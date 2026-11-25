"""Specificatiebestand (TOML, het formaat van metasyn): ontwerp maken, lezen en controleren.

Een specificatie beschrijft per kolom wat de instelling weet: de soort, een omschrijving, of
de kolom uniek is of uitgesloten, en eventueel de verdeling. ``init-spec`` maakt een ontwerp
uit de bron; ``fit --spec`` gebruikt het bij het fitten; ``synthesize --spec`` maakt er zonder
data en zonder model dummydata mee (data-free).

Het bestand volgt het configuratieformaat van metasyn 3.1:

- ``config_version = "1.2"`` (ook 1.0 en 1.1): één tabel, met ``[[var]]`` per kolom en
  optioneel ``name`` en ``n_rows``;
- ``config_version = "2.0"``: meer tabellen, met ``[[table]]`` (``name``, ``n_rows``) en
  ``[[table.var]]`` per kolom.

Per kolom (``[[var]]``) gelden de sleutels van metasyn (``name``, ``description``,
``distribution``, ``prop_missing``, ``var_type``, ``data_free``) plus die van synthese-pipeline:
``soort`` (zie :data:`synthese_pipeline.column_types.KINDS`), ``uitgesloten``, ``brontype``,
``categorieen`` en ``faker``. Een fout noemt alleen tabel- en kolomnamen en sleutels, nooit
waarden uit het bestand.
"""

from __future__ import annotations

import logging
import tomllib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import tomli_w
from metasyn.builder import ConfigV1XParser, ConfigV2Parser

from synthese_pipeline.column_types import (
    FAKER_LOCALE,
    KINDS,
    check_faker_provider,
    dtype_to_dict,
    kinds_for_table,
)
from synthese_pipeline.errors import SyntheseError
from synthese_pipeline.storage import check_table_names

log = logging.getLogger("synthese_pipeline")

SINGLE_VERSIONS = tuple(ConfigV1XParser.supports)
MULTI_VERSIONS = tuple(ConfigV2Parser.supports)
SUPPORTED_VERSIONS = SINGLE_VERSIONS + MULTI_VERSIONS
SINGLE_VERSION = max(SINGLE_VERSIONS)
MULTI_VERSION = max(MULTI_VERSIONS)

DEFAULT_TABLE = "data"
EXCLUDED_KIND = "freetext"
RARE_CATEGORY_COUNT = 10

BRONTYPES = (
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Float32",
    "Float64",
    "Boolean",
    "String",
    "Categorical",
    "Date",
    "Datetime",
    "Time",
)
INTEGER_BRONTYPES = tuple(name for name in BRONTYPES if name.startswith(("Int", "UInt")))

_SINGLE_KEYS = {"config_version", "name", "n_rows", "var"}
_MULTI_KEYS = {"config_version", "table"}
_TABLE_KEYS = {"name", "n_rows", "var"}
_VAR_KEYS = {
    "name",
    "soort",
    "description",
    "uitgesloten",
    "distribution",
    "prop_missing",
    "categorieen",
    "faker",
    "brontype",
    "var_type",
    "data_free",
}
_DISTRIBUTION_KEYS = {"implements", "name", "parameters", "unique", "version"}

HEADER = f"""\
# Specificatie voor synthese-pipeline (configuratieformaat van metasyn).
# Gemaakt met 'synthese-pipeline init-spec'. Kijk het na en vul aan:
# - soort: {", ".join(KINDS)}
# - description: omschrijving van de kolom. Zet hier nooit echte waarden in.
# - uitgesloten = true: de kolom gaat niet in het model (standaard bij vrije tekst).
# - distribution: een verdeling van metasyn, bijv.
#   distribution = {{implements = "core.uniform", parameters = {{lower = 1, upper = 10}}}}
# - categorieen: de categorieën van een categorische kolom (nodig voor data-free).
# Gebruik: synthese-pipeline fit BRON MODEL --spec SPEC
#      of: synthese-pipeline synthesize --spec SPEC --rows N DOEL (data-free)

"""


@dataclass(frozen=True)
class ColumnSpec:
    """Wat de specificatie over één kolom zegt; ``None`` = niet opgegeven."""

    name: str
    kind: str | None = None
    excluded: bool = False
    description: str | None = None
    distribution: dict[str, Any] | None = None
    unique: bool | None = None
    prop_missing: float | None = None
    categories: list[Any] | None = None
    faker: str | None = None
    dtype: str | None = None
    var_type: str | None = None

    @property
    def sets_model(self) -> bool:
        """Zegt de kolom iets over soort of verdeling? Dat gaat opzij voor --type en --faker."""
        return (
            self.kind is not None
            or self.distribution is not None
            or self.unique is not None
            or self.faker is not None
        )


@dataclass(frozen=True)
class TableSpec:
    """Eén tabel uit de specificatie. ``name`` is ``None`` bij een specificatie zonder naam."""

    name: str | None
    n_rows: int | None
    columns: dict[str, ColumnSpec]


@dataclass(frozen=True)
class Spec:
    """Een gelezen en gecontroleerde specificatie."""

    version: str
    tables: list[TableSpec]

    @property
    def multi(self) -> bool:
        return self.version in MULTI_VERSIONS

    def table_names(self) -> list[str]:
        """Naam per tabel; een tabel zonder naam heet ``data``."""
        return [table.name or DEFAULT_TABLE for table in self.tables]

    def match_tables(self, tables: dict[str, pl.DataFrame]) -> dict[str, TableSpec]:
        """Koppel de tabellen van de specificatie aan die van de bron.

        Eén tabel aan beide kanten: de namen tellen niet. Anders op naam; een bron-tabel zonder
        specificatie krijgt de afgeleide soorten. Elke kolom in de specificatie moet in de bron
        staan.
        """
        if not self.multi and len(tables) != 1:
            raise SyntheseError(
                f"De specificatie beschrijft één tabel (config_version {self.version}), maar de "
                f"bron heeft {len(tables)} tabellen. Gebruik config_version {MULTI_VERSION} met "
                "[[table]] per tabel."
            )
        if len(self.tables) == 1 and len(tables) == 1:
            matched = {next(iter(tables)): self.tables[0]}
        else:
            matched = {}
            for table in self.tables:
                name = table.name or DEFAULT_TABLE
                if name not in tables:
                    raise SyntheseError(
                        f"De specificatie noemt tabel '{name}', maar die staat niet in de bron."
                    )
                matched[name] = table
        for name, table in matched.items():
            for column in table.columns:
                if column not in tables[name].columns:
                    raise SyntheseError(
                        f"De specificatie noemt kolom '{column}' in tabel '{name}', maar die "
                        "bestaat niet in de bron."
                    )
        return matched


# ---------------------------------------------------------------------------- ontwerp maken


def _categories(series: pl.Series) -> list[Any]:
    filled = series.drop_nulls()
    if isinstance(filled.dtype, (pl.Categorical, pl.Enum)):
        filled = filled.cast(pl.String)
    elif filled.dtype.is_float():
        filled = filled.cast(pl.Int64)
    return filled.unique().sort().to_list()


def _rare_categories(series: pl.Series) -> int:
    counts = series.drop_nulls().value_counts(name="aantal")
    return int((counts["aantal"] < RARE_CATEGORY_COUNT).sum())


def _draft_var(table: str, series: pl.Series, kind: str, with_categories: bool) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": series.name,
        "soort": kind,
        "brontype": dtype_to_dict(series.dtype)["naam"],
    }
    if kind == EXCLUDED_KIND:
        entry["uitgesloten"] = True
    entry["description"] = ""
    if kind == "key":
        entry["distribution"] = {"unique": True}
    if with_categories and kind == "categorical":
        entry["categorieen"] = _categories(series)
        rare = _rare_categories(series)
        if rare:
            log.warning(
                "tabel '%s', kolom '%s': %d categorie(ën) komen minder dan %d keer voor; "
                "kijk na of die in de specificatie mogen staan",
                table,
                series.name,
                rare,
                RARE_CATEGORY_COUNT,
            )
    return entry


def build_draft(
    tables: dict[str, pl.DataFrame], *, with_categories: bool = False
) -> dict[str, Any]:
    """Maak een ontwerp-specificatie uit de bron.

    Per kolom: naam, afgeleide soort, brontype en een lege omschrijving; een kandidaat-sleutel
    krijgt ``distribution = {unique = true}``, vrije tekst ``uitgesloten = true``. Zonder
    ``with_categories`` staan er geen waarden uit de bron in, alleen namen, soorten, types en
    aantallen. Eén tabel geeft config_version 1.2, meer tabellen 2.0 met ``[[table]]``.
    """
    check_table_names(tables)
    drafts = []
    for name, df in tables.items():
        kinds = kinds_for_table(name, df, {})
        variables = [_draft_var(name, df[c], kinds[c], with_categories) for c in df.columns]
        drafts.append({"name": name, "n_rows": df.height, "var": variables})
        for column in df.columns:
            log.info("tabel '%s', kolom '%s': soort %s", name, column, kinds[column])
    if len(drafts) == 1:
        return {"config_version": SINGLE_VERSION, **drafts[0]}
    return {"config_version": MULTI_VERSION, "table": drafts}


def write_spec(spec: dict[str, Any], path: str | Path) -> None:
    """Schrijf een specificatie als TOML, met uitleg bovenaan."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(HEADER + tomli_w.dumps(spec), encoding="utf-8")


# ---------------------------------------------------------------------------------- lezen


def _where(table: str | None, column: str | None = None) -> str:
    parts = []
    if table is not None:
        parts.append(f"tabel '{table}'")
    if column is not None:
        parts.append(f"kolom '{column}'")
    return ", ".join(parts)


def _fail(path: Path, where: str, text: str) -> SyntheseError:
    prefix = f"De specificatie {path}"
    return SyntheseError(f"{prefix}: {where}: {text}" if where else f"{prefix}: {text}")


def _check_keys(path: Path, where: str, data: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise _fail(
            path,
            where,
            f"onbekende sleutel(s) {', '.join(unknown)}. Toegestaan: {', '.join(sorted(allowed))}.",
        )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _read_n_rows(path: Path, where: str, value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise _fail(path, where, "n_rows moet een geheel getal van minstens 1 zijn.")
    return value


def _read_distribution(path: Path, where: str, value: Any) -> tuple[dict[str, Any] | None, Any]:
    """Geef (verdeling zonder ``unique``, ``unique``) terug; ``implements`` wordt ``name``."""
    if isinstance(value, str):
        value = {"name": value}
    if not isinstance(value, dict):
        raise _fail(path, where, "distribution moet een tabel zijn, bijv. {unique = true}.")
    _check_keys(path, f"{where}, distribution", value, _DISTRIBUTION_KEYS)
    dist = deepcopy(value)
    if "implements" in dist:
        if "name" in dist:
            raise _fail(path, where, "geef in distribution implements of name, niet allebei.")
        dist["name"] = dist.pop("implements")
    if "name" in dist and not isinstance(dist["name"], str):
        raise _fail(path, where, "implements in distribution moet tekst zijn.")
    if "parameters" in dist and not isinstance(dist["parameters"], dict):
        raise _fail(path, where, "parameters in distribution moet een tabel zijn.")
    if "parameters" in dist and "name" not in dist:
        raise _fail(path, where, "distribution heeft parameters maar geen implements.")
    unique = dist.pop("unique", None)
    if unique is not None and not isinstance(unique, bool):
        raise _fail(path, where, "unique in distribution moet true of false zijn.")
    return (dist or None), unique


def _read_categories(path: Path, where: str, value: Any) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise _fail(path, where, "categorieen moet een niet-lege lijst zijn.")
    kinds = {
        "bool"
        if isinstance(item, bool)
        else "int"
        if isinstance(item, int)
        else type(item).__name__
        for item in value
    }
    if len(kinds) != 1 or not kinds <= {"bool", "int", "str"}:
        raise _fail(
            path,
            where,
            "categorieen moet allemaal tekst, allemaal gehele getallen of allemaal true/false "
            "zijn.",
        )
    if len(set(value)) != len(value):
        raise _fail(path, where, "categorieen bevat een categorie meer dan één keer.")
    return list(value)


def _read_column(path: Path, table: str | None, data: Any, number: int) -> ColumnSpec:
    if not isinstance(data, dict):
        raise _fail(path, _where(table), f"kolom {number} is geen tabel ([[var]]).")
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise _fail(path, _where(table), f"kolom {number} heeft geen name.")
    where = _where(table, name)
    _check_keys(path, where, data, _VAR_KEYS)

    kind = data.get("soort")
    if kind is not None and (not isinstance(kind, str) or kind.lower() not in KINDS):
        raise _fail(path, where, f"onbekende soort. Kies uit: {', '.join(KINDS)}.")
    kind = kind.lower() if isinstance(kind, str) else None

    excluded = data.get("uitgesloten")
    if excluded is not None and not isinstance(excluded, bool):
        raise _fail(path, where, "uitgesloten moet true of false zijn.")
    if excluded is False and kind == EXCLUDED_KIND:
        raise _fail(
            path,
            where,
            "soort freetext gaat nooit in het model; kies soort text of faker om hem op te nemen.",
        )

    description = data.get("description")
    if description is not None and not isinstance(description, str):
        raise _fail(path, where, "description moet tekst zijn.")

    distribution, unique = (None, None)
    if "distribution" in data:
        distribution, unique = _read_distribution(path, where, data["distribution"])

    prop_missing = data.get("prop_missing")
    if prop_missing is not None and (not _is_number(prop_missing) or not 0 <= prop_missing <= 1):
        raise _fail(path, where, "prop_missing moet een getal tussen 0 en 1 zijn.")

    faker = data.get("faker")
    if faker is not None:
        if not isinstance(faker, str):
            raise _fail(path, where, "faker moet de naam van een Faker-provider zijn.")
        if kind not in (None, "faker"):
            raise _fail(path, where, f"faker hoort bij soort faker, niet bij soort {kind}.")
        if distribution is not None:
            raise _fail(path, where, "geef faker of distribution, niet allebei.")
        check_faker_provider(faker, option=f"de specificatie ({where})")
        kind = "faker"
    elif kind == "faker":
        raise _fail(path, where, 'soort faker vraagt een provider: faker = "name".')

    categories = None
    if "categorieen" in data:
        categories = _read_categories(path, where, data["categorieen"])
        if kind not in (None, "categorical"):
            raise _fail(path, where, f"categorieen hoort bij soort categorical, niet bij {kind}.")
        kind = kind or "categorical"

    if unique is True:
        if kind not in (None, "key", EXCLUDED_KIND) and not excluded:
            raise _fail(path, where, f"unique = true hoort bij soort key, niet bij {kind}.")
        kind = kind or "key"
    elif unique is False and kind == "key":
        raise _fail(path, where, "soort key is altijd uniek; unique = false past daar niet bij.")

    if excluded:
        kind = EXCLUDED_KIND

    dtype = data.get("brontype")
    if dtype is not None and dtype not in BRONTYPES:
        raise _fail(path, where, f"onbekend brontype. Kies uit: {', '.join(BRONTYPES)}.")
    var_type = data.get("var_type")
    if var_type is not None and not isinstance(var_type, str):
        raise _fail(path, where, "var_type moet tekst zijn.")
    data_free = data.get("data_free")
    if data_free is not None and not isinstance(data_free, bool):
        raise _fail(path, where, "data_free moet true of false zijn.")

    return ColumnSpec(
        name=name,
        kind=kind,
        excluded=kind == EXCLUDED_KIND,
        description=description or None,
        distribution=distribution,
        unique=unique,
        prop_missing=None if prop_missing is None else float(prop_missing),
        categories=categories,
        faker=faker,
        dtype=dtype,
        var_type=var_type,
    )


def _read_table(path: Path, data: Any, name: str | None, number: int | None) -> TableSpec:
    where = _where(name) if name is not None else (f"tabel {number}" if number else "")
    variables = data.get("var")
    if not isinstance(variables, list) or not variables:
        raise _fail(path, where, "geen kolommen: geef per kolom een [[var]] met een name.")
    columns: dict[str, ColumnSpec] = {}
    for index, item in enumerate(variables, start=1):
        column = _read_column(path, name, item, index)
        if column.name in columns:
            raise _fail(path, _where(name, column.name), "deze kolom staat er meer dan één keer.")
        columns[column.name] = column
    return TableSpec(
        name=name, n_rows=_read_n_rows(path, where, data.get("n_rows")), columns=columns
    )


def _version(path: Path, data: dict[str, Any]) -> str:
    value = data.get("config_version")
    if value is None:
        return MULTI_VERSION if "table" in data else SINGLE_VERSION
    if _is_number(value):
        value = str(value)
    if value not in SUPPORTED_VERSIONS:
        raise _fail(
            path,
            "",
            "deze config_version wordt niet ondersteund. metasyn kent: "
            f"{', '.join(SUPPORTED_VERSIONS)}.",
        )
    return value


def parse_spec(data: dict[str, Any], path: str | Path = "specificatie") -> Spec:
    """Controleer een gelezen specificatie (een dict uit TOML) en geef een :class:`Spec`."""
    source = Path(path)
    version = _version(source, data)
    if version in SINGLE_VERSIONS:
        _check_keys(source, "", data, _SINGLE_KEYS)
        name = data.get("name")
        if name is not None and not isinstance(name, str):
            raise _fail(source, "", "name moet tekst zijn.")
        tables = [_read_table(source, data, name, None)]
    else:
        _check_keys(source, "", data, _MULTI_KEYS)
        items = data.get("table")
        if not isinstance(items, list) or not items:
            raise _fail(source, "", "geen tabellen: geef per tabel een [[table]] met een name.")
        tables = []
        for number, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                raise _fail(source, f"tabel {number}", "is geen tabel ([[table]]).")
            name = item.get("name")
            if not isinstance(name, str) or not name:
                raise _fail(source, f"tabel {number}", "heeft geen name.")
            if "relations" in item:
                raise _fail(
                    source, _where(name), "relaties tussen tabellen worden nog niet ondersteund."
                )
            _check_keys(source, _where(name), item, _TABLE_KEYS)
            tables.append(_read_table(source, item, name, number))
    spec = Spec(version=version, tables=tables)
    try:
        check_table_names(spec.table_names())
    except SyntheseError as exc:
        raise _fail(source, "", str(exc)) from None
    return spec


def load_spec(path: str | Path) -> Spec:
    """Lees en controleer een specificatiebestand (TOML)."""
    source = Path(path)
    if not source.is_file():
        raise SyntheseError(f"De specificatie bestaat niet: {source}")
    try:
        data = tomllib.loads(source.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        raise SyntheseError(f"De specificatie is geen tekstbestand (UTF-8): {source}") from None
    except tomllib.TOMLDecodeError as exc:
        # Alleen de plek: de melding van tomllib kan een stukje van het bestand bevatten.
        line = getattr(exc, "lineno", None)
        place = f" (regel {line})" if line else ""
        raise SyntheseError(
            f"De specificatie is geen geldig TOML-bestand: {source}{place}."
        ) from None
    return parse_spec(data, source)


# --------------------------------------------------------------- toepassen bij fit en data-free


def fit_var_spec(
    base: dict[str, Any] | None, column: ColumnSpec, kind: str, *, use_model: bool
) -> dict[str, Any]:
    """``var_spec`` voor metasyn bij ``fit``: ``base`` (uit de soort) plus de specificatie.

    Met ``use_model`` (geen ``--type``/``--faker`` voor deze kolom) gaat een verdeling met
    ``implements`` uit de specificatie voor die van de soort; ``unique`` wordt overgenomen. Een
    omschrijving en ``prop_missing`` gelden altijd.
    """
    result = deepcopy(base) if base else {}
    if use_model:
        if column.distribution is not None and "name" in column.distribution:
            dist = deepcopy(column.distribution)
            if kind == "key":
                dist["unique"] = True
            elif column.unique is not None:
                dist["unique"] = column.unique
            result["distribution"] = dist
        elif column.unique is not None:
            dist = dict(result.get("distribution") or {})
            dist["unique"] = column.unique or kind == "key"
            result["distribution"] = dist
    if column.description:
        result["description"] = column.description
    if column.prop_missing is not None:
        result["prop_missing"] = column.prop_missing
    return result


def _label_dtype(labels: list[Any]) -> str | None:
    if all(isinstance(label, bool) for label in labels):
        return "Boolean"
    if all(isinstance(label, int) for label in labels):
        return "Int64"
    if all(isinstance(label, str) for label in labels):
        return "String"
    return None


def _default_dtype(kind: str | None, dist: dict[str, Any]) -> str | None:
    if kind == "categorical":
        labels = (dist.get("parameters") or {}).get("labels")
        return _label_dtype(labels) if isinstance(labels, list) and labels else None
    if kind == "key":
        return "String" if dist.get("name", "").endswith("regex") else "Int64"
    return {
        "discrete": "Int64",
        "continuous": "Float64",
        "date": "Date",
        "text": "String",
        "faker": "String",
    }.get(kind or "")


def data_free_var(table: str, column: ColumnSpec) -> dict[str, Any]:
    """Een kolom van de configuratie voor metasyn zonder data (``data_free = true``).

    Zonder verdeling met ``implements`` vult de soort die aan waar dat kan: ``key`` wordt een
    oplopende unieke sleutel vanaf 1, ``categorical`` met ``categorieen`` een keuze met gelijke
    kansen, ``faker`` de Faker-provider. Andere soorten hebben een verdeling met parameters nodig.
    """
    where = _where(table, column.name)
    kind = column.kind
    dist: dict[str, Any] | None = deepcopy(column.distribution) if column.distribution else None
    if dist is None or "name" not in dist:
        if kind == "categorical" and column.categories:
            n = len(column.categories)
            dist = {
                "name": "core.multinoulli",
                "parameters": {"labels": list(column.categories), "probs": [1 / n] * n},
            }
        elif kind == "key" and (column.dtype is None or column.dtype in INTEGER_BRONTYPES):
            dist = {"name": "core.unique_key", "parameters": {"lower": 1, "consecutive": True}}
        elif kind == "faker":
            dist = {
                "name": "core.faker",
                "parameters": {"faker_type": column.faker, "locale": FAKER_LOCALE},
            }
        else:
            hint = " of categorieen" if kind in (None, "categorical") else ""
            raise SyntheseError(
                f"Specificatie, {where}: zonder data is een verdeling nodig. Geef distribution "
                f"met implements en parameters{hint}, bijv. distribution = {{implements = "
                '"core.uniform", parameters = {lower = 1, upper = 10}}.'
            )
    if kind == "key" or column.unique:
        dist["unique"] = True
    elif column.unique is False:
        dist["unique"] = False
    var: dict[str, Any] = {
        "name": column.name,
        "data_free": True,
        "distribution": dist,
        "prop_missing": column.prop_missing or 0.0,
    }
    var_type = column.var_type or column.dtype or _default_dtype(kind, dist)
    if var_type is not None:
        var["var_type"] = var_type
    if column.description:
        var["description"] = column.description
    return var


def kind_from_var(var: dict[str, Any]) -> str:
    """Soort van een kolom zonder ``soort`` in de specificatie, uit de verdeling en het type."""
    dist = var.get("distribution") or {}
    name = str(dist.get("name", "")).split(".")[-1]
    if dist.get("unique"):
        return "key"
    if name == "multinoulli":
        return "categorical"
    if name == "faker":
        return "faker"
    dtype = str(var.get("dtype", "")).split("(")[0]
    if dtype.startswith(("Int", "UInt")):
        return "discrete"
    if dtype.startswith("Float"):
        return "continuous"
    if dtype in ("Date", "Datetime", "Time"):
        return "date"
    if dtype == "Boolean":
        return "categorical"
    return "text"
