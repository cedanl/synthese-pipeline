"""Het model (GMF-bestand): fitten, opslaan, laden en synthetiseren met metasyn.

Een GMF-bestand van synthese-pipeline is een gewoon metasyn-GMF (versie 2.1, met één of meer
tabellen) plus een blok ``synthese_pipeline`` met per kolom de soort en het brontype, en of
disclosure control aan stond. Kolommen met vrije tekst (soort ``freetext``) staan niet in het
model: het blok noemt ze per tabel onder ``uitgesloten``, met alleen de naam en het brontype.
Alleen dit bestand hoeft de beveiligde omgeving te verlaten.

Met een specificatie (:mod:`synthese_pipeline.spec`) gelden bij het fitten de soorten,
verdelingen, omschrijvingen, uitgesloten kolommen en ``n_rows`` daaruit; ``--type`` en
``--faker`` gaan voor. :func:`data_free_model` maakt een model uit alleen de specificatie,
zonder data (``data_free``).
"""

from __future__ import annotations

import json
import logging
import warnings
from collections import Counter
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from metasyn import MetaFrame
from metasyn.gmf import parse_gmf_dict, validate_gmf_dict
from metasyn.util import set_global_seeds
from metasyn.var import MetaVar

from synthese_pipeline import __version__
from synthese_pipeline.column_types import (
    check_faker_conflicts,
    check_overrides,
    dtype_from_dict,
    dtype_to_dict,
    faker_types_for_table,
    kinds_for_table,
    prepare_column,
    restore_column,
)
from synthese_pipeline.errors import SyntheseError
from synthese_pipeline.privacy import PLUGIN_NAME, disclosure_info, disclosure_privacy
from synthese_pipeline.spec import (
    SINGLE_VERSION,
    ColumnSpec,
    Spec,
    data_free_var,
    fit_var_spec,
    kind_from_var,
)
from synthese_pipeline.storage import check_table_names

TOOL_KEY = "synthese_pipeline"
GMF_VERSION = "2.1"
EXCLUDED_KIND = "freetext"

log = logging.getLogger("synthese_pipeline")


def _jsonable(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: _jsonable(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_jsonable(item) for item in data]
    if isinstance(data, np.ndarray):
        return _jsonable(data.tolist())
    if isinstance(data, np.generic):
        return data.item()
    return data


def _log_warnings(table: str, caught: list[warnings.WarningMessage], step: str) -> None:
    # Alleen het aantal en de soort waarschuwing: de tekst kan waarden uit de data bevatten.
    counts = Counter(item.category.__name__ for item in caught)
    for category, count in sorted(counts.items()):
        log.info(
            "tabel '%s': %d waarschuwing(en) van metasyn bij %s (%s)", table, count, step, category
        )


def fit_tables(
    tables: dict[str, pl.DataFrame],
    *,
    overrides: dict[str, str] | None = None,
    fakers: dict[str, str] | None = None,
    disclosure: bool = True,
    spec: Spec | None = None,
) -> dict[str, Any]:
    """Fit per tabel een metasyn-model en geef het GMF als dict terug.

    ``overrides`` zijn de soorten uit ``--type``, ``fakers`` de Faker-providers uit ``--faker``.
    Kolommen van soort ``freetext`` gaan niet in het model. ``spec`` is een gelezen
    specificatie; ``--type`` en ``--faker`` gaan daar per kolom voor.
    """
    overrides = overrides or {}
    fakers = fakers or {}
    check_table_names(tables)
    check_overrides(overrides, tables)
    check_overrides(fakers, tables, option="--faker")
    check_faker_conflicts(overrides, fakers)
    table_specs = spec.match_tables(tables) if spec is not None else {}
    privacy = disclosure_privacy() if disclosure else None
    disclosure_block = disclosure_info(privacy)

    gmf_tables: list[dict[str, Any]] = []
    table_info: dict[str, Any] = {}
    for name, df in tables.items():
        table_spec = table_specs.get(name)
        column_specs = table_spec.columns if table_spec is not None else {}
        from_cli = _cli_columns(name, df, overrides, fakers)
        spec_kinds = {
            column: col.kind
            for column, col in column_specs.items()
            if col.kind is not None and column not in from_cli
        }
        kinds = kinds_for_table(name, df, overrides, fakers, spec_kinds)
        faker_types = faker_types_for_table(name, df, fakers)
        for column, col in column_specs.items():
            if column in from_cli:
                if col.sets_model:
                    log.info(
                        "tabel '%s', kolom '%s': --type/--faker gaat voor de specificatie",
                        name,
                        column,
                    )
            elif col.faker is not None:
                faker_types[column] = col.faker
        excluded = [column for column in df.columns if kinds[column] == EXCLUDED_KIND]
        if len(excluded) == df.width:
            raise SyntheseError(
                f"Tabel '{name}' heeft alleen kolommen met vrije tekst; die gaan niet in het "
                "model. Geef met --type kolom=text of --faker kolom=provider aan wat erin moet."
            )
        columns, var_specs, column_info, excluded_info = [], [], {}, {}
        for column in df.columns:
            col_spec = column_specs.get(column)
            if column in excluded:
                excluded_info[column] = {"brontype": dtype_to_dict(df.schema[column])}
                if col_spec is not None and col_spec.description:
                    excluded_info[column]["omschrijving"] = col_spec.description
                log.info(
                    "tabel '%s', kolom '%s': soort %s, uitgesloten (vrije tekst)",
                    name,
                    column,
                    EXCLUDED_KIND,
                )
                continue
            prepared, var_spec = prepare_column(
                df[column], kinds[column], faker_type=faker_types.get(column)
            )
            columns.append(prepared)
            if col_spec is not None:
                var_spec = fit_var_spec(
                    var_spec, col_spec, kinds[column], use_model=column not in from_cli
                )
            if var_spec:
                var_specs.append({"name": column, **var_spec})
            column_info[column] = {
                "soort": kinds[column],
                "brontype": dtype_to_dict(df.schema[column]),
            }
            log.info("tabel '%s', kolom '%s': soort %s", name, column, kinds[column])
        fit_df = pl.DataFrame(columns)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metaframe = MetaFrame.fit_dataframe(
                fit_df,
                var_specs=var_specs,
                privacy=privacy,
                n_rows=table_spec.n_rows if table_spec is not None else None,
                progress_bar=False,
                name=name,
            )
        _log_warnings(name, caught, "het fitten")
        gmf_tables.extend(_jsonable(metaframe.to_dict())["tables"])
        table_info[name] = {"kolommen": column_info}
        if excluded_info:
            table_info[name]["uitgesloten"] = excluded_info
            log.info(
                "tabel '%s': %d kolom(men) uitgesloten (vrije tekst): %s",
                name,
                len(excluded),
                ", ".join(excluded),
            )
        log.info("tabel '%s': %d rijen, %d kolommen gefit", name, df.height, fit_df.width)

    gmf = {
        "gmf_version": GMF_VERSION,
        "provenance": {
            "created by": {"name": "metasyn", "version": version("metasyn")},
            "creation time": datetime.now().isoformat(),
            "synthese-pipeline": __version__,
            "disclosure control": (
                f"{disclosure_block['plugin']} {disclosure_block['versie']}"
                if disclosure_block["gebruikt"]
                else "uit"
            ),
        },
        "relations": [],
        "tables": gmf_tables,
        TOOL_KEY: {
            "versie": __version__,
            "disclosure_control": disclosure_block,
            "tabellen": table_info,
        },
    }
    validate_gmf_dict(gmf)
    return gmf


def _cli_columns(
    table: str, df: pl.DataFrame, overrides: dict[str, str], fakers: dict[str, str]
) -> set[str]:
    """Kolommen waarvoor ``--type`` of ``--faker`` iets zegt (die gaan voor de specificatie)."""
    return {
        column
        for column in df.columns
        if any(key in overrides or key in fakers for key in (f"{table}.{column}", column))
    }


def _data_free_var(table: str, column: ColumnSpec, n_rows: int) -> MetaVar:
    """Maak één kolom met metasyn uit alleen de specificatie en trek er één proefwaarde uit."""
    config = {
        "config_version": SINGLE_VERSION,
        "n_rows": n_rows,
        "var": [data_free_var(table, column)],
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            metaframe = MetaFrame.fit_dataframe(None, config=config, progress_bar=False, name=table)
            metaframe.synthesize(1, progress_bar=False)
        except Exception as exc:
            # De melding van metasyn kan waarden uit de specificatie bevatten: alleen het type.
            raise SyntheseError(
                f"Specificatie, tabel '{table}', kolom '{column.name}': metasyn kan de verdeling "
                f"niet gebruiken ({type(exc).__name__}). Kijk implements, parameters en brontype "
                "na."
            ) from None
    _log_warnings(table, caught, "het maken zonder data")
    return metaframe.meta_vars[0]


def data_free_model(spec: Spec, *, rows: int | None = None) -> dict[str, Any]:
    """Maak een GMF uit alleen de specificatie, zonder data (data-free).

    Elke kolom krijgt de soort en verdeling uit de specificatie; uitgesloten kolommen (vrije
    tekst) gaan er niet in. Het aantal rijen komt uit ``rows`` of anders uit ``n_rows``. Het
    blok van synthese-pipeline krijgt ``data_free = true``; disclosure control is niet nodig,
    want er is geen data.
    """
    gmf_tables: list[dict[str, Any]] = []
    table_info: dict[str, Any] = {}
    for table_spec, name in zip(spec.tables, spec.table_names(), strict=True):
        n_rows = rows if rows is not None else table_spec.n_rows
        if n_rows is None:
            raise SyntheseError(
                f"Specificatie, tabel '{name}': geen aantal rijen. Geef --rows of n_rows op."
            )
        meta_vars: list[MetaVar] = []
        column_info: dict[str, Any] = {}
        excluded_info: dict[str, Any] = {}
        for column in table_spec.columns.values():
            if column.excluded:
                excluded_info[column.name] = {"brontype": {"naam": column.dtype or "String"}}
                if column.description:
                    excluded_info[column.name]["omschrijving"] = column.description
                continue
            meta_var = _data_free_var(name, column, n_rows)
            meta_vars.append(meta_var)
            var_dict = meta_var.to_dict()
            dtype = column.dtype or str(var_dict["dtype"]).split("(")[0]
            column_info[column.name] = {
                "soort": column.kind or kind_from_var(var_dict),
                "brontype": dtype_to_dict(dtype_from_dict({"naam": dtype})),
            }
            log.info(
                "tabel '%s', kolom '%s': soort %s (zonder data)",
                name,
                column.name,
                column_info[column.name]["soort"],
            )
        if not meta_vars:
            raise SyntheseError(
                f"Specificatie, tabel '{name}': alle kolommen zijn uitgesloten; er blijft niets "
                "over om te maken."
            )
        metaframe = MetaFrame(meta_vars, n_rows, name=name)
        gmf_tables.extend(_jsonable(metaframe.to_dict())["tables"])
        table_info[name] = {"kolommen": column_info}
        if excluded_info:
            table_info[name]["uitgesloten"] = excluded_info
            log.info(
                "tabel '%s': %d kolom(men) uitgesloten (vrije tekst): %s",
                name,
                len(excluded_info),
                ", ".join(excluded_info),
            )
    gmf = {
        "gmf_version": GMF_VERSION,
        "provenance": {
            "created by": {"name": "metasyn", "version": version("metasyn")},
            "creation time": datetime.now().isoformat(),
            "synthese-pipeline": __version__,
            "disclosure control": "niet van toepassing (data-free)",
        },
        "relations": [],
        "tables": gmf_tables,
        TOOL_KEY: {
            "versie": __version__,
            "data_free": True,
            "disclosure_control": {
                "gebruikt": False,
                "plugin": PLUGIN_NAME,
                "toelichting": "niet van toepassing: gemaakt uit alleen de specificatie, "
                "zonder data",
            },
            "tabellen": table_info,
        },
    }
    validate_gmf_dict(gmf)
    return gmf


def save_model(gmf: dict[str, Any], path: str | Path) -> None:
    """Schrijf het GMF naar een JSON-bestand."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(gmf, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_model(path: str | Path) -> dict[str, Any]:
    """Lees en controleer een GMF-bestand.

    Naast het GMF-schema worden de tabelnamen gecontroleerd (:func:`check_table_names`): die
    worden bij ``synthesize`` map- of bestandsnamen, en het model kan met de hand zijn aangepast.
    """
    source = Path(path)
    if not source.is_file():
        raise SyntheseError(f"Het model bestaat niet: {source}")
    try:
        with source.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SyntheseError(f"Het model is geen geldig JSON-bestand: {source}") from exc
    try:
        gmf = parse_gmf_dict(raw, validate=True)
    except Exception as exc:
        raise SyntheseError(f"Het model is geen geldig GMF-bestand: {source}") from exc
    try:
        check_table_names(table.get("name") for table in gmf["tables"])
    except SyntheseError as exc:
        raise SyntheseError(f"Het model is niet bruikbaar: {source}. {exc}") from None
    return gmf


def metaframes(gmf: dict[str, Any]) -> dict[str, MetaFrame]:
    """Een metasyn-``MetaFrame`` per tabel in het GMF."""
    return {
        table["name"]: MetaFrame.from_dict(gmf, table_name=table["name"]) for table in gmf["tables"]
    }


def tool_info(gmf: dict[str, Any]) -> dict[str, Any]:
    """Het blok van synthese-pipeline uit het GMF (leeg bij een gewoon metasyn-GMF)."""
    return gmf.get(TOOL_KEY) or {}


def column_kinds(gmf: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Soort per kolom per tabel, voor zover het model die vermeldt."""
    tables = tool_info(gmf).get("tabellen", {})
    return {
        table: {column: info.get("soort", "") for column, info in data["kolommen"].items()}
        for table, data in tables.items()
    }


def excluded_columns(gmf: dict[str, Any]) -> dict[str, list[str]]:
    """Uitgesloten kolommen (vrije tekst) per tabel; leeg bij een model zonder uitgesloten kolommen.

    Een ouder model zonder ``uitgesloten`` geeft gewoon niets.
    """
    tables = tool_info(gmf).get("tabellen", {})
    return {
        table: list(data["uitgesloten"])
        for table, data in tables.items()
        if isinstance(data, dict) and data.get("uitgesloten")
    }


def expected_schemas(gmf: dict[str, Any]) -> dict[str, dict[str, pl.DataType]]:
    """Kolomnamen en brontypes per tabel volgens het model."""
    stored = tool_info(gmf).get("tabellen", {})
    schemas: dict[str, dict[str, pl.DataType]] = {}
    for table in gmf["tables"]:
        name = table["name"]
        columns = stored.get(name, {}).get("kolommen", {})
        schema = {}
        for var in table["vars"]:
            if var["name"] in columns:
                schema[var["name"]] = dtype_from_dict(columns[var["name"]]["brontype"])
            else:
                schema[var["name"]] = dtype_from_dict({"naam": var["dtype"].split("(")[0]})
        schemas[name] = schema
    return schemas


def synthesize_tables(
    gmf: dict[str, Any], *, rows: int | None = None, seed: int
) -> dict[str, pl.DataFrame]:
    """Maak synthetische tabellen uit alleen het GMF.

    Hetzelfde GMF met dezelfde ``seed`` geeft dezelfde uitvoer. Zonder ``rows`` krijgt elke
    tabel het aantal rijen uit het model.
    """
    schemas = expected_schemas(gmf)
    excluded = excluded_columns(gmf)
    set_global_seeds(seed)
    result: dict[str, pl.DataFrame] = {}
    for name, metaframe in metaframes(gmf).items():
        n_rows = rows if rows is not None else metaframe.n_rows
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            synthetic = metaframe.synthesize(n_rows, progress_bar=False)
        _log_warnings(name, caught, "het synthetiseren")
        restored = []
        for column in synthetic.columns:
            series, failed = restore_column(synthetic[column], schemas[name][column])
            if failed:
                log.info(
                    "tabel '%s', kolom '%s': %d waarden niet terug te zetten naar het brontype",
                    name,
                    column,
                    failed,
                )
            restored.append(series)
        result[name] = pl.DataFrame(restored)
        log.info("tabel '%s': %d rijen gemaakt", name, n_rows)
        if excluded.get(name):
            log.info(
                "tabel '%s': niet in de uitvoer, uitgesloten (vrije tekst): %s",
                name,
                ", ".join(excluded[name]),
            )
    return result
