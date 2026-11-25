"""Kolomtypes afleiden en vertalen naar instellingen voor metasyn.

Elke kolom krijgt een *soort* (afgeleid, eerste regel die past):

- ``categorical``: waar/onwaar, of een geheel getal (max. 12 unieke waarden) of tekst
  (max. 30 unieke waarden) met hooguit half zoveel unieke als ingevulde waarden, bijv.
  Likert 1-5 of M/V. metasyn krijgt ``core.multinoulli``.
- ``key``: geheel getal of tekst zonder lege waarden, alle waarden uniek, minstens 20 rijen.
  metasyn krijgt ``unique``; een code krijgt een uniek patroon uit de vorm.
- ``discrete``: overige gehele getallen, ook kommagetallen die allemaal op ,0 eindigen.
- ``continuous``: overige kommagetallen.
- ``date``: datum, datum met tijd of tijd.
- ``freetext``: tekst die op vrije tekst lijkt (zie hieronder). Deze kolom gaat standaard niet
  in het model en ontbreekt in de synthetische data.
- ``text``: overige tekst. Een code (zie hieronder) krijgt een patroon dat alleen uit de vorm
  van de waarden is gemaakt, ook met disclosure control; anders kiest metasyn zelf.
- ``faker``: alleen als je die zelf kiest (``--faker kolom=provider``): verzonnen waarden van
  Faker (locale ``nl_NL``), zonder iets uit de echte data.

Vrije tekst (na ``categorical``, vóór ``key``, zodat een kolom met unieke zinnen of namen geen
sleutel wordt): een tekstkolom is ``freetext`` als hij geen code is en

- meer dan de helft van de ingevulde waarden een spatie (of ander wit) bevat
  (``FREETEXT_MIN_SPACE_SHARE``), zoals zinnen en "voornaam achternaam", of
- de ingevulde waarden gemiddeld minstens 20 tekens lang zijn (``FREETEXT_MIN_MEAN_LENGTH``).

Een code heeft een vaste vorm. De vorm van een waarde is de waarde met elk cijfer als ``9``,
elke hoofdletter als ``A`` en elke kleine letter als ``a`` (``1234 AB`` wordt ``9999 AA``). Een
kolom is een code als minstens 95% van de ingevulde waarden (``CODE_MIN_SHARE``) één van de drie
meest voorkomende vormen heeft (``CODE_MAX_SHAPES``) en die vormen een cijfer bevatten en
hooguit 12 tekens lang zijn (``CODE_MAX_LENGTH``). Zo blijven postcodes (``1234 AB``) en
opleidingscodes (``34567``) een patroon, en is ``B Informatica`` (geen cijfer, wisselende vorm)
geen code. Het patroon voor metasyn bevat alleen vormen die minstens 10 keer voorkomen
(``CODE_MIN_SHAPE_COUNT``, de CBS-vuistregel van minimaal 10 eenheden), dus geen waarden en
geen zeldzame vormen: ``9999 AA`` wordt ``[0-9]{4} [A-Z]{2}``.

De regels zijn voor deze repo zelf geschreven. Met ``--type kolom=soort`` is elke soort te
overschrijven: ``--type kolom=text`` neemt een uitgesloten kolom toch op, ``--type
kolom=freetext`` sluit een kolom uit. Het oorspronkelijke datatype wordt in het model bewaard
en na het synthetiseren teruggezet, zodat kolomnamen en types gelijk blijven aan de bron.
"""

from __future__ import annotations

import re
from functools import cache
from itertools import groupby
from typing import Any

import polars as pl

from synthese_pipeline.errors import SyntheseError

KINDS = ("categorical", "discrete", "continuous", "date", "key", "text", "freetext", "faker")

INT_CATEGORICAL_MAX_UNIQUE = 12
TEXT_CATEGORICAL_MAX_UNIQUE = 30
KEY_MIN_ROWS = 20

FREETEXT_MIN_SPACE_SHARE = 0.5
FREETEXT_MIN_MEAN_LENGTH = 20
CODE_MAX_SHAPES = 3
CODE_MIN_SHARE = 0.95
CODE_MAX_LENGTH = 12
CODE_MIN_SHAPE_COUNT = 10

FAKER_LOCALE = "nl_NL"
_PROVIDER_NAME = re.compile(r"^[a-z][a-z0-9_]*$")

_SIMPLE_DTYPES: dict[str, pl.DataType] = {
    "Int8": pl.Int8(),
    "Int16": pl.Int16(),
    "Int32": pl.Int32(),
    "Int64": pl.Int64(),
    "UInt8": pl.UInt8(),
    "UInt16": pl.UInt16(),
    "UInt32": pl.UInt32(),
    "UInt64": pl.UInt64(),
    "Float32": pl.Float32(),
    "Float64": pl.Float64(),
    "Boolean": pl.Boolean(),
    "String": pl.String(),
    "Categorical": pl.Categorical(),
    "Date": pl.Date(),
    "Time": pl.Time(),
}


def _is_text(dtype: pl.DataType) -> bool:
    return isinstance(dtype, (pl.String, pl.Categorical, pl.Enum))


def _is_date(dtype: pl.DataType) -> bool:
    return isinstance(dtype, (pl.Date, pl.Datetime, pl.Time))


def _is_float(dtype: pl.DataType) -> bool:
    return dtype.is_float() or isinstance(dtype, pl.Decimal)


def _all_integral(series: pl.Series) -> bool:
    filled = series.drop_nulls().cast(pl.Float64)
    filled = filled.filter(filled.is_finite())
    return bool((filled == filled.round(0)).all())


def _few(n_unique: int, n_filled: int, max_unique: int) -> bool:
    return n_unique <= max_unique and n_unique * 2 <= n_filled


def _is_key(series: pl.Series, n_unique: int) -> bool:
    return series.null_count() == 0 and series.len() >= KEY_MIN_ROWS and n_unique == series.len()


# Tekens met een betekenis in de regex-taal van regexmodel (metasyn); die krijgen een backslash.
_REGEX_SPECIAL = set(".+*?^$()[]#&{}|\\-~")
_SHAPE_CLASSES = {"9": "[0-9]", "A": "[A-Z]", "a": "[a-z]"}


def _code_shapes(text: pl.Series) -> pl.DataFrame | None:
    """De vormen van een codekolom met hun aantal, of ``None`` als het geen code is."""
    shapes = (
        text.str.replace_all(r"\d", "9")
        .str.replace_all(r"[\p{Ll}\p{Lm}\p{Lo}]", "a")
        .str.replace_all(r"[\p{Lu}\p{Lt}]", "A")
        .alias("vorm")
    )
    counts = (
        shapes.value_counts(name="aantal")
        .sort(["aantal", "vorm"], descending=[True, False])
        .head(CODE_MAX_SHAPES)
    )
    code_shapes = counts.filter(
        pl.col("vorm").str.contains("9", literal=True)
        & (pl.col("vorm").str.len_chars() <= CODE_MAX_LENGTH)
    )
    if code_shapes["aantal"].sum() < CODE_MIN_SHARE * text.len():
        return None
    return code_shapes


def _is_code(text: pl.Series) -> bool:
    return _code_shapes(text) is not None


def _shape_regex(shape: str) -> str | None:
    parts = []
    for char, run in ((key, len(list(group))) for key, group in groupby(shape)):
        if char in _SHAPE_CLASSES:
            parts.append(_SHAPE_CLASSES[char] + (f"{{{run}}}" if run > 1 else ""))
        elif char.isspace() and char != " ":
            return None
        else:
            parts.append(("\\" + char if char in _REGEX_SPECIAL else char) * run)
    return "".join(parts)


def code_pattern(series: pl.Series) -> str | None:
    """Regex voor een codekolom, gemaakt uit alleen de vormen (geen waarden); anders ``None``.

    Alleen vormen die minstens ``CODE_MIN_SHAPE_COUNT`` keer voorkomen tellen mee.
    """
    if not _is_text(series.dtype):
        return None
    text = series.drop_nulls().cast(pl.String)
    if text.len() == 0:
        return None
    shapes = _code_shapes(text)
    if shapes is None:
        return None
    parts = [
        regex
        for shape, count in shapes.iter_rows()
        if count >= CODE_MIN_SHAPE_COUNT and (regex := _shape_regex(shape)) is not None
    ]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else "(" + "|".join(parts) + ")"


def _looks_like_prose(filled: pl.Series) -> bool:
    """Vrije tekst: veel waarden met een spatie, of lange waarden; geen code van vaste vorm."""
    text = filled.cast(pl.String)
    with_space = text.str.contains(r"\s").sum() / text.len()
    mean_length = text.str.len_chars().mean()
    long = isinstance(mean_length, (int, float)) and mean_length >= FREETEXT_MIN_MEAN_LENGTH
    if with_space <= FREETEXT_MIN_SPACE_SHARE and not long:
        return False
    return not _is_code(text)


def infer_kind(series: pl.Series) -> str:
    """Leid de soort van een kolom af uit het datatype en de waarden."""
    dtype = series.dtype
    if isinstance(dtype, pl.Boolean):
        return "categorical"
    if _is_date(dtype):
        return "date"
    if not (dtype.is_integer() or _is_float(dtype) or _is_text(dtype)):
        raise SyntheseError(
            f"Kolom '{series.name}' heeft een type dat niet wordt ondersteund "
            f"({_dtype_name(dtype)})."
        )

    filled = series.drop_nulls()
    n_filled = filled.len()
    if n_filled == 0:
        return "text" if _is_text(dtype) else "discrete" if dtype.is_integer() else "continuous"
    n_unique = filled.n_unique()

    if dtype.is_integer():
        if _is_key(series, n_unique):
            return "key"
        return "categorical" if _few(n_unique, n_filled, INT_CATEGORICAL_MAX_UNIQUE) else "discrete"
    if _is_float(dtype):
        if _all_integral(series):
            if _few(n_unique, n_filled, INT_CATEGORICAL_MAX_UNIQUE):
                return "categorical"
            return "discrete"
        return "continuous"
    if _few(n_unique, n_filled, TEXT_CATEGORICAL_MAX_UNIQUE):
        return "categorical"
    if _looks_like_prose(filled):
        return "freetext"
    if _is_key(series, n_unique):
        return "key"
    return "text"


def parse_type_overrides(items: list[str] | None) -> dict[str, str]:
    """Lees ``kolom=soort`` of ``tabel.kolom=soort`` uit ``--type``.

    Een kolomnaam zonder tabel geldt voor elke tabel met die kolom.
    """
    overrides: dict[str, str] = {}
    for item in items or []:
        name, sep, kind = item.partition("=")
        name, kind = name.strip(), kind.strip().lower()
        if not sep or not name:
            raise SyntheseError(f"Ongeldige --type '{item}': gebruik kolom=soort.")
        if kind not in KINDS:
            raise SyntheseError(
                f"Onbekende soort '{kind}' in --type. Kies uit: {', '.join(KINDS)}."
            )
        overrides[name] = kind
    return overrides


@cache
def _faker():
    from faker import Faker

    return Faker(FAKER_LOCALE)


def check_faker_provider(provider: str) -> None:
    """Geef een fout als ``provider`` geen Faker-provider is die tekst maakt."""
    message = (
        f"Onbekende Faker-provider '{provider}' in --faker. Kies een provider die tekst maakt, "
        "bijv. name, first_name, last_name, email, city, street_address, sentence of text."
    )
    if not _PROVIDER_NAME.match(provider) or provider.startswith("seed"):
        raise SyntheseError(message)
    try:
        method = getattr(_faker(), provider)
        example = method() if callable(method) else None
    except Exception:
        raise SyntheseError(message) from None
    if not isinstance(example, str):
        raise SyntheseError(message)


def parse_faker_overrides(items: list[str] | None) -> dict[str, str]:
    """Lees ``kolom=provider`` of ``tabel.kolom=provider`` uit ``--faker``."""
    fakers: dict[str, str] = {}
    for item in items or []:
        name, sep, provider = item.partition("=")
        name, provider = name.strip(), provider.strip()
        if not sep or not name:
            raise SyntheseError(f"Ongeldige --faker '{item}': gebruik kolom=provider.")
        check_faker_provider(provider)
        fakers[name] = provider
    return fakers


def check_faker_conflicts(overrides: dict[str, str], fakers: dict[str, str]) -> None:
    """Geef een fout als een kolom in ``--faker`` ook met een andere soort in ``--type`` staat."""
    for name in fakers:
        kind = overrides.get(name)
        if kind is not None and kind != "faker":
            raise SyntheseError(
                f"Kolom '{name}' staat in --faker en met soort '{kind}' in --type: kies er één."
            )


def check_overrides(
    overrides: dict[str, str], tables: dict[str, pl.DataFrame], option: str = "--type"
) -> None:
    """Geef een fout als ``--type`` of ``--faker`` (``option``) een onbekende kolom noemt."""
    known = set()
    for table, df in tables.items():
        known.update(df.columns)
        known.update(f"{table}.{column}" for column in df.columns)
    for name in overrides:
        if name not in known:
            raise SyntheseError(f"{option} noemt kolom '{name}', maar die bestaat niet in de bron.")


def faker_types_for_table(
    table: str, df: pl.DataFrame, fakers: dict[str, str] | None
) -> dict[str, str]:
    """Faker-provider per kolom met ``--faker``: eerst ``tabel.kolom``, dan ``kolom``."""
    fakers = fakers or {}
    result = {}
    for column in df.columns:
        provider = fakers.get(f"{table}.{column}") or fakers.get(column)
        if provider:
            result[column] = provider
    return result


def kinds_for_table(
    table: str,
    df: pl.DataFrame,
    overrides: dict[str, str],
    fakers: dict[str, str] | None = None,
) -> dict[str, str]:
    """Soort per kolom: ``--faker``, dan ``tabel.kolom``, dan ``kolom``, anders afgeleid."""
    faker_types = faker_types_for_table(table, df, fakers)
    kinds = {}
    for column in df.columns:
        if column in faker_types:
            kinds[column] = "faker"
            continue
        override = overrides.get(f"{table}.{column}") or overrides.get(column)
        if override == "faker":
            raise SyntheseError(
                f"Kolom '{column}' heeft soort 'faker' maar geen provider: gebruik --faker "
                f"{column}=<provider>, bijv. --faker {column}=name."
            )
        kinds[column] = override or infer_kind(df[column])
    return kinds


def _cast_checked(series: pl.Series, dtype: pl.DataType, kind: str) -> pl.Series:
    converted = series.cast(dtype, strict=False)
    failed = converted.null_count() - series.null_count()
    if failed > 0:
        raise SyntheseError(
            f"Kolom '{series.name}' past niet bij soort '{kind}': {failed} waarden zijn niet om "
            "te zetten."
        )
    return converted


def prepare_column(
    series: pl.Series, kind: str, *, faker_type: str | None = None
) -> tuple[pl.Series, dict[str, Any] | None]:
    """Zet een kolom klaar voor metasyn en geef de bijbehorende ``var_spec`` (zonder naam).

    ``freetext`` komt niet in het model en wordt hier geweigerd; ``faker`` vraagt ``faker_type``.
    """
    dtype = series.dtype
    if kind == "categorical":
        if dtype.is_integer() or (_is_float(dtype) and _all_integral(series)):
            prepared = series.cast(pl.Int64)
        elif isinstance(dtype, pl.Boolean):
            prepared = series
        else:
            prepared = series.cast(pl.String)
        return prepared, {"distribution": {"name": "core.multinoulli"}}
    if kind == "discrete":
        if _is_date(dtype):
            raise SyntheseError(f"Kolom '{series.name}' is een datum en kan niet 'discrete' zijn.")
        source = series.cast(pl.Float64).round(0) if _is_float(dtype) else series
        return _cast_checked(source, pl.Int64(), kind), None
    if kind == "continuous":
        if _is_date(dtype):
            raise SyntheseError(
                f"Kolom '{series.name}' is een datum en kan niet 'continuous' zijn."
            )
        return _cast_checked(series, pl.Float64(), kind), None
    if kind == "date":
        if _is_date(dtype):
            return series, None
        if _is_text(dtype):
            return _cast_checked(series.cast(pl.String), pl.Date(), kind), None
        raise SyntheseError(f"Kolom '{series.name}' kan niet als 'date' worden gebruikt.")
    if kind == "key":
        if dtype.is_integer() or (_is_float(dtype) and _all_integral(series)):
            prepared = series.cast(pl.Int64)
        elif _is_text(dtype):
            prepared = series.cast(pl.String)
            pattern = code_pattern(prepared)
            if pattern is not None:
                return prepared, {"distribution": _regex_spec(pattern, unique=True)}
        else:
            raise SyntheseError(f"Kolom '{series.name}' kan niet als 'key' worden gebruikt.")
        return prepared, {"distribution": {"unique": True}}
    if kind == "text":
        prepared = series.cast(pl.String)
        pattern = code_pattern(prepared)
        return prepared, None if pattern is None else {"distribution": _regex_spec(pattern)}
    if kind == "faker":
        if not _is_text(dtype):
            raise SyntheseError(f"Kolom '{series.name}' is geen tekst en kan niet 'faker' zijn.")
        if not faker_type:
            raise SyntheseError(f"Kolom '{series.name}' heeft soort 'faker' maar geen provider.")
        spec = {
            "name": "core.faker",
            "parameters": {"faker_type": faker_type, "locale": FAKER_LOCALE},
        }
        return series.cast(pl.String), {"distribution": spec}
    if kind == "freetext":
        raise SyntheseError(f"Kolom '{series.name}' is vrije tekst en gaat niet in het model.")
    raise SyntheseError(f"Onbekende soort '{kind}'.")


def _regex_spec(pattern: str, *, unique: bool = False) -> dict[str, Any]:
    spec: dict[str, Any] = {"name": "core.regex", "parameters": {"regex_data": pattern}}
    if unique:
        spec["unique"] = True
    return spec


def _dtype_name(dtype: pl.DataType) -> str:
    return type(dtype).__name__ if isinstance(dtype, pl.DataType) else str(dtype)


def dtype_to_dict(dtype: pl.DataType) -> dict[str, Any]:
    """Beschrijf een polars-datatype als JSON, zonder waarden.

    Een ``Enum`` wordt als ``Categorical`` bewaard: de lijst met categorieën zijn waarden uit de
    data en hoort niet in het model.
    """
    if isinstance(dtype, pl.Datetime):
        return {"naam": "Datetime", "time_unit": dtype.time_unit, "time_zone": dtype.time_zone}
    if isinstance(dtype, pl.Decimal):
        return {"naam": "Decimal", "precision": dtype.precision, "scale": dtype.scale}
    if isinstance(dtype, pl.Enum):
        return {"naam": "Categorical"}
    name = _dtype_name(dtype)
    if name not in _SIMPLE_DTYPES:
        raise SyntheseError(f"Datatype {name} wordt niet ondersteund.")
    return {"naam": name}


def dtype_from_dict(data: dict[str, Any]) -> pl.DataType:
    """Tegenhanger van :func:`dtype_to_dict`."""
    name = data.get("naam")
    if name == "Datetime":
        return pl.Datetime(time_unit=data.get("time_unit") or "us", time_zone=data.get("time_zone"))
    if name == "Decimal":
        return pl.Decimal(precision=data.get("precision"), scale=data.get("scale") or 0)
    if name in _SIMPLE_DTYPES:
        return _SIMPLE_DTYPES[name]
    raise SyntheseError(f"Het model noemt een onbekend datatype ({name}).")


def dtype_label(dtype: pl.DataType) -> str:
    """Korte naam van een datatype voor rapporten (zonder categorieën van een ``Enum``)."""
    return dtype_to_dict(dtype)["naam"] if _dtype_name(dtype) != "Null" else "Null"


def restore_column(series: pl.Series, dtype: pl.DataType) -> tuple[pl.Series, int]:
    """Zet een synthetische kolom terug naar het brontype.

    Geeft de kolom terug en het aantal waarden dat niet om te zetten was (die worden leeg).
    """
    if series.dtype == dtype:
        return series, 0
    source = series
    if dtype.is_integer() and _is_float(series.dtype):
        source = series.round(0)
    if isinstance(dtype, pl.Categorical):
        restored = source.cast(pl.String).cast(pl.Categorical)
    else:
        restored = source.cast(dtype, strict=False)
    return restored, restored.null_count() - series.null_count()
