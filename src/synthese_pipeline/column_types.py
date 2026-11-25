"""Kolomtypes afleiden en vertalen naar instellingen voor metasyn.

Elke kolom krijgt een *soort* (afgeleid, eerste regel die past):

- ``categorical``: waar/onwaar, of een geheel getal (max. 12 unieke waarden) of tekst
  (max. 30 unieke waarden) met hooguit half zoveel unieke als ingevulde waarden, bijv.
  Likert 1-5 of M/V. metasyn krijgt ``core.multinoulli``.
- ``key``: geheel getal of tekst zonder lege waarden, alle waarden uniek, minstens 20 rijen.
  metasyn krijgt ``unique``.
- ``discrete``: overige gehele getallen, ook kommagetallen die allemaal op ,0 eindigen.
- ``continuous``: overige kommagetallen.
- ``date``: datum, datum met tijd of tijd.
- ``text``: overige tekst (metasyn kiest zelf).

De regels zijn voor deze repo zelf geschreven. Met ``--type kolom=soort`` is elke soort te
overschrijven. Het oorspronkelijke datatype wordt in het model bewaard en na het synthetiseren
teruggezet, zodat kolomnamen en types gelijk blijven aan de bron.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from synthese_pipeline.errors import SyntheseError

KINDS = ("categorical", "discrete", "continuous", "date", "key", "text")

INT_CATEGORICAL_MAX_UNIQUE = 12
TEXT_CATEGORICAL_MAX_UNIQUE = 30
KEY_MIN_ROWS = 20

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
    if _is_key(series, n_unique):
        return "key"
    if _few(n_unique, n_filled, TEXT_CATEGORICAL_MAX_UNIQUE):
        return "categorical"
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


def check_overrides(overrides: dict[str, str], tables: dict[str, pl.DataFrame]) -> None:
    """Geef een fout als een ``--type`` naar een kolom verwijst die nergens bestaat."""
    known = set()
    for table, df in tables.items():
        known.update(df.columns)
        known.update(f"{table}.{column}" for column in df.columns)
    for name in overrides:
        if name not in known:
            raise SyntheseError(f"--type noemt kolom '{name}', maar die bestaat niet in de bron.")


def kinds_for_table(table: str, df: pl.DataFrame, overrides: dict[str, str]) -> dict[str, str]:
    """Soort per kolom: eerst ``tabel.kolom``, dan ``kolom``, anders afgeleid."""
    kinds = {}
    for column in df.columns:
        override = overrides.get(f"{table}.{column}") or overrides.get(column)
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


def prepare_column(series: pl.Series, kind: str) -> tuple[pl.Series, dict[str, Any] | None]:
    """Zet een kolom klaar voor metasyn en geef de bijbehorende ``var_spec`` (zonder naam)."""
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
        else:
            raise SyntheseError(f"Kolom '{series.name}' kan niet als 'key' worden gebruikt.")
        return prepared, {"distribution": {"unique": True}}
    if kind == "text":
        return series.cast(pl.String), None
    raise SyntheseError(f"Onbekende soort '{kind}'.")


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
