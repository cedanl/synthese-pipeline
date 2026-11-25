"""Tabellen inlezen en wegschrijven: een map, een lakehouse-tabel (Delta), Parquet of CSV.

Alles gaat via polars; pandas wordt niet gebruikt. CSV-bestanden worden eerst als tekst gelezen
en daarna per kolom omgezet (zie :func:`infer_text_column`), zodat codes met voorloopnullen
(``01234``) tekst blijven.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path, PureWindowsPath
from typing import TypeGuard

import polars as pl

from synthese_pipeline.errors import SyntheseError

ABFSS_PREFIX = "abfss://"
TOKEN_ENV = "SYNTHESE_PIPELINE_TOKEN"
OUTPUT_FORMATS = ("delta", "parquet", "csv")
FILE_SUFFIXES = {".csv": "csv", ".parquet": "parquet"}
CSV_SEPARATORS = (";", ",")
DEFAULT_CSV_SEPARATOR = ";"

_LEADING_ZERO = r"^[+-]?0\d+$"
_DECIMAL_POINT = r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$"
_DECIMAL_COMMA = r"^[+-]?\d*,?\d+$"
_DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y")
_DATETIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
)


def is_remote(path: str | Path) -> bool:
    """Geeft aan of een pad naar OneLake/Azure wijst (``abfss://``)."""
    return str(path).startswith(ABFSS_PREFIX)


def fabric_storage_options(bearer_token: str | None = None) -> dict[str, str]:
    """Bouw de ``storage_options`` voor een ``abfss://``-pad in Microsoft Fabric.

    Het token komt uit ``bearer_token`` of anders uit de omgevingsvariabele
    ``SYNTHESE_PIPELINE_TOKEN`` (in een Fabric-notebook:
    ``notebookutils.credentials.getToken("storage")``). Zonder token blijft alleen
    ``use_fabric_endpoint`` staan; delta-rs zoekt dan zelf naar Azure-inloggegevens.
    """
    token = bearer_token if bearer_token is not None else os.environ.get(TOKEN_ENV)
    options = {"use_fabric_endpoint": "true"}
    if token:
        options["bearer_token"] = token
    return options


def _is_safe_table_name(name: object) -> TypeGuard[str]:
    if not isinstance(name, str) or name in ("", ".", ".."):
        return False
    if any(char in name for char in ("/", "\\", "\0")):
        return False
    return not PureWindowsPath(name).drive


def check_table_names(names: Iterable[object]) -> None:
    """Weiger tabelnamen die geen gewone map- of bestandsnaam zijn.

    Bij het wegschrijven wordt een tabelnaam een map- of bestandsnaam in het doel. De namen
    komen uit het model, en dat is een JSON-bestand dat met de hand kan zijn aangepast. Zonder
    deze controle zou een naam als ``../bron`` of een absoluut pad buiten het doel schrijven.
    Een naam mag dus niet leeg, ``.`` of ``..`` zijn, geen ``/``, ``\\`` of NUL-teken bevatten
    en niet met een schijfletter beginnen; elke naam komt één keer voor. De melding noemt
    alleen het volgnummer van de tabel, niet de naam.
    """
    seen: set[str] = set()
    for number, name in enumerate(names, start=1):
        if not _is_safe_table_name(name):
            raise SyntheseError(
                f"Tabel {number} heeft een ongeldige naam. Een tabelnaam wordt een map- of "
                "bestandsnaam in het doel: niet leeg, niet '.' of '..', zonder '/', '\\' of "
                "NUL-teken en zonder schijfletter."
            )
        if name in seen:
            raise SyntheseError(f"Tabel {number} heeft dezelfde naam als een eerdere tabel.")
        seen.add(name)


def table_name_from_uri(uri: str) -> str:
    """Laatste deel van een pad of URI, bijvoorbeeld ``studenten`` voor ``.../Tables/studenten``."""
    parts = [part for part in uri.rstrip("/").split("/") if part]
    if not parts:
        raise SyntheseError("Kan geen tabelnaam afleiden uit het opgegeven pad.")
    return parts[-1]


def _is_delta(path: Path) -> bool:
    return path.is_dir() and (path / "_delta_log").is_dir()


def read_tables(
    source: str | Path, *, storage_options: dict[str, str] | None = None
) -> dict[str, pl.DataFrame]:
    """Lees alle tabellen uit een bron.

    De bron is een CSV- of Parquet-bestand, een Delta-tabel (een map met ``_delta_log``), een
    map met zulke tabellen, of een ``abfss://``-pad naar één Delta-tabel. De sleutel van het
    resultaat is de tabelnaam (bestandsnaam zonder extensie of mapnaam).
    """
    if is_remote(source):
        uri = str(source)
        options = storage_options if storage_options is not None else fabric_storage_options()
        return {table_name_from_uri(uri): pl.read_delta(uri, storage_options=options)}

    path = Path(source)
    if not path.exists():
        raise SyntheseError(f"De bron bestaat niet: {path}")
    if path.is_file():
        return {path.stem: _read_file(path)}
    if _is_delta(path):
        return {path.name: pl.read_delta(str(path))}

    tables: dict[str, pl.DataFrame] = {}
    for entry in sorted(path.iterdir()):
        if entry.name.startswith((".", "_")):
            continue
        if entry.is_dir():
            if not _is_delta(entry):
                continue
            name, reader = entry.name, _read_delta_dir
        elif entry.suffix.lower() in FILE_SUFFIXES:
            name, reader = entry.stem, _read_file
        else:
            continue
        if name in tables:
            raise SyntheseError(
                f"Tabel '{name}' staat meer dan één keer in de bron (bijvoorbeeld als CSV en "
                "als Parquet). Houd één versie over."
            )
        tables[name] = reader(entry)
    if not tables:
        raise SyntheseError(f"Geen tabellen gevonden in {path} (verwacht: Delta, Parquet of CSV).")
    return tables


def _read_delta_dir(path: Path) -> pl.DataFrame:
    return pl.read_delta(str(path))


def _read_file(path: Path) -> pl.DataFrame:
    kind = FILE_SUFFIXES.get(path.suffix.lower())
    if kind == "csv":
        return read_csv(path)
    if kind == "parquet":
        return pl.read_parquet(path)
    raise SyntheseError(
        f"Bestandstype '{path.suffix}' wordt niet ondersteund (gebruik CSV, Parquet of Delta)."
    )


def _sniff_separator(path: Path) -> str:
    with path.open("rb") as handle:
        header = handle.readline().decode("utf-8", errors="ignore")
    counts = {sep: header.count(sep) for sep in (",", ";", "\t")}
    best = max(counts, key=lambda sep: counts[sep])
    return best if counts[best] > 0 else ","


def read_csv(path: Path) -> pl.DataFrame:
    """Lees een CSV-bestand en leid per kolom het type af.

    Het scheidingsteken (komma, puntkomma of tab) wordt uit de kopregel afgeleid. Bij een
    puntkomma mag een kommagetal decimalen bevatten (``7,5``).
    """
    separator = _sniff_separator(path)
    raw = pl.read_csv(path, separator=separator, infer_schema=False)
    decimal_comma = separator == ";"
    return raw.with_columns(
        infer_text_column(raw[name], decimal_comma=decimal_comma) for name in raw.columns
    )


def infer_text_column(series: pl.Series, *, decimal_comma: bool = False) -> pl.Series:
    """Zet een tekstkolom om naar het best passende type.

    Volgorde: tekst met voorloopnullen blijft tekst, dan waar/onwaar, geheel getal,
    kommagetal, datum, datum met tijd. Past niets, dan blijft het tekst. Een type wordt alleen
    gekozen als élke ingevulde waarde past.
    """
    values = series.str.strip_chars()
    filled = values.drop_nulls()
    n_missing = series.null_count()
    if filled.len() == 0:
        return series
    if filled.str.contains(_LEADING_ZERO).any():
        return series

    lower = values.str.to_lowercase()
    if lower.drop_nulls().is_in(["true", "false"]).all():
        return (lower == "true").alias(series.name)

    integers = values.cast(pl.Int64, strict=False)
    if integers.null_count() == n_missing:
        return integers

    if filled.str.contains(_DECIMAL_POINT).all():
        return values.cast(pl.Float64, strict=False)
    if decimal_comma and filled.str.contains(_DECIMAL_COMMA).all():
        return values.str.replace(",", ".", literal=True).cast(pl.Float64, strict=False)

    for fmt in _DATE_FORMATS:
        dates = values.str.to_date(fmt, strict=False)
        if dates.null_count() == n_missing:
            return dates
    for fmt in _DATETIME_FORMATS:
        stamps = values.str.to_datetime(fmt, strict=False)
        if stamps.null_count() == n_missing:
            return stamps
    return series


def parse_formats(fmt: str) -> tuple[str, ...]:
    """Lees een formaat of een lijst met komma's (``parquet,csv``) als tuple.

    De volgorde is die van :data:`OUTPUT_FORMATS` (Parquet voor CSV), niet die van de invoer.
    Een onbekend of leeg formaat, een formaat dat twee keer voorkomt of Delta samen met een
    ander formaat geeft een fout.
    """
    kinds = [part.strip().lower() for part in fmt.split(",")]
    for kind in kinds:
        if kind not in OUTPUT_FORMATS:
            raise SyntheseError(
                f"Onbekend formaat '{kind}'. Kies uit: {', '.join(OUTPUT_FORMATS)}, of "
                "parquet,csv voor allebei."
            )
    if len(set(kinds)) != len(kinds):
        raise SyntheseError(f"Een formaat staat meer dan één keer in '{fmt}'.")
    if "delta" in kinds and len(kinds) > 1:
        raise SyntheseError("Delta kan niet samen met een ander formaat worden geschreven.")
    return tuple(kind for kind in OUTPUT_FORMATS if kind in kinds)


def resolve_formats(target: str | Path, fmt: str | None) -> tuple[str, ...]:
    """Bepaal de uitvoerformaten: uit ``fmt``, de extensie van het doel, of Delta.

    Meer dan één formaat kan alleen bij een lokale map als doel; naar een ``abfss://``-pad gaat
    alleen Delta.
    """
    suffix_kind = None if is_remote(target) else FILE_SUFFIXES.get(Path(target).suffix.lower())
    if fmt is None:
        return (suffix_kind or "delta",)
    kinds = parse_formats(fmt)
    if len(kinds) > 1 and (is_remote(target) or suffix_kind is not None):
        raise SyntheseError(
            f"Het formaat '{','.join(kinds)}' geeft meer dan één bestand per tabel; geef een "
            "lokale map als doel."
        )
    if is_remote(target) and kinds != ("delta",):
        raise SyntheseError("Naar een abfss://-pad kan alleen Delta worden geschreven.")
    if suffix_kind is not None and kinds != (suffix_kind,):
        raise SyntheseError(
            f"Het doel eindigt op '{Path(target).suffix}' maar het formaat is '{kinds[0]}'."
        )
    return kinds


def resolve_format(target: str | Path, fmt: str | None) -> str:
    """Als :func:`resolve_formats`, met de formaten als tekst (``delta`` of ``parquet,csv``)."""
    return ",".join(resolve_formats(target, fmt))


def check_csv_separator(separator: str) -> str:
    """Geef het scheidingsteken terug als het ``;`` of ``,`` is; anders een fout."""
    if separator not in CSV_SEPARATORS:
        raise SyntheseError(
            "Het scheidingsteken voor CSV moet ';' of ',' zijn, niet een ander teken."
        )
    return separator


def is_single_file_target(target: str | Path) -> bool:
    """Een doel met extensie .csv of .parquet is één bestand; anders is het een map."""
    return not is_remote(target) and Path(target).suffix.lower() in FILE_SUFFIXES


def write_tables(
    tables: dict[str, pl.DataFrame],
    target: str | Path,
    *,
    fmt: str | None = None,
    csv_separator: str = DEFAULT_CSV_SEPARATOR,
    storage_options: dict[str, str] | None = None,
) -> dict[str, str]:
    """Schrijf tabellen weg en geef per tabel de plek terug.

    - ``uit.csv`` of ``uit.parquet``: één bestand (alleen bij één tabel).
    - Een map: per tabel een Delta-tabel ``<map>/<tabel>`` (standaard) of
      ``<map>/<tabel>.parquet`` / ``<map>/<tabel>.csv``. Met ``fmt="parquet,csv"`` komen
      beide bestanden in de map; de teruggegeven plek is dan die van het Parquet-bestand.
    - ``abfss://``: per tabel een Delta-tabel ``<pad>/<tabel>``.

    CSV is UTF-8 (zonder BOM) met ``csv_separator`` (``;`` of ``,``) als scheidingsteken;
    tekst staat tussen aanhalingstekens, getallen, datums en lege waarden niet.

    Bestaande uitvoer wordt overschreven. Er wordt nooit buiten het doel geschreven: een
    ongeldige tabelnaam (zie :func:`check_table_names`) of een symbolische link in de doelmap
    die naar buiten wijst, geeft een fout voordat er iets is geschreven.
    """
    check_table_names(tables)
    kinds = resolve_formats(target, fmt)
    check_csv_separator(csv_separator)
    if is_remote(target):
        options = storage_options if storage_options is not None else fabric_storage_options()
        base = str(target).rstrip("/")
        written = {}
        for name, df in tables.items():
            uri = f"{base}/{name}"
            _write_delta(df, uri, options)
            written[name] = uri
        return written

    path = Path(target)
    if is_single_file_target(path):
        if len(tables) != 1:
            raise SyntheseError(
                f"Het model bevat {len(tables)} tabellen; geef een map als doel in plaats van "
                "één bestand."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        (df,) = tables.values()
        (kind,) = kinds
        _write_file(df, path, kind, csv_separator)
        return {next(iter(tables)): str(path)}

    outs = {
        name: [(kind, path / (name if kind == "delta" else f"{name}.{kind}")) for kind in kinds]
        for name in tables
    }
    root = path.resolve()
    for number, entries in enumerate(outs.values(), start=1):
        if any(out.resolve().parent != root for _, out in entries):
            raise SyntheseError(
                f"Tabel {number} zou buiten de doelmap {path} komen (via een symbolische link). "
                "Kies een ander doel."
            )
    path.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        for kind, out in outs[name]:
            if kind == "delta":
                _write_delta(df, str(out), None)
            else:
                _write_file(df, out, kind, csv_separator)
    return {name: str(entries[0][1]) for name, entries in outs.items()}


def _write_delta(df: pl.DataFrame, uri: str, storage_options: dict[str, str] | None) -> None:
    df.write_delta(
        uri,
        mode="overwrite",
        storage_options=storage_options,
        delta_write_options={"schema_mode": "overwrite"},
    )


def _quote(text: str) -> str:
    return '"' + text.replace('"', '""') + '"'


def _write_file(
    df: pl.DataFrame, path: Path, kind: str, separator: str = DEFAULT_CSV_SEPARATOR
) -> None:
    if kind == "csv":
        _write_csv(df, path, separator)
    else:
        df.write_parquet(path)


def _write_csv(df: pl.DataFrame, path: Path, separator: str) -> None:
    """Schrijf CSV volgens de dataconventies: UTF-8, ``separator``, tekst tussen aanhalingstekens.

    polars' ``quote_style="non_numeric"`` zet ook datums en waar/onwaar tussen
    aanhalingstekens. Daarom krijgen hier alleen tekst (elke kolom die geen getal, datum, tijd of
    waar/onwaar is) en de kolomnamen aanhalingstekens, met ``""`` voor een aanhalingsteken in de
    tekst; de rest schrijft polars zonder. Een lege waarde blijft leeg, lege tekst wordt ``""``.
    """
    text_columns = [
        name
        for name, dtype in df.schema.items()
        if not (dtype.is_numeric() or dtype.is_temporal() or dtype in (pl.Boolean, pl.Null))
    ]
    quoted = df.with_columns(
        pl.concat_str(
            [
                pl.lit('"'),
                pl.col(name).cast(pl.String).str.replace_all('"', '""', literal=True),
                pl.lit('"'),
            ]
        ).alias(name)
        for name in text_columns
    ).rename({name: _quote(name) for name in df.columns})
    quoted.write_csv(path, separator=separator, quote_style="never")
