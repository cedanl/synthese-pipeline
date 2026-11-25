"""Het model (GMF-bestand): fitten, opslaan, laden en synthetiseren met metasyn.

Een GMF-bestand van synthese-pipeline is een gewoon metasyn-GMF (versie 2.1, met één of meer
tabellen) plus een blok ``synthese_pipeline`` met per kolom de soort en het brontype, en of
disclosure control aan stond. Alleen dit bestand hoeft de beveiligde omgeving te verlaten.
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

from synthese_pipeline import __version__
from synthese_pipeline.column_types import (
    check_overrides,
    dtype_from_dict,
    dtype_to_dict,
    kinds_for_table,
    prepare_column,
    restore_column,
)
from synthese_pipeline.errors import SyntheseError
from synthese_pipeline.privacy import disclosure_info, disclosure_privacy

TOOL_KEY = "synthese_pipeline"
GMF_VERSION = "2.1"

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
    disclosure: bool = True,
) -> dict[str, Any]:
    """Fit per tabel een metasyn-model en geef het GMF als dict terug."""
    overrides = overrides or {}
    check_overrides(overrides, tables)
    privacy = disclosure_privacy() if disclosure else None
    disclosure_block = disclosure_info(privacy)

    gmf_tables: list[dict[str, Any]] = []
    table_info: dict[str, Any] = {}
    for name, df in tables.items():
        kinds = kinds_for_table(name, df, overrides)
        columns, var_specs, column_info = [], [], {}
        for column in df.columns:
            prepared, spec = prepare_column(df[column], kinds[column])
            columns.append(prepared)
            if spec is not None:
                var_specs.append({"name": column, **spec})
            column_info[column] = {
                "soort": kinds[column],
                "brontype": dtype_to_dict(df.schema[column]),
            }
            log.info("tabel '%s', kolom '%s': soort %s", name, column, kinds[column])
        fit_df = pl.DataFrame(columns)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metaframe = MetaFrame.fit_dataframe(
                fit_df, var_specs=var_specs, privacy=privacy, progress_bar=False, name=name
            )
        _log_warnings(name, caught, "het fitten")
        gmf_tables.extend(_jsonable(metaframe.to_dict())["tables"])
        table_info[name] = {"kolommen": column_info}
        log.info("tabel '%s': %d rijen, %d kolommen gefit", name, df.height, df.width)

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


def save_model(gmf: dict[str, Any], path: str | Path) -> None:
    """Schrijf het GMF naar een JSON-bestand."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(gmf, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_model(path: str | Path) -> dict[str, Any]:
    """Lees en controleer een GMF-bestand."""
    source = Path(path)
    if not source.is_file():
        raise SyntheseError(f"Het model bestaat niet: {source}")
    try:
        with source.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SyntheseError(f"Het model is geen geldig JSON-bestand: {source}") from exc
    try:
        return parse_gmf_dict(raw, validate=True)
    except Exception as exc:
        raise SyntheseError(f"Het model is geen geldig GMF-bestand: {source}") from exc


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
    return result
