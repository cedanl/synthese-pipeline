"""De twee stappen van de pipeline als Python-functies: ``run_fit`` en ``run_synthesize``.

De commandoregel (``cli.py``) en een notebook (bijv. in Microsoft Fabric) roepen deze aan.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from synthese_pipeline.column_types import parse_type_overrides
from synthese_pipeline.errors import SyntheseError
from synthese_pipeline.model import (
    column_kinds,
    expected_schemas,
    fit_tables,
    load_model,
    save_model,
    synthesize_tables,
    tool_info,
)
from synthese_pipeline.report import build_report, default_report_path, write_report
from synthese_pipeline.storage import (
    is_remote,
    is_single_file_target,
    read_tables,
    resolve_format,
    write_tables,
)

log = logging.getLogger("synthese_pipeline")

MAX_SEED = 2**32 - 1


def _log_tables(tables: dict[str, pl.DataFrame], what: str) -> None:
    log.info("%s: %d tabel(len)", what, len(tables))
    for name, df in tables.items():
        log.info("tabel '%s': %d rijen, %d kolommen", name, df.height, df.width)


def run_fit(
    source: str | Path,
    model_path: str | Path,
    *,
    types: list[str] | None = None,
    disclosure: bool = True,
    storage_options: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Lees de bron, fit het model en schrijf alleen het GMF-bestand."""
    overrides = parse_type_overrides(types)
    tables = read_tables(source, storage_options=storage_options)
    _log_tables(tables, "bron gelezen")
    log.info("disclosure control: %s", "aan" if disclosure else "UIT")
    gmf = fit_tables(tables, overrides=overrides, disclosure=disclosure)
    save_model(gmf, model_path)
    log.info("model geschreven met %d tabel(len)", len(gmf["tables"]))
    return gmf


@dataclass
class SynthesizeResult:
    """Uitkomst van :func:`run_synthesize`."""

    report: dict[str, Any]
    report_path: Path
    written: dict[str, str]

    @property
    def exit_code(self) -> int:
        return int(self.report["exitcode"])


def run_synthesize(
    model_path: str | Path,
    target: str | Path,
    *,
    rows: int | None = None,
    seed: int | None = None,
    fmt: str | None = None,
    source: str | Path | None = None,
    report_path: str | Path | None = None,
    storage_options: dict[str, str] | None = None,
) -> SynthesizeResult:
    """Maak synthetische data uit alleen het GMF, schrijf die weg met een rapport.

    Met ``source`` wordt de synthetische data ook met de echte data vergeleken (verdelingen,
    letterlijke kopieën). Zonder ``seed`` wordt er een gekozen en in het rapport gezet.
    """
    if rows is not None and rows < 1:
        raise SyntheseError("--rows moet minstens 1 zijn.")
    if seed is not None and not 0 <= seed <= MAX_SEED:
        raise SyntheseError(f"--seed moet tussen 0 en {MAX_SEED} liggen.")
    resolve_format(target, fmt)
    if report_path is None:
        if is_remote(target):
            raise SyntheseError("Geef bij een abfss://-doel met --report op waar het rapport komt.")
        report_path = default_report_path(target, is_single_file_target(target))

    gmf = load_model(model_path)
    real = None
    if source is not None:
        real = read_tables(source, storage_options=storage_options)
        _log_tables(real, "bron gelezen voor de vergelijking")

    used_seed = seed if seed is not None else secrets.randbelow(MAX_SEED)
    synthetic = synthesize_tables(gmf, rows=rows, seed=used_seed)
    written = write_tables(synthetic, target, fmt=fmt, storage_options=storage_options)
    log.info("synthetische data geschreven: %d tabel(len)", len(written))

    report = build_report(
        gmf_tool_info=tool_info(gmf),
        expected=expected_schemas(gmf),
        kinds=column_kinds(gmf),
        synthetic=synthetic,
        real=real,
        seed=used_seed,
    )
    write_report(report, report_path)
    log.info(
        "rapport geschreven; structuur gelijk: %s; letterlijke kopieën: %s; exitcode %d",
        "ja" if report["structuur"]["gelijk"] else "nee",
        "niet gecontroleerd"
        if report["letterlijke_kopieen"] is None
        else report["letterlijke_kopieen"],
        report["exitcode"],
    )
    return SynthesizeResult(report=report, report_path=Path(report_path), written=written)
