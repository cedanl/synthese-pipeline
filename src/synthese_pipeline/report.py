"""Rapport en exitcode van een ``synthesize``-run.

Het rapport bevat alleen aantallen, scores, kolomnamen, tabelnamen en datatypes: nooit waarden
uit de echte of de synthetische data. Kolommen die het model uitsluit (vrije tekst) staan per
tabel onder ``uitgesloten`` en tellen niet mee in de structuurcontrole. Bij relaties tussen
tabellen telt ``relaties`` per relatie de wezen (verwijzende waarden zonder sleutel) en dubbele
sleutels; ``beperkingen`` zegt wat de synthetische data dan niet nabootst.
"""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import polars as pl
from scipy.stats import wasserstein_distance

from synthese_pipeline import __version__
from synthese_pipeline.column_types import dtype_label
from synthese_pipeline.privacy import count_literal_copies

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_STRUCTURE = 3
EXIT_COPIES = 4
EXIT_RELATIONS = 5

EXIT_CODES = {
    EXIT_OK: "geslaagd",
    EXIT_ERROR: "fout (invoer, model of omgeving)",
    EXIT_USAGE: "verkeerd gebruik van de commandoregel",
    EXIT_STRUCTURE: "structuur wijkt af van de bron of het model",
    EXIT_COPIES: "letterlijke kopieën van echte rijen gevonden",
    EXIT_RELATIONS: "relaties tussen tabellen kloppen niet (wezen of dubbele sleutels)",
}

RELATION_LIMITATION = (
    "Relaties: het aantal rijen per sleutel (bijv. inschrijvingen per student) volgt niet de "
    "echte verdeling. metasyn trekt de verwijzende sleutels gelijkmatig met teruglegging; "
    "gemeten op dummydata: hoogstens 9 rijen per sleutel tegen 5 in de bron. Lege "
    "verwijzingen in de bron komen niet terug."
)

_DISTANCE_KINDS = {"discrete", "continuous", "date"}


def pair_tables(
    expected: dict[str, Any], actual: dict[str, Any]
) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """Koppel tabellen op naam. Bij precies één tabel aan beide kanten tellen de namen niet.

    Geeft (paren, ontbrekend in ``actual``, extra in ``actual``).
    """
    if len(expected) == 1 and len(actual) == 1:
        return [(next(iter(expected)), next(iter(actual)))], [], []
    pairs = [(name, name) for name in expected if name in actual]
    missing = [name for name in expected if name not in actual]
    extra = [name for name in actual if name not in expected]
    return pairs, missing, extra


def check_structure(
    expected: dict[str, pl.DataType],
    synthetic: pl.DataFrame,
    *,
    real: pl.DataFrame | None = None,
    kinds: dict[str, str] | None = None,
) -> list[str]:
    """Vergelijk kolommen, volgorde en datatypes; met echte data ook categorieën en lege waarden.

    Geeft een lijst met afwijkingen (leeg = gelijk).
    """
    kinds = kinds or {}
    deviations = []
    for column in expected:
        if column not in synthetic.columns:
            deviations.append(f"kolom '{column}' ontbreekt")
    for column in synthetic.columns:
        if column not in expected:
            deviations.append(f"kolom '{column}' is extra")
    common = [column for column in expected if column in synthetic.columns]
    if [c for c in synthetic.columns if c in expected] != common:
        deviations.append("de volgorde van de kolommen wijkt af")
    for column in common:
        want, got = expected[column], synthetic.schema[column]
        if dtype_label(want) != dtype_label(got) or (isinstance(want, pl.Datetime) and want != got):
            deviations.append(
                f"kolom '{column}': type {dtype_label(want)} verwacht, {dtype_label(got)} gevonden"
            )
    if real is None:
        return deviations
    for column in common:
        if column not in real.columns or real.schema[column] != synthetic.schema[column]:
            continue
        if real[column].null_count() == 0 and synthetic[column].null_count() > 0:
            deviations.append(
                f"kolom '{column}': {synthetic[column].null_count()} lege waarden, "
                "de bron heeft er geen"
            )
        if kinds.get(column) == "categorical":
            n_new = (
                synthetic.select(column)
                .drop_nulls()
                .unique()
                .join(real.select(column).drop_nulls().unique(), on=column, how="anti")
                .height
            )
            if n_new:
                deviations.append(
                    f"kolom '{column}': {n_new} categorieën die niet in de bron voorkomen"
                )
    return deviations


def _as_number(series: pl.Series) -> pl.Series:
    dtype = series.dtype
    if isinstance(dtype, pl.Date):
        return series.cast(pl.Int32).cast(pl.Float64)
    if isinstance(dtype, pl.Datetime):
        return series.dt.epoch("s").cast(pl.Float64) / 86400
    if isinstance(dtype, pl.Time):
        return series.cast(pl.Int64).cast(pl.Float64) / 3.6e12
    return series.cast(pl.Float64)


def tv_distance(real: pl.Series, synthetic: pl.Series) -> float | None:
    """Totale-variatieafstand tussen de verdelingen van twee kolommen (0 = gelijk, 1 = disjunct).

    Lege waarden tellen niet mee; die staan apart in het rapport.
    """
    real_counts = real.drop_nulls().cast(pl.String).value_counts(name="n_echt")
    syn_counts = synthetic.drop_nulls().cast(pl.String).value_counts(name="n_syn")
    n_real, n_syn = real_counts["n_echt"].sum(), syn_counts["n_syn"].sum()
    if not n_real or not n_syn:
        return None
    name = real_counts.columns[0]
    joined = real_counts.join(
        syn_counts.rename({syn_counts.columns[0]: name}), on=name, how="full", coalesce=True
    ).fill_null(0)
    diff = (joined["n_echt"] / n_real - joined["n_syn"] / n_syn).abs().sum()
    return float(diff) / 2


def column_measure(real: pl.Series, synthetic: pl.Series, kind: str | None) -> dict[str, Any]:
    """Verdelingsmaat voor één kolom: Wasserstein voor getallen/datums, anders TV-afstand."""
    result: dict[str, Any] = {
        "soort": kind,
        "aandeel_leeg_echt": _null_share(real),
        "aandeel_leeg_synthetisch": _null_share(synthetic),
    }
    if kind == "key":
        result.update(maat=None, waarde=None, toelichting="sleutel: geen verdeling vergeleken")
        return result
    if kind == "text":
        result.update(maat=None, waarde=None, toelichting="tekst: geen verdeling vergeleken")
        return result
    if kind == "faker":
        result.update(
            maat=None, waarde=None, toelichting="verzonnen met Faker: geen verdeling vergeleken"
        )
        return result
    if (
        kind in _DISTANCE_KINDS
        and (real.dtype.is_numeric() or real.dtype.is_temporal())
        and (synthetic.dtype.is_numeric() or synthetic.dtype.is_temporal())
    ):
        left = _as_number(real.drop_nulls())
        right = _as_number(synthetic.drop_nulls())
        value = (
            float(wasserstein_distance(left.to_numpy(), right.to_numpy()))
            if left.len() and right.len()
            else None
        )
        result.update(maat="wasserstein_dagen" if kind == "date" else "wasserstein", waarde=value)
        return result
    result.update(maat="tv_afstand", waarde=tv_distance(real, synthetic))
    return result


def _null_share(series: pl.Series) -> float | None:
    return None if series.len() == 0 else series.null_count() / series.len()


def table_report(
    expected: dict[str, pl.DataType],
    synthetic: pl.DataFrame,
    *,
    real: pl.DataFrame | None,
    kinds: dict[str, str],
    excluded: list[str] | None = None,
) -> dict[str, Any]:
    """Rapport voor één tabel. ``excluded``: kolommen die het model bewust weglaat."""
    excluded = excluded or []
    if real is not None:
        real = real.drop([column for column in excluded if column in real.columns])
    reference = expected if real is None else dict(real.schema)
    deviations = check_structure(reference, synthetic, real=real, kinds=kinds)
    entry: dict[str, Any] = {
        "rijen": synthetic.height,
        "kolommen": synthetic.width,
        "uitgesloten": list(excluded),
        "structuur": {"gelijk": not deviations, "afwijkingen": deviations},
    }
    if real is None:
        entry["verdelingen"] = None
        entry["letterlijke_kopieen"] = None
        return entry
    entry["verdelingen"] = {
        column: column_measure(real[column], synthetic[column], kinds.get(column))
        for column in synthetic.columns
        if column in real.columns
    }
    entry["letterlijke_kopieen"] = count_literal_copies(real, synthetic)
    return entry


def _relation_text(rel: dict[str, Any]) -> str:
    return (
        f"{rel['foreign_table']}[{rel['foreign_key']}] {rel['relation_type']} "
        f"{rel['primary_table']}[{rel['primary_key']}]"
    )


def relation_report(rel: dict[str, Any], synthetic: dict[str, pl.DataFrame]) -> dict[str, Any]:
    """Controleer één relatie in de synthetische data: alleen aantallen, nooit waarden.

    ``wezen``: ingevulde verwijzende waarden die niet in de sleutel staan. ``dubbele_sleutels``:
    ingevulde sleutelwaarden die vaker dan één keer voorkomen (aantal rijen boven het eerste).
    """
    entry: dict[str, Any] = {
        "relatie": _relation_text(rel),
        "tabel": rel["foreign_table"],
        "kolom": rel["foreign_key"],
        "verwijst_naar_tabel": rel["primary_table"],
        "verwijst_naar_kolom": rel["primary_key"],
        "rijen_gecontroleerd": None,
        "lege_verwijzingen": None,
        "wezen": None,
        "dubbele_sleutels": None,
    }
    foreign_df = synthetic.get(rel["foreign_table"])
    primary_df = synthetic.get(rel["primary_table"])
    if (
        foreign_df is None
        or primary_df is None
        or rel["foreign_key"] not in foreign_df.columns
        or rel["primary_key"] not in primary_df.columns
    ):
        entry["toelichting"] = "niet gecontroleerd: tabel of kolom ontbreekt"
        return entry
    foreign = foreign_df[rel["foreign_key"]].cast(pl.String).rename("sleutel")
    primary = primary_df[rel["primary_key"]].cast(pl.String).rename("sleutel").drop_nulls()
    orphans = (
        foreign.drop_nulls()
        .to_frame()
        .join(primary.unique().to_frame(), on="sleutel", how="anti")
        .height
    )
    entry.update(
        rijen_gecontroleerd=foreign.len(),
        lege_verwijzingen=foreign.null_count(),
        wezen=orphans,
        dubbele_sleutels=primary.len() - primary.n_unique(),
    )
    return entry


def build_report(
    *,
    gmf_tool_info: dict[str, Any],
    expected: dict[str, dict[str, pl.DataType]],
    kinds: dict[str, dict[str, str]],
    synthetic: dict[str, pl.DataFrame],
    real: dict[str, pl.DataFrame] | None,
    seed: int,
    excluded: dict[str, list[str]] | None = None,
    relations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Maak het rapport van een run en bepaal de exitcode.

    ``relations`` zijn de relaties uit het GMF (``relations``); de verwijzende kolommen krijgen
    geen verdelingsmaat, want hun waarden komen uit de sleutel.
    """
    excluded = excluded or {}
    relations = relations or []
    kinds = {table: dict(columns) for table, columns in kinds.items()}
    for rel in relations:
        kinds.setdefault(rel["foreign_table"], {})[rel["foreign_key"]] = "key"
    reference: dict[str, Any] = expected if real is None else real
    pairs, missing, extra = pair_tables(reference, synthetic)
    tables: dict[str, Any] = {}
    general: list[str] = [f"tabel '{name}' ontbreekt" for name in missing]
    general += [f"tabel '{name}' is extra" for name in extra]
    for ref_name, syn_name in pairs:
        tables[syn_name] = table_report(
            expected.get(syn_name, {}),
            synthetic[syn_name],
            real=None if real is None else real[ref_name],
            kinds=kinds.get(syn_name, {}),
            excluded=excluded.get(syn_name, []),
        )

    structure_equal = not general and all(t["structuur"]["gelijk"] for t in tables.values())
    copies = (
        None if real is None else sum(int(t["letterlijke_kopieen"] or 0) for t in tables.values())
    )
    relation_entries = [relation_report(rel, synthetic) for rel in relations]
    relations_ok = all(
        not entry["wezen"] and not entry["dubbele_sleutels"] for entry in relation_entries
    )
    if copies:
        code = EXIT_COPIES
    elif not structure_equal:
        code = EXIT_STRUCTURE
    elif not relations_ok:
        code = EXIT_RELATIONS
    else:
        code = EXIT_OK

    disclosure = gmf_tool_info.get("disclosure_control") or {"gebruikt": None}
    report: dict[str, Any] = {
        "synthese_pipeline": __version__,
        "metasyn": version("metasyn"),
        "seed": seed,
        "disclosure_control": disclosure,
        "data_free": bool(gmf_tool_info.get("data_free", False)),
        "bron_vergeleken": real is not None,
        "structuur": {"gelijk": structure_equal, "afwijkingen": general},
        "letterlijke_kopieen": copies,
        "toelichting_kopieen": (
            "niet gecontroleerd: geef --source op"
            if real is None
            else "synthetische rijen die gelijk zijn aan een rij die in de bron uniek is"
        ),
        "tabellen": tables,
        "relaties": relation_entries,
        "exitcode": code,
        "uitkomst": EXIT_CODES[code],
    }
    if relations:
        report["beperkingen"] = [RELATION_LIMITATION]
    return report


def default_report_path(target: str | Path, single_file: bool) -> Path:
    """Naast de data: ``uit.csv`` -> ``uit.rapport.json``; een map -> ``<map>/rapport.json``."""
    path = Path(target)
    if single_file:
        return path.with_name(f"{path.stem}.rapport.json")
    return path / "rapport.json"


def write_report(report: dict[str, Any], path: str | Path) -> None:
    """Schrijf het rapport als JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
