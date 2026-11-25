"""Privacy: disclosure control bij het fitten en de controle op letterlijke kopieën."""

from __future__ import annotations

from importlib.metadata import version
from typing import Any

import polars as pl
from metasyncontrib.disclosure import DisclosurePrivacy

PLUGIN_NAME = "metasyn-disclosure"


def disclosure_privacy() -> DisclosurePrivacy:
    """De standaardinstelling van de disclosure-control-plugin (CBS-vuistregels)."""
    return DisclosurePrivacy()


def disclosure_info(privacy: DisclosurePrivacy | None) -> dict[str, Any]:
    """Beschrijving voor het model en het rapport: stond de plugin aan, welke versie."""
    if privacy is None:
        return {"gebruikt": False, "plugin": PLUGIN_NAME}
    return {
        "gebruikt": True,
        "plugin": PLUGIN_NAME,
        "versie": version(PLUGIN_NAME),
        "parameters": privacy.to_dict()["parameters"],
    }


def count_literal_copies(real: pl.DataFrame, synthetic: pl.DataFrame) -> int:
    """Tel synthetische rijen die gelijk zijn aan een rij die in de echte data uniek is.

    Een combinatie die bij veel mensen in de echte data voorkomt (bijv. M + Likert 3) is geen
    kopie van een persoon. Een rij die in de echte data maar één keer voorkomt wel. Vergeleken
    wordt op de kolommen die in beide tabellen met hetzelfde type staan. Lege waarden tellen als
    gelijk. Het resultaat is alleen een aantal.
    """
    columns = [
        name
        for name in real.columns
        if name in synthetic.columns and real.schema[name] == synthetic.schema[name]
    ]
    if not columns or real.height == 0 or synthetic.height == 0:
        return 0
    unique_rows = (
        real.select(columns).group_by(columns).len(name="__n").filter(pl.col("__n") == 1)
    ).drop("__n")
    if unique_rows.height == 0:
        return 0
    copies = synthetic.select(columns).join(unique_rows, on=columns, how="semi", nulls_equal=True)
    return copies.height
