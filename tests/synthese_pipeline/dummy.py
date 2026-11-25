"""Gegenereerde dummydata voor de tests. Nooit echte data."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

SENTINEL = "SENTINEL-ZQX-7391"
OUTLIER_AGE = 71


def students(n: int = 200, seed: int = 1, *, sentinel: bool = True) -> pl.DataFrame:
    """Studententabel met Likert-items met lege waarden, korte categorieën en een uitschieter."""
    rng = np.random.default_rng(seed)

    def likert() -> list[int | None]:
        return [None if rng.random() < 0.1 else int(rng.integers(1, 6)) for _ in range(n)]

    gender = [str(x) for x in rng.choice(["M", "V"], n)]
    if sentinel:
        gender[7] = SENTINEL
    ages = [int(x) for x in rng.integers(18, 31, n)]
    ages[3] = OUTLIER_AGE
    start = dt.date(2020, 9, 1)
    return pl.DataFrame(
        {
            "student_id": pl.Series(range(100_000, 100_000 + n), dtype=pl.Int64),
            "geslacht": pl.Series(gender, dtype=pl.String),
            "q1": pl.Series(likert(), dtype=pl.Int64),
            "q2": pl.Series(likert(), dtype=pl.Int64),
            "opleidingscode": pl.Series(
                [str(x) for x in rng.choice(["01234", "02345", "00456"], n)], dtype=pl.String
            ),
            "leeftijd": pl.Series(ages, dtype=pl.Int64),
            "inschrijfdatum": pl.Series(
                [start + dt.timedelta(days=int(d)) for d in rng.integers(0, 365, n)],
                dtype=pl.Date,
            ),
            "gemiddeld_cijfer": pl.Series(np.round(rng.normal(6.8, 0.9, n), 2), dtype=pl.Float64),
        }
    )


def results(n: int = 120, seed: int = 2) -> pl.DataFrame:
    """Tweede tabel voor tests met meerdere tabellen."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "vak": pl.Series([str(x) for x in rng.choice(["wiskunde", "nederlands"], n)]),
            "cijfer": pl.Series(np.round(rng.uniform(1, 10, n), 1), dtype=pl.Float64),
            "poging": pl.Series(rng.integers(1, 4, n), dtype=pl.Int32),
        }
    )
