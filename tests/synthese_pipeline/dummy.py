"""Gegenereerde dummydata voor de tests. Nooit echte data."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

SENTINEL = "ZQX7391"
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


FREETEXT_MARK = "Qzxw"
_NAME_PARTS = ("Qzxwa", "Qzxwo", "Qzxwi", "Qzxwu", "Qzxwe", "Qzxwy")
_NAME_ENDS = ("brq", "vlq", "tkq", "mpq", "dzq", "fxq", "ghq", "jkq")


def freetext(n: int = 200, seed: int = 3) -> pl.DataFrame:
    """Tabel met vrije tekst (zinnen, namen), postcodes, opleidingscodes en een cijfer.

    Elke tekstwaarde in ``opmerking`` en ``naam`` bevat ``FREETEXT_MARK``: zo'n woord komt niet
    voor in de Nederlandse woorden en namen van Faker, dus een gelijke waarde is een kopie.
    """
    rng = np.random.default_rng(seed)

    def name() -> str:
        first = str(rng.choice(_NAME_PARTS)) + str(rng.choice(_NAME_ENDS))
        last = str(rng.choice(_NAME_PARTS)) + str(rng.choice(_NAME_ENDS)) + "son"
        return f"{first} {last}"

    letters = list("ABCDEFGHJKLMNPRSTVWXZ")
    return pl.DataFrame(
        {
            "opmerking": pl.Series(
                [
                    f"{FREETEXT_MARK} schreef op dag {i} dat het rooster {rng.integers(1, 99)} "
                    "keer veranderde"
                    for i in range(n)
                ],
                dtype=pl.String,
            ),
            "naam": pl.Series([name() for _ in range(n)], dtype=pl.String),
            "postcode": pl.Series(
                [
                    f"{rng.integers(1000, 10000)} {rng.choice(letters)}{rng.choice(letters)}"
                    for _ in range(n)
                ],
                dtype=pl.String,
            ),
            "opleidingscode": pl.Series(
                [f"{x}" for x in rng.integers(30000, 30300, n)], dtype=pl.String
            ),
            "cijfer": pl.Series(np.round(rng.normal(6.8, 0.9, n), 1), dtype=pl.Float64),
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


def survey(n: int = 200, seed: int = 4) -> pl.DataFrame:
    """Enquêtetabel voor de specificatie: sleutel, Likert, code, datum, vrije tekst, categorie.

    ``geslacht`` bevat één keer ``SENTINEL`` en ``opmerking`` in elke waarde ``FREETEXT_MARK``.
    """
    rng = np.random.default_rng(seed)
    gender = [str(x) for x in rng.choice(["M", "V"], n)]
    gender[5] = SENTINEL
    start = dt.date(2024, 9, 1)
    return pl.DataFrame(
        {
            "studentnummer": pl.Series(range(500_000, 500_000 + n), dtype=pl.Int64),
            "tevredenheid": pl.Series(
                [None if rng.random() < 0.1 else int(rng.integers(1, 6)) for _ in range(n)],
                dtype=pl.Int64,
            ),
            "opleidingscode": pl.Series(
                [f"0{x}" for x in rng.integers(30000, 30300, n)], dtype=pl.String
            ),
            "inschrijfdatum": pl.Series(
                [start + dt.timedelta(days=int(d)) for d in rng.integers(0, 365, n)],
                dtype=pl.Date,
            ),
            "opmerking": pl.Series(
                [
                    f"{FREETEXT_MARK} vond college {i} {rng.choice(['leuk', 'saai'])}"
                    for i in range(n)
                ],
                dtype=pl.String,
            ),
            "geslacht": pl.Series(gender, dtype=pl.String),
        }
    )
