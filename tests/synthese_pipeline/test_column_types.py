"""Tests voor het afleiden van kolomtypes."""

import datetime as dt

import polars as pl
import pytest

from synthese_pipeline import column_types as ct
from synthese_pipeline.errors import SyntheseError
from tests.synthese_pipeline.dummy import SENTINEL, students


def test_inferred_kinds_for_student_table():
    df = students()
    kinds = {column: ct.infer_kind(df[column]) for column in df.columns}
    assert kinds == {
        "student_id": "key",
        "geslacht": "categorical",
        "q1": "categorical",
        "q2": "categorical",
        "opleidingscode": "categorical",
        "leeftijd": "discrete",
        "inschrijfdatum": "date",
        "gemiddeld_cijfer": "continuous",
    }


@pytest.mark.parametrize(
    ("series", "kind"),
    [
        (pl.Series("b", [True, False, None] * 10), "categorical"),
        (pl.Series("f", [1.0, 2.0, None, 3.0] * 10), "categorical"),
        (pl.Series("f", [float(i % 40) for i in range(200)]), "discrete"),
        (pl.Series("t", [f"woord {i}" for i in range(15)] + [None]), "text"),
        (pl.Series("k", [f"S{i:05d}" for i in range(50)]), "key"),
        (pl.Series("k", list(range(10))), "discrete"),
        (pl.Series("d", [dt.datetime(2024, 1, 1)] * 3), "date"),
        (pl.Series("leeg", [None, None], dtype=pl.String), "text"),
        (pl.Series("leeg", [None, None], dtype=pl.Int64), "discrete"),
    ],
)
def test_infer_kind_rules(series, kind):
    assert ct.infer_kind(series) == kind


def test_unsupported_dtype_names_column_not_values():
    series = pl.Series("lijst", [[SENTINEL], ["b"]])
    with pytest.raises(SyntheseError, match="lijst") as exc:
        ct.infer_kind(series)
    assert SENTINEL not in str(exc.value)


def test_parse_overrides():
    assert ct.parse_type_overrides(["code=categorical", "studenten.leeftijd = Discrete"]) == {
        "code": "categorical",
        "studenten.leeftijd": "discrete",
    }
    with pytest.raises(SyntheseError, match="kolom=soort"):
        ct.parse_type_overrides(["code"])
    with pytest.raises(SyntheseError, match="Onbekende soort"):
        ct.parse_type_overrides(["code=getal"])


def test_overrides_must_name_existing_columns():
    tables = {"studenten": students()}
    ct.check_overrides({"leeftijd": "categorical", "studenten.q1": "discrete"}, tables)
    with pytest.raises(SyntheseError, match="bestaat niet"):
        ct.check_overrides({"postcode": "text"}, tables)


def test_kinds_for_table_prefers_table_specific_override():
    df = students()
    kinds = ct.kinds_for_table(
        "studenten", df, {"leeftijd": "continuous", "studenten.leeftijd": "categorical"}
    )
    assert kinds["leeftijd"] == "categorical"
    assert kinds["q1"] == "categorical"


def test_prepare_categorical_uses_multinoulli_and_keeps_integers():
    series = pl.Series("q1", [1, None, 5, 3], dtype=pl.Int8)
    prepared, spec = ct.prepare_column(series, "categorical")
    assert prepared.dtype == pl.Int64
    assert spec == {"distribution": {"name": "core.multinoulli"}}
    text, _ = ct.prepare_column(pl.Series("g", ["M", "V"], dtype=pl.Categorical), "categorical")
    assert text.dtype == pl.String


def test_prepare_key_and_discrete():
    _, spec = ct.prepare_column(pl.Series("id", [1, 2, 3]), "key")
    assert spec == {"distribution": {"unique": True}}
    prepared, spec = ct.prepare_column(pl.Series("n", [1.0, 2.0, None]), "discrete")
    assert prepared.dtype == pl.Int64 and spec is None


def test_prepare_wrong_kind_reports_count_not_values():
    series = pl.Series("geslacht", ["M", "V", SENTINEL])
    with pytest.raises(SyntheseError, match="3 waarden") as exc:
        ct.prepare_column(series, "discrete")
    assert SENTINEL not in str(exc.value)


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Int8(),
        pl.UInt32(),
        pl.Float32(),
        pl.String(),
        pl.Boolean(),
        pl.Date(),
        pl.Datetime("ms", "Europe/Amsterdam"),
        pl.Decimal(10, 2),
        pl.Categorical(),
    ],
)
def test_dtype_roundtrip(dtype):
    assert ct.dtype_from_dict(ct.dtype_to_dict(dtype)) == dtype


def test_enum_categories_are_not_stored():
    stored = ct.dtype_to_dict(pl.Enum([SENTINEL, "b"]))
    assert stored == {"naam": "Categorical"}


def test_restore_column_rounds_floats_to_integers():
    restored, failed = ct.restore_column(pl.Series("x", [1.4, 2.6, None]), pl.Int32())
    assert restored.to_list() == [1, 3, None]
    assert restored.dtype == pl.Int32 and failed == 0
    restored, failed = ct.restore_column(pl.Series("x", ["1", "b"]), pl.Int64())
    assert failed == 1
