"""Tests voor de structuurcheck, de verdelingsmaten en de exitcode in het rapport."""

import json

import polars as pl
import pytest

from synthese_pipeline import report
from tests.synthese_pipeline.dummy import SENTINEL


def test_structure_check_finds_missing_extra_and_type_changes():
    expected = {"a": pl.Int64(), "b": pl.String(), "c": pl.Date()}
    synthetic = pl.DataFrame({"b": ["x"], "a": ["1"], "d": [1]})
    deviations = report.check_structure(expected, synthetic)
    assert "kolom 'c' ontbreekt" in deviations
    assert "kolom 'd' is extra" in deviations
    assert "de volgorde van de kolommen wijkt af" in deviations
    assert "kolom 'a': type Int64 verwacht, String gevonden" in deviations


def test_structure_check_with_real_data_counts_new_categories_and_nulls():
    real = pl.DataFrame({"g": ["M", "V", SENTINEL], "n": [1, 2, 3]})
    synthetic = pl.DataFrame({"g": ["M", "X", "Y"], "n": [1, None, 2]})
    deviations = report.check_structure(
        dict(real.schema), synthetic, real=real, kinds={"g": "categorical"}
    )
    assert "kolom 'g': 2 categorieën die niet in de bron voorkomen" in deviations
    assert "kolom 'n': 1 lege waarden, de bron heeft er geen" in deviations
    assert SENTINEL not in json.dumps(deviations)


def test_tv_distance():
    same = pl.Series(["a", "b", "a", None])
    assert report.tv_distance(same, same) == 0
    assert report.tv_distance(pl.Series(["a"]), pl.Series(["b"])) == 1
    assert report.tv_distance(pl.Series(["a", "b"]), pl.Series(["a", "a"])) == pytest.approx(0.5)


def test_wasserstein_for_numbers_and_dates():
    measure = report.column_measure(pl.Series([1.0, 2.0]), pl.Series([2.0, 3.0]), "continuous")
    assert measure["maat"] == "wasserstein" and measure["waarde"] == pytest.approx(1.0)
    import datetime as dt

    days = report.column_measure(
        pl.Series([dt.date(2024, 1, 1)]), pl.Series([dt.date(2024, 1, 3)]), "date"
    )
    assert days["maat"] == "wasserstein_dagen" and days["waarde"] == pytest.approx(2.0)
    assert report.column_measure(pl.Series([1]), pl.Series([1]), "key")["maat"] is None


def _build(real, synthetic):
    return report.build_report(
        gmf_tool_info={"disclosure_control": {"gebruikt": True}},
        expected={"t": dict(synthetic.schema)},
        kinds={"t": {"a": "categorical", "b": "discrete"}},
        synthetic={"t": synthetic},
        real=None if real is None else {"echt": real},
        seed=5,
    )


def test_exit_codes_in_report():
    real = pl.DataFrame({"a": ["x", "y", "y"], "b": [1, 2, 2]})
    clean = pl.DataFrame({"a": ["y", "y"], "b": [2, 2]})
    copied = pl.DataFrame({"a": ["x", "y"], "b": [1, 2]})
    other_type = pl.DataFrame({"a": ["y"], "b": [2.0]})
    assert _build(real, clean)["exitcode"] == report.EXIT_OK
    assert _build(real, copied)["exitcode"] == report.EXIT_COPIES
    assert _build(real, copied)["letterlijke_kopieen"] == 1
    assert _build(real, other_type)["exitcode"] == report.EXIT_STRUCTURE
    without_source = _build(None, copied)
    assert without_source["exitcode"] == report.EXIT_OK
    assert without_source["letterlijke_kopieen"] is None
    assert "niet gecontroleerd" in without_source["toelichting_kopieen"]


def test_missing_table_is_a_structure_deviation():
    result = report.build_report(
        gmf_tool_info={},
        expected={"a": {"x": pl.Int64()}, "b": {"x": pl.Int64()}},
        kinds={},
        synthetic={"a": pl.DataFrame({"x": [1]})},
        real=None,
        seed=1,
    )
    assert result["structuur"]["afwijkingen"] == ["tabel 'b' ontbreekt"]
    assert result["exitcode"] == report.EXIT_STRUCTURE


def test_default_report_path(tmp_path):
    assert report.default_report_path(tmp_path / "uit.csv", True) == tmp_path / "uit.rapport.json"
    assert report.default_report_path(tmp_path / "uit", False) == tmp_path / "uit" / "rapport.json"
