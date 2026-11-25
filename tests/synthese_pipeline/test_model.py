"""Tests voor fitten en synthetiseren met metasyn (kolomtypes, disclosure, reproduceerbaarheid)."""

import json

import polars as pl
import pytest

from synthese_pipeline import model
from tests.synthese_pipeline.dummy import OUTLIER_AGE, SENTINEL, results, students


@pytest.fixture(scope="module")
def gmf():
    return model.fit_tables({"studenten": students()})


@pytest.fixture(scope="module")
def gmf_plain():
    return model.fit_tables({"studenten": students()}, disclosure=False)


def _var(gmf, table, column):
    (table_dict,) = [t for t in gmf["tables"] if t["name"] == table]
    (var,) = [v for v in table_dict["vars"] if v["name"] == column]
    return var


def test_likert_with_nulls_stays_integer(gmf):
    synthetic = model.synthesize_tables(gmf, rows=1000, seed=1)["studenten"]
    for column in ("q1", "q2"):
        assert synthetic.schema[column] == pl.Int64
        values = set(synthetic[column].drop_nulls().to_list())
        assert values <= {1, 2, 3, 4, 5}
        assert synthetic[column].null_count() > 0


def test_short_text_becomes_categorical_not_regex(gmf):
    var = _var(gmf, "studenten", "geslacht")
    assert var["distribution"]["name"] == "core.multinoulli"
    assert var["distribution"]["class_name"] != "RegexDistribution"


def test_leading_zero_codes_stay_strings(gmf, tmp_path):
    synthetic = model.synthesize_tables(gmf, rows=300, seed=2)["studenten"]
    assert synthetic.schema["opleidingscode"] == pl.String
    assert set(synthetic["opleidingscode"].to_list()) <= {"01234", "02345", "00456"}


def test_output_schema_equals_source(gmf):
    synthetic = model.synthesize_tables(gmf, seed=3)["studenten"]
    assert synthetic.schema == students().schema
    assert synthetic.height == 200


def test_unique_key_stays_unique(gmf):
    synthetic = model.synthesize_tables(gmf, rows=500, seed=4)["studenten"]
    assert synthetic["student_id"].n_unique() == 500


def test_same_model_and_seed_give_same_output(gmf, tmp_path):
    path = tmp_path / "model.json"
    model.save_model(gmf, path)
    loaded = model.load_model(path)
    first = model.synthesize_tables(loaded, rows=300, seed=7)["studenten"]
    second = model.synthesize_tables(model.load_model(path), rows=300, seed=7)["studenten"]
    other = model.synthesize_tables(loaded, rows=300, seed=8)["studenten"]
    assert first.equals(second)
    assert not first.equals(other)


def test_disclosure_is_on_by_default_and_recorded(gmf):
    info = gmf[model.TOOL_KEY]["disclosure_control"]
    assert info["gebruikt"] is True
    assert info["plugin"] == "metasyn-disclosure"
    assert info["versie"] == "0.3.3"
    assert gmf["provenance"]["disclosure control"] == "metasyn-disclosure 0.3.3"
    for var in gmf["tables"][0]["vars"]:
        assert var["creation_method"]["fitter"]["privacy"]["name"] == "disclosure"


def _numbers(data):
    if isinstance(data, dict):
        return [n for value in data.values() for n in _numbers(value)]
    if isinstance(data, list):
        return [n for value in data for n in _numbers(value)]
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        return [data]
    return []


def test_unique_outlier_is_not_in_model_with_disclosure(gmf, gmf_plain):
    with_dc = _numbers(_var(gmf, "studenten", "leeftijd")["distribution"]["parameters"])
    without_dc = _numbers(_var(gmf_plain, "studenten", "leeftijd")["distribution"]["parameters"])
    assert OUTLIER_AGE in without_dc  # controle: zonder plugin staat 71 in het model
    assert OUTLIER_AGE not in with_dc
    assert SENTINEL not in json.dumps(gmf)
    assert gmf_plain[model.TOOL_KEY]["disclosure_control"]["gebruikt"] is False
    assert gmf_plain["provenance"]["disclosure control"] == "uit"


def test_type_override_is_used():
    gmf = model.fit_tables(
        {"studenten": students()}, overrides={"leeftijd": "categorical"}, disclosure=False
    )
    assert _var(gmf, "studenten", "leeftijd")["distribution"]["name"] == "core.multinoulli"
    assert gmf[model.TOOL_KEY]["tabellen"]["studenten"]["kolommen"]["leeftijd"]["soort"] == (
        "categorical"
    )


def test_multiple_tables_in_one_model(tmp_path):
    gmf = model.fit_tables({"studenten": students(), "resultaten": results()})
    path = tmp_path / "model.json"
    model.save_model(gmf, path)
    synthetic = model.synthesize_tables(model.load_model(path), rows=50, seed=1)
    assert set(synthetic) == {"studenten", "resultaten"}
    assert synthetic["resultaten"].schema == results().schema
    assert synthetic["resultaten"].height == 50


def test_load_model_errors(tmp_path):
    from synthese_pipeline.errors import SyntheseError

    with pytest.raises(SyntheseError, match="bestaat niet"):
        model.load_model(tmp_path / "geen.json")
    bad = tmp_path / "kapot.json"
    bad.write_text("{niet json")
    with pytest.raises(SyntheseError, match="geen geldig JSON"):
        model.load_model(bad)
    wrong = tmp_path / "fout.json"
    wrong.write_text('{"gmf_version": "2.1", "tables": [{"vars": 3}]}')
    with pytest.raises(SyntheseError, match="geen geldig GMF"):
        model.load_model(wrong)


def test_other_dtypes_are_restored():
    import datetime as dt
    import decimal

    import numpy as np

    rng = np.random.default_rng(3)
    n = 120
    stamps = [
        dt.datetime(2024, 1, 1) + dt.timedelta(hours=int(h)) for h in rng.integers(0, 5000, n)
    ]
    df = pl.DataFrame(
        {
            "u8": pl.Series(rng.integers(0, 5, n), dtype=pl.UInt8),
            "f32": pl.Series(rng.random(n), dtype=pl.Float32),
            "cat": pl.Series([str(x) for x in rng.choice(["a", "b", "c"], n)]).cast(pl.Categorical),
            "vlag": pl.Series(rng.random(n) < 0.4),
            "leeg": pl.Series([None] * n, dtype=pl.String),
            "dec": pl.Series(
                [decimal.Decimal(f"{x:.2f}") for x in rng.random(n) * 100],
                dtype=pl.Decimal(10, 2),
            ),
            "ts": pl.Series(stamps).dt.replace_time_zone("Europe/Amsterdam"),
            "tijd": pl.Series([dt.time(int(h), 0) for h in rng.integers(8, 18, n)]),
            "enum": pl.Series([str(x) for x in rng.choice(["x", "y"], n)]).cast(
                pl.Enum(["x", "y"])
            ),
        }
    )
    synthetic = model.synthesize_tables(model.fit_tables({"t": df}), seed=1)["t"]
    expected = dict(df.schema)
    expected["enum"] = pl.Categorical()  # de categorieën van een Enum komen niet in het model
    assert dict(synthetic.schema) == expected
