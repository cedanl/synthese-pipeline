"""Vrije tekst: standaard uitgesloten, of met ``--faker`` verzonnen; nooit letterlijk gekopieerd."""

import json
import re

import polars as pl
import pytest

from synthese_pipeline import describe, model
from synthese_pipeline.errors import SyntheseError
from synthese_pipeline.report import EXIT_OK
from tests.synthese_pipeline.dummy import FREETEXT_MARK, freetext

POSTCODE = re.compile(r"^\d{4} [A-Z]{2}$")
OPLEIDINGSCODE = re.compile(r"^\d{5}$")


@pytest.fixture(scope="module")
def excluded():
    return model.fit_tables({"t": freetext()})


@pytest.fixture(scope="module")
def faked():
    return model.fit_tables({"t": freetext()}, fakers={"naam": "name", "t.opmerking": "sentence"})


def _var_names(gmf, table="t"):
    (table_dict,) = [t for t in gmf["tables"] if t["name"] == table]
    return [var["name"] for var in table_dict["vars"]]


def test_freetext_is_excluded_from_the_model(excluded):
    assert _var_names(excluded) == ["postcode", "opleidingscode", "cijfer"]
    info = excluded[model.TOOL_KEY]["tabellen"]["t"]
    assert info["uitgesloten"] == {
        "opmerking": {"brontype": {"naam": "String"}},
        "naam": {"brontype": {"naam": "String"}},
    }
    assert set(info["kolommen"]) == {"postcode", "opleidingscode", "cijfer"}
    assert model.excluded_columns(excluded) == {"t": ["opmerking", "naam"]}


def test_no_real_freetext_value_in_gmf(excluded, faked):
    df = freetext()
    for gmf in (excluded, faked):
        text = json.dumps(gmf, ensure_ascii=False)
        assert FREETEXT_MARK not in text
        for column in ("opmerking", "naam"):
            for value in df[column].to_list():
                assert value not in text


def test_excluded_column_is_absent_and_codes_keep_shape(excluded, tmp_path):
    path = tmp_path / "model.json"
    model.save_model(excluded, path)
    synthetic = model.synthesize_tables(model.load_model(path), rows=300, seed=1)["t"]
    assert synthetic.columns == ["postcode", "opleidingscode", "cijfer"]
    assert synthetic.schema["postcode"] == pl.String
    assert all(POSTCODE.match(value) for value in synthetic["postcode"].to_list())
    assert all(OPLEIDINGSCODE.match(value) for value in synthetic["opleidingscode"].to_list())


def test_faker_columns_have_no_real_values(faked):
    info = faked[model.TOOL_KEY]["tabellen"]["t"]
    assert info.get("uitgesloten", {}) == {}
    assert info["kolommen"]["naam"]["soort"] == "faker"
    for column, provider in (("naam", "name"), ("opmerking", "sentence")):
        (var,) = [v for v in faked["tables"][0]["vars"] if v["name"] == column]
        assert var["distribution"]["name"] == "core.faker"
        assert var["distribution"]["parameters"] == {"faker_type": provider, "locale": "nl_NL"}

    real = freetext()
    synthetic = model.synthesize_tables(faked, rows=500, seed=2)["t"]
    assert synthetic.columns == real.columns
    assert synthetic.schema == real.schema
    for column in ("naam", "opmerking"):
        fake = set(synthetic[column].drop_nulls().to_list())
        assert len(fake) > 50
        assert fake.isdisjoint(set(real[column].to_list()))
        assert not any(FREETEXT_MARK in value for value in fake)
    assert all(POSTCODE.match(value) for value in synthetic["postcode"].to_list())
    assert all(OPLEIDINGSCODE.match(value) for value in synthetic["opleidingscode"].to_list())


def test_faker_is_reproducible_with_seed(faked):
    first = model.synthesize_tables(faked, rows=50, seed=5)["t"]
    second = model.synthesize_tables(faked, rows=50, seed=5)["t"]
    assert first.equals(second)


def test_type_override_brings_column_back():
    gmf = model.fit_tables({"t": freetext()}, overrides={"naam": "text"}, disclosure=False)
    assert "naam" in _var_names(gmf)
    assert list(model.excluded_columns(gmf)["t"]) == ["opmerking"]


def test_forced_freetext_is_excluded():
    gmf = model.fit_tables({"t": freetext()}, overrides={"t.postcode": "freetext"})
    assert "postcode" not in _var_names(gmf)
    assert model.excluded_columns(gmf) == {"t": ["opmerking", "naam", "postcode"]}


def test_all_columns_excluded_is_an_error():
    df = freetext().select("opmerking", "naam")
    with pytest.raises(SyntheseError, match="vrije tekst") as exc:
        model.fit_tables({"t": df})
    assert FREETEXT_MARK not in str(exc.value)


def test_faker_unknown_column_is_an_error():
    with pytest.raises(SyntheseError, match="--faker noemt kolom 'bestaatniet'"):
        model.fit_tables({"t": freetext()}, fakers={"bestaatniet": "name"})


def test_toon_shows_excluded_and_faker(excluded, faked):
    text = describe.describe_model(excluded)
    assert "- opmerking: uitgesloten (vrije tekst)" in text
    assert "- naam: uitgesloten (vrije tekst)" in text
    assert "Tabel 't': 200 rijen, 3 kolommen, 2 uitgesloten" in text
    text = describe.describe_model(faked)
    assert "uitgesloten" not in text
    assert "- naam (verzonnen met Faker): verzonnen waarden (Faker, soort 'name')" in text


def test_model_without_excluded_key_still_works(excluded):
    old = json.loads(json.dumps(excluded))
    del old[model.TOOL_KEY]["tabellen"]["t"]["uitgesloten"]
    assert model.excluded_columns(old) == {}
    assert "uitgesloten" not in describe.describe_model(old)
    synthetic = model.synthesize_tables(old, rows=10, seed=1)["t"]
    assert synthetic.columns == ["postcode", "opleidingscode", "cijfer"]


def test_report_names_excluded_columns_and_structure_is_equal(excluded, tmp_path):
    from synthese_pipeline.pipeline import run_synthesize

    source = tmp_path / "bron.parquet"
    freetext().write_parquet(source)
    path = tmp_path / "model.json"
    model.save_model(excluded, path)
    result = run_synthesize(path, tmp_path / "uit.parquet", seed=1, source=source)
    report = result.report
    assert report["exitcode"] == EXIT_OK, report
    assert report["structuur"]["gelijk"] is True
    assert report["tabellen"]["t"]["uitgesloten"] == ["opmerking", "naam"]
    assert FREETEXT_MARK not in json.dumps(report)
    assert pl.read_parquet(tmp_path / "uit.parquet").columns == [
        "postcode",
        "opleidingscode",
        "cijfer",
    ]
