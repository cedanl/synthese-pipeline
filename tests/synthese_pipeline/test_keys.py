"""Tests voor sleutels in het GMF (#33): geen sleutelwaarde of sleutelbereik uit de echte data.

Alleen gegenereerde dummydata (zie ``dummy.py``).
"""

import json
from pathlib import Path

import polars as pl
import pytest

from synthese_pipeline import model as model_module
from synthese_pipeline.cli import main
from synthese_pipeline.model import fit_tables, synthesize_tables
from synthese_pipeline.report import EXIT_OK
from synthese_pipeline.spec import parse_spec
from tests.synthese_pipeline.dummy import keyed_tables

# Het GMF begint elke gefitte sleutel bij deze vaste waarde, los van de data.
KEY_OFFSET = 1

README = Path(__file__).resolve().parents[2] / "README.md"

RELATION = "resultaten[studentnummer] SUBSET OF studenten[studentnummer]"

SPEC_TOML = f"""
config_version = "2.0"
relations = ["{RELATION}"]

[[table]]
name = "studenten"
[[table.var]]
name = "studentnummer"
[[table.var]]
name = "tevredenheid"

[[table]]
name = "resultaten"
[[table.var]]
name = "studentnummer"
[[table.var]]
name = "cijfer"
"""


def _spec():
    return parse_spec(
        {
            "config_version": "2.0",
            "relations": [RELATION],
            "table": [
                {"name": "studenten", "var": [{"name": "studentnummer"}]},
                {"name": "resultaten", "var": [{"name": "studentnummer"}]},
            ],
        }
    )


def _numbers(data):
    """Alle getallen in geparste JSON (geen booleans), ook diep in lijsten en dicts."""
    if isinstance(data, dict):
        for value in data.values():
            yield from _numbers(value)
    elif isinstance(data, list):
        for item in data:
            yield from _numbers(item)
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        yield data


def _key_vars(gmf):
    for table in gmf["tables"]:
        for var in table["vars"]:
            if var["distribution"]["name"] == "core.unique_key":
                yield table["name"], var


def _real_keys(tables) -> set[int]:
    keys: set[int] = set()
    for df in tables.values():
        keys |= set(df["studentnummer"].to_list())
    return keys


def _assert_no_real_key(gmf, tables):
    real = _real_keys(tables)
    leaked = [number for number in _numbers(json.loads(json.dumps(gmf))) if number in real]
    assert leaked == [], f"{len(leaked)} echte sleutelwaarde(n) in het GMF"


# ------------------------------------------------------------------------------- geen waarden


@pytest.mark.parametrize("disclosure", [True, False], ids=["disclosure", "no-disclosure"])
def test_gmf_has_no_real_key_value(disclosure):
    tables = keyed_tables()
    gmf = fit_tables({"studenten": tables["studenten"]}, disclosure=disclosure)
    (found,) = list(_key_vars(gmf))
    assert found[1]["distribution"]["parameters"]["lower"] == KEY_OFFSET
    _assert_no_real_key(gmf, {"studenten": tables["studenten"]})


@pytest.mark.parametrize("disclosure", [True, False], ids=["disclosure", "no-disclosure"])
def test_referencing_column_has_no_real_key_value(disclosure):
    tables = keyed_tables()
    gmf = fit_tables(tables, disclosure=disclosure, spec=_spec())
    found = {(table, var["name"]): var for table, var in _key_vars(gmf)}
    assert set(found) == {("studenten", "studentnummer"), ("resultaten", "studentnummer")}
    for var in found.values():
        assert var["distribution"]["parameters"]["lower"] == KEY_OFFSET
    _assert_no_real_key(gmf, tables)


def test_cli_fit_no_disclosure_gmf_text_has_no_real_key(tmp_path):
    tables = keyed_tables()
    source = tmp_path / "studenten.csv"
    tables["studenten"].write_csv(source)
    model = tmp_path / "model.json"
    assert main(["fit", str(source), str(model), "--no-disclosure"]) == EXIT_OK
    text = model.read_text(encoding="utf-8")
    _assert_no_real_key(json.loads(text), {"studenten": tables["studenten"]})
    # Ook niet als stuk tekst (bijv. in een omschrijving of log van metasyn).
    assert not any(str(key) in text for key in _real_keys({"s": tables["studenten"]}))


# ------------------------------------------------------------------------- vaste beginwaarde


@pytest.mark.parametrize("disclosure", [True, False], ids=["disclosure", "no-disclosure"])
def test_lower_is_fixed_and_independent_of_the_data(disclosure):
    ranges = [(81_234_000, 81_240_000), (500_000, 503_000), (20_000, 21_000)]
    lowers = []
    for seed, (low, high) in zip((6, 7, 8), ranges, strict=True):
        tables = keyed_tables(seed, low=low, high=high)
        real_min = tables["studenten"]["studentnummer"].min()
        gmf = fit_tables({"studenten": tables["studenten"]}, disclosure=disclosure)
        ((_, var),) = list(_key_vars(gmf))
        lower = var["distribution"]["parameters"]["lower"]
        lowers.append(lower)
        assert real_min - lower >= 10_000
    assert lowers == [KEY_OFFSET] * len(ranges)
    assert model_module.KEY_OFFSET == KEY_OFFSET


def test_same_data_and_seed_give_same_output():
    tables = keyed_tables()
    first = synthesize_tables(fit_tables(tables, spec=_spec()), seed=3)
    second = synthesize_tables(fit_tables(tables, spec=_spec()), seed=3)
    other = synthesize_tables(fit_tables(tables, spec=_spec()), seed=4)
    for name in tables:
        assert first[name].equals(second[name])
    assert not first["studenten"].equals(other["studenten"])


def test_key_fixed_in_the_spec_is_kept():
    """Een verdeling met parameters uit de specificatie komt niet uit de data: die blijft."""
    tables = keyed_tables()
    spec = parse_spec(
        {
            "config_version": "1.0",
            "var": [
                {
                    "name": "studentnummer",
                    "distribution": {
                        "implements": "core.unique_key",
                        "unique": True,
                        "parameters": {"lower": 5000, "consecutive": True},
                    },
                }
            ],
        }
    )
    gmf = fit_tables({"studenten": tables["studenten"]}, spec=spec)
    ((_, var),) = list(_key_vars(gmf))
    assert var["distribution"]["parameters"] == {"lower": 5000, "consecutive": True}


# --------------------------------------------------------------------- uniek, geen wezen


@pytest.mark.parametrize("flags", [[], ["--no-disclosure"]], ids=["disclosure", "no-disclosure"])
def test_synthetic_keys_unique_and_referencing_keys_exist(tmp_path, flags):
    tables = keyed_tables()
    folder = tmp_path / "bron"
    folder.mkdir()
    for name, df in tables.items():
        df.write_parquet(folder / f"{name}.parquet")
    spec_path = tmp_path / "spec.toml"
    spec_path.write_text(SPEC_TOML, encoding="utf-8")
    model = tmp_path / "model.json"
    assert main(["fit", str(folder), str(model), "--spec", str(spec_path), *flags]) == EXIT_OK
    target = tmp_path / "uit"
    args = ["synthesize", str(model), str(target), "--format", "parquet", "--seed", "2"]
    assert main([*args, "--source", str(folder)]) == EXIT_OK
    student = pl.read_parquet(target / "studenten.parquet")
    resultaat = pl.read_parquet(target / "resultaten.parquet")
    keys = student["studentnummer"]
    assert keys.dtype == pl.Int64
    assert keys.null_count() == 0 and keys.n_unique() == student.height == 200
    # Niet-opeenvolgende sleutels: metasyn trekt uit [1, 1 + 2 * rijen).
    assert keys.min() >= KEY_OFFSET and keys.max() <= 2 * student.height
    assert resultaat["studentnummer"].is_in(keys.implode()).all()
    report = json.loads((target / "rapport.json").read_text(encoding="utf-8"))
    assert report["exitcode"] == EXIT_OK
    assert report["structuur"]["gelijk"] is True
    (entry,) = report["relaties"]
    assert entry == {
        "relatie": RELATION,
        "tabel": "resultaten",
        "kolom": "studentnummer",
        "verwijst_naar_tabel": "studenten",
        "verwijst_naar_kolom": "studentnummer",
        "rijen_gecontroleerd": 300,
        "lege_verwijzingen": 0,
        "wezen": 0,
        "dubbele_sleutels": 0,
    }


# ------------------------------------------------------------------------------------- docs


def test_readme_documents_key_offset():
    text = README.read_text(encoding="utf-8")
    assert "lower = 1" in text
    assert "geen sleutelbereik" in text
