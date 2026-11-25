"""Tests voor de specificatie: init-spec, fit --spec en synthesize --spec (data-free)."""

import datetime as dt
import json
import re
import tomllib
from pathlib import Path

import polars as pl
import pytest

from synthese_pipeline.cli import main
from synthese_pipeline.report import EXIT_ERROR, EXIT_OK, EXIT_USAGE
from synthese_pipeline.spec import SUPPORTED_VERSIONS, load_spec, parse_spec
from tests.synthese_pipeline.dummy import FREETEXT_MARK, SENTINEL, results, survey

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "spec.toml"


def _vars(spec: dict) -> dict[str, dict]:
    return {var["name"]: var for var in spec["var"]}


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def source(tmp_path):
    data = tmp_path / "enquete.csv"
    survey().write_csv(data)
    return data


# ---------------------------------------------------------------------------------- init-spec


def test_init_spec_writes_draft_without_values(tmp_path, source):
    spec_path = tmp_path / "spec.toml"
    log = tmp_path / "init.log"
    assert main(["init-spec", str(source), str(spec_path), "--log", str(log)]) == EXIT_OK
    text = spec_path.read_text()
    spec = tomllib.loads(text)
    assert spec["config_version"] == "1.2"
    assert spec["n_rows"] == 200
    variables = _vars(spec)
    assert list(variables) == survey().columns
    assert {name: var["soort"] for name, var in variables.items()} == {
        "studentnummer": "key",
        "tevredenheid": "categorical",
        "opleidingscode": "text",
        "inschrijfdatum": "date",
        "opmerking": "freetext",
        "geslacht": "categorical",
    }
    assert variables["studentnummer"]["distribution"] == {"unique": True}
    assert variables["opmerking"]["uitgesloten"] is True
    assert all(var["description"] == "" for var in variables.values())
    assert all("categorieen" not in var for var in variables.values())
    assert [name for name, var in variables.items() if "distribution" in var] == ["studentnummer"]
    for content in (text, log.read_text()):
        assert SENTINEL not in content
        assert FREETEXT_MARK not in content
        assert re.search(r"03[0-9]{4}", content) is None


def test_init_spec_with_categories(tmp_path, source):
    spec_path = tmp_path / "spec.toml"
    log = tmp_path / "init.log"
    args = ["init-spec", str(source), str(spec_path), "--with-categories", "--log", str(log)]
    assert main(args) == EXIT_OK
    variables = _vars(tomllib.loads(spec_path.read_text()))
    assert variables["tevredenheid"]["categorieen"] == [1, 2, 3, 4, 5]
    assert variables["geslacht"]["categorieen"] == sorted(["M", "V", SENTINEL])
    assert [name for name, var in variables.items() if "categorieen" in var] == [
        "tevredenheid",
        "geslacht",
    ]
    log_text = log.read_text()
    assert "kolom 'geslacht': 1 categorie(ën) komen minder dan 10 keer voor" in log_text
    assert SENTINEL not in log_text


def test_init_spec_multi_table_uses_config_version_2(tmp_path):
    folder = tmp_path / "bron"
    folder.mkdir()
    survey().write_parquet(folder / "enquete.parquet")
    results().write_parquet(folder / "resultaten.parquet")
    spec_path = tmp_path / "spec.toml"
    assert main(["init-spec", str(folder), str(spec_path)]) == EXIT_OK
    data = tomllib.loads(spec_path.read_text())
    assert data["config_version"] == "2.0"
    assert [table["name"] for table in data["table"]] == ["enquete", "resultaten"]
    assert [var["name"] for var in data["table"][1]["var"]] == results().columns
    spec = load_spec(spec_path)
    assert spec.multi
    assert [table.n_rows for table in spec.tables] == [200, 120]

    model = tmp_path / "model.json"
    assert main(["fit", str(folder), str(model), "--spec", str(spec_path)]) == EXIT_OK
    gmf = json.loads(model.read_text())
    assert [table["name"] for table in gmf["tables"]] == ["enquete", "resultaten"]
    assert list(gmf["synthese_pipeline"]["tabellen"]["enquete"]["uitgesloten"]) == ["opmerking"]


# -------------------------------------------------------------------------------- versions


SINGLE = """
config_version = "{version}"
name = "enquete"
n_rows = 10

[[var]]
name = "studentnummer"
soort = "key"

[[var]]
name = "tevredenheid"
categorieen = [1, 2, 3]
"""

MULTI = """
config_version = "2.0"

[[table]]
name = "enquete"
n_rows = 10

[[table.var]]
name = "studentnummer"
distribution = {unique = true}

[[table]]
name = "resultaten"
n_rows = 7

[[table.var]]
name = "cijfer"
soort = "continuous"
distribution = {implements = "core.uniform", parameters = {lower = 1.0, upper = 10.0}}
"""


@pytest.mark.parametrize("version", ["1.0", "1.1", "1.2"])
def test_single_table_versions_are_read(tmp_path, version):
    spec = load_spec(_write(tmp_path / "spec.toml", SINGLE.format(version=version)))
    assert spec.version == version and not spec.multi
    (table,) = spec.tables
    assert table.name == "enquete" and table.n_rows == 10
    assert table.columns["studentnummer"].kind == "key"
    assert table.columns["tevredenheid"].kind == "categorical"


def test_multi_table_version_is_read(tmp_path):
    spec = load_spec(_write(tmp_path / "spec.toml", MULTI))
    assert spec.version == "2.0" and spec.multi
    assert spec.table_names() == ["enquete", "resultaten"]
    assert spec.tables[0].columns["studentnummer"].kind == "key"
    assert spec.tables[1].columns["cijfer"].distribution == {
        "name": "core.uniform",
        "parameters": {"lower": 1.0, "upper": 10.0},
    }


def test_supported_versions_follow_metasyn():
    from metasyn.builder import ConfigV1XParser, ConfigV2Parser

    assert set(SUPPORTED_VERSIONS) == set(ConfigV1XParser.supports) | set(ConfigV2Parser.supports)
    assert parse_spec({"var": [{"name": "a"}]}).version == "1.2"
    assert parse_spec({"table": [{"name": "t", "var": [{"name": "a"}]}]}).version == "2.0"


# --------------------------------------------------------------------------- invalid specs


INVALID = {
    "version": f'config_version = "{SENTINEL}"\n[[var]]\nname = "geslacht"\n',
    "toml": f'config_version = "1.2"\n[[var]\nname = "{SENTINEL}"\n',
    "soort": f'config_version = "1.2"\n[[var]]\nname = "geslacht"\nsoort = "{SENTINEL}"\n',
    "key": 'config_version = "1.2"\n[[var]]\nname = "onbekend"\nsoort = "text"\n',
    "unknown_key": 'config_version = "1.2"\nprivacy = {}\n[[var]]\nname = "geslacht"\n',
    "categories": (
        f'config_version = "1.2"\n[[var]]\nname = "geslacht"\ncategorieen = ["{SENTINEL}", 1]\n'
    ),
    "unique": (
        'config_version = "1.2"\n[[var]]\nname = "geslacht"\nsoort = "categorical"\n'
        "distribution = {unique = true}\n"
    ),
    "faker": 'config_version = "1.2"\n[[var]]\nname = "opmerking"\nfaker = "bestaatniet"\n',
    "relations": (
        'config_version = "2.0"\n[[table]]\nname = "enquete"\nrelations = []\n'
        '[[table.var]]\nname = "geslacht"\n'
    ),
    "table": 'config_version = "2.0"\n[[table]]\nname = "anders"\n[[table.var]]\nname = "a"\n'
    '[[table]]\nname = "nog_een"\n[[table.var]]\nname = "b"\n',
    "freetext_kept": (
        'config_version = "1.2"\n[[var]]\nname = "opmerking"\nsoort = "freetext"\n'
        "uitgesloten = false\n"
    ),
}


@pytest.mark.parametrize("case", sorted(INVALID))
def test_invalid_spec_gives_dutch_error_without_values(tmp_path, source, capsys, case):
    spec_path = _write(tmp_path / "spec.toml", INVALID[case])
    model = tmp_path / "model.json"
    assert main(["fit", str(source), str(model), "--spec", str(spec_path)]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert err.splitlines()[-1].startswith("fout: ")
    assert "Traceback" not in err
    assert SENTINEL not in err
    assert not model.exists()


def test_single_table_spec_refuses_multi_table_source(tmp_path, capsys):
    folder = tmp_path / "bron"
    folder.mkdir()
    survey().write_parquet(folder / "enquete.parquet")
    results().write_parquet(folder / "resultaten.parquet")
    spec_path = _write(tmp_path / "spec.toml", 'config_version = "1.2"\n[[var]]\nname = "vak"\n')
    assert main(["fit", str(folder), str(tmp_path / "m.json"), "--spec", str(spec_path)]) == 1
    assert "config_version 2.0" in capsys.readouterr().err


def test_missing_spec_file(tmp_path, source, capsys):
    args = ["fit", str(source), str(tmp_path / "m.json"), "--spec", str(tmp_path / "geen.toml")]
    assert main(args) == EXIT_ERROR
    assert "De specificatie bestaat niet" in capsys.readouterr().err


# ------------------------------------------------------------------------------ fit --spec


SPEC_FOR_FIT = """
config_version = "1.2"
n_rows = 50

[[var]]
name = "studentnummer"
distribution = {unique = true}
description = "Uniek nummer per student."

[[var]]
name = "tevredenheid"
soort = "discrete"
description = "Likert-item over tevredenheid."
distribution = {implements = "core.uniform"}

[[var]]
name = "opleidingscode"
uitgesloten = true
description = "Code van de opleiding."

[[var]]
name = "inschrijfdatum"
soort = "date"
prop_missing = 0.5

[[var]]
name = "geslacht"
soort = "categorical"
[var.distribution]
implements = "core.multinoulli"
parameters = {labels = ["A", "B"], probs = [0.5, 0.5]}
"""


def test_fit_applies_spec(tmp_path, source, capsys):
    spec_path = _write(tmp_path / "spec.toml", SPEC_FOR_FIT)
    model = tmp_path / "model.json"
    assert main(["fit", str(source), str(model), "--spec", str(spec_path)]) == EXIT_OK
    gmf = json.loads(model.read_text())
    (table,) = gmf["tables"]
    assert table["n_rows"] == 50
    variables = {var["name"]: var for var in table["vars"]}
    assert list(variables) == ["studentnummer", "tevredenheid", "inschrijfdatum", "geslacht"]
    assert variables["studentnummer"]["distribution"]["unique"] is True
    assert variables["studentnummer"]["description"] == "Uniek nummer per student."
    assert variables["tevredenheid"]["description"] == "Likert-item over tevredenheid."
    assert "description" not in variables["inschrijfdatum"]
    assert variables["inschrijfdatum"]["prop_missing"] == 0.5
    assert variables["tevredenheid"]["distribution"]["name"] == "core.uniform"
    assert variables["geslacht"]["distribution"]["parameters"]["labels"] == ["A", "B"]
    info = gmf["synthese_pipeline"]["tabellen"]["enquete"]
    assert info["kolommen"]["studentnummer"]["soort"] == "key"
    assert info["kolommen"]["tevredenheid"]["soort"] == "discrete"
    assert info["uitgesloten"]["opleidingscode"]["omschrijving"] == "Code van de opleiding."
    # opmerking staat niet in de specificatie: vrije tekst blijft standaard uitgesloten.
    assert set(info["uitgesloten"]) == {"opleidingscode", "opmerking"}
    capsys.readouterr()
    assert main(["toon", str(model)]) == EXIT_OK
    assert "  omschrijving: Uniek nummer per student." in capsys.readouterr().out


def test_command_line_wins_over_spec(tmp_path, source):
    spec_path = _write(tmp_path / "spec.toml", SPEC_FOR_FIT)
    model = tmp_path / "model.json"
    log = tmp_path / "fit.log"
    args = ["fit", str(source), str(model), "--spec", str(spec_path), "--log", str(log)]
    args += ["--type", "tevredenheid=categorical", "--faker", "enquete.geslacht=first_name"]
    assert main(args) == EXIT_OK
    gmf = json.loads(model.read_text())
    kolommen = gmf["synthese_pipeline"]["tabellen"]["enquete"]["kolommen"]
    assert kolommen["tevredenheid"]["soort"] == "categorical"
    assert kolommen["geslacht"]["soort"] == "faker"
    variables = {var["name"]: var for var in gmf["tables"][0]["vars"]}
    assert variables["geslacht"]["distribution"]["name"] == "core.faker"
    # De omschrijving uit de specificatie blijft.
    assert variables["tevredenheid"]["description"] == "Likert-item over tevredenheid."
    text = log.read_text()
    assert "kolom 'tevredenheid': --type/--faker gaat voor de specificatie" in text
    assert "kolom 'geslacht': --type/--faker gaat voor de specificatie" in text


def test_spec_faker_and_type_text_reinclude_freetext(tmp_path, source):
    spec_path = _write(
        tmp_path / "spec.toml",
        'config_version = "1.2"\n[[var]]\nname = "opmerking"\nfaker = "sentence"\n',
    )
    model = tmp_path / "model.json"
    assert main(["fit", str(source), str(model), "--spec", str(spec_path)]) == EXIT_OK
    gmf_text = model.read_text()
    assert FREETEXT_MARK not in gmf_text
    info = json.loads(gmf_text)["synthese_pipeline"]["tabellen"]["enquete"]
    assert info["kolommen"]["opmerking"]["soort"] == "faker"
    assert "uitgesloten" not in info


def test_init_spec_fit_synthesize(tmp_path, source):
    spec_path = tmp_path / "spec.toml"
    assert main(["init-spec", str(source), str(spec_path)]) == EXIT_OK
    model = tmp_path / "model.json"
    assert main(["fit", str(source), str(model), "--spec", str(spec_path)]) == EXIT_OK
    assert SENTINEL not in model.read_text() and FREETEXT_MARK not in model.read_text()
    output = tmp_path / "uit.parquet"
    assert main(["synthesize", str(model), str(output), "--seed", "1"]) == EXIT_OK
    synthetic = pl.read_parquet(output)
    expected = survey().drop("opmerking")
    assert synthetic.schema == expected.schema
    assert synthetic.height == 200
    assert synthetic["studentnummer"].n_unique() == synthetic.height
    assert synthetic["studentnummer"].null_count() == 0
    assert set(synthetic["tevredenheid"].drop_nulls().to_list()) <= {1, 2, 3, 4, 5}
    assert synthetic["opleidingscode"].str.contains(r"^[0-9]{6}$").all()
    report = json.loads((tmp_path / "uit.rapport.json").read_text())
    assert report["structuur"]["gelijk"] is True
    assert report["tabellen"]["enquete"]["uitgesloten"] == ["opmerking"]
    assert report["data_free"] is False


# ----------------------------------------------------------------------- data-free (#11)


def test_example_spec_data_free(tmp_path):
    output = tmp_path / "uit.parquet"
    args = ["synthesize", "--spec", str(EXAMPLE), "--rows", "100", str(output), "--seed", "5"]
    assert main(args) == EXIT_OK
    synthetic = pl.read_parquet(output)
    assert synthetic.height == 100
    assert synthetic.schema == pl.Schema(
        {
            "studentnummer": pl.Int64,
            "tevredenheid": pl.Int64,
            "opleidingscode": pl.String,
            "inschrijfdatum": pl.Date,
            "gemiddeld_cijfer": pl.Float64,
        }
    )
    assert synthetic["studentnummer"].to_list() == list(range(100_000, 100_100))
    assert set(synthetic["tevredenheid"].drop_nulls().to_list()) <= {1, 2, 3, 4, 5}
    assert synthetic["tevredenheid"].null_count() > 0
    assert synthetic["opleidingscode"].str.contains(r"^[0-9]{5}$").all()
    dates = synthetic["inschrijfdatum"]
    assert dates.min() >= dt.date(2024, 9, 1) and dates.max() <= dt.date(2025, 8, 31)
    assert synthetic["gemiddeld_cijfer"].is_between(1.0, 10.0).all()
    report = json.loads((tmp_path / "uit.rapport.json").read_text())
    assert report["exitcode"] == EXIT_OK
    assert report["data_free"] is True
    assert report["structuur"]["gelijk"] is True
    assert report["tabellen"]["studenten"]["uitgesloten"] == ["opmerking"]
    assert report["tabellen"]["studenten"]["rijen"] == 100


def test_data_free_is_reproducible_and_uses_n_rows(tmp_path):
    for name, seed in (("a.csv", "7"), ("b.csv", "7"), ("c.csv", "8")):
        assert (
            main(["synthesize", "--spec", str(EXAMPLE), str(tmp_path / name), "--seed", seed]) == 0
        )
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()
    assert (tmp_path / "a.csv").read_bytes() != (tmp_path / "c.csv").read_bytes()
    # n_rows = 100 in het voorbeeld; plus de kopregel.
    assert len((tmp_path / "a.csv").read_text().splitlines()) == 101


def test_data_free_multi_table(tmp_path):
    spec_path = _write(tmp_path / "spec.toml", MULTI)
    target = tmp_path / "synthetisch"
    args = ["synthesize", "--spec", str(spec_path), str(target), "--format", "parquet"]
    assert main([*args, "--seed", "1"]) == EXIT_OK
    enquete = pl.read_parquet(target / "enquete.parquet")
    resultaten = pl.read_parquet(target / "resultaten.parquet")
    assert enquete.height == 10 and resultaten.height == 7
    assert enquete["studentnummer"].n_unique() == 10
    assert resultaten.schema == pl.Schema({"cijfer": pl.Float64})
    report = json.loads((target / "rapport.json").read_text())
    assert report["exitcode"] == EXIT_OK
    assert set(report["tabellen"]) == {"enquete", "resultaten"}


def test_data_free_from_draft_needs_distributions(tmp_path, source, capsys):
    spec_path = tmp_path / "spec.toml"
    assert main(["init-spec", str(source), str(spec_path)]) == EXIT_OK
    capsys.readouterr()
    code = main(["synthesize", "--spec", str(spec_path), str(tmp_path / "uit.csv")])
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "kolom 'tevredenheid'" in err and "zonder data is een verdeling nodig" in err
    assert not (tmp_path / "uit.csv").exists()


def test_data_free_bad_distribution_hides_values(tmp_path, capsys):
    spec_path = _write(
        tmp_path / "spec.toml",
        'config_version = "1.2"\nn_rows = 5\n[[var]]\nname = "x"\nsoort = "discrete"\n'
        f'distribution = {{implements = "core.uniform", parameters = {{lowr = "{SENTINEL}"}}}}\n',
    )
    assert main(["synthesize", "--spec", str(spec_path), str(tmp_path / "uit.csv")]) == 1
    err = capsys.readouterr().err
    assert err.splitlines()[-1].startswith("fout: ") and "kolom 'x'" in err
    assert SENTINEL not in err and "Traceback" not in err


def test_data_free_faker_and_text_categories(tmp_path):
    spec_path = _write(
        tmp_path / "spec.toml",
        'config_version = "1.2"\nname = "personen"\nn_rows = 20\n'
        '[[var]]\nname = "naam"\nfaker = "name"\ndescription = "Verzonnen naam."\n'
        '[[var]]\nname = "geslacht"\ncategorieen = ["M", "V", "X"]\n',
    )
    output = tmp_path / "uit.parquet"
    assert main(["synthesize", "--spec", str(spec_path), str(output), "--seed", "1"]) == 0
    synthetic = pl.read_parquet(output)
    assert synthetic.schema == pl.Schema({"naam": pl.String, "geslacht": pl.String})
    assert synthetic.height == 20
    assert synthetic["naam"].str.contains(" ").any()
    assert set(synthetic["geslacht"].to_list()) <= {"M", "V", "X"}
    report = json.loads((tmp_path / "uit.rapport.json").read_text())
    assert report["exitcode"] == EXIT_OK


def test_data_free_needs_rows(tmp_path, capsys):
    spec_path = _write(
        tmp_path / "spec.toml", 'config_version = "1.2"\n[[var]]\nname = "id"\nsoort = "key"\n'
    )
    assert main(["synthesize", "--spec", str(spec_path), str(tmp_path / "uit.csv")]) == 1
    assert "--rows of n_rows" in capsys.readouterr().err
    assert (
        main(["synthesize", "--spec", str(spec_path), "--rows", "3", str(tmp_path / "u.csv")]) == 0
    )
    assert pl.read_csv(tmp_path / "u.csv")["id"].to_list() == [1, 2, 3]


def test_synthesize_model_and_spec_are_exclusive(tmp_path):
    for args in (
        ["synthesize", "model.json", "--spec", str(EXAMPLE), str(tmp_path / "uit.csv")],
        ["synthesize", "--spec", str(EXAMPLE)],
        ["synthesize", str(tmp_path / "uit.csv")],
    ):
        with pytest.raises(SystemExit) as exc:
            main(args)
        assert exc.value.code == EXIT_USAGE
    assert not (tmp_path / "uit.csv").exists()


def test_ci_runs_the_example():
    workflow = (EXAMPLE.parents[1] / ".github" / "workflows" / "ci.yml").read_text()
    assert "synthesize --spec examples/spec.toml --rows 100" in workflow
