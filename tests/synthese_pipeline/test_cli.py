"""Tests voor de commandoregel: fit, synthesize, toon, rapport en exitcodes."""

import importlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from synthese_pipeline import __version__
from synthese_pipeline.cli import main
from synthese_pipeline.report import EXIT_COPIES, EXIT_ERROR, EXIT_OK, EXIT_STRUCTURE, EXIT_USAGE
from tests.synthese_pipeline.dummy import FREETEXT_MARK, SENTINEL, freetext, results, students


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_version_command_prints_package_version():
    # Het echte commando uit [project.scripts], niet main(): zo telt ook de installatie mee.
    command = shutil.which("synthese-pipeline", path=str(Path(sys.executable).parent))
    assert command is not None
    out = subprocess.run([command, "--version"], capture_output=True, text=True, check=True)
    package_version = importlib.metadata.version("synthese-pipeline")
    assert out.stdout.strip() == f"synthese-pipeline {package_version}"


def test_version_has_one_source(monkeypatch):
    # __version__ komt uit de pakketgegevens (pyproject.toml), niet uit een tweede getal.
    import synthese_pipeline

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.9.9")
    try:
        assert importlib.reload(synthese_pipeline).__version__ == "9.9.9"
    finally:
        monkeypatch.undo()
        importlib.reload(synthese_pipeline)


@pytest.fixture
def fitted(tmp_path):
    """data.csv en model.json in een eigen map; geeft (map, data, model)."""
    data = tmp_path / "data.csv"
    students().write_csv(data)
    model = tmp_path / "model.json"
    assert main(["fit", str(data), str(model), "--log", str(tmp_path / "fit.log")]) == EXIT_OK
    return tmp_path, data, model


def test_fit_writes_only_the_model(tmp_path):
    work = tmp_path / "werk"
    work.mkdir()
    data = work / "data.csv"
    students().write_csv(data)
    assert main(["fit", str(data), str(work / "model.json")]) == EXIT_OK
    assert sorted(p.name for p in work.iterdir()) == ["data.csv", "model.json"]
    gmf = json.loads((work / "model.json").read_text())
    assert gmf["gmf_version"] == "2.1"
    assert gmf["synthese_pipeline"]["disclosure_control"]["gebruikt"] is True


def test_synthesize_from_model_only(fitted):
    work, data, model = fitted
    data.unlink()
    output = work / "output.csv"
    code = main(["synthesize", str(model), str(output), "--rows", "1000", "--seed", "1"])
    assert code == EXIT_OK
    synthetic = pl.read_csv(output, schema_overrides={"opleidingscode": pl.String})
    assert synthetic.height == 1000
    assert synthetic.columns == students().columns
    assert set(synthetic["opleidingscode"].to_list()) <= {"01234", "02345", "00456"}
    report = json.loads((work / "output.rapport.json").read_text())
    assert report["exitcode"] == EXIT_OK
    assert report["seed"] == 1
    assert report["disclosure_control"]["gebruikt"] is True
    assert report["letterlijke_kopieen"] is None
    assert report["structuur"]["gelijk"] is True


def test_same_model_and_seed_give_identical_files(fitted):
    work, _, model = fitted
    for name in ("a.csv", "b.csv", "c.csv"):
        seed = "9" if name != "c.csv" else "10"
        assert main(["synthesize", str(model), str(work / name), "--seed", seed]) == EXIT_OK
    assert (work / "a.csv").read_bytes() == (work / "b.csv").read_bytes()
    assert (work / "a.csv").read_bytes() != (work / "c.csv").read_bytes()


def test_seed_is_chosen_and_reported_when_missing(fitted):
    work, _, model = fitted
    assert main(["synthesize", str(model), str(work / "uit.csv")]) == EXIT_OK
    report = json.loads((work / "uit.rapport.json").read_text())
    assert isinstance(report["seed"], int)


def test_toon_prints_columns_in_dutch(fitted, capsys):
    _, _, model = fitted
    capsys.readouterr()
    assert main(["toon", str(model)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "Disclosure control: aan" in out
    for column in students().columns:
        assert f"- {column} (" in out
    assert "lege waarden" in out


def test_folder_to_delta_keeps_schemas(tmp_path):
    source = tmp_path / "bron"
    source.mkdir()
    students().write_parquet(source / "studenten.parquet")
    results().write_delta(str(source / "resultaten"))
    model = tmp_path / "model.json"
    assert main(["fit", str(source), str(model)]) == EXIT_OK
    target = tmp_path / "synthetisch"
    code = main(["synthesize", str(model), str(target), "--seed", "2", "--source", str(source)])
    assert code == EXIT_OK
    assert pl.read_delta(str(target / "studenten")).schema == students().schema
    assert pl.read_delta(str(target / "resultaten")).schema == results().schema
    report = json.loads((target / "rapport.json").read_text())
    assert report["bron_vergeleken"] is True
    assert report["letterlijke_kopieen"] == 0
    measures = report["tabellen"]["studenten"]["verdelingen"]
    assert measures["geslacht"]["maat"] == "tv_afstand"
    assert measures["gemiddeld_cijfer"]["maat"] == "wasserstein"
    assert measures["inschrijfdatum"]["maat"] == "wasserstein_dagen"


def test_parquet_format_for_folder(fitted):
    work, _, model = fitted
    target = work / "pq"
    assert main(["synthesize", str(model), str(target), "--format", "parquet", "--seed", "1"]) == 0
    assert pl.read_parquet(target / "data.parquet").schema == students().schema


def test_exit_code_structure_deviates(fitted):
    work, _, model = fitted
    other = work / "anders.csv"
    students().with_columns(pl.col("leeftijd").cast(pl.Float64) + 0.5).write_csv(other)
    code = main(
        ["synthesize", str(model), str(work / "uit.csv"), "--seed", "1", "--source", str(other)]
    )
    assert code == EXIT_STRUCTURE
    report = json.loads((work / "uit.rapport.json").read_text())
    assert report["structuur"]["gelijk"] is False
    assert report["tabellen"]["data"]["structuur"]["afwijkingen"] == [
        "kolom 'leeftijd': type Float64 verwacht, Int64 gevonden"
    ]


def test_exit_code_copies_found(fitted):
    work, _, model = fitted
    first = work / "eerste.csv"
    assert main(["synthesize", str(model), str(first), "--seed", "4"]) == EXIT_OK
    # Doe alsof de synthetische data de bron is: elke unieke rij is dan een letterlijke kopie.
    code = main(
        ["synthesize", str(model), str(work / "tweede.csv"), "--seed", "4", "--source", str(first)]
    )
    assert code == EXIT_COPIES
    report = json.loads((work / "tweede.rapport.json").read_text())
    assert report["letterlijke_kopieen"] == 200


def test_exit_code_error_and_usage(tmp_path, capsys):
    assert main(["fit", str(tmp_path / "nergens.csv"), str(tmp_path / "m.json")]) == EXIT_ERROR
    assert "bestaat niet" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        main(["synthesize", "m.json", "uit.csv", "--rows", "0"])
    assert exc.value.code == EXIT_USAGE
    assert main([]) == EXIT_USAGE


@pytest.mark.parametrize("fmt", ["csv", "delta"])
def test_synthesize_refuses_model_with_table_name_outside_target(fitted, capsys, fmt):
    # Een met de hand aangepast model mag niet buiten het doel schrijven, zoals over de bron.
    work, data, model = fitted
    before = data.read_bytes()
    gmf = json.loads(model.read_text())
    gmf["tables"][0]["name"] = "../data" if fmt == "csv" else str(work / "elders")
    edited = work / "bewerkt.json"
    edited.write_text(json.dumps(gmf))
    code = main(["synthesize", str(edited), str(work / "uit"), "--format", fmt, "--seed", "1"])
    assert code == EXIT_ERROR
    assert data.read_bytes() == before
    assert not (work / "uit").exists()
    assert not (work / "elders").exists()
    err = capsys.readouterr().err
    assert "ongeldige naam" in err
    assert gmf["tables"][0]["name"] not in err


def test_type_override_on_fit(tmp_path, capsys):
    data = tmp_path / "data.csv"
    students().write_csv(data)
    model = tmp_path / "model.json"
    assert main(["fit", str(data), str(model), "--type", "leeftijd=categorical"]) == EXIT_OK
    gmf = json.loads(model.read_text())
    assert gmf["synthese_pipeline"]["tabellen"]["data"]["kolommen"]["leeftijd"]["soort"] == (
        "categorical"
    )
    capsys.readouterr()
    assert main(["fit", str(data), str(model), "--type", "geslacht=discrete"]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert "geslacht" in err and SENTINEL not in err


def test_no_disclosure_flag(tmp_path):
    data = tmp_path / "data.csv"
    students().write_csv(data)
    model = tmp_path / "model.json"
    assert main(["fit", str(data), str(model), "--no-disclosure"]) == EXIT_OK
    gmf = json.loads(model.read_text())
    assert gmf["synthese_pipeline"]["disclosure_control"]["gebruikt"] is False
    assert main(["synthesize", str(model), str(tmp_path / "uit.csv"), "--seed", "1"]) == EXIT_OK
    report = json.loads((tmp_path / "uit.rapport.json").read_text())
    assert report["disclosure_control"]["gebruikt"] is False


def test_log_and_report_contain_no_values(fitted):
    work, data, model = fitted
    log = work / "syn.log"
    code = main(
        [
            "synthesize",
            str(model),
            str(work / "uit"),
            "--seed",
            "1",
            "--source",
            str(data),
            "--log",
            str(log),
        ]
    )
    assert code == EXIT_OK
    texts = {
        "fit.log": (work / "fit.log").read_text(),
        "syn.log": log.read_text(),
        "rapport": (work / "uit" / "rapport.json").read_text(),
    }
    for name, text in texts.items():
        assert SENTINEL not in text, name
    assert "kolom 'gemiddeld_cijfer'" in texts["fit.log"]
    assert "tabel 'data': 200 rijen" in texts["fit.log"]
    assert "gemiddeld_cijfer" in texts["rapport"]
    assert SENTINEL not in model.read_text()


def test_help_lists_exit_codes():
    out = subprocess.run(
        [sys.executable, "-m", "synthese_pipeline.cli", "--help"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for code in ("0", "1", "2", "3", "4"):
        assert f"  {code}  " in out
    assert "structuur wijkt af" in out


def test_freetext_excluded_by_default_and_named_in_log_and_report(tmp_path, capsys):
    data = tmp_path / "data.csv"
    freetext().write_csv(data)
    model = tmp_path / "model.json"
    fit_log = tmp_path / "fit.log"
    assert main(["fit", str(data), str(model), "--log", str(fit_log)]) == EXIT_OK
    gmf_text = model.read_text()
    assert FREETEXT_MARK not in gmf_text
    gmf = json.loads(gmf_text)
    assert set(gmf["synthese_pipeline"]["tabellen"]["data"]["uitgesloten"]) == {
        "opmerking",
        "naam",
    }
    syn_log = tmp_path / "syn.log"
    output = tmp_path / "uit.csv"
    args = ["synthesize", str(model), str(output), "--seed", "1", "--source", str(data)]
    assert main([*args, "--log", str(syn_log)]) == EXIT_OK
    synthetic = pl.read_csv(output, schema_overrides={"opleidingscode": pl.String})
    assert synthetic.columns == ["postcode", "opleidingscode", "cijfer"]
    report_text = (tmp_path / "uit.rapport.json").read_text()
    report = json.loads(report_text)
    assert report["tabellen"]["data"]["uitgesloten"] == ["opmerking", "naam"]
    for text in (fit_log.read_text(), syn_log.read_text(), report_text):
        assert FREETEXT_MARK not in text
    assert "uitgesloten (vrije tekst): opmerking, naam" in fit_log.read_text()
    assert "uitgesloten (vrije tekst): opmerking, naam" in syn_log.read_text()
    capsys.readouterr()
    assert main(["toon", str(model)]) == EXIT_OK
    assert "- opmerking: uitgesloten (vrije tekst)" in capsys.readouterr().out


def test_faker_option_on_fit(tmp_path):
    data = tmp_path / "data.csv"
    freetext().write_csv(data)
    model = tmp_path / "model.json"
    args = [
        "fit",
        str(data),
        str(model),
        "--faker",
        "naam=name",
        "--faker",
        "data.opmerking=sentence",
    ]
    assert main(args) == EXIT_OK
    gmf_text = model.read_text()
    assert FREETEXT_MARK not in gmf_text
    kolommen = json.loads(gmf_text)["synthese_pipeline"]["tabellen"]["data"]["kolommen"]
    assert kolommen["naam"]["soort"] == "faker"
    assert kolommen["opmerking"]["soort"] == "faker"
    output = tmp_path / "uit.csv"
    assert main(["synthesize", str(model), str(output), "--seed", "1"]) == EXIT_OK
    synthetic = pl.read_csv(output, schema_overrides={"opleidingscode": pl.String})
    assert synthetic.columns == freetext().columns
    real = freetext()
    for column in ("naam", "opmerking"):
        assert set(synthetic[column].to_list()).isdisjoint(real[column].to_list())


def test_type_override_reincludes_freetext(tmp_path):
    data = tmp_path / "data.csv"
    freetext().write_csv(data)
    model = tmp_path / "model.json"
    assert main(["fit", str(data), str(model), "--type", "naam=text"]) == EXIT_OK
    gmf = json.loads(model.read_text())
    assert gmf["synthese_pipeline"]["tabellen"]["data"]["kolommen"]["naam"]["soort"] == "text"
    assert list(gmf["synthese_pipeline"]["tabellen"]["data"]["uitgesloten"]) == ["opmerking"]


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ("naam=bestaatniet", "bestaatniet"),
        ("onbekend=name", "--faker noemt kolom 'onbekend'"),
        ("naam", "kolom=provider"),
    ],
)
def test_faker_errors_exit_non_zero(tmp_path, capsys, option, message):
    data = tmp_path / "data.csv"
    freetext().write_csv(data)
    model = tmp_path / "model.json"
    assert main(["fit", str(data), str(model), "--faker", option]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert message in err
    assert FREETEXT_MARK not in err
    assert not model.exists()


def test_fit_help_lists_new_kinds_and_faker():
    out = subprocess.run(
        [sys.executable, "-m", "synthese_pipeline.cli", "fit", "--help"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "freetext" in out and "faker" in out
    assert "--faker" in out
