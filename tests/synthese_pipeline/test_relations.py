"""Tests voor relaties tussen tabellen (metasyn MultiFrame): specificatie, fit, data-free, rapport.

Alleen gegenereerde dummydata (zie ``dummy.py``).
"""

import json
from pathlib import Path

import polars as pl
import pytest

from synthese_pipeline.cli import main
from synthese_pipeline.describe import describe_model
from synthese_pipeline.errors import SyntheseError
from synthese_pipeline.model import (
    column_kinds,
    data_free_model,
    expected_schemas,
    fit_tables,
    load_model,
    save_model,
    synthesize_tables,
    tool_info,
)
from synthese_pipeline.report import (
    EXIT_CODES,
    EXIT_ERROR,
    EXIT_OK,
    EXIT_RELATIONS,
    RELATION_LIMITATION,
    build_report,
)
from synthese_pipeline.spec import INFER, SUBSET, load_spec, parse_relation, parse_spec
from tests.synthese_pipeline.dummy import SENTINEL, linked_tables, results, survey

README = Path(__file__).resolve().parents[2] / "README.md"

RELATIONS = [
    "inschrijving[student_id] SUBSET OF student[student_id]",
    "resultaat[inschrijving_id] INFER FROM inschrijving[inschrijving_id]",
]


def _spec_dict(relations: list[str] | None = None) -> dict:
    data: dict = {
        "config_version": "2.0",
        "table": [
            {"name": "student", "var": [{"name": "student_id"}, {"name": "leeftijd"}]},
            {"name": "inschrijving", "var": [{"name": "inschrijving_id"}, {"name": "student_id"}]},
            {"name": "resultaat", "var": [{"name": "inschrijving_id"}]},
        ],
    }
    if relations is not None:
        data["relations"] = relations
    return data


def _orphans(child: pl.DataFrame, column: str, parent: pl.DataFrame, key: str) -> int:
    return (
        child.select(column)
        .join(parent.select(key), left_on=column, right_on=key, how="anti")
        .height
    )


def _write_tables(folder: Path) -> Path:
    folder.mkdir()
    for name, df in linked_tables().items():
        df.write_parquet(folder / f"{name}.parquet")
    return folder


SPEC_TOML = """
config_version = "2.0"
relations = [
  "inschrijving[student_id] SUBSET OF student[student_id]",
  "resultaat[inschrijving_id] INFER FROM inschrijving[inschrijving_id]",
]

[[table]]
name = "student"
[[table.var]]
name = "student_id"
description = "Studentnummer."

[[table]]
name = "inschrijving"
[[table.var]]
name = "inschrijving_id"
[[table.var]]
name = "student_id"

[[table]]
name = "resultaat"
[[table.var]]
name = "inschrijving_id"
"""


# ------------------------------------------------------------------------------ specificatie


def test_relations_are_parsed():
    spec = parse_spec(_spec_dict(RELATIONS))
    assert [str(rel) for rel in spec.relations] == RELATIONS
    first, second = spec.relations
    assert (first.table, first.column, first.kind) == ("inschrijving", "student_id", SUBSET)
    assert (first.ref_table, first.ref_column) == ("student", "student_id")
    assert second.kind == INFER
    # De kolom waarnaar verwezen wordt is een sleutel.
    assert spec.tables[0].columns["student_id"].kind == "key"
    assert spec.tables[1].columns["inschrijving_id"].kind == "key"
    # De verwijzende kolom houdt geen soort.
    assert spec.tables[1].columns["student_id"].kind is None


def test_relation_syntax_is_lenient_about_case_and_spaces():
    rel = parse_relation("  inschrijving [student_id]  subset   of student[student_id] ")
    assert str(rel) == "inschrijving[student_id] SUBSET OF student[student_id]"


def test_spec_without_relations_has_none():
    assert parse_spec(_spec_dict()).relations == ()
    assert parse_spec(_spec_dict([])).relations == ()


INVALID_RELATIONS = {
    "syntax": ([f"inschrijving[student_id] HOORT BIJ student[{SENTINEL}]"], "is ongeldig"),
    "equals": (["inschrijving[student_id] EQUALS student[student_id]"], "is ongeldig"),
    "table": (["inschrijving[student_id] SUBSET OF studenten[student_id]"], "tabel 'studenten'"),
    "column": (["inschrijving[student_nr] SUBSET OF student[student_id]"], "kolom 'student_nr'"),
    "self": (["student[student_id] SUBSET OF student[student_id]"], "naar zichzelf"),
    "chain": (
        [
            "inschrijving[student_id] SUBSET OF student[student_id]",
            "student[student_id] SUBSET OF resultaat[inschrijving_id]",
        ],
        "tegelijk verwijzen en sleutel",
    ),
    "twice": (
        [
            "resultaat[inschrijving_id] SUBSET OF inschrijving[inschrijving_id]",
            "resultaat[inschrijving_id] SUBSET OF student[student_id]",
        ],
        "verwijst al",
    ),
    "not_text": ([1], "lijst met teksten"),
}


@pytest.mark.parametrize("case", sorted(INVALID_RELATIONS))
def test_invalid_relation_gives_dutch_error(case):
    relations, expected = INVALID_RELATIONS[case]
    with pytest.raises(SyntheseError) as exc:
        parse_spec(_spec_dict(relations))
    message = str(exc.value)
    assert message.startswith("De specificatie")
    assert expected in message
    if isinstance(relations[0], str) and case != "syntax":
        assert any(f"relatie '{rel}'" in message for rel in relations)


def test_relation_to_non_key_column_is_refused():
    data = _spec_dict(["inschrijving[student_id] SUBSET OF student[leeftijd]"])
    data["table"][0]["var"][1]["soort"] = "discrete"
    with pytest.raises(SyntheseError, match="moet uniek zijn"):
        parse_spec(data)


def test_relation_with_excluded_column_is_refused():
    data = _spec_dict(RELATIONS)
    data["table"][2]["var"][0]["uitgesloten"] = True
    with pytest.raises(SyntheseError, match="is uitgesloten"):
        parse_spec(data)


def test_relations_in_single_table_spec_are_refused():
    data = {
        "config_version": "1.2",
        "relations": ["data[a] SUBSET OF data[b]"],
        "var": [{"name": "a"}, {"name": "b"}],
    }
    with pytest.raises(SyntheseError) as exc:
        parse_spec(data)
    assert "relatie 'data[a] SUBSET OF data[b]'" in str(exc.value)
    assert 'config_version = "2.0"' in str(exc.value)


def test_relations_per_table_are_refused():
    data = _spec_dict()
    data["table"][1]["relations"] = RELATIONS
    with pytest.raises(SyntheseError, match="bovenaan het bestand"):
        parse_spec(data)


def test_infer_error_via_cli_has_no_values(tmp_path, capsys):
    folder = tmp_path / "bron"
    folder.mkdir()
    tables = linked_tables()
    tables["inschrijving"] = tables["inschrijving"].with_columns(
        pl.when(pl.int_range(pl.len()) < 4)
        .then(pl.lit(f"{SENTINEL}9999"))
        .otherwise(pl.col("student_id"))
        .alias("student_id")
    )
    for name, df in tables.items():
        df.write_parquet(folder / f"{name}.parquet")
    spec_path = tmp_path / "spec.toml"
    spec_path.write_text(SPEC_TOML.replace("SUBSET OF student", "INFER FROM student"), "utf-8")
    model = tmp_path / "model.json"
    assert main(["fit", str(folder), str(model), "--spec", str(spec_path)]) == EXIT_ERROR
    err = capsys.readouterr().err
    last = err.splitlines()[-1]
    assert last.startswith(
        "fout: Relatie 'inschrijving[student_id] INFER FROM student[student_id]'"
    )
    assert "4 waarden" in last
    assert "Traceback" not in err
    assert SENTINEL not in err
    assert not model.exists()


# ------------------------------------------------------------------------------------- fit


@pytest.fixture(scope="module")
def fitted() -> dict:
    return fit_tables(linked_tables(), spec=parse_spec(_spec_dict(RELATIONS)))


def test_fit_writes_relations_in_gmf(fitted):
    assert fitted["relations"] == [
        {
            "foreign_table": "inschrijving",
            "foreign_key": "student_id",
            "primary_table": "student",
            "primary_key": "student_id",
            "relation_type": "SUBSET OF",
        },
        {
            "foreign_table": "resultaat",
            "foreign_key": "inschrijving_id",
            "primary_table": "inschrijving",
            "primary_key": "inschrijving_id",
            "relation_type": "SUBSET OF",
        },
    ]
    tables = tool_info(fitted)["tabellen"]
    assert tables["inschrijving"]["kolommen"]["student_id"]["verwijst_naar"] == (
        "student[student_id]"
    )
    assert column_kinds(fitted)["student"]["student_id"] == "key"


def test_foreign_key_takes_the_distribution_of_the_key(fitted):
    by_name = {
        (table["name"], var["name"]): var for table in fitted["tables"] for var in table["vars"]
    }
    primary = by_name[("student", "student_id")]["distribution"]
    foreign = by_name[("inschrijving", "student_id")]["distribution"]
    assert primary["unique"] is True
    assert foreign["unique"] is False
    assert foreign["name"] == primary["name"]
    assert foreign["parameters"] == primary["parameters"]
    assert by_name[("inschrijving", "student_id")]["prop_missing"] == 0.0
    # Er staat geen sleutelwaarde uit de bron in het model.
    assert SENTINEL not in json.dumps(fitted)


def test_synthetic_keys_are_consistent_three_levels(fitted):
    synthetic = synthesize_tables(fitted, seed=11)
    student, inschrijving, resultaat = (
        synthetic["student"],
        synthetic["inschrijving"],
        synthetic["resultaat"],
    )
    assert (student.height, inschrijving.height, resultaat.height) == (60, 120, 300)
    assert _orphans(inschrijving, "student_id", student, "student_id") == 0
    assert _orphans(resultaat, "inschrijving_id", inschrijving, "inschrijving_id") == 0
    assert student["student_id"].is_unique().all()
    assert inschrijving["inschrijving_id"].is_unique().all()
    # Types blijven die van de bron.
    for name, df in linked_tables().items():
        assert synthetic[name].schema == df.schema


def test_synthesize_with_relations_is_reproducible(fitted):
    first = synthesize_tables(fitted, seed=3)
    second = synthesize_tables(fitted, seed=3)
    other = synthesize_tables(fitted, seed=4)
    assert all(first[name].equals(second[name]) for name in first)
    assert not first["inschrijving"].equals(other["inschrijving"])


def test_synthesize_from_saved_gmf_only(tmp_path, fitted):
    path = tmp_path / "model.json"
    save_model(fitted, path)
    loaded = load_model(path)
    assert loaded["relations"] == fitted["relations"]
    from_file = synthesize_tables(loaded, seed=9, rows=50)
    direct = synthesize_tables(fitted, seed=9, rows=50)
    assert all(from_file[name].equals(direct[name]) for name in direct)
    assert {df.height for df in from_file.values()} == {50}
    assert (
        _orphans(
            from_file["resultaat"], "inschrijving_id", from_file["inschrijving"], "inschrijving_id"
        )
        == 0
    )


def test_infer_with_orphans_in_source_is_refused():
    tables = linked_tables()
    tables["resultaat"] = tables["resultaat"].with_columns(
        pl.when(pl.int_range(pl.len()) < 3)
        .then(pl.lit(1, dtype=pl.Int64))
        .otherwise(pl.col("inschrijving_id"))
        .alias("inschrijving_id")
    )
    with pytest.raises(SyntheseError) as exc:
        fit_tables(tables, spec=parse_spec(_spec_dict(RELATIONS)))
    assert "3 waarden" in str(exc.value) and "SUBSET OF" in str(exc.value)


def test_type_override_on_key_column_is_refused():
    with pytest.raises(SyntheseError, match="moet soort key zijn"):
        fit_tables(
            linked_tables(),
            spec=parse_spec(_spec_dict(RELATIONS)),
            overrides={"inschrijving.inschrijving_id": "discrete"},
        )


def test_describe_shows_relations(fitted):
    text = describe_model(fitted)
    assert "Relaties" in text
    assert "inschrijving[student_id] SUBSET OF student[student_id]" in text


# ------------------------------------------------------------------------------- data-free


DATA_FREE = """
config_version = "2.0"
relations = [
  "inschrijving[student_id] SUBSET OF student[student_id]",
  "resultaat[inschrijving_id] INFER FROM inschrijving[inschrijving_id]",
]

[[table]]
name = "student"
n_rows = 40
[[table.var]]
name = "student_id"
brontype = "Int64"

[[table]]
name = "inschrijving"
n_rows = 80
[[table.var]]
name = "inschrijving_id"
brontype = "String"
distribution = {implements = "core.regex", parameters = {regex_data = "I[0-9]{6}"}}
[[table.var]]
name = "student_id"
brontype = "Int64"
description = "Verwijst naar student."

[[table]]
name = "resultaat"
n_rows = 200
[[table.var]]
name = "inschrijving_id"
brontype = "String"
[[table.var]]
name = "cijfer"
brontype = "Float64"
distribution = {implements = "core.uniform", parameters = {lower = 1.0, upper = 10.0}}
"""


def test_data_free_with_relations(tmp_path):
    spec_path = tmp_path / "spec.toml"
    spec_path.write_text(DATA_FREE, encoding="utf-8")
    gmf = data_free_model(load_spec(spec_path))
    assert [rel["relation_type"] for rel in gmf["relations"]] == ["SUBSET OF", "SUBSET OF"]
    first = synthesize_tables(gmf, seed=2)
    second = synthesize_tables(gmf, seed=2)
    assert all(first[name].equals(second[name]) for name in first)
    assert (first["student"].height, first["inschrijving"].height) == (40, 80)
    assert first["resultaat"].height == 200
    assert _orphans(first["inschrijving"], "student_id", first["student"], "student_id") == 0
    assert (
        _orphans(first["resultaat"], "inschrijving_id", first["inschrijving"], "inschrijving_id")
        == 0
    )
    assert first["student"]["student_id"].is_unique().all()
    assert first["inschrijving"]["inschrijving_id"].is_unique().all()
    assert first["inschrijving"]["inschrijving_id"].str.contains(r"^I[0-9]{6}$").all()
    kinds = column_kinds(gmf)
    assert kinds["inschrijving"]["inschrijving_id"] == "key"
    assert kinds["inschrijving"]["student_id"] == "discrete"
    assert kinds["resultaat"]["inschrijving_id"] == "text"


def test_data_free_cli_with_relations(tmp_path):
    spec_path = tmp_path / "spec.toml"
    spec_path.write_text(DATA_FREE, encoding="utf-8")
    target = tmp_path / "uit"
    args = ["synthesize", "--spec", str(spec_path), str(target), "--format", "csv", "--seed", "1"]
    assert main(args) == EXIT_OK
    report = json.loads((target / "rapport.json").read_text())
    assert [entry["wezen"] for entry in report["relaties"]] == [0, 0]
    assert report["beperkingen"] == [RELATION_LIMITATION]


# ---------------------------------------------------------------------------------- rapport


def _report(gmf: dict, synthetic: dict[str, pl.DataFrame]) -> dict:
    return build_report(
        gmf_tool_info=tool_info(gmf),
        expected=expected_schemas(gmf),
        kinds=column_kinds(gmf),
        synthetic=synthetic,
        real=None,
        seed=1,
        relations=gmf["relations"],
    )


def test_report_counts_orphans_per_relation(fitted):
    synthetic = synthesize_tables(fitted, seed=5)
    report = _report(fitted, synthetic)
    assert report["exitcode"] == EXIT_OK
    entry = report["relaties"][0]
    assert entry == {
        "relatie": "inschrijving[student_id] SUBSET OF student[student_id]",
        "tabel": "inschrijving",
        "kolom": "student_id",
        "verwijst_naar_tabel": "student",
        "verwijst_naar_kolom": "student_id",
        "rijen_gecontroleerd": 120,
        "lege_verwijzingen": 0,
        "wezen": 0,
        "dubbele_sleutels": 0,
    }
    assert report["beperkingen"] == [RELATION_LIMITATION]
    assert "9" in RELATION_LIMITATION and "5" in RELATION_LIMITATION


def test_broken_synthetic_table_gives_orphans_and_exit_code(fitted):
    synthetic = synthesize_tables(fitted, seed=5)
    broken = synthetic["resultaat"].with_columns(
        pl.when(pl.int_range(pl.len()) < 7)
        .then(pl.lit(-1, dtype=pl.Int64))
        .otherwise(pl.col("inschrijving_id"))
        .alias("inschrijving_id")
    )
    report = _report(fitted, {**synthetic, "resultaat": broken})
    entry = report["relaties"][1]
    assert entry["wezen"] == 7 and entry["rijen_gecontroleerd"] == 300
    assert report["exitcode"] == EXIT_RELATIONS
    assert report["uitkomst"] == EXIT_CODES[EXIT_RELATIONS]


def test_duplicate_primary_keys_give_exit_code(fitted):
    synthetic = synthesize_tables(fitted, seed=5)
    student = synthetic["student"]
    duplicated = pl.concat([student, student.head(2)])
    report = _report(fitted, {**synthetic, "student": duplicated})
    assert report["relaties"][0]["dubbele_sleutels"] == 2
    assert report["exitcode"] == EXIT_RELATIONS


def test_cli_end_to_end_with_relations_and_no_values(tmp_path):
    folder = _write_tables(tmp_path / "bron")
    spec_path = tmp_path / "spec.toml"
    spec_path.write_text(SPEC_TOML, encoding="utf-8")
    model = tmp_path / "model.json"
    log_fit = tmp_path / "fit.log"
    args = ["fit", str(folder), str(model), "--spec", str(spec_path), "--log", str(log_fit)]
    assert main(args) == EXIT_OK
    assert json.loads(model.read_text())["relations"]
    target = tmp_path / "uit"
    log_syn = tmp_path / "synthesize.log"
    args = [
        "synthesize",
        str(model),
        str(target),
        "--format",
        "parquet",
        "--seed",
        "8",
        "--source",
        str(folder),
        "--log",
        str(log_syn),
    ]
    assert main(args) == EXIT_OK
    report_text = (target / "rapport.json").read_text()
    report = json.loads(report_text)
    assert [entry["wezen"] for entry in report["relaties"]] == [0, 0]
    assert report["letterlijke_kopieen"] == 0
    # Verwijzende kolommen krijgen geen verdelingsmaat: hun waarden komen uit de sleutel.
    measure = report["tabellen"]["resultaat"]["verdelingen"]["inschrijving_id"]
    assert measure["maat"] is None
    for text in (report_text, model.read_text(), log_fit.read_text(), log_syn.read_text()):
        assert SENTINEL not in text
    student = pl.read_parquet(target / "student.parquet")
    inschrijving = pl.read_parquet(target / "inschrijving.parquet")
    assert _orphans(inschrijving, "student_id", student, "student_id") == 0


# ---------------------------------------------------------------------- zonder relaties gelijk


def test_spec_without_relations_report_and_gmf_unchanged(tmp_path):
    folder = tmp_path / "bron"
    folder.mkdir()
    survey().write_parquet(folder / "enquete.parquet")
    results().write_parquet(folder / "resultaten.parquet")
    model = tmp_path / "model.json"
    assert main(["fit", str(folder), str(model)]) == EXIT_OK
    gmf = json.loads(model.read_text())
    assert gmf["relations"] == []
    target = tmp_path / "uit"
    assert main(["synthesize", str(model), str(target), "--format", "csv", "--seed", "1"]) == 0
    report = json.loads((target / "rapport.json").read_text())
    assert report["relaties"] == []
    assert "beperkingen" not in report


def test_model_without_relations_key_still_loads(tmp_path):
    gmf = fit_tables({"resultaten": results()})
    del gmf["relations"]
    path = tmp_path / "oud.json"
    path.write_text(json.dumps(gmf), encoding="utf-8")
    loaded = load_model(path)
    synthetic = synthesize_tables(loaded, seed=1)
    assert synthetic["resultaten"].height == results().height


# ------------------------------------------------------------------------------------- docs


def test_readme_documents_relations():
    text = README.read_text(encoding="utf-8")
    assert "Relaties tussen tabellen worden nog niet ondersteund" not in text
    assert "SUBSET OF" in text and "INFER FROM" in text
    assert f"| {EXIT_RELATIONS} |" in text
    assert "max 9" in text
