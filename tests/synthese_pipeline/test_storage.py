"""Tests voor inlezen en wegschrijven (CSV, Parquet, Delta, abfss)."""

import subprocess
import sys

import polars as pl
import pytest

from synthese_pipeline import storage
from synthese_pipeline.errors import SyntheseError
from tests.synthese_pipeline.dummy import results, students


def test_read_folder_with_csv_parquet_and_delta(tmp_path):
    source = tmp_path / "bron"
    source.mkdir()
    students().write_csv(source / "studenten.csv")
    results().write_parquet(source / "resultaten.parquet")
    results(n=30, seed=5).write_delta(str(source / "herkansingen"))
    (source / "notities.txt").write_text("geen tabel")

    tables = storage.read_tables(source)

    assert sorted(tables) == ["herkansingen", "resultaten", "studenten"]
    assert tables["studenten"].height == 200
    assert tables["resultaten"].schema == results().schema
    assert tables["herkansingen"].height == 30


def test_read_single_delta_table(tmp_path):
    results().write_delta(str(tmp_path / "resultaten"))
    tables = storage.read_tables(tmp_path / "resultaten")
    assert list(tables) == ["resultaten"]


def test_csv_types_are_inferred_and_leading_zeros_stay_text(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        "code,likert,cijfer,datum,vlag,naam\n"
        "01234,1,7.5,2024-01-31,true,a\n"
        "00456,,6.0,2024-02-01,false,b\n"
        "12345,5,8.25,,true,c\n"
    )
    df = storage.read_tables(path)["data"]
    assert df.schema == pl.Schema(
        {
            "code": pl.String,
            "likert": pl.Int64,
            "cijfer": pl.Float64,
            "datum": pl.Date,
            "vlag": pl.Boolean,
            "naam": pl.String,
        }
    )
    assert df["code"].to_list() == ["01234", "00456", "12345"]
    assert df["likert"].to_list() == [1, None, 5]


def test_csv_with_semicolon_and_decimal_comma(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("cijfer;datum\n7,5;31-01-2024\n6,25;01-02-2024\n")
    df = storage.read_tables(path)["data"]
    assert df["cijfer"].to_list() == [7.5, 6.25]
    assert df.schema["datum"] == pl.Date


def test_duplicate_table_names_are_refused(tmp_path):
    results().write_csv(tmp_path / "resultaten.csv")
    results().write_parquet(tmp_path / "resultaten.parquet")
    with pytest.raises(SyntheseError, match="meer dan één keer"):
        storage.read_tables(tmp_path)


def test_missing_source_and_empty_folder(tmp_path):
    with pytest.raises(SyntheseError, match="bestaat niet"):
        storage.read_tables(tmp_path / "nergens")
    with pytest.raises(SyntheseError, match="Geen tabellen"):
        storage.read_tables(tmp_path)


def test_reading_does_not_import_pandas(tmp_path):
    source = tmp_path / "bron"
    source.mkdir()
    students().write_csv(source / "studenten.csv")
    results().write_parquet(source / "resultaten.parquet")
    results().write_delta(str(source / "delta"))
    code = (
        "import sys\n"
        "from synthese_pipeline.storage import read_tables\n"
        f"tables = read_tables({str(source)!r})\n"
        "assert len(tables) == 3\n"
        "print('pandas' in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "False"


def test_write_delta_folder_keeps_names_and_dtypes(tmp_path):
    tables = {"studenten": students(), "resultaten": results()}
    written = storage.write_tables(tables, tmp_path / "uit")
    assert set(written) == {"studenten", "resultaten"}
    for name, df in tables.items():
        back = pl.read_delta(str(tmp_path / "uit" / name))
        assert back.schema == df.schema
    # Opnieuw schrijven overschrijft.
    storage.write_tables(tables, tmp_path / "uit")
    assert pl.read_delta(str(tmp_path / "uit" / "studenten")).height == 200


def test_write_parquet_and_csv_folder(tmp_path):
    storage.write_tables({"resultaten": results()}, tmp_path / "p", fmt="parquet")
    assert pl.read_parquet(tmp_path / "p" / "resultaten.parquet").schema == results().schema
    storage.write_tables({"resultaten": results()}, tmp_path / "c", fmt="csv")
    assert (tmp_path / "c" / "resultaten.csv").is_file()


def test_single_file_target_needs_one_table(tmp_path):
    with pytest.raises(SyntheseError, match="2 tabellen"):
        storage.write_tables({"a": results(), "b": results()}, tmp_path / "uit.csv")


def test_format_must_match_suffix(tmp_path):
    with pytest.raises(SyntheseError, match="formaat"):
        storage.resolve_format(tmp_path / "uit.csv", "parquet")
    assert storage.resolve_format(tmp_path / "uit", None) == "delta"
    assert storage.resolve_format(tmp_path / "uit.parquet", None) == "parquet"


def test_fabric_storage_options_from_argument_and_environment(monkeypatch):
    monkeypatch.delenv(storage.TOKEN_ENV, raising=False)
    assert storage.fabric_storage_options() == {"use_fabric_endpoint": "true"}
    assert storage.fabric_storage_options("abc") == {
        "use_fabric_endpoint": "true",
        "bearer_token": "abc",
    }
    monkeypatch.setenv(storage.TOKEN_ENV, "uit-omgeving")
    assert storage.fabric_storage_options()["bearer_token"] == "uit-omgeving"


ABFSS_TABLE = "abfss://ws-id@onelake.dfs.fabric.microsoft.com/lh-id/Tables/studenten"


def test_abfss_read_passes_storage_options(monkeypatch):
    calls = {}

    def fake_read_delta(uri, storage_options=None):
        calls["uri"], calls["options"] = uri, storage_options
        return results()

    monkeypatch.setenv(storage.TOKEN_ENV, "token-123")
    monkeypatch.setattr(storage.pl, "read_delta", fake_read_delta)
    tables = storage.read_tables(ABFSS_TABLE)
    assert list(tables) == ["studenten"]
    assert calls["uri"] == ABFSS_TABLE
    assert calls["options"] == {"use_fabric_endpoint": "true", "bearer_token": "token-123"}


def test_abfss_write_passes_storage_options(monkeypatch):
    calls = []

    def fake_write_delta(self, uri, **kwargs):
        calls.append((uri, kwargs["storage_options"]))

    monkeypatch.setattr(pl.DataFrame, "write_delta", fake_write_delta)
    target = "abfss://ws-id@onelake.dfs.fabric.microsoft.com/lh-synth/Tables/"
    written = storage.write_tables(
        {"studenten": results()}, target, storage_options={"bearer_token": "t"}
    )
    assert written == {"studenten": target.rstrip("/") + "/studenten"}
    assert calls == [(target.rstrip("/") + "/studenten", {"bearer_token": "t"})]
    with pytest.raises(SyntheseError, match="alleen Delta"):
        storage.write_tables({"studenten": results()}, target, fmt="parquet")
