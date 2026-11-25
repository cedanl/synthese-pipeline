"""Tests voor inlezen en wegschrijven (CSV, Parquet, Delta, abfss)."""

import datetime as dt
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
    assert storage.read_csv(tmp_path / "c" / "resultaten.csv").equals(results())


# Een tabelnaam komt uit het model en wordt een map- of bestandsnaam in het doel.
UNSAFE_TABLE_NAMES = ["", ".", "..", "../buiten", "a/b", "a\\b", "C:buiten", "nul\0teken"]


@pytest.mark.parametrize("fmt", [*storage.OUTPUT_FORMATS, "parquet,csv"])
@pytest.mark.parametrize("name", [*UNSAFE_TABLE_NAMES, "ABSOLUUT"])
def test_table_names_that_are_paths_are_refused(tmp_path, fmt, name):
    if name == "ABSOLUUT":
        name = str(tmp_path / "elders" / "studenten")
    target = tmp_path / "doel" / "uit"
    with pytest.raises(SyntheseError, match="ongeldige naam"):
        storage.write_tables({"goed": results(), name: results()}, target, fmt=fmt)
    # Niets geschreven: niet buiten het doel, en ook de goede tabel en de doelmap niet.
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fmt", storage.OUTPUT_FORMATS)
def test_write_through_symlink_outside_target_is_refused(tmp_path, fmt):
    outside = tmp_path / "elders"
    outside.mkdir()
    target = tmp_path / "uit"
    target.mkdir()
    entry = "studenten" if fmt == "delta" else f"studenten.{fmt}"
    (target / entry).symlink_to(outside / entry, target_is_directory=fmt == "delta")
    with pytest.raises(SyntheseError, match="buiten de doelmap"):
        storage.write_tables({"studenten": results()}, target, fmt=fmt)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("name", ["../lh-bron/Tables/studenten", "", ".."])
def test_abfss_write_refuses_table_names_that_are_paths(monkeypatch, name):
    calls = []

    def fake_write_delta(self, uri, **kwargs):
        calls.append(uri)

    monkeypatch.setattr(pl.DataFrame, "write_delta", fake_write_delta)
    target = "abfss://ws-id@onelake.dfs.fabric.microsoft.com/lh-synth/Tables"
    with pytest.raises(SyntheseError, match="ongeldige naam"):
        storage.write_tables({"goed": results(), name: results()}, target)
    assert calls == []


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


def _conventions_frame() -> pl.DataFrame:
    """Tekst met scheidingstekens, aanhalingstekens en niet-ASCII, plus getallen en lege waarden."""
    return pl.DataFrame(
        {
            "id": pl.Series([1, 2, 3], dtype=pl.Int64),
            "naam": pl.Series(['Zoë; "de" Groot', "Ångström, A.", None], dtype=pl.String),
            "code": pl.Series(["01234", "00456", "12345"], dtype=pl.String),
            "cijfer": pl.Series([7.5, None, 6.25], dtype=pl.Float64),
            "datum": pl.Series([dt.date(2024, 1, 31), dt.date(2024, 2, 1), None], dtype=pl.Date),
            "vlag": pl.Series([True, False, None], dtype=pl.Boolean),
        }
    )


@pytest.mark.parametrize("separator", [";", ","])
def test_csv_round_trip_keeps_names_dtypes_and_values(tmp_path, separator):
    df = _conventions_frame()
    kwargs = {} if separator == ";" else {"csv_separator": separator}
    written = storage.write_tables({"t": df}, tmp_path / "uit", fmt="csv", **kwargs)
    path = tmp_path / "uit" / "t.csv"
    assert written == {"t": str(path)}
    assert path.read_bytes().decode("utf-8").splitlines()[0].count(separator) == df.width - 1
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf")  # geen BOM

    back = storage.read_csv(path)
    assert back.columns == df.columns
    assert back.schema == df.schema
    assert back.to_dicts() == df.to_dicts()

    plain = pl.read_csv(path, separator=separator, schema_overrides={"code": pl.String})
    assert plain.columns == df.columns
    assert plain["naam"].to_list() == df["naam"].to_list()
    assert plain["cijfer"].dtype == pl.Float64


def test_csv_quotes_text_but_not_numbers_dates_or_nulls(tmp_path):
    storage.write_tables({"t": _conventions_frame()}, tmp_path / "uit", fmt="csv")
    lines = (tmp_path / "uit" / "t.csv").read_text(encoding="utf-8").splitlines()
    assert lines[1] == '1;"Zoë; ""de"" Groot";"01234";7.5;2024-01-31;true'
    assert lines[3] == '3;;"12345";6.25;;'


def test_single_file_csv_uses_semicolon_with_or_without_format(tmp_path):
    for fmt in (None, "csv"):
        storage.write_tables({"t": _conventions_frame()}, tmp_path / "uit.csv", fmt=fmt)
        header = (tmp_path / "uit.csv").read_text(encoding="utf-8").splitlines()[0]
        assert header.split(";") == [f'"{name}"' for name in _conventions_frame().columns]


def test_parquet_and_csv_in_one_run(tmp_path):
    tables = {"studenten": students(), "resultaten": results()}
    written = storage.write_tables(tables, tmp_path / "uit", fmt="parquet,csv")
    assert written == {name: str(tmp_path / "uit" / f"{name}.parquet") for name in tables}
    assert sorted(p.name for p in (tmp_path / "uit").iterdir()) == [
        "resultaten.csv",
        "resultaten.parquet",
        "studenten.csv",
        "studenten.parquet",
    ]
    for name, df in tables.items():
        assert pl.read_parquet(tmp_path / "uit" / f"{name}.parquet").equals(df)
        assert storage.read_csv(tmp_path / "uit" / f"{name}.csv").equals(df)


def test_format_list_is_parsed_and_ordered():
    assert storage.parse_formats("parquet,csv") == ("parquet", "csv")
    assert storage.parse_formats(" csv , parquet ") == ("parquet", "csv")
    assert storage.parse_formats("delta") == ("delta",)
    assert storage.resolve_formats("uit", None) == ("delta",)
    assert storage.resolve_formats("uit", "parquet,csv") == ("parquet", "csv")
    assert storage.resolve_format("uit", "csv,parquet") == "parquet,csv"


@pytest.mark.parametrize(
    ("fmt", "message"),
    [
        ("xml", "Onbekend formaat"),
        ("parquet,xml", "Onbekend formaat"),
        ("parquet,", "Onbekend formaat"),
        ("csv,csv", "meer dan één keer"),
        ("delta,parquet", "Delta"),
    ],
)
def test_invalid_format_lists_write_nothing(tmp_path, fmt, message):
    with pytest.raises(SyntheseError, match=message):
        storage.write_tables({"resultaten": results()}, tmp_path / "uit", fmt=fmt)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "target",
    ["uit.csv", "uit.parquet", "abfss://ws-id@onelake.dfs.fabric.microsoft.com/lh/Tables"],
)
def test_two_formats_need_a_local_folder(tmp_path, target):
    if not target.startswith("abfss://"):
        target = str(tmp_path / target)
    with pytest.raises(SyntheseError, match="map"):
        storage.write_tables({"resultaten": results()}, target, fmt="parquet,csv")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("separator", ["\t", "|", ";;", ""])
def test_invalid_csv_separator_is_refused(tmp_path, separator):
    with pytest.raises(SyntheseError, match="scheidingsteken"):
        storage.write_tables(
            {"resultaten": results()}, tmp_path / "uit", fmt="csv", csv_separator=separator
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("linked", ["parquet", "csv"])
def test_symlink_escape_is_refused_with_two_formats(tmp_path, linked):
    outside = tmp_path / "elders"
    outside.mkdir()
    target = tmp_path / "uit"
    target.mkdir()
    (target / f"studenten.{linked}").symlink_to(outside / f"studenten.{linked}")
    with pytest.raises(SyntheseError, match="buiten de doelmap"):
        storage.write_tables({"studenten": results()}, target, fmt="parquet,csv")
    assert list(outside.iterdir()) == []
    assert [p.name for p in target.iterdir()] == [f"studenten.{linked}"]
