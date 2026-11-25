"""Tests voor het afleiden van kolomtypes."""

import datetime as dt

import polars as pl
import pytest

from synthese_pipeline import column_types as ct
from synthese_pipeline.errors import SyntheseError
from tests.synthese_pipeline.dummy import SENTINEL, freetext, students


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


def test_freetext_columns_in_dummy_frame_are_detected():
    df = freetext()
    kinds = {column: ct.infer_kind(df[column]) for column in df.columns}
    assert kinds["opmerking"] == "freetext"
    assert kinds["naam"] == "freetext"
    assert kinds["postcode"] in ("text", "key")
    assert kinds["opleidingscode"] == "text"
    assert kinds["cijfer"] == "continuous"


@pytest.mark.parametrize(
    ("values", "kind"),
    [
        # Een meerderheid met een spatie: zinnen en "voornaam achternaam".
        ([f"Qzxw zegt {i} dingen over les" for i in range(40)], "freetext"),
        (
            [f"Qzx{'abcdefgh'[i % 8]}{'ijklm'[i % 5]} Qzxson{'pq'[i % 2]}" for i in range(40)],
            "freetext",
        ),
        # Lang (gemiddeld >= 20 tekens) zonder spatie.
        ([f"qzxw.persoon{i:03d}@voorbeeld.nl" for i in range(40)] + [None], "freetext"),
        # Ook met lege waarden en dubbele waarden die niet categorisch zijn.
        ([f"opmerking nummer {i % 35} hier" for i in range(70)] + [None], "freetext"),
        # Codes van een vaste vorm blijven een patroon.
        ([f"{1000 + i * 7} {chr(65 + i % 26)}{chr(66 + i % 24)}" for i in range(60)], "key"),
        (
            [f"{1000 + i * 7} {chr(65 + i % 26)}{chr(66 + i % 24)}" for i in range(60)] + [None],
            "text",
        ),
        ([f"{30000 + i}" for i in range(60)] + [None], "text"),
        ([f"B{i:02d} C{i % 7}" for i in range(60)] + [None], "text"),
        # Korte unieke woorden zonder spatie.
        ([f"tok{i}x" for i in range(60)] + [None], "text"),
        # Weinig verschillende namen van opleidingen: een categorie, geen vrije tekst.
        (["B Informatica", "M Rechten", "B Psychologie"] * 20, "categorical"),
    ],
)
def test_freetext_rules(values, kind):
    assert ct.infer_kind(pl.Series("kolom", values, dtype=pl.String)) == kind


def test_opleidingsnaam_is_not_a_code_shape():
    names = [f"B Opleiding{'x' * (i % 9)} {chr(65 + i % 26)}" for i in range(60)]
    assert ct.infer_kind(pl.Series("opleidingsnaam", names)) == "freetext"


def test_freetext_in_categorical_dtype_is_detected():
    values = [f"Qzxw zegt {i} dingen over les" for i in range(40)]
    assert ct.infer_kind(pl.Series("o", values).cast(pl.Categorical)) == "freetext"


def test_kinds_list_new_kinds_and_overrides_win():
    assert "freetext" in ct.KINDS and "faker" in ct.KINDS
    df = freetext()
    kinds = ct.kinds_for_table("t", df, {"opmerking": "text", "t.opleidingscode": "freetext"})
    assert kinds["opmerking"] == "text"
    assert kinds["opleidingscode"] == "freetext"
    assert kinds["naam"] == "freetext"


def test_faker_overrides_set_kind_and_provider():
    fakers = ct.parse_faker_overrides(["naam=name", "t.opmerking = sentence"])
    assert fakers == {"naam": "name", "t.opmerking": "sentence"}
    df = freetext()
    kinds = ct.kinds_for_table("t", df, {}, fakers)
    assert kinds["naam"] == "faker" and kinds["opmerking"] == "faker"
    assert ct.faker_types_for_table("t", df, fakers) == {"naam": "name", "opmerking": "sentence"}
    assert ct.faker_types_for_table("andere", df, fakers) == {"naam": "name"}


@pytest.mark.parametrize("item", ["naam=bestaatniet", "naam=random_int", "naam=_private", "naam="])
def test_faker_unknown_provider_is_refused(item):
    with pytest.raises(SyntheseError, match="--faker"):
        ct.parse_faker_overrides([item])


def test_faker_needs_kolom_is_provider():
    with pytest.raises(SyntheseError, match="kolom=provider"):
        ct.parse_faker_overrides(["naam"])


def test_faker_kind_without_provider_is_refused():
    with pytest.raises(SyntheseError, match="--faker"):
        ct.kinds_for_table("t", freetext(), {"naam": "faker"})


def test_faker_and_other_type_for_same_column_conflict():
    with pytest.raises(SyntheseError, match="naam"):
        ct.check_faker_conflicts({"naam": "text"}, {"naam": "name"})
    ct.check_faker_conflicts({"naam": "faker", "opmerking": "text"}, {"naam": "name"})


def test_prepare_faker_gives_faker_distribution_without_values():
    series = freetext()["naam"]
    prepared, spec = ct.prepare_column(series, "faker", faker_type="name")
    assert prepared.dtype == pl.String
    assert spec == {
        "distribution": {
            "name": "core.faker",
            "parameters": {"faker_type": "name", "locale": "nl_NL"},
        }
    }
    with pytest.raises(SyntheseError, match="tekst"):
        ct.prepare_column(pl.Series("n", [1, 2]), "faker", faker_type="name")
    with pytest.raises(SyntheseError):
        ct.prepare_column(series, "freetext")


def test_code_pattern_is_made_from_shapes_only():
    postcodes = [f"{1000 + i * 7} {chr(65 + i % 26)}{chr(66 + i % 24)}" for i in range(60)]
    assert ct.code_pattern(pl.Series("pc", postcodes + [None])) == "[0-9]{4} [A-Z]{2}"
    codes = [f"{30000 + i}" for i in range(40)] + [f"a-{i}.x" for i in range(10, 30)]
    assert ct.code_pattern(pl.Series("c", codes)) == "([0-9]{5}|[a-z]\\-[0-9]{2}\\.[a-z])"
    # Een vorm die minder dan 10 keer voorkomt komt niet in het patroon.
    rare = [f"{30000 + i}" for i in range(60)] + [f"{SENTINEL}9"]
    assert ct.code_pattern(pl.Series("c", rare)) == "[0-9]{5}"
    assert ct.code_pattern(pl.Series("c", rare).cast(pl.Categorical)) == "[0-9]{5}"


@pytest.mark.parametrize(
    "series",
    [
        pl.Series("t", [f"Qzxw zegt {i} dingen" for i in range(40)]),
        pl.Series("t", [f"tok{'x' * (i % 9)}{i}" for i in range(40)]),
        pl.Series("t", ["12"] * 5),
        pl.Series("t", [None, None], dtype=pl.String),
        pl.Series("n", list(range(40))),
    ],
)
def test_code_pattern_none_for_non_codes(series):
    assert ct.code_pattern(series) is None


def test_prepare_text_and_key_codes_get_shape_regex():
    codes = pl.Series("c", [f"S{i:05d}" for i in range(50)])
    _, spec = ct.prepare_column(codes, "key")
    assert spec == {
        "distribution": {
            "name": "core.regex",
            "parameters": {"regex_data": "[A-Z][0-9]{5}"},
            "unique": True,
        }
    }
    _, spec = ct.prepare_column(pl.Series("c", [f"{30000 + i % 40}" for i in range(80)]), "text")
    assert spec == {
        "distribution": {"name": "core.regex", "parameters": {"regex_data": "[0-9]{5}"}}
    }
    _, spec = ct.prepare_column(
        pl.Series("t", [f"woord{'x' * (i % 9)}" for i in range(40)]), "text"
    )
    assert spec is None
