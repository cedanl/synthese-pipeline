"""Tests voor de controle op letterlijke kopieën en de disclosure-instelling."""

import polars as pl

from synthese_pipeline import privacy


def test_copy_of_unique_real_row_is_counted():
    real = pl.DataFrame({"a": [1, 2, 2], "b": ["x", "y", "y"]})
    synthetic = pl.DataFrame({"a": [1, 2, 3, 1], "b": ["x", "y", "z", "x"]})
    # (1, x) is uniek in de echte data en komt 2 keer voor; (2, y) is niet uniek.
    assert privacy.count_literal_copies(real, synthetic) == 2


def test_nulls_count_as_equal():
    real = pl.DataFrame({"a": [1, None], "b": ["x", None]})
    synthetic = pl.DataFrame({"a": [None], "b": [None]}, schema={"a": pl.Int64, "b": pl.String})
    assert privacy.count_literal_copies(real, synthetic) == 1


def test_only_common_columns_with_same_type_are_compared():
    real = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    synthetic = pl.DataFrame({"a": [1], "b": [5]})
    assert privacy.count_literal_copies(real, synthetic) == 1
    assert privacy.count_literal_copies(real, pl.DataFrame({"c": [1]})) == 0


def test_disclosure_info():
    assert privacy.disclosure_info(None) == {"gebruikt": False, "plugin": "metasyn-disclosure"}
    info = privacy.disclosure_info(privacy.disclosure_privacy())
    assert info["gebruikt"] is True
    assert info["parameters"]["partition_size"] == 11
