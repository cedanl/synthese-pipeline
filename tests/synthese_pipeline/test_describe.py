"""Tests voor ``toon``: het model in gewone taal."""

from synthese_pipeline import describe, model
from tests.synthese_pipeline.dummy import students


def test_describe_model_lists_every_column_in_dutch():
    gmf = model.fit_tables({"studenten": students()})
    text = describe.describe_model(gmf)
    assert "Disclosure control: aan (metasyn-disclosure 0.3.3)" in text
    assert "Tabel 'studenten': 200 rijen, 8 kolommen" in text
    for column in students().columns:
        assert f"- {column} (" in text
    assert "- geslacht (categorie): keuze uit 2 waarden" in text
    assert "- student_id (unieke sleutel)" in text
    assert "lege waarden: " in text


def test_describe_distributions():
    assert (
        describe.describe_distribution(
            {"name": "core.normal", "parameters": {"mean": 6.5, "sd": 1.25}}
        )
        == "normaal verdeeld rond 6,5 (standaardafwijking 1,25)"
    )
    assert (
        describe.describe_distribution(
            {"name": "core.uniform", "parameters": {"lower": "2020-01-01", "upper": "2020-12-31"}}
        )
        == "gelijkmatig verdeeld tussen 2020-01-01 en 2020-12-31"
    )
    many = {"labels": list(range(15)), "probs": [1 / 15] * 15}
    text = describe.describe_distribution({"name": "core.multinoulli", "parameters": many})
    assert text.startswith("keuze uit 15 waarden") and text.endswith("en nog 5")
    assert describe.describe_distribution({"name": "x.onbekend"}) == "verdeling 'x.onbekend'"
