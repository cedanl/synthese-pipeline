"""``toon``: het model in gewone taal, per tabel en per kolom.

Dit leest alleen het GMF-bestand. Wat hier staat, staat ook in het model zelf: het is een
leeshulp voor wie het model nakijkt voordat het de beveiligde omgeving verlaat.
"""

from __future__ import annotations

from typing import Any

from synthese_pipeline.model import column_kinds, tool_info

MAX_LABELS = 10

_KIND_LABELS = {
    "categorical": "categorie",
    "discrete": "geheel getal",
    "continuous": "kommagetal",
    "date": "datum/tijd",
    "key": "unieke sleutel",
    "text": "tekst",
}


def _num(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value)}"
    return f"{value:.4g}".replace(".", ",")


def _percent(share: float) -> str:
    return f"{share * 100:.1f}%".replace(".", ",")


def describe_distribution(dist: dict[str, Any]) -> str:
    """Eén zin over een metasyn-verdeling."""
    name = dist.get("name", "")
    params = dist.get("parameters") or {}
    short = name.split(".")[-1]
    if short == "multinoulli":
        labels = list(params.get("labels") or [])
        probs = list(params.get("probs") or [])
        parts = [f"{label} ({_percent(prob)})" for label, prob in zip(labels, probs, strict=False)]
        shown = ", ".join(parts[:MAX_LABELS])
        more = f", en nog {len(parts) - MAX_LABELS}" if len(parts) > MAX_LABELS else ""
        return f"keuze uit {len(labels)} waarden met vaste kansen: {shown}{more}"
    if short == "uniform":
        lower, upper = _num(params.get("lower")), _num(params.get("upper"))
        return f"gelijkmatig verdeeld tussen {lower} en {upper}"
    if short == "normal":
        return (
            f"normaal verdeeld rond {_num(params.get('mean'))} "
            f"(standaardafwijking {_num(params.get('sd'))})"
        )
    if short == "truncated_normal":
        return (
            f"normaal verdeeld rond {_num(params.get('mean'))} "
            f"(standaardafwijking {_num(params.get('sd'))}), afgekapt tussen "
            f"{_num(params.get('lower'))} en {_num(params.get('upper'))}"
        )
    if short == "lognormal":
        return "scheef verdeeld (lognormaal)"
    if short == "exponential":
        return f"exponentieel verdeeld (rate {_num(params.get('rate'))})"
    if short == "poisson":
        return f"Poisson-verdeeld met gemiddelde {_num(params.get('rate'))}"
    if short == "geometric":
        return "geometrisch verdeeld"
    if short == "unique_key":
        return "unieke sleutel, oplopend genummerd"
    if short in ("regex", "unique_regex"):
        return "tekst volgens een patroon (reguliere expressie)"
    if short == "faker":
        return f"verzonnen waarden (Faker, soort '{params.get('faker_type')}')"
    if short == "freetext":
        return "verzonnen vrije tekst"
    if short == "constant":
        return f"steeds dezelfde waarde: {_num(params.get('value'))}"
    if short == "na":
        return "alleen lege waarden"
    return f"verdeling '{name}'"


def describe_model(gmf: dict[str, Any]) -> str:
    """Tekst voor ``synthese-pipeline toon``."""
    info = tool_info(gmf)
    kinds = column_kinds(gmf)
    disclosure = info.get("disclosure_control", {})
    if disclosure.get("gebruikt") is True:
        disclosure_text = f"aan ({disclosure.get('plugin')} {disclosure.get('versie')})"
    elif disclosure.get("gebruikt") is False:
        disclosure_text = "uit"
    else:
        disclosure_text = "onbekend (model niet gemaakt met synthese-pipeline)"

    lines = [f"Model met {len(gmf['tables'])} tabel(len). Disclosure control: {disclosure_text}."]
    for table in gmf["tables"]:
        name = table["name"]
        lines.append("")
        lines.append(f"Tabel '{name}': {table['n_rows']} rijen, {table['n_columns']} kolommen")
        for var in table["vars"]:
            kind = kinds.get(name, {}).get(var["name"])
            kind_text = _KIND_LABELS.get(kind or "", var.get("dtype", "?"))
            lines.append(
                f"- {var['name']} ({kind_text}): {describe_distribution(var['distribution'])}; "
                f"lege waarden: {_percent(var.get('prop_missing', 0.0))}"
            )
    return "\n".join(lines)
