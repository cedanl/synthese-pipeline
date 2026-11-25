"""Foutmeldingen van synthese-pipeline."""

from __future__ import annotations


class SyntheseError(Exception):
    """Fout die de gebruiker kan oplossen.

    De melding bevat alleen tabelnamen, kolomnamen, paden en aantallen, nooit waarden uit de
    data. De commandoregel toont de melding daarom letterlijk.
    """
