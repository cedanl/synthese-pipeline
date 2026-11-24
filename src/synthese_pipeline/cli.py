"""Commandoregel voor synthese-pipeline."""

from __future__ import annotations

import argparse

from synthese_pipeline import __version__


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="synthese-pipeline",
        description="Synthetische data maken als stap in een datapipeline.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    _parse_args(argv)
