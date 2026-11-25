"""Logging per run: naar stderr en optioneel naar een logbestand.

Logregels noemen alleen tabelnamen, kolomnamen en aantallen, nooit waarden uit de data.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

LOGGER_NAME = "synthese_pipeline"
_FORMAT = "%(asctime)s %(levelname)s %(message)s"


@contextmanager
def run_logging(log_file: str | Path | None = None) -> Generator[logging.Logger]:
    """Zet logging aan voor de duur van één run en ruim de handlers daarna op."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    formatter = logging.Formatter(_FORMAT)
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    try:
        yield logger
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
