"""Tests voor de commandoregel."""

import pytest

from synthese_pipeline import __version__
from synthese_pipeline.cli import main


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out
