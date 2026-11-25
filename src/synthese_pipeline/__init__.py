"""synthese-pipeline — synthetische data maken als stap in een datapipeline."""

from importlib.metadata import version as _version

# Eén bron voor de versie: `version` in pyproject.toml, via de pakketgegevens.
__version__ = _version("synthese-pipeline")
