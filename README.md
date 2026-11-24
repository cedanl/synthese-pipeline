# synthese-pipeline

Synthetische data maken als stap in een datapipeline. De synthetische data heeft dezelfde structuur als de echte data: dezelfde kolommen, types, categorieën en lege waarden. Verbanden tussen kolommen worden niet overgenomen. Zo kun je analyses en producten bouwen en testen voordat je de echte data gebruikt.

Gebouwd op [metasyn](https://github.com/sodascience/metasyn).

Status: in ontwikkeling.

## Installatie

Vereist [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/cedanl/synthese-pipeline
cd synthese-pipeline
uv sync
```

## Gebruik

```bash
uv run synthese-pipeline --version
```

## Ontwikkeling

```bash
uv sync --group dev
uv run pytest
uv run ruff check src tests
```

Een devcontainer staat in `.devcontainer/`.

## Licentie

MIT
