# synthese-pipeline

## Overview

Commandoregel-tool die synthetische data maakt als stap in de datapipeline van een onderwijsinstelling. Analyses en producten worden gebouwd en getest op de synthetische versie en draaien daarna op de echte data. De synthetische data heeft dezelfde structuur als de echte (kolommen, types, categorieën, lege waarden). Verbanden tussen kolommen worden bewust niet overgenomen.

Engine: [metasyn](https://github.com/sodascience/metasyn). Metasyn modelleert elke kolom los en slaat het model op als leesbaar JSON-bestand (GMF). Alleen dat bestand hoeft de beveiligde omgeving te verlaten.

## Standards

Follow CEDA technical standards: https://github.com/cedanl/.github/blob/main/docs/standards/index.md

## Architectuurprincipes

### Geen app
De primaire interface is de commandoregel (`synthese-pipeline`), aangeroepen vanuit de pipeline van een instelling. Logica leeft in modules onder `src/synthese_pipeline/`; `cli.py` blijft een dun laagje.

### Gelaagd
Laag 1 is metasyn: losse kolommen, structuur behouden, privacy voorop. Synthese met verbanden tussen kolommen (SDV) zit in [edusynth](https://github.com/cedanl/edusynth) en hoort niet in deze repo.

### Code uit edusynth kopiëren, niet importeren
Functies uit edusynth (inlezen, kolomtype-detectie, structuurcheck) worden overgenomen in deze repo. edusynth is geen afhankelijkheid: dat zou SDV en Streamlit meeslepen.

### Geen waarden uit de echte data in rapporten
Rapporten en logbestanden bevatten alleen aantallen, scores en kolomnamen.

## Tech Stack

- Python ≥3.12 met uv
- metasyn, polars
- pytest, ruff, ty

## Project Structure

```
src/synthese_pipeline/
├── __init__.py
├── cli.py            # dun — commandoregel
├── pipeline.py       # run_fit / run_synthesize (ook voor notebooks)
├── storage.py        # inlezen/wegschrijven: CSV, Parquet, Delta, abfss
├── column_types.py   # kolomsoorten afleiden → metasyn var_specs
├── model.py          # GMF fitten, laden, synthetiseren
├── privacy.py        # disclosure control, letterlijke kopieën
├── report.py         # structuurcheck, verdelingsmaten, exitcodes
├── describe.py       # `toon`
├── logs.py           # logging zonder waarden
└── errors.py
tests/synthese_pipeline/   # spiegelt src/synthese_pipeline/
data/
├── 01-raw/demo/      # alleen demo-data in git
├── 02-prepared/demo/
└── 03-output/demo/
```

## How to Run

```bash
uv sync --group dev
uv run synthese-pipeline --version
uv run pytest
uv run ruff check src tests
uv run ruff format src tests
```

## Data

Invoer: CSV, Parquet of Delta (map of `abfss://`); SPSS (.sav) nog niet (#7). Uitvoer: een GMF-bestand (model) en synthetische data als Delta, Parquet of CSV, met een rapport. Echte data staat nooit in git; alleen `data/*/demo/`.
