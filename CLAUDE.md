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

## Afwijkingen van de standaard

Bewuste keuzes waar deze repo afwijkt van de CEDA-standaarden, elk met de reden:

- **Geen app** (`app/`, Streamlit): de tool is een stap in de pipeline van een instelling; een app is een No-Go in #1.
- **`argparse` in plaats van typer/rich**: alleen de standaardbibliotheek, dus een kleinere wheelhouse voor een offline installatie in Fabric.
- **Delta als standaarduitvoer** voor een map of lakehouse als doel (zonder `--format`): Fabric-lakehouses werken met Delta-tabellen. Een doel `uit.csv` of `uit.parquet`, of `--format csv`/`parquet`, geeft dat formaat (dataconventies voor lokale uitvoer: #24).
- **hatchling als build-backend**: werkt met de src-layout en `uv build`; `uv build` en `twine check` slagen ermee in de CI.
- **Exacte pins** van metasyn, metasyn-disclosure en faker: de offline wheelhouse moet overal dezelfde versies hebben, en dat hetzelfde model met dezelfde seed dezelfde data geeft, is alleen met dezelfde versies zeker.
- **Geen vaste `data_dictionary.csv`**: het schema is van de instelling, niet van deze repo; het GMF, `toon` en het rapport laten per kolom de soort en het datatype zien.
- **metasyn als engine** (afweging volgens principe §11; bron: het onderzoeksrapport bij #1):
  - Onderhoud: actief; versie 3.1.0 verscheen op 2026-11-20, met onder meer `MultiFrame` (sinds 2.1), data-free en een TOML-specificatie.
  - Eigenaar: ODISSEI Social Data Science (SoDa), met Research Engineering van de Universiteit Utrecht.
  - Licentie: MIT, net als deze repo. De plugin metasyn-disclosure (0.3.3) is ook MIT, maar nog Alpha.
  - Alternatief: SDV neemt verbanden tussen kolommen over (hoger ONS-niveau dan metasyn, dat niveau 3 levert), maar is zwaarder en zit al in edusynth; synthpop (R) neemt standaard echte waarden over in de uitvoer. metasyn is pure Python zonder PyTorch en past daarom in een Fabric-notebook.

## Tech Stack

- Python 3.12 of 3.13 met uv
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
uv run ty check
```

## Data

Invoer: CSV, Parquet of Delta (map of `abfss://`); SPSS (.sav) nog niet (#7). Uitvoer: een GMF-bestand (model) en synthetische data als Delta, Parquet of CSV, met een rapport. Echte data staat nooit in git; alleen `data/*/demo/`.
