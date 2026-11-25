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

## Data

| | Wat | Waar |
|---|---|---|
| Invoer | Eén tabel of een map met tabellen: CSV, Parquet of Delta, ook in OneLake (`abfss://`). SPSS (`.sav`) wordt nog niet gelezen ([#7](https://github.com/cedanl/synthese-pipeline/issues/7)). | `data/01-raw/` (zoals geleverd) of `data/02-prepared/` (voorbewerkt) |
| Model | Een GMF-bestand (JSON) van metasyn, met per kolom een verdeling | gemaakt in de beveiligde omgeving; alleen dit bestand gaat naar buiten |
| Uitvoer | Synthetische data, plus een rapport (JSON). Lakehouse: per tabel een Delta-tabel (de standaard). Lokale map: Parquet en CSV met `--format parquet,csv`. CSV volgt de dataconventies: `;` als scheidingsteken, UTF-8, tekst tussen aanhalingstekens (zie [Bron en doel](#bron-en-doel)). | `data/03-output/` of een aparte lakehouse |

**`fit` is de privacygrens.** De tool leest de data in `01-raw` of `02-prepared` zoals de instelling die aanlevert, in de beveiligde omgeving. Alleen het GMF-bestand verlaat die omgeving. `synthesize` en `toon` gebruiken alleen het model, nooit de echte data (behalve `synthesize --source`, en dat draait dan weer binnen). Lees het model na met `toon` voordat het naar buiten gaat (zie [Privacy](#privacy)).

Echte data staat nooit in git: `.gitignore` laat in `data/` alleen de mappen `demo/` toe. De tests gebruiken alleen gegenereerde dummydata.

## Gebruik

De tool werkt in twee stappen. In de beveiligde omgeving maakt `fit` een model (een GMF-bestand) uit de echte data. Alleen dat bestand gaat naar buiten. Daarna maakt `synthesize` synthetische data uit alleen het model.

```bash
uv run synthese-pipeline fit data.csv model.json
uv run synthese-pipeline toon model.json
uv run synthese-pipeline synthesize model.json output.csv --rows 1000 --seed 42
```

| Opdracht | Wat het doet |
|---|---|
| `fit BRON MODEL` | Leest de bron en schrijft alleen het GMF-bestand. Disclosure control staat aan. Met `--spec SPEC` gelden de keuzes uit een [specificatie](#specificatie). |
| `synthesize MODEL DOEL` | Maakt synthetische data uit alleen het model en schrijft een rapport (JSON) naast de data. |
| `synthesize --spec SPEC DOEL` | Maakt dummydata uit alleen een specificatie, zonder model en zonder bron ([stap 0](#stap-0-dummydata-zonder-data)). |
| `init-spec BRON SPEC` | Schrijft een ontwerp-specificatie (TOML) uit de bron, zonder waarden uit de data. |
| `toon MODEL` | Laat per tabel en kolom in gewone taal zien wat er in het model staat: verdeling, aandeel lege waarden en omschrijving. |

`uv run synthese-pipeline <opdracht> --help` toont alle opties.

### Bron en doel

| Bron (`fit`, `--source`) | Wat wordt gelezen |
|---|---|
| `data.csv`, `data.parquet` | Eén tabel; de naam is de bestandsnaam zonder extensie |
| een map met `_delta_log` | Eén Delta-tabel; de naam is de mapnaam |
| een map | Alle tabellen erin: Delta-mappen, `*.parquet` en `*.csv` |
| `abfss://…/Tables/<tabel>` | Eén Delta-tabel in OneLake (Microsoft Fabric) |

| Doel (`synthesize`) | Wat wordt geschreven |
|---|---|
| `uit.csv`, `uit.parquet` | Eén bestand (alleen bij een model met één tabel); rapport: `uit.rapport.json` |
| een map | Per tabel een Delta-tabel `<map>/<tabel>` (standaard); met `--format parquet` of `--format csv` een bestand per tabel, met `--format parquet,csv` allebei (`<tabel>.parquet` en `<tabel>.csv`); rapport: `<map>/rapport.json` |
| `abfss://…/Tables` | Per tabel een Delta-tabel (alleen Delta); geef het rapport op met `--report` |

| Waar komt de data | Aanbevolen uitvoer |
|---|---|
| Een lakehouse (Microsoft Fabric/OneLake, `abfss://`) | Delta: de standaard, zonder `--format` |
| Een lokale map | Parquet en CSV in één keer: `--format parquet,csv` (zonder `--format` wordt het Delta) |

CSV-uitvoer volgt de [dataconventies](https://github.com/cedanl/.github/tree/main/standards/README.md): UTF-8 zonder BOM, `;` als scheidingsteken en tekst tussen aanhalingstekens (`"Zoë; ""de"" Groot"`); getallen, datums, waar/onwaar en lege waarden staan er zonder. Getallen hebben een decimale punt. Wil je een komma als scheidingsteken, geef dan `--csv-separator ","`; andere tekens zijn niet toegestaan. `--format parquet,csv` kan alleen bij een lokale map, niet bij `uit.csv` of `abfss://`.

**Let op, gewijzigd:** CSV-uitvoer had eerst een komma als scheidingsteken en geen aanhalingstekens. Lees je de uitvoer met polars, geef dan `separator=";"` mee (`pl.read_csv(pad, separator=";")`), of schrijf met `--csv-separator ","`. Deze tool leidt het scheidingsteken zelf af.

Alles wordt gelezen met polars, niet met pandas. Een CSV-bestand wordt eerst als tekst gelezen; daarna krijgt elke kolom het type waar alle ingevulde waarden in passen (geheel getal, kommagetal, datum, waar/onwaar of tekst). Codes met voorloopnullen, zoals `01234`, blijven tekst. Het scheidingsteken (komma, puntkomma of tab) wordt uit de kopregel afgeleid; bij een puntkomma mag een getal een decimale komma hebben. De synthetische tabellen krijgen dezelfde kolomnamen en datatypes als de bron. Bestaande uitvoer wordt overschreven, maar nooit iets buiten het doel: een tabelnaam uit het model wordt een map- of bestandsnaam, dus `synthesize` en `toon` weigeren een model waarin een tabelnaam een pad is (zoals `../bron` of `/data/bron`).

Voor `abfss://`-paden geeft de tool `use_fabric_endpoint` en een bearer token mee aan delta-rs. Het token komt uit de omgevingsvariabele `SYNTHESE_PIPELINE_TOKEN`, in een Fabric-notebook bijvoorbeeld `notebookutils.credentials.getToken("storage")`. Vanuit Python kan het ook direct: `run_fit(bron, model, storage_options=fabric_storage_options(token))` uit `synthese_pipeline.pipeline` en `synthese_pipeline.storage`. Dit is nog niet getest in een echte Fabric-omgeving.

### Kolomtypes

Elke kolom krijgt een soort. De tool leidt die af uit het datatype en de waarden; de eerste regel die past wint:

| Soort | Wanneer | Wat metasyn ervan maakt |
|---|---|---|
| `categorical` | waar/onwaar; een geheel getal met hooguit 12 unieke waarden of tekst met hooguit 30, en hooguit half zoveel unieke als ingevulde waarden (bijv. Likert 1-5, M/V) | een keuze uit de categorieën met hun kansen |
| `freetext` | tekst die op vrije tekst lijkt: meer dan de helft van de ingevulde waarden heeft een spatie, of de waarden zijn gemiddeld minstens 20 tekens lang, en het is geen code (zie hieronder) | niets: de kolom gaat niet in het model en ontbreekt in de synthetische data |
| `key` | geheel getal of tekst, geen lege waarden, alle waarden uniek, minstens 20 rijen | unieke waarden vanaf 1 (zie [Privacy](#privacy)); bij een code een uniek patroon |
| `discrete` | overige gehele getallen, ook kommagetallen die allemaal op ,0 eindigen | een verdeling van gehele getallen |
| `continuous` | overige kommagetallen | een continue verdeling |
| `date` | datum, datum met tijd, tijd | een verdeling van datums |
| `text` | overige tekst | bij een code een patroon, anders kiest metasyn zelf |
| `faker` | alleen als je het zelf kiest met `--faker` | verzonnen waarden van [Faker](https://faker.readthedocs.io/) (`nl_NL`) |

Gehele getallen met lege waarden blijven gehele getallen. Wijkt een kolom af, geef dan zelf de soort op bij `fit`, eventueel voor één tabel:

```bash
uv run synthese-pipeline fit bron/ model.json --type opleidingscode=categorical --type studenten.leeftijd=discrete
```

Een **code** heeft een vaste vorm: tel elk cijfer als `9`, elke hoofdletter als `A` en elke kleine letter als `a`. Minstens 95% van de waarden heeft dan een van hooguit drie vormen, met een cijfer en hooguit 12 tekens, zoals postcodes (`1234 AB`, vorm `9999 AA`) of opleidingscodes (`34567`). Een code is nooit `freetext`. Het patroon in het model wordt gemaakt uit alleen die vormen (`[0-9]{4} [A-Z]{2}`), niet uit de waarden, en alleen vormen die minstens 10 keer voorkomen tellen mee. Zo blijft de vorm behouden, ook met disclosure control. `B Informatica` is geen code: die heeft geen cijfer en geen vaste vorm.

### Vrije tekst

Vrije tekst, zoals opmerkingen of "voornaam achternaam", kan letterlijk namen en andere gegevens bevatten. Daarom gaat een kolom van soort `freetext` **standaard niet in het model**. Het model noemt de kolom alleen met naam en brontype (`synthese_pipeline.tabellen.<tabel>.uitgesloten`); `toon` laat hem zien als "uitgesloten (vrije tekst)". De synthetische data heeft die kolom niet. Het log en het rapport (`tabellen.<tabel>.uitgesloten`) noemen de uitgesloten kolommen, alleen met hun naam. De structuurcontrole telt ze niet mee.

Wil je de kolom toch, dan zijn er twee keuzes:

| Keuze | Wat er gebeurt |
|---|---|
| `--type opmerking=text` | De kolom gaat als gewone tekst in het model. Kijk het model dan na met `toon`. |
| `--faker opmerking=sentence` | De kolom krijgt verzonnen waarden van Faker (`nl_NL`), zonder iets uit de echte data. Ook `tabel.kolom=provider`; mag vaker. |

Een provider is een functie van Faker die tekst maakt, bijvoorbeeld `name`, `first_name`, `last_name`, `email`, `city`, `street_address`, `sentence` of `text` (zie de [lijst van providers](https://faker.readthedocs.io/en/master/locales/nl_NL.html)). Een onbekende provider of kolom geeft een fout. Faker verzint de waarden, maar een verzonnen naam kan bij toeval gelijk zijn aan een echte naam. Met `--type kolom=freetext` sluit je zelf een kolom uit.

```bash
uv run synthese-pipeline fit bron.csv model.json --faker naam=name --faker opmerking=sentence
```

### Specificatie

Een specificatie is een TOML-bestand in het [configuratieformaat van metasyn](https://metasyn.readthedocs.io/en/latest/improve_synth.html). Daarin legt de instelling per kolom vast wat ze weet: de soort, een omschrijving, of de kolom uniek is of eruit moet, en eventueel de verdeling. `init-spec` maakt een ontwerp uit de bron:

```bash
uv run synthese-pipeline init-spec data.csv spec.toml
uv run synthese-pipeline fit data.csv model.json --spec spec.toml
```

Het ontwerp bevat per kolom de afgeleide soort, het brontype en een lege `description`. Een kandidaat-sleutel krijgt `distribution = {unique = true}`, vrije tekst `uitgesloten = true`. Er staan geen waarden uit de data in, alleen namen, soorten, types en het aantal rijen. Met `--with-categories` komen de categorieën van categorische kolommen erbij. Dat zijn wel waarden uit de data: het log meldt per kolom hoeveel categorieën minder dan 10 keer voorkomen, en je kijkt het bestand na voordat het de beveiligde omgeving verlaat.

Eén tabel geeft `config_version = "1.2"` met `[[var]]` per kolom (1.0 en 1.1 worden ook gelezen). Meer tabellen geven `config_version = "2.0"` met `[[table]]` per tabel en `[[table.var]]` per kolom. Relaties tussen tabellen staan dan bovenaan (zie [Relaties tussen tabellen](#relaties-tussen-tabellen)).

| Sleutel per kolom | Betekenis | Bij `fit --spec` | Bij data-free |
|---|---|---|---|
| `name` | kolomnaam (verplicht) | moet in de bron staan | wordt een kolom |
| `soort` | een [soort](#kolomtypes) | gaat voor de afgeleide soort | bepaalt het type |
| `description` | omschrijving | komt in het model (`toon`) | komt in het model |
| `uitgesloten = true` | kolom gaat niet in het model, net als vrije tekst | ja | ja |
| `distribution` | verdeling van metasyn: `{unique = true}`, of `{implements = "core.uniform", parameters = {lower = 1, upper = 10}}` | met `implements` en `parameters`: die vaste verdeling; met alleen `implements`: metasyn fit die verdeling | nodig, behalve bij `key`, `categorieen` en `faker` |
| `prop_missing` | aandeel lege waarden (0 tot 1) | gaat voor de data | standaard 0 |
| `categorieen` | lijst met categorieën | genegeerd (de data bepaalt ze) | gelijke kansen |
| `faker` | Faker-provider, bijv. `"name"` (soort `faker`) | ja | ja |
| `brontype` | datatype, bijv. `Int64`, `String`, `Date` | genegeerd (de bron bepaalt het) | type van de kolom in de uitvoer |

Bovenaan mogen `n_rows` (aantal rijen in het model) en bij 1.x `name` (tabelnaam bij data-free) staan. Een onbekende sleutel, versie of soort geeft een fout met exitcode 1, zonder waarden uit het bestand.

**`--type` en `--faker` gaan voor de specificatie.** Noemt de commandoregel een kolom, dan gelden voor die kolom de soort en verdeling van de commandoregel; `description` en `prop_missing` uit de specificatie blijven. Het log meldt per kolom dat de commandoregel voorging.

Zet in `description` nooit echte waarden (namen, nummers, voorbeelden uit de data): de omschrijving gaat mee in het model en in de specificatie.

### Relaties tussen tabellen

Een kolom die naar de sleutel van een andere tabel verwijst (bijv. `student_id` in `inschrijving`), geef je op als relatie. Relaties staan bovenaan een specificatie met `config_version = "2.0"`, als lijst met teksten, vóór de eerste `[[table]]`:

```toml
config_version = "2.0"
relations = [
  "inschrijving[student_id] SUBSET OF student[student_id]",
  "resultaat[inschrijving_id] INFER FROM inschrijving[inschrijving_id]",
]

[[table]]
name = "student"
[[table.var]]
name = "student_id"
# ... en de andere tabellen met hun kolommen
```

| Vorm | Betekenis |
|---|---|
| `tabel_a[kolom] SUBSET OF tabel_b[kolom]` | elke ingevulde waarde in `tabel_a[kolom]` staat in `tabel_b[kolom]` |
| `tabel_a[kolom] INFER FROM tabel_b[kolom]` | metasyn leidt het soort relatie bij `fit` af uit de data (`SUBSET OF`, of `EQUALS` / `EQUAL ORDERED` bij een één-op-één-relatie); zonder data geldt `SUBSET OF` |

- Beide tabellen en kolommen moeten in de specificatie staan. De kolom waarnaar verwezen wordt (`tabel_b[kolom]`) is de sleutel: hij krijgt soort `key` en blijft uniek. Een andere soort geeft een fout.
- Een kolom kan maar naar één sleutel verwijzen, en kan niet tegelijk verwijzen en sleutel zijn. Drie niveaus (student, inschrijving, resultaat) gaan wel, met verschillende kolommen.
- Bij `fit` komen de relaties in `relations` van het model. Synthetiseren gaat dan met [`MultiFrame` van metasyn](https://metasyn.readthedocs.io/en/latest/multiframe.html): de verwijzende kolom krijgt waarden uit de synthetische sleutel, dus er zijn geen wezen (verwijzingen zonder sleutel). Dezelfde seed geeft dezelfde data.
- De verwijzende kolom krijgt in het model de verdeling van de sleutel. Er staat dus niets uit de verwijzende kolom zelf in het model. Met `INFER FROM` en verwijzende waarden zonder sleutel in de bron geeft `fit` een fout met alleen het aantal; gebruik dan `SUBSET OF`.
- Zonder data (data-free) heeft de verwijzende kolom geen eigen verdeling nodig.
- Een relatie in een specificatie met één tabel (1.x), een ongeldige tekst, of een onbekende tabel of kolom geeft een fout (exitcode 1) die de relatie noemt, zonder waarden uit de data.

**Beperking.** Het aantal rijen per sleutel (bijv. inschrijvingen per student) volgt niet de echte verdeling. metasyn trekt de verwijzende sleutels gelijkmatig met teruglegging. Op dummydata gemeten: max 9 inschrijvingen per student tegen echt 5. Lege verwijzingen in de bron komen niet terug. Het rapport noemt dit onder `beperkingen` zodra er relaties zijn.

### Stap 0: dummydata zonder data

Een instelling kan beginnen zonder dat er echte data of een model de beveiligde omgeving verlaat. Ze schrijft een specificatie met per kolom een verdeling, en maakt daar dummydata mee ([data-free in metasyn](https://metasyn.readthedocs.io/en/latest/datafree.html)):

```bash
uv run synthese-pipeline synthesize --spec examples/spec.toml --rows 100 --format csv uit --seed 1
```

[`examples/spec.toml`](examples/spec.toml) heeft twee tabellen. `studenten` heeft een sleutel, een Likert-item, een code, een datum, een kommagetal en een uitgesloten kolom met vrije tekst. `resultaten` verwijst met `studentnummer` naar `studenten` (een relatie), met een vak en een cijfer. Met twee tabellen is het doel een map (`uit/studenten.csv`, `uit/resultaten.csv`, `uit/rapport.json`). Zonder `--rows` geldt `n_rows` uit de specificatie. Het rapport werkt zoals altijd: de structuur wordt vergeleken met de specificatie, met dezelfde exitcodes, en `data_free` staat op `true`. Dezelfde specificatie met dezelfde seed geeft dezelfde data. Geef `MODEL` en `--spec` niet samen op: dat is een gebruiksfout (exitcode 2).

Zonder data is per kolom een verdeling nodig. Een `key` zonder verdeling wordt een oplopend nummer vanaf 1, een kolom met `categorieen` een keuze met gelijke kansen, en een kolom met `faker` krijgt verzonnen waarden. Een ontwerp van `init-spec` heeft die verdelingen nog niet: vul ze aan. Dummydata uit een specificatie is **niet statistisch geldig**. Ze heeft de juiste kolommen en types, maar de verdelingen zijn wat je opschrijft, niet wat er in de echte data zit. Gebruik ze om code te bouwen en te testen, niet voor uitkomsten.

### Privacy

- **Disclosure control staat standaard aan.** `fit` gebruikt de plugin [metasyn-disclosure](https://github.com/sodascience/metasyn-disclosure-control) (versie 0.3.3). Zeldzame waarden en uitschieters, zoals een leeftijd die maar bij één persoon voorkomt, komen dan niet letterlijk in het model. Het model (`synthese_pipeline.disclosure_control` en `provenance`) en het rapport vermelden of de plugin aan stond. Uitzetten kan met `--no-disclosure`; dat is afgeraden.
- **Logs en rapporten bevatten geen waarden uit de data**, alleen tabelnamen, kolomnamen, datatypes, aantallen en scores. Met `--log PAD` schrijven `fit` en `synthesize` dezelfde regels ook naar een bestand. Een onverwachte fout toont alleen het soort fout, omdat de melding waarden kan bevatten.
- **Vrije tekst gaat standaard niet in het model** (zie [Vrije tekst](#vrije-tekst)).
- **Het model bevat geen sleutelbereik.** metasyn bewaart bij een sleutel van gehele getallen (soort `key`, verdeling `core.unique_key`) de kleinste waarde als `lower`. `fit` zet die altijd op de vaste waarde `lower = 1`, ook met `--no-disclosure` en ook bij een kolom die naar de sleutel verwijst. Het GMF verraadt dus geen sleutelbereik en geen echte sleutelwaarde. Synthetische sleutels blijven uniek, maar lijken niet meer op de echte: een studentnummer van 8 cijfers wordt een klein getal vanaf 1 (bij niet-opeenvolgende sleutels tot ongeveer 2× het aantal rijen). Een analyse die op de lengte of het bereik van een sleutel steunt, geeft op de synthetische data dus iets anders. Een sleutel met eigen `parameters` in een [specificatie](#specificatie) houdt die waarden.
- **Een ontwerp-specificatie bevat geen waarden uit de data**, behalve met `--with-categories`. Omschrijvingen schrijft de instelling zelf: zet daar geen echte waarden in.
- Het model zelf bevat wel samenvattingen van de data, zoals categorieën en grenzen. Lees het na met `toon` voordat het de beveiligde omgeving verlaat.

### Rapport

`synthesize` schrijft een rapport (JSON) met:

- de versie van synthese-pipeline en metasyn, de seed, en of disclosure control aan stond;
- per tabel: of de structuur gelijk is (kolommen, volgorde, datatypes) met de afwijkingen, en welke kolommen zijn uitgesloten (vrije tekst);
- per relatie (`relaties`): de relatie als tekst (tabel- en kolomnamen), het aantal gecontroleerde rijen, lege verwijzingen, **wezen** (verwijzende waarden die niet in de sleutel staan) en dubbele sleutels. Zonder relaties is de lijst leeg. Met relaties staat de beperking hierboven onder `beperkingen`;
- met `--source BRON`: per kolom een verdelingsmaat (TV-afstand voor categorieën, Wasserstein-afstand voor getallen, in dagen voor datums), het aandeel lege waarden echt en synthetisch, nieuwe categorieën, en het aantal **letterlijke kopieën**: synthetische rijen die gelijk zijn aan een rij die in de echte data maar één keer voorkomt.

Zonder `--source` vergelijkt het rapport de structuur met het model en staan de kopieën op "niet gecontroleerd"; de exitcode verandert daar niet door. Zonder `--seed` kiest de tool een seed en zet die in het rapport. Hetzelfde model met dezelfde seed geeft dezelfde data.

### Exitcodes

| Code | Betekenis |
|---|---|
| 0 | geslaagd |
| 1 | fout (invoer, model of omgeving) |
| 2 | verkeerd gebruik van de commandoregel |
| 3 | structuur wijkt af van de bron of het model |
| 4 | letterlijke kopieën van echte rijen gevonden (gaat voor 3 en 5) |
| 5 | relaties tussen tabellen kloppen niet: wezen of dubbele sleutels (3 gaat voor) |

### Voorbeeld van de uitvoer

Op dummydata, zonder echte data: eerst dummydata uit [`examples/spec.toml`](examples/spec.toml), daarna een model uit die dummydata en synthetische data met een vergelijking.

```bash
uv run synthese-pipeline synthesize --spec examples/spec.toml --rows 100 --format csv uit --seed 1
uv run synthese-pipeline fit uit/studenten.csv model.json
uv run synthese-pipeline synthesize model.json syn.csv --source uit/studenten.csv --seed 1
```

Uitvoer van de laatste opdracht:

```text
2026-11-25 12:47:38,762 INFO bron gelezen voor de vergelijking: 1 tabel(len)
2026-11-25 12:47:38,762 INFO tabel 'studenten': 100 rijen, 5 kolommen
2026-11-25 12:47:38,820 INFO tabel 'studenten': 100 rijen gemaakt
2026-11-25 12:47:38,822 INFO synthetische data geschreven: 1 tabel(len)
2026-11-25 12:47:38,832 INFO rapport geschreven; structuur gelijk: ja; letterlijke kopieën: 0; exitcode 0
rapport: syn.rapport.json (geslaagd)
```

Een deel van `syn.rapport.json` (`…` is weggelaten). Er staan alleen namen, aantallen en scores in:

```json
{
  "synthese_pipeline": "0.1.0",
  "metasyn": "3.1.0",
  "seed": 1,
  "disclosure_control": {
    "gebruikt": true,
    "plugin": "metasyn-disclosure",
    "versie": "0.3.3",
    …
  },
  "bron_vergeleken": true,
  "structuur": {
    "gelijk": true,
    "afwijkingen": []
  },
  "letterlijke_kopieen": 0,
  "tabellen": {
    "studenten": {
      "rijen": 100,
      "kolommen": 5,
      "verdelingen": {
        "tevredenheid": {
          "soort": "categorical",
          "aandeel_leeg_echt": 0.1,
          "aandeel_leeg_synthetisch": 0.1,
          "maat": "tv_afstand",
          "waarde": 0.0222222222222222
        },
        "inschrijfdatum": {
          "soort": "date",
          "aandeel_leeg_echt": 0.0,
          "aandeel_leeg_synthetisch": 0.0,
          "maat": "wasserstein_dagen",
          "waarde": 9.02
        },
        …
      },
      "letterlijke_kopieen": 0
    }
  },
  "exitcode": 0,
  "uitkomst": "geslaagd"
}
```

## Ontwikkeling

```bash
uv sync --group dev
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run ty check
```

Tests gebruiken alleen gegenereerde dummydata (`tests/synthese_pipeline/dummy.py`).

Een devcontainer staat in `.devcontainer/`.

## Release maken

Het pakket `synthese-pipeline` gaat naar [PyPI](https://pypi.org/p/synthese-pipeline) via Trusted Publishing: GitHub Actions logt in met een kortlevend OIDC-token, er staan geen API-tokens in de repo. Een versie op PyPI is definitief: dezelfde versie kan niet opnieuw worden geüpload.

1. **Versie ophogen.** Op een branch, via een pull request naar `main`:

   ```bash
   uv version --bump patch   # of minor / major
   ```

   Dit past `version` in `pyproject.toml` en `uv.lock` samen aan. Pas je alleen `pyproject.toml` aan, dan faalt `uv sync --locked` in de CI.
2. **Taggen.** Na de merge, op de bijgewerkte `main`:

   ```bash
   git switch main && git pull
   git tag v0.2.0            # v + precies de versie uit pyproject.toml
   git push origin v0.2.0
   ```

3. **De workflow** [`pypi-publish.yml`](.github/workflows/pypi-publish.yml) start alleen op een tag `vX.Y.Z` en doet:
   - controleren dat de tag gelijk is aan `project.version` (tag `v0.2.0` hoort bij versie `0.2.0`); anders stopt hij vóór het publiceren;
   - testen: ruff, ruff format, ty en pytest, zoals de CI;
   - bouwen met `uv build` en controleren met `twine check --strict`;
   - publiceren naar PyPI vanuit de GitHub-omgeving `pypi`, met precies de bestanden die in de bouwstap zijn gemaakt.
4. **Controleren** in een schone omgeving:

   ```bash
   uvx --from synthese-pipeline==0.2.0 synthese-pipeline --version
   # of: python -m venv /tmp/check && /tmp/check/bin/pip install synthese-pipeline==0.2.0
   ```

5. **Release-notes.** Maak na een geslaagde publicatie een GitHub Release op dezelfde tag, met notes in het Nederlands:

   ```bash
   gh release create v0.2.0 --title "v0.2.0" --generate-notes
   ```

   Herschrijf de gegenereerde lijst daarna (`gh release edit v0.2.0 --notes-file notes.md`) tot Nederlandse notes voor gebruikers:
   - een korte inleiding: wat deze versie oplevert;
   - kopjes **Nieuw**, **Verbeterd**, **Opgelost** en, als het nodig is, **Let op** (wijzigingen die bestaande pipelines raken, zoals een andere optie of uitvoer);
   - per punt één zin over wat de gebruiker merkt, met het issue- of PR-nummer (`#22`);
   - geen echte data, ook geen voorbeeldwaarden uit echte bestanden.

### Preview op TestPyPI

Elke push naar `main` (en een handmatige start via *Actions → Preview on TestPyPI → Run workflow*) draait [`preview.yml`](.github/workflows/preview.yml). Die bouwt een previewversie `X.Y.Z.devN`, waarbij `N` het volgnummer van de workflowrun is. De versie wordt alleen in de runner aangepast, niet gecommit. De build gaat naar [TestPyPI](https://test.pypi.org/p/synthese-pipeline) vanuit de GitHub-omgeving `testpypi`; bestanden die er al staan worden overgeslagen.

Een preview proberen: haal alleen het pakket zelf van TestPyPI en de afhankelijkheden van PyPI (op TestPyPI kan iedereen pakketten met dezelfde naam zetten):

```bash
uvx pip download --no-deps --index-url https://test.pypi.org/simple/ \
  "synthese-pipeline==0.1.0.dev12" -d /tmp/preview
uvx --from /tmp/preview/synthese_pipeline-0.1.0.dev12-py3-none-any.whl synthese-pipeline --version
```

Het volgnummer staat in de log van de stap *Set preview version* en in de lijst met versies op TestPyPI.

## Achtergrond en verwijzingen

- [metasyn](https://github.com/sodascience/metasyn) (MIT) van ODISSEI Social Data Science (SoDa), met Research Engineering van de Universiteit Utrecht: modelleert elke kolom los en slaat het model op als GMF. Documentatie: [metasyn.readthedocs.io](https://metasyn.readthedocs.io/).
- [metasyn-disclosure](https://github.com/sodascience/metasyn-disclosure-control) (MIT): de plugin voor disclosure control, volgens de vuistregels van het CBS.
- **ONS-niveau 3.** Op de schaal van het Britse ONS (1 tot 6) levert metasyn niveau 3: plausibele synthetische data met de juiste structuur en verdelingen per kolom, zonder verbanden tussen kolommen ([metasyn FAQ](https://metasyn.readthedocs.io/en/latest/faq.html)). Voor analyses met verbanden is deze data niet bedoeld.
- Het onderzoek en de keuzes achter deze tool staan in [#1](https://github.com/cedanl/synthese-pipeline/issues/1).
- Synthese met verbanden tussen kolommen (SDV) zit in [edusynth](https://github.com/cedanl/edusynth).
- Deze repo volgt de [CEDA-standaarden](https://github.com/cedanl/.github/blob/main/docs/standards/index.md).

## Contact en bijdragen

- Een vraag, fout of idee: maak een [issue](https://github.com/cedanl/synthese-pipeline/issues) aan. Zet daar nooit echte data in, ook geen voorbeelden of foutmeldingen met waarden: alleen kolomnamen, types en aantallen.
- Bijdragen gaan via een pull request naar `main`. De CI draait ruff, ty en pytest op Linux en Windows (zie [Ontwikkeling](#ontwikkeling)); tests gebruiken alleen gegenereerde dummydata.
- De repo wordt onderhouden door CEDA.

## Licentie

MIT
