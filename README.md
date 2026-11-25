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
| een map | Per tabel een Delta-tabel `<map>/<tabel>`; met `--format parquet` of `--format csv` een bestand per tabel; rapport: `<map>/rapport.json` |
| `abfss://…/Tables` | Per tabel een Delta-tabel; geef het rapport op met `--report` |

Alles wordt gelezen met polars, niet met pandas. Een CSV-bestand wordt eerst als tekst gelezen; daarna krijgt elke kolom het type waar alle ingevulde waarden in passen (geheel getal, kommagetal, datum, waar/onwaar of tekst). Codes met voorloopnullen, zoals `01234`, blijven tekst. Het scheidingsteken (komma, puntkomma of tab) wordt uit de kopregel afgeleid; bij een puntkomma mag een getal een decimale komma hebben. De synthetische tabellen krijgen dezelfde kolomnamen en datatypes als de bron. Bestaande uitvoer wordt overschreven, maar nooit iets buiten het doel: een tabelnaam uit het model wordt een map- of bestandsnaam, dus `synthesize` en `toon` weigeren een model waarin een tabelnaam een pad is (zoals `../bron` of `/data/bron`).

Voor `abfss://`-paden geeft de tool `use_fabric_endpoint` en een bearer token mee aan delta-rs. Het token komt uit de omgevingsvariabele `SYNTHESE_PIPELINE_TOKEN`, in een Fabric-notebook bijvoorbeeld `notebookutils.credentials.getToken("storage")`. Vanuit Python kan het ook direct: `run_fit(bron, model, storage_options=fabric_storage_options(token))` uit `synthese_pipeline.pipeline` en `synthese_pipeline.storage`. Dit is nog niet getest in een echte Fabric-omgeving.

### Kolomtypes

Elke kolom krijgt een soort. De tool leidt die af uit het datatype en de waarden; de eerste regel die past wint:

| Soort | Wanneer | Wat metasyn ervan maakt |
|---|---|---|
| `categorical` | waar/onwaar; een geheel getal met hooguit 12 unieke waarden of tekst met hooguit 30, en hooguit half zoveel unieke als ingevulde waarden (bijv. Likert 1-5, M/V) | een keuze uit de categorieën met hun kansen |
| `freetext` | tekst die op vrije tekst lijkt: meer dan de helft van de ingevulde waarden heeft een spatie, of de waarden zijn gemiddeld minstens 20 tekens lang, en het is geen code (zie hieronder) | niets: de kolom gaat niet in het model en ontbreekt in de synthetische data |
| `key` | geheel getal of tekst, geen lege waarden, alle waarden uniek, minstens 20 rijen | unieke waarden; bij een code een uniek patroon |
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

Eén tabel geeft `config_version = "1.2"` met `[[var]]` per kolom (1.0 en 1.1 worden ook gelezen). Meer tabellen geven `config_version = "2.0"` met `[[table]]` per tabel en `[[table.var]]` per kolom. Relaties tussen tabellen worden nog niet ondersteund.

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

### Stap 0: dummydata zonder data

Een instelling kan beginnen zonder dat er echte data of een model de beveiligde omgeving verlaat. Ze schrijft een specificatie met per kolom een verdeling, en maakt daar dummydata mee ([data-free in metasyn](https://metasyn.readthedocs.io/en/latest/datafree.html)):

```bash
uv run synthese-pipeline synthesize --spec examples/spec.toml --rows 100 uit.csv --seed 1
```

[`examples/spec.toml`](examples/spec.toml) heeft een sleutel, een Likert-item, een code, een datum, een kommagetal en een uitgesloten kolom met vrije tekst. Zonder `--rows` geldt `n_rows` uit de specificatie. Het rapport werkt zoals altijd: de structuur wordt vergeleken met de specificatie, met dezelfde exitcodes, en `data_free` staat op `true`. Dezelfde specificatie met dezelfde seed geeft dezelfde data. Geef `MODEL` en `--spec` niet samen op: dat is een gebruiksfout (exitcode 2).

Zonder data is per kolom een verdeling nodig. Een `key` zonder verdeling wordt een oplopend nummer vanaf 1, een kolom met `categorieen` een keuze met gelijke kansen, en een kolom met `faker` krijgt verzonnen waarden. Een ontwerp van `init-spec` heeft die verdelingen nog niet: vul ze aan. Dummydata uit een specificatie is **niet statistisch geldig**. Ze heeft de juiste kolommen en types, maar de verdelingen zijn wat je opschrijft, niet wat er in de echte data zit. Gebruik ze om code te bouwen en te testen, niet voor uitkomsten.

### Privacy

- **Disclosure control staat standaard aan.** `fit` gebruikt de plugin [metasyn-disclosure](https://github.com/sodascience/metasyn-disclosure-control) (versie 0.3.3). Zeldzame waarden en uitschieters, zoals een leeftijd die maar bij één persoon voorkomt, komen dan niet letterlijk in het model. Het model (`synthese_pipeline.disclosure_control` en `provenance`) en het rapport vermelden of de plugin aan stond. Uitzetten kan met `--no-disclosure`; dat is afgeraden.
- **Logs en rapporten bevatten geen waarden uit de data**, alleen tabelnamen, kolomnamen, datatypes, aantallen en scores. Met `--log PAD` schrijven `fit` en `synthesize` dezelfde regels ook naar een bestand. Een onverwachte fout toont alleen het soort fout, omdat de melding waarden kan bevatten.
- **Vrije tekst gaat standaard niet in het model** (zie [Vrije tekst](#vrije-tekst)).
- **Een ontwerp-specificatie bevat geen waarden uit de data**, behalve met `--with-categories`. Omschrijvingen schrijft de instelling zelf: zet daar geen echte waarden in.
- Het model zelf bevat wel samenvattingen van de data, zoals categorieën en grenzen. Lees het na met `toon` voordat het de beveiligde omgeving verlaat.

### Rapport

`synthesize` schrijft een rapport (JSON) met:

- de versie van synthese-pipeline en metasyn, de seed, en of disclosure control aan stond;
- per tabel: of de structuur gelijk is (kolommen, volgorde, datatypes) met de afwijkingen, en welke kolommen zijn uitgesloten (vrije tekst);
- met `--source BRON`: per kolom een verdelingsmaat (TV-afstand voor categorieën, Wasserstein-afstand voor getallen, in dagen voor datums), het aandeel lege waarden echt en synthetisch, nieuwe categorieën, en het aantal **letterlijke kopieën**: synthetische rijen die gelijk zijn aan een rij die in de echte data maar één keer voorkomt.

Zonder `--source` vergelijkt het rapport de structuur met het model en staan de kopieën op "niet gecontroleerd"; de exitcode verandert daar niet door. Zonder `--seed` kiest de tool een seed en zet die in het rapport. Hetzelfde model met dezelfde seed geeft dezelfde data.

### Exitcodes

| Code | Betekenis |
|---|---|
| 0 | geslaagd |
| 1 | fout (invoer, model of omgeving) |
| 2 | verkeerd gebruik van de commandoregel |
| 3 | structuur wijkt af van de bron of het model |
| 4 | letterlijke kopieën van echte rijen gevonden (gaat voor 3) |

## Ontwikkeling

```bash
uv sync --group dev
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

Tests gebruiken alleen gegenereerde dummydata (`tests/synthese_pipeline/dummy.py`).

Een devcontainer staat in `.devcontainer/`.

## Licentie

MIT
