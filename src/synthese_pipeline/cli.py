"""Commandoregel voor synthese-pipeline."""

from __future__ import annotations

import argparse
import sys

from synthese_pipeline import __version__
from synthese_pipeline.column_types import KINDS
from synthese_pipeline.describe import describe_model
from synthese_pipeline.errors import SyntheseError
from synthese_pipeline.logs import run_logging
from synthese_pipeline.model import load_model
from synthese_pipeline.pipeline import run_fit, run_synthesize
from synthese_pipeline.report import EXIT_CODES, EXIT_ERROR
from synthese_pipeline.storage import OUTPUT_FORMATS, TOKEN_ENV

EXIT_HELP = "exitcodes:\n" + "\n".join(f"  {code}  {text}" for code, text in EXIT_CODES.items())


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("geef een geheel getal") from None
    if value < 1:
        raise argparse.ArgumentTypeError("moet minstens 1 zijn")
    return value


def _seed(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("geef een geheel getal") from None
    if not 0 <= value < 2**32:
        raise argparse.ArgumentTypeError("moet tussen 0 en 4294967295 liggen")
    return value


def build_parser() -> argparse.ArgumentParser:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(
        prog="synthese-pipeline",
        description="Synthetische data maken als stap in een datapipeline.",
        epilog=EXIT_HELP,
        formatter_class=formatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="opdracht")

    fit = commands.add_parser(
        "fit",
        help="model (GMF-bestand) maken uit de echte data",
        description=(
            "Lees de bron en schrijf alleen het model (GMF-bestand). Disclosure control "
            "(metasyn-disclosure) staat standaard aan."
        ),
        epilog=(
            f"soorten voor --type: {', '.join(KINDS)}\n"
            "vrije tekst (soort freetext) gaat standaard niet in het model; --type kolom=text\n"
            "neemt de kolom toch op, --faker kolom=provider vult hem met verzonnen waarden.\n"
            f"abfss://-paden: het token komt uit de omgevingsvariabele {TOKEN_ENV}.\n\n" + EXIT_HELP
        ),
        formatter_class=formatter,
    )
    fit.add_argument(
        "source",
        metavar="BRON",
        help="CSV- of Parquet-bestand, Delta-tabel, map met tabellen of abfss://-pad",
    )
    fit.add_argument("model", metavar="MODEL", help="pad van het GMF-bestand (JSON)")
    fit.add_argument(
        "--type",
        dest="types",
        action="append",
        metavar="KOLOM=SOORT",
        help="soort van een kolom zelf opgeven, bijv. --type opleidingscode=categorical of "
        "--type studenten.leeftijd=discrete (mag vaker)",
    )
    fit.add_argument(
        "--faker",
        dest="fakers",
        action="append",
        metavar="KOLOM=PROVIDER",
        help="kolom vullen met verzonnen waarden van Faker (nl_NL), bijv. --faker naam=name of "
        "--faker studenten.opmerking=sentence (mag vaker)",
    )
    fit.add_argument(
        "--no-disclosure",
        action="store_true",
        help="disclosure control uitzetten (afgeraden: uitschieters kunnen in het model komen)",
    )
    fit.add_argument("--log", metavar="PAD", help="logbestand (alleen namen en aantallen)")

    synthesize = commands.add_parser(
        "synthesize",
        help="synthetische data maken uit alleen het model",
        description=(
            "Maak synthetische data uit alleen het GMF-bestand en schrijf een rapport (JSON) "
            "naast de data."
        ),
        epilog=EXIT_HELP,
        formatter_class=formatter,
    )
    synthesize.add_argument("model", metavar="MODEL", help="het GMF-bestand")
    synthesize.add_argument(
        "target",
        metavar="DOEL",
        help="uit.csv of uit.parquet (één tabel), een map of een abfss://-pad",
    )
    synthesize.add_argument(
        "--rows", type=_positive_int, help="aantal rijen per tabel (standaard: zoals in het model)"
    )
    synthesize.add_argument(
        "--seed", type=_seed, help="seed; zelfde model en seed geven dezelfde data"
    )
    synthesize.add_argument(
        "--format",
        dest="fmt",
        choices=OUTPUT_FORMATS,
        help="formaat bij een map als doel (standaard delta)",
    )
    synthesize.add_argument(
        "--source",
        metavar="BRON",
        help="echte data om mee te vergelijken (verdelingen, structuur, letterlijke kopieën)",
    )
    synthesize.add_argument(
        "--report",
        metavar="PAD",
        help="pad van het rapport (standaard naast de data)",
    )
    synthesize.add_argument("--log", metavar="PAD", help="logbestand (alleen namen en aantallen)")

    show = commands.add_parser(
        "toon",
        help="laat zien wat er in het model staat",
        description="Laat per tabel en kolom in gewone taal zien wat er in het model staat.",
    )
    show.add_argument("model", metavar="MODEL", help="het GMF-bestand")
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.command == "toon":
        print(describe_model(load_model(args.model)))
        return 0
    with run_logging(args.log):
        if args.command == "fit":
            run_fit(
                args.source,
                args.model,
                types=args.types,
                fakers=args.fakers,
                disclosure=not args.no_disclosure,
            )
            return 0
        result = run_synthesize(
            args.model,
            args.target,
            rows=args.rows,
            seed=args.seed,
            fmt=args.fmt,
            source=args.source,
            report_path=args.report,
        )
    print(f"rapport: {result.report_path} ({result.report['uitkomst']})")
    return result.exit_code


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return 2
    try:
        return _run(args)
    except SyntheseError as exc:
        print(f"fout: {exc}", file=sys.stderr)
    except Exception as exc:
        # De melding van een onbekende fout kan waarden uit de data bevatten: alleen het type.
        print(
            f"fout: onverwachte fout ({type(exc).__name__}). De details zijn weggelaten omdat "
            "ze waarden uit de data kunnen bevatten.",
            file=sys.stderr,
        )
    return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
