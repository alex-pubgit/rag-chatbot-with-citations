"""Command line: ``ragbot ingest | ask | serve | eval``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from pydantic import ValidationError

from ragbot.chatbot import build_chatbot, build_embedder
from ragbot.errors import RagbotError
from ragbot.evaluation import evaluate, format_table, load_golden, suggest_min_score
from ragbot.index import Index
from ragbot.ingest import ingest
from ragbot.retrieval import Retriever
from ragbot.settings import Settings


def _ingest(args: argparse.Namespace, settings: Settings) -> int:
    report = ingest(
        Path(args.folder),
        settings.index_dir,
        build_embedder(settings),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        rebuild=args.rebuild,
    )
    print(
        f"Indexed {report.documents} documents as {report.chunks} passages in {settings.index_dir}"
    )
    print(
        f"added {len(report.added)}, updated {len(report.updated)}, "
        f"unchanged {len(report.unchanged)}, removed {len(report.removed)}"
    )
    for source, reason in report.failed.items():
        print(f"skipped {source}: {reason}", file=sys.stderr)
    return 0


def _ask(args: argparse.Namespace, settings: Settings) -> int:
    answer = build_chatbot(settings).ask(args.question)
    if args.json:
        print(json.dumps(asdict(answer), ensure_ascii=False, indent=2))
        return 0
    print(answer.text)
    for citation in answer.citations:
        where = citation.heading or citation.title
        print(f"  [{citation.n}] {citation.source} - {where} (score {citation.score:.2f})")
    return 0


def _serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    from ragbot.api import create_app

    uvicorn.run(create_app(settings), host=args.host, port=args.port)
    return 0


def _eval(args: argparse.Namespace, settings: Settings) -> int:
    items = load_golden(Path(args.golden))
    embedder = build_embedder(settings)
    index = Index.load(settings.index_dir)
    if args.calibrate:
        retriever = Retriever.from_index(index, embedder, mode=settings.retrieval)
        threshold = suggest_min_score(retriever, items)
        print(f"Suggested RAGBOT_MIN_SCORE for '{embedder.fingerprint}': {threshold:.2f}")
        return 0

    modes = ("bm25", "dense", "hybrid") if args.compare else (settings.retrieval,)
    rows = []
    for mode in modes:
        chatbot = build_chatbot(
            settings.model_copy(update={"retrieval": mode}), index=index, embedder=embedder
        )
        rows.append((mode, evaluate(chatbot, items, settings.top_k)))
    print(format_table(rows))

    scores = dict(rows)[settings.retrieval]
    floors = {
        "hit rate": (scores.hit_rate, args.min_hit_rate),
        "correct answers": (scores.correct, args.min_correct),
        "refusal recall": (scores.refusal_recall, args.min_refusal_recall),
    }
    failed = [
        f"{name} {value:.0%} is below the required {floor:.0%}"
        for name, (value, floor) in floors.items()
        if floor is not None and value < floor
    ]
    for message in failed:
        print(f"error: {message}", file=sys.stderr)
    return 1 if failed else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragbot", description="Answer questions from your documents, with citations."
    )
    commands = parser.add_subparsers(required=True, metavar="command")

    ingest_cmd = commands.add_parser("ingest", help="index a folder of documents")
    ingest_cmd.add_argument("folder", help="folder with .md, .txt, .html or .pdf files")
    ingest_cmd.add_argument("--rebuild", action="store_true", help="ignore the existing index")
    ingest_cmd.set_defaults(run=_ingest)

    ask_cmd = commands.add_parser("ask", help="ask one question")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("--json", action="store_true", help="print the full answer as JSON")
    ask_cmd.set_defaults(run=_ask)

    serve_cmd = commands.add_parser("serve", help="start the API and the web page")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8000)
    serve_cmd.set_defaults(run=_serve)

    eval_cmd = commands.add_parser("eval", help="score the bot against a golden set")
    eval_cmd.add_argument("--golden", default="eval/golden.jsonl", help="path to the golden set")
    eval_cmd.add_argument("--compare", action="store_true", help="score bm25, dense and hybrid")
    eval_cmd.add_argument(
        "--calibrate", action="store_true", help="suggest a refusal threshold and exit"
    )
    eval_cmd.add_argument("--min-hit-rate", type=float, help="fail below this hit rate (0-1)")
    eval_cmd.add_argument(
        "--min-correct", type=float, help="fail below this share of correct answers"
    )
    eval_cmd.add_argument("--min-refusal-recall", type=float, help="fail below this refusal recall")
    eval_cmd.set_defaults(run=_eval)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    try:
        return args.run(args, Settings())
    except ValidationError as exc:
        problems = "; ".join(
            f"RAGBOT_{'_'.join(map(str, error['loc'])).upper()}: {error['msg']}"
            for error in exc.errors()
        )
        print(f"error: invalid settings: {problems}", file=sys.stderr)
        return 1
    except RagbotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
