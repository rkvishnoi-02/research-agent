"""Simple CLI for running the research graph."""

from __future__ import annotations

import argparse
import json

from langgraph.checkpoint.memory import InMemorySaver

from .graph import build_research_graph
from .models import ResearchRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the research agent from the command line.")
    parser.add_argument("--request", required=True, help="Research brief or goal.")
    parser.add_argument("--product", required=True, help="Product or market name.")
    parser.add_argument("--audience", required=True, help="Target audience.")
    parser.add_argument("--industry", default=None, help="Industry label.")
    parser.add_argument("--company-size", default=None, help="Company size or segment.")
    parser.add_argument("--mode", choices=["live", "strict"], default="live", help="Validation profile.")
    parser.add_argument("--thread-id", default="research-agent-cli", help="LangGraph thread id.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    graph = build_research_graph(checkpointer=InMemorySaver())
    request = ResearchRequest(
        research_request=args.request,
        product_name=args.product,
        audience=args.audience,
        industry=args.industry,
        company_size=args.company_size,
        research_mode=args.mode,
    )

    result = graph.invoke_request(request, config={"configurable": {"thread_id": args.thread_id}})
    print(json.dumps(result, default=lambda value: value.model_dump() if hasattr(value, "model_dump") else str(value), indent=2))


if __name__ == "__main__":
    main()
