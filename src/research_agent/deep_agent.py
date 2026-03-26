"""Optional Deep Agents wrapper for operating the research graph."""

from __future__ import annotations

from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from .graph import build_research_graph
from .models import ResearchRequest, ValidationThresholds
from .services import ResearchServices


def create_research_deep_agent(
    model: str = "openai:gpt-4.1-mini",
    services: ResearchServices | None = None,
    thresholds: ValidationThresholds | None = None,
):
    try:
        from deepagents import create_deep_agent
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "deepagents is not installed. Install deepagents>=0.4.12 to use the operator wrapper."
        ) from exc

    graph = build_research_graph(
        services=services,
        thresholds=thresholds,
        checkpointer=InMemorySaver(),
    )

    def run_research_workflow(
        research_request: str,
        product_name: str,
        audience: str,
        product_url: str = "",
        industry: str = "",
        company_size: str = "",
        focus_notes: str = "",
    ) -> str:
        request = ResearchRequest(
            research_request=research_request,
            product_name=product_name,
            product_url=product_url or None,
            audience=audience,
            industry=industry or None,
            company_size=company_size or None,
            focus_notes=focus_notes or None,
        )
        result = graph.invoke_request(
            request,
            config={"configurable": {"thread_id": f"research-{product_name.lower().replace(' ', '-')}"}} ,
        )
        if result.get("final_research_packet"):
            packet = result["final_research_packet"]
            return (
                f"Approved packet with {packet.total_quotes} quotes, "
                f"{packet.total_insights} insights, truth density {packet.truth_density:.2f}."
            )
        report = result.get("escalation_report")
        if report:
            return f"Escalated after validation failures: {', '.join(item.code for item in report.validation_failures)}"
        return "Workflow ended without a packet."

    tool = StructuredTool.from_function(
        name="run_research_workflow",
        description=(
            "Run the LangGraph-first research workflow to collect verbatim buyer evidence, "
            "validate quote authenticity, and produce an approved research packet."
        ),
        func=run_research_workflow,
    )

    return create_deep_agent(
        model=model,
        tools=[tool],
        checkpointer=InMemorySaver(),
        system_prompt=(
            "You are an operator for the research workflow. Gather the required request fields, "
            "run the workflow tool, and report whether the run produced an approved packet or escalated."
        ),
    )
