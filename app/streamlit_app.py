"""Cost Optimizer Streamlit app — minimal v1: single recommendations tab."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import streamlit as st

from cost_optimizer.llm.mock import MockLLM
from cost_optimizer.observability.jsonl_tracer import JsonlTracer
from cost_optimizer.providers.aws import AwsProvider
from cost_optimizer.runner import Runner


def main() -> None:
    st.set_page_config(page_title="Cost Optimizer Agent", layout="wide")
    st.title("Cloud Cost Optimizer Agent")
    st.caption("Upload an AWS CUR CSV; the agent will produce ranked optimization recommendations.")

    with st.sidebar:
        st.header("Run config")
        uploaded = st.file_uploader("AWS CUR CSV", type=["csv"])
        top_n = st.slider("Top-N resources", 1, 100, 10)
        llm_choice = st.selectbox("LLM", options=["mock", "claude"], index=0)
        run_clicked = st.button("Run analysis", type="primary", disabled=uploaded is None)

    if not run_clicked or uploaded is None:
        st.info("Upload a CSV and click **Run analysis** to begin.")
        return

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        csv_path = Path(tmp.name)

    llm = _build_llm(llm_choice)
    if llm is None:
        return

    runs_dir = Path(__file__).resolve().parents[1] / "runs"
    tracer = JsonlTracer(output_dir=runs_dir)
    runner = Runner(provider=AwsProvider(), llm=llm, tracer=tracer)

    with st.spinner("Analyzing resources..."):
        result = runner.run(csv_path, top_n=top_n)

    st.success(
        f"Analyzed {result.analyzed_count} resource(s); produced "
        f"{len(result.recommendations)} recommendation(s); failed {result.failed_count}."
    )

    if not result.recommendations:
        st.write("No recommendations.")
        return

    table = [
        {
            "type": r.type.value,
            "resource": r.resource_id,
            "monthly_savings_usd": r.monthly_savings_usd,
            "annual_savings_usd": r.annual_savings_usd,
            "confidence": r.confidence,
            "risk": r.risk_level,
        }
        for r in sorted(result.recommendations, key=lambda r: -r.annual_savings_usd)
    ]
    st.subheader("Recommendations")
    st.dataframe(table, use_container_width=True)

    st.subheader("Drill-down")
    for r in sorted(result.recommendations, key=lambda r: -r.annual_savings_usd):
        with st.expander(
            f"{r.type.value} — {r.resource_id} — ${r.annual_savings_usd:.0f}/yr"
        ):
            st.markdown(f"**Reasoning:** {r.reasoning}")
            st.markdown(f"**Confidence:** {r.confidence:.2f} • **Risk:** {r.risk_level} • "
                        f"**Effort:** {r.effort}")
            st.markdown("**Evidence:**")
            for e in r.evidence:
                st.code(json.dumps({"description": e.description,
                                    "source": e.source, "data": e.data}, indent=2),
                        language="json")
            if r.prerequisites:
                st.markdown("**Prerequisites:**")
                for p in r.prerequisites:
                    st.write(f"- {p}")
            if r.rollback_plan:
                st.markdown(f"**Rollback plan:** {r.rollback_plan}")
            if r.trace_id:
                lf_host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
                st.markdown(f"**Trace:** [`{r.trace_id}`]({lf_host}/traces/{r.trace_id}) "
                            f"(or check `runs/` for JSONL)")


def _build_llm(name: str):
    if name == "mock":
        return MockLLM()
    if name == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.error("ANTHROPIC_API_KEY not set; switch to 'mock' or set the key.")
            return None
        from cost_optimizer.llm.claude import ClaudeLLM
        return ClaudeLLM()
    st.error(f"Unknown LLM: {name}")
    return None


if __name__ == "__main__":
    main()
