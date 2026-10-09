"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import (
    FINALIZER_PATH,
    REPORT_PATH,
    SOURCES_PATH,
    VALIDATOR_PATH,
    WORKDIR,
    build_lead_agent,
)
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"  # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    if not topic or not isinstance(topic, str):
        return "topic"
    slug = topic.lower().strip()
    # Replace non-alphanumeric characters with hyphens
    slug = re.sub(r"[^\w]+", "-", slug).strip("-_")
    if not slug:
        return "topic"
    return slug[:60].rstrip("-_") or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Please conduct an in-depth, comprehensive research survey on the topic: '{topic}'.\n\n"
        "Follow your required research workflow strictly:\n"
        "1. Plan with `write_todos` and decompose this topic into at least 3-4 independent sub-questions.\n"
        "2. Delegate each sub-question to the `researcher` subagent in parallel using the `task` tool "
        "(ensure each delegation message contains the topic, sub-question, notes file path, and note format).\n"
        "3. Review all notes and compile all sources into `/tmp/work/research/sources.json`.\n"
        "   IMPORTANT: The sources MUST represent at least 3 distinct source families (among 'arxiv', 'hf-daily', 'hf-search', 'web'). "
        "   If fewer than 3 families are present, delegate additional researcher tasks to find missing families.\n"
        "4. Write the survey body to `/tmp/work/report/report.md` following the REPORT_TEMPLATE structure. "
        "   DO NOT write the `## References` section.\n"
        "5. Run `/tmp/work/research/finalize_citations.py` via the `execute` tool to generate references and renumber citations.\n"
        "6. Run `/tmp/work/research/check_citations.py` via the `execute` tool and ensure it passes with OK.\n"
        "7. Use the `citation-checker` subagent to spot-check 2-3 key claims against source URLs."
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}."""
    tool_counts = Counter()
    input_tokens = 0
    output_tokens = 0

    for msg in messages:
        # Check tool calls
        tcalls = getattr(msg, "tool_calls", None)
        if not tcalls and isinstance(msg, dict):
            tcalls = msg.get("tool_calls")
        if tcalls and isinstance(tcalls, list):
            for tc in tcalls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name:
                    tool_counts[str(name)] += 1

        # Check token usage
        usage = getattr(msg, "usage_metadata", None)
        if not usage and isinstance(msg, dict):
            usage = msg.get("usage_metadata")
        if not usage:
            resp_meta = getattr(msg, "response_metadata", None) or (
                msg.get("response_metadata") if isinstance(msg, dict) else None
            )
            if isinstance(resp_meta, dict):
                usage = resp_meta.get("token_usage")

        if isinstance(usage, dict):
            in_tok = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            out_tok = usage.get("output_tokens") or usage.get("completion_tokens") or 0
            try:
                input_tokens += int(in_tok)
                output_tokens += int(out_tok)
            except (ValueError, TypeError):
                pass

    subagent_calls = tool_counts.get("task", 0)

    return {
        "model": str(model_name),
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": subagent_calls,
        "tool_calls": dict(tool_counts),
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
        },
    }


def _load_sources_safe(raw_text):
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    # Handle multiple concatenated JSON arrays or objects (e.g. [...] [...])
    decoder = json.JSONDecoder()
    pos = 0
    items = []
    while pos < len(text):
        remaining = text[pos:].lstrip()
        pos = len(text) - len(remaining)
        if pos >= len(text):
            break
        try:
            val, end = decoder.raw_decode(text, pos)
            if isinstance(val, list):
                items.extend(val)
            elif isinstance(val, dict):
                items.append(val)
            pos = end
        except Exception:
            pos += 1
    if items:
        return items
    return json.loads(text)


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)

    if not report_bytes or not report_bytes.strip():
        raise RuntimeError(f"Report is missing or empty in sandbox at {REPORT_PATH}")
    if not sources_bytes or not sources_bytes.strip():
        raise RuntimeError(f"Sources file is missing or empty in sandbox at {SOURCES_PATH}")

    try:
        sources_text = sources_bytes.decode("utf-8")
        sources = _load_sources_safe(sources_text)
        if not isinstance(sources, list) or len(sources) == 0:
            raise ValueError("sources.json must be a non-empty list of source records")
    except Exception as exc:
        raise RuntimeError(f"sources.json is invalid JSON: {exc}")

    # Build metadata
    summary = summarize(messages, elapsed, model_name)
    families = sorted(list({s.get("source") for s in sources if isinstance(s, dict) and s.get("source")}))

    meta = {
        "topic": topic,
        **summary,
        "n_sources": len(sources),
        "source_families": families,
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)
    md_path = reports_dir / f"{slug}.md"
    sources_path = reports_dir / f"{slug}.sources.json"
    meta_path = reports_dir / f"{slug}.meta.json"

    md_path.write_bytes(report_bytes)
    # Always write cleanly serialized JSON
    sources_path.write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return md_path


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    clean_topic = topic.strip()
    if not clean_topic:
        print("Usage: python research.py \"<topic>\"", file=sys.stderr)
        return 2

    model = make_model()
    model_name = (
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or os.getenv("LAB_MODEL", "model")
    )

    start = time.monotonic()
    try:
        with open_sandbox() as backend:
            backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            upload(
                backend,
                {
                    VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                    FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
                },
            )
            agent = build_lead_agent(backend, model)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": build_prompt(clean_topic)}]},
                config={"recursion_limit": 1000},
            )
            elapsed = time.monotonic() - start
            messages = result.get("messages", []) if isinstance(result, dict) else []

            # Ensure finalizer is run inside the sandbox before downloading
            backend.execute(f"python3 {FINALIZER_PATH}")
            backend.execute(f"python3 {VALIDATOR_PATH}")

            saved_report = save_outputs(backend, clean_topic, messages, elapsed, model_name)
            print(f"SUCCESS: Report saved to {saved_report}")
            return 0
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
