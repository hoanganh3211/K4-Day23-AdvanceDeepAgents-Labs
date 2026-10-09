"""Runs INSIDE the sandbox: python3 finalize_citations.py [report.md] [sources.json]

Makes the citations valid BY CONSTRUCTION instead of trusting an LLM to hand-maintain numbered references:
  * merges sources that share a URL, drops sources the text never cites,
  * renumbers the survivors 1..k in order of first appearance and rewrites every citation in the body
    (grouped forms [1, 2] / [1-3] are normalised to [1][2] / [1][2][3]; code spans and Markdown links are left alone),
  * regenerates the `## References` section (one line per source, exactly one URL) and rewrites sources.json.
Idempotent. If the body cites a number that is not in sources.json nothing is written and the problem is reported.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")   # [3]  [1, 2]  [1-3]  [2-3]; not [3](link)
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")


def _clean(text):
    return " ".join(str(text).split())


def _group_numbers(group):
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            numbers.append(int(part))
    return numbers


def _body_of(report_text):
    matches = list(_REF_HEADING.finditer(report_text))
    return (report_text[:matches[-1].start()] if matches else report_text).rstrip()


def finalize(report_text, sources):
    if not isinstance(sources, list) or not all(isinstance(e, dict) for e in sources):
        return report_text, sources, ["sources.json must be a JSON list of objects"]
    bad = [f"source n={e.get('n')!r} has no url" for e in sources if not e.get("url")]
    if bad:
        return report_text, sources, bad
    body = _body_of(report_text)
    by_n = {s["n"]: s for s in sources if isinstance(s.get("n"), int)}
    canonical = {}
    for entry in sources:
        if isinstance(entry.get("n"), int):
            canonical.setdefault(entry["url"], entry["n"])
    alias = {n: canonical[entry["url"]] for n, entry in by_n.items()}  # duplicate url -> first number
    segments = _CODE.split(body)                                       # odd indexes are code: never touched
    order, problems = [], []
    for i, segment in enumerate(segments):
        if i % 2:
            continue
        for match in _GROUP.finditer(segment):
            for n in _group_numbers(match.group(1)):
                if n not in alias:
                    message = f"[{n}] is cited in the text but missing from sources.json"
                    if message not in problems:
                        problems.append(message)
                elif alias[n] not in order:
                    order.append(alias[n])
    if not order and not problems:
        problems.append("no citations found in the report body")
    new_number = {old: k + 1 for k, old in enumerate(order)}

    def rewrite(match):
        return "".join(f"[{new_number[alias[n]]}]" if n in alias else f"[{n}]" for n in _group_numbers(match.group(1)))

    new_body = "".join(seg if i % 2 else _GROUP.sub(rewrite, seg) for i, seg in enumerate(segments))
    new_sources = [dict(by_n[old], n=new_number[old]) for old in order]
    lines = []
    for s in new_sources:
        title = _clean(re.sub(r"https?://\S+", "", str(s.get("title") or ""))) or "Untitled"
        lines.append(f"[{s['n']}] {title}. {s.get('source', 'web')}. {s['url']} ({s.get('date') or 'n.d.'})")
    report = new_body + ("\n\n## References\n" + "\n".join(lines) + "\n" if lines else "\n")
    return report, new_sources, problems


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


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            raw_sources = f.read()
        sources = _load_sources_safe(raw_sources)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    new_report, new_sources, problems = finalize(report, sources)
    if problems:
        print("NOT finalized (fix these in the report body, then run again):")
        print("\n".join(problems))
        return 1
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(new_report)
    with open(sources_path, "w", encoding="utf-8") as f:
        json.dump(new_sources, f, ensure_ascii=False, indent=2)
    print(f"FINALIZED: kept {len(new_sources)} of {len(sources)} sources; References regenerated")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
