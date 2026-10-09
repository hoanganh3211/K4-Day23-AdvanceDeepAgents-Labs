"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")   # [3], [1, 2], [1-3]; not [3](link)
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_REF_LINE = re.compile(r"^\[(\d+)\]\s*(.*)$")
_URL_REGEX = re.compile(r"https?://\S+")


def _group_numbers(group):
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        elif part.isdigit():
            numbers.append(int(part))
    return numbers


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    problems = []
    if not isinstance(sources, list) or len(sources) == 0:
        return ["no sources in sources.json"]

    source_ns = set()
    source_by_n = {}
    seen_urls = set()

    for idx, s in enumerate(sources):
        if not isinstance(s, dict):
            problems.append(f"source entry {idx} is not an object")
            continue
        n = s.get("n")
        url = s.get("url")
        if not isinstance(n, int):
            problems.append(f"source {idx}: n must be an int, got {n!r}")
        elif n in source_ns:
            problems.append(f"source {n}: duplicate n in sources.json")
        else:
            source_ns.add(n)
            source_by_n[n] = s

        if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"source n={n!r}: url must start with http:// or https://, got {url!r}")
        elif url in seen_urls:
            problems.append(f"duplicate url in sources.json: {url}")
        else:
            seen_urls.add(url)

    # Split report_text at the heading "## References"
    ref_matches = list(_REF_HEADING.finditer(report_text))
    if not ref_matches:
        problems.append("missing '## References' heading in report")
        return problems

    ref_heading_match = ref_matches[-1]
    body = report_text[:ref_heading_match.start()]
    refs_section = report_text[ref_heading_match.end():]

    # Find citations in body only (ignoring code blocks and markdown links)
    segments = _CODE.split(body)
    cited = set()
    for i, segment in enumerate(segments):
        if i % 2 == 1:
            continue  # inside code block / span
        for match in _GROUP.finditer(segment):
            for num in _group_numbers(match.group(1)):
                cited.add(num)

    # Check every cited number exists in sources
    for num in sorted(cited):
        if num not in source_ns:
            problems.append(f"[{num}] cited in body but missing from sources.json")

    # Check every source number is cited in body
    for num in sorted(source_ns):
        if num not in cited:
            problems.append(f"source [{num}] never cited in body")

    # Reference lines check
    ref_lines = [line.strip() for line in refs_section.splitlines() if line.strip()]
    seen_ref_numbers = []

    for line in ref_lines:
        m = _REF_LINE.match(line)
        if not m:
            continue
        ref_n = int(m.group(1))
        seen_ref_numbers.append(ref_n)

        raw_urls = _URL_REGEX.findall(line)
        expected_url = str(source_by_n.get(ref_n, {}).get("url") or "") if ref_n in source_by_n else None

        urls_clean = []
        for u in raw_urls:
            if expected_url and u == expected_url:
                urls_clean.append(u)
            elif expected_url and u.rstrip(").,;") == expected_url.rstrip(").,;"):
                urls_clean.append(expected_url)
            else:
                urls_clean.append(re.sub(r"[\.,\);>]+$", "", u))

        if len(urls_clean) != 1:
            problems.append(
                f"reference [{ref_n}] must contain exactly ONE URL, found {len(urls_clean)} in line: {line[:80]}"
            )
        else:
            line_url = urls_clean[0]
            if expected_url and line_url.rstrip("/") != expected_url.rstrip("/"):
                problems.append(f"reference [{ref_n}] URL mismatch: expected {expected_url!r}, got {line_url!r}")

    ref_counts = {}
    for num in seen_ref_numbers:
        ref_counts[num] = ref_counts.get(num, 0) + 1

    for num in sorted(source_ns):
        cnt = ref_counts.get(num, 0)
        if cnt == 0:
            problems.append(f"source [{num}] missing from References section")
        elif cnt > 1:
            problems.append(f"reference [{num}] appears {cnt} times in References section")

    for num in sorted(ref_counts.keys()):
        if num not in source_ns:
            problems.append(f"reference [{num}] in References section is not in sources.json")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
