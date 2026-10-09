"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import time
import xml.etree.ElementTree as ET

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"

# Module-level state for arXiv etiquette (>= 3s between calls)
_LAST_ARXIV_CALL = 0.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def _clean_str(text) -> str:
    return " ".join(str(text or "").split())


def _redact_secrets(text: str) -> str:
    """Ensure API keys (especially EXA_API_KEY, TAVILY_API_KEY) never appear in agent-facing strings."""
    exa_key = (os.getenv("EXA_API_KEY") or "").strip()
    tavily_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    result = str(text)
    if exa_key and len(exa_key) > 5:
        result = result.replace(exa_key, "[REDACTED]")
    if tavily_key and len(tavily_key) > 5:
        result = result.replace(tavily_key, "[REDACTED]")
    result = re.sub(r"tvly-[A-Za-z0-9_\-]+", "[REDACTED]", result)
    result = re.sub(r"exaApiKey=[A-Za-z0-9_\-]+", "exaApiKey=[REDACTED]", result)
    result = re.sub(r"Bearer\s+[A-Za-z0-9_\-]+", "Bearer [REDACTED]", result)
    return result


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    PSEUDO-CODE:
      for attempt in 0 .. attempts-1:
          try: return fn()
          except RetryableError as e:
              if this was the last attempt: raise
              delay = e.retry_after if the server told us, else exponential backoff base * 2**attempt
              cap the delay at `cap` seconds; add random jitter to the exponential case
              sleep(delay)
    Use it to wrap EVERY network call below. Also treat these as retryable: HTTP 429/500/502/503/504,
    httpx.TransportError (timeouts, connection resets). Read the Retry-After header when present.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except (RetryableError, httpx.TransportError) as exc:
            if attempt == attempts - 1:
                raise

            retry_after = getattr(exc, "retry_after", None)
            if retry_after is not None:
                try:
                    delay = float(retry_after)
                except (ValueError, TypeError):
                    delay = base * (2**attempt) + random.uniform(0.1, 1.0)
            else:
                delay = base * (2**attempt) + random.uniform(0.1, 1.0)

            delay = min(cap, delay)
            time.sleep(delay)


def _http_get_with_status_retry(client: httpx.Client, url: str, **kwargs):
    """Make GET request, translating 429/5xx into RetryableError so with_retry handles them."""
    resp = client.get(url, **kwargs)
    if resp.status_code in (429, 500, 502, 503, 504):
        retry_after = resp.headers.get("Retry-After")
        ra_val = None
        if retry_after:
            try:
                ra_val = float(retry_after)
            except ValueError:
                pass
        raise RetryableError(f"HTTP {resp.status_code}", retry_after=ra_val)
    return resp


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    global _LAST_ARXIV_CALL
    try:
        # Keep only word characters / hyphens
        terms = re.findall(r"[\w\-]+", query)
        if not terms:
            return "NO RESULTS"

        # Respect arXiv etiquette: at least 3 seconds between two arXiv calls
        now = time.time()
        elapsed = now - _LAST_ARXIV_CALL
        if elapsed < 3.0:
            time.sleep(3.0 - elapsed)
        _LAST_ARXIV_CALL = time.time()

        clamped_results = max(1, min(30, max_results))
        search_query = " AND ".join(f"all:{t}" for t in terms)
        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": clamped_results,
        }

        def _fetch():
            with httpx.Client(timeout=30.0) as client:
                resp = _http_get_with_status_retry(client, ARXIV_URL, params=params)
                resp.raise_for_status()
                return resp.text

        # Use with_retry for arXiv call
        try:
            xml_content = with_retry(_fetch, attempts=3, base=1.0, cap=10.0)
        except Exception as exc:
            # Fallback to finding real arXiv papers via HF papers index when export.arxiv.org returns 429
            try:
                clean_terms = " ".join(terms)
                with httpx.Client(timeout=20.0) as client:
                    hf_resp = client.get(HF_SEARCH_URL, params={"q": clean_terms, "limit": clamped_results})
                    if hf_resp.status_code == 200:
                        items = hf_resp.json()
                        records = []
                        for item in items:
                            paper = item.get("paper") if isinstance(item, dict) and "paper" in item else item
                            if not isinstance(paper, dict) or not paper.get("id"):
                                continue
                            p_id = str(paper["id"])
                            records.append({
                                "id": p_id,
                                "url": f"https://arxiv.org/abs/{p_id}",
                                "published": str(paper.get("publishedAt") or "")[:10],
                                "title": _clean_str(paper.get("title") or ""),
                                "summary": _clean_str(paper.get("ai_summary") or paper.get("summary") or "")[:600],
                            })
                        if records:
                            return json.dumps(records, ensure_ascii=False)
            except Exception:
                pass
            return f"ERROR: {type(exc).__name__}: {_redact_secrets(str(exc))}"

        # Parse Atom XML
        root = ET.fromstring(xml_content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        records = []

        for entry in root.findall("atom:entry", ns):
            id_elem = entry.find("atom:id", ns)
            if id_elem is None or not id_elem.text:
                continue

            raw_id = id_elem.text.strip().split("/abs/")[-1]
            paper_id = re.sub(r"v\d+$", "", raw_id)
            url = f"https://arxiv.org/abs/{paper_id}"

            pub_elem = entry.find("atom:published", ns)
            published = pub_elem.text.strip()[:10] if pub_elem is not None and pub_elem.text else ""

            title_elem = entry.find("atom:title", ns)
            title = _clean_str(title_elem.text) if title_elem is not None else ""

            sum_elem = entry.find("atom:summary", ns)
            summary = _clean_str(sum_elem.text)[:600] if sum_elem is not None else ""

            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {_redact_secrets(str(exc))}"


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        clamped_limit = max(1, min(100, limit))
        params = {"limit": clamped_limit}
        if date.strip():
            params["date"] = date.strip()

        def _fetch():
            with httpx.Client(timeout=30.0) as client:
                resp = _http_get_with_status_retry(client, HF_DAILY_URL, params=params)
                resp.raise_for_status()
                return resp.json()

        items = with_retry(_fetch, attempts=5, base=1.0, cap=30.0)
        if not isinstance(items, list):
            return "NO RESULTS"

        records = []
        kw = keyword.strip().lower()

        for item in items:
            paper = item.get("paper") if isinstance(item, dict) and "paper" in item else item
            if not isinstance(paper, dict) or not paper.get("id"):
                continue

            paper_id = str(paper["id"])
            url = f"https://huggingface.co/papers/{paper_id}"
            published = str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10]
            title = _clean_str(paper.get("title") or item.get("title") or "")
            summary = _clean_str(paper.get("summary") or item.get("summary") or "")[:600]
            upvotes = paper.get("upvotes") or 0
            github = paper.get("githubRepo") or None
            stars = paper.get("githubStars") or None

            if kw and (kw not in title.lower() and kw not in summary.lower()):
                continue

            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        records.sort(key=lambda r: r.get("upvotes") or 0, reverse=True)
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {_redact_secrets(str(exc))}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        clean_q = query.strip()
        if not clean_q:
            return "NO RESULTS"

        clamped_limit = max(1, min(50, limit))
        params = {"q": clean_q, "limit": clamped_limit}

        def _fetch():
            with httpx.Client(timeout=30.0) as client:
                resp = _http_get_with_status_retry(client, HF_SEARCH_URL, params=params)
                resp.raise_for_status()
                return resp.json()

        items = with_retry(_fetch, attempts=5, base=1.0, cap=30.0)
        if not isinstance(items, list):
            return "NO RESULTS"

        records = []
        for item in items:
            paper = item.get("paper") if isinstance(item, dict) and "paper" in item else item
            if not isinstance(paper, dict) or not paper.get("id"):
                continue

            paper_id = str(paper["id"])
            url = f"https://huggingface.co/papers/{paper_id}"
            published = str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10]
            title = _clean_str(paper.get("title") or item.get("title") or "")
            summary = _clean_str(paper.get("ai_summary") or paper.get("summary") or item.get("summary") or "")[:600]
            upvotes = paper.get("upvotes") or 0
            github = paper.get("githubRepo") or None
            stars = paper.get("githubStars") or None

            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {_redact_secrets(str(exc))}"


# ---- Helper for Tavily API ----
def _call_tavily_search(query: str, num_results: int = 5) -> str:
    """Call Tavily Search API when TAVILY_API_KEY is configured."""
    tavily_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not tavily_key:
        return ""

    def _execute():
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                TAVILY_SEARCH_URL,
                json={
                    "api_key": tavily_key,
                    "query": query,
                    "max_results": max(1, min(10, num_results)),
                    "include_raw_content": False,
                },
            )
            if resp.status_code in (429, 500, 502, 503, 504):
                ra = resp.headers.get("Retry-After")
                ra_val = float(ra) if ra and ra.isdigit() else 2.0
                raise RetryableError(f"Tavily HTTP {resp.status_code}", retry_after=ra_val)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return "NO RESULTS"
            formatted = []
            for r in results:
                title = _clean_str(r.get("title", ""))
                url = str(r.get("url", ""))
                content = _clean_str(r.get("content", ""))
                formatted.append(f"Title: {title}\nURL: {url}\nSnippet: {content}")
            return "\n\n---\n\n".join(formatted)

    return with_retry(_execute, attempts=5, base=1.0, cap=30.0)


def _call_tavily_fetch(url: str) -> str:
    """Call Tavily Extract API (with fallback to direct GET) when TAVILY_API_KEY is configured."""
    tavily_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not tavily_key:
        return ""

    def _execute():
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                TAVILY_EXTRACT_URL,
                json={
                    "api_key": tavily_key,
                    "urls": [url],
                },
            )
            if resp.status_code in (429, 500, 502, 503, 504):
                ra = resp.headers.get("Retry-After")
                ra_val = float(ra) if ra and ra.isdigit() else 2.0
                raise RetryableError(f"Tavily HTTP {resp.status_code}", retry_after=ra_val)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if results and isinstance(results, list):
                    raw = results[0].get("raw_content") or ""
                    if raw.strip():
                        return raw[:12000]

            # Fallback direct HTTP GET if extract didn't return content
            get_resp = client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                timeout=20.0,
                follow_redirects=True,
            )
            if get_resp.status_code == 200:
                text = get_resp.text
                text = re.sub(r"<script.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<[^>]+>", " ", text)
                return _clean_str(text)[:12000]
            return ""

    return with_retry(_execute, attempts=5, base=1.0, cap=30.0)


# ---- Helper for Exa MCP ----
def _call_exa_mcp(tool_name: str, arguments: dict) -> str:
    """Call Exa MCP over HTTP POST JSON-RPC. Detects rate limits in status code, _meta, and body."""
    exa_key = (os.getenv("EXA_API_KEY") or "").strip()
    target_url = f"{EXA_URL}?exaApiKey={exa_key}" if exa_key else EXA_URL

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if exa_key:
        headers["Authorization"] = f"Bearer {exa_key}"

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }

    def _execute():
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(target_url, headers=headers, json=payload)
            if resp.status_code in (429, 500, 502, 503, 504):
                ra = resp.headers.get("Retry-After")
                ra_val = float(ra) if ra and ra.isdigit() else 20.0
                raise RetryableError(f"Exa HTTP {resp.status_code}", retry_after=ra_val)
            resp.raise_for_status()

            # Parse SSE or JSON
            raw_text = resp.text.strip()
            data = None
            for line in raw_text.splitlines():
                if line.startswith("data:"):
                    try:
                        data = json.loads(line[5:].strip())
                        break
                    except json.JSONDecodeError:
                        pass
            if data is None:
                try:
                    data = resp.json()
                except Exception:
                    pass

            if not data or not isinstance(data, dict):
                return ""

            # Check JSON-RPC error
            if "error" in data:
                err = data["error"]
                msg = str(err.get("message") if isinstance(err, dict) else err)
                if "rate limit" in msg.lower() or (isinstance(err, dict) and err.get("code") == -32000):
                    raise RetryableError(f"Exa rate limit: {msg}", retry_after=20.0)
                raise RuntimeError(f"Exa JSON-RPC error: {msg}")

            result = data.get("result", {})
            # Check rate limiting in _meta (GUIDE 1.4: Exa returns HTTP 200 with _meta flag)
            meta = result.get("_meta", {})
            if meta.get("rateLimited") or "rate limit" in str(meta).lower():
                raise RetryableError("Exa rate limit flag in _meta", retry_after=20.0)

            content = result.get("content", [])
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    t = item.get("text", "")
                    texts.append(t)
            combined = "\n\n".join(texts).strip()

            # Check rate limiting message in text
            if "rate limit" in combined.lower() and len(combined) < 500:
                raise RetryableError("Exa rate limit in text content", retry_after=20.0)

            return combined

    return with_retry(_execute, attempts=5, base=2.0, cap=60.0)


# ---- TODO 4: web search / fetch through Tavily or Exa MCP ----
@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web. Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        clean_q = query.strip()
        if not clean_q:
            return "NO RESULTS"
        clamped_n = max(1, min(10, num_results))

        if os.getenv("TAVILY_API_KEY"):
            text = _call_tavily_search(clean_q, clamped_n)
        else:
            obj = objective.strip() if objective.strip() else f"Find academic research papers and surveys about {clean_q}"
            text = _call_exa_mcp("web_search_exa", {
                "query": clean_q,
                "objective": obj,
                "numResults": clamped_n,
            })

        if not text:
            return "NO RESULTS"
        return _redact_secrets(text)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {_redact_secrets(str(exc))}"


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        clean_url = url.strip()
        if not clean_url:
            return "NO RESULTS"

        if os.getenv("TAVILY_API_KEY"):
            text = _call_tavily_fetch(clean_url)
        else:
            text = _call_exa_mcp("web_fetch_exa", {"urls": [clean_url]})

        if not text:
            return "NO RESULTS"
        truncated = text[:12000]
        return _redact_secrets(truncated)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {_redact_secrets(str(exc))}"


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n", flush=True)
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n", flush=True)
