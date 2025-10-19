# os_probe.py

import httpx

from lexical import norm_tokens

def probe_has_hits(query: str, os_url: str, index: str, timeout: float = 0.4) -> bool:
    """
    POST /{index}/_search with size:0 using query_string.
    Returns True if total hits > 0. Swallows errors -> False (fail closed).
    """
    q = " ".join(norm_tokens(query))[:512] or "*"
    url = f"{os_url.rstrip('/')}/{index}/_search"
    payload = {"size": 0, "query": {"query_string": {"query": q}}}
    try:
        r = httpx.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        total = r.json().get("hits", {}).get("total", 0)
        if isinstance(total, dict):  # OS7+ style
            return int(total.get("value", 0)) > 0
        return int(total) > 0
    except Exception:
        return False
