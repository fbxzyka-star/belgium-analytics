"""
Belgium Analytics - AI research step.

Runs on GitHub Actions every 30 minutes:
  1. Gets the current FX rates (Coinbase).
  2. Asks Gemini (with Google Search) WHY each rate moved today.
  3. Saves the answer to research.json, which index.html shows.

Only the Python standard library is used, so nothing needs installing.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

# ------------------------------------------------------------
# Change these lines to change the research
# ------------------------------------------------------------
# Tried in this order; if one has no free quota, the next is used
MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]   # free tier: 20 calls/day each
LANGUAGE = "Greek"                 # e.g. "English" for clients
PAIRS = ["USD", "GBP", "CHF"]      # currencies against EUR

QUESTION = """
You are an FX analyst. Today is {date}.
Current rates (1 EUR =): {rates}

Search the news of the last 24 hours and explain WHY each of these
currency pairs moved today: EUR/USD, EUR/GBP, EUR/CHF.
Mention the concrete events (central bank statements, data releases,
politics, risk sentiment). If nothing important happened, say so.

Answer in {language}. Reply ONLY with JSON, no other text, in this shape:
{{
  "headline": "one short sentence about the day",
  "pairs": [
    {{"pair": "EUR/USD", "direction": "up" | "down" | "flat",
      "reason": "2-3 sentences"}}
  ]
}}
"""
# ------------------------------------------------------------

OUTPUT_FILE = "research.json"


def get_json(url, data=None):
    headers = {"Content-Type": "application/json", "User-Agent": "belgium-analytics"}
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def get_rates():
    data = get_json("https://api.coinbase.com/v2/exchange-rates?currency=EUR")
    all_rates = data["data"]["rates"]
    return {cur: round(float(all_rates[cur]), 4) for cur in PAIRS}


def ask_gemini(prompt, api_key, MODEL):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL}:generateContent?key={api_key}"
    )
    request = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],   # lets Gemini search the web
    }
    reply = get_json(url, request)
    candidate = reply["candidates"][0]

    text = "".join(p.get("text", "") for p in candidate["content"]["parts"])

    # Web pages Gemini used, so visitors can check the facts
    sources = []
    for chunk in candidate.get("groundingMetadata", {}).get("groundingChunks", []):
        web = chunk.get("web")
        if web and web.get("uri"):
            sources.append({"title": web.get("title", "source"), "url": web["uri"]})

    return text, sources[:6]


def parse_json(text):
    # Gemini sometimes wraps the JSON in ```json ... ```
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON in Gemini reply:\n" + text)
    return json.loads(match.group(0))


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY is missing (add it in GitHub Secrets).")

    now = datetime.now(timezone.utc)
    rates = get_rates()

    prompt = QUESTION.format(
        date=now.strftime("%Y-%m-%d"),
        rates=", ".join(f"{cur} {val}" for cur, val in rates.items()),
        language=LANGUAGE,
    )

    errors = []
    for model in MODELS:
        try:
            text, sources = ask_gemini(prompt, api_key, model)
            break
        except urllib.error.HTTPError as e:
            if e.code not in (404, 429):
                raise
            detail = e.read().decode(errors="replace")[:300].replace("\n", " ")
            errors.append(f"{model}: HTTP {e.code} {detail}")
            print("Skipping", model, "->", e.code)
    else:
        sys.exit("No model worked: " + ", ".join(errors))
    result = parse_json(text)

    result["updated"] = now.isoformat(timespec="minutes")
    result["rates"] = rates
    result["sources"] = sources
    result["model"] = model

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Saved", OUTPUT_FILE)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as e:
        # Show Google's real error message (the key is not in it)
        raw = e.read().decode(errors="replace")
        try:
            err = json.loads(raw)["error"]
            details = [d.get("violations", d.get("retryDelay", "")) for d in err.get("details", [])]
            body = f'{err.get("message", "")} | {json.dumps(details)}'[:1500]
        except Exception:
            body = raw[:1500].replace("\n", " ")
        print(f"::error::HTTP {e.code} from {e.url.split('?')[0]}: {body}")
        sys.exit(1)
    except SystemExit as e:
        if e.code not in (0, None):
            print(f"::error::{e.code}")
        raise
    except Exception as e:
        print(f"::error::{type(e).__name__}: {str(e)[:500]}".replace("\n", " "))
        sys.exit(1)
