"""
Belgium Analytics - AI research step.

Runs on GitHub Actions every 2 hours:
  1. Gets the current FX rates (Coinbase) and yesterday's official ECB rates.
     The code (not the AI) decides if each pair went up, down or stayed flat.
  2. Collects REAL news headlines from free RSS feeds (central banks + news).
  3. Gives the headlines to Gemini and asks WHY each pair moved.
     Gemini may only use these headlines, so it cannot invent events.
  4. Saves everything to research.json, which index.html shows.

Only the Python standard library is used, so nothing needs installing.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

# ------------------------------------------------------------
# Change these lines to change the research
# ------------------------------------------------------------
# Tried in this order; if one has no free quota, the next is used
MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
LANGUAGE = "Greek"                 # e.g. "English" for clients
PAIRS = ["USD", "GBP", "CHF"]      # currencies against EUR
NEWS_HOURS = 36                    # only headlines newer than this
FLAT_LIMIT = 0.05                  # % change below this counts as "flat"

# Official central bank feeds (used for every pair)
BANK_FEEDS = {
    "ECB": "https://www.ecb.europa.eu/rss/press.html",
    "Federal Reserve": "https://www.federalreserve.gov/feeds/press_all.xml",
    "Bank of England": "https://www.bankofengland.co.uk/rss/news",
    "Swiss National Bank": "https://www.snb.ch/public/en/rss/news",
}

# News search per pair (Google News RSS, free)
NEWS_QUERY = {
    "USD": "EUR USD euro dollar",
    "GBP": "EUR GBP euro pound sterling",
    "CHF": "EUR CHF euro swiss franc",
}

QUESTION = """
You are an FX analyst. Today is {date}.

Facts computed from real prices (do NOT change them):
{moves}

Real news headlines from the last {hours} hours:
{headlines}

Explain WHY each pair (EUR/USD, EUR/GBP, EUR/CHF) moved, using ONLY the
headlines above. Do not invent events, numbers or quotes. If the headlines
do not explain a move, say clearly that there is no clear news driver.

Answer in {language}. Reply ONLY with JSON, no other text, in this shape:
{{
  "headline": "one short sentence about the day",
  "pairs": [
    {{"pair": "EUR/USD", "reason": "2-3 sentences"}}
  ]
}}
"""
# ------------------------------------------------------------

OUTPUT_FILE = "research.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (belgium-analytics research bot)"}


def fetch(url, data=None):
    headers = dict(HEADERS)
    body = None
    if data is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def get_json(url, data=None):
    return json.loads(fetch(url, data).decode())


# ---------- 1. Prices: the code decides up / down / flat ----------

def get_rates():
    data = get_json("https://api.coinbase.com/v2/exchange-rates?currency=EUR")
    all_rates = data["data"]["rates"]
    return {cur: round(float(all_rates[cur]), 4) for cur in PAIRS}


def get_previous_ecb_rates(today):
    """Official ECB rates of the last working day before today."""
    start = (today - timedelta(days=7)).isoformat()
    url = f"https://api.frankfurter.dev/v1/{start}..?base=EUR&symbols={','.join(PAIRS)}"
    days = get_json(url)["rates"]
    previous = [d for d in sorted(days) if d < today.isoformat()]
    if not previous:
        raise ValueError("No previous ECB rates found")
    date = previous[-1]
    return date, days[date]


def compute_moves(rates, previous):
    moves = {}
    for cur in PAIRS:
        change = (rates[cur] - previous[cur]) / previous[cur] * 100
        if change > FLAT_LIMIT:
            direction = "up"
        elif change < -FLAT_LIMIT:
            direction = "down"
        else:
            direction = "flat"
        moves[cur] = {"change": round(change, 2), "direction": direction}
    return moves


# ---------- 2. Real news headlines ----------

def read_feed(url, source, since):
    """Return [{title, url, source, date}] from an RSS feed, newest only."""
    items = []
    try:
        root = ET.fromstring(fetch(url))
    except Exception as e:
        print(f"Feed skipped ({source}): {e}")
        return items

    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = item.findtext("pubDate")
        if not title or not link.startswith("https://"):
            continue
        try:
            date = parsedate_to_datetime(pub)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if date < since:
            continue
        # Google News puts the outlet name in <source>
        outlet = item.findtext("source") or source
        items.append({"title": title, "url": link, "source": outlet,
                      "date": date.strftime("%d/%m %H:%M")})
    return items


def get_headlines(since):
    news = []
    for name, url in BANK_FEEDS.items():
        news += read_feed(url, name, since)[:4]
    for cur, query in NEWS_QUERY.items():
        q = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={q}+when:2d&hl=en-US&gl=US&ceid=US:en"
        news += read_feed(url, "Google News", since)[:6]

    # remove duplicates (same title)
    seen, unique = set(), []
    for n in news:
        key = n["title"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(n)
    return unique


# ---------- 3. Gemini ----------

def ask_gemini(prompt, api_key, model):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    request = {"contents": [{"parts": [{"text": prompt}]}]}
    reply = get_json(url, request)
    parts = reply["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts)


def ask_any_model(prompt, api_key):
    errors = []
    for model in MODELS:
        try:
            return ask_gemini(prompt, api_key, model), model
        except urllib.error.HTTPError as e:
            if e.code not in (404, 429):
                raise
            detail = e.read().decode(errors="replace")[:300].replace("\n", " ")
            errors.append(f"{model}: HTTP {e.code} {detail}")
            print("Skipping", model, "->", e.code)
    sys.exit("No model worked: " + ", ".join(errors))


def parse_json(text):
    # Gemini sometimes wraps the JSON in ```json ... ```
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON in Gemini reply:\n" + text)
    return json.loads(match.group(0))


# ---------- main ----------

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY is missing (add it in GitHub Secrets).")

    now = datetime.now(timezone.utc)

    rates = get_rates()
    prev_date, prev_rates = get_previous_ecb_rates(now.date())
    moves = compute_moves(rates, prev_rates)

    headlines = get_headlines(now - timedelta(hours=NEWS_HOURS))
    print(f"Found {len(headlines)} headlines")
    if not headlines:
        sys.exit("No news headlines found (all feeds failed?)")

    moves_text = "\n".join(
        f"- EUR/{cur}: now {rates[cur]}, ECB close {prev_date} {prev_rates[cur]}, "
        f"change {m['change']:+.2f}% ({m['direction']})"
        for cur, m in moves.items()
    )
    headlines_text = "\n".join(
        f"- [{n['date']}] {n['source']}: {n['title']}" for n in headlines
    )

    prompt = QUESTION.format(
        date=now.strftime("%Y-%m-%d"),
        moves=moves_text,
        hours=NEWS_HOURS,
        headlines=headlines_text,
        language=LANGUAGE,
    )

    text, model = ask_any_model(prompt, api_key)
    answer = parse_json(text)

    # Direction and % always come from real prices, never from the AI
    reasons = {p.get("pair", ""): p.get("reason", "") for p in answer.get("pairs", [])}
    pairs = []
    for cur in PAIRS:
        name = f"EUR/{cur}"
        pairs.append({
            "pair": name,
            "direction": moves[cur]["direction"],
            "change": moves[cur]["change"],
            "reason": reasons.get(name, ""),
        })

    result = {
        "headline": answer.get("headline", ""),
        "pairs": pairs,
        "updated": now.isoformat(timespec="minutes"),
        "compared_to": f"ECB {prev_date}",
        "rates": rates,
        "sources": [{"title": f"{n['source']}: {n['title']}"[:90], "url": n["url"]}
                    for n in headlines[:8]],
        "model": model,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Saved", OUTPUT_FILE)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:800].replace("\n", " ")
        print(f"::error::HTTP {e.code} from {e.url.split('?')[0]}: {body}")
        sys.exit(1)
    except SystemExit as e:
        if e.code not in (0, None):
            print(f"::error::{e.code}")
        raise
    except Exception as e:
        print(f"::error::{type(e).__name__}: {str(e)[:500]}".replace("\n", " "))
        sys.exit(1)
