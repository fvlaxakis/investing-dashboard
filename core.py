# -*- coding: utf-8 -*-
"""
core.py — Καθαρή λογική (δεδομένα, δείκτες, σκορ, φόρος, ειδοποιήσεις).

ΔΕΝ εξαρτάται από το Streamlit, ώστε να μπορεί να τη χρησιμοποιεί ΚΑΙ το app.py
(dashboard) ΚΑΙ το scan_email.py (πρωινό email μέσω GitHub Actions).
"""

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ----------------------------------------------------------------------------
# Supabase (πραγματικό sync χαρτοφυλακίου cross-device μέσω REST/PostgREST)
# Πίνακας: portfolio(id bigint identity PK, symbol text, qty numeric, buy_price numeric)
# ----------------------------------------------------------------------------
def supabase_load(url: str, key: str) -> list:
    """Διαβάζει όλες τις θέσεις. Επιστρέφει list από dicts symbol/qty/buy_price."""
    r = requests.get(
        f"{url}/rest/v1/portfolio",
        params={"select": "symbol,qty,buy_price", "order": "id"},
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
        timeout=12,
    )
    r.raise_for_status()
    return r.json()


def supabase_save(url: str, key: str, records: list) -> None:
    """Αντικαθιστά όλο το χαρτοφυλάκιο (delete-all + insert)."""
    h = {"apikey": key, "Authorization": f"Bearer {key}",
         "Content-Type": "application/json"}
    # PostgREST: χρειάζεται φίλτρο για μαζικό delete — id>0 πιάνει τα πάντα
    requests.delete(f"{url}/rest/v1/portfolio", params={"id": "gt.0"},
                    headers=h, timeout=12).raise_for_status()
    if records:
        requests.post(
            f"{url}/rest/v1/portfolio",
            headers={**h, "Prefer": "return=minimal"},
            json=records, timeout=12,
        ).raise_for_status()


# ----------------------------------------------------------------------------
# Δείκτες
# ----------------------------------------------------------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def fetch_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df.empty:
        return df
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["RSI"] = rsi(df["Close"])
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["Close"])
    return df


def fetch_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}
    return {
        "name": info.get("shortName") or ticker,
        "currency": info.get("currency", ""),
        "pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "dividend_yield": info.get("dividendYield"),
        "market_cap": info.get("marketCap"),
        "sector": info.get("sector", "—"),
        "target_mean": info.get("targetMeanPrice"),
        "quote_type": (info.get("quoteType") or "").upper(),
    }


def fetch_news(ticker: str, limit: int = 6) -> list:
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    out = []
    for item in raw[:limit]:
        c = item.get("content", item)
        title = c.get("title") or item.get("title")
        if not title:
            continue
        link = (
            item.get("link")
            or (c.get("clickThroughUrl") or {}).get("url")
            or (c.get("canonicalUrl") or {}).get("url")
            or ""
        )
        publisher = (
            item.get("publisher")
            or (c.get("provider") or {}).get("displayName")
            or ""
        )
        out.append({"title": title, "link": link, "publisher": publisher})
    return out


def fx_to_eur(currency: str) -> float:
    """Πόσα EUR κάνει 1 μονάδα του νομίσματος. EUR->1.0."""
    cur = (currency or "EUR").upper()
    if cur in ("EUR", ""):
        return 1.0
    try:
        hist = yf.Ticker(f"{cur}EUR=X").history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return 1.0


# ----------------------------------------------------------------------------
# Σκορ σήματος
# ----------------------------------------------------------------------------
def compute_signal(df: pd.DataFrame):
    if df.empty or len(df) < 200:
        return 50, "Ανεπαρκή δεδομένα", ["Δεν υπάρχει αρκετό ιστορικό για ανάλυση."]

    last = df.iloc[-1]
    price = last["Close"]
    reasons = []
    score = 50

    if price > last["SMA50"] > last["SMA200"]:
        score += 20
        reasons.append("📈 Ανοδική τάση: τιμή > MA50 > MA200.")
    elif price < last["SMA50"] < last["SMA200"]:
        score -= 20
        reasons.append("📉 Καθοδική τάση: τιμή < MA50 < MA200.")
    else:
        reasons.append("➖ Μικτή τάση στους κινητούς μέσους.")

    r = last["RSI"]
    if not np.isnan(r):
        if r < 30:
            score += 10
            reasons.append(f"🟢 RSI {r:.0f} — υπερπουλημένο (πιθανή ευκαιρία).")
        elif r > 70:
            score -= 10
            reasons.append(f"🔴 RSI {r:.0f} — υπεραγορασμένο (προσοχή).")
        else:
            reasons.append(f"➖ RSI {r:.0f} — ουδέτερη ζώνη.")

    if last["MACD"] > last["MACD_signal"]:
        score += 10
        reasons.append("🟢 MACD πάνω από τη γραμμή σήματος (θετική ορμή).")
    else:
        score -= 10
        reasons.append("🔴 MACD κάτω από τη γραμμή σήματος (αρνητική ορμή).")

    if len(df) > 126:
        ret6m = price / df["Close"].iloc[-126] - 1
        if ret6m > 0.10:
            score += 10
            reasons.append(f"🟢 Απόδοση 6μήνου +{ret6m*100:.0f}%.")
        elif ret6m < -0.10:
            score -= 10
            reasons.append(f"🔴 Απόδοση 6μήνου {ret6m*100:.0f}%.")

    score = int(max(0, min(100, score)))
    if score >= 65:
        label = "Ισχυρό"
    elif score >= 45:
        label = "Ουδέτερο"
    else:
        label = "Αδύναμο"
    return score, label, reasons


# ----------------------------------------------------------------------------
# Φορολογικός χαρακτηρισμός (ιδιώτης ΦΚ Ελλάδας — Ν.4172/2013 άρθρ. 42-43)
# ----------------------------------------------------------------------------
EU_SUFFIXES = (".DE", ".AS", ".L", ".MI", ".PA", ".AT", ".BR", ".MC",
               ".SW", ".VI", ".HE", ".ST", ".OL", ".CO", ".LS", ".IR", ".F")


def tax_profile(ticker: str, info: dict):
    """Επιστρέφει (tier, badge, note). tier: 'green' | 'yellow' | 'red'."""
    qt = info.get("quote_type", "")
    dy = info.get("dividend_yield") or 0
    has_div = bool(dy and dy > 0)
    european = ticker.upper().endswith(EU_SUFFIXES)

    if qt in ("ETF", "MUTUALFUND"):
        if not european:
            return ("red", "⛔ US/μη-UCITS",
                    "Πιθανώς μη διαθέσιμο σε EU retail (PRIIPs). Ψάξε το UCITS "
                    "αντίστοιχο σε ευρωπαϊκό χρηματιστήριο (π.χ. .DE/.AS).")
        if has_div:
            return ("yellow", "✅ Υπεραξία 0% · ⚠️ Μέρισμα 5%",
                    "UCITS distributing: κέρδος πώλησης αφορολόγητο, μέρισμα 5%. "
                    "Για μηδενικό φόρο ψάξε την 'Acc' (accumulating) έκδοση.")
        return ("green", "✅✅ Πλήρως αφορολόγητο",
                "Accumulating UCITS: χωρίς μέρισμα (0%) & υπεραξία αφορολόγητη.")

    if qt == "EQUITY":
        if has_div:
            return ("yellow", "✅ Υπεραξία 0% · ⚠️ Μέρισμα 5%",
                    "Εισηγμένη μετοχή, κατοχή <0,5%: κέρδος πώλησης αφορολόγητο· "
                    "το μέρισμα φορολογείται 5%.")
        return ("green", "✅✅ Πλήρως αφορολόγητο",
                "Μετοχή χωρίς μέρισμα, κατοχή <0,5%: υπεραξία αφορολόγητη, "
                "χωρίς μέρισμα να φορολογηθεί.")

    return ("red", "⚠️ Έλεγξε (15%;)",
            "Ομόλογο/άγνωστος τύπος: η υπεραξία μπορεί να φορολογείται 15%. "
            "Επιβεβαίωσε με φοροτεχνικό.")


# ----------------------------------------------------------------------------
# Market Radar — ανιχνευτής ασυνήθιστων κινήσεων
# ----------------------------------------------------------------------------
def scan_alerts(ticker: str, df: pd.DataFrame) -> list:
    """Επιστρέφει λίστα από (severity, text). severity: 'red'|'green'|'info'."""
    if df.empty or len(df) < 60:
        return []
    alerts = []
    last = df.iloc[-1]
    close = last["Close"]
    prev = df["Close"].iloc[-2]
    day_ret = close / prev - 1

    if "Volume" in df and df["Volume"].iloc[-20:].mean() > 0:
        avg_vol = df["Volume"].iloc[-21:-1].mean()
        if avg_vol > 0 and last["Volume"] > 2 * avg_vol:
            mult = last["Volume"] / avg_vol
            arrow = "🟢📈" if day_ret >= 0 else "🔴📉"
            alerts.append(
                ("info", f"{arrow} Ασυνήθιστος όγκος {mult:.1f}× του μέσου "
                         f"(τιμή {day_ret*100:+.1f}%) — συνήθως υπάρχει είδηση.")
            )

    daily = df["Close"].pct_change().iloc[-21:]
    std = daily.std()
    if std and std > 0:
        z = day_ret / std
        if z <= -2.5:
            alerts.append(("red", f"🔴 Απότομη πτώση {day_ret*100:+.1f}% "
                                  f"({abs(z):.1f}× πάνω από το κανονικό)."))
        elif z >= 2.5:
            alerts.append(("green", f"🟢 Απότομη άνοδος {day_ret*100:+.1f}% "
                                    f"({z:.1f}× πάνω από το κανονικό)."))

    window = df["Close"].iloc[-252:]
    hi, lo = window.max(), window.min()
    if close >= hi * 0.995:
        alerts.append(("green", "🟢 Νέο υψηλό 52 εβδομάδων — δυνατή ανοδική ορμή."))
    elif close <= lo * 1.005:
        alerts.append(("red", "🔴 Νέο χαμηλό 52 εβδομάδων — προσοχή."))

    if not np.isnan(last["SMA50"]) and not np.isnan(last["SMA200"]):
        recent = df.iloc[-6:]
        above = recent["SMA50"] > recent["SMA200"]
        if above.iloc[0] != above.iloc[-1]:
            if above.iloc[-1]:
                alerts.append(("green", "🟢 Golden Cross: ο MA50 πέρασε πάνω "
                                        "από τον MA200 (μακροπρόθεσμα θετικό)."))
            else:
                alerts.append(("red", "🔴 Death Cross: ο MA50 πέρασε κάτω από "
                                      "τον MA200 (μακροπρόθεσμα αρνητικό)."))

    r = last["RSI"]
    if not np.isnan(r):
        if r >= 75:
            alerts.append(("red", f"🔴 RSI {r:.0f} — έντονα υπεραγορασμένο."))
        elif r <= 25:
            alerts.append(("green", f"🟢 RSI {r:.0f} — έντονα υπερπουλημένο "
                                    "(πιθανή ευκαιρία)."))
    return alerts


# ----------------------------------------------------------------------------
# Σύμπαν υποψηφίων για τον Screener (φορολογικά αποδοτικά, συνήθως στο DeGiro)
# ----------------------------------------------------------------------------
SCREENER_UNIVERSE = [
    # Accumulating UCITS ETF (πλήρως αφορολόγητα)
    "VUAA.DE", "SXR8.DE", "IWDA.AS", "EUNL.DE", "VWCE.DE", "SXRV.DE",
    "IS3N.DE", "MEUD.PA", "XDWD.DE", "SPPW.DE",
    # Μετοχές χαμηλού/μηδενικού μερίσματος (υπεραξία αφορολόγητη)
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "ASML.AS",
    "AMD", "NFLX", "ADBE", "CRM", "NVO", "LVMH.PA", "SAP.DE",
]
