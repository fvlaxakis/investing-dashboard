# -*- coding: utf-8 -*-
"""
Επενδυτικό Dashboard — βοηθός απόφασης (ΟΧΙ επίσημη επενδυτική συμβουλή).

Live τιμές & γραφήματα από Yahoo Finance (δωρεάν), τεχνικοί & θεμελιώδεις
δείκτες, ένα διαφανές σκορ ανά μετοχή, και σχεδιασμός κατανομής κεφαλαίου.

Τρέξε τοπικά:      py -m streamlit run app.py
Δείχνει στο κινητό: κάνε deploy δωρεάν στο Streamlit Community Cloud (δες README).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Επενδυτικό Dashboard", page_icon="📈", layout="wide")

# ----------------------------------------------------------------------------
# Ρυθμίσεις / προεπιλογές
# ----------------------------------------------------------------------------
DEFAULT_WATCHLIST = "AAPL, MSFT, NVDA, VUAA.DE, IWDA.AS, ASML.AS"
# VUAA.DE = S&P 500 ETF (EUR, Xetra) · IWDA.AS = MSCI World ETF (Amsterdam)


# ----------------------------------------------------------------------------
# Δείκτες (υπολογισμένοι με pandas — χωρίς εξωτερικές βιβλιοθήκες TA)
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


@st.cache_data(ttl=900, show_spinner=False)
def load_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df.empty:
        return df
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["RSI"] = rsi(df["Close"])
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["Close"])
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_info(ticker: str) -> dict:
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
        "quote_type": (info.get("quoteType") or "").upper(),  # EQUITY / ETF / MUTUALFUND
    }


# ----------------------------------------------------------------------------
# Φορολογικός χαρακτηρισμός για φορολογικό κάτοικο Ελλάδας (ΙΔΙΩΤΗ)
# Βάση: Ν.4172/2013 άρθρα 42-43. ΔΕΝ είναι φορολογική συμβουλή — heuristic.
#   • Μετοχές εισηγμένες, κατοχή <0,5% (πάντα ισχύει για ιδιώτη): υπεραξία 0%
#   • UCITS ETF (ευρωπαϊκά, IE/LU domicile): υπεραξία 0% για ιδιώτες
#       - accumulating (χωρίς μέρισμα) => και 0% σε μερίσματα => πλήρως αφορολόγητο
#       - distributing => 5% στο μέρισμα
#   • US ETF: συνήθως μη διαθέσιμα σε EU retail (PRIIPs) — τα σημειώνουμε κόκκινα
# ----------------------------------------------------------------------------
_EU_SUFFIXES = (".DE", ".AS", ".L", ".MI", ".PA", ".AT", ".BR", ".MC",
                ".SW", ".VI", ".HE", ".ST", ".OL", ".CO", ".LS", ".IR", ".F")


def tax_profile(ticker: str, info: dict):
    """Επιστρέφει (tier, badge, note). tier: 'green' | 'yellow' | 'red'."""
    qt = info.get("quote_type", "")
    dy = info.get("dividend_yield") or 0
    has_div = bool(dy and dy > 0)
    european = ticker.upper().endswith(_EU_SUFFIXES)

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
# Διαφανές σκορ σήματος (heuristic — ΟΧΙ εγγύηση)
# ----------------------------------------------------------------------------
def compute_signal(df: pd.DataFrame):
    """Επιστρέφει (score 0-100, label, λίστα από λόγους)."""
    if df.empty or len(df) < 200:
        return 50, "Ανεπαρκή δεδομένα", ["Δεν υπάρχει αρκετό ιστορικό για ανάλυση."]

    last = df.iloc[-1]
    price = last["Close"]
    reasons = []
    score = 50  # ουδέτερο ξεκίνημα

    # 1) Τάση: τιμή vs κινητοί μέσοι
    if price > last["SMA50"] > last["SMA200"]:
        score += 20
        reasons.append("📈 Ανοδική τάση: τιμή > MA50 > MA200.")
    elif price < last["SMA50"] < last["SMA200"]:
        score -= 20
        reasons.append("📉 Καθοδική τάση: τιμή < MA50 < MA200.")
    else:
        reasons.append("➖ Μικτή τάση στους κινητούς μέσους.")

    # 2) RSI (υπεραγορασμένο / υπερπουλημένο)
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

    # 3) MACD
    if last["MACD"] > last["MACD_signal"]:
        score += 10
        reasons.append("🟢 MACD πάνω από τη γραμμή σήματος (θετική ορμή).")
    else:
        score -= 10
        reasons.append("🔴 MACD κάτω από τη γραμμή σήματος (αρνητική ορμή).")

    # 4) Απόδοση 6 μηνών
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


def fmt_money(x, cur=""):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if abs(x) >= 1e9:
        return f"{x/1e9:.1f}B {cur}".strip()
    if abs(x) >= 1e6:
        return f"{x/1e6:.1f}M {cur}".strip()
    return f"{x:,.2f} {cur}".strip()


# ----------------------------------------------------------------------------
# 📰 Νέα ανά μετοχή
# ----------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def load_news(ticker: str, limit: int = 6) -> list:
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    out = []
    for item in raw[:limit]:
        # yfinance επιστρέφει είτε flat dict είτε {'content': {...}}
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


# ----------------------------------------------------------------------------
# 💱 Μετατροπή σε EUR (για το χαρτοφυλάκιο)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fx_to_eur(currency: str) -> float:
    """Πόσα EUR κάνει 1 μονάδα του νομίσματος. EUR->1.0."""
    cur = (currency or "EUR").upper()
    if cur in ("EUR", ""):
        return 1.0
    # Yahoo: 'USDEUR=X' = EUR ανά 1 USD
    try:
        hist = yf.Ticker(f"{cur}EUR=X").history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return 1.0  # fallback: αν αποτύχει, μη μετατρέπεις


# ----------------------------------------------------------------------------
# 🔔 Market Radar — ανιχνευτής ασυνήθιστων κινήσεων
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

    # 1) Ασυνήθιστος όγκος
    if "Volume" in df and df["Volume"].iloc[-20:].mean() > 0:
        avg_vol = df["Volume"].iloc[-21:-1].mean()
        if avg_vol > 0 and last["Volume"] > 2 * avg_vol:
            mult = last["Volume"] / avg_vol
            arrow = "🟢📈" if day_ret >= 0 else "🔴📉"
            alerts.append(
                ("info", f"{arrow} Ασυνήθιστος όγκος {mult:.1f}× του μέσου "
                         f"(τιμή {day_ret*100:+.1f}%) — συνήθως υπάρχει είδηση.")
            )

    # 2) Απότομη κίνηση σε σχέση με τη μεταβλητότητα (z-score)
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

    # 3) 52-εβδομάδων high / low
    window = df["Close"].iloc[-252:]
    hi, lo = window.max(), window.min()
    if close >= hi * 0.995:
        alerts.append(("green", "🟢 Νέο υψηλό 52 εβδομάδων — δυνατή ανοδική ορμή."))
    elif close <= lo * 1.005:
        alerts.append(("red", "🔴 Νέο χαμηλό 52 εβδομάδων — προσοχή."))

    # 4) Golden / Death cross (MA50 x MA200 τις τελευταίες ~5 μέρες)
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

    # 5) RSI ακραίο
    r = last["RSI"]
    if not np.isnan(r):
        if r >= 75:
            alerts.append(("red", f"🔴 RSI {r:.0f} — έντονα υπεραγορασμένο."))
        elif r <= 25:
            alerts.append(("green", f"🟢 RSI {r:.0f} — έντονα υπερπουλημένο "
                                    "(πιθανή ευκαιρία)."))
    return alerts


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("📈 Επενδυτικό Dashboard")
st.caption(
    "Βοηθός απόφασης με δεδομένα & δείκτες — **ΔΕΝ** είναι επίσημη επενδυτική "
    "συμβουλή. Δεδομένα: Yahoo Finance (καθυστέρηση ~15').  Οι εντολές γίνονται "
    "χειροκίνητα στο DeGiro."
)
st.warning(
    "⚠️ **Η διαθεσιμότητα στο DeGiro δεν είναι επιβεβαιωμένη αυτόματα.** Τα "
    "δεδομένα είναι από Yahoo Finance, που έχει περισσότερα προϊόντα απ' όσα "
    "προσφέρει το DeGiro. **Πριν αγοράσεις, ψάξε το προϊόν στο ίδιο το DeGiro** "
    "(κατά προτίμηση με ISIN). Δες τον οδηγό ↓."
)
with st.expander("🔎 Πώς επιβεβαιώνω ένα προϊόν στο DeGiro (& σίγουρα αφορολόγητα UCITS ETF)"):
    st.markdown(
        "**Βήματα επιβεβαίωσης:**\n"
        "1. Στο DeGiro, χρησιμοποίησε την μπάρα **αναζήτησης** πάνω-πάνω.\n"
        "2. Ψάξε με **ISIN** (πιο σίγουρο) ή με το όνομα της εταιρείας/ETF.\n"
        "3. Αν εμφανιστεί → είναι διαθέσιμο. Τσέκαρε το **χρηματιστήριο** "
        "(π.χ. Xetra/Euronext Amsterdam) και τυχόν προμήθεια.\n"
        "4. Για ETF, βεβαιώσου ότι λέει **UCITS** και (για μηδενικό φόρο) "
        "**Accumulating / Acc**.\n\n"
        "**Βασικά αφορολόγητα UCITS ETF — επιβεβαιωμένα ISIN** (σχεδόν σίγουρα "
        "στο DeGiro):"
    )
    st.table(pd.DataFrame([
        {"ETF": "Vanguard S&P 500 (Acc)", "Σύμβολο": "VUAA", "ISIN": "IE00BFMXXD54"},
        {"ETF": "iShares Core S&P 500 (Acc)", "Σύμβολο": "SXR8 / CSPX", "ISIN": "IE00B5BMR087"},
        {"ETF": "iShares Core MSCI World (Acc)", "Σύμβολο": "IWDA / EUNL", "ISIN": "IE00B4L5Y983"},
        {"ETF": "Vanguard FTSE All-World (Acc)", "Σύμβολο": "VWCE", "ISIN": "IE00BK5BQT80"},
    ]))
    st.caption(
        "Το Yahoo δίνει αναξιόπιστα/λάθος ISIN, γι' αυτό δεν εμφανίζονται αυτόματα "
        "στους πίνακες. Αυτά τα 4 είναι επιβεβαιωμένα χειροκίνητα."
    )

with st.sidebar:
    st.header("⚙️ Ρυθμίσεις")
    capital = st.number_input("Διαθέσιμο κεφάλαιο (€)", value=40000, step=1000, min_value=0)
    watchlist_raw = st.text_area(
        "Watchlist (σύμβολα, χωρισμένα με κόμμα)",
        value=DEFAULT_WATCHLIST,
        help="Χρησιμοποίησε σύμβολα Yahoo Finance. π.χ. AAPL, VUAA.DE, ASML.AS",
    )
    period = st.select_slider(
        "Ιστορικό γραφήματος", options=["6mo", "1y", "2y", "5y"], value="2y"
    )
    st.markdown("---")
    tax_filter = st.radio(
        "🧾 Φορολογικό φίλτρο (ιδιώτης ΦΚ Ελλάδας)",
        options=["Όλα", "Μόνο αφορολόγητη υπεραξία", "Μόνο πλήρως αφορολόγητα"],
        index=0,
        help="Πλήρως αφορολόγητα = accumulating UCITS ETF ή μετοχές χωρίς μέρισμα.",
    )
    st.markdown("---")
    st.markdown(
        "**Σκορ:** 0–100 από τάση + RSI + MACD + ορμή. Είναι διαφανές heuristic, "
        "όχι πρόβλεψη. Πάντα διασταύρωσε και διαφοροποίησε το ρίσκο."
    )
    st.caption(
        "⚠️ Ο φορολογικός χαρακτηρισμός είναι ενδεικτικός (Ν.4172/2013 άρθρ. 42-43). "
        "Δεν είναι φορολογική συμβουλή — επιβεβαίωσε με φοροτεχνικό."
    )

tickers = [t.strip().upper() for t in watchlist_raw.split(",") if t.strip()]

tab_overview, tab_detail, tab_portfolio = st.tabs(
    ["🏠 Επισκόπηση & Κατανομή", "🔍 Ανάλυση μετοχής", "💼 Χαρτοφυλάκιο"]
)

# --- Επισκόπηση: πίνακας σκορ όλων + πρόταση κατανομής ------------------------
with tab_overview:
    rows = []
    signals = {}
    all_alerts = []  # (ticker, severity, text)
    with st.spinner("Φόρτωση δεδομένων..."):
        for t in tickers:
            df = load_history(t, period="2y")
            info = load_info(t)
            score, label, reasons = compute_signal(df)
            signals[t] = (score, label, reasons)
            for sev, txt in scan_alerts(t, df):
                all_alerts.append((t, sev, txt))
            price = df["Close"].iloc[-1] if not df.empty else np.nan
            chg = (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100 if len(df) > 1 else np.nan
            tier, badge, note = tax_profile(t, info)
            rows.append(
                {
                    "Σύμβολο": t,
                    "Όνομα": info["name"],
                    "Τιμή": price,
                    "Ημ. %": chg,
                    "Σκορ": score,
                    "Σήμα": label,
                    "Φόρος": badge,
                    "_tier": tier,
                    "P/E": info["pe"],
                    "Μέρισμα %": info["dividend_yield"] if info["dividend_yield"] else None,
                    "Κλάδος": info["sector"],
                }
            )

    # --- 🔔 Market Radar -----------------------------------------------------
    st.subheader("🔔 Market Radar")
    if all_alerts:
        st.caption(f"{len(all_alerts)} ασυνήθιστες κινήσεις εντοπίστηκαν στη watchlist:")
        # Ταξινόμηση: κόκκινα πρώτα, μετά πράσινα, μετά info
        order = {"red": 0, "green": 1, "info": 2}
        for t, sev, txt in sorted(all_alerts, key=lambda a: order.get(a[1], 3)):
            line = f"**{t}** — {txt}"
            if sev == "red":
                st.error(line)
            elif sev == "green":
                st.success(line)
            else:
                st.info(line)
    else:
        st.caption("✅ Τίποτα ασυνήθιστο αυτή τη στιγμή στη watchlist — ήρεμα νερά.")
    st.divider()

    if rows:
        table = pd.DataFrame(rows).sort_values("Σκορ", ascending=False)

        # Εφαρμογή φορολογικού φίλτρου
        if tax_filter == "Μόνο αφορολόγητη υπεραξία":
            table = table[table["_tier"].isin(["green", "yellow"])]
        elif tax_filter == "Μόνο πλήρως αφορολόγητα":
            table = table[table["_tier"] == "green"]

        st.subheader("Κατάταξη watchlist")
        if table.empty:
            st.warning(
                "Κανένα προϊόν της watchlist δεν περνά αυτό το φορολογικό φίλτρο. "
                "Δοκίμασε accumulating UCITS ETF (π.χ. VUAA.DE, SXR8.DE, IWDA.AS) "
                "ή μετοχές χωρίς μέρισμα."
            )
        else:
            show = table.drop(columns=["_tier"])
            st.dataframe(
                show.style.format(
                    {"Τιμή": "{:.2f}", "Ημ. %": "{:+.2f}", "P/E": "{:.1f}", "Μέρισμα %": "{:.2f}"},
                    na_rep="—",
                ).background_gradient(subset=["Σκορ"], cmap="RdYlGn", vmin=0, vmax=100),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "🟢 Πλήρως αφορολόγητο · 🟡 Υπεραξία 0% αλλά μέρισμα 5% · "
                "🔴 Πιθανό 15% ή μη διαθέσιμο σε EU retail."
            )

        # --- Πρόταση κατανομής βασισμένη στο σκορ ----------------------------
        st.subheader("💡 Ιδέα κατανομής κεφαλαίου")
        st.caption(
            "Απλή κατανομή σταθμισμένη με το σκορ, μόνο σε μετοχές με σήμα ≥ 'Ουδέτερο'. "
            "Δείχνει *πώς* θα μπορούσες να μοιράσεις το ρίσκο — όχι εντολή αγοράς."
        )
        elig = table[table["Σκορ"] >= 45].copy()
        if not elig.empty:
            weights = elig["Σκορ"] / elig["Σκορ"].sum()
            elig["Ποσό €"] = (weights * capital).round(0)
            elig["Ποσοστό %"] = (weights * 100).round(1)
            c1, c2 = st.columns([1, 1])
            with c1:
                st.dataframe(
                    elig[["Σύμβολο", "Όνομα", "Ποσοστό %", "Ποσό €"]],
                    use_container_width=True,
                    hide_index=True,
                )
            with c2:
                pie = go.Figure(
                    go.Pie(labels=elig["Σύμβολο"], values=elig["Ποσό €"], hole=0.4)
                )
                pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
                st.plotly_chart(pie, use_container_width=True)
            st.info(
                f"⚠️ Εμπειρικός κανόνας ρίσκου: μην βάζεις πάνω από ~20–25% "
                f"(≈{capital*0.22:,.0f}€) σε μία μόνο θέση, και κράτα ένα ποσοστό "
                "σε μετρητά/ETF ευρείας αγοράς για σταθερότητα."
            )
        else:
            st.warning("Καμία μετοχή δεν έχει σήμα ≥ Ουδέτερο αυτή τη στιγμή.")
    else:
        st.info("Πρόσθεσε σύμβολα στη watchlist από το πλαϊνό μενού.")

# --- Λεπτομερής ανάλυση μίας μετοχής ----------------------------------------
with tab_detail:
    if not tickers:
        st.info("Πρόσθεσε σύμβολα στη watchlist.")
    else:
        sel = st.selectbox("Διάλεξε μετοχή", tickers)
        df = load_history(sel, period=period)
        info = load_info(sel)
        if df.empty:
            st.error(f"Δεν βρέθηκαν δεδομένα για {sel}. Έλεγξε το σύμβολο.")
        else:
            score, label, reasons = compute_signal(load_history(sel, period="2y"))
            last = df.iloc[-1]
            cur = info["currency"]

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Τιμή", f"{last['Close']:.2f} {cur}")
            k2.metric("Σκορ σήματος", f"{score}/100", label)
            k3.metric("P/E", f"{info['pe']:.1f}" if info["pe"] else "—")
            k4.metric(
                "Μέρισμα",
                f"{info['dividend_yield']:.2f}%" if info["dividend_yield"] else "—",
            )

            tier, badge, note = tax_profile(sel, info)
            tier_icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}[tier]
            st.markdown(f"**Φορολογία (ιδιώτης ΦΚ Ελλάδας):** {tier_icon} {badge}")
            st.caption(note)

            with st.expander("🧠 Γιατί αυτό το σκορ;", expanded=True):
                for rline in reasons:
                    st.write("•", rline)
                st.caption(
                    "Θυμήσου: δείκτες = ενδείξεις, όχι βεβαιότητα. Καμία στρατηγική "
                    "δεν κερδίζει πάντα."
                )

            # Γράφημα τιμής + κινητοί μέσοι
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name=sel,
            ))
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="MA50", line=dict(width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="MA200", line=dict(width=1)))
            fig.update_layout(
                height=420, margin=dict(t=20, b=10, l=10, r=10),
                xaxis_rangeslider_visible=False, legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                rfig = go.Figure()
                rfig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
                rfig.add_hline(y=70, line_dash="dash", line_color="red")
                rfig.add_hline(y=30, line_dash="dash", line_color="green")
                rfig.update_layout(title="RSI (14)", height=250, margin=dict(t=30, b=10, l=10, r=10))
                st.plotly_chart(rfig, use_container_width=True)
            with c2:
                mfig = go.Figure()
                mfig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Histogram"))
                mfig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
                mfig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal"))
                mfig.update_layout(title="MACD", height=250, margin=dict(t=30, b=10, l=10, r=10))
                st.plotly_chart(mfig, use_container_width=True)

            # 🔔 Ειδοποιήσεις radar για τη συγκεκριμένη μετοχή
            sel_alerts = scan_alerts(sel, load_history(sel, period="2y"))
            if sel_alerts:
                st.markdown("**🔔 Ασυνήθιστες κινήσεις:**")
                for sev, txt in sel_alerts:
                    (st.error if sev == "red" else st.success if sev == "green" else st.info)(txt)

            # 📰 Νέα
            st.markdown("**📰 Πρόσφατα νέα**")
            news = load_news(sel)
            if news:
                for n in news:
                    pub = f" — *{n['publisher']}*" if n["publisher"] else ""
                    if n["link"]:
                        st.markdown(f"- [{n['title']}]({n['link']}){pub}")
                    else:
                        st.markdown(f"- {n['title']}{pub}")
            else:
                st.caption("Δεν βρέθηκαν πρόσφατα νέα για αυτό το σύμβολο.")

# --- 💼 Χαρτοφυλάκιο: πραγματικές αγορές, κέρδος/ζημιά, καθαρό μετά φόρου -----
with tab_portfolio:
    st.subheader("💼 Το χαρτοφυλάκιό μου")
    st.caption(
        "Βάλε τις αγορές σου: σύμβολο, ποσότητα (τεμάχια) και μέση τιμή αγοράς "
        "(στο νόμισμα της μετοχής). Τα κενά αγνοούνται. Οι τιμές είναι live· τα "
        "σύνολα μετατρέπονται σε € αυτόματα."
    )
    seed = pd.DataFrame(
        [{"Σύμβολο": "", "Ποσότητα": None, "Τιμή αγοράς": None}]
    )
    edited = st.data_editor(
        seed, num_rows="dynamic", use_container_width=True, hide_index=True,
        column_config={
            "Σύμβολο": st.column_config.TextColumn(help="π.χ. VUAA.DE, AAPL"),
            "Ποσότητα": st.column_config.NumberColumn(format="%.4f", min_value=0),
            "Τιμή αγοράς": st.column_config.NumberColumn(format="%.2f", min_value=0),
        },
        key="portfolio_editor",
    )

    prows, tot_val_eur, tot_cost_eur, tot_div_eur = [], 0.0, 0.0, 0.0
    for _, r in edited.iterrows():
        sym = str(r["Σύμβολο"]).strip().upper()
        qty, buy = r["Ποσότητα"], r["Τιμή αγοράς"]
        if not sym or not qty or qty <= 0:
            continue
        df = load_history(sym, period="6mo")
        info = load_info(sym)
        if df.empty:
            prows.append({"Σύμβολο": sym, "Κατάσταση": "❌ άγνωστο σύμβολο"})
            continue
        cur = info["currency"] or "EUR"
        price = float(df["Close"].iloc[-1])
        rate = fx_to_eur(cur)
        value_eur = price * qty * rate
        cost_eur = (buy or 0) * qty * rate
        pl_eur = value_eur - cost_eur
        pl_pct = (price / buy - 1) * 100 if buy else np.nan
        # Ετήσιο μέρισμα (φορ. 5%) — dividend_yield είναι ήδη %
        dy = info["dividend_yield"] or 0
        div_eur = value_eur * (dy / 100)
        tot_val_eur += value_eur
        tot_cost_eur += cost_eur
        tot_div_eur += div_eur
        tier, badge, _ = tax_profile(sym, info)
        prows.append({
            "Σύμβολο": sym,
            "Ποσότητα": qty,
            "Τιμή τώρα": f"{price:.2f} {cur}",
            "Αξία €": round(value_eur, 2),
            "Κέρδος/Ζημιά €": round(pl_eur, 2),
            "Απόδοση %": round(pl_pct, 1) if not np.isnan(pl_pct) else None,
            "Φόρος υπεραξίας": "0% ✅" if tier in ("green", "yellow") else "15%; ⚠️",
        })

    if any("Αξία €" in p for p in prows):
        st.dataframe(
            pd.DataFrame(prows),
            use_container_width=True, hide_index=True,
        )
        pl_total = tot_val_eur - tot_cost_eur
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Συνολική αξία", f"{tot_val_eur:,.0f} €")
        m2.metric("Κόστος", f"{tot_cost_eur:,.0f} €")
        m3.metric(
            "Κέρδος/Ζημιά", f"{pl_total:,.0f} €",
            f"{(pl_total/tot_cost_eur*100):+.1f}%" if tot_cost_eur else None,
        )
        m4.metric("Ετήσιο μέρισμα (~)", f"{tot_div_eur:,.0f} €",
                  f"-{tot_div_eur*0.05:,.0f} € φόρος 5%")
        st.success(
            f"💡 Καλά νέα: το κέρδος από την πώληση (υπεραξία) των μετοχών & UCITS "
            f"ETF είναι **αφορολόγητο** για σένα → τα **{pl_total:,.0f} €** κέρδος "
            "είναι καθαρά. Φόρος πληρώνεις μόνο ~5% στα μερίσματα (πάνω)."
        )
        st.caption(
            "⚠️ Ενδεικτικός υπολογισμός. Οι τιμές έχουν καθυστέρηση ~15'. Το "
            "χαρτοφυλάκιο δεν αποθηκεύεται μόνιμα — αν κλείσεις το app, ξαναβάζεις "
            "τις γραμμές (μπορώ να προσθέσω μόνιμη αποθήκευση αργότερα)."
        )
    else:
        st.info("Πρόσθεσε γραμμές παραπάνω για να δεις κέρδος/ζημιά και σύνολα.")

st.markdown("---")
st.caption(
    "Εκπαιδευτικό εργαλείο. Οι επενδύσεις έχουν κίνδυνο απώλειας κεφαλαίου. "
    "Δεν αποτελεί επενδυτική συμβουλή."
)
