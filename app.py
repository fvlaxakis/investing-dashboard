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
# UI
# ----------------------------------------------------------------------------
st.title("📈 Επενδυτικό Dashboard")
st.caption(
    "Βοηθός απόφασης με δεδομένα & δείκτες — **ΔΕΝ** είναι επίσημη επενδυτική "
    "συμβουλή. Δεδομένα: Yahoo Finance (καθυστέρηση ~15').  Οι εντολές γίνονται "
    "χειροκίνητα στο DeGiro."
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

tab_overview, tab_detail = st.tabs(["🏠 Επισκόπηση & Κατανομή", "🔍 Ανάλυση μετοχής"])

# --- Επισκόπηση: πίνακας σκορ όλων + πρόταση κατανομής ------------------------
with tab_overview:
    rows = []
    signals = {}
    with st.spinner("Φόρτωση δεδομένων..."):
        for t in tickers:
            df = load_history(t, period="2y")
            info = load_info(t)
            score, label, reasons = compute_signal(df)
            signals[t] = (score, label, reasons)
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

st.markdown("---")
st.caption(
    "Εκπαιδευτικό εργαλείο. Οι επενδύσεις έχουν κίνδυνο απώλειας κεφαλαίου. "
    "Δεν αποτελεί επενδυτική συμβουλή."
)
