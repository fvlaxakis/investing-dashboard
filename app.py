# -*- coding: utf-8 -*-
"""
Επενδυτικό Dashboard — βοηθός απόφασης (ΟΧΙ επίσημη επενδυτική συμβουλή).

Live τιμές & γραφήματα από Yahoo Finance (δωρεάν), τεχνικοί & θεμελιώδεις
δείκτες, ένα διαφανές σκορ ανά μετοχή, και σχεδιασμός κατανομής κεφαλαίου.

Τρέξε τοπικά:      py -m streamlit run app.py
Δείχνει στο κινητό: κάνε deploy δωρεάν στο Streamlit Community Cloud (δες README).
"""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core

try:
    from streamlit_local_storage import LocalStorage
except Exception:  # η βιβλιοθήκη μπορεί να λείπει σε κάποιο περιβάλλον
    LocalStorage = None

st.set_page_config(page_title="Επενδυτικό Dashboard", page_icon="📈", layout="wide")

# ----------------------------------------------------------------------------
# Ρυθμίσεις / προεπιλογές
# ----------------------------------------------------------------------------
DEFAULT_WATCHLIST = "AAPL, MSFT, NVDA, VUAA.DE, IWDA.AS, ASML.AS"
# VUAA.DE = S&P 500 ETF (EUR, Xetra) · IWDA.AS = MSCI World ETF (Amsterdam)

# ----------------------------------------------------------------------------
# Cached wrappers γύρω από την καθαρή λογική του core.py
# (η ίδια λογική χρησιμοποιείται και από το scan_email.py για το πρωινό email)
# ----------------------------------------------------------------------------
load_history = st.cache_data(ttl=900, show_spinner=False)(core.fetch_history)
load_info = st.cache_data(ttl=3600, show_spinner=False)(core.fetch_info)
load_news = st.cache_data(ttl=1800, show_spinner=False)(core.fetch_news)
fx_to_eur = st.cache_data(ttl=3600, show_spinner=False)(core.fx_to_eur)
compute_signal = core.compute_signal
tax_profile = core.tax_profile
scan_alerts = core.scan_alerts


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

tab_overview, tab_detail, tab_ideas, tab_portfolio = st.tabs(
    ["🏠 Επισκόπηση & Κατανομή", "🔍 Ανάλυση μετοχής", "💡 Ιδέες (Screener)",
     "💼 Χαρτοφυλάκιο"]
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
    # --- 📋 Τι να κάνω σήμερα -----------------------------------------------
    st.subheader("📋 Τι να κάνω σήμερα")
    reds = sorted({t for t, sev, _ in all_alerts if sev == "red"})
    greens = sorted({t for t, sev, _ in all_alerts if sev == "green"})
    if reds:
        st.error(
            f"🔴 **Πρόσεξε: {', '.join(reds)}.** Κάτι ασυνήθιστο (αρνητικό) "
            "συμβαίνει. Πήγαινε στο tab «Ανάλυση μετοχής», διάβασε τα 📰 Νέα και "
            "**μη βιάζεσαι** — μια απότομη πτώση δεν σημαίνει αυτόματα «πούλα». "
            "Αν είναι θέση που κρατάς μακροπρόθεσμα (π.χ. ETF), συνήθως δεν κάνεις "
            "τίποτα."
        )
    if greens:
        st.success(
            f"🟢 **Δυνατή κίνηση: {', '.join(greens)}.** Θετικό σήμα — αλλά "
            "μην κυνηγήσεις την τιμή. Αν σε ενδιαφέρει, δες την ήρεμα στο tab "
            "«Ανάλυση μετοχής»."
        )
    if not reds and not greens:
        st.success(
            "🟢 **Ήρεμα νερά — δεν χρειάζεται καμία ενέργεια σήμερα.** Αυτό είναι "
            "το φυσιολογικό τις περισσότερες μέρες. Το να ΜΗΝ κάνεις κίνηση είναι "
            "σωστή κίνηση. Μένεις στο πλάνο σου."
        )
    st.caption(
        "💡 Ρυθμός: κοίτα το Radar ~1×/βδομάδα, το χαρτοφυλάκιο ~1×/μήνα. Η υπομονή "
        "είναι το μεγαλύτερο πλεονέκτημά σου — όχι οι συχνές κινήσεις."
    )
    st.divider()

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

# --- 💡 Ιδέες (Screener): προτάσεις νέων φορολογικά αποδοτικών προϊόντων ------
with tab_ideas:
    st.subheader("💡 Ιδέες για νέες θέσεις")
    st.caption(
        "Σαρώνει ένα σύνολο δημοφιλών φορολογικά αποδοτικών προϊόντων (accumulating "
        "UCITS ETF & μετοχές χαμηλού μερίσματος, συνήθως στο DeGiro) και προτείνει "
        "όσα ΔΕΝ έχεις ήδη στη watchlist, με το καλύτερο σκορ. Είναι σημείο "
        "εκκίνησης για έρευνα — **όχι εντολή αγοράς**. Πάντα επιβεβαίωσε στο DeGiro."
    )
    only_taxfree = st.checkbox("Μόνο πλήρως αφορολόγητα (🟢)", value=True)
    if st.button("🔍 Σάρωση αγοράς"):
        cands = [c for c in core.SCREENER_UNIVERSE if c not in tickers]
        res = []
        prog = st.progress(0.0, text="Σάρωση...")
        for i, c in enumerate(cands):
            prog.progress((i + 1) / len(cands), text=f"Σάρωση {c}...")
            dfc = load_history(c, period="2y")
            if dfc.empty:
                continue
            sc, lbl, _ = compute_signal(dfc)
            infoc = load_info(c)
            tier, badge, _ = tax_profile(c, infoc)
            if only_taxfree and tier != "green":
                continue
            if not only_taxfree and tier == "red":
                continue
            res.append({"Σύμβολο": c, "Όνομα": infoc["name"], "Σκορ": sc,
                        "Σήμα": lbl, "Φόρος": badge})
        prog.empty()
        if res:
            resdf = pd.DataFrame(res).sort_values("Σκορ", ascending=False).head(12)
            st.dataframe(
                resdf.style.background_gradient(subset=["Σκορ"], cmap="RdYlGn",
                                                vmin=0, vmax=100),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "Ταξινομημένα με σκορ. 🟢 = πλήρως αφορολόγητο. Πρόσθεσε όποιο σε "
                "ενδιαφέρει στη watchlist για να το παρακολουθείς, και επιβεβαίωσε "
                "διαθεσιμότητα & ISIN στο DeGiro πριν αγοράσεις."
            )
        else:
            st.warning("Δεν βρέθηκαν ιδέες με αυτά τα κριτήρια αυτή τη στιγμή.")
    else:
        st.info("Πάτα «🔍 Σάρωση αγοράς» για προτάσεις (παίρνει ~30 δευτερόλεπτα).")

# --- 💼 Χαρτοφυλάκιο: πραγματικές αγορές, κέρδος/ζημιά, καθαρό μετά φόρου -----
with tab_portfolio:
    st.subheader("💼 Το χαρτοφυλάκιό μου")
    st.caption(
        "Βάλε τις αγορές σου: σύμβολο, ποσότητα (τεμάχια) και μέση τιμή αγοράς "
        "(στο νόμισμα της μετοχής). Τα κενά αγνοούνται. Οι τιμές είναι live· τα "
        "σύνολα μετατρέπονται σε € αυτόματα."
    )

    default_seed = [{"Σύμβολο": "", "Ποσότητα": None, "Τιμή αγοράς": None}]
    COLS = ["Σύμβολο", "Ποσότητα", "Τιμή αγοράς"]

    def _get_secret(k):
        try:
            return st.secrets[k]
        except Exception:
            return None

    SUPA_URL = _get_secret("SUPABASE_URL")
    SUPA_KEY = _get_secret("SUPABASE_KEY")
    use_supabase = bool(SUPA_URL and SUPA_KEY)

    def records_to_df(recs):
        df = pd.DataFrame(recs)
        if df.empty:
            return pd.DataFrame(default_seed)
        return df.rename(columns={"symbol": "Σύμβολο", "qty": "Ποσότητα",
                                  "buy_price": "Τιμή αγοράς"})[COLS]

    def df_to_records(df):
        recs = []
        for _, r in df.iterrows():
            s = str(r["Σύμβολο"]).strip().upper()
            if not s or pd.isna(r["Ποσότητα"]):
                continue
            recs.append({"symbol": s,
                         "qty": float(r["Ποσότητα"]),
                         "buy_price": float(r["Τιμή αγοράς"]) if pd.notna(r["Τιμή αγοράς"]) else None})
        return recs

    # localStorage (fallback όταν δεν υπάρχει Supabase)
    local_store = None
    if not use_supabase and LocalStorage is not None:
        try:
            local_store = LocalStorage()
        except Exception:
            local_store = None

    # --- Φόρτωση αρχικών δεδομένων (μία φορά ανά session) --------------------
    if "pf_seed" not in st.session_state:
        seed_df = pd.DataFrame(default_seed)
        if use_supabase:
            try:
                seed_df = records_to_df(core.supabase_load(SUPA_URL, SUPA_KEY))
            except Exception as e:
                st.warning(f"Δεν φορτώθηκε από Supabase: {e}")
        elif local_store is not None:
            try:
                saved = local_store.getItem("portfolio_v1")
                if saved:
                    seed_df = pd.DataFrame(json.loads(saved))
            except Exception:
                pass
        st.session_state.pf_seed = seed_df

    if use_supabase:
        st.caption("🔄 **Sync ενεργό (Supabase)** — το χαρτοφυλάκιο είναι ίδιο σε "
                   "κινητό & υπολογιστή.")

    edited = st.data_editor(
        st.session_state.pf_seed, num_rows="dynamic",
        use_container_width=True, hide_index=True,
        column_config={
            "Σύμβολο": st.column_config.TextColumn(help="π.χ. VUAA.DE, AAPL"),
            "Ποσότητα": st.column_config.NumberColumn(format="%.4f", min_value=0),
            "Τιμή αγοράς": st.column_config.NumberColumn(format="%.2f", min_value=0),
        },
        key="portfolio_editor",
    )

    # --- Κουμπιά αποθήκευσης / backup ---------------------------------------
    b1, b2 = st.columns([1, 1])
    with b1:
        if st.button("💾 Αποθήκευση", use_container_width=True):
            if use_supabase:
                try:
                    core.supabase_save(SUPA_URL, SUPA_KEY, df_to_records(edited))
                    st.success("✅ Αποθηκεύτηκε στο cloud — συγχρονισμένο παντού.")
                except Exception as e:
                    st.error(f"Αποτυχία Supabase: {e}")
            elif local_store is not None:
                try:
                    local_store.setItem(
                        "portfolio_v1", edited.to_json(orient="records", force_ascii=False))
                    st.success("✅ Αποθηκεύτηκε στη συσκευή σου (μόνο εδώ — για sync "
                               "παντού ρύθμισε Supabase).")
                except Exception:
                    st.warning("Δεν έγινε αποθήκευση — κατέβασε CSV.")
            else:
                st.warning("Αποθήκευση μη διαθέσιμη — χρησιμοποίησε το CSV.")
    with b2:
        st.download_button(
            "⬇️ Backup CSV", edited.to_csv(index=False).encode("utf-8"),
            file_name="portfolio.csv", mime="text/csv", use_container_width=True,
        )
    up = st.file_uploader("⬆️ Επαναφορά από CSV", type="csv")
    if up is not None:
        try:
            st.session_state.pf_seed = pd.read_csv(up)
            st.success("Φορτώθηκε. Πάτα «💾 Αποθήκευση» για να το κρατήσεις.")
            st.rerun()
        except Exception:
            st.error("Δεν διαβάστηκε το CSV.")

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
            "⚠️ Ενδεικτικός υπολογισμός. Οι τιμές έχουν καθυστέρηση ~15'. Πάτα "
            "«💾 Αποθήκευση» για να θυμάται το χαρτοφυλάκιο στη συσκευή σου. Για "
            "συγχρονισμό μεταξύ κινητού & υπολογιστή, χρησιμοποίησε το Backup CSV."
        )
    else:
        st.info("Πρόσθεσε γραμμές παραπάνω για να δεις κέρδος/ζημιά και σύνολα.")

st.markdown("---")
st.caption(
    "Εκπαιδευτικό εργαλείο. Οι επενδύσεις έχουν κίνδυνο απώλειας κεφαλαίου. "
    "Δεν αποτελεί επενδυτική συμβουλή."
)
