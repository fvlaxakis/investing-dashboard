# -*- coding: utf-8 -*-
"""
scan_email.py — Πρωινή σάρωση watchlist & αποστολή email.

Τρέχει αυτόματα από το GitHub Actions (δες .github/workflows/morning-scan.yml).
Χρησιμοποιεί την ίδια λογική με το dashboard (core.py) — καμία εξάρτηση Streamlit.

Ρυθμίσεις μέσω μεταβλητών περιβάλλοντος (GitHub Secrets):
  SMTP_USER  — το Gmail σου (π.χ. fvlaxakis@gmail.com)
  SMTP_PASS  — Gmail App Password (16 χαρακτήρες, ΟΧΙ ο κανονικός κωδικός)
  MAIL_TO    — πού να σταλεί (default: SMTP_USER)
  WATCHLIST  — σύμβολα χωρισμένα με κόμμα (αλλιώς διαβάζει το watchlist.txt)
"""

import os
import smtplib
import ssl
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import core


def get_watchlist() -> list:
    env = os.environ.get("WATCHLIST", "").strip()
    if env:
        raw = env
    else:
        try:
            with open("watchlist.txt", encoding="utf-8") as f:
                raw = f.read()
        except FileNotFoundError:
            raw = "AAPL, MSFT, NVDA, VUAA.DE, IWDA.AS, ASML.AS"
    parts = raw.replace("\n", ",").split(",")
    return [p.strip().upper() for p in parts if p.strip()]


def build_report(tickers: list):
    reds, greens, infos, scores = [], [], [], []
    for t in tickers:
        df = core.fetch_history(t, period="2y")
        if df.empty:
            continue
        sc, lbl, _ = core.compute_signal(df)
        scores.append((t, sc, lbl))
        for sev, txt in core.scan_alerts(t, df):
            entry = f"<b>{t}</b> — {txt}"
            if sev == "red":
                reds.append(entry)
            elif sev == "green":
                greens.append(entry)
            else:
                infos.append(entry)
    scores.sort(key=lambda x: x[1], reverse=True)
    return reds, greens, infos, scores


def to_html(reds, greens, infos, scores) -> str:
    today = date.today().strftime("%d/%m/%Y")
    if reds:
        headline = ("🔴 <b>Πρόσεξε σήμερα.</b> Εντοπίστηκαν αρνητικές ασυνήθιστες "
                    "κινήσεις — δες παρακάτω και μη βιάζεσαι.")
    elif greens or infos:
        headline = ("🟡 Υπάρχουν κάποιες κινήσεις να δεις — αλλά τίποτα ανησυχητικό.")
    else:
        headline = ("🟢 <b>Ήρεμα νερά — καμία ενέργεια σήμερα.</b> Μένεις στο πλάνο σου.")

    def block(title, items):
        if not items:
            return ""
        lis = "".join(f"<li style='margin:4px 0'>{x}</li>" for x in items)
        return f"<h3 style='margin:16px 0 4px'>{title}</h3><ul>{lis}</ul>"

    top = "".join(
        f"<li>{t}: <b>{sc}/100</b> ({lbl})</li>" for t, sc, lbl in scores[:5]
    )
    return f"""<div style="font-family:Arial,sans-serif;max-width:640px;color:#1a1a1a">
      <h2>📈 Πρωινή σάρωση — {today}</h2>
      <p style="font-size:15px">{headline}</p>
      {block("🔴 Αρνητικά σήματα", reds)}
      {block("🟢 Θετικά σήματα", greens)}
      {block("ℹ️ Ασυνήθιστος όγκος / λοιπά", infos)}
      <h3 style="margin:16px 0 4px">🏆 Κορυφαία σκορ</h3>
      <ul>{top}</ul>
      <hr style="margin-top:20px;border:none;border-top:1px solid #ddd">
      <p style="font-size:12px;color:#888">
        Αυτόματη σάρωση από το Επενδυτικό σου Dashboard. Δείκτες = ενδείξεις, όχι
        βεβαιότητα. ΔΕΝ είναι επενδυτική συμβουλή. Οι εντολές γίνονται χειροκίνητα
        στο DeGiro.
      </p>
    </div>"""


def send_email(html: str):
    user = os.environ["SMTP_USER"]
    password = os.environ["SMTP_PASS"]
    to_addr = os.environ.get("MAIL_TO", user)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Επενδυτική σάρωση — {date.today().strftime('%d/%m/%Y')}"
    msg["From"] = user
    msg["To"] = to_addr
    msg.attach(MIMEText(html, "html", "utf-8"))

    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
        server.login(user, password)
        server.sendmail(user, [to_addr], msg.as_string())
    print(f"Email sent to {to_addr}")


def main():
    tickers = get_watchlist()
    print(f"Scanning {len(tickers)} tickers: {tickers}")
    reds, greens, infos, scores = build_report(tickers)
    html = to_html(reds, greens, infos, scores)
    if os.environ.get("SMTP_USER") and os.environ.get("SMTP_PASS"):
        send_email(html)
    else:
        print("SMTP_USER/SMTP_PASS δεν έχουν οριστεί — προεπισκόπηση μόνο:\n")
        print(html)


if __name__ == "__main__":
    main()
