"""Launchable web interface for the Customer Support Ticket Classifier.

In VS Code, open this file and click Run Python File.
The local website will open in your default browser.
"""
from __future__ import annotations

import threading
import time
import webbrowser
from pathlib import Path

from flask import Flask, render_template_string, request

from src.ticket_classifier import load_dataset, train_and_evaluate, predict_ticket

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
DATA = load_dataset(ROOT / "data" / "support_tickets.csv")
RESULT = train_and_evaluate(DATA)
MODEL = RESULT["model"]

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Customer Support Ticket Classifier</title>
<style>
:root { color-scheme: light; --navy:#16324f; --blue:#2563eb; --bg:#f3f6fb; --muted:#5c6b7a; }
* { box-sizing:border-box; }
body { margin:0; font-family:Segoe UI,Arial,sans-serif; background:var(--bg); color:#172435; }
header { background:linear-gradient(120deg,#16324f,#2563eb); color:white; padding:32px 20px; }
header .wrap, main { max-width:980px; margin:auto; }
h1 { margin:0 0 8px; font-size:clamp(24px,4vw,36px); }
header p { margin:0; color:#e4edff; line-height:1.5; }
main { padding:24px 16px 40px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:14px; margin-bottom:18px; }
.card { background:white; border:1px solid #e1e8f0; border-radius:14px; padding:20px; box-shadow:0 4px 14px #17324f0a; }
.stat { font-size:27px; font-weight:700; color:var(--navy); margin-top:8px; }
.label { color:var(--muted); font-size:13px; text-transform:uppercase; letter-spacing:.06em; font-weight:700; }
textarea { width:100%; min-height:140px; resize:vertical; padding:14px; border:1px solid #cbd5e1; border-radius:10px; font:inherit; line-height:1.5; }
textarea:focus { outline:3px solid #bfdbfe; border-color:var(--blue); }
button { background:var(--blue); color:white; border:0; border-radius:9px; padding:12px 18px; font-weight:700; font-size:15px; cursor:pointer; margin-top:12px; }
button:hover { background:#1d4ed8; }
.result { border-left:5px solid var(--blue); }
.pill { display:inline-block; padding:6px 11px; border-radius:999px; font-weight:700; font-size:13px; text-transform:uppercase; }
.high { background:#fee2e2; color:#991b1b; }
.medium { background:#fef3c7; color:#92400e; }
.low { background:#dcfce7; color:#166534; }
.small { color:var(--muted); font-size:14px; line-height:1.6; }
footer { text-align:center; color:var(--muted); padding:0 16px 28px; font-size:13px; }
ul { padding-left:20px; line-height:1.8; }
</style>
</head>
<body>
<header><div class="wrap">
<h1>Customer Support Ticket Classifier</h1>
<p>Classify incoming tickets with machine learning and assign a transparent priority level.</p>
</div></header>
<main>
<div class="grid">
<div class="card"><div class="label">Training examples</div><div class="stat">{{ rows }}</div><div class="small">Illustrative sample tickets</div></div>
<div class="card"><div class="label">Ticket categories</div><div class="stat">{{ categories }}</div><div class="small">Billing, technical, account, shipping, cancellation/refund</div></div>
<div class="card"><div class="label">Test accuracy</div><div class="stat">{{ accuracy }}%</div><div class="small">Held-out sample split; educational only</div></div>
</div>
<div class="card">
<h2>Analyze a support ticket</h2>
<p class="small">Enter the customer's message below. Avoid entering real personal or sensitive information.</p>
<form method="post">
<textarea name="ticket" placeholder="Example: I was charged twice and need a refund immediately." required>{{ ticket }}</textarea>
<br><button type="submit">Classify ticket</button>
</form>
</div>
{% if prediction %}
<div class="card result" style="margin-top:18px">
<h2>Prediction result</h2>
<div class="grid">
<div><div class="label">Predicted category</div><div class="stat" style="font-size:22px">{{ prediction.category.replace('_',' ').title() }}</div></div>
<div><div class="label">Assigned priority</div><p><span class="pill {{ prediction.priority }}">{{ prediction.priority }}</span></p></div>
</div>
<p><strong>Why this priority?</strong> {{ prediction.reason }}</p>
</div>
{% endif %}
<div class="card" style="margin-top:18px">
<h2>How this project works</h2>
<ul>
<li>Text preprocessing and TF-IDF feature extraction.</li>
<li>Scikit-learn Logistic Regression predicts the ticket category.</li>
<li>Readable priority rules assign high, medium, or low priority.</li>
<li>Model performance is evaluated using a held-out test split.</li>
</ul>
<p class="small">This is an educational prototype trained on a small synthetic dataset. Review predictions before acting on them; real deployment needs representative, privacy-reviewed data and validated support policies.</p>
</div>
</main>
<footer>Python · Flask · Pandas · Scikit-learn · Local demo</footer>
</body></html>"""

@app.route("/", methods=["GET", "POST"])
def home():
    ticket = ""
    prediction = None
    if request.method == "POST":
        ticket = request.form.get("ticket", "").strip()
        if ticket:
            result = predict_ticket(MODEL, ticket)
            prediction = {
                "category": result["category"],
                "priority": result["priority"],
                "reason": result["priority_reason"],
            }
    return render_template_string(
        PAGE,
        rows=len(DATA),
        categories=DATA["category"].nunique(),
        accuracy=round(RESULT["accuracy"] * 100, 1),
        ticket=ticket,
        prediction=prediction,
    )

def open_browser():
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    print("Customer Support Ticket Classifier is starting.")
    print("If your browser does not open automatically, visit http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
