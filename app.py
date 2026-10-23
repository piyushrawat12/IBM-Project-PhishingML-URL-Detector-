# coding: utf-8
"""
Phishing URL Detection Using Machine Learning
=============================================
A beginner-friendly Flask web application that uses a
Logistic Regression model (trained on hand-crafted URL features)
to detect phishing URLs in real time.

Run:
    pip install flask scikit-learn numpy
    python app.py
Then open: http://127.0.0.1:5000
"""

from flask import Flask, render_template_string, request, jsonify
import re
import math
import datetime
import numpy as np
from urllib.parse import urlparse

# ─────────────────────────────────────────────────────────────
#  1.  FEATURE EXTRACTION
#      We pull 18 measurable signals from every URL.
# ─────────────────────────────────────────────────────────────

SUSPICIOUS_TLDS   = {".tk", ".ml", ".ga", ".cf", ".gq", ".pw", ".top",
                     ".xyz", ".online", ".site", ".click", ".link", ".win"}
SUSPICIOUS_WORDS  = ["secure", "account", "update", "confirm", "login",
                     "signin", "banking", "verify", "webscr", "ebayisapi",
                     "paypal", "password", "credential", "free", "lucky",
                     "prize", "winner", "click", "urgent", "suspended"]
TRUSTED_DOMAINS   = {"google.com", "facebook.com", "amazon.com", "apple.com",
                     "microsoft.com", "github.com", "youtube.com", "twitter.com",
                     "instagram.com", "linkedin.com", "wikipedia.org"}


def url_length(url):
    return len(url)

def count_dots(url):
    return url.count(".")

def count_hyphens(url):
    return url.count("-")

def count_at(url):
    return url.count("@")

def count_slashes(url):
    return url.count("/")

def count_digits(url):
    return sum(c.isdigit() for c in url)

def has_ip_address(url):
    """1 if the host looks like a raw IP address."""
    host = urlparse(url).netloc.split(":")[0]
    pattern = r"^\d{1,3}(\.\d{1,3}){3}$"
    return 1 if re.match(pattern, host) else 0

def has_https(url):
    return 1 if url.startswith("https://") else 0

def domain_length(url):
    host = urlparse(url).netloc
    return len(host)

def subdomain_count(url):
    host = urlparse(url).netloc.split(":")[0]
    parts = host.split(".")
    return max(0, len(parts) - 2)   # e.g. a.b.example.com -> 2 subdomains

def suspicious_tld(url):
    host = urlparse(url).netloc.split(":")[0]
    for tld in SUSPICIOUS_TLDS:
        if host.endswith(tld):
            return 1
    return 0

def suspicious_word_count(url):
    lower = url.lower()
    return sum(1 for w in SUSPICIOUS_WORDS if w in lower)

def path_length(url):
    return len(urlparse(url).path)

def query_length(url):
    return len(urlparse(url).query)

def count_special_chars(url):
    return sum(1 for c in url if c in "!#$%^&*()+=[]{}|;:,<>?")

def has_port(url):
    netloc = urlparse(url).netloc
    return 1 if ":" in netloc else 0

def url_entropy(url):
    """Shannon entropy — high entropy can indicate random/obfuscated strings."""
    if not url:
        return 0.0
    freq = {}
    for c in url:
        freq[c] = freq.get(c, 0) + 1
    n = len(url)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())

def is_trusted_domain(url):
    host = urlparse(url).netloc.split(":")[0].lower()
    # strip leading www.
    if host.startswith("www."):
        host = host[4:]
    return 1 if host in TRUSTED_DOMAINS else 0


def extract_features(url):
    """Return a numpy array of 18 features for a given URL."""
    return np.array([
        url_length(url),          # 0
        count_dots(url),          # 1
        count_hyphens(url),       # 2
        count_at(url),            # 3
        count_slashes(url),       # 4
        count_digits(url),        # 5
        has_ip_address(url),      # 6
        has_https(url),           # 7
        domain_length(url),       # 8
        subdomain_count(url),     # 9
        suspicious_tld(url),      # 10
        suspicious_word_count(url),  # 11
        path_length(url),         # 12
        query_length(url),        # 13
        count_special_chars(url), # 14
        has_port(url),            # 15
        url_entropy(url),         # 16
        is_trusted_domain(url),   # 17
    ], dtype=float)


# ─────────────────────────────────────────────────────────────
#  2.  TRAIN A LOGISTIC REGRESSION MODEL  (at startup)
#      We use a small hand-crafted training set so the app
#      works without any external dataset file.
# ─────────────────────────────────────────────────────────────

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

TRAINING_URLS = [
    # --- LEGITIMATE (label = 0) ---
    ("https://www.google.com/search?q=flask", 0),
    ("https://github.com/pallets/flask", 0),
    ("https://www.amazon.com/dp/B08N5WRWNW", 0),
    ("https://en.wikipedia.org/wiki/Python", 0),
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", 0),
    ("https://stackoverflow.com/questions/tagged/python", 0),
    ("https://www.linkedin.com/in/someone", 0),
    ("https://twitter.com/home", 0),
    ("https://www.microsoft.com/en-us/windows", 0),
    ("https://apple.com/iphone", 0),
    ("https://docs.python.org/3/library/re.html", 0),
    ("https://flask.palletsprojects.com/en/3.0.x/", 0),
    ("https://www.reddit.com/r/python", 0),
    ("https://news.ycombinator.com/", 0),
    ("https://www.bbc.com/news", 0),
    ("https://developer.mozilla.org/en-US/docs/Web", 0),
    ("https://www.nytimes.com/section/technology", 0),
    ("https://pypi.org/project/scikit-learn/", 0),
    ("https://scikit-learn.org/stable/", 0),
    ("https://www.ibm.com/cloud", 0),
    # --- PHISHING (label = 1) ---
    ("http://192.168.1.1/secure-login.html", 1),
    ("http://paypal-secure-update.tk/account/verify", 1),
    ("http://amazon-prize-winner.xyz/claim?id=12345", 1),
    ("http://login-facebook-confirm.ml/signin", 1),
    ("http://secure-banking-verify.ga/update-password", 1),
    ("http://apple-id-suspended.pw/confirm-account", 1),
    ("http://microsoft-account-locked.top/reset", 1),
    ("http://free-gift-winner.online/prize/claim", 1),
    ("http://ebayisapi-webscr.click/account-verify", 1),
    ("http://urgent-credential-update.win/login", 1),
    ("http://10.0.0.1/bank/signin?user=admin", 1),
    ("http://sub1.sub2.sub3.phish.cf/steal-data", 1),
    ("http://g00gle-secure.site/accounts/login", 1),
    ("http://amaz0n-update.link/verify-account", 1),
    ("http://payp4l-verify.pw/confirm?token=abc123", 1),
    ("http://support-apple-id.gq/locked-account", 1),
    ("http://faceb00k-login.ml/checkpoint", 1),
    ("http://bank-account-verify.tk/secure/login", 1),
    ("http://free-iphone-winner.xyz/claim-prize", 1),
    ("http://credential-phishing-test.online/login-verify?id=999&token=xyz", 1),
]


def train_model():
    X = np.array([extract_features(url) for url, _ in TRAINING_URLS])
    y = np.array([label for _, label in TRAINING_URLS])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler


print("Training ML model on startup...")
MODEL, SCALER = train_model()
print("Model ready.")

# ─────────────────────────────────────────────────────────────
#  3.  PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "URL Length", "Dot Count", "Hyphen Count", "@ Symbol",
    "Slash Count", "Digit Count", "Has IP Address", "Has HTTPS",
    "Domain Length", "Subdomain Count", "Suspicious TLD",
    "Suspicious Words", "Path Length", "Query Length",
    "Special Chars", "Has Port", "URL Entropy", "Trusted Domain",
]

def predict_url(url):
    """Run the ML model on a URL and return a rich result dict."""
    # ensure it has a scheme so urlparse works
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    features = extract_features(url)
    features_scaled = SCALER.transform(features.reshape(1, -1))
    prediction = MODEL.predict(features_scaled)[0]
    proba = MODEL.predict_proba(features_scaled)[0]  # [legit_prob, phish_prob]

    phish_score = round(proba[1] * 100, 1)
    legit_score = round(proba[0] * 100, 1)

    if phish_score >= 70:
        verdict = "PHISHING"
        verdict_color = "#ef4444"
        risk_level = "HIGH RISK"
    elif phish_score >= 40:
        verdict = "SUSPICIOUS"
        verdict_color = "#f97316"
        risk_level = "MEDIUM RISK"
    else:
        verdict = "LEGITIMATE"
        verdict_color = "#22c55e"
        risk_level = "LOW RISK"

    # Build per-feature breakdown for the UI
    breakdown = []
    for i, name in enumerate(FEATURE_NAMES):
        val = features[i]
        flag = False
        # flag suspicious values
        if name == "Has IP Address"    and val == 1: flag = True
        if name == "Has HTTPS"         and val == 0: flag = True
        if name == "Suspicious TLD"    and val == 1: flag = True
        if name == "Suspicious Words"  and val >= 1: flag = True
        if name == "@ Symbol"          and val >= 1: flag = True
        if name == "Subdomain Count"   and val >= 3: flag = True
        if name == "URL Length"        and val > 75:  flag = True
        if name == "Has Port"          and val == 1: flag = True
        if name == "URL Entropy"       and val > 4.5: flag = True
        if name == "Trusted Domain"    and val == 1: flag = False  # trusted = good

        breakdown.append({
            "name":  name,
            "value": round(float(val), 3),
            "flag":  flag,
        })

    return {
        "url":           url,
        "verdict":       verdict,
        "verdict_color": verdict_color,
        "risk_level":    risk_level,
        "phish_score":   phish_score,
        "legit_score":   legit_score,
        "breakdown":     breakdown,
    }


# ─────────────────────────────────────────────────────────────
#  4.  SCAN LOG  (in-memory)
# ─────────────────────────────────────────────────────────────

scan_log = []


# ─────────────────────────────────────────────────────────────
#  5.  HTML TEMPLATE  (dark theme, all-in-one)
# ─────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Phishing URL Detection Using Machine Learning</title>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#080c14;color:#c9d1d9;min-height:100vh;}

/* NAV */
nav{background:#0d1117;border-bottom:1px solid #21262d;padding:0 2rem;display:flex;
    align-items:center;justify-content:space-between;height:62px;position:sticky;top:0;z-index:100;}
.logo{font-size:1.15rem;font-weight:700;color:#58a6ff;letter-spacing:.5px;}
.logo span{color:#a371f7;}
nav ul{list-style:none;display:flex;gap:1.5rem;}
nav ul a{color:#8b949e;text-decoration:none;font-size:.88rem;transition:color .2s;}
nav ul a:hover{color:#58a6ff;}

/* HERO */
.hero{text-align:center;padding:3.5rem 1rem 1.5rem;background:linear-gradient(175deg,#0d1117 0%,#080c14 100%);}
.hero h1{font-size:2.1rem;color:#e6edf3;margin-bottom:.5rem;}
.hero h1 span{color:#a371f7;}
.hero p{color:#8b949e;font-size:.95rem;max-width:560px;margin:.5rem auto 0;}
.badge-ml{display:inline-block;background:#a371f722;color:#a371f7;border:1px solid #a371f744;
          border-radius:999px;font-size:.75rem;font-weight:700;padding:.2rem .8rem;margin-bottom:.75rem;
          letter-spacing:.5px;}

/* STATS */
.stats-bar{display:flex;justify-content:center;gap:2rem;flex-wrap:wrap;
           padding:1.2rem 2rem;background:#0d1117;
           border-top:1px solid #21262d;border-bottom:1px solid #21262d;margin-top:1.5rem;}
.stat{text-align:center;}
.stat-num{font-size:1.5rem;font-weight:700;color:#58a6ff;}
.stat-label{font-size:.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;}

/* LAYOUT */
.container{max-width:1120px;margin:0 auto;padding:2rem 1rem;}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;}
@media(max-width:750px){.grid,.grid-3{grid-template-columns:1fr;}}

/* CARDS */
.card{background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:1.5rem;}
.card h2{color:#e6edf3;font-size:1rem;margin-bottom:1rem;display:flex;align-items:center;gap:.5rem;}
.card h2 .icon{color:#58a6ff;}

/* URL INPUT */
.url-row{display:flex;gap:.6rem;}
.url-row input{flex:1;background:#161b22;border:1px solid #30363d;border-radius:6px;
               color:#c9d1d9;padding:.7rem 1rem;font-size:.95rem;outline:none;transition:border-color .2s;}
.url-row input:focus{border-color:#a371f7;}
.url-row input::placeholder{color:#4a5261;}
.btn-scan{background:#6e40c9;color:#fff;border:none;border-radius:6px;padding:.7rem 1.5rem;
          font-size:.9rem;cursor:pointer;font-weight:700;transition:background .2s;white-space:nowrap;}
.btn-scan:hover{background:#8957e5;}
.btn-sm{background:#21262d;color:#c9d1d9;border:none;border-radius:5px;padding:.3rem .7rem;
        font-size:.78rem;cursor:pointer;transition:background .2s;}
.btn-sm:hover{background:#30363d;}

/* RESULT */
#result{margin-top:1.2rem;display:none;}
.result-box{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:1.2rem;}
.verdict-row{display:flex;align-items:center;gap:1rem;margin-bottom:.75rem;flex-wrap:wrap;}
.verdict-text{font-size:1.6rem;font-weight:800;}
.risk-badge{font-size:.8rem;font-weight:700;padding:.25rem .75rem;border-radius:999px;}
.score-row{display:flex;gap:1.5rem;margin-bottom:1rem;flex-wrap:wrap;}
.score-item{text-align:center;}
.score-num{font-size:1.3rem;font-weight:700;}
.score-label{font-size:.72rem;color:#8b949e;}

/* PROGRESS BAR */
.progress-wrap{margin-bottom:1rem;}
.progress-label{font-size:.8rem;color:#8b949e;margin-bottom:.3rem;display:flex;justify-content:space-between;}
.progress-bg{background:#21262d;border-radius:999px;height:8px;overflow:hidden;}
.progress-fill{height:100%;border-radius:999px;transition:width .4s;}

/* FEATURE TABLE */
.feature-grid{display:grid;grid-template-columns:1fr 1fr;gap:.4rem;}
@media(max-width:500px){.feature-grid{grid-template-columns:1fr;}}
.feat-item{background:#0a0e1a;border:1px solid #21262d;border-radius:6px;
           padding:.45rem .75rem;display:flex;justify-content:space-between;align-items:center;font-size:.8rem;}
.feat-name{color:#8b949e;}
.feat-val{font-weight:700;color:#e6edf3;}
.feat-flag{color:#ef4444;}

/* LOG TABLE */
.log-wrap{overflow-x:auto;}
table{width:100%;border-collapse:collapse;font-size:.83rem;}
th{background:#161b22;color:#8b949e;text-align:left;padding:.6rem .75rem;
   border-bottom:1px solid #21262d;font-weight:600;}
td{padding:.5rem .75rem;border-bottom:1px solid #161b22;}
tr:hover td{background:#161b22;}
.badge{display:inline-block;padding:.15rem .55rem;border-radius:999px;font-size:.72rem;font-weight:700;}

/* EXAMPLES */
.ex-list{display:flex;flex-direction:column;gap:.5rem;}
.ex-item{background:#161b22;border:1px solid #21262d;border-radius:6px;
         padding:.55rem .85rem;display:flex;justify-content:space-between;align-items:center;gap:.5rem;}
.ex-url{font-size:.78rem;color:#8b949e;font-family:'Courier New',monospace;word-break:break-all;flex:1;}
.ex-label{font-size:.72rem;font-weight:700;padding:.15rem .5rem;border-radius:999px;white-space:nowrap;}

/* HOW IT WORKS */
.step-list{display:flex;flex-direction:column;gap:.75rem;}
.step{display:flex;gap:.85rem;align-items:flex-start;}
.step-num{background:#a371f722;color:#a371f7;border:1px solid #a371f744;border-radius:50%;
          width:28px;height:28px;display:flex;align-items:center;justify-content:center;
          font-size:.8rem;font-weight:700;flex-shrink:0;margin-top:.1rem;}
.step-text h4{color:#e6edf3;font-size:.88rem;margin-bottom:.15rem;}
.step-text p{color:#8b949e;font-size:.8rem;line-height:1.5;}

/* FOOTER */
footer{text-align:center;padding:1.5rem;color:#6e7681;font-size:.78rem;
       border-top:1px solid #21262d;margin-top:3rem;}
</style>
</head>
<body>

<!-- NAV -->
<nav>
  <div class="logo">Phishing<span>ML</span> &mdash; URL Detector</div>
  <ul>
    <li><a href="#scanner">Scanner</a></li>
    <li><a href="#examples">Examples</a></li>
    <li><a href="#how">How It Works</a></li>
    <li><a href="#log">Log</a></li>
  </ul>
</nav>

<!-- HERO -->
<div class="hero">
  <div class="badge-ml">&#129302; Machine Learning Powered</div>
  <h1>Phishing URL Detection<br/><span>Using Machine Learning</span></h1>
  <p>Enter any URL and our Logistic Regression model analyzes 18 features to instantly determine whether it is legitimate or a phishing attack.</p>
</div>

<!-- STATS -->
<div class="stats-bar">
  <div class="stat"><div class="stat-num" id="s-total">0</div><div class="stat-label">URLs Scanned</div></div>
  <div class="stat"><div class="stat-num" id="s-phish">0</div><div class="stat-label">Phishing Found</div></div>
  <div class="stat"><div class="stat-num" id="s-legit">0</div><div class="stat-label">Legitimate</div></div>
  <div class="stat"><div class="stat-num" id="s-susp">0</div><div class="stat-label">Suspicious</div></div>
  <div class="stat"><div class="stat-num" id="s-acc" style="color:#a371f7">18</div><div class="stat-label">Features Used</div></div>
</div>

<div class="container">

  <!-- SCANNER CARD -->
  <div class="card" id="scanner" style="margin-bottom:1.5rem">
    <h2><span class="icon">&#128279;</span> URL Scanner</h2>
    <div class="url-row">
      <input type="text" id="url-input" placeholder="https://example.com/login or paste any suspicious URL..."
             onkeydown="if(event.key==='Enter') scanURL()"/>
      <button class="btn-scan" onclick="scanURL()">&#9889; Detect</button>
    </div>
    <div style="margin-top:.6rem;display:flex;gap:.5rem;flex-wrap:wrap">
      <button class="btn-sm" onclick="fillURL('http://paypal-secure-update.tk/account/verify')">Try Phishing URL</button>
      <button class="btn-sm" onclick="fillURL('https://www.google.com/search?q=python')">Try Legit URL</button>
      <button class="btn-sm" onclick="fillURL('http://192.168.1.1/login.html')">Try IP URL</button>
    </div>

    <!-- RESULT -->
    <div id="result">
      <div class="result-box" id="result-box">
        <div class="verdict-row">
          <div class="verdict-text" id="r-verdict"></div>
          <div class="risk-badge" id="r-risk"></div>
        </div>
        <div style="font-size:.82rem;color:#8b949e;margin-bottom:.85rem;word-break:break-all" id="r-url"></div>

        <div class="score-row">
          <div class="score-item">
            <div class="score-num" id="r-phish-score"></div>
            <div class="score-label">Phishing Probability</div>
          </div>
          <div class="score-item">
            <div class="score-num" id="r-legit-score"></div>
            <div class="score-label">Legitimate Probability</div>
          </div>
        </div>

        <div class="progress-wrap">
          <div class="progress-label">
            <span>Phishing Risk</span>
            <span id="r-bar-label"></span>
          </div>
          <div class="progress-bg">
            <div class="progress-fill" id="r-bar" style="width:0%"></div>
          </div>
        </div>

        <div style="font-size:.85rem;color:#8b949e;margin-bottom:.6rem;font-weight:600">Feature Analysis:</div>
        <div class="feature-grid" id="r-features"></div>
      </div>
    </div>
  </div>

  <!-- ROW 2 -->
  <div class="grid" style="margin-bottom:1.5rem">

    <!-- EXAMPLES -->
    <div class="card" id="examples">
      <h2><span class="icon">&#128196;</span> Example URLs</h2>
      <div class="ex-list">
        <div class="ex-item">
          <span class="ex-url">https://www.google.com</span>
          <span class="ex-label" style="background:#22c55e22;color:#22c55e">Legitimate</span>
        </div>
        <div class="ex-item">
          <span class="ex-url">http://paypal-secure-update.tk/verify</span>
          <span class="ex-label" style="background:#ef444422;color:#ef4444">Phishing</span>
        </div>
        <div class="ex-item">
          <span class="ex-url">http://192.168.1.1/bank/login</span>
          <span class="ex-label" style="background:#ef444422;color:#ef4444">Phishing</span>
        </div>
        <div class="ex-item">
          <span class="ex-url">https://github.com/pallets/flask</span>
          <span class="ex-label" style="background:#22c55e22;color:#22c55e">Legitimate</span>
        </div>
        <div class="ex-item">
          <span class="ex-url">http://apple-id-suspended.pw/confirm</span>
          <span class="ex-label" style="background:#ef444422;color:#ef4444">Phishing</span>
        </div>
        <div class="ex-item">
          <span class="ex-url">http://g00gle-secure.site/accounts</span>
          <span class="ex-label" style="background:#f9731622;color:#f97316">Suspicious</span>
        </div>
      </div>
    </div>

    <!-- HOW IT WORKS -->
    <div class="card" id="how">
      <h2><span class="icon">&#129302;</span> How It Works</h2>
      <div class="step-list">
        <div class="step">
          <div class="step-num">1</div>
          <div class="step-text">
            <h4>Feature Extraction</h4>
            <p>18 signals are pulled from the URL: length, dots, hyphens, IP address, HTTPS, subdomains, suspicious TLD, suspicious keywords, entropy, and more.</p>
          </div>
        </div>
        <div class="step">
          <div class="step-num">2</div>
          <div class="step-text">
            <h4>Feature Scaling</h4>
            <p>All features are normalized using Standard Scaler so large numbers don't dominate the model.</p>
          </div>
        </div>
        <div class="step">
          <div class="step-num">3</div>
          <div class="step-text">
            <h4>ML Prediction</h4>
            <p>A Logistic Regression model (trained on 40 URLs) outputs a probability score between 0% and 100% for phishing.</p>
          </div>
        </div>
        <div class="step">
          <div class="step-num">4</div>
          <div class="step-text">
            <h4>Verdict</h4>
            <p>&ge;70% = Phishing &nbsp;|&nbsp; 40-69% = Suspicious &nbsp;|&nbsp; &lt;40% = Legitimate</p>
          </div>
        </div>
      </div>
    </div>

  </div>

  <!-- FEATURE INFO -->
  <div class="card" style="margin-bottom:1.5rem">
    <h2><span class="icon">&#128200;</span> 18 ML Features Explained</h2>
    <div class="grid-3">
      <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:.85rem">
        <div style="color:#a371f7;font-size:.8rem;font-weight:700;margin-bottom:.3rem">Structural</div>
        <ul style="color:#8b949e;font-size:.78rem;line-height:1.9;padding-left:1rem">
          <li>URL Length</li><li>Domain Length</li><li>Path Length</li>
          <li>Query Length</li><li>Dot Count</li><li>Slash Count</li>
        </ul>
      </div>
      <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:.85rem">
        <div style="color:#f97316;font-size:.8rem;font-weight:700;margin-bottom:.3rem">Content Signals</div>
        <ul style="color:#8b949e;font-size:.78rem;line-height:1.9;padding-left:1rem">
          <li>Suspicious Words</li><li>Suspicious TLD</li><li>@ Symbol</li>
          <li>Hyphen Count</li><li>Digit Count</li><li>Special Chars</li>
        </ul>
      </div>
      <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:.85rem">
        <div style="color:#58a6ff;font-size:.8rem;font-weight:700;margin-bottom:.3rem">Trust Indicators</div>
        <ul style="color:#8b949e;font-size:.78rem;line-height:1.9;padding-left:1rem">
          <li>Has HTTPS</li><li>Has IP Address</li><li>Subdomain Count</li>
          <li>Has Port</li><li>URL Entropy</li><li>Trusted Domain</li>
        </ul>
      </div>
    </div>
  </div>

  <!-- SCAN LOG -->
  <div class="card" id="log">
    <h2><span class="icon">&#128203;</span> Scan History</h2>
    <div class="log-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th><th>Time</th><th>URL Preview</th>
            <th>Verdict</th><th>Phish %</th>
          </tr>
        </thead>
        <tbody id="log-body">
          <tr><td colspan="5" style="color:#6e7681;text-align:center;padding:1.2rem">No scans yet.</td></tr>
        </tbody>
      </table>
    </div>
  </div>

</div>

<footer>
  Phishing URL Detection Using Machine Learning &mdash; Built with Python, Flask &amp; scikit-learn &nbsp;|&nbsp; For educational purposes only
</footer>

<script>
let totals = {total:0, phish:0, legit:0, susp:0};
let logCount = 0;

function fillURL(url) {
  document.getElementById("url-input").value = url;
  scanURL();
}

async function scanURL() {
  const url = document.getElementById("url-input").value.trim();
  if (!url) { alert("Please enter a URL to scan."); return; }

  const btn = document.querySelector(".btn-scan");
  btn.textContent = "Scanning...";
  btn.disabled = true;

  try {
    const res  = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url })
    });
    const d = await res.json();

    // verdict
    document.getElementById("result").style.display = "block";
    document.getElementById("r-verdict").textContent = d.verdict;
    document.getElementById("r-verdict").style.color = d.verdict_color;
    document.getElementById("r-url").textContent = d.url;

    const risk = document.getElementById("r-risk");
    risk.textContent = d.risk_level;
    risk.style.background = d.verdict_color + "22";
    risk.style.color = d.verdict_color;

    document.getElementById("r-phish-score").textContent = d.phish_score + "%";
    document.getElementById("r-phish-score").style.color = d.verdict_color;
    document.getElementById("r-legit-score").textContent = d.legit_score + "%";
    document.getElementById("r-legit-score").style.color = "#22c55e";

    const bar = document.getElementById("r-bar");
    bar.style.width = d.phish_score + "%";
    bar.style.background = d.verdict_color;
    document.getElementById("r-bar-label").textContent = d.phish_score + "%";

    // feature breakdown
    const fg = document.getElementById("r-features");
    fg.innerHTML = d.breakdown.map(f =>
      `<div class="feat-item">
        <span class="feat-name">${f.name}</span>
        <span class="feat-val ${f.flag ? 'feat-flag' : ''}">${f.value}${f.flag ? ' ⚑' : ''}</span>
      </div>`
    ).join("");

    // stats
    totals.total++;
    if (d.verdict === "PHISHING")    totals.phish++;
    else if (d.verdict === "LEGITIMATE") totals.legit++;
    else totals.susp++;
    document.getElementById("s-total").textContent = totals.total;
    document.getElementById("s-phish").textContent = totals.phish;
    document.getElementById("s-legit").textContent = totals.legit;
    document.getElementById("s-susp").textContent  = totals.susp;

    // log
    logCount++;
    const tbody = document.getElementById("log-body");
    if (logCount === 1) tbody.innerHTML = "";
    const preview = d.url.length > 45 ? d.url.substring(0,45) + "…" : d.url;
    const now = new Date().toLocaleTimeString();
    tbody.insertAdjacentHTML("afterbegin", `<tr>
      <td>${logCount}</td>
      <td>${now}</td>
      <td><code style="font-size:.78rem">${escHtml(preview)}</code></td>
      <td><span class="badge" style="background:${d.verdict_color}22;color:${d.verdict_color}">${d.verdict}</span></td>
      <td style="color:${d.verdict_color};font-weight:700">${d.phish_score}%</td>
    </tr>`);

    // scroll to result
    document.getElementById("result").scrollIntoView({behavior:"smooth", block:"nearest"});

  } finally {
    btn.textContent = "\u26A1 Detect";
    btn.disabled = false;
  }
}

function escHtml(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────
#  6.  FLASK ROUTES
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Receive a URL, run ML prediction, return JSON."""
    data = request.get_json()
    url  = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    result = predict_url(url)

    scan_log.append({
        "time":        datetime.datetime.now().isoformat(),
        "url":         result["url"],
        "verdict":     result["verdict"],
        "phish_score": result["phish_score"],
    })

    return jsonify(result)


@app.route("/api/logs")
def api_logs():
    """Return all scan logs as JSON."""
    return jsonify(scan_log)


# ─────────────────────────────────────────────────────────────
#  7.  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 58)
    print("  Phishing URL Detection Using Machine Learning")
    print("  Open your browser at: http://127.0.0.1:5000")
    print("=" * 58)
    app.run(debug=True)
