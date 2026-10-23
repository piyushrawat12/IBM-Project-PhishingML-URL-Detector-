# Project Report

## Phishing URL Detection Using Machine Learning

---

<div align="center">

**Submitted by**

| Field | Details |
|---|---|
| **Name** | Piyush Rawat |
| **Roll Number** | 2400001730165 |
| **Branch** | BTech Cyber Security |
| **Semester / Year** | 5th Semester, 3rd Year |
| **Institution** | Faculty of Technology |
| **University** | Veer Madho Singh Bhandari Uttarakhand Technical University |

</div>

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [System Architecture](#4-system-architecture)
5. [Feature Engineering](#5-feature-engineering)
6. [Machine Learning Model](#6-machine-learning-model)
7. [Web Application](#7-web-application)
8. [Verdict Thresholds](#8-verdict-thresholds)
9. [Limitations](#9-limitations)
10. [Future Improvements](#10-future-improvements)
11. [How to Run](#11-how-to-run)
12. [Dependencies](#12-dependencies)
13. [Conclusion](#13-conclusion)

---

## 1. Introduction

Phishing is one of the most prevalent and damaging forms of cybercrime in the
digital age. Attackers craft malicious URLs that closely mimic the appearance
of legitimate websites — such as banks, payment platforms, and social media
networks — to trick users into surrendering their credentials, financial
details, and personal information.

Traditional defenses such as blacklists are reactive; they can only block
URLs that have already been reported. A machine-learning-based approach
allows the system to generalize from known patterns and flag previously
unseen phishing URLs proactively.

This project implements a **real-time phishing URL detector** as a fully
functional web application. A user submits any URL and immediately receives:

- A verdict: **LEGITIMATE**, **SUSPICIOUS**, or **PHISHING**
- A probability score (0–100%)
- A breakdown of all 18 URL features analyzed
- A history log of all scans in the session

---

## 2. Problem Statement

> Given a URL string, classify it as **Legitimate (0)** or **Phishing (1)**
> using machine learning techniques applied to structural and content-based
> features extracted from the URL itself — without making any network
> requests to the target server.

---

## 3. Objectives

1. Study the characteristics that distinguish phishing URLs from legitimate ones.
2. Design and implement 18 measurable feature functions for URL analysis.
3. Train a Logistic Regression classifier on a labeled URL dataset embedded in the code.
4. Build a clean, user-friendly web interface using Flask and a modern dark theme.
5. Display a transparent per-feature breakdown alongside every prediction.
6. Keep the entire implementation in a single, beginner-readable Python file.

---

## 4. System Architecture

```
┌──────────────────────────────────────────┐
│            Browser (User)                │
│   Dark-theme single-page web interface   │
└───────────────────┬──────────────────────┘
                    │  HTTP  POST /api/predict
                    ▼
┌──────────────────────────────────────────┐
│         Flask Web Server (app.py)        │
│  Route: GET  /            → HTML Page    │
│  Route: POST /api/predict → ML Predict   │
│  Route: GET  /api/logs    → Scan History │
└───────────────────┬──────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│       Feature Extraction Layer           │
│  18 Python functions extract numeric     │
│  signals from the raw URL string         │
└───────────────────┬──────────────────────┘
                    │  numpy array  shape=(1, 18)
                    ▼
┌──────────────────────────────────────────┐
│     StandardScaler  (scikit-learn)       │
│  Normalizes each feature to mean=0,      │
│  std=1 so large values don't dominate    │
└───────────────────┬──────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│  Logistic Regression  (scikit-learn)     │
│  Trained at startup on 40 labeled URLs   │
│  Outputs probability  P(phishing) ∈ [0,1]│
└───────────────────┬──────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│            Verdict Engine                │
│  P ≥ 0.70  →  PHISHING   (HIGH RISK)    │
│  P ≥ 0.40  →  SUSPICIOUS (MEDIUM RISK)  │
│  P <  0.40 →  LEGITIMATE (LOW RISK)     │
└──────────────────────────────────────────┘
```

---

## 5. Feature Engineering

### 5.1 Structural Features

| # | Feature | Description | Phishing Signal |
|---|---|---|---|
| 1 | URL Length | Total character count of the URL | Very long URLs hide the real domain |
| 2 | Dot Count | Number of `.` characters | Many dots suggest deep subdomains |
| 3 | Hyphen Count | Number of `-` characters | `paypal-secure-login.tk` is a common pattern |
| 4 | Slash Count | Number of `/` characters | Deep paths are used to obscure intent |
| 5 | Domain Length | Length of the host/domain part | Longer domains are often fake |
| 6 | Path Length | Length of the URL path | Long paths hide malicious redirects |
| 7 | Query Length | Length of query string after `?` | Excessive parameters suggest token harvesting |

### 5.2 Content-Based Signals

| # | Feature | Description | Phishing Signal |
|---|---|---|---|
| 8 | @ Symbol | Count of `@` in the URL | Browser ignores everything before `@` — used to deceive |
| 9 | Digit Count | Count of numeric characters | High digit count suggests IP obfuscation |
| 10 | Suspicious TLD | Matches free/abused TLDs (.tk, .ml, .xyz, .pw…) | Free TLDs are heavily exploited by attackers |
| 11 | Suspicious Word Count | Count of words like `login`, `verify`, `secure`… | Social engineering bait words |
| 12 | Special Characters | Count of `!#$%^&*` etc. | Used for encoding tricks and obfuscation |
| 13 | Subdomain Count | Number of subdomains (parts before root domain) | `a.b.c.evil.com` hides the real registrant |

### 5.3 Trust Indicators

| # | Feature | Description | Phishing Signal |
|---|---|---|---|
| 14 | Has HTTPS | 1 if scheme is `https://`, else 0 | Legitimate sites almost always use HTTPS |
| 15 | Has IP Address | 1 if host is a raw IPv4 address | Real companies use domain names, not IPs |
| 16 | Has Port | 1 if an explicit port like `:8080` is present | Unusual for production web services |
| 17 | URL Entropy | Shannon entropy of character distribution | High entropy → random-looking, obfuscated strings |
| 18 | Trusted Domain | 1 if domain is in known-safe whitelist | google.com, github.com etc. are baseline safe |

---

## 6. Machine Learning Model

### 6.1 Algorithm — Logistic Regression

Logistic Regression is a supervised binary classification algorithm. It learns
a weight `w_i` for each of the 18 features during training. Given a new URL,
the model computes:

```
z = w0 + w1*f1 + w2*f2 + ... + w18*f18
P(phishing) = 1 / (1 + e^(-z))
```

**Why Logistic Regression?**

- Simple, transparent, and interpretable — ideal for beginners
- Trains in milliseconds; no GPU required
- Natively outputs a calibrated probability (not just a label)
- Works well when features are informative and reasonably separable

### 6.2 Training Dataset

A hand-crafted dataset of **40 labeled URLs** is embedded directly in `app.py`:

| Class | Count | Sample URLs |
|---|---|---|
| Legitimate (0) | 20 | google.com, github.com, amazon.com, wikipedia.org |
| Phishing (1) | 20 | paypal-secure-update.tk, 192.168.1.1/login, apple-id-suspended.pw |

### 6.3 Training Pipeline

```python
# 1. Extract features from all 40 training URLs
X = np.array([extract_features(url) for url, _ in TRAINING_URLS])  # shape (40, 18)
y = np.array([label for _, label in TRAINING_URLS])                 # shape (40,)

# 2. Normalize features
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train the classifier
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_scaled, y)
```

The model is trained **once at application startup** and reused for all
incoming prediction requests during the session.

---

## 7. Web Application

### 7.1 Frontend

Built with plain **HTML5, CSS3, and Vanilla JavaScript** — no external frameworks.
All styles are inlined; the page is fully self-contained.

| Component | Description |
|---|---|
| Sticky Navigation Bar | Links to Scanner, Examples, How It Works, Log |
| Hero Banner | Title, ML badge, short description |
| Live Stats Bar | Real-time counters: scans, phishing, legitimate, suspicious |
| URL Scanner Card | Input box, quick-fill buttons, animated result panel |
| Result Panel | Verdict, risk badge, probability scores, progress bar, feature grid |
| Example URLs | 6 pre-labelled URLs for quick reference and testing |
| How It Works | 4-step ML pipeline explanation for beginners |
| 18 Features Panel | Grouped feature reference organized into 3 categories |
| Scan History Table | Timestamped log of every URL checked in the session |

### 7.2 Backend (Flask)

| Route | Method | Description |
|---|---|---|
| `/` | GET | Renders the full single-page HTML interface |
| `/api/predict` | POST | Accepts `{"url": "..."}`, returns prediction JSON |
| `/api/logs` | GET | Returns all in-session scan records as JSON array |

### 7.3 API Response Format

```json
{
  "url": "http://paypal-secure-update.tk/account/verify",
  "verdict": "PHISHING",
  "verdict_color": "#ef4444",
  "risk_level": "HIGH RISK",
  "phish_score": 91.4,
  "legit_score": 8.6,
  "breakdown": [
    { "name": "URL Length",      "value": 46,   "flag": false },
    { "name": "Suspicious TLD",  "value": 1,    "flag": true  },
    { "name": "Has HTTPS",       "value": 0,    "flag": true  },
    { "name": "Suspicious Words","value": 3,    "flag": true  }
  ]
}
```

---

## 8. Verdict Thresholds

| Phishing Probability | Verdict | Risk Level | UI Color |
|---|---|---|---|
| ≥ 70% | PHISHING | HIGH RISK | Red `#ef4444` |
| 40% – 69% | SUSPICIOUS | MEDIUM RISK | Orange `#f97316` |
| < 40% | LEGITIMATE | LOW RISK | Green `#22c55e` |

---

## 9. Limitations

| # | Limitation | Impact |
|---|---|---|
| 1 | Small training set (40 URLs) | Model may misclassify edge cases |
| 2 | No DNS / WHOIS lookups | Cannot verify domain registration age |
| 3 | No network requests to target | Cannot check page content or redirects |
| 4 | In-memory scan log | History lost when server restarts |
| 5 | Rule-based features only | Sophisticated obfuscation may bypass all signals |

---

## 10. Future Improvements

- [ ] Train on a real-world large dataset (PhishTank, OpenPhish, UCI Phishing Dataset)
- [ ] Add WHOIS and DNS-based features (domain age, registrar, IP reputation)
- [ ] Save trained model to disk using `joblib.dump()` for faster startup
- [ ] Replace Logistic Regression with Random Forest or XGBoost for higher accuracy
- [ ] Add persistent scan history using SQLite database
- [ ] Support bulk URL scanning via CSV file upload
- [ ] Add user authentication and per-user dashboards
- [ ] Deploy to cloud (Render, Railway, or Heroku)

---

## 11. How to Run

```bash
# Step 1 — Install all dependencies
pip install -r requirements.txt

# Step 2 — Start the Flask server
python app.py

# Step 3 — Open in browser
# http://127.0.0.1:5000
```

Expected console output:
```
Training ML model on startup...
Model ready.
==========================================================
  Phishing URL Detection Using Machine Learning
  Open your browser at: http://127.0.0.1:5000
==========================================================
```

---

## 12. Dependencies

| Package | Version | Purpose |
|---|---|---|
| `flask` | ≥ 3.0.0 | Web server and HTTP routing |
| `scikit-learn` | ≥ 1.3.0 | Logistic Regression, StandardScaler |
| `numpy` | ≥ 1.26.0 | Numerical arrays for feature vectors |

All listed in `requirements.txt`. Install with:
```bash
pip install -r requirements.txt
```

---

## 13. Conclusion

This project successfully demonstrates the application of machine learning
to a real-world cybersecurity challenge — phishing URL detection — in a
beginner-accessible, fully functional web application.

By extracting 18 carefully chosen features from raw URL strings and training
a Logistic Regression classifier, the system can accurately identify common
phishing URL patterns and provide a transparent, feature-level explanation
for every prediction.

The project reinforces core concepts in:
- **Cybersecurity** — phishing attack vectors and URL-based indicators of compromise
- **Machine Learning** — feature engineering, supervised classification, probability calibration
- **Web Development** — REST APIs, single-page applications, dark-theme UI design

The entire solution — ML training, REST API, and interactive dashboard — is
contained in a single Python file, making it straightforward to understand,
extend, and present.

---

*Submitted by: **Piyush Rawat** | Roll No: 240000173016*
*BTech Cyber Security — 5th Semester, 3rd Year*
*Faculty of Technology, Veer Madho Singh Bhandari Uttarakhand Technical University*
