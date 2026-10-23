# Phishing URL Detection Using Machine Learning

> A beginner-friendly cybersecurity web application built with **Python**, **Flask**, and **scikit-learn**.  
> Detects phishing URLs in real time using a Logistic Regression model trained on 18 hand-crafted URL features.

---

## 📸 Features

| Feature | Description |
|---|---|
| 🔗 **URL Scanner** | Paste any URL and get an instant verdict with probability score |
| 🧠 **ML Model** | Logistic Regression trained at startup — no external dataset needed |
| 📊 **18 Feature Analysis** | Every scan shows all 18 URL features with suspicious ones flagged |
| 📈 **Live Stats Bar** | Tracks total scans, phishing, legitimate, and suspicious counts |
| 📋 **Scan History Log** | Full history table of every URL checked with timestamp and verdict |
| 🎨 **Dark Theme UI** | Modern dark-themed single-page interface |
| 🔌 **REST API** | `/api/predict` and `/api/logs` endpoints for programmatic use |

---

## 🗂️ Project Structure

```
project/
│
├── app.py               ← Main application (all code in one file)
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
└── PROJECT_REPORT.md    ← Detailed project report
```

---

## ⚙️ Requirements

- Python 3.9 or higher
- pip

---

## 🚀 Installation & Running

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the application

```bash
python app.py
```

### 3. Open in browser

```
http://127.0.0.1:5000
```

---

## 🧪 Example URLs to Test

| URL | Expected Verdict |
|---|---|
| `https://www.google.com` | ✅ LEGITIMATE |
| `https://github.com/pallets/flask` | ✅ LEGITIMATE |
| `http://paypal-secure-update.tk/account/verify` | 🔴 PHISHING |
| `http://192.168.1.1/login.html` | 🔴 PHISHING |
| `http://apple-id-suspended.pw/confirm` | 🔴 PHISHING |
| `http://g00gle-secure.site/accounts` | 🟠 SUSPICIOUS |

---

## 🧠 How the ML Model Works

```
URL Input
   │
   ▼
Feature Extraction  (18 signals pulled from the URL)
   │
   ▼
StandardScaler      (normalize features)
   │
   ▼
Logistic Regression (predict probability 0–100%)
   │
   ▼
Verdict:  ≥70% → PHISHING | 40–69% → SUSPICIOUS | <40% → LEGITIMATE
```

---

## 📐 The 18 ML Features

### Structural
- URL Length, Domain Length, Path Length, Query Length, Dot Count, Slash Count

### Content Signals
- Suspicious Words, Suspicious TLD, @ Symbol, Hyphen Count, Digit Count, Special Chars

### Trust Indicators
- Has HTTPS, Has IP Address, Subdomain Count, Has Port, URL Entropy, Trusted Domain

---

## 🔌 API Reference

### `POST /api/predict`

**Request body:**
```json
{ "url": "http://example.com/login" }
```

**Response:**
```json
{
  "url": "http://example.com/login",
  "verdict": "PHISHING",
  "verdict_color": "#ef4444",
  "risk_level": "HIGH RISK",
  "phish_score": 85.3,
  "legit_score": 14.7,
  "breakdown": [
    { "name": "URL Length", "value": 28, "flag": false },
    ...
  ]
}
```

### `GET /api/logs`

Returns a JSON array of all scans performed in the current session.

---

## 🛑 Disclaimer

This project is **for educational purposes only**.  
Do not use it as the sole tool for making security decisions.

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3, Flask |
| ML | scikit-learn (Logistic Regression, StandardScaler) |
| Numerics | NumPy |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Deployment | Flask dev server (for learning); use Gunicorn for production |
