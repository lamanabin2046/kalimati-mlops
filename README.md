# 🍅 Kalimati Tomato Price Intelligence System

> **MLOps Pipeline for Tomato Price Forecasting — Asian Institute of Technology**

[![CI/CD](https://github.com/lamanabin2046/kalimati-mlops/actions/workflows/deploy-lambdas.yml/badge.svg)](https://github.com/lamanabin2046/kalimati-mlops/actions)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![AWS](https://img.shields.io/badge/AWS-MLOps-orange)](https://aws.amazon.com/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)](https://xgboost.readthedocs.io/)

---

## 📋 Overview

An end-to-end MLOps system that automatically collects, processes, and forecasts tomato prices at the Kalimati Fruits and Vegetables Market in Kathmandu, Nepal. The system predicts prices for the next **1 to 7 days (t+1 to t+7)** using XGBoost models trained on multi-source data including historical prices, weather, fuel prices, exchange rates, and news events.

- **Live App:** http://52.201.197.75
- **Student:** Nabin Gangtan Lama (st125985@ait.ac.th)
- **Supervisor:** Dr. Chantri Polprasert
- **Institution:** Asian Institute of Technology (AIT)

---

## 🏗️ Architecture

```
EventBridge (Daily 1:15 AM UTC)
        │
        ▼
AWS Step Functions — Orchestration Pipeline
        │
        ├── PARALLEL: Data Collection
        │     ├── Lambda: weather_ingestion
        │     ├── Lambda: noc_diesel_scraper
        │     ├── Lambda: news_event_ingestion
        │     ├── Lambda: nrb_exchange_rate
        │     ├── Lambda: nrb_inflation
        │     ├── ECS Fargate: kalimati_scraper (Selenium)
        │     └── ECS Fargate: kalimati_supply_scraper (Selenium)
        │
        ├── Lambda: build_event_features
        │
        ├── Lambda: build_dataset
        │
        ├── SageMaker Training Job (XGBoost t+1 to t+7)
        │
        ├── Lambda: drift_detection
        │
        └── POST /api/reload-model → EC2 (FastAPI + React)
```

**Data Flow:**
```
Scrapers → S3 (Raw) → Lambda (Processing) → S3 (Processed) → SageMaker → S3 (Models) → FastAPI
```

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| **Scraping** | Python, Selenium, Requests, BeautifulSoup |
| **Compute** | AWS Lambda, ECS Fargate, EC2 (t2.micro) |
| **Orchestration** | AWS Step Functions |
| **Storage** | Amazon S3 (Data Lake) |
| **ML Training** | Amazon SageMaker, XGBoost |
| **Backend** | FastAPI (Python) |
| **Frontend** | React.js |
| **Scheduling** | Amazon EventBridge |
| **Monitoring** | AWS CloudWatch, Drift Detection, SNS Alerts |
| **CI/CD** | GitHub Actions |
| **Infrastructure** | AWS IAM, CloudWatch Logs |

---

## 📂 Project Structure

```
kalimati-mlops/
├── .github/
│   └── workflows/
│       └── deploy-lambdas.yml       # CI/CD pipeline
│
├── lambdas/
│   ├── drift-detector/              # Model & data drift detection
│   ├── weather-ingestion/           # Open-Meteo weather API
│   ├── noc-diesel-scraper/          # NOC fuel price scraper
│   ├── news-event-ingestion/        # RSS news scraper
│   ├── nrb-exchange-rate/           # NRB exchange rate scraper
│   ├── nrb-inflation/               # NRB inflation scraper
│   ├── build-event-features/        # Feature engineering
│   └── build-dataset/               # Dataset builder
│
├── scrapers/                        # Selenium-based scrapers (ECS Fargate)
│   ├── kalimati_scraper.py          # Kalimati daily price scraper
│   └── kalimati_supply_scraper.py   # Kalimati supply scraper
│
├── Tomato_price_prediction/
│   ├── modeling.py                  # XGBoost training (t+1 to t+7)
│   ├── preprocessing.py             # Feature preprocessing
│   └── data/                        # Local data (gitignored)
│
├── app/
│   ├── backend/                     # FastAPI REST API
│   │   ├── main.py
│   │   └── requirements.txt
│   └── frontend/                    # React web app
│       ├── src/
│       └── package.json
│
├── step-functions/
│   └── pipeline.json                # Step Functions state machine definition
│
├── .gitignore
└── README.md
```

---

## 🤖 ML Models

The system trains **7 separate XGBoost models**, one for each forecast horizon:

| Model | Forecast Horizon | Features |
|---|---|---|
| model_t1.pkl | Next day (t+1) | Price history, weather, diesel, exchange rate, news |
| model_t2.pkl | 2 days ahead (t+2) | Same features with lag adjustments |
| model_t3.pkl | 3 days ahead (t+3) | Same features |
| model_t4.pkl | 4 days ahead (t+4) | Same features |
| model_t5.pkl | 5 days ahead (t+5) | Same features |
| model_t6.pkl | 6 days ahead (t+6) | Same features |
| model_t7.pkl | 7 days ahead (t+7) | Same features |

**Features used:**
- Kalimati tomato price (historical)
- Kalimati supply volume
- Weather data (temperature, rainfall, humidity)
- NOC diesel price
- NRB exchange rate (USD/NPR)
- NRB inflation index
- News event indicators

---

## 🚀 CI/CD Pipeline

The project uses **GitHub Actions** for automated deployments:

```
Local Code Changes
      │
      ▼
git push origin main
      │
      ▼
GitHub Actions (.github/workflows/deploy-lambdas.yml)
      │
      ├── Configure AWS Credentials (from GitHub Secrets)
      ├── Deploy Lambda Functions (zip + aws lambda update)
      └── ✅ Auto-deployed to AWS
```

**GitHub Secrets required:**
```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_SESSION_TOKEN
```

---

## 📊 Data Sources

| Source | Data | Method |
|---|---|---|
| Kalimati Market | Daily tomato price & supply | Selenium (ECS Fargate) |
| Open-Meteo API | Weather data (Kathmandu) | Lambda + REST API |
| NOC Nepal | Diesel fuel prices | Lambda + Web scraping |
| Nepal Rastra Bank | Exchange rates, inflation | Lambda + Web scraping |
| RSS Feeds | News events | Lambda + RSS parsing |

---

## 🔍 Drift Detection

The system monitors for both **data drift** and **model drift**:

- **Data Drift** — Detects when input feature distributions shift significantly
- **Model Drift** — Monitors prediction accuracy over time
- **Alert** — SNS email notification when drift is detected
- **Action** — Triggers automatic retraining via Step Functions

---

## ⚙️ Setup & Deployment

### Prerequisites
- AWS Account with IAM permissions
- GitHub repository with AWS secrets configured
- Python 3.11+
- Node.js 18+ (for React frontend)

### Local Development

```bash
# Clone the repo
git clone https://github.com/lamanabin2046/kalimati-mlops.git
cd kalimati-mlops

# Backend
cd app/backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd app/frontend
npm install
npm start
```

### Deploy Lambda Functions

```bash
# Push to main branch — GitHub Actions deploys automatically
git add .
git commit -m "Update lambda function"
git push origin main
```

### Manual Lambda Deploy

```bash
cd lambdas/drift-detector
zip -r function.zip lambda_function.py
aws lambda update-function-code \
  --function-name drift-detection \
  --zip-file fileb://function.zip
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/predict` | Get 7-day price forecast |
| GET | `/api/history` | Get historical price data |
| POST | `/api/reload-model` | Reload ML model from S3 |
| GET | `/api/health` | Health check |

---

## 📈 Monitoring

- **CloudWatch Logs** — All Lambda and EC2 logs
- **Step Functions Console** — Pipeline execution history
- **Drift Detection** — Daily automated checks
- **SNS Alerts** — Email notifications for failures and drift

---

## 👤 Author

**Nabin Gangtan Lama**
- Student ID: st125985
- Email: st125985@ait.ac.th
- GitHub: [@lamanabin2046](https://github.com/lamanabin2046)

**Supervisor:** Dr. Chantri Polprasert
**Institution:** Asian Institute of Technology (AIT), Thailand

---

## 📄 License

This project is developed as part of an Project at AIT. All rights reserved.
