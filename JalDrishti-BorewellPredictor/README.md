# 💧 JalDrishti जलदृष्टि — Borewell Suitability & Risk Analysis Tool

> **Know Before You Drill** — AI-powered groundwater analysis for Hyderabad region using real CGWB data, satellite APIs, and ML models.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![Leaflet](https://img.shields.io/badge/Leaflet-1.9-brightgreen?logo=leaflet)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Features

- **Interactive Map** — Click anywhere on satellite imagery to get instant borewell risk analysis
- **ML Risk Prediction** — XGBoost model trained on 36 Hyderabad mandals with CGWB ground truth data
- **SHAP Explainability** — Visual breakdown of which factors increase/decrease risk
- **Precision Mode** — Crosshair targeting for exact coordinate selection
- **📍 GPS Location Detection** — One-click current location via HTML5 Geolocation API
- **Find Best Spot** — Radius-based grid search for optimal drilling location
- **Seasonal Risk Matrix** — Month-by-month drilling risk calendar
- **Rainfall History** — 5-year precipitation & ET₀ trends
- **Print Report** — Export analysis as a printable report

## 📁 Project Structure

```
borewell-predictor/
├── app/                        # FastAPI backend
│   ├── main.py                 # Server entry point
│   ├── api/predict.py          # All API endpoints
│   ├── schemas/                # Pydantic models
│   └── services/
│       ├── feature_engine.py   # API calls + ML features
│       ├── grid_analyzer.py    # Spatial grid analysis
│       ├── risk_matrix.py      # Seasonal risk computation
│       └── geo_lookup.py       # Nominatim geocoding
├── frontend/                   # Static web frontend
│   ├── index.html              # Main page
│   ├── style.css               # Glassmorphism UI
│   └── app.js                  # Leaflet map + dashboard
├── data/                       # CGWB ground truth data
│   ├── hyderabad_mandals.json  # 36 mandals with GWL/extraction
│   ├── hyderabad_waterbodies.json
│   ├── hyderabad_lineaments.json
│   └── processed/              # Training CSV
├── models/                     # Trained ML models
│   ├── risk_model.joblib       # XGBoost classifier
│   ├── depth_model.joblib      # Depth regressor
│   └── feature_config.joblib   # Feature encoders
├── ml/                         # Model training scripts
│   └── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/JalDrishti-BorewellPredictor.git
cd JalDrishti-BorewellPredictor

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Server

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in your browser.

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Main risk prediction |
| `/api/grid-analysis` | POST | Grid search for best spot |
| `/api/rainfall-history` | GET | 5-year rainfall trends |
| `/api/monthly-risk` | GET | 12-month risk calendar |
| `/api/heatmap` | GET | All mandals heatmap data |
| `/api/geocode` | GET | Address search (Nominatim) |
| `/api/health` | GET | Server health check |

## 🌐 Data Sources (Free, No API Keys Required)

| Source | Data Provided |
|--------|--------------|
| **CGWB** | Ground water levels, extraction %, aquifer type |
| **Open-Meteo** | 5-year daily rainfall & evapotranspiration |
| **Open-Elevation** | Elevation & slope analysis |
| **SoilGrids** | Clay %, sand %, soil pH |
| **Nominatim** | Reverse geocoding |
| **Esri/OSM** | Satellite & street map tiles |

## 🧠 ML Model

- **Algorithm**: XGBoost (classification for risk, regression for depth)
- **Features**: 22 features including hydrology, geology, terrain, soil, and user inputs
- **Training Data**: Synthetic dataset generated from 36 real CGWB mandal profiles
- **Explainability**: SHAP TreeExplainer for per-prediction factor attribution

## 📄 License

MIT License — free for personal and commercial use.
