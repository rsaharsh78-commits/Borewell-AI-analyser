# 💧 JalDrishti जलदृष्टि — Borewell Suitability & Risk Analysis Tool

> **Know Before You Drill** — AI-powered groundwater analysis for Hyderabad region using real CGWB data, satellite APIs, and ML models.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![Leaflet](https://img.shields.io/badge/Leaflet-1.9-brightgreen?logo=leaflet)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📸 Screenshots

> _Add your screenshots here after deploying._
>
> | Map View | Risk Analysis | AI Advisory |
> |----------|--------------|-------------|
> | ![Map](screenshots/map.png) | ![Risk](screenshots/risk.png) | ![AI](screenshots/ai.png) |

---

## 🎯 Problem Statement

In India, **70% of borewells fail** in hard-rock terrain due to poor site selection. Farmers and homeowners spend ₹2–5 lakhs drilling without knowing if water exists underground. JalDrishti solves this by combining **real government groundwater data**, **satellite APIs**, and **machine learning** to predict borewell success before a single rupee is spent.

---

## 🌟 Features

- 🗺️ **Interactive Satellite Map** — Click anywhere on Esri/Sentinel-2 imagery to get instant risk analysis
- 🤖 **ML Risk Prediction** — XGBoost model trained on 36 Hyderabad mandals with CGWB ground truth data
- 📊 **SHAP Explainability** — Visual breakdown of which factors increase/decrease drilling risk
- 🎯 **Precision Mode** — Crosshair targeting for exact coordinate selection
- 📍 **GPS Location Detection** — One-click current location via HTML5 Geolocation API
- 🔍 **Find Best Spot** — Radius-based grid search to find the optimal drilling location nearby
- 📅 **Seasonal Risk Matrix** — Month-by-month drilling risk calendar showing best times to drill
- 🌧️ **Rainfall History** — 5-year precipitation & evapotranspiration trend charts
- 🧠 **AI Drilling Advisor** — Natural-language advice powered by Featherless AI (Qwen3-14B)
- 🖨️ **Print Report** — Export analysis as a printable PDF-ready report
- 🔥 **Risk Heatmap** — Overlay showing risk across all 36 mandals

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (Leaflet.js)                 │
│   index.html + style.css + app.js (Glassmorphism UI)    │
└───────────────┬─────────────────────────────────────────┘
                │ REST API (JSON)
┌───────────────▼─────────────────────────────────────────┐
│                 FastAPI Backend (Python)                 │
│                                                         │
│  ┌──────────┐ ┌──────────────┐ ┌─────────────────────┐  │
│  │predict.py│ │feature_engine│ │   grid_analyzer.py   │  │
│  │(7 routes)│ │  .py (APIs)  │ │ (spatial grid search)│  │
│  └──────────┘ └──────────────┘ └─────────────────────┘  │
│  ┌──────────────┐ ┌─────────────┐ ┌─────────────────┐   │
│  │risk_matrix.py│ │geo_lookup.py│ │ ai_advisory.py  │   │
│  │(seasonal risk)│ │ (geocoding) │ │(Featherless AI) │   │
│  └──────────────┘ └─────────────┘ └─────────────────┘   │
└───────────────┬─────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────┐
│              ML Models (XGBoost + SHAP)                 │
│  risk_model.joblib │ depth_model.joblib │ feature_config │
└───────────────┬─────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────┐
│                  External Data APIs                     │
│  Open-Meteo │ Open-Elevation │ SoilGrids │ Nominatim   │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
JalDrishti-BorewellPredictor/
├── app/                           # FastAPI backend
│   ├── main.py                    # Server entry point
│   ├── api/predict.py             # All API endpoints (8 routes)
│   ├── schemas/                   # Pydantic request/response models
│   └── services/
│       ├── feature_engine.py      # Parallel API calls + ML feature extraction
│       ├── grid_analyzer.py       # Spatial grid analysis for "Find Best Spot"
│       ├── risk_matrix.py         # 12-month seasonal risk computation
│       ├── geo_lookup.py          # Nominatim geocoding
│       └── ai_advisory.py         # Featherless AI integration (Qwen3-14B)
├── frontend/                      # Static web frontend
│   ├── index.html                 # Main page (glassmorphism UI)
│   ├── style.css                  # Custom CSS with dark mode
│   └── app.js                     # Leaflet map + dashboard logic
├── data/                          # CGWB ground truth data
│   ├── hyderabad_mandals.json     # 36 mandals with GWL, extraction %, aquifer data
│   ├── hyderabad_waterbodies.json # Major water bodies for proximity analysis
│   ├── hyderabad_lineaments.json  # Geological lineament/fracture zones
│   ├── district_lookup.json       # District boundary mapping
│   └── processed/                 # Training CSV (auto-generated)
├── models/                        # Pre-trained ML models
│   ├── risk_model.joblib          # XGBoost classifier (risk prediction)
│   ├── depth_model.joblib         # XGBoost regressor (depth recommendation)
│   ├── feature_config.joblib      # Feature encoders & scalers
│   └── shap_summary.png           # SHAP feature importance plot
├── ml/                            # Model training scripts
│   ├── train_model.py             # Training pipeline
│   └── feature_config.py          # Feature definitions
├── requirements.txt
├── .gitignore
├── test_api.py                    # API test suite
└── README.md
```

---

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

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Main risk prediction with SHAP factors |
| `/api/grid-analysis` | POST | Grid search for best spot in radius |
| `/api/ai-advisory` | POST | AI-generated drilling advice (Featherless AI) |
| `/api/rainfall-history` | GET | 5-year rainfall & ET₀ trends |
| `/api/monthly-risk` | GET | 12-month seasonal risk calendar |
| `/api/heatmap` | GET | All mandals heatmap data |
| `/api/geocode` | GET | Address search (Nominatim) |
| `/api/health` | GET | Server health check |

---

## 🌐 Data Sources

All core data sources are **free and require no API keys**:

| Source | Data Provided | Usage |
|--------|--------------|-------|
| **CGWB** | Ground water levels, extraction %, aquifer type | Local JSON (36 mandals) |
| **Open-Meteo** | Daily rainfall & evapotranspiration (5 years) | REST API (free) |
| **Open-Elevation** | Elevation & slope analysis | REST API (free) |
| **SoilGrids** | Clay %, sand %, silt %, soil pH, depth to bedrock | REST API (free) |
| **Nominatim** | Reverse geocoding (location names) | REST API (free) |
| **Esri / OSM** | Satellite & street map tiles | Tile server (free) |
| **Featherless AI** | AI advisory text generation (Qwen3-14B) | REST API (key included) |

---

## 🧠 ML Model Details

| Aspect | Details |
|--------|---------|
| **Algorithm** | XGBoost (classification for risk, regression for depth) |
| **Features** | 22 features: hydrology, geology, terrain, soil, user inputs |
| **Training Data** | Synthetic dataset generated from 36 real CGWB mandal profiles |
| **Explainability** | SHAP TreeExplainer for per-prediction factor attribution |
| **Risk Classes** | Low (0-25), Moderate (26-50), High (51-75), Critical (76-100) |
| **Depth Model** | Recommends optimal drilling depth range (ft) |

### Key Features Used by the Model

| Category | Features |
|----------|----------|
| **Hydrology** | Monsoon rainfall, recharge rate, pre/post-monsoon GWL |
| **Geology** | Rock type, weathered zone thickness, lineament density |
| **Terrain** | Slope, elevation, distance to water body, impervious surface % |
| **Soil** | Clay %, sand %, soil pH, depth to bedrock |
| **Context** | Extraction stage %, planned depth, planned month |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML5, CSS3 (Glassmorphism), Vanilla JavaScript |
| **Mapping** | Leaflet.js 1.9 with Esri/Sentinel-2/OSM tile layers |
| **Backend** | FastAPI (Python) with Uvicorn ASGI server |
| **ML** | XGBoost, SHAP, scikit-learn, pandas, numpy |
| **AI** | Featherless AI API (Qwen/Qwen3-14B) |
| **APIs** | Open-Meteo, Open-Elevation, SoilGrids, Nominatim |

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🙏 Acknowledgments

- **Central Ground Water Board (CGWB)** — Real groundwater data for Telangana
- **Open-Meteo** — Free weather & climate API
- **SoilGrids (ISRIC)** — Global soil data
- **Featherless AI** — LLM inference API
- **Leaflet.js** — Open-source interactive maps

---

## 📄 License

MIT License — free for personal and commercial use.

---

<p align="center">
  Made with 💧 for India's groundwater future
</p>
