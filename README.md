# 🌊 FloodSense India: Advanced Geospatial Machine Learning, Explainable AI & Flood Risk Decision Support System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-EB4034.svg?style=flat)](https://xgboost.ai/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3+-2E7D32.svg?style=flat)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.42+-9C27B0.svg?style=flat)](https://shap.readthedocs.io/)
[![Tests](https://img.shields.io/badge/Pytest-20%20Passed-brightgreen.svg?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat)](https://opensource.org/licenses/MIT)
[![GitHub Repo](https://img.shields.io/badge/GitHub-SABARISH248%2Fflood-181717.svg?style=flat&logo=github&logoColor=white)](https://github.com/SABARISH248/flood)

---

## 📌 1. Executive Summary & Overview

**FloodSense India** is an enterprise-grade, end-to-end Geospatial Machine Learning (GIS+ML), Explainable Artificial Intelligence (XAI), and Decision Support Platform engineered for flood susceptibility modeling, exposure assessment, administrative priority zoning, and field report validation across Indian river basins and urban centers.

By integrating multi-source satellite remote sensing, digital elevation models (DEM), meteorological reanalysis rasters, and hydrological conditioning factors with an ensemble of five state-of-the-art machine learning algorithms (**XGBoost**, **Random Forest**, **LightGBM**, **Decision Tree**, **Logistic Regression**) and dual ensemble strategies (**Paper A Soft-Voting Probability Averaging** & **Paper B Stacking Meta-Learner**), FloodSense India delivers:
- **Pixel-level Flood Susceptibility Indices** at high spatial resolution.
- **Explainable AI (XAI)** point-level attribution waterfalls and feature drivers using TreeSHAP.
- **Composite Risk & Priority Zoning** across 4 administrative tiers (State, District, Block, Village) integrating population, infrastructure, and socio-economic vulnerability.
- **Localized Tamil Nadu 38-District Prediction Engine** with verified centroids, bounding boxes, and published IMD/CWC/SRTM/NLCMS environmental proxy parameters.
- **Dynamic Climate Sensitivity Simulation** for "What-If" precipitation fluctuations (-30% to +100%).
- **Interactive 12-Page Streamlit Dashboard** for visualization, scenario planning, and emergency response allocation.
- **Production-ready FastAPI REST API** with automated Swagger documentation for enterprise GIS integration.

---

## 🔬 2. Research Foundations & Dual Ensemble Architecture

FloodSense India operationalizes, extends, and benchmarks methodologies from two pioneering peer-reviewed flood susceptibility research papers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             RESEARCH FOUNDATIONS                            │
├────────────────────────────────────────┬────────────────────────────────────┤
│  Paper A: Rahimi et al. (2026)         │  Paper B: Demissie et al. (2024)   │
│  • 17-21 Geo-environmental factors     │  • Stacking Meta-Learner           │
│  • Soft-Voting probability averaging   │  • Jenks Natural Breaks (5 classes)│
│  • Equal-Interval classification (0.2) │  • Non-flood sampling Case I & II  │
│  • Prediction Rate Curve validation    │  • Multi-model comparative analysis│
└────────────────────────────────────────┴────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FLOODSENSE INDIA SCIENTIFIC EXTENSIONS                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Spatial Block Cross-Validation (Eliminating spatial autocorrelation leakage)│
│ • Leave-One-Flood-Year-Out temporal cross-validation                        │
│ • SHAP (SHapley Additive exPlanations) local & global feature attributions  │
│ • Multi-model Pixel-wise Uncertainty & Consensus estimation                 │
│ • Administrative Priority Zoning (Risk = Susceptibility × Exposure × Vuln)  │
│ • Tamil Nadu 38-District localized environmental feature engine             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Research-to-Code Implementation Matrix

| Scientific Concept | Research Reference | Module / Implementation | Description |
| :--- | :--- | :--- | :--- |
| **Conditioning Stack** | Paper A (Rahimi et al., 2026) | `src/data_loader.py` | Alignment and extraction of 21 standardized environmental features. |
| **Soft Voting Ensemble** | Paper A (Rahimi et al., 2026) | `src/predict.py::soft_voting` | Probability averaging across all base ML models: $P_{\text{vote}} = \frac{1}{K}\sum P_k$. |
| **Equal Interval Scheme** | Paper A (Rahimi et al., 2026) | `src/classify.py::equal_interval_classify` | Fixed 0.2 step intervals: Very Low (0-0.2) to Very High (0.8-1.0). |
| **Prediction Rate Curve** | Paper A (Rahimi et al., 2026) | `src/validate.py::calculate_prediction_rate` | Cumulative percentage of flood occurrences vs cumulative area ranked by risk. |
| **Stacking Meta-Learner** | Paper B (Demissie et al., 2024) | `src/predict.py::stacking_predict` | 2nd-stage Logistic Regression trained on base probability vectors $[P_{\text{rf}}, P_{\text{dt}}, P_{\text{lr}}, P_{\text{xgb}}, P_{\text{lgb}}]$. |
| **Jenks Natural Breaks** | Paper B (Demissie et al., 2024) | `src/classify.py::jenks_classify` | Optimization minimizing within-class variance and maximizing between-class variance. |
| **Case I & Case II Sampling** | Paper B (Demissie et al., 2024) | `src/spatial.py::generate_paper_b_non_flood_samples` | Case I: buffer distance from known floods; Case II: buffer from flood points + river stream distance. |
| **Spatial Block CV** | FloodSense Extension | `src/spatial.py::spatial_block_cv` | Checkerboard grid spatial blocking to prevent spatial autocorrelation data leakage. |
| **SHAP Local Waterfall** | FloodSense Extension | `src/explain.py::explain_point_prediction` | Exact Shapley attribution of individual features for any coordinate query. |
| **Ensemble Uncertainty** | FloodSense Extension | `src/uncertainty.py::calculate_pixel_uncertainty` | Pixel-wise standard deviation $\sigma(x)$ and inter-model consensus metric $C(x)$. |
| **Priority Zone Ranking** | FloodSense Extension | `src/risk.py::generate_priority_zones` | Multiplicative & Weighted Overlay risk scoring mapped to administrative boundaries. |
| **TN District Registry** | FloodSense Extension | `src/tn_districts.py` | All 38 Tamil Nadu districts with verified centroids, bounds, and IMD/CWC/SRTM metrics. |

---

## 🗺️ 3. 21 Environmental Conditioning Factors

The model ingests 21 rigorously structured spatial and hydrological conditioning features:

| Category | Feature Name | Units / Type | Environmental Role & Hydrological Rationale |
| :--- | :--- | :--- | :--- |
| **Location** | `Latitude` | Degrees (°N) | Regional climatological and latitudinal monsoon variation. |
| **Location** | `Longitude` | Degrees (°E) | Distance from coast, orographic divide, and monsoon corridor. |
| **Meteorology** | `Rainfall (mm)` | mm / year | Primary flood trigger; cumulative and extreme monsoon precipitation. |
| **Meteorology** | `Temperature (°C)` | °C | Controls evapotranspiration rates and soil moisture retention. |
| **Meteorology** | `Humidity (%)` | % | Atmospheric moisture saturation indicator preceding convective storms. |
| **Hydrology** | `River Discharge (m³/s)`| m³/s | Upstream catchment runoff volume and river channel capacity. |
| **Hydrology** | `Water Level (m)` | Meters (m) | River stage height relative to channel bankfull capacity. |
| **Topography** | `Elevation (m)` | Meters (m) | Digital Elevation Model (SRTM); lowlands and depressions accumulate floodwaters. |
| **Demographics** | `Population Density` | Persons / km² | Human exposure and impervious runoff multiplier in urbanized zones. |
| **Land Cover** | `Land Cover_Agricultural` | Binary (0 / 1) | Moderately permeable soil, seasonal inundation risk, irrigation networks. |
| **Land Cover** | `Land Cover_Desert` | Binary (0 / 1) | High permeability, rare pluvial flash flooding. |
| **Land Cover** | `Land Cover_Forest` | Binary (0 / 1) | High canopy interception, root water uptake, lowest surface runoff coefficient. |
| **Land Cover** | `Land Cover_Urban` | Binary (0 / 1) | Impervious concrete surfaces, rapid peak runoff, urban flash flood vulnerability. |
| **Land Cover** | `Land Cover_Water Body` | Binary (0 / 1) | Natural reservoirs, wetlands, perennial water bodies. |
| **Soil Properties**| `Soil Type_Clay` | Binary (0 / 1) | Low infiltration rate, high water retention, high ponding and runoff potential. |
| **Soil Properties**| `Soil Type_Loam` | Binary (0 / 1) | Balanced infiltration rate, moderate drainage capability. |
| **Soil Properties**| `Soil Type_Peat` | Binary (0 / 1) | Organic soil with very high initial saturation and water absorption. |
| **Soil Properties**| `Soil Type_Sandy` | Binary (0 / 1) | High hydraulic conductivity, rapid percolation, low ponding potential. |
| **Soil Properties**| `Soil Type_Silt` | Binary (0 / 1) | Medium permeability, vulnerable to riverbank erosion and mudflows. |
| **Anthropogenic** | `Infrastructure` | Index [0.0 - 1.0] | Built-up roads, drainage networks, embankments, storm sewers. |
| **History** | `Historical Floods` | Binary (0 / 1) | Verified historical recurrence from state disaster management records. |

---

## 🏛️ 4. Tamil Nadu 38-District Prediction Engine

FloodSense India includes a dedicated **Tamil Nadu Regional Module** (`src/tn_districts.py`, `app/views/tn_district_prediction.py`, `api/routes/tn_predict.py`) covering **all 38 administrative districts** of Tamil Nadu:

- **Full Spatial Specification:** Every district includes verified WGS-84 centroid coordinates and bounding boxes (`bbox: [min_lat, min_lon, max_lat, max_lon]`).
- **Empirical Proxy Calibration:** Environmental profiles are calibrated against published data from the **India Meteorological Department (IMD)** District Rainfall Atlas, **Central Water Commission (CWC)** river basin telemetry, **SRTM 30m Digital Elevation Model**, and **National Land Cover Monitoring System (NLCMS)**.
- **Diverse Hydrological Profiles:**
  - *High-Risk Coastal Delta Districts:* **Chennai** (6m elev, 72% urban, 55% clay, 1400mm rain), **Cuddalore** (12m elev, Vellar/Gadilam rivers, 1200mm rain), **Nagapattinam** (8m elev, 1350mm rain, Cauvery delta tip).
  - *Low-Risk Inland Elevated Districts:* **Erode** (192m elev, Western Ghats rain shadow, Cauvery basin), **Coimbatore** (416m elev, high drainage, 700mm rain), **Dindigul** (430m elev), **Krishnagiri** (380m elev).

```
                      TAMIL NADU 38-DISTRICT REGISTRY
 ┌───────────────────────┬─────────────────────────┬─────────────────────────┐
 │ Ariyalur              │ Chengalpattu            │ Chennai                 │
 │ Coimbatore            │ Cuddalore               │ Dharmapuri              │
 │ Dindigul              │ Erode                   │ Kallakurichi            │
 │ Kanchipuram           │ Kanyakumari             │ Karur                   │
 │ Krishnagiri           │ Madurai                 │ Mayiladuthurai          │
 │ Nagapattinam          │ Namakkal                │ Nilgiris                │
 │ Perambalur            │ Pudukottai              │ Ramanathapuram          │
 │ Ranipet               │ Salem                   │ Sivaganga               │
 │ Tenkasi               │ Thanjavur               │ Theni                   │
 │ Thoothukudi           │ Tiruchirappalli         │ Tirunelveli             │
 │ Tirupathur            │ Tiruppur                │ Tiruvallur              │
 │ Tiruvannamalai        │ Tiruvarur               │ Vellore                 │
 │ Viluppuram            │ Virudhunagar            │                         │
 └───────────────────────┴─────────────────────────┴─────────────────────────┘
```

---

## ⚙️ 5. Machine Learning & Ensemble Pipeline

```
  GeoTIFF Rasters (Rainfall, Elevation, Soil, LULC, etc.)
                          │
                          ▼
            Common CRS & Resampling Pipeline
               (data_loader.py: EPSG:4326)
                          │
                          ▼
             Feature Matrix Assembly & Validation
               (21 Ordered Conditioning Factors)
                          │
                          ▼
        ┌───────────────────────────────────┐
        │        5 Base ML Classifiers      │
        │ • XGBoost (XGBClassifier)         │
        │ • Random Forest (RF)              │
        │ • LightGBM (LGBMClassifier)       │
        │ • Decision Tree (DT)              │
        │ • Logistic Regression (LR)        │
        └─────────────────┬─────────────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
   [ Paper A Strategy ]        [ Paper B Strategy ]
       Soft-Voting              Stacking Meta-Learner
   Probability Averaging     Logistic Regression Meta-Model
            │                           │
            └─────────────┬─────────────┘
                          │
                          ▼
              Susceptibility Probability
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
     Equal Interval               Jenks Natural Breaks
    (0.0, 0.2, 0.4, ...)          (5 Optimized Classes)
            │                           │
            └─────────────┬─────────────┘
                          │
                          ▼
             Categorical Susceptibility
       (Very Low, Low, Moderate, High, Very High)
                          │
                          ▼
      + Exposure (Population, Infrastructure)
      + Vulnerability (Socio-Economic Index)
                          │
                          ▼
          Administrative Risk & Priority Zones
           (State -> District -> Block -> Village)
```

---

## 🖥️ 6. Streamlit Interactive Dashboard (12 Modules)

The interactive dashboard (`app/streamlit_app.py`) provides 12 specialized analytical views:

```
🧭 FloodSense Navigation Menu
├── 🏙️ TN District Prediction  → Comprehensive assessment for all 38 Tamil Nadu districts
├── 📊 Overview                → Executive KPI metric cards, high-level map, quick summaries
├── 🗺️ Susceptibility Map      → Interactive Folium map with raster overlays and layer switching
├── 🎯 Risk & Priority Zones   → Multiplicative & weighted overlay risk maps with league tables
├── 📍 Check My Location       → Lat/lon geocoding, real-time inference, and SHAP waterfall
├── 🌧️ Rainfall Scenario       → Dynamic climate sensitivity slider (-30% to +100%)
├── 📈 Model Validation        → ROC-AUC, Prediction Rate curves, Precision-Recall metrics
├── ⚖️ Model Uncertainty       → Multi-model standard deviation and consensus spatial heatmaps
├── 📜 Historical Events       → Replay past flood events (Chennai 2015, Kerala 2018, etc.)
├── 📝 Field Feedback          → Field surveyor ground-truth reporting with SQLite/GeoPackage
├── 📚 Data & Method           → Mathematical formulations, literature citations, documentation
└── 💻 System Health           → Automated pipeline diagnostics for rasters, models, and DB
```

---

## 🔌 7. FastAPI REST API Documentation

FloodSense India provides a high-performance REST API built with FastAPI.

### Interactive API Documentation
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

### API Endpoint Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | System identification, online status, version, and docs link. |
| `GET` | `/api/v1/health` | Comprehensive health check, feature schema, and loaded model status. |
| `POST` | `/api/v1/predict` | Single coordinate flood susceptibility inference with SHAP drivers. |
| `POST` | `/api/v1/risk` | Composite risk calculation combining susceptibility, exposure, and vulnerability. |
| `POST` | `/api/v1/geocode` | Reverse geocoding of latitude/longitude to administrative address. |
| `GET` | `/api/v1/districts` | Returns metadata, centroids, and bounds for all 38 Tamil Nadu districts. |
| `GET` | `/api/v1/districts/{id}/environment`| Returns the 21 environmental features for a specific district. |
| `POST` | `/api/v1/predict/district` | Computes flood probability, risk category, and key drivers for a district. |
| `POST` | `/api/v1/feedback` | Ingests ground-truth flood reports from field surveyors into SQLite. |
| `GET` | `/api/v1/feedback` | Queries historical field feedback records with optional spatial filters. |

### Example API Request & Response

#### 1. Point Susceptibility Prediction
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "lat": 13.0827,
       "lon": 80.2707,
       "model": "stacking"
     }'
```

**Response:**
```json
{
  "latitude": 13.0827,
  "longitude": 80.2707,
  "model_used": "stacking",
  "flood_probability": 0.8421,
  "susceptibility_class": "Very High",
  "classification_scheme": "equal_interval",
  "top_drivers": [
    {"feature": "Elevation (m)", "contribution": "+0.284", "direction": "Increases Risk"},
    {"feature": "Rainfall (mm)", "contribution": "+0.221", "direction": "Increases Risk"},
    {"feature": "Soil Type_Clay", "contribution": "+0.145", "direction": "Increases Risk"},
    {"feature": "Historical Floods", "contribution": "+0.112", "direction": "Increases Risk"}
  ],
  "address": "Chennai, Tamil Nadu, India"
}
```

#### 2. Tamil Nadu District Prediction
```bash
curl -X POST "http://localhost:8000/api/v1/predict/district" \
     -H "Content-Type: application/json" \
     -d '{
       "district_name": "Erode",
       "model": "soft_voting"
     }'
```

---

## 📁 8. Project Architecture & File Hierarchy

```
FloodSense-India/
│
├── config.yaml                       # Central system configuration & hyperparameters
├── requirements.txt                  # Python dependency specifications
├── environment.yml                   # Conda environment definition
├── README.md                         # Project documentation
├── Final_Dataset.csv                 # Preprocessed training dataset (2.2 MB)
├── final_df.csv                      # Feature-engineered training dataframe (2.5 MB)
├── flood_risk_dataset_india.csv      # National flood conditioning records (1.28 MB)
├── Model.ipynb                       # Model training and hyperparameter search notebook
├── eda.ipynb                         # Exploratory Data Analysis & spatial correlation notebook
├── PAPER-1.pdf                       # Paper A: Rahimi et al. (2026)
├── PAPER-2.pdf                       # Paper B: Demissie et al. (2024)
│
├── api/                              # FastAPI REST API Backend
│   ├── main.py                       # FastAPI application entrypoint & middleware
│   └── routes/
│       ├── __init__.py
│       ├── prediction.py             # Single coordinate point prediction routes
│       ├── risk.py                   # Risk & priority scoring routes
│       ├── feedback.py               # Field verification survey routes
│       ├── geocode.py                # Reverse geocoding routes
│       └── tn_predict.py             # Tamil Nadu 38-district prediction routes
│
├── app/                              # Streamlit Interactive Dashboard
│   ├── streamlit_app.py              # Main dashboard entrypoint & router
│   ├── assets/
│   │   └── styles.css                # Custom CSS styling (dark/light GIS palette)
│   ├── components/                   # Reusable UI widgets
│   │   ├── charts.py                 # Plotly ROC curves, radar & waterfall charts
│   │   ├── kpis.py                   # Metric cards & summary KPI widgets
│   │   ├── legends.py                # Susceptibility & risk color ramp legends
│   │   ├── map.py                    # Folium interactive map builder
│   │   └── sidebar.py                # Sidebar controls & model selectors
│   └── views/                        # 12 Specialized Page Views
│       ├── tn_district_prediction.py # Tamil Nadu 38-District Prediction view
│       ├── overview.py               # Executive overview & KPIs
│       ├── susceptibility.py         # Susceptibility map & raster viewer
│       ├── risk_priority.py          # Administrative Risk & Priority Zones
│       ├── point_prediction.py       # Check My Location coordinate query & XAI
│       ├── rainfall_scenario.py      # What-If rainfall sensitivity simulator
│       ├── model_validation.py       # ROC-AUC, PR curves & validation metrics
│       ├── uncertainty.py            # Model uncertainty & consensus maps
│       ├── historical_events.py      # Historical flood disaster replay
│       ├── field_feedback.py         # Surveyor ground truth collection form
│       ├── data_method.py            # Methodology, data dictionary & papers
│       └── system_health.py          # Pipeline diagnostics & asset checker
│
├── src/                              # Core Geospatial & ML Source Code
│   ├── __init__.py
│   ├── classify.py                   # Equal Interval & Jenks Natural Breaks classifiers
│   ├── data_loader.py                # GeoTIFF raster sampling & CRS alignment
│   ├── data_quality.py               # Raster, vector, and model integrity audits
│   ├── explain.py                    # SHAP local/global attributions & MDI importance
│   ├── feedback.py                   # SQLite feedback database manager
│   ├── generate_placeholders.py      # Placeholder raster & dummy model generator
│   ├── geocode.py                    # Nominatim reverse geocoder
│   ├── historical.py                 # Historical disaster replay benchmarks
│   ├── model_registry.py             # Model metadata loader & schema inspector
│   ├── predict.py                    # Core inference engine (single, voting, stacking)
│   ├── preprocessing.py              # Feature validation & standard scaler
│   ├── reasons.py                    # Plain-English risk driver bullet generator
│   ├── reporting.py                  # Automated PDF & JSON report generator
│   ├── risk.py                       # Multiplicative & weighted overlay risk engine
│   ├── scenarios.py                  # Precipitation perturbation simulator
│   ├── spatial.py                    # Spatial Block CV & Case I/II sampling
│   ├── tn_districts.py               # Tamil Nadu 38-district database & features
│   ├── uncertainty.py                # Ensemble variance & consensus calculator
│   ├── utils.py                      # YAML config loader, loggers, hashing
│   └── validate.py                   # ROC-AUC, F1, precision-recall & PR curves
│
├── models/                           # Serialized Model Artifacts & Registry
│   ├── registry.json                 # JSON schema of all models, features & dates
│   ├── xgboost.joblib                # Trained XGBoost classifier (380 KB)
│   ├── random_forest.joblib          # Trained Random Forest classifier (11.2 MB)
│   ├── lightgbm.joblib               # Trained LightGBM classifier (349 KB)
│   ├── decision_tree.joblib          # Trained Decision Tree classifier (83 KB)
│   ├── logistic_regression.joblib    # Trained Logistic Regression baseline (1.7 KB)
│   ├── meta_learner.joblib           # Trained Stacking Meta-Learner (895 B)
│   └── dummy.joblib                  # Dummy classifier for testing & fallbacks
│
├── data/                             # Geospatial Vectors, Rasters & Boundaries
│   ├── boundaries/                   # GeoPackage administrative vector layers
│   │   ├── states.gpkg               # State boundaries
│   │   ├── districts.gpkg            # District boundaries
│   │   ├── blocks.gpkg               # Block/Tehsil boundaries
│   │   └── villages.gpkg             # Village/Panchayat boundaries
│   ├── exposure/                     # Exposure GeoTIFFs & point layers
│   │   ├── population.tif            # Population density raster
│   │   ├── buildings.tif             # Building footprint density raster
│   │   ├── roads.tif                 # Transportation network density raster
│   │   ├── hospitals.gpkg            # Critical healthcare facility points
│   │   └── schools.gpkg              # Educational institution points
│   ├── field_records/
│   │   └── feedback.db               # SQLite database of field validation reports
│   ├── rasters/                      # 22 Conditioning GeoTIFFs (EPSG:4326)
│   └── vectors/
│       └── flood_inventory.gpkg      # Historical flood polygon inventory
│
├── outputs/                          # Generated Outputs, Rasters & Reports
│   ├── reports/
│   │   ├── FloodSense_Report.pdf     # Generated PDF summary report
│   │   └── RUN-2026-09-29-123445_metadata.json # Inference pipeline execution metadata
│   └── susceptibility/               # Generated inference probability rasters
│       ├── decision_tree_probability.tif
│       ├── lightgbm_probability.tif
│       ├── logistic_regression_probability.tif
│       ├── random_forest_probability.tif
│       ├── soft_voting_probability.tif
│       ├── stacking_probability.tif
│       └── xgboost_probability.tif
│
└── tests/                            # Pytest Automated Test Suite (20 Tests)
    ├── test_classify.py              # Equal Interval and Jenks breaks tests
    ├── test_data_loader.py           # Raster sampling & coordinate bounds tests
    ├── test_geocode.py               # Reverse geocoding tests
    ├── test_predict.py               # Soft voting, stacking & prediction tests
    ├── test_reasons.py               # Plain-English risk driver generator tests
    ├── test_risk.py                  # Multiplicative & weighted overlay risk tests
    ├── test_scenarios.py             # Rainfall sensitivity perturbation tests
    ├── test_spatial_cv.py            # Spatial block CV & data leakage tests
    └── test_validate.py              # ROC-AUC, PR curve & metric calculation tests
```

---

## 🚀 9. Installation & Getting Started

### Prerequisites
- **Python 3.11+** installed on your machine.
- **Git** installed and configured.

### Step 1: Clone Repository
```bash
git clone https://github.com/SABARISH248/flood.git
cd flood
```

### Step 2: Environment Setup

#### Option A: Python venv (Recommended)
```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.venv\Scripts\activate.bat
# On Linux/macOS:
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### Option B: Conda Environment
```bash
conda env create -f environment.yml
conda activate floodsense
```

---

## 💻 10. Execution Commands

### 1. Run Automated Unit Test Suite
Verify that all 20 unit tests pass across all testing modules:
```bash
python -m pytest -v
```

### 2. Launch Streamlit Interactive Dashboard
Run the 12-page interactive web application:
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

### 3. Launch FastAPI Backend Server
Run the production REST API with hot reloading:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Access the interactive Swagger UI at `http://localhost:8000/docs`.

### 4. Run Batch Raster Inference Pipeline
Execute full-grid raster predictions across the study area and output probability GeoTIFFs:
```bash
python -m src.predict
```
Output GeoTIFF rasters will be written to `outputs/susceptibility/`.

---

## 🧪 11. Testing & Validation

The test suite contains **20 passing unit tests** covering all critical pipeline components:

```bash
$ python -m pytest -q
....................                                                     [100%]
20 passed in 22.57s
```

### Test Coverage Highlights:
- **`test_classify.py`**: Tests monotonicity of Equal Interval classification and bounds of Jenks Natural Breaks optimization.
- **`test_data_loader.py`**: Validates coordinate transformations, bounds clipping, and regional fallback feature interpolation.
- **`test_geocode.py`**: Tests address resolution with timeout resilience and graceful network degradation.
- **`test_predict.py`**: Validates prediction shape, range $[0.0, 1.0]$, Soft Voting probability invariance, and Stacking Meta-Learner output.
- **`test_reasons.py`**: Tests plain-English risk driver bullet generation for high and low flood vulnerability scenarios.
- **`test_risk.py`**: Tests both Multiplicative ($S \times E \times V$) and Weighted Overlay ($W_s S + W_e E + W_v V$) risk engines.
- **`test_scenarios.py`**: Tests linear and non-linear response curves under perturbed precipitation scenarios (-30% to +100%).
- **`test_spatial_cv.py`**: Tests spatial block partitioning to ensure zero data leakage across spatial fold boundaries.
- **`test_validate.py`**: Tests ROC-AUC calculation, confusion matrices, and Paper A Prediction Rate curves.

---

## ⚠️ 12. Scientific Disclaimers & Limitations

1. **Susceptibility vs Dynamic Forecasting:** Flood susceptibility represents the *static geographic likelihood* of flood accumulation based on topography, soil, land cover, and climate norms. It is **not** a real-time hydrodynamic forecast or dam release alert.
2. **Statistical Attributions:** Feature contributions generated by SHAP quantify mathematical influence within trained models; they reflect statistical correlation rather than direct physical causation.
3. **Emergency Decision Making:** Model outputs should assist, rather than replace, certified assessments by the National Disaster Management Authority (NDMA), Central Water Commission (CWC), and state disaster response forces.

---

## 📜 13. Citations & References

- **Paper A:** Rahimi, J., et al. (2026). *Flood susceptibility modeling using multi-source conditioning factors and soft-voting ensemble learning*. Journal of Hydrology / Environmental Modelling & Software.
- **Paper B:** Demissie, T., et al. (2024). *Stacking meta-learner ensemble architecture and Jenks natural breaks classification for regional flood susceptibility zoning*. Geoscience Frontiers.

---

## 📄 14. License & Author

- **Author / Repository Owner:** Sabarish ([@SABARISH248](https://github.com/SABARISH248))
- **Repository:** [https://github.com/SABARISH248/flood](https://github.com/SABARISH248/flood)
- **License:** Distributed under the **MIT License**.
