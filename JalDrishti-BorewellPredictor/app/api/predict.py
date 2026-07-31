"""
predict.py — Real ML-powered prediction endpoints for JalDrishti v2.
Uses feature_engine's built-in ML inference, SHAP factors, bilingual advisory.
"""
from fastapi import APIRouter, HTTPException
from typing import List
import numpy as np
import pandas as pd
from pathlib import Path

router = APIRouter()

# ------------------------------------------------------------------ #
# Model Loading (at import time — for SHAP explainer)
# ------------------------------------------------------------------ #
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

_shap_explainer = None

try:
    import shap
    import joblib
    _risk_model_path = MODELS_DIR / "risk_model.joblib"
    if _risk_model_path.exists():
        _risk_model_for_shap = joblib.load(_risk_model_path)
        _shap_explainer = shap.TreeExplainer(_risk_model_for_shap)
        _model_config_for_shap = joblib.load(MODELS_DIR / "feature_config.joblib")
        print("[OK] SHAP explainer ready in predict.py")
except Exception as e:
    print(f"[WARN] SHAP setup in predict.py: {e}")
    _model_config_for_shap = None

# ------------------------------------------------------------------ #
# Imports
# ------------------------------------------------------------------ #
from app.services.feature_engine import feature_engine_instance
from app.services.geo_lookup import geo_lookup_service
from app.services.grid_analyzer import grid_analyzer_instance
from app.services.risk_matrix import risk_matrix_instance
from app.schemas.request import PredictRequest, GridAnalysisRequest
from app.schemas.response import (
    PredictResponse, GeocodeResponse, DepthRange, Factor, LocationInfo,
    NearbyStats, SoilProperties, GeologyInfo, HydrologySummary, TerrainInfo,
    RainfallHistoryResponse, HeatmapDataPoint,
    GridAnalysisResponse, GridPointScore,
    MonthlyRiskResponse, MonthlyRisk,
    DetailedExplanation, DataSourceInfo,
)


# ------------------------------------------------------------------ #
# SHAP-based factor breakdown
# ------------------------------------------------------------------ #
_FEATURE_LABELS = {
    'extraction_stage_pct': 'Extraction Stage',
    'gwl_pre_monsoon_mbgl': 'Pre-Monsoon GWL',
    'gwl_post_monsoon_mbgl': 'Post-Monsoon GWL',
    'gwl_seasonal_fluctuation_m': 'GWL Seasonal Fluctuation',
    'lineament_density_per_km2': 'Lineament Density',
    'weathered_zone_thickness_m': 'Weathered Zone Thickness',
    'depth_to_bedrock_m': 'Depth to Bedrock',
    'rainfall_15yr_annual_avg_mm': 'Annual Rainfall (15y avg)',
    'rainfall_monsoon_avg_mm': 'Monsoon Rainfall',
    'et0_annual_avg_mm': 'Evapotranspiration (ET₀)',
    'effective_recharge_mm': 'Effective Recharge',
    'heavy_rain_days_per_yr': 'Heavy Rain Days/Year',
    'moderate_rain_days_per_yr': 'Moderate Rain Days/Year',
    'rainfall_intensity_ratio': 'Rainfall Intensity',
    'drought_years_in_15': 'Drought Years (in 15)',
    'elevation_m': 'Elevation',
    'slope_degrees': 'Slope',
    'twi': 'Topographic Wetness Index',
    'distance_to_waterbody_m': 'Distance to Water Body',
    'impervious_surface_pct': 'Impervious Surface %',
    'clay_pct': 'Clay Content',
    'sand_pct': 'Sand Content',
    'soil_ph': 'Soil pH',
    'planned_depth_ft': 'Planned Depth',
    'planned_month': 'Planned Month',
    'rock_type_encoded': 'Rock Type',
    'aquifer_type_encoded': 'Aquifer Type',
    'land_cover_class_encoded': 'Land Cover',
}


def _build_shap_vector(features_dict: dict) -> np.ndarray:
    """Build a feature vector matching the model's training column order."""
    if _model_config_for_shap is None:
        return None
    feature_cols = _model_config_for_shap.get('features', [])
    encoders = _model_config_for_shap.get('encoders', {})
    row = {}
    for col in feature_cols:
        if col.endswith('_encoded'):
            base = col.replace('_encoded', '')
            if base in encoders and base in features_dict:
                le = encoders[base]
                val = features_dict[base]
                row[col] = le.transform([val])[0] if val in le.classes_ else 0
            else:
                row[col] = 0
        else:
            row[col] = features_dict.get(col, 0)
    return np.array([[row.get(c, 0) for c in feature_cols]], dtype=np.float64), feature_cols


def _compute_shap_factors(features_dict: dict) -> List[dict]:
    """Return top-6 SHAP-based risk factors with human-readable labels."""
    if _shap_explainer is None:
        return []
    try:
        result = _build_shap_vector(features_dict)
        if result is None:
            return []
        X, cols = result
        sv = _shap_explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        vals = sv[0]

        pairs = sorted(zip(cols, vals), key=lambda x: abs(x[1]), reverse=True)
        max_abs = max(abs(v) for _, v in pairs) if pairs else 1.0

        factors = []
        for name, shap_val in pairs[:6]:
            norm = abs(shap_val) / max_abs if max_abs else 0
            label = _FEATURE_LABELS.get(name, name.replace('_', ' ').title())

            raw_key = name.replace('_encoded', '')
            raw_val = features_dict.get(raw_key, '')
            disp_val = f"{raw_val:.1f}" if isinstance(raw_val, float) else str(raw_val)

            impact = 'High' if norm > 0.6 else ('Medium' if norm > 0.3 else 'Low')
            contrib = 'Positive' if shap_val > 0 else 'Negative'

            factors.append({
                'name': label, 'impact': impact, 'value': disp_val,
                'contribution': contrib, 'score': round(norm, 3),
            })
        return factors
    except Exception as e:
        print(f"[WARN] SHAP failed: {e}")
        return []


# ------------------------------------------------------------------ #
# Bilingual advisory generator
# ------------------------------------------------------------------ #
def _generate_advisory(risk_score, features_dict):
    extraction = features_dict.get('extraction_stage_pct', 80)
    lineament = features_dict.get('lineament_density_per_km2', 2.0)
    recharge = features_dict.get('effective_recharge_mm', 100)

    en, hi = [], []

    if risk_score > 75:
        en.append(f"⚠️ CRITICAL: This location scores {risk_score}/100 for borewell failure risk.")
        hi.append(f"⚠️ गंभीर: इस स्थान का बोरवेल विफलता जोखिम स्कोर {risk_score}/100 है।")
    elif risk_score > 50:
        en.append(f"This location has moderate-to-high failure risk (score {risk_score}/100).")
        hi.append(f"इस स्थान पर मध्यम-से-उच्च विफलता जोखिम है (स्कोर {risk_score}/100)।")
    elif risk_score > 30:
        en.append(f"Moderate borewell prospects at this location (score {risk_score}/100).")
        hi.append(f"इस स्थान पर बोरवेल की मध्यम संभावना है (स्कोर {risk_score}/100)।")
    else:
        en.append(f"✅ Favorable conditions for borewell drilling (score {risk_score}/100).")
        hi.append(f"✅ बोरवेल खुदाई के लिए अनुकूल स्थितियाँ (स्कोर {risk_score}/100)।")

    if extraction > 100:
        en.append(f"Groundwater extraction at {extraction:.0f}% (over-exploited) — water table declining rapidly.")
        hi.append(f"भूजल निकासी {extraction:.0f}% (अत्यधिक दोहन) — जल स्तर तेजी से गिर रहा है।")
    elif extraction > 90:
        en.append(f"Extraction stage at {extraction:.0f}% — approaching critical levels.")
        hi.append(f"निकासी चरण {extraction:.0f}% — गंभीर स्तर के करीब पहुँच रहा है।")

    if lineament > 2.5:
        en.append(f"Good lineament density ({lineament:.1f}/km²) suggests fracture-fed aquifers may sustain yield.")
        hi.append(f"अच्छी रेखीय घनत्व ({lineament:.1f}/km²) — फ्रैक्चर एक्विफर उपज बनाए रख सकते हैं।")
    elif lineament < 1.5:
        en.append(f"Low lineament density ({lineament:.1f}/km²) reduces chances of hitting a productive fracture.")
        hi.append(f"कम रेखीय घनत्व ({lineament:.1f}/km²) — उत्पादक फ्रैक्चर मिलने की संभावना कम है।")

    if recharge < 60:
        en.append("Effective recharge is low due to impervious surfaces. Consider rainwater harvesting.")
        hi.append("अभेद्य सतहों के कारण प्रभावी पुनर्भरण कम है। वर्षा जल संचयन पर विचार करें।")

    return ' '.join(en), ' '.join(hi)


def _risk_category(score: int) -> str:
    if score > 75: return "Critical Risk"
    if score > 50: return "High Risk"
    if score > 30: return "Moderate Risk"
    return "Low Risk"


# ================================================================== #
#                           ENDPOINTS                                #
# ================================================================== #

@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    lat, lng = req.latitude, req.longitude

    # Geocode if only address supplied
    if lat is None or lng is None:
        if req.address:
            results = feature_engine_instance.geocode_address(req.address)
            if not results:
                raise HTTPException(400, "Could not geocode address")
            lat, lng = float(results[0]['lat']), float(results[0]['lon'])
        else:
            raise HTTPException(400, "Provide lat/lng or address")

    planned_depth = req.planned_depth_ft or 500
    planned_month = req.planned_month or 1

    # 1. Build features (returns tuple: features_dict, mandal, zone, rain_extras)
    features_dict, mandal, zone, rain_extras = feature_engine_instance.build_features(
        lat, lng, planned_depth, planned_month
    )

    # 2. ML inference (uses feature_engine's built-in predict)
    risk_score, confidence, rec_depth = feature_engine_instance.predict(features_dict)

    risk_cat = _risk_category(risk_score)
    rec_min = max(100, rec_depth - 75)
    rec_max = rec_depth + 75

    # 3. SHAP factors
    factors_list = _compute_shap_factors(features_dict)

    # Fallback factors if SHAP unavailable
    if not factors_list:
        factors_list = [
            {'name': 'Extraction Stage', 'impact': 'High',
             'value': f"{features_dict.get('extraction_stage_pct', 0):.0f}%",
             'contribution': 'Negative', 'score': 0.8},
            {'name': 'Lineament Density', 'impact': 'Medium',
             'value': f"{features_dict.get('lineament_density_per_km2', 0):.1f}/km²",
             'contribution': 'Positive', 'score': 0.5},
            {'name': 'Effective Recharge', 'impact': 'Medium',
             'value': f"{features_dict.get('effective_recharge_mm', 0):.0f} mm",
             'contribution': 'Positive', 'score': 0.4},
            {'name': 'Pre-Monsoon GWL', 'impact': 'Medium',
             'value': f"{features_dict.get('gwl_pre_monsoon_mbgl', 0):.1f} m",
             'contribution': 'Negative', 'score': 0.6},
        ]

    advisory_en, advisory_hi = _generate_advisory(risk_score, features_dict)
    factors = [Factor(**f) for f in factors_list]

    # Soil values from features_dict
    clay = features_dict.get('clay_pct', 35.0)
    sand = features_dict.get('sand_pct', 40.0)
    silt = max(0, round(100 - clay - sand, 1))
    ph = features_dict.get('soil_ph', 7.2)

    # Lineament zone info
    dominant_trend = zone.get('dominant_trend', 'NE-SW') if zone else 'NE-SW'
    frac_range = zone.get('fracture_depth_range_m', [30, 80]) if zone else [30, 80]
    frac_min = frac_range[0] if isinstance(frac_range, list) and len(frac_range) >= 2 else 30
    frac_max = frac_range[1] if isinstance(frac_range, list) and len(frac_range) >= 2 else 80
    
    detailed_explanations = [DetailedExplanation(**d) for d in feature_engine_instance._generate_detailed_explanations(features_dict, mandal, zone)]
    
    risk_data = risk_matrix_instance.assess_all_months(lat, lng, features_dict, mandal)
    monthly_risk = next((MonthlyRisk(**m) for m in risk_data['months'] if m['month'] == planned_month), None)
    
    data_sources = [
        DataSourceInfo(field="Rainfall & Evapotranspiration", source="Open-Meteo API", status="live"),
        DataSourceInfo(field="Elevation & Slope", source="Open-Elevation API", status="live"),
        DataSourceInfo(field="Soil Properties", source="SoilGrids API", status="live"),
        DataSourceInfo(field="Groundwater Levels & Geology", source="CGWB / Local DB", status="static"),
        DataSourceInfo(field="Land Cover & Imperviousness", source="Heuristics", status="static"),
    ]

    return PredictResponse(
        risk_score=risk_score,
        risk_category=risk_cat,
        confidence=confidence,
        recommended_depth_ft=DepthRange(min=rec_min, max=rec_max),
        factors=factors,
        advisory=advisory_en,
        advisory_hi=advisory_hi,
        detailed_explanations=detailed_explanations,
        data_sources=data_sources,
        monthly_risk=monthly_risk,
        location_info=LocationInfo(
            mandal=mandal.get('mandal', 'Unknown'),
            district=mandal.get('district', 'Hyderabad'),
            state="Telangana",
            rock_type=features_dict.get('rock_type', 'Granite-Gneiss'),
            aquifer_type=features_dict.get('aquifer_type', 'Weathered'),
        ),
        nearby_stats=NearbyStats(
            avg_depth_ft=mandal.get('avg_borewell_depth_ft', 500),
            estimated_success_rate=round(mandal.get('estimated_success_rate', 0.5), 2),
            extraction_stage_pct=mandal.get('extraction_stage_pct', 80.0),
        ),
        soil_properties=SoilProperties(
            clay_pct=clay, sand_pct=sand, silt_pct=silt,
            ph=ph, organic_carbon=1.2, bulk_density=1.35,
        ),
        geology_info=GeologyInfo(
            lineament_density=features_dict.get('lineament_density_per_km2', 2.0),
            weathered_zone_thickness_m=features_dict.get('weathered_zone_thickness_m', 18.0),
            depth_to_bedrock_m=features_dict.get('depth_to_bedrock_m', 22.0),
            dominant_trend=dominant_trend,
            fracture_depth_range=DepthRange(min=frac_min, max=frac_max),
        ),
        hydrology_summary=HydrologySummary(
            rainfall_15yr_avg_mm=features_dict.get('rainfall_15yr_annual_avg_mm', 820.0),
            monsoon_avg_mm=features_dict.get('rainfall_monsoon_avg_mm', 650.0),
            et0_avg_mm=features_dict.get('et0_annual_avg_mm', 1400.0),
            effective_recharge_mm=features_dict.get('effective_recharge_mm', 100.0),
            heavy_rain_days=int(features_dict.get('heavy_rain_days_per_yr', 6)),
            drought_years=int(features_dict.get('drought_years_in_15', 2)),
        ),
        terrain_info=TerrainInfo(
            elevation_m=features_dict.get('elevation_m', 500.0),
            slope_degrees=features_dict.get('slope_degrees', 2.5),
            twi=features_dict.get('twi', 10.0),
            distance_to_waterbody_m=features_dict.get('distance_to_waterbody_m', 1200.0),
            land_cover_class=features_dict.get('land_cover_class', 'Built Area'),
            impervious_pct=features_dict.get('impervious_surface_pct', 50.0),
        ),
    )


@router.get("/geocode", response_model=List[GeocodeResponse])
def geocode(q: str):
    results = feature_engine_instance.geocode_address(q)
    out = []
    for r in results:
        addr = r.get('address', {})
        out.append(GeocodeResponse(
            lat=float(r['lat']),
            lng=float(r['lon']),
            display_name=r.get('display_name', ''),
            mandal=addr.get('suburb') or addr.get('county'),
            district=addr.get('state_district'),
            state=addr.get('state'),
        ))
    return out


@router.get("/rainfall-history", response_model=RainfallHistoryResponse)
def rainfall_history(lat: float, lng: float):
    """Generate 15-year rainfall history from feature engine data."""
    planned_depth = 500
    planned_month = 1
    _, _, _, rain_extras = feature_engine_instance.build_features(lat, lng, planned_depth, planned_month)

    yearly_rain = rain_extras.get('yearly_rain', [])
    yearly_et0 = rain_extras.get('yearly_et0', [])

    if not yearly_rain:
        return RainfallHistoryResponse(
            years=list(range(2010, 2025)),
            annual_rainfall=[800.0] * 15,
            annual_et0=[1200.0] * 15,
            effective_recharge=[80.0] * 15,
            monthly_data=[],
        )

    years = list(range(2010, 2010 + len(yearly_rain)))
    recharge = [round(max(0, (r - e) * 0.25), 1) for r, e in zip(yearly_rain, yearly_et0)]

    return RainfallHistoryResponse(
        years=years,
        annual_rainfall=yearly_rain,
        annual_et0=yearly_et0,
        effective_recharge=recharge,
        monthly_data=[],
    )


@router.get("/heatmap", response_model=List[HeatmapDataPoint])
def get_heatmap():
    return geo_lookup_service.get_heatmap_grid()


@router.post('/grid-analysis', response_model=GridAnalysisResponse)
def grid_analysis(req: GridAnalysisRequest):
    results = grid_analyzer_instance.analyze_grid(req.latitude, req.longitude, req.radius_km)
    return GridAnalysisResponse(**results)


@router.get('/monthly-risk', response_model=MonthlyRiskResponse)
def monthly_risk(lat: float, lng: float):
    features_dict, mandal, zone, _ = feature_engine_instance.build_features(lat, lng, 500, 1)
    result = risk_matrix_instance.assess_all_months(lat, lng, features_dict, mandal)
    return MonthlyRiskResponse(**result)


@router.get("/health")
def health_check():
    has_model = feature_engine_instance.risk_model is not None
    version = 'not loaded'
    if feature_engine_instance.model_config:
        version = feature_engine_instance.model_config.get('model_version', 'unknown')
    return {
        "status": "ok",
        "models_loaded": has_model,
        "model_version": version,
    }
