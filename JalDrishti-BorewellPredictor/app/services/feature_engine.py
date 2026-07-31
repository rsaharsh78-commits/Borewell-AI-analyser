"""
Feature engine for JalDrishti v2 - computes features for ML model prediction.
Uses local JSON data (mandals, lineaments, waterbodies) plus optional live API calls.
"""
import json
import math
import os
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
import traceback

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

MANDALS_PATH = DATA_DIR / "hyderabad_mandals.json"
LINEAMENTS_PATH = DATA_DIR / "hyderabad_lineaments.json"
WATERBODIES_PATH = DATA_DIR / "hyderabad_waterbodies.json"

# Hyderabad-region defaults (used when APIs fail)
HYD_DEFAULTS = {
    "rainfall_15yr_annual_avg_mm": 820.0,
    "rainfall_monsoon_avg_mm": 650.0,
    "et0_annual_avg_mm": 1500.0,
    "effective_recharge_mm": 80.0,
    "heavy_rain_days_per_yr": 8.0,
    "moderate_rain_days_per_yr": 30.0,
    "rainfall_intensity_ratio": 0.21,
    "drought_years_in_15": 3,
    "elevation_m": 540.0,
    "slope_degrees": 2.5,
    "twi": 10.0,
    "clay_pct": 35.0,
    "sand_pct": 40.0,
    "soil_ph": 7.2,
}

IMPERVIOUS_PCT_MAP = {
    'Water': 0, 'Trees': 5, 'Flooded Vegetation': 10,
    'Crops': 15, 'Built Area': 90, 'Bare Ground': 5,
    'Snow/Ice': 0, 'Clouds': 0, 'Rangeland': 10
}


def _haversine_m(lat1, lng1, lat2, lng2):
    """Haversine distance in meters."""
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _api_get(url, timeout=5):
    """GET request with error handling, returns parsed JSON or None."""
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'JalDrishti/2.0 (borewell-predictor)',
            'Accept': 'application/json'
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print("[WARN] API call failed: " + str(e)[:80])
        return None


def _api_post(url, data, timeout=5):
    """POST request with error handling."""
    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={
            'User-Agent': 'JalDrishti/2.0 (borewell-predictor)',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print("[WARN] POST API call failed: " + str(e)[:80])
        return None


class FeatureEngine:
    def __init__(self):
        self.mandals_data = []
        self.lineaments_zones = []
        self.waterbodies_data = []
        self.risk_model = None
        self.depth_model = None
        self.model_config = None

        # Load mandals
        try:
            if MANDALS_PATH.exists():
                with open(MANDALS_PATH, 'r', encoding='utf-8') as f:
                    self.mandals_data = json.load(f)
                print("[OK] Loaded mandals: " + str(len(self.mandals_data)))
        except Exception as e:
            print("[WARN] Could not load mandals: " + str(e))

        # Load waterbodies
        try:
            if WATERBODIES_PATH.exists():
                with open(WATERBODIES_PATH, 'r', encoding='utf-8') as f:
                    self.waterbodies_data = json.load(f)
                print("[OK] Loaded waterbodies: " + str(len(self.waterbodies_data)))
        except Exception as e:
            print("[WARN] Could not load waterbodies: " + str(e))

        # Load lineaments
        try:
            if LINEAMENTS_PATH.exists():
                with open(LINEAMENTS_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.lineaments_zones = data.get('zones', [])
                print("[OK] Loaded lineaments: " + str(len(self.lineaments_zones)) + " zones")
        except Exception as e:
            print("[WARN] Could not load lineaments: " + str(e))

        # Load ML models
        try:
            import joblib
            risk_path = MODELS_DIR / "risk_model.joblib"
            depth_path = MODELS_DIR / "depth_model.joblib"
            config_path = MODELS_DIR / "feature_config.joblib"
            if risk_path.exists() and config_path.exists():
                self.risk_model = joblib.load(str(risk_path))
                self.model_config = joblib.load(str(config_path))
                if depth_path.exists():
                    self.depth_model = joblib.load(str(depth_path))
                print("[OK] ML models loaded successfully")
                print("[OK] Model version: " + str(self.model_config.get('model_version', 'unknown')))
            else:
                print("[WARN] ML model files not found at: " + str(MODELS_DIR))
        except Exception as e:
            print("[WARN] Could not load ML models: " + str(e))
            traceback.print_exc()

    # ---- Nearest-lookup methods ----

    def find_nearest_mandal(self, lat, lng):
        """Find nearest mandal by haversine distance."""
        best = {}
        best_dist = float('inf')
        for m in self.mandals_data:
            d = _haversine_m(lat, lng, m.get('lat', 0), m.get('lng', 0))
            if d < best_dist:
                best_dist = d
                best = m
        return best

    def find_lineament_zone(self, lat, lng):
        """Find which lineament zone contains this point."""
        for z in self.lineaments_zones:
            bb = z.get('bbox', {})
            if (bb.get('min_lat', 0) <= lat <= bb.get('max_lat', 0) and
                    bb.get('min_lng', 0) <= lng <= bb.get('max_lng', 0)):
                return z
        # Fallback to nearest zone
        best = {}
        best_dist = float('inf')
        for z in self.lineaments_zones:
            bb = z.get('bbox', {})
            clat = (bb.get('min_lat', 0) + bb.get('max_lat', 0)) / 2
            clng = (bb.get('min_lng', 0) + bb.get('max_lng', 0)) / 2
            d = _haversine_m(lat, lng, clat, clng)
            if d < best_dist:
                best_dist = d
                best = z
        return best

    def distance_to_nearest_water(self, lat, lng):
        """Haversine distance in meters to nearest water body."""
        best = float('inf')
        for wb in self.waterbodies_data:
            d = _haversine_m(lat, lng, wb.get('lat', 0), wb.get('lng', 0))
            if d < best:
                best = d
        return best if best < float('inf') else 2000.0

    # ---- Geocoding ----

    @lru_cache(maxsize=128)
    def geocode_address(self, query):
        """Geocode an address using Nominatim."""
        encoded_query = urllib.parse.quote(query)
        url = ("https://nominatim.openstreetmap.org/search?"
               "q=" + encoded_query + "&format=jsonv2&addressdetails=1&limit=5&countrycodes=in")
        return _api_get(url) or []

    # ---- Main feature builder ----

    @lru_cache(maxsize=128)
    def _fetch_open_meteo(self, lat, lng):
        rlat, rlng = round(lat, 2), round(lng, 2)
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={rlat}&longitude={rlng}&start_date=2020-01-01&end_date=2024-12-31&daily=precipitation_sum,et0_fao_evapotranspiration&timezone=Asia%2FKolkata"
        data = _api_get(url)
        if not data or 'daily' not in data:
            return {
                "rainfall_15yr_annual_avg_mm": HYD_DEFAULTS["rainfall_15yr_annual_avg_mm"],
                "rainfall_monsoon_avg_mm": HYD_DEFAULTS["rainfall_monsoon_avg_mm"],
                "et0_annual_avg_mm": HYD_DEFAULTS["et0_annual_avg_mm"],
                "effective_recharge_mm": HYD_DEFAULTS["effective_recharge_mm"],
                "heavy_rain_days_per_yr": HYD_DEFAULTS["heavy_rain_days_per_yr"],
                "moderate_rain_days_per_yr": HYD_DEFAULTS["moderate_rain_days_per_yr"],
                "rainfall_intensity_ratio": HYD_DEFAULTS["rainfall_intensity_ratio"],
                "drought_years_in_15": HYD_DEFAULTS["drought_years_in_15"],
                "yearly_rain": [HYD_DEFAULTS["rainfall_15yr_annual_avg_mm"]] * 15,
                "yearly_et0": [HYD_DEFAULTS["et0_annual_avg_mm"]] * 15,
            }
        
        daily_time = data['daily'].get('time', [])
        daily_precip = data['daily'].get('precipitation_sum', [])
        daily_et0 = data['daily'].get('et0_fao_evapotranspiration', [])
        
        yearly_rain = {}
        yearly_et0 = {}
        yearly_heavy = {}
        yearly_moderate = {}
        yearly_monsoon = {}
        
        for i, t in enumerate(daily_time):
            y = int(t[0:4])
            m = int(t[5:7])
            p = daily_precip[i] or 0.0
            e = daily_et0[i] or 0.0
            
            yearly_rain[y] = yearly_rain.get(y, 0) + p
            yearly_et0[y] = yearly_et0.get(y, 0) + e
            if 6 <= m <= 9:
                yearly_monsoon[y] = yearly_monsoon.get(y, 0) + p
            if p > 20:
                yearly_heavy[y] = yearly_heavy.get(y, 0) + 1
            elif p > 5:
                yearly_moderate[y] = yearly_moderate.get(y, 0) + 1
                
        years = list(yearly_rain.keys())
        if not years:
            return HYD_DEFAULTS
            
        avg_rain = sum(yearly_rain.values()) / len(years)
        avg_monsoon = sum(yearly_monsoon.values()) / len(years)
        avg_et0 = sum(yearly_et0.values()) / len(years)
        avg_heavy = sum(yearly_heavy.values()) / len(years)
        avg_moderate = sum(yearly_moderate.values()) / len(years)
        drought_years = sum(1 for y in years if yearly_rain[y] < 600)
        
        return {
            "rainfall_15yr_annual_avg_mm": round(avg_rain, 1),
            "rainfall_monsoon_avg_mm": round(avg_monsoon, 1),
            "et0_annual_avg_mm": round(avg_et0, 1),
            "effective_recharge_mm": 0, # Will be computed in build_features
            "heavy_rain_days_per_yr": round(avg_heavy, 1),
            "moderate_rain_days_per_yr": round(avg_moderate, 1),
            "rainfall_intensity_ratio": round(avg_heavy / max(1, avg_moderate + avg_heavy), 3),
            "drought_years_in_15": int(drought_years),
            "yearly_rain": [round(yearly_rain[y], 1) for y in years],
            "yearly_et0": [round(yearly_et0[y], 1) for y in years]
        }

    @lru_cache(maxsize=128)
    def _fetch_elevation(self, lat, lng):
        pts = [
            {"latitude": lat, "longitude": lng},
            {"latitude": lat+0.001, "longitude": lng},
            {"latitude": lat-0.001, "longitude": lng},
            {"latitude": lat, "longitude": lng+0.001},
            {"latitude": lat, "longitude": lng-0.001}
        ]
        url = "https://api.open-elevation.com/api/v1/lookup"
        data = _api_post(url, {"locations": pts})
        if not data or 'results' not in data or not data['results']:
            return {
                "elevation_m": HYD_DEFAULTS["elevation_m"],
                "slope_degrees": HYD_DEFAULTS["slope_degrees"],
                "twi": HYD_DEFAULTS["twi"]
            }
            
        res = data['results']
        elevs = [r['elevation'] for r in res]
        center_el = elevs[0]
        max_diff = max(abs(e - center_el) for e in elevs[1:])
        slope = math.degrees(math.atan(max_diff / 111.0)) # ~111m for 0.001 deg
        
        twi = 10.0
        if slope > 0:
            twi = math.log(1000 / math.tan(math.radians(slope)))
            
        return {
            "elevation_m": round(center_el, 1),
            "slope_degrees": round(slope, 2),
            "twi": round(twi, 2)
        }

    @lru_cache(maxsize=128)
    def _fetch_soilgrids(self, lat, lng):
        # Single attempt only — no retries for urban offsets (too slow)
        url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lng}&lat={lat}&property=clay&property=sand&property=phh2o&depth=0-30cm&value=mean"
        data = _api_get(url)
        if data and 'properties' in data and data['properties'].get('layers'):
            layers = data['properties']['layers']
            vals = {}
            for l in layers:
                name = l['name']
                if l['depths'] and l['depths'][0].get('values') and l['depths'][0]['values'].get('mean') is not None:
                    vals[name] = l['depths'][0]['values']['mean']
            
            if 'clay' in vals and 'sand' in vals and 'phh2o' in vals:
                return {
                    "clay_pct": round(vals['clay'] / 10.0, 1),
                    "sand_pct": round(vals['sand'] / 10.0, 1),
                    "soil_ph": round(vals['phh2o'] / 10.0, 1)
                }
                    
        return {
            "clay_pct": HYD_DEFAULTS["clay_pct"],
            "sand_pct": HYD_DEFAULTS["sand_pct"],
            "soil_ph": HYD_DEFAULTS["soil_ph"]
        }

    def build_features(self, lat, lng, planned_depth_ft, planned_month):
        """
        Build complete feature dict for prediction.
        Returns (features_dict, mandal_data, lineament_zone, rain_extras).
        """
        mandal = self.find_nearest_mandal(lat, lng)
        zone = self.find_lineament_zone(lat, lng)
        dist_water = self.distance_to_nearest_water(lat, lng)

        # Determine land cover heuristically based on distance from city center
        dist_center = _haversine_m(lat, lng, 17.385, 78.486) / 1000  # km
        if dist_center < 10:
            land_cover = "Built Area"
        elif dist_center < 20:
            land_cover = "Built Area" if mandal.get('extraction_stage_pct', 0) > 120 else "Rangeland"
        elif dist_center < 35:
            land_cover = "Crops"
        else:
            land_cover = "Rangeland"
        impervious = IMPERVIOUS_PCT_MAP.get(land_cover, 20)

        # --- Run all 3 external API calls IN PARALLEL with hard timeout ---
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_meteo = executor.submit(self._fetch_open_meteo, lat, lng)
            future_elev = executor.submit(self._fetch_elevation, lat, lng)
            future_soil = executor.submit(self._fetch_soilgrids, lat, lng)
            
            try:
                meteo_data = future_meteo.result(timeout=6)
            except Exception:
                meteo_data = {
                    "rainfall_15yr_annual_avg_mm": HYD_DEFAULTS["rainfall_15yr_annual_avg_mm"],
                    "rainfall_monsoon_avg_mm": HYD_DEFAULTS["rainfall_monsoon_avg_mm"],
                    "et0_annual_avg_mm": HYD_DEFAULTS["et0_annual_avg_mm"],
                    "effective_recharge_mm": HYD_DEFAULTS["effective_recharge_mm"],
                    "heavy_rain_days_per_yr": HYD_DEFAULTS["heavy_rain_days_per_yr"],
                    "moderate_rain_days_per_yr": HYD_DEFAULTS["moderate_rain_days_per_yr"],
                    "rainfall_intensity_ratio": HYD_DEFAULTS["rainfall_intensity_ratio"],
                    "drought_years_in_15": HYD_DEFAULTS["drought_years_in_15"],
                }
            try:
                elev_data = future_elev.result(timeout=6)
            except Exception:
                elev_data = {"elevation_m": HYD_DEFAULTS["elevation_m"], "slope_degrees": HYD_DEFAULTS["slope_degrees"], "twi": HYD_DEFAULTS["twi"]}
            try:
                soil_data = future_soil.result(timeout=6)
            except Exception:
                soil_data = {"clay_pct": HYD_DEFAULTS["clay_pct"], "sand_pct": HYD_DEFAULTS["sand_pct"], "soil_ph": HYD_DEFAULTS["soil_ph"]}

        # CGWB-standard recharge estimation for Deccan hard rock terrain:
        monsoon_rain = meteo_data.get('rainfall_monsoon_avg_mm', 650)
        sand_fraction = soil_data.get('sand_pct', 40) / 100
        recharge_coeff = 0.08 + 0.04 * sand_fraction  # 8-12% range based on soil
        imperv_factor = max(0.1, 1 - impervious / 120)  # Urbanization penalty
        eff_recharge = monsoon_rain * recharge_coeff * imperv_factor
        meteo_data['effective_recharge_mm'] = round(eff_recharge, 1)

        rain_extras = {
            "yearly_rain": meteo_data.pop("yearly_rain", []),
            "yearly_et0": meteo_data.pop("yearly_et0", []),
        }

        features = {
            # Hydrology
            **meteo_data,
            # Terrain
            **elev_data,
            "distance_to_waterbody_m": round(dist_water, 1),
            # Surface
            "impervious_surface_pct": impervious,
            # CGWB (from mandal)
            "gwl_pre_monsoon_mbgl": mandal.get("gwl_pre_monsoon_mbgl", 20),
            "gwl_post_monsoon_mbgl": mandal.get("gwl_post_monsoon_mbgl", 12),
            "gwl_seasonal_fluctuation_m": mandal.get("gwl_pre_monsoon_mbgl", 20) - mandal.get("gwl_post_monsoon_mbgl", 12),
            "extraction_stage_pct": mandal.get("extraction_stage_pct", 100),
            "weathered_zone_thickness_m": mandal.get("weathered_zone_thickness_m", 15),
            "depth_to_bedrock_m": mandal.get("depth_to_bedrock_m", 20),
            "lineament_density_per_km2": zone.get("lineament_density_per_km2", mandal.get("lineament_density_per_km2", 2.0)),
            # Soil
            **soil_data,
            # User input
            "planned_depth_ft": planned_depth_ft,
            "planned_month": planned_month,
            # Categoricals
            "rock_type": mandal.get("rock_type", "Granite-Gneiss"),
            "aquifer_type": mandal.get("aquifer_type", "Fractured"),
            "land_cover_class": land_cover,
        }

        return features, mandal, zone, rain_extras

    def predict(self, features_dict):
        """
        Run ML model prediction. Returns (risk_score, confidence, recommended_depth_ft).
        Falls back to heuristic if model not loaded.
        """
        if self.risk_model is not None and self.model_config is not None:
            return self._ml_predict(features_dict)
        return self._heuristic_predict(features_dict)

    def _ml_predict(self, features_dict):
        """Use trained XGBoost models."""
        import numpy as np

        feature_cols = self.model_config.get('features', [])
        encoders = self.model_config.get('encoders', {})

        # Build feature vector
        row = {}
        for col in feature_cols:
            if col.endswith('_encoded'):
                base = col.replace('_encoded', '')
                if base in encoders and base in features_dict:
                    le = encoders[base]
                    val = features_dict[base]
                    if val in le.classes_:
                        row[col] = le.transform([val])[0]
                    else:
                        row[col] = 0
                else:
                    row[col] = 0
            else:
                row[col] = features_dict.get(col, 0)

        X = np.array([[row.get(c, 0) for c in feature_cols]], dtype=np.float64)

        # Risk prediction (model predicts success probability -> invert for risk)
        success_prob = self.risk_model.predict_proba(X)[0][1]
        risk_score = int(round((1 - success_prob) * 100))
        risk_score = max(0, min(100, risk_score))
        confidence = round(max(success_prob, 1 - success_prob) * 100, 1)

        # Depth prediction
        rec_depth = 500
        if self.depth_model is not None:
            rec_depth = int(round(self.depth_model.predict(X)[0]))
            rec_depth = max(200, min(1200, rec_depth))

        return risk_score, confidence, rec_depth

    def _heuristic_predict(self, features_dict):
        """Fallback heuristic when ML models are unavailable."""
        gwl = features_dict.get("gwl_pre_monsoon_mbgl", 20)
        extraction = features_dict.get("extraction_stage_pct", 100)
        lineament = features_dict.get("lineament_density_per_km2", 2.0)
        impervious = features_dict.get("impervious_surface_pct", 50)
        recharge = features_dict.get("effective_recharge_mm", 80)

        risk = 50
        risk += (gwl - 15) * 2
        risk += (extraction - 100) * 0.2
        risk -= (lineament - 2.0) * 10
        risk += (impervious - 30) * 0.3
        risk -= recharge * 0.05
        risk = max(5, min(95, int(risk)))

        confidence = 72.0
        rec_depth = int(gwl * 3.28 * 10 + 200)
        rec_depth = max(200, min(1000, rec_depth))

        return risk, confidence, rec_depth

    def _generate_detailed_explanations(self, features_dict, mandal, zone):
        explanations = []

        # Extraction Stage
        ext = features_dict.get("extraction_stage_pct", 100)
        if ext > 100:
            assessment = "risk"
            narrative = "The aquifer is over-exploited, meaning more water is drawn than recharged. The water table is likely declining, increasing the risk of early borewell failure."
        elif ext >= 70:
            assessment = "neutral"
            narrative = "The area is semi-critical. Extraction and recharge are balanced but vulnerable to drought seasons."
        else:
            assessment = "advantage"
            narrative = "The area is in a safe zone with sustainable extraction levels. Groundwater reserves are relatively stable."
        explanations.append({"factor": "Extraction Stage", "value": f"{ext}", "unit": "%", "assessment": assessment, "narrative": narrative, "source": "CGWB", "recommendation": "Monitor yields."})

        # Pre-Monsoon GWL
        pre_gwl = features_dict.get("gwl_pre_monsoon_mbgl", 20)
        if pre_gwl > 20:
            assessment = "risk"
            narrative = "The water table is deep, requiring deeper drilling and more powerful pumps. Aquifer may be stressed."
        elif pre_gwl >= 10:
            assessment = "neutral"
            narrative = "Water table is at a moderate depth. Standard drilling equipment is sufficient."
        else:
            assessment = "advantage"
            narrative = "Water table is shallow, indicating good groundwater availability and easier access."
        explanations.append({"factor": "Pre-Monsoon GWL", "value": f"{pre_gwl}", "unit": "m", "assessment": assessment, "narrative": narrative, "source": "CGWB", "recommendation": "Plan drilling depth accordingly."})

        # Post-Monsoon GWL
        post_gwl = features_dict.get("gwl_post_monsoon_mbgl", 12)
        if post_gwl > 20:
            assessment = "risk"
            narrative = "Even after monsoon, the water table remains deep, showing poor recharge."
        elif post_gwl >= 10:
            assessment = "neutral"
            narrative = "Post-monsoon water table is at a moderate depth."
        else:
            assessment = "advantage"
            narrative = "Water table recovers well after monsoon, indicating excellent recharge potential."
        explanations.append({"factor": "Post-Monsoon GWL", "value": f"{post_gwl}", "unit": "m", "assessment": assessment, "narrative": narrative, "source": "CGWB", "recommendation": "Use this as baseline for maximum yield."})

        # Lineament Density
        lin = features_dict.get("lineament_density_per_km2", 2.0)
        if lin > 2.5:
            assessment = "advantage"
            narrative = "High density of fractures provides excellent conduits for groundwater flow and storage."
        elif lin >= 1.5:
            assessment = "neutral"
            narrative = "Moderate fracture density offers reasonable chances of hitting water-bearing zones."
        else:
            assessment = "risk"
            narrative = "Low fracture density means fewer pathways for groundwater. Hitting an aquifer is more difficult."
        explanations.append({"factor": "Lineament Density", "value": f"{lin}", "unit": "km/km2", "assessment": assessment, "narrative": narrative, "source": "CGWB", "recommendation": "Conduct detailed geophysical survey."})

        # Weathered Zone Thickness
        wzt = features_dict.get("weathered_zone_thickness_m", 15)
        if wzt > 20:
            assessment = "advantage"
            narrative = "A thick weathered zone acts as a good reservoir for groundwater storage and recharge."
        elif wzt >= 10:
            assessment = "neutral"
            narrative = "Moderate weathered zone thickness provides typical storage capacity."
        else:
            assessment = "risk"
            narrative = "Thin weathered zone offers limited storage capacity and poorer recharge potential."
        explanations.append({"factor": "Weathered Zone Thickness", "value": f"{wzt}", "unit": "m", "assessment": assessment, "narrative": narrative, "source": "CGWB", "recommendation": "Casing is required through this zone."})

        # Rock Type
        rock = features_dict.get("rock_type", "Granite-Gneiss")
        if rock == "Granite-Gneiss":
            assessment = "neutral"
            narrative = "Granite-Gneiss is a hard rock. Groundwater exists only in fractures and weathered portions."
        elif rock == "Deccan-Basalt":
            assessment = "advantage"
            narrative = "Deccan-Basalt often has vesicular and fractured zones that can serve as good aquifers."
        else:
            assessment = "neutral"
            narrative = f"{rock} requires careful site selection for successful drilling."
        explanations.append({"factor": "Rock Type", "value": rock, "unit": "", "assessment": assessment, "narrative": narrative, "source": "CGWB", "recommendation": "Select rig suitable for rock type."})

        # Annual Rainfall
        rain = features_dict.get("rainfall_15yr_annual_avg_mm", 800)
        if rain > 900:
            assessment = "advantage"
            narrative = "High annual rainfall ensures good potential for aquifer replenishment."
        elif rain >= 700:
            assessment = "neutral"
            narrative = "Moderate rainfall provides average groundwater recharge."
        else:
            assessment = "risk"
            narrative = "Low rainfall severely limits natural recharge, increasing dependency on existing storage."
        explanations.append({"factor": "Annual Rainfall", "value": f"{rain}", "unit": "mm", "assessment": assessment, "narrative": narrative, "source": "Open-Meteo API", "recommendation": "Implement rainwater harvesting."})

        # Effective Recharge
        rech = features_dict.get("effective_recharge_mm", 80)
        if rech > 100:
            assessment = "advantage"
            narrative = "High effective recharge indicates strong natural replenishment of the aquifer."
        elif rech >= 50:
            assessment = "neutral"
            narrative = "Moderate effective recharge supports sustainable extraction if managed well."
        else:
            assessment = "risk"
            narrative = "Low effective recharge means the aquifer takes longer to recover from extraction."
        explanations.append({"factor": "Effective Recharge", "value": f"{rech}", "unit": "mm", "assessment": assessment, "narrative": narrative, "source": "Open-Meteo API", "recommendation": "Limit excessive pumping."})

        # Elevation and Slope
        slope = features_dict.get("slope_degrees", 2.0)
        if slope < 2.0:
            assessment = "advantage"
            narrative = "Flat terrain slows runoff, allowing more time for water to percolate and recharge the aquifer."
        elif slope <= 5.0:
            assessment = "neutral"
            narrative = "Gentle slope provides moderate runoff and reasonable recharge conditions."
        else:
            assessment = "risk"
            narrative = "Steep slope causes rapid runoff, minimizing water infiltration and reducing recharge."
        explanations.append({"factor": "Slope", "value": f"{slope}", "unit": "°", "assessment": assessment, "narrative": narrative, "source": "Open-Elevation API", "recommendation": "Consider contour trenching."})

        # Soil
        sand = features_dict.get("sand_pct", 40)
        clay = features_dict.get("clay_pct", 35)
        if sand > 50:
            assessment = "advantage"
            narrative = "Sandy soil is highly permeable, allowing excellent infiltration of surface water."
        elif clay > 40:
            assessment = "risk"
            narrative = "High clay content makes the soil impermeable, significantly reducing groundwater recharge."
        else:
            assessment = "neutral"
            narrative = "Loamy soil offers a balance between water retention and infiltration."
        explanations.append({"factor": "Soil (Sand/Clay)", "value": f"{sand}% Sand, {clay}% Clay", "unit": "", "assessment": assessment, "narrative": narrative, "source": "SoilGrids API", "recommendation": ""})

        # Distance to Water Body
        dist = features_dict.get("distance_to_waterbody_m", 1000)
        if dist < 500:
            assessment = "advantage"
            narrative = "Close proximity to a water body significantly enhances local groundwater recharge."
        elif dist <= 2000:
            assessment = "neutral"
            narrative = "Moderate distance from surface water provides some beneficial influence on the local aquifer."
        else:
            assessment = "risk"
            narrative = "Far from surface water, meaning recharge relies entirely on local rainfall and infiltration."
        explanations.append({"factor": "Distance to Water Body", "value": f"{dist}", "unit": "m", "assessment": assessment, "narrative": narrative, "source": "Heuristic", "recommendation": ""})

        # Impervious Surface
        imp = features_dict.get("impervious_surface_pct", 50)
        if imp < 30:
            assessment = "advantage"
            narrative = "Low impervious cover allows maximum natural infiltration of rainfall."
        elif imp <= 60:
            assessment = "neutral"
            narrative = "Moderate impervious cover restricts some recharge but still allows significant infiltration."
        else:
            assessment = "risk"
            narrative = "High impervious cover (pavement/buildings) blocks rainfall from entering the aquifer."
        explanations.append({"factor": "Impervious Surface", "value": f"{imp}", "unit": "%", "assessment": assessment, "narrative": narrative, "source": "Heuristic", "recommendation": "Mandatory rainwater harvesting."})

        return explanations



# Singleton instance
feature_engine_instance = FeatureEngine()
