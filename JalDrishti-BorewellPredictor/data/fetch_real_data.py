"""
fetch_real_data.py - Builds training dataset from real APIs for Hyderabad region.
Usage:
    python data/fetch_real_data.py          # Full ~200 points
    python data/fetch_real_data.py --quick  # Quick 30 points for testing
"""
import json, os, sys, time, math, random, hashlib
import urllib.request, urllib.parse, urllib.error
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = DATA_DIR / "processed"
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Load curated data
with open(DATA_DIR / "hyderabad_mandals.json") as f:
    MANDALS = json.load(f)
with open(DATA_DIR / "hyderabad_waterbodies.json") as f:
    WATERBODIES = json.load(f)
with open(DATA_DIR / "hyderabad_lineaments.json") as f:
    LINEAMENTS = json.load(f)

# Hyderabad bounding box
BBOX = {"min_lat": 17.10, "max_lat": 17.70, "min_lng": 77.50, "max_lng": 78.75}


def cache_key(prefix, lat, lng):
    s = f"{prefix}_{lat:.4f}_{lng:.4f}"
    return CACHE_DIR / f"{hashlib.md5(s.encode()).hexdigest()}.json"


def cached_api_call(prefix, lat, lng, fetch_fn):
    """Cache API responses to avoid re-fetching."""
    path = cache_key(prefix, lat, lng)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    data = fetch_fn()
    if data is not None:
        with open(path, "w") as f:
            json.dump(data, f)
    return data


def api_get(url, headers=None, timeout=20):
    """Simple GET with error handling."""
    try:
        req = urllib.request.Request(url, headers=headers or {})
        req.add_header("User-Agent", "JalDrishti/2.0 (borewell-predictor)")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  [WARN] API error for {url[:80]}...: {e}")
        return None


def haversine(lat1, lng1, lat2, lng2):
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


# ---- API Fetchers ----

def fetch_rainfall(lat, lng):
    """Fetch 15 years of daily precipitation + ET0 from Open-Meteo."""
    def _fetch():
        all_precip = []
        all_et0 = []
        for start_yr, end_yr in [(2010,2014),(2015,2019),(2020,2024)]:
            url = (f"https://archive-api.open-meteo.com/v1/archive?"
                   f"latitude={lat}&longitude={lng}"
                   f"&start_date={start_yr}-01-01&end_date={end_yr}-12-31"
                   f"&daily=precipitation_sum,et0_fao_evapotranspiration"
                   f"&timezone=Asia/Kolkata")
            data = api_get(url)
            if data and "daily" in data:
                all_precip.extend(data["daily"].get("precipitation_sum", []))
                all_et0.extend(data["daily"].get("et0_fao_evapotranspiration", []))
            time.sleep(0.3)
        return {"precipitation": all_precip, "et0": all_et0}
    return cached_api_call("rain", lat, lng, _fetch)


def compute_rainfall_features(rain_data):
    """Compute 8 hydrology features from 15 years of daily data."""
    if not rain_data:
        return {
            "rainfall_15yr_annual_avg_mm": 800, "rainfall_monsoon_avg_mm": 600,
            "et0_annual_avg_mm": 1500, "effective_recharge_mm": 0,
            "heavy_rain_days_per_yr": 3, "moderate_rain_days_per_yr": 25,
            "rainfall_intensity_ratio": 0.1, "drought_years_in_15": 3
        }
    precip = [p if p is not None else 0.0 for p in rain_data.get("precipitation", [])]
    et0 = [e if e is not None else 0.0 for e in rain_data.get("et0", [])]

    if len(precip) < 365:
        return compute_rainfall_features(None)  # fallback

    # Reshape into years (approximate 365 days/year)
    n_years = len(precip) // 365
    yearly_rain = []
    yearly_monsoon = []
    yearly_et0_total = []
    heavy_days_total = 0
    moderate_days_total = 0
    total_rain_days = 0

    for yr in range(n_years):
        start = yr * 365
        end = start + 365
        yr_precip = precip[start:end]
        yr_et0 = et0[start:end]

        yearly_rain.append(sum(yr_precip))
        yearly_et0_total.append(sum(yr_et0))

        # Monsoon: days 152-273 (Jun 1 - Sep 30 approx)
        monsoon_precip = yr_precip[152:274]
        yearly_monsoon.append(sum(monsoon_precip))

        for p in yr_precip:
            if p > 50:
                heavy_days_total += 1
                total_rain_days += 1
            elif p > 10:
                moderate_days_total += 1
                total_rain_days += 1
            elif p > 0.1:
                total_rain_days += 1

    avg_annual = np.mean(yearly_rain) if yearly_rain else 800
    avg_monsoon = np.mean(yearly_monsoon) if yearly_monsoon else 600
    avg_et0 = np.mean(yearly_et0_total) if yearly_et0_total else 1500
    eff_recharge = max(0, avg_annual - avg_et0)
    heavy_per_yr = heavy_days_total / max(1, n_years)
    moderate_per_yr = moderate_days_total / max(1, n_years)
    intensity_ratio = heavy_days_total / max(1, total_rain_days)
    drought_yrs = sum(1 for r in yearly_rain if r < 0.7 * avg_annual)

    return {
        "rainfall_15yr_annual_avg_mm": round(avg_annual, 1),
        "rainfall_monsoon_avg_mm": round(avg_monsoon, 1),
        "et0_annual_avg_mm": round(avg_et0, 1),
        "effective_recharge_mm": round(eff_recharge, 1),
        "heavy_rain_days_per_yr": round(heavy_per_yr, 1),
        "moderate_rain_days_per_yr": round(moderate_per_yr, 1),
        "rainfall_intensity_ratio": round(intensity_ratio, 3),
        "drought_years_in_15": drought_yrs,
    }


def fetch_soil(lat, lng):
    """Fetch soil properties from SoilGrids."""
    def _fetch():
        url = (f"https://rest.isric.org/soilgrids/v2.0/properties/query?"
               f"lat={lat}&lon={lng}"
               f"&property=clay&property=sand&property=silt&property=soc&property=bdod&property=phh2o"
               f"&depth=0-5cm&value=mean")
        return api_get(url)
    return cached_api_call("soil", lat, lng, _fetch)


def parse_soil(soil_data):
    """Extract soil features from SoilGrids response."""
    defaults = {"clay_pct": 25.0, "sand_pct": 40.0, "silt_pct": 35.0, "soil_ph": 7.0}
    if not soil_data or "properties" not in soil_data:
        return defaults
    try:
        layers = soil_data["properties"]["layers"]
        result = {}
        for layer in layers:
            name = layer["name"]
            val = layer["depths"][0]["values"].get("mean")
            if val is None:
                continue
            if name == "clay":
                result["clay_pct"] = round(val / 10.0, 1)
            elif name == "sand":
                result["sand_pct"] = round(val / 10.0, 1)
            elif name == "silt":
                result["silt_pct"] = round(val / 10.0, 1)
            elif name == "phh2o":
                result["soil_ph"] = round(val / 10.0, 1)
        for k, v in defaults.items():
            result.setdefault(k, v)
        return result
    except Exception:
        return defaults


def fetch_elevation_grid(lat, lng):
    """Fetch 5-point cross elevation pattern from SRTM for slope/TWI."""
    def _fetch():
        d = 0.001  # ~111m spacing
        locs = f"{lat},{lng}|{lat+d},{lng}|{lat-d},{lng}|{lat},{lng+d}|{lat},{lng-d}"
        url = f"https://api.opentopodata.org/v1/srtm30m?locations={locs}"
        return api_get(url)
    return cached_api_call("elev", lat, lng, _fetch)


def compute_terrain(elev_data):
    """Compute elevation, slope, and TWI from 5-point cross pattern."""
    defaults = {"elevation_m": 500, "slope_degrees": 2.0, "twi": 8.0}
    if not elev_data or "results" not in elev_data or len(elev_data["results"]) < 5:
        return defaults
    try:
        elevs = [r["elevation"] for r in elev_data["results"]]
        center = elevs[0]
        north, south, east, west = elevs[1], elevs[2], elevs[3], elevs[4]

        dx = 111.0  # meters per 0.001 degree (approximate)
        dz_ns = (north - south) / (2 * dx)
        dz_ew = (east - west) / (2 * dx)
        slope_rad = math.atan(math.sqrt(dz_ns**2 + dz_ew**2))
        slope_deg = math.degrees(slope_rad)

        # Simplified TWI
        tan_slope = max(math.tan(slope_rad), 0.01)
        twi = math.log(1000.0 / tan_slope)

        return {
            "elevation_m": round(center, 1),
            "slope_degrees": round(slope_deg, 2),
            "twi": round(twi, 2),
        }
    except Exception:
        return defaults


def fetch_land_cover(lat, lng):
    """Fetch land use class from Esri 10m Land Cover."""
    def _fetch():
        mosaic = urllib.parse.quote('{"where":"Year=2023"}')
        url = (f"https://ic.imagery1.arcgis.com/arcgis/rest/services/"
               f"Sentinel2_10m_LandCover/ImageServer/identify?"
               f"geometry={lng},{lat}&geometryType=esriGeometryPoint"
               f"&sr=4326&mosaicRule={mosaic}&f=json")
        return api_get(url)
    return cached_api_call("lulc", lat, lng, _fetch)


def parse_land_cover(lc_data):
    """Extract land cover class and impervious %."""
    from ml.feature_config import LAND_COVER_CLASSES, IMPERVIOUS_PCT_MAP
    defaults = {"land_cover_class": "Built Area", "impervious_surface_pct": 50}
    if not lc_data:
        return defaults
    try:
        val = int(lc_data.get("value", 7))
        cls_name = LAND_COVER_CLASSES.get(val, "Built Area")
        imp_pct = IMPERVIOUS_PCT_MAP.get(cls_name, 50)
        return {"land_cover_class": cls_name, "impervious_surface_pct": imp_pct}
    except Exception:
        return defaults


def find_nearest_mandal(lat, lng):
    """Find the closest mandal by haversine distance."""
    best = None
    best_dist = float("inf")
    for m in MANDALS:
        d = haversine(lat, lng, m["lat"], m["lng"])
        if d < best_dist:
            best_dist = d
            best = m
    return best


def find_lineament_zone(lat, lng):
    """Find which lineament zone contains this point."""
    for zone in LINEAMENTS.get("zones", []):
        bb = zone["bbox"]
        if bb["min_lat"] <= lat <= bb["max_lat"] and bb["min_lng"] <= lng <= bb["max_lng"]:
            return zone
    # Default to nearest zone by centroid
    return LINEAMENTS["zones"][0] if LINEAMENTS.get("zones") else None


def distance_to_nearest_waterbody(lat, lng):
    """Compute distance to nearest water body in meters."""
    min_dist = float("inf")
    for wb in WATERBODIES:
        d = haversine(lat, lng, wb["lat"], wb["lng"])
        min_dist = min(min_dist, d)
    return round(min_dist, 0)


def generate_grid(quick=False):
    """Generate sample points across Hyderabad region."""
    if quick:
        n_lat, n_lng = 5, 6
    else:
        n_lat, n_lng = 12, 17

    lats = np.linspace(BBOX["min_lat"], BBOX["max_lat"], n_lat)
    lngs = np.linspace(BBOX["min_lng"], BBOX["max_lng"], n_lng)

    points = []
    for lat in lats:
        for lng in lngs:
            points.append((round(lat, 4), round(lng, 4)))

    # Add mandal centroids
    for m in MANDALS:
        points.append((round(m["lat"], 4), round(m["lng"], 4)))

    # Deduplicate
    points = list(set(points))
    random.shuffle(points)
    print(f"[OK] Generated {len(points)} sample points")
    return points


def build_features_for_point(lat, lng, idx, total):
    """Build the complete feature vector for one point."""
    print(f"  [{idx+1}/{total}] Processing ({lat:.4f}, {lng:.4f})...")

    features = {"latitude": lat, "longitude": lng}

    # 1. Rainfall + ET0 (15 years)
    rain_data = fetch_rainfall(lat, lng)
    features.update(compute_rainfall_features(rain_data))
    time.sleep(0.2)

    # 2. Soil properties
    soil_data = fetch_soil(lat, lng)
    features.update(parse_soil(soil_data))
    time.sleep(0.2)

    # 3. Elevation + Slope + TWI
    elev_data = fetch_elevation_grid(lat, lng)
    features.update(compute_terrain(elev_data))
    time.sleep(1.0)  # OpenTopoData rate limit

    # 4. Land cover
    lc_data = fetch_land_cover(lat, lng)
    features.update(parse_land_cover(lc_data))
    time.sleep(0.3)

    # 5. Nearest mandal (CGWB data)
    mandal = find_nearest_mandal(lat, lng)
    if mandal:
        features["gwl_pre_monsoon_mbgl"] = mandal["gwl_pre_monsoon_mbgl"]
        features["gwl_post_monsoon_mbgl"] = mandal["gwl_post_monsoon_mbgl"]
        features["gwl_seasonal_fluctuation_m"] = round(
            mandal["gwl_pre_monsoon_mbgl"] - mandal["gwl_post_monsoon_mbgl"], 1)
        features["extraction_stage_pct"] = mandal["extraction_stage_pct"]
        features["rock_type"] = mandal["rock_type"]
        features["aquifer_type"] = mandal["aquifer_type"]
        features["weathered_zone_thickness_m"] = mandal["weathered_zone_thickness_m"]
        features["depth_to_bedrock_m"] = mandal["depth_to_bedrock_m"]
        features["lineament_density_per_km2"] = mandal["lineament_density_per_km2"]
        features["avg_borewell_depth_ft"] = mandal["avg_borewell_depth_ft"]
        features["estimated_success_rate"] = mandal["estimated_success_rate"]
    else:
        features.update({
            "gwl_pre_monsoon_mbgl": 15, "gwl_post_monsoon_mbgl": 8,
            "gwl_seasonal_fluctuation_m": 7, "extraction_stage_pct": 70,
            "rock_type": "Granite-Gneiss", "aquifer_type": "Weathered",
            "weathered_zone_thickness_m": 20, "depth_to_bedrock_m": 25,
            "lineament_density_per_km2": 2.0, "avg_borewell_depth_ft": 500,
            "estimated_success_rate": 0.50
        })

    # 6. Lineament zone refinement
    zone = find_lineament_zone(lat, lng)
    if zone:
        features["lineament_density_per_km2"] = zone["lineament_density_per_km2"]

    # 7. Distance to water
    features["distance_to_waterbody_m"] = distance_to_nearest_waterbody(lat, lng)

    # 8. Random planned parameters (for training variety)
    features["planned_depth_ft"] = random.choice([200,300,400,500,600,700,800,900,1000])
    features["planned_month"] = random.randint(1, 12)

    # 9. Simulate borewell outcome
    base_prob = features.get("estimated_success_rate", 0.50)
    # Modifiers based on features
    if features["lineament_density_per_km2"] > 2.5:
        base_prob += 0.10
    if features["distance_to_waterbody_m"] < 500:
        base_prob += 0.05
    if features.get("land_cover_class") == "Built Area":
        base_prob -= 0.12
    if features["effective_recharge_mm"] > 400:
        base_prob += 0.08
    elif features["effective_recharge_mm"] < 100:
        base_prob -= 0.05
    if features["planned_depth_ft"] < features["avg_borewell_depth_ft"] * 0.5:
        base_prob -= 0.08
    if features["slope_degrees"] > 5:
        base_prob -= 0.05
    if features["twi"] > 10:
        base_prob += 0.05
    # Monsoon months slightly better
    if features["planned_month"] in [7, 8, 9, 10]:
        base_prob += 0.05

    base_prob = max(0.05, min(0.95, base_prob))
    features["borewell_success"] = 1 if random.random() < base_prob else 0

    return features


def main():
    quick = "--quick" in sys.argv
    print(f"[OK] JalDrishti v2 Data Collection {'(QUICK MODE)' if quick else '(FULL MODE)'}")
    print(f"[OK] Bounding box: {BBOX}")

    points = generate_grid(quick)
    all_features = []

    for i, (lat, lng) in enumerate(points):
        try:
            feat = build_features_for_point(lat, lng, i, len(points))
            all_features.append(feat)
        except Exception as e:
            print(f"  [ERROR] Failed for ({lat}, {lng}): {e}")
            continue

    df = pd.DataFrame(all_features)
    output_path = OUTPUT_DIR / "hyderabad_training_data.csv"
    df.to_csv(output_path, index=False)

    print(f"\n[OK] Dataset generated: {df.shape}")
    print(f"[OK] Success rate: {df['borewell_success'].mean():.3f}")
    print(f"[OK] Failure rate: {1 - df['borewell_success'].mean():.3f}")
    print(f"[OK] Saved to {output_path}")
    print(f"\nFeature columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
