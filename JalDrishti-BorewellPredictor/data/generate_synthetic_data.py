import os
import json
import pandas as pd
import numpy as np

np.random.seed(42)

def generate_data():
    N = 5000
    
    lookup_path = os.path.join(os.path.dirname(__file__), 'district_lookup.json')
    with open(lookup_path, 'r') as f:
        districts_list = json.load(f)
    
    # Build lookup by state -> district
    districts_by_state = {}
    for d in districts_list:
        state = d['state']
        if state not in districts_by_state:
            districts_by_state[state] = []
        districts_by_state[state].append(d)
    
    records = []
    states = list(districts_by_state.keys())
    
    for _ in range(N):
        state = np.random.choice(states)
        d_info = districts_by_state[state][np.random.randint(len(districts_by_state[state]))]
        district = d_info['district']
        
        rock = d_info.get("rock_type", "Granite")
        soil = d_info.get("soil_type", "Red")
        aquifer = d_info.get("aquifer_type", "Fractured")
        stage = d_info.get("extraction_stage", 1)
        gwl_base = d_info.get("avg_gwl", 50)
        rain_base = d_info.get("avg_rainfall", 800)
        base_lat = d_info.get("lat", 20.0)
        base_lng = d_info.get("lng", 78.0)
        
        # Realistic coordinates near the district center
        lat = base_lat + np.random.uniform(-0.5, 0.5)
        lon = base_lng + np.random.uniform(-0.5, 0.5)
        elev = np.random.uniform(10, 1000)
        
        gwl_current = max(5, np.random.normal(gwl_base, gwl_base * 0.2))
        rain_ann = max(100, np.random.normal(rain_base, rain_base * 0.15))
        
        gwl_pre = gwl_current + np.random.uniform(0, 10)
        gwl_post = max(5, gwl_current - np.random.uniform(5, 20))
        decline_rate = np.random.uniform(-0.5, 2.5) if stage > 1 else np.random.uniform(-0.1, 0.5)
        
        ext_pct = np.random.uniform(80, 150) if stage >= 2 else np.random.uniform(20, 80)
        rain_mon = rain_ann * np.random.uniform(0.6, 0.9)
        rain_def = np.random.uniform(-10, 50)
        drought_yrs = np.random.choice([0, 1, 2, 3], p=[0.6, 0.2, 0.1, 0.1])
        rain_trend = np.random.uniform(-20, 20)
        
        dist_river = np.random.exponential(5)
        dist_wb = np.random.exponential(2)
        pop_dens = d_info.get("population_density", np.random.uniform(100, 1000))
        well_dens = d_info.get("well_density", np.random.uniform(5, 50))
        slope = np.random.uniform(0, 15)
        month = np.random.randint(1, 13)
        planned_depth = np.random.uniform(100, 1200)
        
        # Correlated Target Generation
        score = 0
        if stage >= 3: score -= 3
        elif stage >= 2: score -= 1.5
        
        if rock in ["Granite", "Schist", "Gneiss"]: score -= 1
        elif rock in ["Alluvium", "Sandstone"]: score += 2
        elif rock == "Laterite": score += 1.5
        
        if gwl_current > 100: score -= 2
        elif gwl_current > 60: score -= 0.5
        elif gwl_current < 30: score += 2
        
        if rain_ann < 500: score -= 1.5
        elif rain_ann > 1200: score += 1.5
        elif rain_ann > 800: score += 0.5
        
        if state in ["Rajasthan"]: score -= 1.5
        elif state in ["Madhya Pradesh"]: score -= 0.5
        if state in ["Kerala"]: score += 2
        elif state in ["Gujarat"] and rock == "Alluvium": score += 1
        
        if decline_rate > 1.5: score -= 1
        if drought_yrs >= 2: score -= 0.5
        if dist_river < 2: score += 0.5
        
        prob_success = 1 / (1 + np.exp(-score))
        success = int(np.random.binomial(1, prob_success))
        
        actual_yield = np.random.uniform(1, 10) if success else np.random.uniform(0, 0.5)
        rec_depth = gwl_current * 3.28 + np.random.uniform(50, 150)  # Convert m to ft approx
        
        records.append({
            "latitude": round(lat, 4), "longitude": round(lon, 4),
            "state": state, "district": district,
            "elevation_m": round(elev, 1), "rock_type": rock,
            "soil_type": soil, "aquifer_type": aquifer,
            "gwl_current_mbgl": round(gwl_current, 2),
            "gwl_pre_monsoon_mbgl": round(gwl_pre, 2),
            "gwl_post_monsoon_mbgl": round(gwl_post, 2),
            "gwl_decline_rate_m_per_yr": round(decline_rate, 2),
            "extraction_stage": stage,
            "extraction_percentage": round(ext_pct, 1),
            "rainfall_annual_mm": round(rain_ann, 1),
            "rainfall_monsoon_mm": round(rain_mon, 1),
            "rainfall_deficit_pct": round(rain_def, 1),
            "consecutive_drought_years": int(drought_yrs),
            "rainfall_trend_5yr": round(rain_trend, 2),
            "dist_to_river_km": round(dist_river, 2),
            "dist_to_waterbody_km": round(dist_wb, 2),
            "population_density": round(pop_dens, 0),
            "well_density_per_km2": round(well_dens, 1),
            "slope_degrees": round(slope, 2),
            "month_of_drilling": int(month),
            "planned_depth_ft": round(planned_depth, 0),
            "borewell_success": success,
            "actual_yield_lps": round(actual_yield, 2),
            "recommended_depth_ft": round(rec_depth, 0)
        })
        
    df = pd.DataFrame(records)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "training_data.csv")
    df.to_csv(out_file, index=False)
    
    print(f"Generated synthetic data: {df.shape}")
    print(f"Success rate: {df['borewell_success'].mean():.3f}")
    print(f"Failure rate: {1 - df['borewell_success'].mean():.3f}")
    print(f"States: {df['state'].nunique()}, Districts: {df['district'].nunique()}")
    print(f"Data saved to {out_file}")

if __name__ == "__main__":
    generate_data()
