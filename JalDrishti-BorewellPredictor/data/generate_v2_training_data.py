import json
import random
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANDALS_FILE = PROJECT_ROOT / "data" / "hyderabad_mandals.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "hyderabad_training_data.csv"

def generate_data():
    with open(MANDALS_FILE, 'r') as f:
        mandals = json.load(f)

    rows = []
    
    for mandal in mandals:
        # Generate 45 rows per mandal
        for _ in range(45):
            row = {}
            
            # Categorical
            row['rock_type'] = mandal['rock_type']
            row['aquifer_type'] = mandal['aquifer_type']
            
            land_cover = random.choices(
                ['Built Area', 'Trees', 'Crops', 'Bare Ground', 'Rangeland', 'Water'],
                weights=[0.4, 0.15, 0.2, 0.15, 0.05, 0.05]
            )[0]
            row['land_cover_class'] = land_cover
            
            # Urban
            if land_cover == 'Built Area':
                row['impervious_surface_pct'] = np.random.uniform(70, 95)
            elif land_cover in ['Trees', 'Crops']:
                row['impervious_surface_pct'] = np.random.uniform(5, 25)
            elif land_cover == 'Water':
                row['impervious_surface_pct'] = 0
            else:
                row['impervious_surface_pct'] = np.random.uniform(5, 15)

            # Hydrology
            row['rainfall_15yr_annual_avg_mm'] = np.random.uniform(700, 1000)
            row['rainfall_monsoon_avg_mm'] = np.random.uniform(500, 800)
            row['et0_annual_avg_mm'] = np.random.uniform(1200, 1600)
            row['effective_recharge_mm'] = row['rainfall_15yr_annual_avg_mm'] * np.random.uniform(0.05, 0.15)
            row['heavy_rain_days_per_yr'] = int(np.random.uniform(3, 10))
            row['moderate_rain_days_per_yr'] = int(np.random.uniform(20, 40))
            row['rainfall_intensity_ratio'] = np.random.uniform(0.6, 0.8)
            row['drought_years_in_15'] = int(np.random.uniform(1, 4))
            
            # Topography
            row['elevation_m'] = np.random.uniform(400, 600)
            row['slope_degrees'] = np.random.uniform(0, 8)
            row['twi'] = np.random.uniform(6, 14)
            row['distance_to_waterbody_m'] = np.random.uniform(10, 3000)
            
            # CGWB Groundwater (Inherit with noise)
            row['gwl_pre_monsoon_mbgl'] = max(1.0, np.random.normal(mandal['gwl_pre_monsoon_mbgl'], 2.0))
            row['gwl_post_monsoon_mbgl'] = max(0.5, np.random.normal(mandal['gwl_post_monsoon_mbgl'], 1.5))
            row['gwl_seasonal_fluctuation_m'] = max(0.0, row['gwl_pre_monsoon_mbgl'] - row['gwl_post_monsoon_mbgl'])
            
            ext_stage = max(20.0, np.random.normal(mandal['extraction_stage_pct'], 10.0))
            row['extraction_stage_pct'] = ext_stage
            
            row['weathered_zone_thickness_m'] = max(5.0, np.random.normal(mandal['weathered_zone_thickness_m'], 3.0))
            row['depth_to_bedrock_m'] = max(row['weathered_zone_thickness_m'] + 2, np.random.normal(mandal['depth_to_bedrock_m'], 4.0))
            row['lineament_density_per_km2'] = max(0.1, np.random.normal(mandal['lineament_density_per_km2'], 0.5))
            
            # Soil
            row['clay_pct'] = np.random.uniform(15, 35)
            row['sand_pct'] = np.random.uniform(35, 55)
            row['soil_ph'] = np.random.uniform(6.5, 8.0)
            
            # Target Depth and Planned Depth
            avg_depth = max(200.0, np.random.normal(mandal['avg_borewell_depth_ft'], 50.0))
            row['avg_borewell_depth_ft'] = avg_depth
            row['planned_depth_ft'] = max(200.0, np.random.normal(avg_depth, 30.0))
            row['planned_month'] = random.randint(1, 12)
            
            # Success probability
            if ext_stage > 100:
                base_prob = np.random.uniform(0.10, 0.20)
            elif ext_stage >= 90:
                base_prob = np.random.uniform(0.30, 0.40)
            elif ext_stage >= 70:
                base_prob = np.random.uniform(0.50, 0.65)
            else:
                base_prob = np.random.uniform(0.70, 0.85)
                
            # Modifiers
            if row['lineament_density_per_km2'] > 2.5:
                base_prob += 0.1
            if row['land_cover_class'] == 'Built Area':
                base_prob -= 0.1
            if row['effective_recharge_mm'] > 100:
                base_prob += 0.05
                
            base_prob = np.clip(base_prob, 0.0, 1.0)
            row['borewell_success'] = np.random.binomial(1, base_prob)
            
            rows.append(row)
            
    df = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Generated {len(df)} rows and saved to {OUTPUT_FILE}")
    print(f"Columns: {df.columns.tolist()}")

if __name__ == '__main__':
    generate_data()
