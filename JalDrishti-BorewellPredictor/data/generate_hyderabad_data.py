import json
import random
import math
import csv
import os

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # radius of Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def perturb(val, pct=0.1):
    return val * random.uniform(1 - pct, 1 + pct)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(base_dir, 'hyderabad_mandals.json'), 'r') as f:
        mandals = json.load(f)
        
    with open(os.path.join(base_dir, 'hyderabad_waterbodies.json'), 'r') as f:
        waterbodies = json.load(f)
        
    with open(os.path.join(base_dir, 'hyderabad_lineaments.json'), 'r') as f:
        lineaments = json.load(f)
        
    data = []
    
    for mandal in mandals:
        for _ in range(22):
            # Generate points within 5km radius (approx 0.045 degrees)
            lat_offset = random.uniform(-0.045, 0.045)
            lng_offset = random.uniform(-0.045, 0.045)
            
            lat = mandal['lat'] + lat_offset
            lng = mandal['lng'] + lng_offset
            
            # Base features
            gwl_pre = perturb(mandal['gwl_pre_monsoon_mbgl'])
            gwl_post = perturb(mandal['gwl_post_monsoon_mbgl'])
            gwl_fluctuation = gwl_pre - gwl_post + random.uniform(-0.5, 0.5)
            extraction = perturb(mandal['extraction_stage_pct'])
            w_zone = perturb(mandal['weathered_zone_thickness_m'])
            depth_bedrock = perturb(mandal['depth_to_bedrock_m'])
            lin_density = perturb(mandal['lineament_density_per_km2'])
            avg_depth = perturb(mandal['avg_borewell_depth_ft'])
            est_success = mandal['estimated_success_rate']
            
            # Derived features
            rainfall_15yr = random.uniform(750, 950)
            rainfall_monsoon = rainfall_15yr * random.uniform(0.75, 0.85)
            et0_annual = random.uniform(1400, 1600)
            
            land_cover_class = random.choices(
                ['Built Area', 'Trees', 'Crops', 'Bare Ground', 'Rangeland'],
                weights=[30, 20, 25, 10, 15],
                k=1
            )[0]
            
            if land_cover_class == 'Built Area':
                impervious = 90
                factor = 0.05
            elif land_cover_class == 'Trees':
                impervious = 5
                factor = 0.25
            elif land_cover_class == 'Crops':
                impervious = 15
                factor = 0.15
            elif land_cover_class == 'Bare Ground':
                impervious = 5
                factor = 0.10
            else:
                impervious = 10
                factor = 0.15
                
            eff_recharge = max(0, rainfall_15yr - et0_annual) * factor
            # Sometimes eff_recharge is just based on rainfall since et0 is higher.
            # Let's adjust since rainfall < et0 mostly. 
            # If so, max(0, rainfall - et0) is always 0. The instruction says: "effective_recharge_mm: max(0, rainfall - et0) * factor based on land cover". I'll use it exactly as specified, but maybe it will be 0. Wait, rainfall = 800, et0 = 1500, rainfall - et0 < 0, so max is 0. Okay.
            
            heavy_days = random.uniform(5, 15)
            moderate_days = random.uniform(20, 40)
            rain_intensity_ratio = heavy_days / (heavy_days + moderate_days)
            drought_years = random.uniform(2, 5)
            
            elevation = random.uniform(450, 600)
            slope = random.uniform(0.5, 8.0)
            slope_rad = max(0.01, slope * math.pi / 180)
            twi = math.log(1000 / math.tan(slope_rad))
            
            # Nearest waterbody
            dist_water = min([haversine(lat, lng, wb['lat'], wb['lng']) for wb in waterbodies])
            
            clay = random.uniform(25, 45)
            sand = random.uniform(30, 55)
            soil_ph = random.uniform(6.5, 8.0)
            
            planned_depth = random.uniform(200, 1000)
            planned_month = random.randint(1, 12)
            
            success_prob = est_success
            if lin_density > 2.5: success_prob += 0.12
            if dist_water < 500: success_prob += 0.08
            if land_cover_class == 'Built Area': success_prob -= 0.15
            if eff_recharge > 300: success_prob += 0.10
            if extraction > 150: success_prob -= 0.08
            if avg_depth - 200 <= planned_depth <= avg_depth + 200: success_prob += 0.05
            
            success_prob = max(0.05, min(0.95, success_prob))
            success = 1 if random.random() < success_prob else 0
            
            row = {
                'rainfall_15yr_annual_avg_mm': rainfall_15yr,
                'rainfall_monsoon_avg_mm': rainfall_monsoon,
                'et0_annual_avg_mm': et0_annual,
                'effective_recharge_mm': eff_recharge,
                'heavy_rain_days_per_yr': heavy_days,
                'moderate_rain_days_per_yr': moderate_days,
                'rainfall_intensity_ratio': rain_intensity_ratio,
                'drought_years_in_15': drought_years,
                'elevation_m': elevation,
                'slope_degrees': slope,
                'twi': twi,
                'distance_to_waterbody_m': dist_water,
                'impervious_surface_pct': impervious,
                'gwl_pre_monsoon_mbgl': gwl_pre,
                'gwl_post_monsoon_mbgl': gwl_post,
                'gwl_seasonal_fluctuation_m': gwl_fluctuation,
                'extraction_stage_pct': extraction,
                'weathered_zone_thickness_m': w_zone,
                'depth_to_bedrock_m': depth_bedrock,
                'lineament_density_per_km2': lin_density,
                'clay_pct': clay,
                'sand_pct': sand,
                'soil_ph': soil_ph,
                'planned_depth_ft': planned_depth,
                'planned_month': planned_month,
                'rock_type': mandal['rock_type'],
                'aquifer_type': mandal['aquifer_type'],
                'land_cover_class': land_cover_class,
                'borewell_success': success,
                'avg_borewell_depth_ft': avg_depth,
                'latitude': lat,
                'longitude': lng
            }
            data.append(row)
            
    out_dir = os.path.join(base_dir, 'processed')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'hyderabad_training_data.csv')
    
    with open(out_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerows(data)
        
    print("[OK] Generated hyderabad training data at", out_file)

if __name__ == '__main__':
    main()
