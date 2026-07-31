"""
Feature configuration for JalDrishti v2 - Hyderabad-focused borewell predictor.
26-feature schema reflecting real hydrogeological data.
"""

NUMERIC_FEATURES = [
    # Hydrology (8)
    'rainfall_15yr_annual_avg_mm',
    'rainfall_monsoon_avg_mm',
    'et0_annual_avg_mm',
    'effective_recharge_mm',
    'heavy_rain_days_per_yr',
    'moderate_rain_days_per_yr',
    'rainfall_intensity_ratio',
    'drought_years_in_15',
    # Topography (4)
    'elevation_m',
    'slope_degrees',
    'twi',
    'distance_to_waterbody_m',
    # Urban (1)
    'impervious_surface_pct',
    # CGWB Groundwater (7)
    'gwl_pre_monsoon_mbgl',
    'gwl_post_monsoon_mbgl',
    'gwl_seasonal_fluctuation_m',
    'extraction_stage_pct',
    'weathered_zone_thickness_m',
    'depth_to_bedrock_m',
    'lineament_density_per_km2',
    # Soil (4) - surface, less weight
    'clay_pct',
    'sand_pct',
    'soil_ph',
    # Input (2)
    'planned_depth_ft',
    'planned_month',
]

CATEGORICAL_FEATURES = [
    'rock_type',
    'aquifer_type',
    'land_cover_class',
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Land cover class mapping (Esri 10m)
LAND_COVER_CLASSES = {
    1: 'Water', 2: 'Trees', 4: 'Flooded Vegetation',
    5: 'Crops', 7: 'Built Area', 8: 'Bare Ground',
    9: 'Snow/Ice', 10: 'Clouds', 11: 'Rangeland'
}

IMPERVIOUS_PCT_MAP = {
    'Water': 0, 'Trees': 5, 'Flooded Vegetation': 10,
    'Crops': 15, 'Built Area': 90, 'Bare Ground': 5,
    'Snow/Ice': 0, 'Clouds': 0, 'Rangeland': 10
}

ROCK_TYPES = ['Granite-Gneiss', 'Deccan-Basalt', 'Sedimentary']
AQUIFER_TYPES = ['Weathered', 'Fractured']
