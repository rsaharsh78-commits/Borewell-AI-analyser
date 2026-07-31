from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class DepthRange(BaseModel):
    min: int
    max: int

class Factor(BaseModel):
    name: str
    impact: str
    value: str
    contribution: str
    score: float = 0.0  # Numeric 0-1 impact for frontend bar visualization

class LocationInfo(BaseModel):
    mandal: str
    district: str
    state: str
    rock_type: str
    aquifer_type: str

class NearbyStats(BaseModel):
    avg_depth_ft: int
    estimated_success_rate: float
    extraction_stage_pct: float

class SoilProperties(BaseModel):
    clay_pct: float
    sand_pct: float
    silt_pct: float
    ph: float
    organic_carbon: float
    bulk_density: float

class GeologyInfo(BaseModel):
    lineament_density: float
    weathered_zone_thickness_m: float
    depth_to_bedrock_m: float
    dominant_trend: str
    fracture_depth_range: DepthRange

class HydrologySummary(BaseModel):
    rainfall_15yr_avg_mm: float
    monsoon_avg_mm: float
    et0_avg_mm: float
    effective_recharge_mm: float
    heavy_rain_days: int
    drought_years: int

class TerrainInfo(BaseModel):
    elevation_m: float
    slope_degrees: float
    twi: float
    distance_to_waterbody_m: float
    land_cover_class: str
    impervious_pct: float

class DetailedExplanation(BaseModel):
    factor: str
    value: str
    unit: str = ''
    assessment: str  # 'advantage', 'risk', 'neutral'
    narrative: str
    source: str
    recommendation: str = ''

class DataSourceInfo(BaseModel):
    field: str
    source: str
    status: str  # 'live', 'fallback', 'static'

class MonthlyRisk(BaseModel):
    month: int
    month_name: str
    risk_score: int
    risk_category: str
    explanation: str
    risks: List[str]
    advantages: List[str]
    recommendation: str

class PredictResponse(BaseModel):
    risk_score: int
    risk_category: str
    confidence: float
    recommended_depth_ft: DepthRange
    factors: List[Factor]
    advisory: str
    advisory_hi: str
    location_info: LocationInfo
    nearby_stats: NearbyStats
    soil_properties: SoilProperties
    geology_info: GeologyInfo
    hydrology_summary: HydrologySummary
    terrain_info: TerrainInfo
    detailed_explanations: List[DetailedExplanation] = []
    data_sources: List[DataSourceInfo] = []
    monthly_risk: Optional[MonthlyRisk] = None

class GeocodeResponse(BaseModel):
    lat: float
    lng: float
    display_name: str
    mandal: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None

class MonthlyRainfall(BaseModel):
    year: int
    month: int
    precipitation: float
    et0: float

class RainfallHistoryResponse(BaseModel):
    years: List[int]
    annual_rainfall: List[float]
    annual_et0: List[float]
    effective_recharge: List[float]
    monthly_data: List[MonthlyRainfall]

class HeatmapDataPoint(BaseModel):
    mandal: Optional[str] = None
    district: Optional[str] = None
    lat: float
    lng: float
    risk_score: float
    extraction_stage: float
    gwl_pre_monsoon: float

class MonthlyRiskResponse(BaseModel):
    location: str
    extraction_stage_pct: float
    months: List[MonthlyRisk]
    best_month: int
    best_month_name: str
    worst_month: int
    worst_month_name: str

class GridPointScore(BaseModel):
    lat: float
    lng: float
    total_score: float
    rank: int
    factor_scores: dict
    reasoning: str
    mandal: str = ''
    rock_type: str = ''
    extraction_category: str = ''

class GridAnalysisResponse(BaseModel):
    center_lat: float
    center_lng: float
    radius_km: float
    total_points: int
    best_location: GridPointScore
    top_3: List[GridPointScore]
    all_points: List[GridPointScore]
