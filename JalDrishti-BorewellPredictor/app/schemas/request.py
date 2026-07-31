from pydantic import BaseModel
from typing import Optional

class PredictRequest(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: Optional[str] = None
    planned_depth_ft: Optional[int] = 500
    planned_month: Optional[int] = None

class GeocodeRequest(BaseModel):
    q: str

class GridAnalysisRequest(BaseModel):
    latitude: float
    longitude: float
    radius_km: float = 2.0
