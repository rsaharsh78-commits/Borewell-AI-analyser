import math
from typing import List, Tuple
from app.services.feature_engine import feature_engine_instance
from app.schemas.response import GridPointScore

class GridAnalyzer:
    def generate_grid(self, center_lat: float, center_lng: float, radius_km: float, grid_size: int = 5) -> List[Tuple[float, float]]:
        """Generate a grid of test points within a circle."""
        points = []
        # Approximate km per degree
        lat_degree_km = 111.0
        lng_degree_km = 111.0 * math.cos(math.radians(center_lat))
        
        lat_step = (radius_km / lat_degree_km) * 2 / max(1, grid_size - 1)
        lng_step = (radius_km / lng_degree_km) * 2 / max(1, grid_size - 1)
        
        start_lat = center_lat - (radius_km / lat_degree_km)
        start_lng = center_lng - (radius_km / lng_degree_km)
        
        for i in range(grid_size):
            for j in range(grid_size):
                lat = start_lat + i * lat_step
                lng = start_lng + j * lng_step
                
                # Check if point is within circle
                from app.services.feature_engine import _haversine_m
                dist_m = _haversine_m(center_lat, center_lng, lat, lng)
                if dist_m <= radius_km * 1000:
                    points.append((lat, lng))
                    
        return points

    def analyze_grid(self, center_lat: float, center_lng: float, radius_km: float = 2.0):
        """Analyze all grid points and return scored results."""
        points = self.generate_grid(center_lat, center_lng, radius_km, grid_size=5)
        
        scored_points = []
        for lat, lng in points:
            features_dict, mandal, zone, rain_extras = feature_engine_instance.build_features(lat, lng, 500, 1)
            score_data = self._score_point(features_dict, mandal, zone)
            
            rock_type = mandal.get("rock_type", "Unknown")
            extraction_pct = mandal.get("extraction_stage_pct", 100)
            if extraction_pct > 150:
                cat = "Critical"
            elif extraction_pct > 100:
                cat = "Over-exploited"
            elif extraction_pct > 70:
                cat = "Semi-critical"
            else:
                cat = "Safe"
                
            scored_points.append({
                "lat": lat,
                "lng": lng,
                "total_score": score_data["total_score"],
                "factor_scores": score_data["factor_scores"],
                "reasoning": score_data["reasoning"],
                "mandal": mandal.get("mandal", "Unknown"),
                "rock_type": rock_type,
                "extraction_category": cat
            })
            
        scored_points.sort(key=lambda x: x["total_score"], reverse=True)
        
        results = []
        for idx, pt in enumerate(scored_points):
            pt["rank"] = idx + 1
            results.append(GridPointScore(**pt))
            
        best = results[0] if results else None
        top_3 = results[:3]
        
        return {
            "center_lat": center_lat,
            "center_lng": center_lng,
            "radius_km": radius_km,
            "total_points": len(results),
            "best_location": best,
            "top_3": top_3,
            "all_points": results
        }

    def _score_point(self, features_dict: dict, mandal: dict, zone: dict) -> dict:
        """Calculate suitability score (0-100) with per-factor breakdown."""
        # Extraction Stage: 25% weight
        extraction_pct = features_dict.get("extraction_stage_pct", 100)
        ext_score = max(0, (200 - extraction_pct) / 200 * 100)
        
        # Slope: 20% weight
        slope_degrees = features_dict.get("slope_degrees", 2.0)
        slope_score = max(0, (10 - slope_degrees) / 10 * 100)
        
        # Soil Permeability (Sand): 15% weight
        sand_pct = features_dict.get("sand_pct", 40.0)
        sand_score = min(100, sand_pct * 1.5)
        
        # Water Proximity: 15% weight
        distance_m = features_dict.get("distance_to_waterbody_m", 1000.0)
        dist_score = max(0, (5000 - distance_m) / 5000 * 100)
        
        # Lineament Density: 15% weight
        lineament_density = features_dict.get("lineament_density_per_km2", 2.0)
        lin_score = min(100, lineament_density * 25)
        
        # Effective Recharge: 10% weight
        recharge_mm = features_dict.get("effective_recharge_mm", 50.0)
        rech_score = min(100, recharge_mm * 0.5)
        
        total_score = (
            ext_score * 0.25 +
            slope_score * 0.20 +
            sand_score * 0.15 +
            dist_score * 0.15 +
            lin_score * 0.15 +
            rech_score * 0.10
        )
        
        factor_scores = {
            "Extraction Stage": {
                "score": round(ext_score, 1),
                "weight": 25,
                "weighted_score": round(ext_score * 0.25, 1),
                "raw_value": extraction_pct,
                "explanation": "Lower extraction stage indicates higher sustainability."
            },
            "Slope": {
                "score": round(slope_score, 1),
                "weight": 20,
                "weighted_score": round(slope_score * 0.20, 1),
                "raw_value": slope_degrees,
                "explanation": "Flatter terrain favors groundwater recharge."
            },
            "Soil Permeability": {
                "score": round(sand_score, 1),
                "weight": 15,
                "weighted_score": round(sand_score * 0.15, 1),
                "raw_value": sand_pct,
                "explanation": "Higher sand content allows better infiltration."
            },
            "Water Proximity": {
                "score": round(dist_score, 1),
                "weight": 15,
                "weighted_score": round(dist_score * 0.15, 1),
                "raw_value": distance_m,
                "explanation": "Closer to surface water typically means better recharge."
            },
            "Lineament Density": {
                "score": round(lin_score, 1),
                "weight": 15,
                "weighted_score": round(lin_score * 0.15, 1),
                "raw_value": lineament_density,
                "explanation": "Higher fracture density indicates better potential yields."
            },
            "Effective Recharge": {
                "score": round(rech_score, 1),
                "weight": 10,
                "weighted_score": round(rech_score * 0.10, 1),
                "raw_value": recharge_mm,
                "explanation": "Higher net recharge implies better aquifer replenishment."
            }
        }
        
        if extraction_pct > 150:
            cat = "Critical"
        elif extraction_pct > 100:
            cat = "Over-exploited"
        elif extraction_pct > 70:
            cat = "Semi-critical"
        else:
            cat = "Safe"

        reasoning = (
            f"Score {round(total_score, 1)}/100. Extraction: {extraction_pct}% ({cat}). "
            f"Terrain: {slope_degrees:.1f}° slope ({('flat, good' if slope_degrees < 3 else 'moderate' if slope_degrees < 5 else 'steep, poor')} for recharge). "
            f"Soil: {sand_pct:.0f}% sand ({('permeable' if sand_pct > 50 else 'moderate' if sand_pct > 30 else 'low permeability')}). "
            f"Fractures: {lineament_density:.1f}/km² ({('excellent' if lineament_density > 2.5 else 'fair' if lineament_density > 1.5 else 'poor')})."
        )
        
        return {
            "total_score": round(total_score, 1),
            "factor_scores": factor_scores,
            "reasoning": reasoning
        }

grid_analyzer_instance = GridAnalyzer()
