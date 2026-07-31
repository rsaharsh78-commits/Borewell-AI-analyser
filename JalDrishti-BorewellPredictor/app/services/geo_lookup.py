"""
Geo-lookup service — mandal interpolation, heatmap grid generation.
"""
import json
import math
import os
from typing import List, Dict, Any

MANDALS_PATH = os.path.join("data", "hyderabad_mandals.json")


class GeoLookupService:
    def __init__(self):
        self.mandals_data = []
        try:
            if os.path.exists(MANDALS_PATH):
                with open(MANDALS_PATH, 'r', encoding='utf-8') as f:
                    self.mandals_data = json.load(f)
        except Exception as e:
            print(f"[ERROR] GeoLookupService init failed: {e}")

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1))
             * math.cos(math.radians(lat2))
             * math.sin(dlon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def find_nearest_mandal(self, lat: float, lng: float) -> Dict[str, Any]:
        if not self.mandals_data:
            return {}
        best, best_dist = None, float('inf')
        for m in self.mandals_data:
            m_lat, m_lng = m.get('lat'), m.get('lng')
            if m_lat is not None and m_lng is not None:
                dist = self._haversine(lat, lng, m_lat, m_lng)
                if dist < best_dist:
                    best_dist = dist
                    best = m
        return best or {}

    def idw_interpolate_gwl(self, lat: float, lng: float, p: int = 2) -> Dict[str, float]:
        if not self.mandals_data:
            return {"pre_monsoon": 0.0, "post_monsoon": 0.0}

        num_pre = num_post = den = 0.0

        for m in self.mandals_data:
            m_lat, m_lng = m.get('lat'), m.get('lng')
            if m_lat is None or m_lng is None:
                continue
            dist = self._haversine(lat, lng, m_lat, m_lng)
            if dist == 0:
                return {
                    "pre_monsoon": m.get('gwl_pre_monsoon_mbgl', 0.0),
                    "post_monsoon": m.get('gwl_post_monsoon_mbgl', 0.0),
                }
            w = 1.0 / (dist ** p)
            num_pre += w * m.get('gwl_pre_monsoon_mbgl', 0.0)
            num_post += w * m.get('gwl_post_monsoon_mbgl', 0.0)
            den += w

        if den == 0:
            return {"pre_monsoon": 0.0, "post_monsoon": 0.0}
        return {
            "pre_monsoon": num_pre / den,
            "post_monsoon": num_post / den,
        }

    def _idw_field(self, lat, lng, field, p=2):
        """IDW-interpolate an arbitrary numeric field across mandals."""
        num = den = 0.0
        for m in self.mandals_data:
            m_lat, m_lng = m.get('lat'), m.get('lng')
            if m_lat is None or m_lng is None:
                continue
            dist = self._haversine(lat, lng, m_lat, m_lng)
            if dist == 0:
                return m.get(field, 0.0)
            w = 1.0 / (dist ** p)
            num += w * m.get(field, 0.0)
            den += w
        return num / den if den else 0.0

    def get_heatmap_grid(self) -> List[Dict[str, Any]]:
        points: List[Dict[str, Any]] = []
        if not self.mandals_data:
            return points

        lats = [m['lat'] for m in self.mandals_data if 'lat' in m]
        lngs = [m['lng'] for m in self.mandals_data if 'lng' in m]
        lat_min, lat_max = min(lats), max(lats)
        lng_min, lng_max = min(lngs), max(lngs)

        steps = 14
        lat_step = (lat_max - lat_min) / steps
        lng_step = (lng_max - lng_min) / steps

        for i in range(steps + 1):
            for j in range(steps + 1):
                c_lat = lat_min + i * lat_step
                c_lng = lng_min + j * lng_step

                gwl = self.idw_interpolate_gwl(c_lat, c_lng)
                nearest = self.find_nearest_mandal(c_lat, c_lng)

                # Composite risk using extraction stage, GWL depth,
                # and lineament density (inverted — lower = riskier)
                ext = self._idw_field(c_lat, c_lng, 'extraction_stage_pct')
                lin = self._idw_field(c_lat, c_lng, 'lineament_density_per_km2')
                gwl_pre = gwl.get('pre_monsoon', 15.0)

                ext_norm = min(1.0, max(0, ext / 200.0))
                gwl_norm = min(1.0, max(0, gwl_pre / 30.0))
                lin_norm = 1.0 - min(1.0, max(0, lin / 4.0))

                risk = min(100.0, max(0.0,
                    ext_norm * 50 + gwl_norm * 30 + lin_norm * 20))

                points.append({
                    "mandal": nearest.get('mandal', 'Unknown'),
                    "district": nearest.get('district', 'Unknown'),
                    "lat": round(c_lat, 5),
                    "lng": round(c_lng, 5),
                    "risk_score": round(risk, 1),
                    "extraction_stage": round(ext, 1),
                    "gwl_pre_monsoon": round(gwl_pre, 1),
                })

        return points


geo_lookup_service = GeoLookupService()
