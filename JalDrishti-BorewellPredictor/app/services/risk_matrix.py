from typing import Dict, Any, List

class RiskMatrixAssessor:
    def assess_all_months(self, lat: float, lng: float, features_dict: dict, mandal: dict) -> dict:
        """Return 12-month risk assessment."""
        months = []
        best_month_idx = -1
        worst_month_idx = -1
        best_score = float('inf')
        worst_score = -float('inf')
        
        for month in range(1, 13):
            month_data = self._assess_month(month, features_dict, mandal)
            months.append(month_data)
            
            if month_data['risk_score'] < best_score:
                best_score = month_data['risk_score']
                best_month_idx = month - 1
                
            if month_data['risk_score'] > worst_score:
                worst_score = month_data['risk_score']
                worst_month_idx = month - 1
                
        extraction_pct = mandal.get("extraction_stage_pct", 100)
        
        return {
            "location": mandal.get("mandal_name", "Unknown"),
            "extraction_stage_pct": extraction_pct,
            "months": months,
            "best_month": months[best_month_idx]['month'],
            "best_month_name": months[best_month_idx]['month_name'],
            "worst_month": months[worst_month_idx]['month'],
            "worst_month_name": months[worst_month_idx]['month_name']
        }

    def _assess_month(self, month: int, features_dict: dict, mandal: dict) -> dict:
        """Assess drilling risk for a specific month."""
        month_names = ["January", "February", "March", "April", "May", "June", 
                       "July", "August", "September", "October", "November", "December"]
        month_name = month_names[month - 1]
        
        base_risks = {
            1: 25, 2: 25, 3: 35, 4: 45, 5: 55, 6: 50,
            7: 75, 8: 80, 9: 60, 10: 15, 11: 20, 12: 25
        }
        
        risk_score = base_risks[month]
        
        extraction_pct = mandal.get("extraction_stage_pct", 100)
        recharge_mm = features_dict.get("effective_recharge_mm", 80)
        lineament = features_dict.get("lineament_density_per_km2", 2.0)
        wzt = mandal.get("weathered_zone_thickness_m", 15)
        mandal_name = mandal.get("mandal_name", "the area")
        
        risks = []
        advantages = []
        
        if extraction_pct > 150:
            risk_score += 10
            risks.append("Extremely over-exploited aquifer increases unreliability in all seasons.")
        elif extraction_pct < 70:
            risk_score -= 5
            advantages.append("Safe extraction levels make this area more forgiving.")
            
        if recharge_mm < 50 and month in [3, 4, 5]:
            risk_score += 5
            risks.append("Low effective recharge compounds the risk of drying out in pre-monsoon months.")
            
        if lineament > 2.5:
            risk_score -= 5
            advantages.append("High lineament density provides more aquifer options, buffering seasonal extremes.")
            
        risk_score = max(0, min(100, risk_score))
        
        if risk_score > 70:
            category = "High"
        elif risk_score > 40:
            category = "Moderate"
        else:
            category = "Low"
            
        if month == 8:
            explanation = (
                f"August represents the highest drilling risk in the Hyderabad region. The southwest monsoon is at peak intensity. "
                f"The weathered zone (currently {wzt}m thick at this location) becomes fully saturated, creating severe borehole "
                f"collapse risk in unconsolidated sections. The water table rises to its seasonal maximum, masking the true sustainable "
                f"yield of fracture aquifers — a borewell that appears to yield 3 inches during monsoon may drop significantly by April. "
                f"The current extraction stage of {extraction_pct}% in {mandal_name} means the aquifer is already stressed, "
                f"compounding monsoon-season measurement unreliability."
            )
        elif month == 10:
            explanation = (
                f"October is typically the optimal month for drilling. Post-monsoon groundwater levels are stabilized, providing "
                f"reliable yield measurements while avoiding the active recharge chaos of the monsoon. The weathered zone ({wzt}m thick) "
                f"is less prone to collapse compared to peak monsoon."
            )
        else:
            explanation = f"Risk in {month_name} is considered {category.lower()} with a score of {risk_score}/100. "
            if risk_score > 50:
                explanation += "Seasonal variations may significantly affect yield reliability."
            else:
                explanation += "Conditions are generally favorable for drilling and yield assessment."
                
        return {
            "month": month,
            "month_name": month_name,
            "risk_score": risk_score,
            "risk_category": category,
            "explanation": explanation,
            "risks": risks,
            "advantages": advantages,
            "recommendation": f"Proceed with {'caution' if risk_score > 50 else 'confidence'} in {month_name}."
        }

risk_matrix_instance = RiskMatrixAssessor()
