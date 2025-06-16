import pandas as pd

class YieldAnalyzer:
    def __init__(self):
        self.data = None
        self.data_dir = 'data'

    def load_data(self, file_path):
        """Load yield data from CSV file"""
        try:
            self.data = pd.read_csv(file_path)
            print(f"Yield data loaded successfully with {len(self.data)} records")
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    def analyze_tree_maturity(self, tree_age):
        status = "Mature"
        issues = []
        if tree_age < 3:
            status = "Young"
            issues.append("Tree is too young for optimal fruit production")
        elif tree_age > 15:
            status = "Aging"
            issues.append("Tree is entering senescence phase")
        return {
            "status": status,
            "issues": issues,
            "details": {
                "age": tree_age,
                "maturity_stage": "Early Growth" if tree_age < 3 else "Prime" if 3 <= tree_age <= 15 else "Late Stage",
                "yield_potential": "Low" if tree_age < 3 else "High" if 3 <= tree_age <= 15 else "Declining"
            }
        }

    def analyze_flowering_potential(self, flower_buds_count, tree_age):
        # Implementation of analyze_flowering_potential method
        pass

    def analyze_leaf_health(self, leaf_color):
        # Implementation of analyze_leaf_health method
        pass

    def analyze_soil_condition(self, soil_moisture):
        # Implementation of analyze_soil_condition method
        pass

    def analyze_nutrient_status(self, fertilizer_used):
        # Implementation of analyze_nutrient_status method
        pass

    def analyze_yield(self, tree_age, flower_buds_count, leaf_color, soil_moisture, fertilizer_used):
        # Input validation
        errors = []
        if not isinstance(tree_age, (int, float)) or tree_age < 0:
            errors.append("Tree age must be a positive number.")
        if not isinstance(flower_buds_count, int) or flower_buds_count < 0:
            errors.append("Flower buds count must be a positive integer.")
        if leaf_color not in ['Green', 'Yellow', 'Brown']:
            errors.append("Leaf color must be Green, Yellow, or Brown.")
        if soil_moisture not in ['Dry', 'Moderate', 'Wet']:
            errors.append("Soil moisture must be Dry, Moderate, or Wet.")
        if not isinstance(fertilizer_used, bool):
            errors.append("Fertilizer used must be true or false.")
        if errors:
            return {"errors": errors}

        # Yield potential calculation
        score = 100
        limiting_factors = []
        if tree_age < 3:
            score *= 0.5
            limiting_factors.append("Tree is too young for optimal fruit production.")
        elif tree_age > 15:
            score *= 0.7
            limiting_factors.append("Tree is aging; yield may decline.")
        expected_buds = 100 if tree_age < 3 else 200 if tree_age < 10 else 150
        bud_ratio = flower_buds_count / expected_buds
        if bud_ratio < 0.5:
            score *= 0.6
            limiting_factors.append("Low flower bud count; may indicate stress or poor nutrition.")
        elif bud_ratio > 1.5:
            score *= 0.9
            limiting_factors.append("Very high bud count; may reduce fruit size.")
        if leaf_color == "Yellow":
            score *= 0.7
            limiting_factors.append("Yellow leaves; possible nutrient deficiency or disease.")
        elif leaf_color == "Brown":
            score *= 0.4
            limiting_factors.append("Brown leaves; severe stress or disease.")
        if soil_moisture == "Dry":
            score *= 0.6
            limiting_factors.append("Dry soil; water stress may limit yield.")
        elif soil_moisture == "Wet":
            score *= 0.7
            limiting_factors.append("Wet soil; risk of root problems.")
        if not fertilizer_used:
            score *= 0.8
            limiting_factors.append("No fertilizer used; possible nutrient deficiency.")

        # Farmer-friendly rating
        def get_yield_rating(score):
            if score >= 80:
                return "Excellent"
            elif score >= 60:
                return "Good"
            elif score >= 40:
                return "Average"
            else:
                return "Poor"
        rating = get_yield_rating(score)
        if score >= 80:
            yield_category = "High"
        elif score >= 60:
            yield_category = "Medium"
        else:
            yield_category = "Low"

        # Fruit quality prediction
        quality = "Good"
        if leaf_color != "Green" or soil_moisture != "Moderate":
            quality = "Fair"
        if not fertilizer_used:
            quality = "Poor"

        # Recommendations
        recommendations = []
        if tree_age < 3:
            recommendations.append({
                "category": "Tree Management",
                "issue": "Young tree",
                "recommendation": "Focus on root development and formative pruning for young trees.",
                "priority": "High"
            })
        elif tree_age > 15:
            recommendations.append({
                "category": "Tree Management",
                "issue": "Aging tree",
                "recommendation": "Consider rejuvenation pruning or replacement for aging trees.",
                "priority": "Medium"
            })
        if flower_buds_count < expected_buds * 0.5:
            recommendations.append({
                "category": "Flowering",
                "issue": "Low flower bud count",
                "recommendation": "Review pruning and winter chilling to improve bud count.",
                "priority": "High"
            })
        if leaf_color != "Green":
            recommendations.append({
                "category": "Leaf Health",
                "issue": f"Abnormal leaf color ({leaf_color})",
                "recommendation": "Investigate nutrient status and check for disease.",
                "priority": "High" if leaf_color == "Brown" else "Medium"
            })
        if soil_moisture != "Moderate":
            recommendations.append({
                "category": "Irrigation",
                "issue": f"{soil_moisture} soil",
                "recommendation": "Adjust irrigation to maintain moderate soil moisture.",
                "priority": "High"
            })
        if not fertilizer_used:
            recommendations.append({
                "category": "Nutrient Management",
                "issue": "No fertilizer used",
                "recommendation": "Apply balanced fertilizer as per guidelines.",
                "priority": "High"
            })
        recommendations.append({
            "category": "Monitoring",
            "issue": "Regular yield assessment",
            "recommendation": "Conduct regular yield assessments and maintain records.",
            "priority": "Medium"
        })

        detailed_prediction = {
            "tree_maturity": self.analyze_tree_maturity(tree_age),
            "flowering_potential": self.analyze_flowering_potential(flower_buds_count, tree_age),
            "leaf_health": self.analyze_leaf_health(leaf_color),
            "soil_condition": self.analyze_soil_condition(soil_moisture),
            "nutrient_status": self.analyze_nutrient_status(fertilizer_used)
        }

        return {
            "yield_potential": {
                "rating": rating,
                "category": yield_category,
                "explanation": f"Based on input factors, the yield potential is {rating}."
            },
            "detailed_prediction": detailed_prediction,
            "fruit_quality_prediction": quality,
            "limiting_factors": limiting_factors,
            "recommendations": recommendations,
            "input_details": {
                "tree_age": tree_age,
                "flower_buds_count": flower_buds_count,
                "leaf_color": leaf_color,
                "soil_moisture": soil_moisture,
                "fertilizer_used": fertilizer_used
            }
        } #comenttttttt