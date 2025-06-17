import pandas as pd
import numpy as np
from ml_predictor import MLPredictor

class YieldAnalysisResult:
    def __init__(self, rating, expected_yield, details, limiting_factors, fruit_quality, recommendations):
        self.rating = rating
        self.expected_yield = expected_yield
        self.details = details
        self.limiting_factors = limiting_factors
        self.fruit_quality = fruit_quality
        self.recommendations = recommendations

    def to_dict(self):
        return {
            'rating': self.rating,
            'expected_yield': self.expected_yield,
            'details': self.details,
            'limiting_factors': self.limiting_factors,
            'fruit_quality_prediction': self.fruit_quality,
            'recommendations': self.recommendations
        }

class YieldAnalyzer:
    LEAF_COLORS = ['Green', 'Yellow', 'Brown']
    SOIL_MOISTURE = ['Dry', 'Moderate', 'Wet']

    def __init__(self):
        self.data = None
        self.ml_predictor = MLPredictor()
        self.data_dir = 'data'

    def load_data(self, file_path):
        """Load yield analysis data"""
        if isinstance(file_path, str):
            self.data = pd.read_csv(file_path)
        else:
            self.data = pd.read_csv(file_path)

    def analyze(self, tree_age, flower_buds_count, leaf_color, soil_moisture, fertilizer_used):
        """Analyze yield parameters using ML model"""
        try:
            # Prepare input data
            input_data = {
                'tree_age': float(tree_age),
                'flower_buds_count': int(flower_buds_count),
                'leaf_color': leaf_color,
                'soil_moisture': soil_moisture,
                'fertilizer_used': fertilizer_used
            }
            
            # Get ML prediction
            ml_result = self.ml_predictor.predict_yield(input_data)
            
            # Calculate yield metrics based on ML prediction
            expected_yield = ml_result['prediction']
            feature_importance = ml_result['feature_importance']
            
            # Calculate yield range based on feature importance
            yield_range = self._calculate_yield_range(expected_yield, feature_importance)
            
            # Generate quality predictions
            quality_prediction = self._predict_quality(input_data, feature_importance)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(input_data, feature_importance, expected_yield)
            
            return {
                'rating': self._get_yield_rating(expected_yield),
                'expected_yield': {
                    'minimum': round(yield_range['min'], 1),
                    'maximum': round(yield_range['max'], 1),
                    'expected': round(expected_yield, 1),
                    'unit': 'kg per tree'
                },
                'details': {
                    'tree_age': self._get_age_status(tree_age),
                    'flower_buds_count': self._get_buds_status(flower_buds_count),
                    'leaf_color': self._get_leaf_status(leaf_color),
                    'soil_moisture': self._get_moisture_status(soil_moisture),
                    'fertilizer_used': self._get_fertilizer_status(fertilizer_used)
                },
                'limiting_factors': self._identify_limiting_factors(input_data, feature_importance),
                'fruit_quality_prediction': quality_prediction,
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {"errors": str(e)}

    def _calculate_yield_range(self, expected_yield, feature_importance):
        """Calculate yield range based on expected yield and feature importance"""
        # Calculate uncertainty based on feature importance
        total_importance = sum(feature_importance.values())
        uncertainty = (1 - total_importance) * 0.2  # 20% maximum uncertainty
        
        return {
            'min': expected_yield * (1 - uncertainty),
            'max': expected_yield * (1 + uncertainty)
        }

    def _predict_quality(self, input_data, feature_importance):
        """Predict fruit quality based on input parameters and feature importance"""
        # Calculate quality scores for each aspect
        size_score = self._calculate_quality_score('size', input_data, feature_importance)
        color_score = self._calculate_quality_score('color', input_data, feature_importance)
        sweetness_score = self._calculate_quality_score('sweetness', input_data, feature_importance)
        firmness_score = self._calculate_quality_score('firmness', input_data, feature_importance)
        
        # Calculate overall quality
        overall_score = (size_score + color_score + sweetness_score + firmness_score) / 4
        
        return {
            'size': self._get_quality_rating(size_score),
            'color': self._get_quality_rating(color_score),
            'sweetness': self._get_quality_rating(sweetness_score),
            'firmness': self._get_quality_rating(firmness_score),
            'overall_quality': self._get_quality_rating(overall_score)
        }

    def _calculate_quality_score(self, aspect, input_data, feature_importance):
        """Calculate quality score for a specific aspect"""
        base_score = 7.0  # Base score on a 0-10 scale
        
        # Adjust based on input parameters
        if aspect == 'size':
            if input_data['soil_moisture'] == 'Moderate':
                base_score += 1
            if input_data['fertilizer_used']:
                base_score += 0.5
        elif aspect == 'color':
            if input_data['leaf_color'] == 'Green':
                base_score += 1
        elif aspect == 'sweetness':
            if input_data['soil_moisture'] == 'Moderate':
                base_score += 0.5
            if input_data['fertilizer_used']:
                base_score += 0.5
        elif aspect == 'firmness':
            if 3 <= input_data['tree_age'] <= 15:
                base_score += 0.5
            if input_data['fertilizer_used']:
                base_score += 0.5
        
        # Adjust based on feature importance
        importance_factor = sum(feature_importance.values()) / len(feature_importance)
        base_score *= (0.8 + 0.4 * importance_factor)
        
        return min(max(base_score, 0), 10)

    def _get_quality_rating(self, score):
        """Convert quality score to rating"""
        if score >= 8.5:
            return "Excellent"
        elif score >= 7.0:
            return "Good"
        elif score >= 5.0:
            return "Fair"
        else:
            return "Poor"

    def _get_yield_rating(self, expected_yield):
        """Get yield rating based on expected yield"""
        if expected_yield >= 20:
            return "Excellent"
        elif expected_yield >= 15:
            return "Good"
        elif expected_yield >= 10:
            return "Average"
        else:
            return "Poor"

    def _get_age_status(self, age):
        """Get tree age status"""
        if 3 <= age <= 15:
            return {"status": "Optimal", "rating": "Excellent"}
        elif age < 3:
            return {"status": "Young", "rating": "Average"}
        else:
            return {"status": "Aging", "rating": "Good"}

    def _get_buds_status(self, count):
        """Get flower buds status"""
        if count >= 150:
            return {"status": "High", "rating": "Excellent"}
        elif count >= 100:
            return {"status": "Moderate", "rating": "Good"}
        else:
            return {"status": "Low", "rating": "Average"}

    def _get_leaf_status(self, color):
        """Get leaf color status"""
        if color == "Green":
            return {"status": "Healthy", "rating": "Excellent"}
        elif color == "Yellow":
            return {"status": "Deficiency/Stress", "rating": "Average"}
        else:
            return {"status": "Unhealthy", "rating": "Poor"}

    def _get_moisture_status(self, moisture):
        """Get soil moisture status"""
        if moisture == "Moderate":
            return {"status": "Optimal", "rating": "Excellent"}
        elif moisture == "Wet":
            return {"status": "High", "rating": "Good"}
        else:
            return {"status": "Low", "rating": "Average"}

    def _get_fertilizer_status(self, used):
        """Get fertilizer status"""
        if used:
            return {"status": "Applied", "rating": "Excellent"}
        else:
            return {"status": "Not Applied", "rating": "Average"}

    def _identify_limiting_factors(self, input_data, feature_importance):
        """Identify limiting factors based on input data and feature importance"""
        limiting_factors = []
        
        # Check each parameter
        if input_data['tree_age'] < 3:
            limiting_factors.append("Young tree age may limit yield.")
        elif input_data['tree_age'] > 15:
            limiting_factors.append("Aging tree may affect yield.")
            
        if input_data['flower_buds_count'] < 100:
            limiting_factors.append("Low flower bud count may limit yield.")
            
        if input_data['leaf_color'] != "Green":
            limiting_factors.append("Leaf color indicates stress or deficiency.")
            
        if input_data['soil_moisture'] != "Moderate":
            limiting_factors.append("Soil moisture is not optimal.")
            
        if not input_data['fertilizer_used']:
            limiting_factors.append("No fertilizer application may limit yield.")
        
        return limiting_factors

    def _generate_recommendations(self, input_data, feature_importance, expected_yield):
        """Generate recommendations based on input data and feature importance"""
        recommendations = []
        
        # Tree age recommendations
        if input_data['tree_age'] < 3:
            recommendations.append("Provide additional care for young tree development.")
        elif input_data['tree_age'] > 15:
            recommendations.append("Consider rejuvenation pruning for aging tree.")
        
        # Flower buds recommendations
        if input_data['flower_buds_count'] < 100:
            recommendations.append("Improve nutrition and management to increase bud count.")
        
        # Leaf color recommendations
        if input_data['leaf_color'] != "Green":
            recommendations.append("Check for nutrient deficiency or water stress.")
        
        # Soil moisture recommendations
        if input_data['soil_moisture'] != "Moderate":
            recommendations.append("Adjust irrigation to maintain moderate soil moisture.")
        
        # Fertilizer recommendations
        if not input_data['fertilizer_used']:
            recommendations.append("Consider applying appropriate fertilizer.")
        
        return recommendations

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