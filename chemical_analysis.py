import enum
import pandas as pd
import numpy as np
from ml_predictor import MLPredictor

class ChemicalAnalysisResult:
    def __init__(self, health_score, status, details, recommendations):
        self.health_score = health_score
        self.status = status
        self.details = details
        self.recommendations = recommendations

    def to_dict(self):
        return {
            'health_score': self.health_score,
            'status': self.status,
            'details': self.details,
            'recommendations': self.recommendations
        }

class ChemicalAnalyzer:
    LEAF_COLORS = ['Green', 'Yellow', 'Brown']
    MOISTURE_LEVELS = ['Low', 'Medium', 'High']
    CHLOROPHYLL_CONTENTS = ['Low', 'Normal', 'High']
    NITROGEN_LEVELS = ['Low', 'Adequate', 'High']

    def __init__(self):
        self.data = None
        self.ml_predictor = MLPredictor()
        
    def load_data(self, file_path):
        """Load chemical analysis data"""
        if isinstance(file_path, str):
            self.data = pd.read_csv(file_path)
        else:
            self.data = pd.read_csv(file_path)
    
    def analyze(self, leaf_color, soil_ph, moisture_level, chlorophyll_content, nitrogen_level):
        """Analyze chemical parameters using ML model"""
        try:
            # Prepare input data
            input_data = {
                'leaf_color': leaf_color,
                'soil_ph': float(soil_ph),
                'moisture_level': moisture_level,
                'chlorophyll_content': chlorophyll_content,
                'nitrogen_level': nitrogen_level
            }
            
            # Get ML prediction
            ml_result = self.ml_predictor.predict_chemical(input_data)
            
            # Calculate chemical compounds based on ML prediction
            health_score = ml_result['prediction']
            feature_importance = ml_result['feature_importance']
            
            # Generate chemical compounds based on health score and feature importance
            compounds = {
                'sugars': self._calculate_compound_value(health_score, feature_importance.get('soil_ph', 0), 2.5, 4.0),
                'malic_acid': self._calculate_compound_value(health_score, feature_importance.get('moisture_level', 0), 0.8, 1.5),
                'vitamin_c': self._calculate_compound_value(health_score, feature_importance.get('chlorophyll_content', 0), 0.4, 0.8),
                'chlorophyll': self._calculate_compound_value(health_score, feature_importance.get('leaf_color', 0), 2.0, 3.0),
                'anthocyanins': self._calculate_compound_value(health_score, feature_importance.get('nitrogen_level', 0), 3.5, 4.5),
                'pectin': self._calculate_compound_value(health_score, feature_importance.get('soil_ph', 0), 1.2, 1.8),
                'actinidin': self._calculate_compound_value(health_score, feature_importance.get('nitrogen_level', 0), 0.8, 1.2),
                'fiber': self._calculate_compound_value(health_score, feature_importance.get('moisture_level', 0), 1.8, 2.4)
            }
            
            # Generate recommendations based on feature importance
            recommendations = self._generate_recommendations(compounds, feature_importance, input_data)
            
            return {
                'chemical_compounds': compounds,
                'environmental_factors': {
                    'pH_level': {
                        'value': soil_ph,
                        'status': 'Optimal' if 6.0 <= soil_ph <= 7.0 else 'Sub-optimal'
                    },
                    'moisture_level': {
                        'value': moisture_level,
                        'status': 'Optimal' if moisture_level == 'Medium' else 'Sub-optimal'
                    },
                    'chlorophyll_content': {
                        'value': chlorophyll_content,
                        'status': 'Optimal' if chlorophyll_content == 'Normal' else 'Sub-optimal'
                    },
                    'nitrogen_level': {
                        'value': nitrogen_level,
                        'status': 'Optimal' if nitrogen_level == 'Adequate' else 'Sub-optimal'
                    }
                },
                'overall_assessment': {
                    'health_score': health_score,
                    'status': self._get_health_status(health_score)
                },
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {"errors": str(e)}
    
    def _calculate_compound_value(self, health_score, feature_importance, min_val, max_val):
        """Calculate compound value based on health score and feature importance"""
        base_value = min_val + (max_val - min_val) * (health_score / 100)
        # Adjust based on feature importance
        adjustment = (max_val - min_val) * 0.1 * feature_importance
        return round(base_value + adjustment, 2)
    
    def _get_health_status(self, health_score):
        """Get health status based on score"""
        if health_score >= 90:
            return "Excellent - All chemical parameters are within optimal ranges"
        elif health_score >= 75:
            return "Good - Most chemical parameters are within optimal ranges"
        elif health_score >= 60:
            return "Fair - Some chemical parameters need attention"
        else:
            return "Poor - Multiple chemical parameters need attention"
    
    def _generate_recommendations(self, compounds, feature_importance, input_data):
        """Generate recommendations based on compound values and feature importance"""
        recommendations = []
        
        # Check each compound against optimal ranges
        optimal_ranges = {
            'sugars': (2.5, 4.0),
            'malic_acid': (0.8, 1.5),
            'vitamin_c': (0.4, 0.8),
            'chlorophyll': (2.0, 3.0),
            'anthocyanins': (3.5, 4.5),
            'pectin': (1.2, 1.8),
            'actinidin': (0.8, 1.2),
            'fiber': (1.8, 2.4)
        }
        
        for compound, (min_val, max_val) in optimal_ranges.items():
            value = compounds[compound]
            if value < min_val:
                recommendations.append({
                    'compound': compound,
                    'issue': f"{compound.replace('_', ' ').title()} levels are low",
                    'recommendation': self._get_recommendation(compound, 'low', input_data)
                })
            elif value > max_val:
                recommendations.append({
                    'compound': compound,
                    'issue': f"{compound.replace('_', ' ').title()} levels are high",
                    'recommendation': self._get_recommendation(compound, 'high', input_data)
                })
        
        return recommendations
    
    def _get_recommendation(self, compound, level, input_data):
        """Get specific recommendation based on compound and level"""
        recommendations = {
            'sugars': {
                'low': 'Consider adjusting fertilization and irrigation.',
                'high': 'Reduce sugar application and monitor water levels.'
            },
            'malic_acid': {
                'low': 'Review fruit maturity and harvest timing.',
                'high': 'Check fruit development stage and storage conditions.'
            },
            'vitamin_c': {
                'low': 'Check sunlight exposure and nutrient balance.',
                'high': 'Monitor fruit development and storage conditions.'
            },
            'chlorophyll': {
                'low': 'Review leaf health and nutrient uptake.',
                'high': 'Check for excessive nitrogen application.'
            },
            'anthocyanins': {
                'low': 'Check light exposure and temperature conditions.',
                'high': 'Monitor fruit development and storage conditions.'
            },
            'pectin': {
                'low': 'Review fruit development stage and harvest timing.',
                'high': 'Check fruit maturity and storage conditions.'
            },
            'actinidin': {
                'low': 'Check fruit ripeness and storage conditions.',
                'high': 'Monitor fruit development and storage conditions.'
            },
            'fiber': {
                'low': 'Review fruit development and harvest timing.',
                'high': 'Check fruit maturity and storage conditions.'
            }
        }
        
        return recommendations.get(compound, {}).get(level, 'Monitor and adjust growing conditions.') 