import pandas as pd

class YieldAnalysisResult:
    def __init__(self, yield_score, rating, details, limiting_factors, recommendations):
        self.yield_score = yield_score
        self.rating = rating
        self.details = details
        self.limiting_factors = limiting_factors
        self.recommendations = recommendations

    def to_dict(self):
        return {
            'yield_score': self.yield_score,
            'rating': self.rating,
            'details': self.details,
            'limiting_factors': self.limiting_factors,
            'recommendations': self.recommendations
        }

class YieldAnalyzer:
    LEAF_COLORS = ['Green', 'Yellow', 'Brown']
    SOIL_MOISTURE = ['Dry', 'Moderate', 'Wet']

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

    def analyze(self, tree_age, flower_buds_count, leaf_color, soil_moisture, fertilizer_used):
        errors = []
        if not isinstance(tree_age, (int, float)) or tree_age < 0:
            errors.append('tree_age must be a non-negative number')
        if not isinstance(flower_buds_count, int) or flower_buds_count < 0:
            errors.append('flower_buds_count must be a non-negative integer')
        if leaf_color not in self.LEAF_COLORS:
            errors.append(f'Invalid leaf_color: {leaf_color}')
        if soil_moisture not in self.SOIL_MOISTURE:
            errors.append(f'Invalid soil_moisture: {soil_moisture}')
        if not isinstance(fertilizer_used, bool):
            errors.append('fertilizer_used must be a boolean')
        if errors:
            return {'errors': errors}

        # Scoring logic
        score = 1.0
        details = {}
        limiting_factors = []
        recommendations = []

        # Tree age
        if 3 <= tree_age <= 15:
            details['tree_age'] = {'status': 'Optimal', 'score': 1.0}
        elif tree_age < 3:
            details['tree_age'] = {'status': 'Young', 'score': 0.6}
            limiting_factors.append('Tree is too young for optimal yield.')
            recommendations.append('Focus on root development and proper pruning for young trees.')
            score *= 0.6
        else:
            details['tree_age'] = {'status': 'Aging', 'score': 0.8}
            limiting_factors.append('Tree is aging, yield may decline.')
            recommendations.append('Consider rejuvenation pruning or replacement for old trees.')
            score *= 0.8

        # Flower buds
        expected_buds = 100 if tree_age < 3 else 200 if tree_age < 10 else 150
        if flower_buds_count >= expected_buds:
            details['flower_buds_count'] = {'status': 'Good', 'score': 1.0}
        elif flower_buds_count >= expected_buds * 0.5:
            details['flower_buds_count'] = {'status': 'Moderate', 'score': 0.7}
            limiting_factors.append('Moderate flower bud count may limit yield.')
            recommendations.append('Improve nutrition and management to increase bud count.')
            score *= 0.7
        else:
            details['flower_buds_count'] = {'status': 'Low', 'score': 0.4}
            limiting_factors.append('Low flower bud count is a major limiting factor.')
            recommendations.append('Review pruning and ensure proper winter chilling.')
            score *= 0.4

        # Leaf color
        if leaf_color == 'Green':
            details['leaf_color'] = {'status': 'Healthy', 'score': 1.0}
        elif leaf_color == 'Yellow':
            details['leaf_color'] = {'status': 'Deficiency/Stress', 'score': 0.6}
            limiting_factors.append('Yellow leaves indicate stress or deficiency.')
            recommendations.append('Check for nutrient deficiency or water stress.')
            score *= 0.6
        else:
            details['leaf_color'] = {'status': 'Severe Stress/Disease', 'score': 0.3}
            limiting_factors.append('Brown leaves indicate severe stress or disease.')
            recommendations.append('Inspect for pests, diseases, or root issues.')
            score *= 0.3

        # Soil moisture
        if soil_moisture == 'Moderate':
            details['soil_moisture'] = {'status': 'Optimal', 'score': 1.0}
        else:
            details['soil_moisture'] = {'status': 'Sub-optimal', 'score': 0.7}
            limiting_factors.append('Soil moisture is not optimal.')
            recommendations.append('Adjust irrigation to maintain moderate soil moisture.')
            score *= 0.7

        # Fertilizer
        if fertilizer_used:
            details['fertilizer_used'] = {'status': 'Applied', 'score': 1.0}
        else:
            details['fertilizer_used'] = {'status': 'Not Applied', 'score': 0.5}
            limiting_factors.append('No fertilizer applied, possible nutrient deficiency.')
            recommendations.append('Apply balanced fertilizer as per recommendations.')
            score *= 0.5

        # Final yield score and rating
        yield_score = round(score * 100, 1)
        if yield_score >= 90:
            rating = 'Excellent'
        elif yield_score >= 70:
            rating = 'Good'
        elif yield_score >= 50:
            rating = 'Average'
        else:
            rating = 'Poor'

        if not recommendations:
            recommendations.append('Maintain current practices and monitor regularly.')

        return YieldAnalysisResult(
            yield_score=yield_score,
            rating=rating,
            details=details,
            limiting_factors=limiting_factors,
            recommendations=recommendations
        ).to_dict()