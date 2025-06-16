import pandas as pd

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

        # Farmer-friendly rating logic
        def to_rating(val):
            if val >= 0.9:
                return 'Excellent'
            elif val >= 0.7:
                return 'Good'
            elif val >= 0.5:
                return 'Average'
            else:
                return 'Poor'

        # Parameter ratings
        details = {}
        limiting_factors = []
        recommendations = []
        score = 1.0

        # Tree age
        if 3 <= tree_age <= 15:
            details['tree_age'] = {'status': 'Optimal', 'rating': 'Excellent'}
            age_factor = 1.0
        elif tree_age < 3:
            details['tree_age'] = {'status': 'Young', 'rating': 'Average'}
            limiting_factors.append('Tree is too young for optimal yield.')
            recommendations.append('Focus on root development and proper pruning for young trees.')
            age_factor = 0.6
            score *= 0.6
        else:
            details['tree_age'] = {'status': 'Aging', 'rating': 'Good'}
            limiting_factors.append('Tree is aging, yield may decline.')
            recommendations.append('Consider rejuvenation pruning or replacement for old trees.')
            age_factor = 0.8
            score *= 0.8

        # Flower buds
        expected_buds = 100 if tree_age < 3 else 200 if tree_age < 10 else 150
        if flower_buds_count >= expected_buds:
            details['flower_buds_count'] = {'status': 'Good', 'rating': 'Excellent'}
            bud_factor = 1.0
        elif flower_buds_count >= expected_buds * 0.5:
            details['flower_buds_count'] = {'status': 'Moderate', 'rating': 'Good'}
            limiting_factors.append('Moderate flower bud count may limit yield.')
            recommendations.append('Improve nutrition and management to increase bud count.')
            bud_factor = 0.7
            score *= 0.7
        else:
            details['flower_buds_count'] = {'status': 'Low', 'rating': 'Poor'}
            limiting_factors.append('Low flower bud count is a major limiting factor.')
            recommendations.append('Review pruning and ensure proper winter chilling.')
            bud_factor = 0.4
            score *= 0.4

        # Leaf color
        if leaf_color == 'Green':
            details['leaf_color'] = {'status': 'Healthy', 'rating': 'Excellent'}
            leaf_factor = 1.0
        elif leaf_color == 'Yellow':
            details['leaf_color'] = {'status': 'Deficiency/Stress', 'rating': 'Average'}
            limiting_factors.append('Yellow leaves indicate stress or deficiency.')
            recommendations.append('Check for nutrient deficiency or water stress.')
            leaf_factor = 0.6
            score *= 0.6
        else:
            details['leaf_color'] = {'status': 'Severe Stress/Disease', 'rating': 'Poor'}
            limiting_factors.append('Brown leaves indicate severe stress or disease.')
            recommendations.append('Inspect for pests, diseases, or root issues.')
            leaf_factor = 0.3
            score *= 0.3

        # Soil moisture
        if soil_moisture == 'Moderate':
            details['soil_moisture'] = {'status': 'Optimal', 'rating': 'Excellent'}
            moisture_factor = 1.0
        else:
            details['soil_moisture'] = {'status': 'Sub-optimal', 'rating': 'Good'}
            limiting_factors.append('Soil moisture is not optimal.')
            recommendations.append('Adjust irrigation to maintain moderate soil moisture.')
            moisture_factor = 0.7
            score *= 0.7

        # Fertilizer
        if fertilizer_used:
            details['fertilizer_used'] = {'status': 'Applied', 'rating': 'Excellent'}
            fert_factor = 1.0
        else:
            details['fertilizer_used'] = {'status': 'Not Applied', 'rating': 'Average'}
            limiting_factors.append('No fertilizer applied, possible nutrient deficiency.')
            recommendations.append('Apply balanced fertilizer as per recommendations.')
            fert_factor = 0.5
            score *= 0.5

        # Final yield rating
        yield_rating = to_rating(score)

        # Expected yield range (farmer-friendly)
        base_yield = {'minimum': 15, 'maximum': 30, 'expected': 22.5, 'unit': 'kg per tree'}
        # Adjust for all factors
        base = 1.0
        for f in [age_factor, bud_factor, leaf_factor, moisture_factor, fert_factor]:
            base *= f
        base_yield = {
            'minimum': round(15 * base, 1),
            'maximum': round(30 * base, 1),
            'expected': round(22.5 * base, 1),
            'unit': 'kg per tree'
        }

        # Fruit quality prediction
        fruit_quality = {
            'size': 'Medium',
            'color': 'Good',
            'sweetness': 'Medium',
            'firmness': 'Good',
            'overall_quality': 'Good'
        }
        if leaf_color != 'Green' or soil_moisture != 'Moderate':
            fruit_quality['color'] = 'Fair'
            fruit_quality['overall_quality'] = 'Fair'
        if not fertilizer_used:
            fruit_quality['size'] = 'Small'
            fruit_quality['sweetness'] = 'Low'
            fruit_quality['overall_quality'] = 'Poor'

        if not recommendations:
            recommendations.append('Maintain current practices and monitor regularly.')

        return YieldAnalysisResult(
            rating=yield_rating,
            expected_yield=base_yield,
            details=details,
            limiting_factors=limiting_factors,
            fruit_quality=fruit_quality,
            recommendations=recommendations
        ).to_dict()
        #comment