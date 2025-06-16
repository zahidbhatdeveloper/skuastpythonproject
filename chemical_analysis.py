import enum

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

    def analyze(self, leaf_color, soil_ph, moisture_level, chlorophyll_content, nitrogen_level):
        errors = []
        if leaf_color not in self.LEAF_COLORS:
            errors.append(f"Invalid leaf_color: {leaf_color}")
        if not (5.5 <= soil_ph <= 7.5):
            errors.append(f"soil_ph must be between 5.5 and 7.5")
        if moisture_level not in self.MOISTURE_LEVELS:
            errors.append(f"Invalid moisture_level: {moisture_level}")
        if chlorophyll_content not in self.CHLOROPHYLL_CONTENTS:
            errors.append(f"Invalid chlorophyll_content: {chlorophyll_content}")
        if nitrogen_level not in self.NITROGEN_LEVELS:
            errors.append(f"Invalid nitrogen_level: {nitrogen_level}")
        if errors:
            return {'errors': errors}

        # Scoring logic
        score = 1.0
        details = {}
        recommendations = []

        # Leaf color
        if leaf_color == 'Green':
            details['leaf_color'] = {'status': 'Healthy', 'score': 1.0}
        elif leaf_color == 'Yellow':
            details['leaf_color'] = {'status': 'Deficiency/Stress', 'score': 0.6}
            recommendations.append('Investigate possible nutrient deficiency or water stress causing yellow leaves.')
            score *= 0.6
        else:
            details['leaf_color'] = {'status': 'Severe Stress/Disease', 'score': 0.3}
            recommendations.append('Brown leaves indicate severe stress or disease. Inspect for pests or root issues.')
            score *= 0.3

        # Soil pH
        if 6.0 <= soil_ph <= 7.0:
            details['soil_ph'] = {'status': 'Optimal', 'score': 1.0}
        else:
            details['soil_ph'] = {'status': 'Sub-optimal', 'score': 0.7}
            recommendations.append('Adjust soil pH to 6.0-7.0 for optimal nutrient uptake.')
            score *= 0.7

        # Moisture
        if moisture_level == 'Medium':
            details['moisture_level'] = {'status': 'Optimal', 'score': 1.0}
        else:
            details['moisture_level'] = {'status': 'Sub-optimal', 'score': 0.7}
            recommendations.append('Adjust irrigation to maintain medium soil moisture.')
            score *= 0.7

        # Chlorophyll
        if chlorophyll_content == 'Normal' or chlorophyll_content == 'High':
            details['chlorophyll_content'] = {'status': 'Good', 'score': 1.0}
        else:
            details['chlorophyll_content'] = {'status': 'Low', 'score': 0.6}
            recommendations.append('Low chlorophyll content: consider nitrogen fertilization and check sunlight exposure.')
            score *= 0.6

        # Nitrogen
        if nitrogen_level == 'Adequate' or nitrogen_level == 'High':
            details['nitrogen_level'] = {'status': 'Sufficient', 'score': 1.0}
        else:
            details['nitrogen_level'] = {'status': 'Low', 'score': 0.6}
            recommendations.append('Apply nitrogen-rich fertilizer as per recommendations.')
            score *= 0.6

        # Final health score and status
        health_score = round(score * 100, 1)
        if health_score >= 90:
            status = 'Excellent'
        elif health_score >= 70:
            status = 'Good'
        elif health_score >= 50:
            status = 'Fair'
        else:
            status = 'Poor'

        if not recommendations:
            recommendations.append('Maintain current practices and monitor regularly.')

        return ChemicalAnalysisResult(
            health_score=health_score,
            status=status,
            details=details,
            recommendations=recommendations
        ).to_dict() 