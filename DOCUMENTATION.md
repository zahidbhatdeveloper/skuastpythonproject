# SKUAST Tree Analysis System Documentation

## Overview
The SKUAST Tree Analysis System provides two main types of analysis:
1. Chemical Analysis - Evaluates tree health based on chemical parameters
2. Yield Analysis - Predicts and evaluates potential fruit yield

## Chemical Analysis

### Input Parameters
The chemical analysis accepts the following parameters:
- `Leaf_Color`: String - Options: "Green", "Yellow", "Brown"
- `Soil_pH`: Float - Range: 5.5 to 7.5
- `Moisture_Level`: String - Options: "Low", "Medium", "High"
- `Chlorophyll_Content`: String - Options: "Low", "Normal", "High"
- `Nitrogen_Level`: String - Options: "Low", "Adequate", "High"

### Analysis Process
1. **Parameter Validation**
   - Validates all input parameters
   - Checks for required values and valid ranges

2. **Chemical Compound Analysis**
   The system analyzes several key compounds:
   - Sugars (Optimal range: 2.5 - 4.0)
   - Malic Acid (Optimal range: 0.8 - 1.5)
   - Vitamin C (Optimal range: 0.4 - 0.8)
   - Chlorophyll (Optimal range: 2.0 - 3.0)
   - Anthocyanins (Optimal range: 3.5 - 4.5)
   - Pectin (Optimal range: 1.2 - 1.8)
   - Actinidin (Optimal range: 0.8 - 1.2)
   - Fiber (Optimal range: 1.8 - 2.4)

3. **Environmental Factors Analysis**
   - pH Level (Optimal range: 6.0 - 7.0)
   - Tree Age Assessment
   - Soil Type Evaluation
   - Fruit Stage Analysis
   - Location Impact

4. **Output Structure**
   ```json
   {
     "chemical_compounds": {
       "compound_name": {
         "mean_concentration": float,
         "range": "min - max",
         "std_dev": float,
         "cv_percent": float,
         "optimal_range": "min - max",
         "status": "Optimal/Sub-optimal"
       }
     },
     "environmental_factors": {
       "pH_level": {...},
       "tree_age": {...},
       "soil_type": string,
       "fruit_stage": string,
       "location": string
     },
     "overall_assessment": {
       "health_score": float,
       "status": string
     },
     "recommendations": [
       {
         "compound": string,
         "issue": string,
         "recommendation": string
       }
     ]
   }
   ```

## Yield Analysis

### Input Parameters
The yield analysis accepts the following parameters:
- `Tree_Age`: Float - Age in years
- `Flower_Buds_Count`: Integer - Number of flowers/fruit buds
- `Leaf_Color`: String - Options: "Green", "Yellow", "Brown"
- `Soil_Moisture`: String - Options: "Dry", "Moderate", "Wet"
- `Fertilizer_Used`: Boolean - Yes/No

### Analysis Process
1. **Parameter Validation**
   - Validates all input parameters
   - Checks for required values and valid ranges

2. **Yield Factors Analysis**
   - **Tree Age Assessment**
     - Optimal: 3-15 years
     - Young: < 3 years
     - Aging: > 15 years

   - **Flower Buds Evaluation**
     - Expected buds based on tree age
     - Quality assessment of bud development

   - **Leaf Health Analysis**
     - Color indicates nutrient status
     - Impact on photosynthesis and yield

   - **Soil Moisture Assessment**
     - Optimal: Moderate
     - Impact on fruit development

   - **Fertilization Status**
     - Impact on nutrient availability
     - Effect on yield potential

3. **Yield Prediction**
   - Calculates expected yield range
   - Predicts fruit quality parameters
   - Identifies limiting factors

4. **Output Structure**
   ```json
   {
     "rating": "Excellent/Good/Average/Poor",
     "expected_yield": {
       "minimum": float,
       "maximum": float,
       "expected": float,
       "unit": "kg per tree"
     },
     "details": {
       "tree_age": {"status": string, "rating": string},
       "flower_buds_count": {"status": string, "rating": string},
       "leaf_color": {"status": string, "rating": string},
       "soil_moisture": {"status": string, "rating": string},
       "fertilizer_used": {"status": string, "rating": string}
     },
     "limiting_factors": [string],
     "fruit_quality_prediction": {
       "size": string,
       "color": string,
       "sweetness": string,
       "firmness": string,
       "overall_quality": string
     },
     "recommendations": [string]
   }
   ```

## API Endpoints

### Chemical Analysis
- **Endpoint**: `/analyze/chemical`
- **Method**: POST
- **Content-Type**: application/json or application/x-www-form-urlencoded

### Yield Analysis
- **Endpoint**: `/analyze/yield`
- **Method**: POST
- **Content-Type**: application/json or application/x-www-form-urlencoded

## Usage Examples

### Chemical Analysis Request
```json
{
  "leaf_color": "Green",
  "soil_ph": 6.5,
  "moisture_level": "Medium",
  "chlorophyll_content": "Normal",
  "nitrogen_level": "Adequate"
}
```

### Yield Analysis Request
```json
{
  "tree_age": 5.0,
  "flower_buds_count": 150,
  "leaf_color": "Green",
  "soil_moisture": "Moderate",
  "fertilizer_used": true
}
```

## Error Handling
- Invalid parameter values return 400 Bad Request
- Missing required parameters return 400 Bad Request
- Server errors return 500 Internal Server Error

## Best Practices
1. Always validate input parameters before sending
2. Monitor soil moisture regularly
3. Maintain proper fertilization schedule
4. Regular leaf color monitoring
5. Keep track of tree age and development stage 