from fastapi import FastAPI, HTTPException, UploadFile, File, Body, Query, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
from tree_disease_analyzer import TreeDiseaseAnalyzer
import pandas as pd
import json
import io
import traceback
import logging
import os
from yield_analysis import YieldAnalyzer
from chemical_analysis import ChemicalAnalyzer
from ml_predictor import MLPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SKUAST Tree Analysis API",
    description="API for analyzing fruit tree chemical and yield data using ML models",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzers
chemical_analyzer = ChemicalAnalyzer()
yield_analyzer = YieldAnalyzer()
disease_analyzer = TreeDiseaseAnalyzer()
ml_predictor = MLPredictor()

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

class TreeAnalysisResponse(BaseModel):
    tree_id: str
    tree_species: str
    chemical_compounds: Dict
    environmental_factors: Dict
    overall_assessment: Dict
    recommendations: List[Dict]

class YieldAnalysisResponse(BaseModel):
    tree_id: str
    tree_species: str
    report_path: Optional[str] = None
    current_yield_status: Dict
    overall_yield_statistics: Dict
    yield_trends: List[Dict]
    environmental_factors_impact: List[Dict]
    recommendations: List[Dict]

class DiseaseAnalysisResponse(BaseModel):
    timestamp: str
    image_path: str
    analysis_summary: Dict
    detected_diseases: List[Dict]
    color_analysis: Dict
    texture_analysis: Dict
    recommendations: List[Dict]

class ChemicalAnalysisRequest(BaseModel):
    Leaf_Color: str  # Green, Yellow, Brown
    Soil_pH: float   # 5.5 to 7.5
    Moisture_Level: str  # Low, Medium, High
    Chlorophyll_Content: str  # Low, Normal, High
    Nitrogen_Level: str  # Low, Adequate, High


class TreeHealthRequest(BaseModel):
    Tree_Age: float  # Age in years
    Flower_Buds_Count: int  # Number of flowers/fruit buds
    Leaf_Color: str  # Green, Yellow, Brown
    Soil_Moisture: str  # Dry, Moderate, Wet
    Fertilizer_Used: bool  # Yes/No

class YieldAnalysisRequest(BaseModel):
    Tree_Age: float  # Age in years
    Flower_Buds_Count: int  # Number of flowers/fruit buds
    Leaf_Color: str  # Green, Yellow, Brown
    Soil_Moisture: str  # Dry, Moderate, Wet
    Fertilizer_Used: bool  # Yes/No



@app.get("/")
def root():
    return {"message": "Welcome to the SKUAST Tree Analysis API", "status": "healthy"}

@app.post("/load-data")
async def load_data(file: UploadFile = File(...)):
    try:
        # Read the uploaded file content
        contents = await file.read()
        # Use io.BytesIO to simulate a file for pandas
        data = io.BytesIO(contents)
        
        # Load data into the analyzer from the uploaded file
        chemical_analyzer.load_data(data)
        
        return {"message": "Data loaded successfully", "filename": file.filename, "tree_count": len(chemical_analyzer.data['Tree ID'].unique())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tree/{tree_id}")
async def get_tree_analysis(tree_id: str):
    if chemical_analyzer.data is None:
        raise HTTPException(status_code=400, detail="Please load data first using /load-data endpoint")
    
    if tree_id not in chemical_analyzer.data['Tree ID'].unique():
        raise HTTPException(status_code=404, detail=f"Tree ID {tree_id} not found")
    
    # Get tree data
    tree_data = chemical_analyzer.data[chemical_analyzer.data['Tree ID'] == tree_id]
    
    # Calculate chemical compound statistics
    chemical_compounds = {}
    for compound in tree_data['Chemical Compound'].unique():
        compound_data = tree_data[tree_data['Chemical Compound'] == compound]
        mean_conc = compound_data['Concentration'].mean()
        min_conc = compound_data['Concentration'].min()
        max_conc = compound_data['Concentration'].max()
        std_dev = compound_data['Concentration'].std()
        cv = (std_dev / mean_conc * 100) if mean_conc != 0 and not pd.isna(std_dev) else float('nan')
        
        # Define optimal ranges and status
        optimal_range = "N/A"
        status = "N/A"
        if compound == 'Sugars':
            optimal_range = "2.5 - 4.0"
            status = "Optimal" if 2.5 <= mean_conc <= 4.0 else "Sub-optimal"
        elif compound == 'Malic Acid':
            optimal_range = "0.8 - 1.5"
            status = "Optimal" if 0.8 <= mean_conc <= 1.5 else "Sub-optimal"
        elif compound == 'Vitamin C':
            optimal_range = "0.4 - 0.8"
            status = "Optimal" if 0.4 <= mean_conc <= 0.8 else "Sub-optimal"
        elif compound == 'Chlorophyll':
            optimal_range = "2.0 - 3.0"
            status = "Optimal" if 2.0 <= mean_conc <= 3.0 else "Sub-optimal"
        elif compound == 'Anthocyanins':
            optimal_range = "3.5 - 4.5"
            status = "Optimal" if 3.5 <= mean_conc <= 4.5 else "Sub-optimal"
        elif compound == 'Pectin':
            optimal_range = "1.2 - 1.8"
            status = "Optimal" if 1.2 <= mean_conc <= 1.8 else "Sub-optimal"
        elif compound == 'Actinidin':
            optimal_range = "0.8 - 1.2"
            status = "Optimal" if 0.8 <= mean_conc <= 1.2 else "Sub-optimal"
        elif compound == 'Fiber':
            optimal_range = "1.8 - 2.4"
            status = "Optimal" if 1.8 <= mean_conc <= 2.4 else "Sub-optimal"
        
        chemical_compounds[compound] = {
            "mean_concentration": round(mean_conc, 3),
            "range": f"{round(min_conc, 3)} - {round(max_conc, 3)}",
            "std_dev": round(std_dev, 3) if not pd.isna(std_dev) else None,
            "cv_percent": round(cv, 1) if not pd.isna(cv) else None,
            "optimal_range": optimal_range,
            "status": status
        }
    
    # Calculate environmental factors
    environmental_factors = {
        "pH_level": {
            "mean": round(tree_data['pH Level'].mean(), 2),
            "range": f"{round(tree_data['pH Level'].min(), 2)} - {round(tree_data['pH Level'].max(), 2)}",
            "optimal_range": "6.0 - 7.0",
            "status": "Optimal" if 6.0 <= tree_data['pH Level'].mean() <= 7.0 else "Sub-optimal"
        },
        "tree_age": {
            "mean": round(tree_data['Tree Age (years)'].mean(), 1),
            "range": f"{round(tree_data['Tree Age (years)'].min(), 1)} - {round(tree_data['Tree Age (years)'].max(), 1)}",
            "growth_stage": "Mature" if tree_data['Tree Age (years)'].mean() > 5 else "Young"
        },
        "soil_type": tree_data['Soil Type'].mode()[0],
        "fruit_stage": tree_data['Fruit Stage'].mode()[0],
        "location": tree_data['Location'].mode()[0]
    }
    
    # Calculate overall assessment
    health_scores = []
    for compound in tree_data['Chemical Compound'].unique():
        compound_data = tree_data[tree_data['Chemical Compound'] == compound]
        mean_conc = compound_data['Concentration'].mean()
        
        score = 1  # Default score if not a key compound
        if compound == 'Sugars':
            score = 1 if 2.5 <= mean_conc <= 4.0 else 0.5
        elif compound == 'Malic Acid':
            score = 1 if 0.8 <= mean_conc <= 1.5 else 0.5
        elif compound == 'Vitamin C':
            score = 1 if 0.4 <= mean_conc <= 0.8 else 0.5
        elif compound == 'Chlorophyll':
            score = 1 if 2.0 <= mean_conc <= 3.0 else 0.5
        elif compound == 'Anthocyanins':
            score = 1 if 3.5 <= mean_conc <= 4.5 else 0.5
        elif compound == 'Pectin':
            score = 1 if 1.2 <= mean_conc <= 1.8 else 0.5
        elif compound == 'Actinidin':
            score = 1 if 0.8 <= mean_conc <= 1.2 else 0.5
        elif compound == 'Fiber':
            score = 1 if 1.8 <= mean_conc <= 2.4 else 0.5
            
        health_scores.append(score)
    
    overall_score = (sum(health_scores) / len(health_scores) * 100) if health_scores else 0
    
    overall_assessment = {
        "health_score": round(overall_score, 1),
        "status": "Excellent - All chemical parameters are within optimal ranges" if overall_score >= 90 else 
                 ("Good - Most chemical parameters are within optimal ranges" if overall_score >= 75 else 
                 ("Fair - Some chemical parameters need attention" if overall_score >= 60 else 
                 "Poor - Multiple chemical parameters need attention"))
    }
    
    # Generate recommendations
    recommendations = []
    for compound in tree_data['Chemical Compound'].unique():
        compound_data = tree_data[tree_data['Chemical Compound'] == compound]
        mean_conc = compound_data['Concentration'].mean()
        
        if compound == 'Sugars' and not (2.5 <= mean_conc <= 4.0):
            recommendations.append({
                "compound": compound,
                "issue": f"Sugar levels are {'low' if mean_conc < 2.5 else 'high'}",
                "recommendation": "Consider adjusting fertilization and irrigation."
            })
        elif compound == 'Malic Acid' and not (0.8 <= mean_conc <= 1.5):
            recommendations.append({
                "compound": compound,
                "issue": f"Malic acid levels are {'low' if mean_conc < 0.8 else 'high'}",
                "recommendation": "Review fruit maturity and harvest timing."
            })
        elif compound == 'Vitamin C' and not (0.4 <= mean_conc <= 0.8):
            recommendations.append({
                "compound": compound,
                "issue": f"Vitamin C levels are {'low' if mean_conc < 0.4 else 'high'}",
                "recommendation": "Check sunlight exposure and nutrient balance."
            })
        elif compound == 'Chlorophyll' and not (2.0 <= mean_conc <= 3.0):
            recommendations.append({
                "compound": compound,
                "issue": f"Chlorophyll levels are {'low' if mean_conc < 2.0 else 'high'}",
                "recommendation": "Review leaf health and nutrient uptake."
            })
        elif compound == 'Anthocyanins' and not (3.5 <= mean_conc <= 4.5):
            recommendations.append({
                "compound": compound,
                "issue": f"Anthocyanin levels are {'low' if mean_conc < 3.5 else 'high'}",
                "recommendation": "Check light exposure and temperature conditions."
            })
        elif compound == 'Pectin' and not (1.2 <= mean_conc <= 1.8):
            recommendations.append({
                "compound": compound,
                "issue": f"Pectin levels are {'low' if mean_conc < 1.2 else 'high'}",
                "recommendation": "Review fruit development stage and harvest timing."
            })
        elif compound == 'Actinidin' and not (0.8 <= mean_conc <= 1.2):
            recommendations.append({
                "compound": compound,
                "issue": f"Actinidin levels are {'low' if mean_conc < 0.8 else 'high'}",
                "recommendation": "Check fruit ripeness and storage conditions."
            })
        elif compound == 'Fiber' and not (1.8 <= mean_conc <= 2.4):
            recommendations.append({
                "compound": compound,
                "issue": f"Fiber levels are {'low' if mean_conc < 1.8 else 'high'}",
                "recommendation": "Review fruit development and harvest timing."
            })
    
    return {
        "tree_id": tree_id,
        "tree_species": tree_data['Tree Species'].iloc[0],
        "chemical_compounds": chemical_compounds,
        "environmental_factors": environmental_factors,
        "overall_assessment": overall_assessment,
        "recommendations": recommendations
    }

@app.post("/yield/load-data")
async def load_yield_data(file: UploadFile = File(...)):
    """Load yield data from a CSV file"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        yield_analyzer.data = df
        return {"message": "Yield data loaded successfully", "rows": len(df)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/yield/{tree_id}", response_model=YieldAnalysisResponse)
async def get_yield_analysis(tree_id: str):
    """Get yield analysis for a specific tree"""
    try:
        if yield_analyzer.data is None:
            raise HTTPException(status_code=400, detail="No yield data loaded. Please load yield data first.")
        
        if tree_id not in yield_analyzer.data['Tree ID'].unique():
            raise HTTPException(status_code=404, detail=f"Tree ID {tree_id} not found in data")
        
        result = yield_analyzer.analyze_yield(tree_id=tree_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/disease-detection/{tree_id}", response_model=DiseaseAnalysisResponse)
async def detect_tree_disease(tree_id: str, image: UploadFile = File(...)):
    """
    Analyze tree image for disease detection
    
    Parameters:
    - tree_id: ID of the tree being analyzed
    - image: Image file (jpg, jpeg, or png)
    
    Returns:
    - Detailed analysis of detected diseases and recommendations
    """
    try:
        # Validate image format
        if not image.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            raise HTTPException(status_code=400, detail="Invalid image format. Please upload JPG or PNG files only.")
        
        # Read image content
        image_data = await image.read()
        
        # Analyze image
        result = disease_analyzer.analyze_image(image_data, tree_id)
        
        return result
    except Exception as e:
        logger.error(f"Error in disease detection: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/chemical")
async def train_chemical_model(file: UploadFile = File(...)):
    """Train chemical analysis ML model using uploaded data"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Train the model
        score = ml_predictor.train_chemical_model(df)
        
        return {
            "message": "Chemical analysis model trained successfully",
            "model_score": score
        }
    except Exception as e:
        logger.error(f"Error training chemical model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/yield")
async def train_yield_model(file: UploadFile = File(...)):
    """Train yield analysis ML model using uploaded data"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Train the model
        score = ml_predictor.train_yield_model(df)
        
        return {
            "message": "Yield analysis model trained successfully",
            "model_score": score
        }
    except Exception as e:
        logger.error(f"Error training yield model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/chemical")
async def analyze_chemical(
    request: Request,
    leaf_color: Optional[str] = Form(None),
    soil_ph: Optional[float] = Form(None),
    moisture_level: Optional[str] = Form(None),
    chlorophyll_content: Optional[str] = Form(None),
    nitrogen_level: Optional[str] = Form(None)
):
    try:
        content_type = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" in content_type:
            if not all([leaf_color, soil_ph is not None, moisture_level, chlorophyll_content, nitrogen_level]):
                raise HTTPException(status_code=400, detail="Missing required form fields")
        else:
            data = await request.json()
            leaf_color = data.get('leaf_color')
            soil_ph = data.get('soil_ph')
            moisture_level = data.get('moisture_level')
            chlorophyll_content = data.get('chlorophyll_content')
            nitrogen_level = data.get('nitrogen_level')

        # Prepare input data for ML model
        input_data = {
            'leaf_color': leaf_color,
            'soil_ph': float(soil_ph),
            'moisture_level': moisture_level,
            'chlorophyll_content': chlorophyll_content,
            'nitrogen_level': nitrogen_level
        }

        # Get ML prediction
        ml_result = ml_predictor.predict_chemical(input_data)
        
        # Get traditional analysis
        traditional_result = chemical_analyzer.analyze(
            leaf_color, float(soil_ph), moisture_level, chlorophyll_content, nitrogen_level
        )

        # Combine results
        result = {
            **traditional_result,
            'ml_prediction': ml_result['prediction'],
            'feature_importance': ml_result['feature_importance']
        }

        return result
    except Exception as e:
        logger.error(f"Error in chemical analysis: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze/yield")
async def analyze_yield(
    request: Request,
    tree_age: Optional[float] = Form(None),
    flower_buds_count: Optional[int] = Form(None),
    leaf_color: Optional[str] = Form(None),
    soil_moisture: Optional[str] = Form(None),
    fertilizer_used: Optional[bool] = Form(None)
):
    try:
        content_type = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" in content_type:
            if not all([tree_age is not None, flower_buds_count is not None, leaf_color, soil_moisture, fertilizer_used is not None]):
                raise HTTPException(status_code=400, detail="Missing required form fields")
        else:
            data = await request.json()
            tree_age = data.get('tree_age')
            flower_buds_count = data.get('flower_buds_count')
            leaf_color = data.get('leaf_color')
            soil_moisture = data.get('soil_moisture')
            fertilizer_used = data.get('fertilizer_used')

        # Convert fertilizer_used to boolean if string
        if isinstance(fertilizer_used, str):
            fertilizer_used = fertilizer_used.lower() == 'true'

        # Prepare input data for ML model
        input_data = {
            'tree_age': float(tree_age),
            'flower_buds_count': int(flower_buds_count),
            'leaf_color': leaf_color,
            'soil_moisture': soil_moisture,
            'fertilizer_used': fertilizer_used
        }

        # Get ML prediction
        ml_result = ml_predictor.predict_yield(input_data)
        
        # Get traditional analysis
        traditional_result = yield_analyzer.analyze(
            float(tree_age), int(flower_buds_count), leaf_color, soil_moisture, fertilizer_used
        )

        # Combine results
        result = {
            **traditional_result,
            'ml_prediction': ml_result['prediction'],
            'feature_importance': ml_result['feature_importance']
        }

        return result
    except Exception as e:
        logger.error(f"Error in yield analysis: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
