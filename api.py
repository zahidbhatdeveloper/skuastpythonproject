from fastapi import FastAPI, HTTPException, UploadFile, File, Body, Query, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
from tree_chemical_analysis import FruitTreeAnalyzer
from tree_yield_analysis import TreeYieldAnalyzer
from tree_disease_analyzer import TreeDiseaseAnalyzer
import pandas as pd
import json
import io
import traceback
import logging
from yield_analysis import YieldAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SKUAST Tree Analysis API",
    description="API for analyzing fruit tree chemical composition data",
    version="1.0.0"
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
chemical_analyzer = FruitTreeAnalyzer()
yield_analyzer = TreeYieldAnalyzer()
disease_analyzer = TreeDiseaseAnalyzer()
yield_analyzer = YieldAnalyzer()

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
async def root():
    return {"message": "Welcome to SKUAST Tree Analysis API"}

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

@app.post("/analyze/chemical")
async def analyze_chemical(
    request: Request,
    leaf_color: str = Form(None),
    soil_ph: float = Form(None),
    moisture_level: str = Form(None),
    chlorophyll_content: str = Form(None),
    nitrogen_level: str = Form(None)
):
    """
    Analyzes tree health based on 5 key parameters and provides detailed predictions and recommendations.
    """
    try:
        content_type = request.headers.get("content-type", "")
        
        if "application/x-www-form-urlencoded" in content_type:
            # Handle form data
            if not all([leaf_color, soil_ph, moisture_level, chlorophyll_content, nitrogen_level]):
                raise HTTPException(status_code=400, detail="Missing required form fields")
            
            # Create a single row DataFrame
            df = pd.DataFrame([{
                'Leaf Color': leaf_color,
                'Soil pH': soil_ph,
                'Moisture Level': moisture_level,
                'Chlorophyll Content': chlorophyll_content,
                'Nitrogen Level': nitrogen_level
            }])
            
        else:
            # Assume JSON
            data = await request.json()
            df = pd.DataFrame(data)
            df = df.rename(columns={
                'Leaf_Color': 'Leaf Color',
                'Soil_pH': 'Soil pH',
                'Moisture_Level': 'Moisture Level',
                'Chlorophyll_Content': 'Chlorophyll Content',
                'Nitrogen_Level': 'Nitrogen Level'
            })

        # Get the first row for analysis
        row = df.iloc[0]
        
        # Validate input values
        if row['Leaf Color'] not in ['Green', 'Yellow', 'Brown']:
            raise HTTPException(status_code=400, detail="Invalid Leaf Color. Must be Green, Yellow, or Brown")
        
        if not 5.5 <= row['Soil pH'] <= 7.5:
            raise HTTPException(status_code=400, detail="Invalid Soil pH. Must be between 5.5 and 7.5")
        
        if row['Moisture Level'] not in ['Low', 'Medium', 'High']:
            raise HTTPException(status_code=400, detail="Invalid Moisture Level. Must be Low, Medium, or High")
        
        if row['Chlorophyll Content'] not in ['Low', 'Normal', 'High']:
            raise HTTPException(status_code=400, detail="Invalid Chlorophyll Content. Must be Low, Normal, or High")
        
        if row['Nitrogen Level'] not in ['Low', 'Adequate', 'High']:
            raise HTTPException(status_code=400, detail="Invalid Nitrogen Level. Must be Low, Adequate, or High")

        # Analyze each parameter and generate predictions
        analysis = {
            "leaf_health": analyze_leaf_health(row['Leaf Color'], row['Chlorophyll Content']),
            "soil_condition": analyze_soil_condition(row['Soil pH'], row['Moisture Level']),
            "nutrient_status": analyze_nutrient_status(row['Nitrogen Level'], row['Soil pH']),
            "overall_health": calculate_overall_health(row),
            "predictions": generate_predictions(row),
            "recommendations": generate_detailed_recommendations(row)
        }

        return analysis

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def analyze_leaf_health(leaf_color, chlorophyll_content):
    """Analyze leaf health based on color and chlorophyll content"""
    health_status = "Good"
    issues = []
    
    if leaf_color == "Yellow":
        health_status = "Poor"
        issues.append("Leaf yellowing indicates potential nutrient deficiency or stress")
    elif leaf_color == "Brown":
        health_status = "Critical"
        issues.append("Leaf browning suggests severe stress or disease")
    
    if chlorophyll_content == "Low":
        health_status = "Poor" if health_status == "Good" else health_status
        issues.append("Low chlorophyll content indicates reduced photosynthetic capacity")
    elif chlorophyll_content == "High":
        health_status = "Excellent"
        issues.append("High chlorophyll content indicates optimal photosynthetic activity")
    
    return {
        "status": health_status,
        "issues": issues,
        "details": {
            "leaf_color": leaf_color,
            "chlorophyll_content": chlorophyll_content,
            "photosynthetic_efficiency": "High" if chlorophyll_content == "High" else "Low" if chlorophyll_content == "Low" else "Normal"
        }
    }

def analyze_soil_condition(soil_ph, moisture_level):
    """Analyze soil condition based on pH and moisture"""
    condition = "Good"
    issues = []
    
    # pH analysis
    if soil_ph < 6.0:
        condition = "Poor"
        issues.append("Soil is too acidic, may affect nutrient availability")
    elif soil_ph > 7.0:
        condition = "Poor"
        issues.append("Soil is too alkaline, may affect nutrient uptake")
    
    # Moisture analysis
    if moisture_level == "Low":
        condition = "Poor" if condition == "Good" else condition
        issues.append("Low moisture levels may cause water stress")
    elif moisture_level == "High":
        condition = "Poor" if condition == "Good" else condition
        issues.append("Excessive moisture may lead to root problems")
    
    return {
        "status": condition,
        "issues": issues,
        "details": {
            "soil_ph": soil_ph,
            "moisture_level": moisture_level,
            "nutrient_availability": "Optimal" if 6.0 <= soil_ph <= 7.0 else "Sub-optimal"
        }
    }

def analyze_nutrient_status(nitrogen_level, soil_ph):
    """Analyze nutrient status based on nitrogen level and soil pH"""
    status = "Good"
    issues = []
    
    if nitrogen_level == "Low":
        status = "Poor"
        issues.append("Low nitrogen levels may affect growth and development")
    elif nitrogen_level == "High":
        status = "Good"
        issues.append("High nitrogen levels may affect fruit quality")
    
    # Consider pH impact on nutrient availability
    if not 6.0 <= soil_ph <= 7.0:
        issues.append("Sub-optimal soil pH may affect nutrient uptake")
    
    return {
        "status": status,
        "issues": issues,
        "details": {
            "nitrogen_level": nitrogen_level,
            "nutrient_uptake_efficiency": "High" if 6.0 <= soil_ph <= 7.0 else "Low",
            "fertilization_needed": "Yes" if nitrogen_level == "Low" else "No"
        }
    }

def calculate_overall_health(row):
    """Calculate overall tree health based on all parameters"""
    health_scores = {
        "leaf_health": 1.0,
        "soil_condition": 1.0,
        "nutrient_status": 1.0
    }
    
    # Leaf health scoring
    if row['Leaf Color'] == "Green":
        health_scores["leaf_health"] = 1.0
    elif row['Leaf Color'] == "Yellow":
        health_scores["leaf_health"] = 0.6
    else:  # Brown
        health_scores["leaf_health"] = 0.3
    
    if row['Chlorophyll Content'] == "High":
        health_scores["leaf_health"] *= 1.2
    elif row['Chlorophyll Content'] == "Low":
        health_scores["leaf_health"] *= 0.7
    
    # Soil condition scoring
    if 6.0 <= row['Soil pH'] <= 7.0:
        health_scores["soil_condition"] = 1.0
    else:
        health_scores["soil_condition"] = 0.7
    
    if row['Moisture Level'] == "Medium":
        health_scores["soil_condition"] *= 1.0
    elif row['Moisture Level'] in ["Low", "High"]:
        health_scores["soil_condition"] *= 0.8
    
    # Nutrient status scoring
    if row['Nitrogen Level'] == "Adequate":
        health_scores["nutrient_status"] = 1.0
    elif row['Nitrogen Level'] == "High":
        health_scores["nutrient_status"] = 0.9
    else:  # Low
        health_scores["nutrient_status"] = 0.6
    
    # Calculate overall score
    overall_score = sum(health_scores.values()) / len(health_scores)
    
    # Determine health status
    if overall_score >= 0.9:
        status = "Excellent"
    elif overall_score >= 0.7:
        status = "Good"
    elif overall_score >= 0.5:
        status = "Fair"
    else:
        status = "Poor"
    
    return {
        "overall_score": round(overall_score * 100, 1),
        "status": status,
        "component_scores": {
            "leaf_health": round(health_scores["leaf_health"] * 100, 1),
            "soil_condition": round(health_scores["soil_condition"] * 100, 1),
            "nutrient_status": round(health_scores["nutrient_status"] * 100, 1)
        }
    }

def generate_predictions(row):
    """Generate predictions based on current conditions"""
    predictions = {
        "growth_potential": "High",
        "fruit_quality": "Good",
        "disease_risk": "Low",
        "yield_potential": "High",
        "recovery_time": "None"
    }
    
    # Adjust predictions based on leaf health
    if row['Leaf Color'] != "Green" or row['Chlorophyll Content'] == "Low":
        predictions["growth_potential"] = "Low"
        predictions["yield_potential"] = "Low"
        predictions["recovery_time"] = "2-3 months"
    
    # Adjust predictions based on soil condition
    if not 6.0 <= row['Soil pH'] <= 7.0 or row['Moisture Level'] != "Medium":
        predictions["fruit_quality"] = "Fair"
        predictions["disease_risk"] = "Medium"
    
    # Adjust predictions based on nutrient status
    if row['Nitrogen Level'] == "Low":
        predictions["growth_potential"] = "Low"
        predictions["yield_potential"] = "Low"
        predictions["recovery_time"] = "1-2 months"
    elif row['Nitrogen Level'] == "High":
        predictions["fruit_quality"] = "Fair"
    
    return predictions

def generate_detailed_recommendations(row):
    """Generate detailed recommendations based on analysis"""
    recommendations = []
    
    # Leaf health recommendations
    if row['Leaf Color'] != "Green":
        recommendations.append({
            "category": "Leaf Health",
            "issue": f"Abnormal leaf color ({row['Leaf Color']})",
            "recommendation": "Conduct leaf tissue analysis and adjust nutrient application accordingly",
            "priority": "High" if row['Leaf Color'] == "Brown" else "Medium"
        })
    
    if row['Chlorophyll Content'] == "Low":
        recommendations.append({
            "category": "Leaf Health",
            "issue": "Low chlorophyll content",
            "recommendation": "Increase nitrogen application and ensure proper sunlight exposure",
            "priority": "High"
        })
    
    # Soil condition recommendations
    if not 6.0 <= row['Soil pH'] <= 7.0:
        recommendations.append({
            "category": "Soil Management",
            "issue": f"Sub-optimal soil pH ({row['Soil pH']})",
            "recommendation": "Apply appropriate soil amendments to adjust pH to optimal range (6.0-7.0)",
            "priority": "High"
        })
    
    if row['Moisture Level'] != "Medium":
        recommendations.append({
            "category": "Irrigation",
            "issue": f"Inappropriate moisture level ({row['Moisture Level']})",
            "recommendation": "Adjust irrigation schedule to maintain optimal soil moisture",
            "priority": "High"
        })
    
    # Nutrient management recommendations
    if row['Nitrogen Level'] == "Low":
        recommendations.append({
            "category": "Nutrient Management",
            "issue": "Low nitrogen levels",
            "recommendation": "Apply nitrogen-rich fertilizer according to recommended rates",
            "priority": "High"
        })
    elif row['Nitrogen Level'] == "High":
        recommendations.append({
            "category": "Nutrient Management",
            "issue": "High nitrogen levels",
            "recommendation": "Reduce nitrogen application and monitor for excessive vegetative growth",
            "priority": "Medium"
        })
    
    # Add general recommendations
    recommendations.append({
        "category": "Monitoring",
        "issue": "Regular health assessment",
        "recommendation": "Conduct regular leaf and soil analysis to monitor changes",
        "priority": "Medium"
    })
    
    return recommendations

<<<<<<< HEAD
@app.post("/analyze/tree-health")
async def analyze_tree_health(
    request: Request,
    tree_age: float = Form(None),
    flower_buds_count: int = Form(None),
    leaf_color: str = Form(None),
    soil_moisture: str = Form(None),
    fertilizer_used: bool = Form(None)
):
    """
    Analyzes tree health based on key parameters and provides detailed predictions and recommendations.
    """
    try:
        content_type = request.headers.get("content-type", "")
        
        if "application/x-www-form-urlencoded" in content_type:
            # Handle form data
            if not all([tree_age, flower_buds_count, leaf_color, soil_moisture, fertilizer_used is not None]):
                raise HTTPException(status_code=400, detail="Missing required form fields")
            
            # Create a single row DataFrame
            df = pd.DataFrame([{
                'Tree Age': tree_age,
                'Flower Buds Count': flower_buds_count,
                'Leaf Color': leaf_color,
                'Soil Moisture': soil_moisture,
                'Fertilizer Used': fertilizer_used
            }])
            
        else:
            # Assume JSON
            data = await request.json()
            df = pd.DataFrame(data)
            df = df.rename(columns={
                'Tree_Age': 'Tree Age',
                'Flower_Buds_Count': 'Flower Buds Count',
                'Leaf_Color': 'Leaf Color',
                'Soil_Moisture': 'Soil Moisture',
                'Fertilizer_Used': 'Fertilizer Used'
            })

        # Get the first row for analysis
        row = df.iloc[0]
        
        # Validate input values
        if row['Tree Age'] < 0:
            raise HTTPException(status_code=400, detail="Tree age cannot be negative")
        
        if row['Flower Buds Count'] < 0:
            raise HTTPException(status_code=400, detail="Flower buds count cannot be negative")
        
        if row['Leaf Color'] not in ['Green', 'Yellow', 'Brown']:
            raise HTTPException(status_code=400, detail="Invalid Leaf Color. Must be Green, Yellow, or Brown")
        
        if row['Soil Moisture'] not in ['Dry', 'Moderate', 'Wet']:
            raise HTTPException(status_code=400, detail="Invalid Soil Moisture. Must be Dry, Moderate, or Wet")

        # Analyze each parameter and generate predictions
        analysis = {
            "tree_maturity": analyze_tree_maturity(row['Tree Age']),
            "flowering_potential": analyze_flowering_potential(row['Flower Buds Count'], row['Tree Age']),
            "leaf_health": analyze_leaf_health(row['Leaf Color']),
            "soil_condition": analyze_soil_condition(row['Soil Moisture']),
            "nutrient_status": analyze_nutrient_status(row['Fertilizer Used']),
            "overall_health": calculate_overall_health(row),
            "predictions": generate_predictions(row),
            "recommendations": generate_detailed_recommendations(row)
        }

        return analysis

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def analyze_tree_maturity(tree_age):
    """Analyze tree maturity based on age"""
    maturity_status = "Mature"
    issues = []
    
    if tree_age < 3:
        maturity_status = "Young"
        issues.append("Tree is too young for optimal fruit production")
    elif tree_age > 15:
        maturity_status = "Aging"
        issues.append("Tree is entering senescence phase")
    
    return {
        "status": maturity_status,
        "issues": issues,
        "details": {
            "age": tree_age,
            "maturity_stage": "Early Growth" if tree_age < 3 else "Prime" if 3 <= tree_age <= 15 else "Late Stage",
            "yield_potential": "Low" if tree_age < 3 else "High" if 3 <= tree_age <= 15 else "Declining"
        }
    }

def analyze_flowering_potential(flower_buds_count, tree_age):
    """Analyze flowering potential based on bud count and tree age"""
    potential = "Good"
    issues = []
    
    # Expected bud count varies with tree age
    expected_buds = 100 if tree_age < 3 else 200 if tree_age < 10 else 150
    
    if flower_buds_count < expected_buds * 0.5:
        potential = "Poor"
        issues.append("Low flower bud count indicates potential stress or nutrient deficiency")
    elif flower_buds_count > expected_buds * 1.5:
        potential = "Excellent"
        issues.append("High flower bud count indicates excellent tree health")
    
    return {
        "status": potential,
        "issues": issues,
        "details": {
            "bud_count": flower_buds_count,
            "expected_buds": expected_buds,
            "flowering_efficiency": "High" if flower_buds_count > expected_buds else "Low" if flower_buds_count < expected_buds * 0.5 else "Normal"
        }
    }

def analyze_leaf_health(leaf_color):
    """Analyze leaf health based on color"""
    health_status = "Good"
    issues = []
    
    if leaf_color == "Yellow":
        health_status = "Poor"
        issues.append("Leaf yellowing indicates potential nutrient deficiency or stress")
    elif leaf_color == "Brown":
        health_status = "Critical"
        issues.append("Leaf browning suggests severe stress or disease")
    
    return {
        "status": health_status,
        "issues": issues,
        "details": {
            "leaf_color": leaf_color,
            "photosynthetic_efficiency": "High" if leaf_color == "Green" else "Low" if leaf_color == "Brown" else "Moderate",
            "nutrient_status": "Optimal" if leaf_color == "Green" else "Deficient" if leaf_color == "Yellow" else "Severe Deficiency"
        }
    }

def analyze_soil_condition(soil_moisture):
    """Analyze soil condition based on moisture level"""
    condition = "Good"
    issues = []
    
    if soil_moisture == "Dry":
        condition = "Poor"
        issues.append("Dry soil may cause water stress and affect fruit development")
    elif soil_moisture == "Wet":
        condition = "Poor"
        issues.append("Excessive moisture may lead to root problems and disease")
    
    return {
        "status": condition,
        "issues": issues,
        "details": {
            "moisture_level": soil_moisture,
            "water_availability": "Optimal" if soil_moisture == "Moderate" else "Low" if soil_moisture == "Dry" else "Excessive",
            "root_health_risk": "Low" if soil_moisture == "Moderate" else "High"
        }
    }

def analyze_nutrient_status(fertilizer_used):
    """Analyze nutrient status based on fertilizer application"""
    status = "Good" if fertilizer_used else "Poor"
    issues = []
    
    if not fertilizer_used:
        issues.append("No fertilizer application may lead to nutrient deficiency")
    
    return {
        "status": status,
        "issues": issues,
        "details": {
            "fertilizer_status": "Applied" if fertilizer_used else "Not Applied",
            "nutrient_availability": "Adequate" if fertilizer_used else "Likely Deficient",
            "fertilization_needed": "No" if fertilizer_used else "Yes"
        }
    }

def calculate_overall_health(row):
    """Calculate overall tree health based on all parameters"""
    health_scores = {
        "maturity": 1.0,
        "flowering": 1.0,
        "leaf_health": 1.0,
        "soil_condition": 1.0,
        "nutrient_status": 1.0
    }
    
    # Maturity scoring
    if 3 <= row['Tree Age'] <= 15:
        health_scores["maturity"] = 1.0
    elif row['Tree Age'] < 3:
        health_scores["maturity"] = 0.6
    else:
        health_scores["maturity"] = 0.8
    
    # Flowering scoring
    expected_buds = 100 if row['Tree Age'] < 3 else 200 if row['Tree Age'] < 10 else 150
    if row['Flower Buds Count'] >= expected_buds:
        health_scores["flowering"] = 1.0
    elif row['Flower Buds Count'] >= expected_buds * 0.5:
        health_scores["flowering"] = 0.7
    else:
        health_scores["flowering"] = 0.4
    
    # Leaf health scoring
    if row['Leaf Color'] == "Green":
        health_scores["leaf_health"] = 1.0
    elif row['Leaf Color'] == "Yellow":
        health_scores["leaf_health"] = 0.6
    else:  # Brown
        health_scores["leaf_health"] = 0.3
    
    # Soil condition scoring
    if row['Soil Moisture'] == "Moderate":
        health_scores["soil_condition"] = 1.0
    else:
        health_scores["soil_condition"] = 0.6
    
    # Nutrient status scoring
    health_scores["nutrient_status"] = 1.0 if row['Fertilizer Used'] else 0.5
    
    # Calculate overall score
    overall_score = sum(health_scores.values()) / len(health_scores)
    
    # Determine health status
    if overall_score >= 0.9:
        status = "Excellent"
    elif overall_score >= 0.7:
        status = "Good"
    elif overall_score >= 0.5:
        status = "Fair"
    else:
        status = "Poor"
    
    return {
        "overall_score": round(overall_score * 100, 1),
        "status": status,
        "component_scores": {
            "maturity": round(health_scores["maturity"] * 100, 1),
            "flowering": round(health_scores["flowering"] * 100, 1),
            "leaf_health": round(health_scores["leaf_health"] * 100, 1),
            "soil_condition": round(health_scores["soil_condition"] * 100, 1),
            "nutrient_status": round(health_scores["nutrient_status"] * 100, 1)
        }
    }

def generate_predictions(row):
    """Generate predictions based on current conditions"""
    predictions = {
        "yield_potential": "High",
        "fruit_quality": "Good",
        "disease_risk": "Low",
        "recovery_time": "None",
        "next_season_potential": "High"
    }
    
    # Adjust predictions based on tree age
    if row['Tree Age'] < 3:
        predictions["yield_potential"] = "Low"
        predictions["next_season_potential"] = "Medium"
    elif row['Tree Age'] > 15:
        predictions["yield_potential"] = "Medium"
        predictions["next_season_potential"] = "Low"
    
    # Adjust predictions based on flower buds
    expected_buds = 100 if row['Tree Age'] < 3 else 200 if row['Tree Age'] < 10 else 150
    if row['Flower Buds Count'] < expected_buds * 0.5:
        predictions["yield_potential"] = "Low"
        predictions["recovery_time"] = "2-3 months"
    
    # Adjust predictions based on leaf health
    if row['Leaf Color'] != "Green":
        predictions["fruit_quality"] = "Fair"
        predictions["disease_risk"] = "High" if row['Leaf Color'] == "Brown" else "Medium"
    
    # Adjust predictions based on soil moisture
    if row['Soil Moisture'] != "Moderate":
        predictions["fruit_quality"] = "Fair"
        predictions["disease_risk"] = "Medium"
    
    # Adjust predictions based on fertilizer
    if not row['Fertilizer Used']:
        predictions["yield_potential"] = "Low"
        predictions["recovery_time"] = "1-2 months"
    
    return predictions

def generate_detailed_recommendations(row):
    """Generate detailed recommendations based on analysis"""
    recommendations = []
    
    # Tree age recommendations
    if row['Tree Age'] < 3:
        recommendations.append({
            "category": "Tree Management",
            "issue": "Young tree",
            "recommendation": "Focus on establishing strong root system and proper pruning",
            "priority": "High"
        })
    elif row['Tree Age'] > 15:
        recommendations.append({
            "category": "Tree Management",
            "issue": "Aging tree",
            "recommendation": "Consider replacement or intensive care program",
            "priority": "High"
        })
    
    # Flowering recommendations
    expected_buds = 100 if row['Tree Age'] < 3 else 200 if row['Tree Age'] < 10 else 150
    if row['Flower Buds Count'] < expected_buds * 0.5:
        recommendations.append({
            "category": "Flowering",
            "issue": "Low flower bud count",
            "recommendation": "Review pruning practices and ensure proper winter chilling",
            "priority": "High"
        })
    
    # Leaf health recommendations
    if row['Leaf Color'] != "Green":
        recommendations.append({
            "category": "Leaf Health",
            "issue": f"Abnormal leaf color ({row['Leaf Color']})",
            "recommendation": "Conduct leaf tissue analysis and adjust nutrient application",
            "priority": "High" if row['Leaf Color'] == "Brown" else "Medium"
        })
    
    # Soil moisture recommendations
    if row['Soil Moisture'] != "Moderate":
        recommendations.append({
            "category": "Irrigation",
            "issue": f"Inappropriate moisture level ({row['Soil Moisture']})",
            "recommendation": "Adjust irrigation schedule to maintain optimal soil moisture",
            "priority": "High"
        })
    
    # Fertilizer recommendations
    if not row['Fertilizer Used']:
        recommendations.append({
            "category": "Nutrient Management",
            "issue": "No fertilizer application",
            "recommendation": "Develop and implement a fertilization program",
            "priority": "High"
        })
    
    # Add general recommendations
    recommendations.append({
        "category": "Monitoring",
        "issue": "Regular health assessment",
        "recommendation": "Conduct regular tree health assessments and maintain records",
        "priority": "Medium"
    })
    
    return recommendations

@app.post("/analyze/yield")
async def analyze_yield(
    request: Request,
    tree_age: float = Form(None),
    flower_buds_count: int = Form(None),
    leaf_color: str = Form(None),
    soil_moisture: str = Form(None),
    fertilizer_used: bool = Form(None)
):
    """
    Analyze yield based on 5 key parameters and provide detailed predictions and recommendations.
    """
    try:
        content_type = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" in content_type:
            # Handle form data
            if not all([tree_age is not None, flower_buds_count is not None, leaf_color, soil_moisture, fertilizer_used is not None]):
                raise HTTPException(status_code=400, detail="Missing required form fields")
        else:
            # Assume JSON
            data = await request.json()
            tree_age = data.get('tree_age')
            flower_buds_count = data.get('flower_buds_count')
            leaf_color = data.get('leaf_color')
            soil_moisture = data.get('soil_moisture')
            fertilizer_used = data.get('fertilizer_used')

        # Convert fertilizer_used to boolean if it's a string
        if isinstance(fertilizer_used, str):
            fertilizer_used = fertilizer_used.lower() == 'true'

        result = yield_analyzer.analyze_yield(
            tree_age, flower_buds_count, leaf_color, soil_moisture, fertilizer_used
        )
        if "errors" in result:
            raise HTTPException(status_code=400, detail=result["errors"])
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

 
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000) 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
