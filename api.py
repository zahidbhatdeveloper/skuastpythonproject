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
    Tree_ID: str
    Tree_Species: str
    Chemical_Compound: str
    Concentration: float
    Previous_Dosage: float
    Measurement_Date: str
    Location: str
    Season: str
    Tree_Age_years: float
    pH_Level: float
    Soil_Type: str
    Fruit_Stage: str

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
async def analyze_chemical(request: Request, tree_id: str = Query(None), data: str = Form(None)):
    """
    Accepts JSON, CSV, or form data (with a 'data' field containing CSV or JSON) in the POST body.
    If CSV, parses it and analyzes the data for the tree.
    If JSON, behaves as before.
    If form, parses the 'data' field.
    """
    try:
        content_type = request.headers.get("content-type", "")
        df = None
        if data is not None:
            # Data provided via form
            if data.strip().startswith("["):
                # JSON array
                import json
                records = json.loads(data)
                df = pd.DataFrame(records)
                df = df.rename(columns={
                    'Tree_ID': 'Tree ID',
                    'Tree_Species': 'Tree Species',
                    'Chemical_Compound': 'Chemical Compound',
                    'Concentration': 'Concentration',
                    'Previous_Dosage': 'Previous Dosage',
                    'Measurement_Date': 'Measurement Date',
                    'Location': 'Location',
                    'Season': 'Season',
                    'Tree_Age_years': 'Tree Age (years)',
                    'pH_Level': 'pH Level',
                    'Soil_Type': 'Soil Type',
                    'Fruit_Stage': 'Fruit Stage',
                })
            else:
                # Assume CSV
                from io import StringIO
                columns = [
                    "Tree ID", "Tree Species", "Chemical Compound", "Concentration", "Previous Dosage",
                    "Measurement Date", "Location", "Season", "Tree Age (years)", "pH Level", "Soil Type", "Fruit Stage"
                ]
                df = pd.read_csv(StringIO(data), names=columns)
        elif "text/csv" in content_type:
            # Read raw body as text
            body = await request.body()
            csv_text = body.decode("utf-8")
            columns = [
                "Tree ID", "Tree Species", "Chemical Compound", "Concentration", "Previous Dosage",
                "Measurement Date", "Location", "Season", "Tree Age (years)", "pH Level", "Soil Type", "Fruit Stage"
            ]
            from io import StringIO
            df = pd.read_csv(StringIO(csv_text), names=columns)
        else:
            # Assume JSON
            data_json = await request.json()
            df = pd.DataFrame(data_json)
            df = df.rename(columns={
                'Tree_ID': 'Tree ID',
                'Tree_Species': 'Tree Species',
                'Chemical_Compound': 'Chemical Compound',
                'Concentration': 'Concentration',
                'Previous_Dosage': 'Previous Dosage',
                'Measurement_Date': 'Measurement Date',
                'Location': 'Location',
                'Season': 'Season',
                'Tree_Age_years': 'Tree Age (years)',
                'pH_Level': 'pH Level',
                'Soil_Type': 'Soil Type',
                'Fruit_Stage': 'Fruit Stage',
            })
        # If tree_id is provided, filter for that tree
        if tree_id is not None:
            df = df[df['Tree ID'] == tree_id]
            if df.empty:
                raise HTTPException(status_code=404, detail=f"Tree ID {tree_id} not found in input data")
        tree_id_val = df['Tree ID'].iloc[0]
        tree_species = df['Tree Species'].iloc[0]
        tree_data = df
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
            "status": "Excellent - All chemical parameters are within optimal ranges" if overall_score >= 90 else \
                     ("Good - Most chemical parameters are within optimal ranges" if overall_score >= 75 else \
                     ("Fair - Some chemical parameters need attention" if overall_score >= 60 else \
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
            "tree_id": tree_id_val,
            "tree_species": tree_species,
            "chemical_compounds": chemical_compounds,
            "environmental_factors": environmental_factors,
            "overall_assessment": overall_assessment,
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000) 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)