from fastapi import FastAPI, HTTPException, UploadFile, File
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
from datetime import datetime

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

# New Pydantic models for single tree data
class ChemicalMeasurement(BaseModel):
    chemical_compound: str
    concentration: float
    dosage: float

class TreeData(BaseModel):
    tree_id: str
    tree_species: str
    measurements: List[ChemicalMeasurement]
    location: str
    season: str
    tree_age: float
    ph_level: float
    soil_type: str
    fruit_stage: str

class TreeDataResponse(BaseModel):
    tree_id: str
    tree_species: str
    analysis_results: Dict
    recommendations: List[Dict]
    status: str

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

@app.post("/tree/data", response_model=TreeDataResponse)
async def analyze_tree_data(tree_data: TreeData):
    try:
        import pandas as pd
        # Use current date/time as measurement date
        measurement_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        records = []
        for measurement in tree_data.measurements:
            record = {
                'Tree ID': tree_data.tree_id,
                'Tree Species': tree_data.tree_species,
                'Chemical Compound': measurement.chemical_compound,
                'Concentration': measurement.concentration,
                'Dosage': measurement.dosage,
                'Measurement Date': measurement_date,
                'Location': tree_data.location,
                'Season': tree_data.season,
                'Tree Age (years)': tree_data.tree_age,
                'pH Level': tree_data.ph_level,
                'Soil Type': tree_data.soil_type,
                'Fruit Stage': tree_data.fruit_stage
            }
            records.append(record)
        df = pd.DataFrame(records)

        # Calculate percentage composition
        chemical_means = df.groupby('Chemical Compound')['Concentration'].mean()
        total_concentration = chemical_means.sum()
        composition_percent = (chemical_means / total_concentration * 100).round(2)

        # Detailed chemical analysis
        chemical_analysis = []
        for compound in df['Chemical Compound'].unique():
            compound_data = df[df['Chemical Compound'] == compound]
            mean_conc = compound_data['Concentration'].mean()
            min_conc = compound_data['Concentration'].min()
            max_conc = compound_data['Concentration'].max()
            std_dev = compound_data['Concentration'].std()
            median = compound_data['Concentration'].median()
            mean_dosage = compound_data['Dosage'].mean()
            percentage = f"{composition_percent[compound]}%" if compound in composition_percent else None
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
            chemical_analysis.append({
                "compound": compound,
                "mean_concentration": round(mean_conc, 3),
                "min": round(min_conc, 3),
                "max": round(max_conc, 3),
                "std": round(std_dev, 3) if std_dev is not None else None,
                "median": round(median, 3),
                "mean_dosage": round(mean_dosage, 3),
                "percentage": percentage,
                "optimal_range": optimal_range,
                "status": status
            })

        # Environmental factors
        environmental_factors = {
            "pH_level": {
                "mean": tree_data.ph_level,
                "optimal_range": "6.0 - 7.0",
                "status": "Optimal" if 6.0 <= tree_data.ph_level <= 7.0 else "Sub-optimal"
            },
            "tree_age": {
                "years": tree_data.tree_age,
                "growth_stage": "Mature" if tree_data.tree_age > 5 else "Young"
            },
            "soil_type": tree_data.soil_type,
            "fruit_stage": tree_data.fruit_stage
        }

        # Recommendations
        recommendations = []
        for chem in chemical_analysis:
            if chem["status"] == "Sub-optimal":
                recommendations.append({
                    "compound": chem["compound"],
                    "issue": f"{chem['compound']} levels are out of optimal range.",
                    "recommendation": "Please review management practices."
                })

        # Overall health score
        health_scores = []
        for chem in chemical_analysis:
            mean_conc = chem["mean_concentration"]
            compound = chem["compound"]
            score = 1
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
        overall_health = {
            "score": round(overall_score, 1),
            "status": "Excellent - All chemical parameters are within optimal ranges" if overall_score >= 90 else \
                     ("Good - Most chemical parameters are within optimal ranges" if overall_score >= 75 else \
                     ("Fair - Some chemical parameters need attention" if overall_score >= 60 else \
                     "Poor - Multiple chemical parameters need attention"))
        }

        # Return broad, detailed response
        return {
            "tree_id": tree_data.tree_id,
            "tree_species": tree_data.tree_species,
            "location": tree_data.location,
            "season": tree_data.season,
            "tree_age": tree_data.tree_age,
            "ph_level": tree_data.ph_level,
            "soil_type": tree_data.soil_type,
            "fruit_stage": tree_data.fruit_stage,
            "measurement_date": measurement_date,
            "chemical_analysis": chemical_analysis,
            "environmental_factors": environmental_factors,
            "overall_health": overall_health,
            "recommendations": recommendations
        }
    except Exception as e:
        logger.error(f"Error in tree data analysis: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

#  if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000) 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)