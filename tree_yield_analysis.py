import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from tabulate import tabulate
from fastapi import HTTPException
warnings.filterwarnings('ignore')

# Set the style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

class TreeYieldAnalyzer:
    def __init__(self):
        self.data = None
        self.data_dir = 'data'
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def load_data(self, file_path):
        """Load data from CSV file"""
        try:
            self.data = pd.read_csv(file_path)
            print(f"Data loaded successfully with {len(self.data)} records")
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def analyze_yield(self, tree_age, flower_buds_count, leaf_color, soil_moisture, fertilizer_used):
        """
        Analyzes yield potential based on key parameters and provides detailed predictions and recommendations.
        """
        try:
            # Validate input values
            if not isinstance(tree_age, (int, float)) or tree_age < 0:
                raise ValueError("Tree age must be a positive number")
            
            if not isinstance(flower_buds_count, int) or flower_buds_count < 0:
                raise ValueError("Flower buds count must be a positive integer")
            
            if leaf_color not in ['Green', 'Yellow', 'Brown']:
                raise ValueError("Invalid Leaf Color. Must be Green, Yellow, or Brown")
            
            if soil_moisture not in ['Dry', 'Moderate', 'Wet']:
                raise ValueError("Invalid Soil Moisture. Must be Dry, Moderate, or Wet")
            
            if not isinstance(fertilizer_used, bool):
                raise ValueError("Fertilizer Used must be a boolean value (True/False)")
            
            # Create a single row DataFrame for analysis
            df = pd.DataFrame([{
                'Tree Age': tree_age,
                'Flower Buds Count': flower_buds_count,
                'Leaf Color': leaf_color,
                'Soil Moisture': soil_moisture,
                'Fertilizer Used': fertilizer_used
            }])
            
            # Get the first row for analysis
            row = df.iloc[0]
            
            # Calculate yield potential based on all factors
            yield_analysis = {
                "yield_potential": self._calculate_yield_potential(row),
                "fruit_quality_prediction": self._predict_fruit_quality(row),
                "yield_limiting_factors": self._identify_limiting_factors(row),
                "expected_yield_range": self._calculate_expected_yield_range(row),
                "recommendations": self._generate_yield_recommendations(row)
            }
            
            # Generate and save report
            report_path = self._generate_yield_report(yield_analysis, row)
            
            return yield_analysis, report_path
            
        except Exception as e:
            print(f"Error in yield analysis: {str(e)}")
            return None, None
    
    def _calculate_yield_potential(self, row):
        """Calculate overall yield potential based on all factors"""
        # Base yield potential score (0-100)
        base_score = 100
        
        # Adjust for tree age
        if row['Tree Age'] < 3:
            base_score *= 0.5  # Young trees have lower yield potential
        elif row['Tree Age'] > 15:
            base_score *= 0.7  # Aging trees have reduced yield potential
        
        # Adjust for flower buds
        expected_buds = 100 if row['Tree Age'] < 3 else 200 if row['Tree Age'] < 10 else 150
        bud_ratio = row['Flower Buds Count'] / expected_buds
        if bud_ratio < 0.5:
            base_score *= 0.6
        elif bud_ratio > 1.5:
            base_score *= 0.9  # Too many buds can reduce fruit size
        
        # Adjust for leaf health
        if row['Leaf Color'] == "Yellow":
            base_score *= 0.7
        elif row['Leaf Color'] == "Brown":
            base_score *= 0.4
        
        # Adjust for soil moisture
        if row['Soil Moisture'] == "Dry":
            base_score *= 0.6
        elif row['Soil Moisture'] == "Wet":
            base_score *= 0.7
        
        # Adjust for fertilizer
        if not row['Fertilizer Used']:
            base_score *= 0.8
        
        # Calculate final score
        final_score = round(base_score, 1)
        
        # Determine yield potential category
        if final_score >= 80:
            potential = "High"
        elif final_score >= 60:
            potential = "Medium"
        else:
            potential = "Low"
        
        return {
            "score": final_score,
            "category": potential,
            "details": {
                "tree_age_factor": "Optimal" if 3 <= row['Tree Age'] <= 15 else "Sub-optimal",
                "flowering_factor": "Optimal" if 0.5 <= bud_ratio <= 1.5 else "Sub-optimal",
                "leaf_health_factor": "Optimal" if row['Leaf Color'] == "Green" else "Sub-optimal",
                "soil_factor": "Optimal" if row['Soil Moisture'] == "Moderate" else "Sub-optimal",
                "nutrient_factor": "Optimal" if row['Fertilizer Used'] else "Sub-optimal"
            }
        }
    
    def _predict_fruit_quality(self, row):
        """Predict fruit quality based on input parameters"""
        quality_factors = {
            "size": "Medium",
            "color": "Good",
            "sweetness": "Medium",
            "firmness": "Good",
            "overall_quality": "Good"
        }
        
        # Adjust for tree age
        if row['Tree Age'] < 3:
            quality_factors["size"] = "Small"
            quality_factors["overall_quality"] = "Fair"
        elif row['Tree Age'] > 15:
            quality_factors["firmness"] = "Medium"
        
        # Adjust for leaf health
        if row['Leaf Color'] != "Green":
            quality_factors["color"] = "Poor"
            quality_factors["sweetness"] = "Low"
            quality_factors["overall_quality"] = "Fair"
        
        # Adjust for soil moisture
        if row['Soil Moisture'] == "Dry":
            quality_factors["size"] = "Small"
            quality_factors["sweetness"] = "High"
        elif row['Soil Moisture'] == "Wet":
            quality_factors["firmness"] = "Poor"
        
        # Adjust for fertilizer
        if not row['Fertilizer Used']:
            quality_factors["size"] = "Small"
            quality_factors["sweetness"] = "Low"
        
        return quality_factors
    
    def _identify_limiting_factors(self, row):
        """Identify factors limiting yield potential"""
        limiting_factors = []
        
        # Tree age factors
        if row['Tree Age'] < 3:
            limiting_factors.append({
                "factor": "Tree Age",
                "issue": "Young tree",
                "impact": "High",
                "description": "Tree is too young for optimal fruit production"
            })
        elif row['Tree Age'] > 15:
            limiting_factors.append({
                "factor": "Tree Age",
                "issue": "Aging tree",
                "impact": "Medium",
                "description": "Tree is entering senescence phase"
            })
        
        # Flowering factors
        expected_buds = 100 if row['Tree Age'] < 3 else 200 if row['Tree Age'] < 10 else 150
        if row['Flower Buds Count'] < expected_buds * 0.5:
            limiting_factors.append({
                "factor": "Flowering",
                "issue": "Low flower bud count",
                "impact": "High",
                "description": "Insufficient flowers for optimal yield"
            })
        
        # Leaf health factors
        if row['Leaf Color'] != "Green":
            limiting_factors.append({
                "factor": "Leaf Health",
                "issue": f"Abnormal leaf color ({row['Leaf Color']})",
                "impact": "High" if row['Leaf Color'] == "Brown" else "Medium",
                "description": "Poor leaf health affects photosynthesis and fruit development"
            })
        
        # Soil moisture factors
        if row['Soil Moisture'] != "Moderate":
            limiting_factors.append({
                "factor": "Soil Moisture",
                "issue": f"Inappropriate moisture level ({row['Soil Moisture']})",
                "impact": "High",
                "description": "Sub-optimal moisture affects fruit development and quality"
            })
        
        # Nutrient factors
        if not row['Fertilizer Used']:
            limiting_factors.append({
                "factor": "Nutrients",
                "issue": "No fertilizer application",
                "impact": "High",
                "description": "Lack of nutrients affects fruit development and yield"
            })
        
        return limiting_factors
    
    def _calculate_expected_yield_range(self, row):
        """Calculate expected yield range based on input parameters"""
        # Base yield range (kg per tree)
        base_yield = {
            "minimum": 15,
            "maximum": 30,
            "expected": 22.5
        }
        
        # Adjust for tree age
        if row['Tree Age'] < 3:
            base_yield["minimum"] *= 0.5
            base_yield["maximum"] *= 0.5
            base_yield["expected"] *= 0.5
        elif row['Tree Age'] > 15:
            base_yield["minimum"] *= 0.7
            base_yield["maximum"] *= 0.7
            base_yield["expected"] *= 0.7
        
        # Adjust for flower buds
        expected_buds = 100 if row['Tree Age'] < 3 else 200 if row['Tree Age'] < 10 else 150
        bud_ratio = row['Flower Buds Count'] / expected_buds
        if bud_ratio < 0.5:
            base_yield["minimum"] *= 0.6
            base_yield["maximum"] *= 0.6
            base_yield["expected"] *= 0.6
        
        # Adjust for leaf health
        if row['Leaf Color'] == "Yellow":
            base_yield["minimum"] *= 0.7
            base_yield["maximum"] *= 0.7
            base_yield["expected"] *= 0.7
        elif row['Leaf Color'] == "Brown":
            base_yield["minimum"] *= 0.4
            base_yield["maximum"] *= 0.4
            base_yield["expected"] *= 0.4
        
        # Adjust for soil moisture
        if row['Soil Moisture'] != "Moderate":
            base_yield["minimum"] *= 0.7
            base_yield["maximum"] *= 0.7
            base_yield["expected"] *= 0.7
        
        # Adjust for fertilizer
        if not row['Fertilizer Used']:
            base_yield["minimum"] *= 0.8
            base_yield["maximum"] *= 0.8
            base_yield["expected"] *= 0.8
        
        # Round all values
        return {
            "minimum": round(base_yield["minimum"], 1),
            "maximum": round(base_yield["maximum"], 1),
            "expected": round(base_yield["expected"], 1),
            "unit": "kg per tree"
        }
    
    def _generate_yield_recommendations(self, row):
        """Generate recommendations to improve yield"""
        recommendations = []
        
        # Tree age recommendations
        if row['Tree Age'] < 3:
            recommendations.append({
                "category": "Tree Management",
                "issue": "Young tree",
                "recommendation": "Focus on establishing strong root system and proper pruning",
                "priority": "High",
                "expected_impact": "Long-term yield improvement"
            })
        elif row['Tree Age'] > 15:
            recommendations.append({
                "category": "Tree Management",
                "issue": "Aging tree",
                "recommendation": "Consider replacement or intensive care program",
                "priority": "High",
                "expected_impact": "Yield stabilization"
            })
        
        # Flowering recommendations
        expected_buds = 100 if row['Tree Age'] < 3 else 200 if row['Tree Age'] < 10 else 150
        if row['Flower Buds Count'] < expected_buds * 0.5:
            recommendations.append({
                "category": "Flowering",
                "issue": "Low flower bud count",
                "recommendation": "Review pruning practices and ensure proper winter chilling",
                "priority": "High",
                "expected_impact": "Immediate yield improvement"
            })
        
        # Leaf health recommendations
        if row['Leaf Color'] != "Green":
            recommendations.append({
                "category": "Leaf Health",
                "issue": f"Abnormal leaf color ({row['Leaf Color']})",
                "recommendation": "Conduct leaf tissue analysis and adjust nutrient application",
                "priority": "High" if row['Leaf Color'] == "Brown" else "Medium",
                "expected_impact": "Medium-term yield improvement"
            })
        
        # Soil moisture recommendations
        if row['Soil Moisture'] != "Moderate":
            recommendations.append({
                "category": "Irrigation",
                "issue": f"Inappropriate moisture level ({row['Soil Moisture']})",
                "recommendation": "Adjust irrigation schedule to maintain optimal soil moisture",
                "priority": "High",
                "expected_impact": "Immediate yield improvement"
            })
        
        # Fertilizer recommendations
        if not row['Fertilizer Used']:
            recommendations.append({
                "category": "Nutrient Management",
                "issue": "No fertilizer application",
                "recommendation": "Develop and implement a fertilization program",
                "priority": "High",
                "expected_impact": "Medium-term yield improvement"
            })
        
        # Add general recommendations
        recommendations.append({
            "category": "Monitoring",
            "issue": "Regular yield assessment",
            "recommendation": "Conduct regular yield assessments and maintain records",
            "priority": "Medium",
            "expected_impact": "Long-term yield optimization"
        })
        
        return recommendations
    
    def _generate_yield_report(self, yield_analysis, row):
        """Generate a detailed HTML report for the yield analysis"""
        report_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Tree Yield Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        .score {{ font-size: 24px; font-weight: bold; color: #27ae60; }}
        .warning {{ color: #e74c3c; }}
        .recommendation {{ background-color: #f8f9fa; padding: 10px; margin: 5px 0; border-left: 4px solid #3498db; }}
        .high-priority {{ border-left-color: #e74c3c; }}
        .medium-priority {{ border-left-color: #f1c40f; }}
        .low-priority {{ border-left-color: #2ecc71; }}
    </style>
</head>
<body>
    <h1>Tree Yield Analysis Report</h1>
    
    <div class="section">
        <h2>Input Parameters</h2>
        <p>Tree Age: {row['Tree Age']} years</p>
        <p>Flower Buds Count: {row['Flower Buds Count']}</p>
        <p>Leaf Color: {row['Leaf Color']}</p>
        <p>Soil Moisture: {row['Soil Moisture']}</p>
        <p>Fertilizer Used: {'Yes' if row['Fertilizer Used'] else 'No'}</p>
    </div>
    
    <div class="section">
        <h2>Yield Potential</h2>
        <p class="score">Score: {yield_analysis['yield_potential']['score']}%</p>
        <p>Category: {yield_analysis['yield_potential']['category']}</p>
        <h3>Factor Analysis:</h3>
        <ul>
            <li>Tree Age: {yield_analysis['yield_potential']['details']['tree_age_factor']}</li>
            <li>Flowering: {yield_analysis['yield_potential']['details']['flowering_factor']}</li>
            <li>Leaf Health: {yield_analysis['yield_potential']['details']['leaf_health_factor']}</li>
            <li>Soil Condition: {yield_analysis['yield_potential']['details']['soil_factor']}</li>
            <li>Nutrient Status: {yield_analysis['yield_potential']['details']['nutrient_factor']}</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Fruit Quality Prediction</h2>
        <ul>
            <li>Size: {yield_analysis['fruit_quality_prediction']['size']}</li>
            <li>Color: {yield_analysis['fruit_quality_prediction']['color']}</li>
            <li>Sweetness: {yield_analysis['fruit_quality_prediction']['sweetness']}</li>
            <li>Firmness: {yield_analysis['fruit_quality_prediction']['firmness']}</li>
            <li>Overall Quality: {yield_analysis['fruit_quality_prediction']['overall_quality']}</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Expected Yield Range</h2>
        <p>Minimum: {yield_analysis['expected_yield_range']['minimum']} {yield_analysis['expected_yield_range']['unit']}</p>
        <p>Maximum: {yield_analysis['expected_yield_range']['maximum']} {yield_analysis['expected_yield_range']['unit']}</p>
        <p>Expected: {yield_analysis['expected_yield_range']['expected']} {yield_analysis['expected_yield_range']['unit']}</p>
    </div>
    
    <div class="section">
        <h2>Limiting Factors</h2>
        <ul>
            {''.join(f"<li><strong>{factor['factor']}:</strong> {factor['issue']} - {factor['description']}</li>" for factor in yield_analysis['yield_limiting_factors'])}
        </ul>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        {''.join(f"""
        <div class="recommendation {'high-priority' if rec['priority'] == 'High' else 'medium-priority' if rec['priority'] == 'Medium' else 'low-priority'}">
            <h3>{rec['category']} - {rec['issue']}</h3>
            <p><strong>Recommendation:</strong> {rec['recommendation']}</p>
            <p><strong>Priority:</strong> {rec['priority']}</p>
            <p><strong>Expected Impact:</strong> {rec['expected_impact']}</p>
        </div>
        """ for rec in yield_analysis['recommendations'])}
    </div>
</body>
</html>
"""
        
        # Save the report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.data_dir, f'yield_analysis_report_{timestamp}.html')
        with open(report_path, 'w') as f:
            f.write(report_html)
        
        return report_path

def main():
    print("SKUAST Tree Yield Analysis System")
    print("--------------------------------")
    
    analyzer = TreeYieldAnalyzer()
    
    while True:
        print("\nYield Analysis Menu:")
        print("1. Analyze Tree Yield")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1-2): ")
        
        if choice == '1':
            try:
                # Get input parameters with validation
                while True:
                    try:
                        tree_age = float(input("Enter tree age (years): "))
                        if tree_age < 0:
                            print("Tree age cannot be negative. Please try again.")
                            continue
                        break
                    except ValueError:
                        print("Please enter a valid number for tree age.")
                
                while True:
                    try:
                        flower_buds_count = int(input("Enter number of flower buds: "))
                        if flower_buds_count < 0:
                            print("Flower buds count cannot be negative. Please try again.")
                            continue
                        break
                    except ValueError:
                        print("Please enter a valid integer for flower buds count.")
                
                while True:
                    leaf_color = input("Enter leaf color (Green/Yellow/Brown): ").strip().capitalize()
                    if leaf_color not in ['Green', 'Yellow', 'Brown']:
                        print("Invalid leaf color. Please enter Green, Yellow, or Brown.")
                        continue
                    break
                
                while True:
                    soil_moisture = input("Enter soil moisture (Dry/Moderate/Wet): ").strip().capitalize()
                    if soil_moisture not in ['Dry', 'Moderate', 'Wet']:
                        print("Invalid soil moisture. Please enter Dry, Moderate, or Wet.")
                        continue
                    break
                
                while True:
                    fertilizer_input = input("Has fertilizer been used? (Yes/No): ").strip().lower()
                    if fertilizer_input not in ['yes', 'no']:
                        print("Please enter Yes or No.")
                        continue
                    fertilizer_used = fertilizer_input == 'yes'
                    break
                
                # Perform analysis
                yield_analysis, report_path = analyzer.analyze_yield(
                    tree_age, flower_buds_count, leaf_color, soil_moisture, fertilizer_used
                )
                
                if yield_analysis and report_path:
                    print(f"\nAnalysis complete! Report saved to: {report_path}")
                    
                    # Print summary
                    print("\nYield Analysis Summary:")
                    print(f"Yield Potential: {yield_analysis['yield_potential']['category']} ({yield_analysis['yield_potential']['score']}%)")
                    print(f"Expected Yield: {yield_analysis['expected_yield_range']['expected']} {yield_analysis['expected_yield_range']['unit']}")
                    print(f"Overall Fruit Quality: {yield_analysis['fruit_quality_prediction']['overall_quality']}")
                    
                    # Open the report in the default web browser
                    import webbrowser
                    webbrowser.open(f'file://{os.path.abspath(report_path)}')
                else:
                    print("Analysis failed. Please check the input parameters and try again.")
                
            except Exception as e:
                print(f"An error occurred: {str(e)}")
        
        elif choice == '2':
            print("Thank you for using the SKUAST Tree Yield Analysis System!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 