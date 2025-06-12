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
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def load_data(self, file_path):
        """Load yield data from CSV file"""
        try:
            self.data = pd.read_csv(file_path)
            print(f"Yield data loaded successfully with {len(self.data)} records")
            # Convert date column to datetime
            self.data['Measurement Date'] = pd.to_datetime(self.data['Measurement Date'])
            # Print available tree IDs
            print("\nAvailable Tree IDs:")
            for species in self.data['Tree Species'].unique():
                tree_ids = self.data[self.data['Tree Species'] == species]['Tree ID'].unique()
                print(f"{species}: {', '.join(tree_ids)}")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
    
    def analyze_yield(self, tree_id=None, tree_species=None):
        """Analyze yield data for a specific tree or species"""
        try:
            if self.data is None:
                raise ValueError("No data loaded. Please load data first.")
            
            # Filter data based on tree_id or tree_species
            if tree_id:
                tree_data = self.data[self.data['Tree ID'] == tree_id]
            elif tree_species:
                tree_data = self.data[self.data['Tree Species'] == tree_species]
            else:
                tree_data = self.data
            
            if tree_data.empty:
                raise ValueError(f"No data found for {'Tree ID: ' + tree_id if tree_id else 'Tree Species: ' + tree_species}")
            
            # Get the latest data point
            latest_data = tree_data.sort_values('Measurement Date').iloc[-1:].reset_index(drop=True)
            
            # Calculate yield trends
            yield_trends = tree_data.groupby('Measurement Date')['Yield (kg)'].mean().reset_index()
            
            # Calculate overall statistics
            overall_avg_yield = tree_data['Yield (kg)'].mean()
            overall_min_yield = tree_data['Yield (kg)'].min()
            overall_max_yield = tree_data['Yield (kg)'].max()
            overall_std_yield = tree_data['Yield (kg)'].std()
            
            # Get environmental factors
            env_factors = {}
            if not latest_data.empty:
                for col in ['Tree Age (years)', 'pH Level', 'Soil Type', 'Fruit Stage', 'Irrigation (mm)', 'Fertilizer Applied (kg)']:
                    if col in latest_data.columns and pd.notna(latest_data[col].iloc[0]):
                        env_factors[col] = latest_data[col].iloc[0]
            
            # Generate recommendations
            recommendations = self._generate_yield_recommendations(env_factors, latest_data['Yield (kg)'].iloc[0] if not latest_data.empty else None)
            
            # Generate HTML report
            report_html = self.generate_yield_report(tree_data, tree_id, tree_species)
            
            # Save the HTML report
            report_file_path = os.path.join(self.data_dir, f'yield_analysis_report_{tree_id if tree_id else tree_species.lower().replace(" ", "_")}.html')
            with open(report_file_path, 'w') as f:
                f.write(report_html)
            
            # Prepare the response dictionary
            yield_analysis_results = {
                "tree_id": tree_id,
                "tree_species": tree_data['Tree Species'].iloc[0],
                "report_path": os.path.abspath(report_file_path),
                "current_yield_status": {
                    "tree_id": latest_data['Tree ID'].iloc[0] if not latest_data.empty else None,
                    "species": latest_data['Tree Species'].iloc[0] if not latest_data.empty else None,
                    "current_yield_kg": float(latest_data['Yield (kg)'].iloc[0]) if not latest_data.empty else None,
                    "fruit_count": int(latest_data['Fruit Count'].iloc[0]) if not latest_data.empty else None,
                    "avg_fruit_weight_g": float(latest_data['Average Fruit Weight (g)'].iloc[0]) if not latest_data.empty else None,
                    "fruit_size_cm": float(latest_data['Fruit Size (cm)'].iloc[0]) if not latest_data.empty else None,
                    "fruit_color": latest_data['Fruit Color'].iloc[0] if not latest_data.empty else None,
                    "harvest_date": latest_data['Harvest Date'].iloc[0] if not latest_data.empty else None
                },
                "overall_yield_statistics": {
                    "average_yield_kg": float(overall_avg_yield),
                    "minimum_yield_kg": float(overall_min_yield),
                    "maximum_yield_kg": float(overall_max_yield),
                    "std_dev_yield_kg": float(overall_std_yield)
                },
                "yield_trends": yield_trends.reset_index().rename(columns={'Measurement Date': 'date', 'Yield (kg)': 'yield_kg'}).to_dict(orient='records'),
                "environmental_factors_impact": [
                    {"factor": f, "value": str(v), "impact_percent": float(self._calculate_environmental_impact(f, v))}
                    for f, v in env_factors.items()
                ],
                "recommendations": [
                    {"factor": r[0], "recommendation": r[1]}
                    for r in recommendations
                ]
            }
            
            return yield_analysis_results
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    def _calculate_environmental_impact(self, factor, value):
        """Calculate the impact of environmental factors on yield"""
        try:
            if pd.isna(value):
                return 0.0
            
            # Define optimal ranges for different factors
            optimal_ranges = {
                'Tree Age (years)': (3, 10),
                'pH Level': (6.0, 7.0),
                'Irrigation (mm)': (30, 40),
                'Fertilizer Applied (kg)': (3.0, 4.0)
            }
            
            if factor in optimal_ranges:
                min_val, max_val = optimal_ranges[factor]
                if min_val <= float(value) <= max_val:
                    return 100.0
                else:
                    # Calculate deviation from optimal range
                    deviation = min(abs(float(value) - min_val), abs(float(value) - max_val))
                    return max(0.0, 100.0 - (deviation * 20.0))  # 20% penalty per unit deviation
            
            return 50.0  # Default impact for other factors
            
        except Exception as e:
            print(f"Error calculating environmental impact: {str(e)}")
            return 0.0

    def _generate_yield_recommendations(self, env_factors, current_yield):
        """Generate recommendations based on environmental factors and current yield"""
        recommendations = []
        
        try:
            # Check tree age
            if 'Tree Age (years)' in env_factors:
                age = float(env_factors['Tree Age (years)'])
                if age < 3:
                    recommendations.append(("Tree Age", "Tree is too young for optimal yield. Consider waiting for maturity."))
                elif age > 10:
                    recommendations.append(("Tree Age", "Tree is aging. Consider replacement or intensive care."))
            
            # Check pH level
            if 'pH Level' in env_factors:
                ph = float(env_factors['pH Level'])
                if ph < 6.0:
                    recommendations.append(("pH Level", "Soil is too acidic. Consider adding lime to raise pH."))
                elif ph > 7.0:
                    recommendations.append(("pH Level", "Soil is too alkaline. Consider adding sulfur to lower pH."))
            
            # Check irrigation
            if 'Irrigation (mm)' in env_factors:
                irrigation = float(env_factors['Irrigation (mm)'])
                if irrigation < 30:
                    recommendations.append(("Irrigation", "Increase irrigation to optimal levels (30-40mm)."))
                elif irrigation > 40:
                    recommendations.append(("Irrigation", "Reduce irrigation to prevent waterlogging."))
            
            # Check fertilizer
            if 'Fertilizer Applied (kg)' in env_factors:
                fertilizer = float(env_factors['Fertilizer Applied (kg)'])
                if fertilizer < 3.0:
                    recommendations.append(("Fertilizer", "Increase fertilizer application to optimal levels (3-4kg)."))
                elif fertilizer > 4.0:
                    recommendations.append(("Fertilizer", "Reduce fertilizer application to prevent nutrient burn."))
            
            # Add yield-specific recommendations
            if current_yield is not None:
                if current_yield < 20:
                    recommendations.append(("Yield", "Current yield is below optimal. Review all environmental factors."))
                elif current_yield > 30:
                    recommendations.append(("Yield", "Excellent yield! Maintain current practices."))
            
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
        
        return recommendations

    def generate_yield_report(self, tree_data, tree_id=None, tree_species=None):
        """Generate an HTML report for yield analysis"""
        try:
            # Get the latest data point
            latest_data = tree_data.sort_values('Measurement Date').iloc[-1:].reset_index(drop=True)
            
            # Calculate statistics
            avg_yield = tree_data['Yield (kg)'].mean()
            max_yield = tree_data['Yield (kg)'].max()
            min_yield = tree_data['Yield (kg)'].min()
            
            # Generate HTML report
            html = f"""
            <html>
            <head>
                <title>Yield Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .container {{ max-width: 800px; margin: 0 auto; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                    table {{ width: 100%; border-collapse: collapse; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f0f0f0; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Yield Analysis Report</h1>
                        <p>Tree ID: {tree_id if tree_id else 'N/A'}</p>
                        <p>Species: {tree_species if tree_species else tree_data['Tree Species'].iloc[0]}</p>
                        <p>Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                    
                    <div class="section">
                        <h2>Current Yield Status</h2>
                        <table>
                            <tr><th>Metric</th><th>Value</th></tr>
                            <tr><td>Current Yield</td><td>{latest_data['Yield (kg)'].iloc[0]:.2f} kg</td></tr>
                            <tr><td>Fruit Count</td><td>{latest_data['Fruit Count'].iloc[0]}</td></tr>
                            <tr><td>Average Fruit Weight</td><td>{latest_data['Average Fruit Weight (g)'].iloc[0]:.2f} g</td></tr>
                            <tr><td>Fruit Size</td><td>{latest_data['Fruit Size (cm)'].iloc[0]:.2f} cm</td></tr>
                            <tr><td>Fruit Color</td><td>{latest_data['Fruit Color'].iloc[0]}</td></tr>
                            <tr><td>Harvest Date</td><td>{latest_data['Harvest Date'].iloc[0]}</td></tr>
                        </table>
                    </div>
                    
                    <div class="section">
                        <h2>Overall Yield Statistics</h2>
                        <table>
                            <tr><th>Metric</th><th>Value</th></tr>
                            <tr><td>Average Yield</td><td>{avg_yield:.2f} kg</td></tr>
                            <tr><td>Maximum Yield</td><td>{max_yield:.2f} kg</td></tr>
                            <tr><td>Minimum Yield</td><td>{min_yield:.2f} kg</td></tr>
                        </table>
                    </div>
                    
                    <div class="section">
                        <h2>Environmental Factors</h2>
                        <table>
                            <tr><th>Factor</th><th>Value</th></tr>
                            <tr><td>Tree Age</td><td>{latest_data['Tree Age (years)'].iloc[0]} years</td></tr>
                            <tr><td>pH Level</td><td>{latest_data['pH Level'].iloc[0]}</td></tr>
                            <tr><td>Soil Type</td><td>{latest_data['Soil Type'].iloc[0]}</td></tr>
                            <tr><td>Irrigation</td><td>{latest_data['Irrigation (mm)'].iloc[0]} mm</td></tr>
                            <tr><td>Fertilizer</td><td>{latest_data['Fertilizer Applied (kg)'].iloc[0]} kg</td></tr>
                        </table>
                    </div>
                </div>
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            print(f"Error generating yield report: {str(e)}")
            return "<html><body><h1>Error generating report</h1></body></html>"

def main():
    analyzer = TreeYieldAnalyzer()
    
    print("SKUAST Fruit Tree Yield Analysis System")
    print("--------------------------------------")
    
    while True:
        print("\nMain Menu:")
        print("1. Load yield dataset")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1-2): ")
        
        if choice == '1':
            file_path = input("Enter the path to the yield analysis data file: ")
            analyzer.load_data(file_path)
            if analyzer.data is not None:
                while True:
                    print("\nAvailable Tree IDs:")
                    for species in analyzer.data['Tree Species'].unique():
                        tree_ids = analyzer.data[analyzer.data['Tree Species'] == species]['Tree ID'].unique()
                        print(f"{species}: {', '.join(tree_ids)}")

                    tree_id_input = input("\nEnter Tree ID to analyze yield (or type 'menu' to return to main menu, 'exit' to quit): ").strip().upper()
                    
                    if tree_id_input == 'MENU':
                        break
                    elif tree_id_input == 'EXIT':
                        print("Thank you for using the SKUAST Fruit Tree Yield Analysis System!")
                        return
                    
                    if tree_id_input in analyzer.data['Tree ID'].unique():
                        print(f"\nAnalyzing yield for Tree {tree_id_input}...")
                        yield_analysis_results = analyzer.analyze_yield(tree_id=tree_id_input)
                        print("\nYield Analysis Results:")
                        print(tabulate(yield_analysis_results.items(), headers=["Key", "Value"]))
                    else:
                        print(f"Invalid Tree ID '{tree_id_input}'. Please use one of the available IDs shown above.")
        
        elif choice == '2':
            print("Thank you for using the SKUAST Fruit Tree Yield Analysis System!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 