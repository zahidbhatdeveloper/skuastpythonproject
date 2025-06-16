import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from tabulate import tabulate
warnings.filterwarnings('ignore')

# Set the style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

# Import the dedicated TreeYieldAnalyzer
from tree_yield_analysis import TreeYieldAnalyzer

class FruitTreeAnalyzer:
    def __init__(self):
        self.data = None
        self.data_dir = 'data'
        self.yield_model = None
        self.scaler = StandardScaler()
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def load_data(self, file_path):
        """Load chemical data from CSV file"""
        try:
            self.data = pd.read_csv(file_path)
            print(f"Data loaded successfully with {len(self.data)} records")
            # Convert date column to datetime
            self.data['Measurement Date'] = pd.to_datetime(self.data['Measurement Date'])
            # Print available tree IDs
            print("\nAvailable Tree IDs:")
            for species in self.data['Tree Species'].unique():
                tree_ids = self.data[self.data['Tree Species'] == species]['Tree ID'].unique()
                print(f"{species}: {', '.join(tree_ids)}")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
    
    def analyze_chemical_compounds(self, tree_id=None, tree_species=None):
        """Analyze chemical compounds for specific tree ID or species"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        if tree_id:
            species_data = self.data[self.data['Tree ID'] == tree_id]
        elif tree_species:
            species_data = self.data[self.data['Tree Species'] == tree_species]
        else:
            species_data = self.data
        
        # Calculate basic statistics
        stats_summary = species_data.groupby('Chemical Compound').agg({
            'Concentration': ['mean', 'std', 'min', 'max', 'median'],
            'Previous Dosage': ['mean', 'std', 'min', 'max', 'median']
        }).round(3)
        
        # Print detailed analysis tables
        print("\nDetailed Chemical Analysis:")
        print("=" * 80)
        print(tabulate(stats_summary, headers='keys', tablefmt='grid'))
        
        # Calculate percentage composition
        chemical_means = species_data.groupby('Chemical Compound')['Concentration'].mean()
        total_concentration = chemical_means.sum()
        composition_percent = (chemical_means / total_concentration * 100).round(2)
        
        print("\nPercentage Composition:")
        print("=" * 80)
        composition_data = [[compound, f"{percentage}%"] for compound, percentage in composition_percent.items()]
        print(tabulate(composition_data, headers=["Compound", "Percentage"], tablefmt='grid'))
        
        # Generate detailed chemical compound analysis with status
        print("\nDetailed Chemical Compound Evaluation:")
        print("=" * 80)
        chemical_compound_data = []
        for compound in species_data['Chemical Compound'].unique():
            compound_data = species_data[species_data['Chemical Compound'] == compound]
            mean_conc = compound_data['Concentration'].mean()
            min_conc = compound_data['Concentration'].min()
            max_conc = compound_data['Concentration'].max()
            std_dev = compound_data['Concentration'].std()
            cv = (std_dev / mean_conc * 100) if mean_conc != 0 and not np.isnan(std_dev) else float('nan')
            
            # Get previous dosage information
            prev_dosage_mean = compound_data['Previous Dosage'].mean()
            prev_dosage_std = compound_data['Previous Dosage'].std()
            
            # Define optimal ranges based on compound type
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

            chemical_compound_data.append([
                compound,
                f"{mean_conc:.3f}",
                f"{min_conc:.3f} - {max_conc:.3f}",
                f"{std_dev:.3f}" if not np.isnan(std_dev) else "NaN",
                f"{cv:.1f}%" if not np.isnan(cv) else "NaN",
                f"{prev_dosage_mean:.3f}",
                f"{prev_dosage_std:.3f}",
                optimal_range,
                status
            ])
        
        chem_detail_table = tabulate(
            chemical_compound_data,
            headers=["Compound", "Avg Conc", "Range", "Std Dev", "CV%", "Prev Dosage Avg", "Prev Dosage Std", "Optimal Range", "Status"],
            tablefmt='grid'
        )
        print(chem_detail_table)
        
        # Calculate overall health score
        health_scores = []
        for compound in species_data['Chemical Compound'].unique():
            compound_data = species_data[species_data['Chemical Compound'] == compound]
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
        
        print("\nOverall Chemical Health Assessment:")
        print("=" * 80)
        assessment_data = [
            ["Overall Chemical Health Score", f"{overall_score:.1f}%"],
            ["Status", "Excellent - All chemical parameters are within optimal ranges" if overall_score >= 90 else 
                      ("Good - Most chemical parameters are within optimal ranges" if overall_score >= 75 else 
                      ("Fair - Some chemical parameters need attention" if overall_score >= 60 else 
                      "Poor - Multiple chemical parameters need attention"))]
        ]
        assessment_table = tabulate(assessment_data, headers=["Assessment", "Details"], tablefmt='grid')
        print(assessment_table)
        
        # Generate recommendations
        print("\nRecommendations:")
        print("=" * 80)
        recommendations_data = []
        for compound in species_data['Chemical Compound'].unique():
            compound_data = species_data[species_data['Chemical Compound'] == compound]
            mean_conc = compound_data['Concentration'].mean()
            
            if compound == 'Sugars' and not (2.5 <= mean_conc <= 4.0):
                recommendations_data.append([compound, f"Sugar levels are {'low' if mean_conc < 2.5 else 'high'}. Consider adjusting fertilization and irrigation."])
            elif compound == 'Malic Acid' and not (0.8 <= mean_conc <= 1.5):
                recommendations_data.append([compound, f"Malic acid levels are {'low' if mean_conc < 0.8 else 'high'}. Review fruit maturity and harvest timing."])
            elif compound == 'Vitamin C' and not (0.4 <= mean_conc <= 0.8):
                recommendations_data.append([compound, f"Vitamin C levels are {'low' if mean_conc < 0.4 else 'high'}. Check sunlight exposure and nutrient balance."])
            elif compound == 'Chlorophyll' and not (2.0 <= mean_conc <= 3.0):
                recommendations_data.append([compound, f"Chlorophyll levels are {'low' if mean_conc < 2.0 else 'high'}. Review leaf health and nutrient uptake."])
            elif compound == 'Anthocyanins' and not (3.5 <= mean_conc <= 4.5):
                recommendations_data.append([compound, f"Anthocyanin levels are {'low' if mean_conc < 3.5 else 'high'}. Check light exposure and temperature conditions."])
            elif compound == 'Pectin' and not (1.2 <= mean_conc <= 1.8):
                recommendations_data.append([compound, f"Pectin levels are {'low' if mean_conc < 1.2 else 'high'}. Review fruit development stage and harvest timing."])
            elif compound == 'Actinidin' and not (0.8 <= mean_conc <= 1.2):
                recommendations_data.append([compound, f"Actinidin levels are {'low' if mean_conc < 0.8 else 'high'}. Check fruit ripeness and storage conditions."])
            elif compound == 'Fiber' and not (1.8 <= mean_conc <= 2.4):
                recommendations_data.append([compound, f"Fiber levels are {'low' if mean_conc < 1.8 else 'high'}. Review fruit development and harvest timing."])

        if recommendations_data:
            recommendations_table = tabulate(recommendations_data, headers=["Compound", "Recommendation"], tablefmt='grid')
            print(recommendations_table)
        else:
            print("No specific recommendations based on optimal ranges.")
        
        # Create and show visualizations
        self.create_chemical_composition_plot(species_data, tree_id or tree_species)
        self.create_correlation_heatmap(species_data, tree_id or tree_species)
        self.create_trend_analysis(species_data, tree_id or tree_species)
        self.create_environmental_impact_plot(species_data, tree_id or tree_species)
        
        return stats_summary
    
    def analyze_fruit_stage_impact(self, tree_id=None):
        """Analyze the impact of fruit development stage on chemical composition"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        if tree_id:
            data_subset = self.data[self.data['Tree ID'] == tree_id]
        else:
            data_subset = self.data
        
        # Group by fruit stage and calculate mean concentrations
        stage_impact = data_subset.groupby(['Fruit Stage', 'Chemical Compound'])['Concentration'].mean().unstack()
        
        # Create heatmap
        plt.figure(figsize=(15, 8))
        sns.heatmap(stage_impact, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title('Impact of Fruit Development Stage on Chemical Composition')
        plt.tight_layout()
        plt.show()
        
        return stage_impact
    
    def compare_fruit_trees(self, tree_ids=None):
        """Compare chemical composition between different fruit trees"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        if tree_ids:
            data_subset = self.data[self.data['Tree ID'].isin(tree_ids)]
        else:
            data_subset = self.data
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        # Define important compounds to compare
        compounds = ['Sugars', 'Proteins', 'Vitamin C', 'Chlorophyll', 
                    'Starch', 'Cellulose', 'Anthocyanins', 'Malic Acid', 'Pectin']
        
        for idx, compound in enumerate(compounds):
            row = idx // 3
            col = idx % 3
            compound_data = data_subset[data_subset['Chemical Compound'] == compound]
            if not compound_data.empty:
                sns.barplot(data=compound_data, x='Tree ID', y='Concentration', ax=axes[row, col])
                axes[row, col].set_title(f'{compound} Content')
                axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_soil_impact(self, tree_id=None):
        """Analyze the impact of soil type on chemical composition"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        if tree_id:
            data_subset = self.data[self.data['Tree ID'] == tree_id]
        else:
            data_subset = self.data
        
        # Group by soil type and calculate mean concentrations
        soil_impact = data_subset.groupby(['Soil Type', 'Chemical Compound'])['Concentration'].mean().unstack()
        
        # Create heatmap
        plt.figure(figsize=(15, 8))
        sns.heatmap(soil_impact, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title('Impact of Soil Type on Chemical Composition')
        plt.tight_layout()
        plt.show()
        
        return soil_impact
    
    def generate_tree_report(self, tree_id=None, tree_species=None):
        """Generate a comprehensive chemical analysis report for a specific tree"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        if tree_id:
            species_data = self.data[self.data['Tree ID'] == tree_id]
            report_name = f"{tree_id}"
        elif tree_species:
            species_data = self.data[self.data['Tree Species'] == tree_species]
            report_name = tree_species
        else:
            print("Please provide either a tree ID or species name")
            return
        
        if species_data.empty:
            print(f"No data found for {report_name}")
            return
            
        self.generate_report_for_data(species_data, report_name)
    
    def analyze_tree_growth(self, tree_id):
        """Analyze the growth and chemical changes over time for a specific tree"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        tree_data = self.data[self.data['Tree ID'] == tree_id]
        if tree_data.empty:
            print(f"No data found for tree {tree_id}")
            return
        
        # Sort data by date
        tree_data = tree_data.sort_values('Measurement Date')
        
        # Create line plots for each chemical compound
        compounds = tree_data['Chemical Compound'].unique()
        fig = make_subplots(rows=len(compounds), cols=1,
                           subplot_titles=[f"{compound} Over Time" for compound in compounds])
        
        for idx, compound in enumerate(compounds, 1):
            compound_data = tree_data[tree_data['Chemical Compound'] == compound]
            fig.add_trace(
                go.Scatter(
                    x=compound_data['Measurement Date'],
                    y=compound_data['Concentration'],
                    mode='lines+markers',
                    name=compound
                ),
                row=idx, col=1
            )
        
        fig.update_layout(
            height=300 * len(compounds),
            title_text=f"Chemical Compound Trends for Tree {tree_id}",
            showlegend=True
        )
        
        fig.show()
        
        # Calculate growth rates
        growth_rates = {}
        for compound in compounds:
            compound_data = tree_data[tree_data['Chemical Compound'] == compound]
            if len(compound_data) > 1:
                first_measurement = compound_data.iloc[0]
                last_measurement = compound_data.iloc[-1]
                days_diff = (last_measurement['Measurement Date'] - first_measurement['Measurement Date']).days
                if days_diff > 0:
                    growth_rate = (last_measurement['Concentration'] - first_measurement['Concentration']) / days_diff
                    growth_rates[compound] = growth_rate
        
        print("\nGrowth Rates (per day):")
        for compound, rate in growth_rates.items():
            print(f"{compound}: {rate:.4f}")
    
    def create_chemical_composition_plot(self, data_subset, species_name):
        """Create an interactive chemical composition plot using Plotly"""
        chemical_means = data_subset.groupby('Chemical Compound')['Concentration'].mean()
        
        # Get tree species from data
        tree_species = data_subset['Tree Species'].iloc[0]
        tree_id = data_subset['Tree ID'].iloc[0] if 'Tree ID' in data_subset.columns else None
        
        # Create a more detailed pie chart
        fig = px.pie(
            values=chemical_means.values,
            names=chemical_means.index,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3,
            template='plotly_white',
            width=900,
            height=700
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label+value',
            hovertemplate="<b>%{label}</b><br>" +
                         "Concentration: %{value:.2f}<br>" +
                         "Percentage: %{percent}<br>" +
                         "<extra></extra>"
        )
        
        # Add annotations for tree species and ID
        title = f"{tree_species} Tree Analysis"
        if tree_id:
            title += f" (ID: {tree_id})"
            
        annotations = [
            dict(
                text=title,
                showarrow=False,
                x=0.5,
                y=0.95,
                font_size=24,
                font_color='#2c3e50'
            ),
            dict(
                text=f"Total Compounds: {len(chemical_means)}",
                showarrow=False,
                x=0.5,
                y=0.5,
                font_size=20,
                font_color='#7f8c8d'
            )
        ]

        fig.update_layout(
            title_x=0.5,
            title_font_size=24,
            title_font_color='#2c3e50',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            annotations=annotations,
            margin=dict(b=200, l=0, r=0, t=50)
        )
        return fig

    def create_correlation_heatmap(self, data_subset, species_name):
        """Create a detailed correlation heatmap for chemical compounds"""
        # Get tree species from data
        tree_species = data_subset['Tree Species'].iloc[0]
        tree_id = data_subset['Tree ID'].iloc[0] if 'Tree ID' in data_subset.columns else None
        
        # Pivot the data to get compounds as columns
        pivot_data = data_subset.pivot_table(
            index='Measurement Date',
            columns='Chemical Compound',
            values='Concentration'
        )
        
        # Calculate correlation matrix
        corr_matrix = pivot_data.corr()
        
        # Create a more detailed heatmap
        fig = px.imshow(
            corr_matrix,
            title=f'Chemical Compound Correlations for {tree_species} Tree' + (f" (ID: {tree_id})" if tree_id else ""),
            color_continuous_scale='RdBu_r',
            aspect='auto',
            template='plotly_white'
        )
        
        fig.update_layout(
            title_x=0.5,
            title_font_size=24,
            title_font_color='#2c3e50',
            xaxis_title="Chemical Compounds",
            yaxis_title="Chemical Compounds",
            coloraxis_colorbar=dict(
                title=dict(
                    text="Correlation",
                    side="right"
                )
            ),
            margin=dict(l=0, r=0, t=50, b=0)
        )
        return fig

    def create_trend_analysis(self, data_subset, species_name):
        """Create a detailed trend analysis plot for chemical compounds over time"""
        # Get tree species from data
        tree_species = data_subset['Tree Species'].iloc[0]
        tree_id = data_subset['Tree ID'].iloc[0] if 'Tree ID' in data_subset.columns else None
        
        # Create a more detailed line plot
        fig = px.line(
            data_subset,
            x='Measurement Date',
            y='Concentration',
            color='Chemical Compound',
            title=f'Chemical Compound Trends Over Time for {tree_species} Tree' + (f" (ID: {tree_id})" if tree_id else ""),
            markers=True,
            template='plotly_white'
        )
        
        # Add trend lines for each compound
        for compound in data_subset['Chemical Compound'].unique():
            compound_data = data_subset[data_subset['Chemical Compound'] == compound]
            x = range(len(compound_data))
            y = compound_data['Concentration']
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            
            fig.add_scatter(
                x=compound_data['Measurement Date'],
                y=p(x),
                mode='lines',
                line=dict(dash='dash', width=1),
                name=f'{compound} Trend',
                showlegend=False
            )
        
        fig.update_layout(
            title_x=0.5,
            title_font_size=24,
            title_font_color='#2c3e50',
            xaxis_title="Date",
            yaxis_title="Concentration",
            legend_title="Chemical Compounds",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add range slider
        fig.update_xaxes(rangeslider_visible=True)
        
        return fig

    def create_environmental_impact_plot(self, data_subset, species_name):
        """Create a detailed plot showing environmental factors impact"""
        # Get tree species from data
        tree_species = data_subset['Tree Species'].iloc[0]
        tree_id = data_subset['Tree ID'].iloc[0] if 'Tree ID' in data_subset.columns else None
        
        # Create subplots for different environmental factors
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'pH Level Impact',
                'Tree Age Impact',
                'Soil Type Impact',
                'Fruit Stage Impact'
            )
        )
        
        # pH Level Impact
        fig.add_trace(
            go.Scatter(
                x=data_subset['pH Level'],
                y=data_subset['Concentration'],
                mode='markers',
                marker=dict(
                    color=data_subset['Concentration'],
                    colorscale='Viridis',
                    showscale=True
                ),
                name='pH Impact'
            ),
            row=1, col=1
        )
        
        # Tree Age Impact
        fig.add_trace(
            go.Scatter(
                x=data_subset['Tree Age (years)'],
                y=data_subset['Concentration'],
                mode='markers',
                marker=dict(
                    color=data_subset['Concentration'],
                    colorscale='Viridis',
                    showscale=True
                ),
                name='Age Impact'
            ),
            row=1, col=2
        )
        
        # Soil Type Impact
        soil_means = data_subset.groupby('Soil Type')['Concentration'].mean()
        fig.add_trace(
            go.Bar(
                x=soil_means.index,
                y=soil_means.values,
                name='Soil Impact'
            ),
            row=2, col=1
        )
        
        # Fruit Stage Impact
        stage_means = data_subset.groupby('Fruit Stage')['Concentration'].mean()
        fig.add_trace(
            go.Bar(
                x=stage_means.index,
                y=stage_means.values,
                name='Stage Impact'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text=f'Environmental Factors Impact Analysis for {tree_species} Tree' + (f" (ID: {tree_id})" if tree_id else ""),
            title_x=0.5,
            title_font_size=24,
            title_font_color='#2c3e50',
            showlegend=True,
            template='plotly_white',
            height=800,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        return fig

    def generate_report_for_data(self, data_subset, species_name):
        """Generate a comprehensive report with enhanced visualizations"""
        # Get tree species and ID
        tree_species = data_subset['Tree Species'].iloc[0]
        tree_id = data_subset['Tree ID'].iloc[0] if 'Tree ID' in data_subset.columns else None
        
        report_title = f"{tree_species} Tree Analysis"
        if tree_id:
            report_title += f" (ID: {tree_id})"
            
        print(f"\n{'='*20} Generating Chemical Analysis Report for {report_title} {'='*20}")
        
        report_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Chemical Analysis Report for {report_title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; margin: 20px; background-color: #f8f9fa; color: #333; }}
        h1, h2, h3 {{ color: #0056b3; margin-top: 20px; margin-bottom: 10px; }}
        h1 {{ text-align: center; color: #003970; margin-bottom: 30px; }}
        .report-section {{ margin-bottom: 40px; border: 1px solid #dee2e6; padding: 20px; border-radius: 8px; background-color: #fff; box-shadow: 0 2px 4px rgba(0,0,0,.05); }}
        .report-section h2 {{ border-bottom: 2px solid #007bff; padding-bottom: 10px; color: #007bff; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; box-shadow: 0 1px 3px rgba(0,0,0,.02); }}
        th, td {{ border: 1px solid #dee2e6; padding: 12px 15px; text-align: left; }}
        th {{ background-color: #007bff; color: white; font-weight: bold; }}
        tbody tr:nth-child(even) {{ background-color: #f2f2f2; }}
        tbody tr:hover {{ background-color: #e9ecef; }}
        .plotly-graph-div {{ margin-top: 20px; border: none; border-radius: 5px; padding: 0px; background-color: #fff; }}
        .tree-info {{ text-align: center; margin-bottom: 30px; padding: 15px; background-color: #e3f2fd; border-radius: 8px; }}
        .tree-info h2 {{ color: #1976d2; margin: 0; }}
        .tree-info p {{ margin: 5px 0; color: #455a64; }}
    </style>
</head>
<body>
    <h1>Chemical Analysis Report</h1>
    
    <div class="tree-info">
        <h2>{report_title}</h2>
        <p>Location: {data_subset['Location'].iloc[0]}</p>
        <p>Tree Age: {data_subset['Tree Age (years)'].iloc[0]} years</p>
        <p>Soil Type: {data_subset['Soil Type'].iloc[0]}</p>
        <p>Current Fruit Stage: {data_subset['Fruit Stage'].iloc[0]}</p>
    </div>
    
"""

        # Create interactive chemical composition plot and add to HTML
        comp_fig = self.create_chemical_composition_plot(data_subset, species_name)
        report_html += f"""
    <div class="report-section">
        <h2>Chemical Composition Analysis</h2>
        {comp_fig.to_html(full_html=False, include_plotlyjs='cdn')}
"""

        # Generate and add Chemical Compound Statistics table to HTML
        stats_summary = data_subset.groupby('Chemical Compound')['Concentration'].agg([
            ('Mean', 'mean'),
            ('Std Dev', 'std'),
            ('Min', 'min'),
            ('Max', 'max'),
            ('Median', 'median'),
            ('CV%', lambda x: (x.std() / x.mean() * 100))  # Coefficient of Variation
        ]).round(3)
        stats_data = stats_summary.reset_index().values.tolist()
        stats_table_html = tabulate(stats_data, headers=stats_summary.reset_index().columns.tolist(), tablefmt="html")
        report_html += f"""
        <h3>Chemical Compound Statistics</h3>
        {stats_table_html}
"""

        # Generate and add Chemical Composition Analysis table to HTML
        chemical_means = data_subset.groupby('Chemical Compound')['Concentration'].mean()
        total_concentration = chemical_means.sum()
        composition_percent = (chemical_means / total_concentration * 100).round(2)
        composition_data = []
        for compound, percentage in composition_percent.items():
            composition_data.append([compound, f"{percentage}%"])
        composition_table_html = tabulate(composition_data, headers=["Compound", "Percentage"], tablefmt="html")
        report_html += f"""
        <h3>Percentage Composition</h3>
        {composition_table_html}
"""

        # Create correlation heatmap and add to HTML
        corr_fig = self.create_correlation_heatmap(data_subset, species_name)
        report_html += f"""
    </div>
    <div class="report-section">
        <h2>Chemical Compound Correlations</h2>
        {corr_fig.to_html(full_html=False, include_plotlyjs='cdn')}
"""
        # Note: Adding detailed correlation text/table below heatmap in HTML is more complex with Plotly's to_html. 
        # The most practical approach is to rely on hover information in the interactive graph or include a separate text block.
        # We will rely on hover for this version.

        # Create environmental impact plot and add to HTML
        env_fig = self.create_environmental_impact_plot(data_subset, species_name)
        report_html += f"""
    </div>
    <div class="report-section">
        <h2>Environmental Factors Analysis</h2>
        {env_fig.to_html(full_html=False, include_plotlyjs='cdn')}
"""

        # Generate and add Environmental Factors Analysis table to HTML
        env_factors = {
            'pH Level': {
                'Mean': data_subset['pH Level'].mean(),
                'Range': f"{data_subset['pH Level'].min():.2f} - {data_subset['pH Level'].max():.2f}",
                'Optimal Range': '6.0 - 7.0',
                'Status': 'Optimal' if 6.0 <= data_subset['pH Level'].mean() <= 7.0 else 'Sub-optimal'
            },
            'Tree Age': {
                'Mean': f"{data_subset['Tree Age (years)'].mean():.1f} years",
                'Range': f"{data_subset['Tree Age (years)'].min():.1f} - {data_subset['Tree Age (years)'].max():.1f} years",
                'Growth Stage': 'Mature' if data_subset['Tree Age (years)'].mean() > 5 else 'Young'
            },
            'Soil Type': {
                'Type': data_subset['Soil Type'].mode()[0],
                'Distribution': data_subset['Soil Type'].value_counts().to_dict()
            },
            'Fruit Stage': {
                'Current Stage': data_subset['Fruit Stage'].mode()[0],
                'Stage Distribution': data_subset['Fruit Stage'].value_counts().to_dict()
            },
            'Location': {
                'Primary Location': data_subset['Location'].mode()[0],
                'Location Distribution': data_subset['Location'].value_counts().to_dict()
            }
        }
        environmental_data = []
        for factor, details in env_factors.items():
            detail_strings = []
            for key, value in details.items():
                if isinstance(value, dict):
                    dist_str = ", ".join([f"{k}: {v}" for k, v in value.items()])
                    detail_strings.append(f"{key}: {dist_str}")
                else:
                    detail_strings.append(f"{key}: {value}")
            # Pass the list of detail strings directly
            # environmental_data.append([factor, detail_strings])
            # Revert to joining with <br> for compatibility
            environmental_data.append([factor, "<br>".join(detail_strings)])

        # Use list_fmt="html" to render lists within cells with line breaks
        # env_table_html = tabulate(environmental_data, headers=["Environmental Factor", "Details"], tablefmt="html", listfmt="html")

        # Revert to simple html format and manually unescape <br>
        env_table_html = tabulate(environmental_data, headers=["Environmental Factor", "Details"], tablefmt="html")

        # Manually unescape <br> tags in the generated HTML table
        # This is necessary because tabulate escapes HTML by default in 'html' format
        env_table_html = env_table_html.replace('&lt;br&gt;', '<br>')

        report_html += f"""
        <h3>Environmental Factors Details</h3>
        {env_table_html}
"""

        # --- Yield Information Section ---
        # Get the latest measurement for yield-related columns
        yield_columns = [
            'Yield (kg)', 'Fruit Count', 'Average Fruit Weight (g)', 'Fruit Size (cm)', 'Fruit Color', 'Harvest Date'
        ]
        latest_row = data_subset.sort_values('Measurement Date').iloc[-1]
        yield_info_data = []
        for col in yield_columns:
            if col in latest_row:
                yield_info_data.append([col, latest_row[col]])
        yield_info_table_html = tabulate(yield_info_data, headers=["Metric", "Value"], tablefmt="html")
        report_html += f"""
    <div class="report-section">
        <h2>Yield Information</h2>
        {yield_info_table_html}
    </div>
"""

        # Generate and add Detailed Chemical Compound Analysis table to HTML
        chemical_compound_data = []
        for compound in data_subset['Chemical Compound'].unique():
            compound_data = data_subset[data_subset['Chemical Compound'] == compound]
            mean_conc = compound_data['Concentration'].mean()
            min_conc = compound_data['Concentration'].min()
            max_conc = compound_data['Concentration'].max()
            std_dev = compound_data['Concentration'].std()
            cv = (std_dev / mean_conc * 100) if mean_conc != 0 and not np.isnan(std_dev) else float('nan')
            
            # Get previous dosage information
            prev_dosage_mean = compound_data['Previous Dosage'].mean()
            prev_dosage_std = compound_data['Previous Dosage'].std()
            
            # Define optimal ranges based on compound type
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

            chemical_compound_data.append([
                compound,
                f"{mean_conc:.3f}",
                f"{min_conc:.3f} - {max_conc:.3f}",
                f"{std_dev:.3f}" if not np.isnan(std_dev) else "NaN",
                f"{cv:.1f}%" if not np.isnan(cv) else "NaN",
                f"{prev_dosage_mean:.3f}",
                f"{prev_dosage_std:.3f}",
                optimal_range,
                status
            ])
        
        chem_detail_table_html = tabulate(
            chemical_compound_data,
            headers=["Compound", "Avg Conc", "Range", "Std Dev", "CV%", "Prev Dosage Avg", "Prev Dosage Std", "Optimal Range", "Status"],
            tablefmt="html"
        )
        report_html += f"""
        <h3>Detailed Chemical Compound Analysis</h3>
        {chem_detail_table_html}
"""

        # Generate and add Overall Assessment table to HTML
        health_scores = []
        for compound in data_subset['Chemical Compound'].unique():
            compound_data = data_subset[data_subset['Chemical Compound'] == compound]
            mean_conc = compound_data['Concentration'].mean()
            
            score = 1 # Default score if not a key compound
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
        
        assessment_data = [
            ["Overall Chemical Health Score", f"{overall_score:.1f}%"],
            ["Status", "Excellent - All chemical parameters are within optimal ranges" if overall_score >= 90 else ("Good - Most chemical parameters are within optimal ranges" if overall_score >= 75 else ("Fair - Some chemical parameters need attention" if overall_score >= 60 else "Poor - Multiple chemical parameters need attention"))]
        ]
        assessment_table_html = tabulate(assessment_data, headers=["Assessment", "Details"], tablefmt="html")
        report_html += f"""
        <h3>Overall Chemical Assessment</h3>
        {assessment_table_html}
"""

        # Generate and add Recommendations table to HTML
        recommendations_data = []
        for compound in data_subset['Chemical Compound'].unique():
            compound_data = data_subset[data_subset['Chemical Compound'] == compound]
            mean_conc = compound_data['Concentration'].mean()
            
            if compound == 'Sugars' and not (2.5 <= mean_conc <= 4.0):
                recommendations_data.append([compound, f"Sugar levels are {'low' if mean_conc < 2.5 else 'high'}. Consider adjusting fertilization and irrigation."])
            elif compound == 'Malic Acid' and not (0.8 <= mean_conc <= 1.5):
                recommendations_data.append([compound, f"Malic acid levels are {'low' if mean_conc < 0.8 else 'high'}. Review fruit maturity and harvest timing."])
            elif compound == 'Vitamin C' and not (0.4 <= mean_conc <= 0.8):
                recommendations_data.append([compound, f"Vitamin C levels are {'low' if mean_conc < 0.4 else 'high'}. Check sunlight exposure and nutrient balance."])
            elif compound == 'Chlorophyll' and not (2.0 <= mean_conc <= 3.0):
                recommendations_data.append([compound, f"Chlorophyll levels are {'low' if mean_conc < 2.0 else 'high'}. Review leaf health and nutrient uptake."])
            elif compound == 'Anthocyanins' and not (3.5 <= mean_conc <= 4.5):
                recommendations_data.append([compound, f"Anthocyanin levels are {'low' if mean_conc < 3.5 else 'high'}. Check light exposure and temperature conditions."])
            elif compound == 'Pectin' and not (1.2 <= mean_conc <= 1.8):
                recommendations_data.append([compound, f"Pectin levels are {'low' if mean_conc < 1.2 else 'high'}. Review fruit development stage and harvest timing."])
            elif compound == 'Actinidin' and not (0.8 <= mean_conc <= 1.2):
                recommendations_data.append([compound, f"Actinidin levels are {'low' if mean_conc < 0.8 else 'high'}. Check fruit ripeness and storage conditions."])
            elif compound == 'Fiber' and not (1.8 <= mean_conc <= 2.4):
                recommendations_data.append([compound, f"Fiber levels are {'low' if mean_conc < 1.8 else 'high'}. Review fruit development and harvest timing."])

        if recommendations_data:
            recommendations_table_html = tabulate(recommendations_data, headers=["Compound", "Recommendation"], tablefmt="html")
            report_html += f"""
        <h3>Recommendations</h3>
        {recommendations_table_html}
"""
        else:
            report_html += f"""
        <h3>Recommendations</h3>
        <p>No specific recommendations based on optimal ranges.</p>
"""

        # Close the main HTML body and document
        report_html += f"""
    </div>
</body>
</html>
"""

        # Save the HTML report to a file
        report_file_path = os.path.join(self.data_dir, f'chemical_analysis_report_{species_name.lower().replace(" ", "_")}.html')
        with open(report_file_path, 'w') as f:
            f.write(report_html)

        print(f"Chemical analysis report saved to {report_file_path}")

        # Open the HTML report in the default web browser
        import webbrowser
        webbrowser.open(f'file://{os.path.abspath(report_file_path)}')

    def evaluate_tree_from_file(self, file_path):
        """Load and evaluate chemical data for a single tree from a CSV file"""
        try:
            single_tree_data = pd.read_csv(file_path)
            if single_tree_data.empty:
                print(f"Error: No data found in {file_path}")
                return
                
            # Get tree ID and species
            tree_id = single_tree_data['Tree ID'].iloc[0]
            tree_species = single_tree_data['Tree Species'].iloc[0]
            print(f"Evaluating data for {tree_species} (ID: {tree_id}) from {file_path}")
            
            # Generate report for this single tree's data
            self.generate_report_for_data(single_tree_data, f"{tree_species} (ID: {tree_id})")
            
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
        except KeyError as e:
            print(f"Error: Missing required column in CSV: {e}")
            print("Please ensure the CSV contains the following columns: Tree ID,Tree Species,Chemical Compound,Concentration,Measurement Date,Location,Season,Tree Age (years),pH Level,Soil Type,Fruit Stage")
        except Exception as e:
            print(f"Error evaluating data: {str(e)}")

    def generate_yield_report(self, tree_data, tree_id=None, tree_species=None):
        """Generate a dedicated yield analysis report"""
        latest_data = tree_data.sort_values('Measurement Date').groupby('Tree ID').last()
        
        # Generate HTML report for yield analysis
        report_title = f"Yield Analysis Report - {tree_id if tree_id else tree_species}"
        report_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report_title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; margin: 20px; background-color: #f8f9fa; color: #333; }}
        h1, h2, h3 {{ color: #0056b3; margin-top: 20px; margin-bottom: 10px; }}
        h1 {{ text-align: center; color: #003970; margin-bottom: 30px; }}
        .report-section {{ margin-bottom: 40px; border: 1px solid #dee2e6; padding: 20px; border-radius: 8px; background-color: #fff; box-shadow: 0 2px 4px rgba(0,0,0,.05); }}
        .report-section h2 {{ border-bottom: 2px solid #007bff; padding-bottom: 10px; color: #007bff; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; box-shadow: 0 1px 3px rgba(0,0,0,.02); }}
        th, td {{ border: 1px solid #dee2e6; padding: 12px 15px; text-align: left; }}
        th {{ background-color: #007bff; color: white; font-weight: bold; }}
        tbody tr:nth-child(even) {{ background-color: #f2f2f2; }}
        tbody tr:hover {{ background-color: #e9ecef; }}
        .plotly-graph-div {{ margin-top: 20px; border: none; border-radius: 5px; padding: 0px; background-color: #fff; }}
        .tree-info {{ text-align: center; margin-bottom: 30px; padding: 15px; background-color: #e3f2fd; border-radius: 8px; }}
        .tree-info h2 {{ color: #1976d2; margin: 0; }}
        .tree-info p {{ margin: 5px 0; color: #455a64; }}
        .yield-metrics {{ display: flex; justify-content: space-around; flex-wrap: wrap; margin: 20px 0; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,.1); margin: 10px; min-width: 200px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; margin: 10px 0; }}
        .metric-label {{ color: #666; }}
    </style>
</head>
<body>
    <h1>{report_title}</h1>
    
    <div class="tree-info">
        <h2>Tree Information</h2>
        <p>Tree ID: {tree_id if tree_id else 'N/A'}</p>
        <p>Species: {tree_data['Tree Species'].iloc[0]}</p>
        <p>Location: {tree_data['Location'].iloc[0]}</p>
        <p>Tree Age: {tree_data['Tree Age (years)'].iloc[0]} years</p>
    </div>
"""
        
        # 1. Current Yield Status
        report_html += """
    <div class="report-section">
        <h2>Current Yield Status</h2>
"""
        current_yield_data = []
        for _, row in latest_data.iterrows():
            current_yield_data.append(["Tree ID", row['Tree ID']])
            current_yield_data.append(["Species", row['Tree Species']])
            current_yield_data.append(["Current Yield", f"{row['Yield (kg)']:.2f} kg"])
            current_yield_data.append(["Fruit Count", row['Fruit Count']])
            current_yield_data.append(["Average Fruit Weight", f"{row['Average Fruit Weight (g)']:.1f} g"])
            current_yield_data.append(["Fruit Size", f"{row['Fruit Size (cm)']:.1f} cm"])
            current_yield_data.append(["Fruit Color", row['Fruit Color']])
            current_yield_data.append(["Harvest Date", row['Harvest Date']])
        
        current_yield_table = tabulate(current_yield_data, 
                                       headers=["Metric", "Value"], 
                                       tablefmt="html")
        report_html += f"{current_yield_table}"
        
        report_html += """
    </div>
"""
        
        # 2. Yield Trends Over Time
        report_html += """
    <div class="report-section">
        <h2>Yield Trends Over Time</h2>
"""
        yield_trends = tree_data.groupby('Measurement Date')['Yield (kg)'].mean()
        
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=yield_trends.index,
                y=yield_trends.values,
                mode='lines+markers',
                name='Yield Trend',
                line=dict(color='green', width=2)
            )
        )
        
        fig1.update_layout(
            title="Yield Progression Over Time",
            xaxis_title="Date",
            yaxis_title="Yield (kg)",
            template='plotly_white'
        )
        
        report_html += f"{fig1.to_html(full_html=False, include_plotlyjs='cdn')}"
        
        # Add yield trends table
        yield_trends_data = []
        for date, yield_value in yield_trends.items():
            yield_trends_data.append([date.strftime('%Y-%m-%d'), f"{yield_value:.2f} kg"])
        
        yield_trends_table = tabulate(yield_trends_data, 
                                    headers=["Date", "Yield"], 
                                    tablefmt="html")
        report_html += f"""
        <h3>Detailed Yield History</h3>
        {yield_trends_table}
"""
        
        report_html += """
    </div>
"""
        
        # 3. Environmental Factors Impact on Yield
        report_html += """
    <div class="report-section">
        <h2>Environmental Factors Impact on Yield</h2>
"""
        env_factors = {
            'Tree Age': tree_data['Tree Age (years)'].mean(),
            'pH Level': tree_data['pH Level'].mean(),
            'Soil Type': tree_data['Soil Type'].mode()[0],
            'Location': tree_data['Location'].mode()[0],
            'Fruit Stage': tree_data['Fruit Stage'].mode()[0],
            'Irrigation': tree_data['Irrigation (mm)'].mean(),
            'Fertilizer': tree_data['Fertilizer Applied (kg)'].mean()
        }
        
        env_factors_data = []
        for factor, value in env_factors.items():
            impact = self._calculate_environmental_impact(factor, value)
            env_factors_data.append([factor, f"{value}", f"{impact:.2f}%"])
        
        env_factors_table = tabulate(env_factors_data, 
                                   headers=["Factor", "Value", "Impact on Yield"], 
                                   tablefmt="html")
        report_html += f"{env_factors_table}"
        
        # Add environmental impact visualization
        impacts = {factor: self._calculate_environmental_impact(factor, value) 
                  for factor, value in env_factors.items()}
        
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                x=list(impacts.keys()),
                y=list(impacts.values()),
                marker_color='orange'
            )
        )
        
        fig2.update_layout(
            title="Environmental Factors Impact on Yield",
            xaxis_title="Factors",
            yaxis_title="Impact (%)",
            template='plotly_white'
        )
        
        report_html += f"{fig2.to_html(full_html=False, include_plotlyjs='cdn')}"
        
        report_html += """
    </div>
"""
        
        # 4. Yield Predictions and Recommendations
        report_html += """
    <div class="report-section">
        <h2>Yield Analysis and Recommendations</h2>
"""
        current_yield = latest_data['Yield (kg)'].mean()
        recommendations = []
        
        # Yield-specific recommendations
        if current_yield < 15:
            recommendations.append(["Yield Level", "Current yield is low. Review all management practices and consider soil testing."])
        elif current_yield > 35:
            recommendations.append(["Yield Level", "Current yield is high. Ensure adequate support and monitor tree health closely."])
        
        # Environmental recommendations
        if env_factors['Tree Age'] < 3:
            recommendations.append(["Tree Age", "Tree is young. Focus on establishing strong root system and proper pruning."])
        elif env_factors['Tree Age'] > 10:
            recommendations.append(["Tree Age", "Tree is mature. Monitor health closely and consider rejuvenation pruning if needed."])
        
        if not 6.0 <= env_factors['pH Level'] <= 7.0:
            recommendations.append(["Soil pH", f"pH ({env_factors['pH Level']:.1f}) is outside optimal range. Consider soil amendment."])
        
        if env_factors['Soil Type'] not in ['Loamy', 'Sandy']:
            recommendations.append(["Soil Type", f"Current soil type ({env_factors['Soil Type']}) may not be optimal. Consider soil improvement."])
        
        if not 30 <= env_factors['Irrigation'] <= 40:
            recommendations.append(["Irrigation", f"Irrigation ({env_factors['Irrigation']:.1f} mm) is outside optimal range. Adjust watering schedule."])
        
        if not 3 <= env_factors['Fertilizer'] <= 4:
            recommendations.append(["Fertilizer", f"Fertilizer application ({env_factors['Fertilizer']:.1f} kg) is outside optimal range. Review fertilization program."])
        
        if recommendations:
            recommendations_table = tabulate(recommendations, 
                                          headers=["Factor", "Recommendation"], 
                                          tablefmt="html")
            report_html += f"""
        <h3>Recommendations for Yield Improvement</h3>
        {recommendations_table}
"""
        else:
            report_html += """
        <h3>Recommendations for Yield Improvement</h3>
        <p>No specific recommendations based on current analysis. Current yield management practices appear optimal.</p>
"""
        
        report_html += """
    </div>
</body>
</html>
"""
        
        # Save the HTML report
        report_file_path = os.path.join(self.data_dir, f'yield_analysis_report_{tree_id if tree_id else tree_species.lower().replace(" ", "_")}.html')
        with open(report_file_path, 'w') as f:
            f.write(report_html)
        
        return report_file_path

    def analyze_yield(self, tree_id=None, tree_species=None):
        """Analyze and predict yield for specific tree ID or species"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        try:
            # Filter data for specific tree or species
            if tree_id:
                tree_data = self.data[self.data['Tree ID'] == tree_id]
            elif tree_species:
                tree_data = self.data[self.data['Tree Species'] == tree_species]
            else:
                tree_data = self.data
            
            if tree_data.empty:
                print("No data found for the specified tree/species")
                return
            
            # Generate yield report
            report_file_path = self.generate_yield_report(tree_data, tree_id, tree_species)
            
            print(f"\nYield analysis report saved to {report_file_path}")
            
            # Open the HTML report in the default web browser
            import webbrowser
            webbrowser.open(f'file://{os.path.abspath(report_file_path)}')
            
            # Print console output
            print("\nYield Analysis Report")
            print("=" * 80)
            
            # Get the latest measurement for each tree
            latest_data = tree_data.sort_values('Measurement Date').groupby('Tree ID').last()
            
            # 1. Current Yield Status
            print("\nCurrent Yield Status:")
            print("-" * 40)
            for _, row in latest_data.iterrows():
                print(f"\nTree ID: {row['Tree ID']}")
                print(f"Species: {row['Tree Species']}")
                print(f"Current Yield: {row['Yield (kg)']:.2f} kg")
                print(f"Fruit Count: {row['Fruit Count']}")
                print(f"Average Fruit Weight: {row['Average Fruit Weight (g)']:.1f} g")
                print(f"Fruit Size: {row['Fruit Size (cm)']:.1f} cm")
                print(f"Fruit Color: {row['Fruit Color']}")
                print(f"Harvest Date: {row['Harvest Date']}")
            
            # 2. Environmental Factors
            env_factors = {
                'Tree Age': tree_data['Tree Age (years)'].mean(),
                'pH Level': tree_data['pH Level'].mean(),
                'Soil Type': tree_data['Soil Type'].mode()[0],
                'Location': tree_data['Location'].mode()[0],
                'Fruit Stage': tree_data['Fruit Stage'].mode()[0],
                'Irrigation': tree_data['Irrigation (mm)'].mean(),
                'Fertilizer': tree_data['Fertilizer Applied (kg)'].mean()
            }
            
            print("\nEnvironmental Factors:")
            print("-" * 40)
            for factor, value in env_factors.items():
                impact = self._calculate_environmental_impact(factor, value)
                print(f"{factor}: {value} ({impact:.2f}% impact on yield)")
            
            # 3. Yield Trends
            yield_trends = tree_data.groupby('Measurement Date')['Yield (kg)'].mean()
            print("\nYield Trends:")
            print("-" * 40)
            if not yield_trends.empty:
                print("Yield progression over time:")
                for date, yield_value in yield_trends.items():
                    print(f"{date}: {yield_value:.2f} kg")
            
            # 4. Generate recommendations
            current_yield = latest_data['Yield (kg)'].mean()
            recommendations = []
            
            # Yield-specific recommendations
            if current_yield < 15:
                recommendations.append(["Yield Level", "Current yield is low. Review all management practices and consider soil testing."])
            elif current_yield > 35:
                recommendations.append(["Yield Level", "Current yield is high. Ensure adequate support and monitor tree health closely."])
            
            # Environmental recommendations
            if env_factors['Tree Age'] < 3:
                recommendations.append(["Tree Age", "Tree is young. Focus on establishing strong root system and proper pruning."])
            elif env_factors['Tree Age'] > 10:
                recommendations.append(["Tree Age", "Tree is mature. Monitor health closely and consider rejuvenation pruning if needed."])
            
            if not 6.0 <= env_factors['pH Level'] <= 7.0:
                recommendations.append(["Soil pH", f"pH ({env_factors['pH Level']:.1f}) is outside optimal range. Consider soil amendment."])
            
            if env_factors['Soil Type'] not in ['Loamy', 'Sandy']:
                recommendations.append(["Soil Type", f"Current soil type ({env_factors['Soil Type']}) may not be optimal. Consider soil improvement."])
            
            if not 30 <= env_factors['Irrigation'] <= 40:
                recommendations.append(["Irrigation", f"Irrigation ({env_factors['Irrigation']:.1f} mm) is outside optimal range. Adjust watering schedule."])
            
            if not 3 <= env_factors['Fertilizer'] <= 4:
                recommendations.append(["Fertilizer", f"Fertilizer application ({env_factors['Fertilizer']:.1f} kg) is outside optimal range. Review fertilization program."])
            
            if recommendations:
                print("\nRecommendations:")
                print("-" * 40)
                for factor, recommendation in recommendations:
                    print(f"- {factor}: {recommendation}")
            
        except Exception as e:
            print(f"Error in yield analysis: {str(e)}")
    
    def _calculate_chemical_impact(self, compound, concentration):
        """Calculate the impact of a chemical compound on yield"""
        # Define optimal ranges and their impact on yield
        optimal_ranges = {
            'Sugars': (2.5, 4.0),
            'Malic Acid': (0.8, 1.5),
            'Vitamin C': (0.4, 0.8),
            'Chlorophyll': (2.0, 3.0),
            'Anthocyanins': (3.5, 4.5),
            'Pectin': (1.2, 1.8),
            'Actinidin': (0.8, 1.2),
            'Fiber': (1.8, 2.4)
        }
        
        if compound in optimal_ranges:
            min_val, max_val = optimal_ranges[compound]
            if min_val <= concentration <= max_val:
                return 100
            else:
                # Calculate impact based on distance from optimal range
                distance = min(abs(concentration - min_val), abs(concentration - max_val))
                return max(0, 100 - (distance * 20))
        return 50  # Default impact for unknown compounds
    
    def _calculate_environmental_impact(self, factor, value):
        """Calculate the impact of environmental factors on yield"""
        if factor == 'Tree Age':
            # Optimal age range: 3-10 years
            if 3 <= value <= 10:
                return 100
            else:
                return max(50, 100 - ((value - 10) * 5))  # Gradual decrease after 10 years
                
        elif factor == 'pH Level':
            # Optimal pH range: 6.0-7.0
            if 6.0 <= value <= 7.0:
                return 100
            else:
                distance = min(abs(value - 6.0), abs(value - 7.0))
                return max(0, 100 - (distance * 20))
                
        elif factor == 'Soil Type':
            # Soil type impact
            soil_impact = {
                'Loamy': 100,
                'Sandy': 80,
                'Clay': 70,
                'Silt': 85
            }
            return soil_impact.get(value, 50)
            
        elif factor == 'Fruit Stage':
            # Fruit stage impact
            stage_impact = {
                'Early Development': 30,
                'Growth': 60,
                'Maturation': 90,
                'Ripening': 100,
                'Harvest Ready': 95
            }
            return stage_impact.get(value, 50)
            
        elif factor == 'Irrigation':
            # Optimal irrigation: 30-40 mm
            if 30 <= value <= 40:
                return 100
            else:
                distance = min(abs(value - 30), abs(value - 40))
                return max(0, 100 - (distance * 5))
                
        elif factor == 'Fertilizer':
            # Optimal fertilizer: 3-4 kg
            if 3 <= value <= 4:
                return 100
            else:
                distance = min(abs(value - 3), abs(value - 4))
                return max(0, 100 - (distance * 20))
            
        return 50  # Default impact for unknown factors
    
    def _generate_yield_recommendations(self, env_factors, current_yield):
        """Generate recommendations based on yield analysis"""
        print("\nRecommendations:")
        print("-" * 40)
        
        # Environmental recommendations
        if env_factors['Tree Age'] < 3:
            print("- Tree is young. Focus on establishing strong root system and proper pruning.")
        elif env_factors['Tree Age'] > 10:
            print("- Tree is mature. Monitor health closely and consider rejuvenation pruning if needed.")
        
        if not 6.0 <= env_factors['pH Level'] <= 7.0:
            print(f"- Soil pH ({env_factors['pH Level']:.1f}) is outside optimal range. Consider soil amendment.")
        
        if env_factors['Soil Type'] not in ['Loamy', 'Sandy']:
            print(f"- Current soil type ({env_factors['Soil Type']}) may not be optimal. Consider soil improvement.")
        
        if not 30 <= env_factors['Irrigation'] <= 40:
            print(f"- Irrigation ({env_factors['Irrigation']:.1f} mm) is outside optimal range. Adjust watering schedule.")
        
        if not 3 <= env_factors['Fertilizer'] <= 4:
            print(f"- Fertilizer application ({env_factors['Fertilizer']:.1f} kg) is outside optimal range. Review fertilization program.")
        
        # Yield-specific recommendations
        if current_yield < 15:
            print("- Expected yield is low. Review all management practices and consider soil testing.")
        elif current_yield > 35:
            print("- Expected yield is high. Ensure adequate support and monitor tree health closely.")

def main():
    print("SKUAST Fruit Tree Analysis System")
    print("--------------------------------")
    
    while True:
        print("\nMain Menu:")
        print("1. Chemical Analysis")
        print("2. Yield Analysis")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            analyzer = FruitTreeAnalyzer()
            print("\nChemical Analysis Menu:")
            print("1. Load main dataset")
            print("2. Compare multiple trees (Enter Tree IDs)")
            print("3. Back to main menu")
            
            sub_choice = input("\nEnter your choice (1-3): ")
            
            if sub_choice == '1':
                file_path = input("Enter the path to the chemical analysis data file: ")
                analyzer.load_data(file_path)
                if analyzer.data is not None:
                    while True:
                        print("\nAvailable Tree IDs:")
                        for species in analyzer.data['Tree Species'].unique():
                            tree_ids = analyzer.data[analyzer.data['Tree Species'] == species]['Tree ID'].unique()
                            print(f"{species}: {', '.join(tree_ids)}")

                        tree_id_input = input("\nEnter Tree ID to analyze (or type 'menu' to return to main menu, 'exit' to quit): ").strip().upper()
                        
                        if tree_id_input == 'MENU':
                            break
                        elif tree_id_input == 'EXIT':
                            print("Thank you for using the SKUAST Fruit Tree Analysis System!")
                            return
                        
                        if tree_id_input in analyzer.data['Tree ID'].unique():
                            print(f"\nAnalyzing Tree {tree_id_input}...")
                            analyzer.generate_tree_report(tree_id=tree_id_input)
                        else:
                            print(f"Invalid Tree ID '{tree_id_input}'. Please use one of the available IDs shown above.")
            
            elif sub_choice == '2':
                if analyzer.data is None:
                    print("Please load data first (Option 1)")
                    continue
                    
                print("\nAvailable Tree IDs:")
                for species in analyzer.data['Tree Species'].unique():
                    tree_ids = analyzer.data[analyzer.data['Tree Species'] == species]['Tree ID'].unique()
                    print(f"{species}: {', '.join(tree_ids)}")
                
                tree_ids = input("\nEnter Tree IDs to compare (comma-separated, e.g., A001,C001): ").strip().upper()
                tree_ids = [id.strip() for id in tree_ids.split(',')]
                
                valid_ids = [id for id in tree_ids if id in analyzer.data['Tree ID'].unique()]
                if valid_ids:
                    print(f"\nComparing Trees: {', '.join(valid_ids)}")
                    analyzer.compare_fruit_trees(valid_ids)
                else:
                    print("Invalid Tree IDs. Please use the available IDs shown above.")
        
        elif choice == '2':
            yield_analyzer = TreeYieldAnalyzer()
            print("\nYield Analysis Menu:")
            print("1. Load yield dataset")
            print("2. Back to main menu")
            
            sub_choice = input("\nEnter your choice (1-2): ")
            
            if sub_choice == '1':
                file_path = input("Enter the path to the yield analysis data file: ")
                yield_analyzer.load_data(file_path)
                if yield_analyzer.data is not None:
                    while True:
                        print("\nAvailable Tree IDs:")
                        for species in yield_analyzer.data['Tree Species'].unique():
                            tree_ids = yield_analyzer.data[yield_analyzer.data['Tree Species'] == species]['Tree ID'].unique()
                            print(f"{species}: {', '.join(tree_ids)}")

                        tree_id_input = input("\nEnter Tree ID to analyze yield (or type 'menu' to return to main menu, 'exit' to quit): ").strip().upper()
                        
                        if tree_id_input == 'MENU':
                            break
                        elif tree_id_input == 'EXIT':
                            print("Thank you for using the SKUAST Fruit Tree Analysis System!")
                            return
                        
                        if tree_id_input in yield_analyzer.data['Tree ID'].unique():
                            print(f"\nAnalyzing yield for Tree {tree_id_input}...")
                            yield_analyzer.analyze_yield(tree_id=tree_id_input)
                        else:
                            print(f"Invalid Tree ID '{tree_id_input}'. Please use one of the available IDs shown above.")
        
        elif choice == '3':
            print("Thank you for using the SKUAST Fruit Tree Analysis System!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 