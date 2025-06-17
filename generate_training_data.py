import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_chemical_dataset(num_samples=1000):
    """Generate realistic chemical analysis dataset"""
    np.random.seed(42)
    
    # Define base parameters
    leaf_colors = ['Green', 'Yellow', 'Brown']
    moisture_levels = ['Low', 'Medium', 'High']
    chlorophyll_contents = ['Low', 'Normal', 'High']
    nitrogen_levels = ['Low', 'Adequate', 'High']
    
    # Generate data
    data = {
        'leaf_color': np.random.choice(leaf_colors, num_samples),
        'soil_ph': np.random.normal(6.5, 0.5, num_samples).clip(5.5, 7.5),
        'moisture_level': np.random.choice(moisture_levels, num_samples),
        'chlorophyll_content': np.random.choice(chlorophyll_contents, num_samples),
        'nitrogen_level': np.random.choice(nitrogen_levels, num_samples),
    }
    
    # Generate chemical compounds with realistic correlations
    data['sugars'] = np.random.normal(3.2, 0.4, num_samples)
    data['malic_acid'] = np.random.normal(1.1, 0.2, num_samples)
    data['vitamin_c'] = np.random.normal(0.6, 0.1, num_samples)
    data['chlorophyll'] = np.random.normal(2.5, 0.3, num_samples)
    data['anthocyanins'] = np.random.normal(4.0, 0.3, num_samples)
    data['pectin'] = np.random.normal(1.5, 0.2, num_samples)
    data['actinidin'] = np.random.normal(1.0, 0.1, num_samples)
    data['fiber'] = np.random.normal(2.1, 0.2, num_samples)
    
    # Add correlations between parameters
    for i in range(num_samples):
        # Leaf color affects chlorophyll
        if data['leaf_color'][i] == 'Green':
            data['chlorophyll'][i] *= 1.2
        elif data['leaf_color'][i] == 'Yellow':
            data['chlorophyll'][i] *= 0.7
        elif data['leaf_color'][i] == 'Brown':
            data['chlorophyll'][i] *= 0.4
            
        # Moisture affects sugar content
        if data['moisture_level'][i] == 'High':
            data['sugars'][i] *= 1.1
        elif data['moisture_level'][i] == 'Low':
            data['sugars'][i] *= 0.9
            
        # Nitrogen level affects protein content
        if data['nitrogen_level'][i] == 'High':
            data['actinidin'][i] *= 1.2
        elif data['nitrogen_level'][i] == 'Low':
            data['actinidin'][i] *= 0.8
            
        # pH affects nutrient availability
        if 6.0 <= data['soil_ph'][i] <= 7.0:
            data['vitamin_c'][i] *= 1.1
            data['anthocyanins'][i] *= 1.1
        else:
            data['vitamin_c'][i] *= 0.9
            data['anthocyanins'][i] *= 0.9
    
    # Calculate health score (target variable)
    data['target'] = np.zeros(num_samples)
    for i in range(num_samples):
        score = 0
        # Chemical compound scores
        if 2.5 <= data['sugars'][i] <= 4.0: score += 1
        if 0.8 <= data['malic_acid'][i] <= 1.5: score += 1
        if 0.4 <= data['vitamin_c'][i] <= 0.8: score += 1
        if 2.0 <= data['chlorophyll'][i] <= 3.0: score += 1
        if 3.5 <= data['anthocyanins'][i] <= 4.5: score += 1
        if 1.2 <= data['pectin'][i] <= 1.8: score += 1
        if 0.8 <= data['actinidin'][i] <= 1.2: score += 1
        if 1.8 <= data['fiber'][i] <= 2.4: score += 1
        
        # Environmental factor scores
        if 6.0 <= data['soil_ph'][i] <= 7.0: score += 1
        if data['moisture_level'][i] == 'Medium': score += 1
        if data['chlorophyll_content'][i] == 'Normal': score += 1
        if data['nitrogen_level'][i] == 'Adequate': score += 1
        
        data['target'][i] = (score / 12) * 100  # Normalize to 0-100
    
    return pd.DataFrame(data)

def generate_yield_dataset(num_samples=1000):
    """Generate realistic yield analysis dataset"""
    np.random.seed(42)
    
    # Define base parameters
    leaf_colors = ['Green', 'Yellow', 'Brown']
    soil_moistures = ['Dry', 'Moderate', 'Wet']
    
    # Generate data
    data = {
        'tree_age': np.random.normal(8, 4, num_samples).clip(1, 20),
        'flower_buds_count': np.random.normal(120, 30, num_samples).clip(20, 200).astype(int),
        'leaf_color': np.random.choice(leaf_colors, num_samples),
        'soil_moisture': np.random.choice(soil_moistures, num_samples),
        'fertilizer_used': np.random.choice([True, False], num_samples, p=[0.7, 0.3])
    }
    
    # Generate yield-related parameters with realistic correlations
    data['fruit_size'] = np.random.normal(150, 20, num_samples)  # grams
    data['fruit_count'] = np.random.normal(80, 15, num_samples)  # per tree
    data['fruit_quality'] = np.random.normal(7.5, 1.0, num_samples)  # 0-10 scale
    
    # Add correlations between parameters
    for i in range(num_samples):
        # Tree age affects yield
        if 3 <= data['tree_age'][i] <= 15:
            data['fruit_count'][i] *= 1.2
            data['fruit_quality'][i] *= 1.1
        elif data['tree_age'][i] < 3:
            data['fruit_count'][i] *= 0.7
            data['fruit_quality'][i] *= 0.9
        else:  # > 15 years
            data['fruit_count'][i] *= 0.9
            data['fruit_quality'][i] *= 0.95
            
        # Flower buds affect fruit count
        data['fruit_count'][i] *= (data['flower_buds_count'][i] / 120)
        
        # Leaf color affects quality
        if data['leaf_color'][i] == 'Green':
            data['fruit_quality'][i] *= 1.1
        elif data['leaf_color'][i] == 'Yellow':
            data['fruit_quality'][i] *= 0.9
        elif data['leaf_color'][i] == 'Brown':
            data['fruit_quality'][i] *= 0.8
            
        # Soil moisture affects size and count
        if data['soil_moisture'][i] == 'Moderate':
            data['fruit_size'][i] *= 1.1
            data['fruit_count'][i] *= 1.1
        elif data['soil_moisture'][i] == 'Dry':
            data['fruit_size'][i] *= 0.9
            data['fruit_count'][i] *= 0.9
            
        # Fertilizer affects all parameters
        if data['fertilizer_used'][i]:
            data['fruit_size'][i] *= 1.1
            data['fruit_count'][i] *= 1.1
            data['fruit_quality'][i] *= 1.1
    
    # Calculate expected yield (target variable)
    data['target'] = data['fruit_size'] * data['fruit_count'] / 1000  # Convert to kg
    
    return pd.DataFrame(data)

def main():
    # Generate datasets
    chemical_data = generate_chemical_dataset(2000)  # 2000 samples for chemical analysis
    yield_data = generate_yield_dataset(2000)  # 2000 samples for yield analysis
    
    # Save datasets
    chemical_data.to_csv('chemical_training_data.csv', index=False)
    yield_data.to_csv('yield_training_data.csv', index=False)
    
    print("Generated datasets:")
    print(f"Chemical Analysis Dataset: {len(chemical_data)} samples")
    print(f"Yield Analysis Dataset: {len(yield_data)} samples")
    
    # Print sample statistics
    print("\nChemical Analysis Dataset Statistics:")
    print(chemical_data.describe())
    print("\nYield Analysis Dataset Statistics:")
    print(yield_data.describe())

if __name__ == "__main__":
    main() 