import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

class MLPredictor:
    def __init__(self):
        self.chemical_model = None
        self.yield_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models_dir = 'models'
        
        # Create models directory if it doesn't exist
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            
        # Initialize label encoders for categorical variables
        self.categorical_features = {
            'leaf_color': ['Green', 'Yellow', 'Brown'],
            'moisture_level': ['Low', 'Medium', 'High'],
            'chlorophyll_content': ['Low', 'Normal', 'High'],
            'nitrogen_level': ['Low', 'Adequate', 'High'],
            'soil_moisture': ['Dry', 'Moderate', 'Wet']
        }
        
        for feature, categories in self.categorical_features.items():
            self.label_encoders[feature] = LabelEncoder()
            self.label_encoders[feature].fit(categories)

    def prepare_chemical_data(self, data):
        """Prepare chemical analysis data for ML model"""
        # Convert categorical variables to numeric
        for feature in ['leaf_color', 'moisture_level', 'chlorophyll_content', 'nitrogen_level']:
            if feature in data.columns:
                data[feature] = self.label_encoders[feature].transform(data[feature])
        
        # Scale numerical features
        numerical_features = ['soil_ph']
        if not data[numerical_features].empty:
            data[numerical_features] = self.scaler.fit_transform(data[numerical_features])
        
        return data

    def prepare_yield_data(self, data):
        """Prepare yield analysis data for ML model"""
        # Convert categorical variables to numeric
        for feature in ['leaf_color', 'soil_moisture']:
            if feature in data.columns:
                data[feature] = self.label_encoders[feature].transform(data[feature])
        
        # Convert boolean to numeric
        if 'fertilizer_used' in data.columns:
            data['fertilizer_used'] = data['fertilizer_used'].astype(int)
        
        # Scale numerical features
        numerical_features = ['tree_age', 'flower_buds_count']
        if not data[numerical_features].empty:
            data[numerical_features] = self.scaler.fit_transform(data[numerical_features])
        
        return data

    def train_chemical_model(self, training_data):
        """Train chemical analysis model"""
        X = self.prepare_chemical_data(training_data.drop('target', axis=1))
        y = training_data['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.chemical_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.chemical_model.fit(X_train, y_train)
        
        # Save the model
        joblib.dump(self.chemical_model, os.path.join(self.models_dir, 'chemical_model.joblib'))
        
        return self.chemical_model.score(X_test, y_test)

    def train_yield_model(self, training_data):
        """Train yield analysis model"""
        X = self.prepare_yield_data(training_data.drop('target', axis=1))
        y = training_data['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.yield_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.yield_model.fit(X_train, y_train)
        
        # Save the model
        joblib.dump(self.yield_model, os.path.join(self.models_dir, 'yield_model.joblib'))
        
        return self.yield_model.score(X_test, y_test)

    def predict_chemical(self, input_data):
        """Predict chemical analysis results"""
        if self.chemical_model is None:
            # Try to load the model if it exists
            model_path = os.path.join(self.models_dir, 'chemical_model.joblib')
            if os.path.exists(model_path):
                self.chemical_model = joblib.load(model_path)
            else:
                raise ValueError("Chemical model not trained or loaded")

        # Prepare input data
        prepared_data = self.prepare_chemical_data(pd.DataFrame([input_data]))
        
        # Make prediction
        prediction = self.chemical_model.predict(prepared_data)[0]
        
        # Get feature importance
        feature_importance = dict(zip(prepared_data.columns, 
                                   self.chemical_model.feature_importances_))
        
        return {
            'prediction': prediction,
            'feature_importance': feature_importance
        }

    def predict_yield(self, input_data):
        """Predict yield analysis results"""
        if self.yield_model is None:
            # Try to load the model if it exists
            model_path = os.path.join(self.models_dir, 'yield_model.joblib')
            if os.path.exists(model_path):
                self.yield_model = joblib.load(model_path)
            else:
                raise ValueError("Yield model not trained or loaded")

        # Prepare input data
        prepared_data = self.prepare_yield_data(pd.DataFrame([input_data]))
        
        # Make prediction
        prediction = self.yield_model.predict(prepared_data)[0]
        
        # Get feature importance
        feature_importance = dict(zip(prepared_data.columns, 
                                   self.yield_model.feature_importances_))
        
        return {
            'prediction': prediction,
            'feature_importance': feature_importance
        }

    def load_models(self):
        """Load saved models if they exist"""
        chemical_model_path = os.path.join(self.models_dir, 'chemical_model.joblib')
        yield_model_path = os.path.join(self.models_dir, 'yield_model.joblib')
        
        if os.path.exists(chemical_model_path):
            self.chemical_model = joblib.load(chemical_model_path)
        
        if os.path.exists(yield_model_path):
            self.yield_model = joblib.load(yield_model_path) 