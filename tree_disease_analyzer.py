import cv2
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
from fastapi import HTTPException

class TreeDiseaseAnalyzer:
    def __init__(self):
        self.data_dir = "data/disease_images"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Define common tree diseases and their characteristics
        self.diseases = {
            "apple_scab": {
                "description": "Fungal disease causing dark, scabby lesions on leaves and fruit",
                "severity_levels": ["Low", "Moderate", "Severe"],
                "treatment": "Apply fungicide, remove infected leaves, improve air circulation"
            },
            "black_rot": {
                "description": "Fungal disease causing circular brown/black spots with dark borders",
                "severity_levels": ["Low", "Moderate", "Severe"],
                "treatment": "Prune infected areas, apply copper-based fungicide, maintain good sanitation"
            },
            "powdery_mildew": {
                "description": "Fungal disease causing white powdery coating on leaves",
                "severity_levels": ["Low", "Moderate", "Severe"],
                "treatment": "Apply sulfur-based fungicide, improve air circulation, avoid overhead watering"
            },
            "cedar_apple_rust": {
                "description": "Fungal disease causing bright orange spots on leaves",
                "severity_levels": ["Low", "Moderate", "Severe"],
                "treatment": "Remove nearby cedar trees, apply fungicide, plant resistant varieties"
            }
        }

    def analyze_image(self, image_data: bytes, tree_id: str = None):
        """Analyze tree image for disease detection"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Could not decode image")
            
            # Save original image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"disease_analysis_{tree_id}_{timestamp}.jpg" if tree_id else f"disease_analysis_{timestamp}.jpg"
            image_path = os.path.join(self.data_dir, image_filename)
            cv2.imwrite(image_path, img)
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Analyze image features
            analysis_results = self._analyze_image_features(img, hsv)
            
            # Generate detailed report
            report = self._generate_analysis_report(analysis_results, image_path)
            
            return report
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

    def _analyze_image_features(self, img, hsv):
        """Analyze various features of the image to detect diseases"""
        results = {
            "detected_diseases": [],
            "healthy_tissue_percentage": 0,
            "affected_areas": [],
            "color_analysis": {},
            "texture_analysis": {}
        }
        
        try:
            # 1. Color Analysis
            # Define color ranges for common disease symptoms
            color_ranges = {
                "healthy_green": ([35, 50, 50], [85, 255, 255]),
                "yellow_spots": ([20, 100, 100], [30, 255, 255]),
                "brown_lesions": ([10, 60, 60], [20, 255, 255]),
                "white_powder": ([0, 0, 200], [180, 30, 255])
            }
            
            for color_name, (lower, upper) in color_ranges.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                percentage = (np.sum(mask > 0) / mask.size) * 100
                results["color_analysis"][color_name] = round(percentage, 2)
            
            # Calculate healthy tissue percentage
            results["healthy_tissue_percentage"] = round(results["color_analysis"]["healthy_green"], 2)
            
            # 2. Disease Detection Logic
            if results["color_analysis"]["brown_lesions"] > 10:
                confidence = min(100, results["color_analysis"]["brown_lesions"] * 2)
                results["detected_diseases"].append({
                    "name": "black_rot",
                    "confidence": confidence,
                    "severity": "Severe" if confidence > 70 else "Moderate" if confidence > 40 else "Low"
                })
            
            if results["color_analysis"]["white_powder"] > 15:
                confidence = min(100, results["color_analysis"]["white_powder"] * 2)
                results["detected_diseases"].append({
                    "name": "powdery_mildew",
                    "confidence": confidence,
                    "severity": "Severe" if confidence > 70 else "Moderate" if confidence > 40 else "Low"
                })
            
            if results["color_analysis"]["yellow_spots"] > 12:
                confidence = min(100, results["color_analysis"]["yellow_spots"] * 2)
                results["detected_diseases"].append({
                    "name": "cedar_apple_rust",
                    "confidence": confidence,
                    "severity": "Severe" if confidence > 70 else "Moderate" if confidence > 40 else "Low"
                })
            
            # 3. Texture Analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            results["texture_analysis"] = self._calculate_texture_features(gray)
            
            # 4. Calculate affected areas
            for disease in results["detected_diseases"]:
                affected_area = self._identify_affected_areas(img, hsv, disease["name"])
                results["affected_areas"].append({
                    "disease": disease["name"],
                    "area_percentage": affected_area
                })
            
            return results
            
        except Exception as e:
            print(f"Error in image feature analysis: {str(e)}")
            return results

    def _calculate_texture_features(self, gray_img):
        """Calculate texture features from grayscale image"""
        features = {}
        
        try:
            # Calculate gradient-based features
            kernel_size = 3
            dx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size)
            dy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size)
            
            gradient_magnitude = np.sqrt(dx**2 + dy**2)
            
            # Calculate texture properties
            features["contrast"] = float(np.std(gradient_magnitude))
            features["smoothness"] = float(1 - (1 / (1 + np.sum(gradient_magnitude))))
            features["uniformity"] = float(np.sum(np.square(cv2.calcHist([gray_img], [0], None, [256], [0, 256]))) / (gray_img.size ** 2))
            
            return features
            
        except Exception as e:
            print(f"Error calculating texture features: {str(e)}")
            return features

    def _identify_affected_areas(self, img, hsv, disease_name):
        """Identify and calculate percentage of affected areas for a specific disease"""
        try:
            if disease_name == "black_rot":
                lower = np.array([10, 60, 60])
                upper = np.array([20, 255, 255])
            elif disease_name == "powdery_mildew":
                lower = np.array([0, 0, 200])
                upper = np.array([180, 30, 255])
            elif disease_name == "cedar_apple_rust":
                lower = np.array([20, 100, 100])
                upper = np.array([30, 255, 255])
            else:
                return 0
            
            mask = cv2.inRange(hsv, lower, upper)
            affected_pixels = np.sum(mask > 0)
            total_pixels = mask.size
            
            return round((affected_pixels / total_pixels) * 100, 2)
            
        except Exception as e:
            print(f"Error identifying affected areas: {str(e)}")
            return 0

    def _generate_analysis_report(self, analysis_results, image_path):
        """Generate a detailed analysis report"""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "image_path": image_path,
                "analysis_summary": {
                    "healthy_tissue_percentage": analysis_results["healthy_tissue_percentage"],
                    "number_of_diseases_detected": len(analysis_results["detected_diseases"]),
                    "overall_health_status": "Healthy" if analysis_results["healthy_tissue_percentage"] > 80 else "Requires Attention"
                },
                "detected_diseases": [],
                "color_analysis": analysis_results["color_analysis"],
                "texture_analysis": analysis_results["texture_analysis"],
                "recommendations": []
            }
            
            # Add detailed disease information
            for disease in analysis_results["detected_diseases"]:
                disease_name = disease["name"]
                disease_info = self.diseases.get(disease_name, {})
                
                # Find affected area percentage
                affected_area = next(
                    (area["area_percentage"] for area in analysis_results["affected_areas"] 
                     if area["disease"] == disease_name),
                    0
                )
                
                disease_details = {
                    "name": disease_name,
                    "confidence": round(disease["confidence"], 2),
                    "severity": disease["severity"],
                    "affected_area_percentage": affected_area,
                    "description": disease_info.get("description", ""),
                    "treatment": disease_info.get("treatment", "")
                }
                report["detected_diseases"].append(disease_details)
            
            # Generate recommendations
            if analysis_results["healthy_tissue_percentage"] < 60:
                report["recommendations"].append({
                    "category": "General Health",
                    "action": "Immediate attention required. Consider consulting a professional arborist.",
                    "priority": "High"
                })
            
            for disease in analysis_results["detected_diseases"]:
                if disease["name"] in self.diseases:
                    report["recommendations"].append({
                        "category": "Disease Treatment",
                        "disease": disease["name"],
                        "severity": disease["severity"],
                        "action": self.diseases[disease["name"]]["treatment"],
                        "priority": "High" if disease["severity"] == "Severe" else "Medium" if disease["severity"] == "Moderate" else "Low"
                    })
            
            if len(analysis_results["detected_diseases"]) > 0:
                report["recommendations"].append({
                    "category": "Preventive Measures",
                    "action": "Implement regular monitoring, maintain good air circulation, and practice proper sanitation",
                    "priority": "Medium"
                })
            
            return report
            
        except Exception as e:
            print(f"Error generating analysis report: {str(e)}")
            return {"error": str(e)} 