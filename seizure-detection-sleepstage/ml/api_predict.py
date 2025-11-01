"""
API Wrapper for Seizure Detection Model
This script wraps the predict_user_file.py to provide JSON output for the API
"""

import sys
import json
import os
from pathlib import Path
import numpy as np
import pickle

# Add the ml directory to the path
ml_dir = Path(__file__).parent
sys.path.insert(0, str(ml_dir))

# Import EEGFeatureExtractor BEFORE any pickle operations
from predict_user_file import EEGFeatureExtractor

def main():
    if len(sys.argv) < 2:
        print(json.dumps({
            "error": "No file path provided",
            "success": False
        }))
        sys.exit(1)
    
    file_path = sys.argv[1]
    file_format = None
    report_path = None
    analysis_type = "seizure"
    
    # Parse command line arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--format" and i + 1 < len(sys.argv):
            file_format = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--report" and i + 1 < len(sys.argv):
            report_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--type" and i + 1 < len(sys.argv):
            analysis_type = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(json.dumps({
            "error": "File not found",
            "success": False
        }))
        sys.exit(1)
    
    try:
        # Load the model
        model_path = ml_dir / "best_model.pkl"
        feature_extractor_path = ml_dir / "feature_extractor.pkl"
        
        if not model_path.exists():
            print(json.dumps({
                "error": "Model file not found. Please train the model first.",
                "success": False
            }))
            sys.exit(1)
        
        # Load model and feature extractor
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        if feature_extractor_path.exists():
            with open(feature_extractor_path, 'rb') as f:
                feature_extractor = pickle.load(f)
        else:
            feature_extractor = EEGFeatureExtractor()
        
        # Determine file format
        if not file_format:
            file_format = Path(file_path).suffix[1:]  # Remove the dot
        
        # Load data using the feature extractor's load_data method
        try:
            X, y_true = feature_extractor.load_data(file_path, file_format)
        except Exception as e:
            print(json.dumps({
                "error": f"Failed to load file: {str(e)}",
                "success": False
            }))
            sys.exit(1)
        
        # Extract features using the feature extractor's transform method
        try:
            features = feature_extractor.transform(X)
        except Exception as e:
            print(json.dumps({
                "error": f"Failed to extract features: {str(e)}",
                "success": False
            }))
            sys.exit(1)
        
        # Make predictions for all samples
        predictions = model.predict(features)
        prediction_probas = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
        
        # Calculate overall statistics
        seizure_count = int(np.sum(predictions == 1))
        total_samples = int(len(predictions))
        seizure_percentage = float((seizure_count / total_samples) * 100)
        
        # Get average confidence
        if prediction_probas is not None:
            avg_confidence = float(np.mean(np.max(prediction_probas, axis=1)) * 100)
            avg_seizure_prob = float(np.mean(prediction_probas[:, 1]) * 100) if prediction_probas.shape[1] > 1 else 0
            avg_normal_prob = float(np.mean(prediction_probas[:, 0]) * 100)
        else:
            avg_confidence = None
            avg_seizure_prob = None
            avg_normal_prob = None
        
        # Prepare result
        result = {
            "success": True,
            "prediction": "Seizure Detected" if seizure_percentage > 50 else "Normal (No Seizure)",
            "seizure_detected": seizure_count > 0,
            "total_samples": total_samples,
            "seizure_samples": seizure_count,
            "normal_samples": total_samples - seizure_count,
            "seizure_percentage": seizure_percentage,
            "confidence": avg_confidence,
            "file_format": file_format,
            "features_extracted": int(features.shape[1]),
            "message": f"Analyzed {total_samples} samples. Found {seizure_count} seizure events ({seizure_percentage:.1f}%)"
        }
        
        # Add probability details if available
        if avg_seizure_prob is not None:
            result["probabilities"] = {
                "normal": avg_normal_prob,
                "seizure": avg_seizure_prob
            }
        
        print(json.dumps(result))
        
    except ImportError as e:
        print(json.dumps({
            "error": f"Missing required Python package: {str(e)}",
            "success": False,
            "suggestion": "Please install required packages: pip install numpy pandas scikit-learn mne scipy matplotlib reportlab"
        }))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({
            "error": f"Analysis failed: {str(e)}",
            "success": False
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()
