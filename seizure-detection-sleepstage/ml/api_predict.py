"""
API Wrapper for Seizure Detection Model
This script wraps the predict_user_file.py to provide JSON output for the API
"""

import sys
import json
import os
from pathlib import Path
import numpy as np

# Add the ml directory to the path
ml_dir = Path(__file__).parent
sys.path.insert(0, str(ml_dir))

def load_file_data(file_path, file_format):
    """Load data from various file formats"""
    try:
        if file_format == 'npz':
            data = np.load(file_path)
            # Try common key names
            for key in ['X', 'data', 'signals', 'eeg']:
                if key in data:
                    return data[key]
            # If no common key, return first array
            return data[list(data.keys())[0]]
        
        elif file_format == 'edf':
            import mne
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            return raw.get_data()
        
        elif file_format == 'csv':
            import pandas as pd
            df = pd.read_csv(file_path)
            return df.values.T  # Transpose to get (channels, time_points)
        
        elif file_format == 'pkl':
            import pickle
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                for key in ['X', 'data', 'signals', 'eeg']:
                    if key in data:
                        return data[key]
                return list(data.values())[0]
            return data
        
        else:
            return None
            
    except Exception as e:
        print(f"Error loading file: {str(e)}", file=sys.stderr)
        return None

def main():
    if len(sys.argv) < 2:
        print(json.dumps({
            "error": "No file path provided",
            "success": False
        }))
        sys.exit(1)
    
    file_path = sys.argv[1]
    file_format = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(json.dumps({
            "error": "File not found",
            "success": False
        }))
        sys.exit(1)
    
    try:
        # Import the prediction module
        from predict_user_file import EEGFeatureExtractor
        import pickle
        import numpy as np
        
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
        
        # Load and process the data
        data = load_file_data(file_path, file_format)
        
        if data is None:
            print(json.dumps({
                "error": f"Failed to load file in {file_format} format",
                "success": False
            }))
            sys.exit(1)
        
        # Extract features
        features = feature_extractor.extract_all_features(data)
        features = features.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else None
        
        # Prepare result
        result = {
            "success": True,
            "prediction": "Seizure Detected" if prediction == 1 else "Normal (No Seizure)",
            "prediction_value": int(prediction),
            "confidence": float(max(prediction_proba) * 100) if prediction_proba is not None else None,
            "file_format": file_format,
            "features_extracted": int(features.shape[1]),
            "message": "Analysis completed successfully"
        }
        
        # Add probability details if available
        if prediction_proba is not None:
            result["probabilities"] = {
                "normal": float(prediction_proba[0] * 100),
                "seizure": float(prediction_proba[1] * 100) if len(prediction_proba) > 1 else 0
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
