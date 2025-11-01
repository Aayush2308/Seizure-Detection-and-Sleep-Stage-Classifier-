import pickle
import numpy as np
import sys
from predict_user_file import EEGFeatureExtractor

# Load the feature extractor
with open('feature_extractor.pkl', 'rb') as f:
    fe = pickle.load(f)

print(f"Type: {type(fe)}")
print(f"Has transform: {hasattr(fe, 'transform')}")
print(f"Has extract_all_features: {hasattr(fe, 'extract_all_features')}")
print(f"\nClass name: {fe.__class__.__name__}")
print(f"\nMethods and attributes:")
for attr in dir(fe):
    if not attr.startswith('_'):
        print(f"  - {attr}")

# Test with sample data
print("\n\nTesting with sample data:")
sample = np.random.randn(23, 256)  # Single sample
print(f"Input shape: {sample.shape}")

try:
    if hasattr(fe, 'extract_all_features'):
        features = fe.extract_all_features(sample)
        print(f"Output features: {features.shape}")
    elif hasattr(fe, 'transform'):
        features = fe.transform(sample.reshape(1, 23, 256))
        print(f"Output features: {features.shape}")
except Exception as e:
    print(f"Error: {e}")
