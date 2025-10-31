"""
Seizure Detection - User File Prediction Script

This script demonstrates how to use the trained pipeline to predict
seizures from user-provided EEG files in various formats.

Usage:
    python predict_user_file.py <file_path> [--format FORMAT]

Examples:
    python predict_user_file.py user_data.npz
    python predict_user_file.py user_data.edf --format edf
    python predict_user_file.py user_data.csv --format csv

Note: Make sure you have trained the model first by running main.ipynb
"""

import pickle
import numpy as np
import argparse
import sys
from pathlib import Path
import pandas as pd
import mne
from scipy import signal, stats
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import tempfile
import os


class EEGFeatureExtractor:
    """
    Comprehensive feature extraction for EEG signals.
    Supports multiple input formats: NPZ, EDF, PKL, CSV
    """
    
    def __init__(self, sampling_rate=256):
        self.sampling_rate = sampling_rate
        self.scaler = StandardScaler()
        
    def extract_statistical_features(self, signal_data):
        """Extract statistical features from signal"""
        features = {}
        features['mean'] = np.mean(signal_data, axis=-1)
        features['std'] = np.std(signal_data, axis=-1)
        features['var'] = np.var(signal_data, axis=-1)
        features['skewness'] = skew(signal_data, axis=-1)
        features['kurtosis'] = kurtosis(signal_data, axis=-1)
        features['min'] = np.min(signal_data, axis=-1)
        features['max'] = np.max(signal_data, axis=-1)
        features['range'] = features['max'] - features['min']
        features['median'] = np.median(signal_data, axis=-1)
        features['q25'] = np.percentile(signal_data, 25, axis=-1)
        features['q75'] = np.percentile(signal_data, 75, axis=-1)
        features['iqr'] = features['q75'] - features['q25']
        
        return features
    
    def extract_frequency_features(self, signal_data):
        """Extract frequency domain features"""
        features = {}
        
        # Compute power spectral density
        freqs, psd = signal.welch(signal_data, fs=self.sampling_rate, axis=-1)
        
        # EEG frequency bands (Hz)
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        for band_name, (low, high) in bands.items():
            band_idx = np.logical_and(freqs >= low, freqs <= high)
            features[f'{band_name}_power'] = np.trapezoid(psd[..., band_idx], freqs[band_idx], axis=-1)
        
        # Total power
        total_power = np.trapezoid(psd, freqs, axis=-1)
        features['total_power'] = total_power
        
        # Relative band powers
        for band_name in bands.keys():
            features[f'{band_name}_relative'] = features[f'{band_name}_power'] / (total_power + 1e-10)
        
        # Spectral entropy
        psd_norm = psd / (np.sum(psd, axis=-1, keepdims=True) + 1e-10)
        features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10), axis=-1)
        
        # Spectral edge frequency (95% power)
        cumsum_psd = np.cumsum(psd, axis=-1)
        total_psd = cumsum_psd[..., -1:]
        edge_idx = np.argmax(cumsum_psd >= 0.95 * total_psd, axis=-1)
        features['spectral_edge'] = freqs[edge_idx]
        
        return features
    
    def extract_time_features(self, signal_data):
        """Extract time domain features"""
        features = {}
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(signal_data), axis=-1) != 0, axis=-1)
        features['zero_crossing_rate'] = zero_crossings / signal_data.shape[-1]
        
        # Line length (sum of absolute differences)
        features['line_length'] = np.sum(np.abs(np.diff(signal_data, axis=-1)), axis=-1)
        
        # Energy
        features['energy'] = np.sum(signal_data ** 2, axis=-1)
        
        # First and second derivatives
        first_diff = np.diff(signal_data, axis=-1)
        second_diff = np.diff(first_diff, axis=-1)
        
        features['first_diff_mean'] = np.mean(np.abs(first_diff), axis=-1)
        features['second_diff_mean'] = np.mean(np.abs(second_diff), axis=-1)
        
        return features
    
    def extract_wavelet_features(self, signal_data):
        """Extract wavelet-based features"""
        features = {}
        
        # Process each channel separately
        if signal_data.ndim == 1:
            signal_data = signal_data.reshape(1, -1)
        
        wavelet_coeffs = []
        
        for i in range(signal_data.shape[0]):
            sig = signal_data[i]
            
            # Multi-scale analysis using different frequency bands
            coeffs = []
            for scale in [2, 4, 8, 16]:
                # Downsample to simulate wavelet decomposition at different scales
                if len(sig) >= scale:
                    downsampled = sig[::scale]
                    coeffs.extend([
                        np.mean(np.abs(downsampled)),
                        np.std(np.abs(downsampled)),
                        np.max(np.abs(downsampled))
                    ])
            
            # Aggregate statistics
            if coeffs:
                wavelet_coeffs.append([
                    np.mean(coeffs),
                    np.std(coeffs),
                    np.max(coeffs)
                ])
            else:
                wavelet_coeffs.append([0.0, 0.0, 0.0])
        
        wavelet_coeffs = np.array(wavelet_coeffs)
        features['wavelet_mean'] = wavelet_coeffs[:, 0]
        features['wavelet_std'] = wavelet_coeffs[:, 1]
        features['wavelet_max'] = wavelet_coeffs[:, 2]
        
        return features
    
    def extract_all_features(self, signal_data):
        """Extract all features from signal"""
        # Ensure 2D array (channels, time_points)
        if signal_data.ndim == 1:
            signal_data = signal_data.reshape(1, -1)
        
        # Extract different feature types
        stat_features = self.extract_statistical_features(signal_data)
        freq_features = self.extract_frequency_features(signal_data)
        time_features = self.extract_time_features(signal_data)
        wavelet_features = self.extract_wavelet_features(signal_data)
        
        # Combine all features
        feature_dict = {**stat_features, **freq_features, **time_features, **wavelet_features}
        
        # Flatten features from all channels
        feature_vector = []
        for key in sorted(feature_dict.keys()):
            val = feature_dict[key]
            if isinstance(val, np.ndarray):
                feature_vector.extend(val.flatten())
            else:
                feature_vector.append(val)
        
        return np.array(feature_vector)
    
    def load_data(self, file_path, file_format=None):
        """
        Load EEG data from various formats
        Supports: NPZ, EDF, PKL, CSV
        """
        if file_format is None:
            file_format = file_path.split('.')[-1].lower()
        
        if file_format == 'npz':
            data = np.load(file_path)
            # Try different possible key names
            if 'X' in data:
                return data['X'], data.get('y', None)
            elif 'train_signals' in data:
                return data['train_signals'], data.get('train_labels', None)
            elif 'val_signals' in data:
                return data['val_signals'], data.get('val_labels', None)
            elif 'test_signals' in data:
                return data['test_signals'], data.get('test_labels', None)
            else:
                # Return first array
                key = list(data.keys())[0]
                return data[key], None
        
        elif file_format == 'edf':
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            data = raw.get_data()
            return data, None
        
        elif file_format == 'pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                return data.get('X', data.get('data')), data.get('y', data.get('labels'))
            return data, None
        
        elif file_format == 'csv':
            df = pd.read_csv(file_path)
            
            # Check if labels exist
            y = None
            if 'label' in df.columns or 'y' in df.columns:
                label_col = 'label' if 'label' in df.columns else 'y'
                y = df[label_col].values
                X_flat = df.drop(columns=[label_col]).values
            else:
                X_flat = df.values
            
            n_samples = X_flat.shape[0]
            n_features = X_flat.shape[1]
            
            # Expected shape: (n_samples, 23 channels, 256 timepoints)
            # Flattened: 23 * 256 = 5888 features per sample
            expected_flat_size = 23 * 256
            
            if n_features == expected_flat_size:
                # Data is already in correct flattened format
                X = X_flat.reshape(n_samples, 23, 256)
                print(f"   ℹ️  Reshaped CSV from ({n_samples}, {n_features}) to ({n_samples}, 23, 256)")
            else:
                # Incompatible format
                error_msg = f"""
   ❌ ERROR: CSV data format incompatible with model requirements!
   
   Your CSV shape: ({n_samples}, {n_features})
   Expected format: ({n_samples}, 5888) where 5888 = 23 channels × 256 timepoints
   
   Required CSV structure:
   - Each row = one 1-second EEG sample
   - Each row should have 5888 columns (23 channels × 256 timepoints, flattened)
   - Optional: 'label' or 'y' column for ground truth
   
   Example: 
   First 256 columns = Channel 1 timepoints
   Next 256 columns = Channel 2 timepoints
   ... (repeat for all 23 channels)
   
   Alternative: Use NPZ or EDF format for easier data handling.
"""
                print(error_msg)
                raise ValueError(f"CSV format mismatch: expected {expected_flat_size} columns, got {n_features}")
            
            return X, y
        
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    def process_dataset(self, X):
        """Process entire dataset and extract features"""
        features_list = []
        
        for i in range(len(X)):
            features = self.extract_all_features(X[i])
            features_list.append(features)
        
        return np.array(features_list)
    
    def fit_transform(self, X):
        """Extract features and fit scaler"""
        features = self.process_dataset(X)
        self.scaler.fit(features)
        return self.scaler.transform(features)
    
    def transform(self, X):
        """Extract features and apply fitted scaler"""
        features = self.process_dataset(X)
        return self.scaler.transform(features)


def load_pipeline(feature_extractor_path='feature_extractor.pkl', 
                  model_path='best_model.pkl'):
    """Load the saved feature extractor and model"""
    try:
        with open(feature_extractor_path, 'rb') as f:
            feature_extractor = pickle.load(f)
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print("✅ Pipeline loaded successfully!")
        return feature_extractor, model
    
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find model files. Please train the model first.")
        print(f"   Missing file: {e.filename}")
        sys.exit(1)


def predict_file(file_path, file_format=None, feature_extractor=None, model=None):
    """
    Predict seizures from a file
    
    Parameters:
    -----------
    file_path : str
        Path to the EEG data file
    file_format : str, optional
        File format ('npz', 'edf', 'pkl', 'csv')
    feature_extractor : object
        Trained feature extractor
    model : object
        Trained classification model
    
    Returns:
    --------
    dict : Prediction results
    """
    # Load data
    print(f"\n📂 Loading data from: {file_path}")
    X, y_true = feature_extractor.load_data(file_path, file_format)
    print(f"   Data shape: {X.shape}")
    
    # Validate data shape
    if X.ndim == 2:
        print(f"   ⚠️  Warning: 2D data detected. Expected 3D (samples, channels, timepoints)")
        print(f"   Attempting to reshape assuming single sample with {X.shape[0]} channels and {X.shape[1]} timepoints...")
        X = X.reshape(1, X.shape[0], X.shape[1])
    elif X.ndim != 3:
        raise ValueError(f"Invalid data shape: {X.shape}. Expected 3D array (samples, channels, timepoints)")
    
    print(f"   Final shape for processing: {X.shape}")
    print(f"   ({X.shape[0]} samples, {X.shape[1]} channels, {X.shape[2]} timepoints)")
    
    # Validate channel count and timepoints match model requirements
    expected_channels = 23
    expected_timepoints = 256
    actual_channels = X.shape[1]
    actual_timepoints = X.shape[2]
    
    # Strict validation - no padding or trimming
    if actual_channels != expected_channels:
        error_msg = f"""
❌ ERROR: Channel count mismatch!
   Model expects: {expected_channels} channels
   Your data has: {actual_channels} channels
   
   This model was trained on EEG data with exactly {expected_channels} channels.
   Please ensure your input data has the same channel configuration.
   
   Supported data format:
   - Shape: (n_samples, {expected_channels} channels, {expected_timepoints} timepoints)
   - Sampling rate: 256 Hz
   - Window size: 1 second (256 timepoints)
"""
        print(error_msg)
        raise ValueError(f"Channel mismatch: expected {expected_channels}, got {actual_channels}")
    
    if actual_timepoints != expected_timepoints:
        error_msg = f"""
❌ ERROR: Timepoint count mismatch!
   Model expects: {expected_timepoints} timepoints (1 second at 256 Hz)
   Your data has: {actual_timepoints} timepoints
   
   This model was trained on 1-second EEG windows sampled at 256 Hz.
   Please ensure your input data has the same temporal configuration.
   
   Supported data format:
   - Shape: (n_samples, {expected_channels} channels, {expected_timepoints} timepoints)
   - Sampling rate: 256 Hz
   - Window size: 1 second
"""
        print(error_msg)
        raise ValueError(f"Timepoint mismatch: expected {expected_timepoints}, got {actual_timepoints}")
    
    # Extract features
    print("🔧 Extracting features...")
    try:
        features = feature_extractor.transform(X)
        print(f"   ✅ Features extracted successfully")
        print(f"   Features shape: {features.shape}")
    except Exception as e:
        error_msg = f"""
❌ Feature extraction failed!
   
   Error: {str(e)}
   
   Input data shape: {X.shape}
   Expected shape: (n_samples, 23 channels, 256 timepoints)
   
   Please ensure your data:
   1. Has exactly 23 EEG channels
   2. Has exactly 256 timepoints per sample (1 second at 256 Hz)
   3. Is in the format: (samples, channels, timepoints)
   
   Use NPZ format for best compatibility.
"""
        print(error_msg)
        raise
    
    # Predict
    print("🔮 Making predictions...")
    predictions = model.predict(features)
    
    # Try to get probabilities, use decision function if not available
    try:
        probabilities = model.predict_proba(features)
    except AttributeError:
        print("   ⚠️  Model doesn't support predict_proba, using decision_function")
        if hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(features)
            # Convert decision scores to pseudo-probabilities
            probabilities = np.column_stack([1 - decision_scores, decision_scores])
        else:
            # Fallback: use predictions as probabilities (0 or 1)
            probabilities = np.column_stack([1 - predictions, predictions])
    
    # Generate report
    seizure_count = np.sum(predictions == 1)
    non_seizure_count = np.sum(predictions == 0)
    seizure_percentage = (seizure_count / len(predictions)) * 100
    
    report = {
        'total_samples': len(predictions),
        'seizure_detected': seizure_count,
        'non_seizure': non_seizure_count,
        'seizure_percentage': seizure_percentage,
        'predictions': predictions,
        'probabilities': probabilities,
        'mean_seizure_probability': np.mean(probabilities[:, 1]),
        'max_seizure_probability': np.max(probabilities[:, 1]),
        'true_labels': y_true
    }
    
    return report


def print_report(report):
    """Print prediction report"""
    print("\n" + "="*70)
    print("📊 SEIZURE DETECTION REPORT")
    print("="*70)
    
    print(f"\n📈 Overview:")
    print(f"   Total samples analyzed    : {report['total_samples']}")
    print(f"   Seizures detected         : {report['seizure_detected']}")
    print(f"   Non-seizure samples       : {report['non_seizure']}")
    print(f"   Seizure percentage        : {report['seizure_percentage']:.2f}%")
    
    print(f"\n🎯 Confidence Metrics:")
    print(f"   Mean seizure probability  : {report['mean_seizure_probability']:.4f}")
    print(f"   Max seizure probability   : {report['max_seizure_probability']:.4f}")
    
    # If true labels are available, show accuracy
    if report['true_labels'] is not None:
        from sklearn.metrics import accuracy_score, classification_report
        
        true_labels = report['true_labels'].astype(int)
        unique_labels = np.unique(true_labels)
        
        # Check if labels are binary (0, 1) as expected
        if not np.array_equal(np.sort(unique_labels), np.array([0, 1])):
            print(f"\n⚠️  WARNING: Label format mismatch!")
            print(f"   Model expects binary labels: 0 (Non-Seizure) and 1 (Seizure)")
            print(f"   Your labels contain: {unique_labels}")
            
            if len(unique_labels) <= 2:
                print(f"   Attempting to map labels to binary...")
                # Map to 0 and 1
                label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
                true_labels = np.array([label_mapping[l] for l in true_labels])
                print(f"   ✅ Labels mapped: {unique_labels[0]}→0, {unique_labels[1]}→1")
            else:
                print(f"\n   Accuracy calculation skipped due to label mismatch.")
                print(f"   Please ensure labels are binary: 0=Non-Seizure, 1=Seizure")
                return
        
        accuracy = accuracy_score(true_labels, report['predictions'])
        print(f"\n✅ Validation (Ground truth available):")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(true_labels, 
                                   report['predictions'],
                                   target_names=['Non-Seizure', 'Seizure'],
                                   zero_division=0))
    
    # Show sample predictions
    print(f"\n🔍 Sample Predictions (first 10):")
    print(f"   {'Index':<10} {'Prediction':<15} {'Probability':<15}")
    print(f"   {'-'*40}")
    for i in range(min(10, len(report['predictions']))):
        pred_label = 'Seizure' if report['predictions'][i] == 1 else 'Non-Seizure'
        prob = report['probabilities'][i, 1]
        print(f"   {i:<10} {pred_label:<15} {prob:.4f}")
    
    print("\n" + "="*70)
    
    # Warning if high seizure percentage
    if report['seizure_percentage'] > 50:
        print("\n⚠️  WARNING: High percentage of seizure activity detected!")
        print("   Please consult with a medical professional.")
    elif report['seizure_percentage'] > 0:
        print("\n⚠️  Note: Some seizure activity detected.")
        print("   Please review the results carefully.")
    else:
        print("\n✅ No seizure activity detected in the provided data.")
    
    print("="*70)


def generate_pdf_report(report, file_path, output_pdf=None):
    """
    Generate a comprehensive PDF report with visualizations
    
    Parameters:
    -----------
    report : dict
        Prediction results dictionary
    file_path : str
        Original input file path
    output_pdf : str, optional
        Output PDF filename. If None, auto-generated based on input filename
    
    Returns:
    --------
    str : Path to generated PDF file
    """
    # Generate output filename if not provided
    if output_pdf is None:
        input_name = Path(file_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_pdf = f"seizure_detection_report_{input_name}_{timestamp}.pdf"
    
    # Ensure .pdf extension
    if not output_pdf.endswith('.pdf'):
        output_pdf += '.pdf'
    
    print(f"\n📄 Generating PDF report: {output_pdf}")
    
    # Create temporary directory for images
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create the PDF document
        doc = SimpleDocTemplate(output_pdf, pagesize=letter,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title
        elements.append(Paragraph("🧠 Seizure Detection Report", title_style))
        elements.append(Spacer(1, 12))
        
        # Metadata
        metadata_style = styles['Normal']
        elements.append(Paragraph(f"<b>Analysis Date:</b> {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", metadata_style))
        elements.append(Paragraph(f"<b>Input File:</b> {Path(file_path).name}", metadata_style))
        elements.append(Spacer(1, 20))
        
        # === SECTION 1: Overview Summary ===
        elements.append(Paragraph("📊 Overview Summary", heading_style))
        
        overview_data = [
            ['Metric', 'Value'],
            ['Total Samples Analyzed', str(report['total_samples'])],
            ['Seizure Samples Detected', str(report['seizure_detected'])],
            ['Non-Seizure Samples', str(report['non_seizure'])],
            ['Seizure Percentage', f"{report['seizure_percentage']:.2f}%"],
        ]
        
        overview_table = Table(overview_data, colWidths=[3*inch, 2*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        elements.append(overview_table)
        elements.append(Spacer(1, 20))
        
        # === SECTION 2: Pie Chart Visualization ===
        elements.append(Paragraph("📈 Detection Distribution", heading_style))
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sizes = [report['non_seizure'], report['seizure_detected']]
        labels = ['Non-Seizure', 'Seizure']
        colors_pie = ['#2ecc71', '#e74c3c']
        explode = (0, 0.1)  # Explode seizure slice
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
               autopct='%1.1f%%', shadow=True, startangle=90,
               textprops={'fontsize': 14, 'weight': 'bold'})
        ax.axis('equal')
        plt.title('Seizure vs Non-Seizure Detection', fontsize=16, weight='bold', pad=20)
        
        # Save pie chart
        pie_chart_path = os.path.join(temp_dir, 'pie_chart.png')
        plt.savefig(pie_chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Add pie chart to PDF
        elements.append(Image(pie_chart_path, width=5*inch, height=3.75*inch))
        elements.append(Spacer(1, 20))
        
        # === SECTION 3: Confidence Metrics ===
        elements.append(Paragraph("🎯 Confidence Metrics", heading_style))
        
        confidence_data = [
            ['Metric', 'Value'],
            ['Mean Seizure Probability', f"{report['mean_seizure_probability']:.4f}"],
            ['Maximum Seizure Probability', f"{report['max_seizure_probability']:.4f}"],
        ]
        
        confidence_table = Table(confidence_data, colWidths=[3*inch, 2*inch])
        confidence_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        
        elements.append(confidence_table)
        elements.append(Spacer(1, 20))
        
        # === SECTION 4: Probability Distribution ===
        elements.append(Paragraph("📊 Probability Distribution", heading_style))
        
        # Create histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        
        seizure_probs = report['probabilities'][:, 1]
        ax.hist(seizure_probs, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
        ax.set_xlabel('Seizure Probability', fontsize=12, weight='bold')
        ax.set_ylabel('Frequency', fontsize=12, weight='bold')
        ax.set_title('Distribution of Seizure Probabilities', fontsize=14, weight='bold', pad=15)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save histogram
        histogram_path = os.path.join(temp_dir, 'probability_histogram.png')
        plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        elements.append(Image(histogram_path, width=5.5*inch, height=3.4*inch))
        elements.append(Spacer(1, 20))
        
        # === SECTION 5: Sample Predictions ===
        elements.append(PageBreak())
        elements.append(Paragraph("🔍 Sample Predictions (First 20)", heading_style))
        
        # Create predictions table
        pred_data = [['Sample #', 'Prediction', 'Probability', 'Confidence']]
        
        for i in range(min(20, len(report['predictions']))):
            pred_label = 'Seizure' if report['predictions'][i] == 1 else 'Non-Seizure'
            prob = report['probabilities'][i, 1]
            
            if prob > 0.8 or prob < 0.2:
                confidence = 'High'
            elif prob > 0.6 or prob < 0.4:
                confidence = 'Medium'
            else:
                confidence = 'Low'
            
            pred_data.append([
                str(i + 1),
                pred_label,
                f"{prob:.4f}",
                confidence
            ])
        
        pred_table = Table(pred_data, colWidths=[1*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        elements.append(pred_table)
        elements.append(Spacer(1, 20))
        
        # === SECTION 6: Validation Results (if available) ===
        if report['true_labels'] is not None:
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            
            elements.append(Paragraph("✅ Validation Results (Ground Truth Available)", heading_style))
            
            true_labels = report['true_labels'].astype(int)
            unique_labels = np.unique(true_labels)
            
            # Handle label mapping if needed
            if not np.array_equal(np.sort(unique_labels), np.array([0, 1])):
                if len(unique_labels) <= 2:
                    label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
                    true_labels = np.array([label_mapping[l] for l in true_labels])
            
            accuracy = accuracy_score(true_labels, report['predictions'])
            
            # Accuracy table
            accuracy_data = [
                ['Metric', 'Value'],
                ['Overall Accuracy', f"{accuracy:.4f} ({accuracy*100:.2f}%)"],
            ]
            
            accuracy_table = Table(accuracy_data, colWidths=[3*inch, 2*inch])
            accuracy_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
            ]))
            
            elements.append(accuracy_table)
            elements.append(Spacer(1, 15))
            
            # Confusion Matrix
            cm = confusion_matrix(true_labels, report['predictions'])
            
            cm_data = [
                ['', 'Predicted Non-Seizure', 'Predicted Seizure'],
                ['Actual Non-Seizure', str(cm[0, 0]), str(cm[0, 1])],
                ['Actual Seizure', str(cm[1, 0]), str(cm[1, 1])],
            ]
            
            cm_table = Table(cm_data, colWidths=[2*inch, 1.75*inch, 1.75*inch])
            cm_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#34495e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (1, 1), (-1, -1), colors.lightblue),
            ]))
            
            elements.append(Paragraph("Confusion Matrix:", styles['Normal']))
            elements.append(Spacer(1, 10))
            elements.append(cm_table)
        
        # === SECTION 7: Clinical Recommendation ===
        elements.append(Spacer(1, 30))
        elements.append(Paragraph("⚕️ Clinical Recommendation", heading_style))
        
        if report['seizure_percentage'] > 50:
            recommendation = """
            <b><font color='red'>⚠️ HIGH ALERT:</font></b> A significant percentage ({:.1f}%) of seizure activity 
            has been detected in the analyzed data. Immediate consultation with a qualified neurologist 
            or medical professional is strongly recommended. This report should not be used as the sole 
            basis for clinical decision-making.
            """.format(report['seizure_percentage'])
        elif report['seizure_percentage'] > 0:
            recommendation = """
            <b><font color='orange'>⚠️ CAUTION:</font></b> Some seizure activity ({:.1f}%) has been detected. 
            Please review these results with a healthcare provider. Further clinical evaluation may be necessary.
            """.format(report['seizure_percentage'])
        else:
            recommendation = """
            <b><font color='green'>✅ NORMAL:</font></b> No seizure activity was detected in the provided data. 
            However, this automated analysis should be interpreted in the context of clinical findings and 
            patient history.
            """
        
        elements.append(Paragraph(recommendation, styles['Normal']))
        
        # Disclaimer
        elements.append(Spacer(1, 30))
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        
        disclaimer_text = """
        <b>DISCLAIMER:</b> This report is generated by an automated machine learning system for research and 
        educational purposes. It should NOT replace professional medical diagnosis or treatment. Always consult 
        qualified healthcare professionals for medical advice.
        """
        
        elements.append(Paragraph(disclaimer_text, disclaimer_style))
        
        # Build PDF
        doc.build(elements)
        
        print(f"   ✅ PDF report generated successfully!")
        print(f"   📍 Location: {os.path.abspath(output_pdf)}")
        
        return output_pdf
        
    finally:
        # Cleanup temporary files
        try:
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
        except:
            pass


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Predict seizures from EEG data files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_user_file.py data.npz
  python predict_user_file.py data.edf --format edf
  python predict_user_file.py data.csv --format csv
  python predict_user_file.py data.pkl --format pkl

Supported formats: NPZ, EDF, PKL, CSV
        """)
    
    parser.add_argument('file_path', type=str,
                       help='Path to the EEG data file')
    parser.add_argument('--format', type=str, default=None,
                       choices=['npz', 'edf', 'pkl', 'csv'],
                       help='File format (auto-detected if not specified)')
    parser.add_argument('--save', type=str, default=None,
                       help='Save predictions to file (NPZ format)')
    parser.add_argument('--pdf', type=str, default=None,
                       help='Output PDF report filename (auto-generated if not specified)')
    parser.add_argument('--no-pdf', action='store_true',
                       help='Skip PDF generation')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.file_path).exists():
        print(f"❌ Error: File not found: {args.file_path}")
        sys.exit(1)
    
    # Load pipeline
    print("🚀 Initializing Seizure Detection Pipeline...")
    feature_extractor, model = load_pipeline()
    
    # Predict
    try:
        report = predict_file(args.file_path, args.format, 
                            feature_extractor, model)
        
        # Print report
        print_report(report)
        
        # Generate PDF report (unless disabled)
        if not args.no_pdf:
            try:
                pdf_path = generate_pdf_report(report, args.file_path, args.pdf)
                print(f"\n✅ PDF report ready for download: {pdf_path}")
            except Exception as pdf_error:
                print(f"\n⚠️  Warning: Could not generate PDF report")
                print(f"   Error: {str(pdf_error)}")
                print(f"   Prediction results are still valid (shown above)")
        
        # Save if requested
        if args.save:
            np.savez(args.save,
                    predictions=report['predictions'],
                    probabilities=report['probabilities'])
            print(f"\n💾 Predictions saved to: {args.save}")
    
    except Exception as e:
        print(f"\n❌ Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
