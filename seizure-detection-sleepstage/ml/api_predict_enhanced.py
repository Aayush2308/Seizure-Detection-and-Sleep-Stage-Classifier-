"""
API Wrapper for Seizure Detection Model with PDF Report Generation
"""

import sys
import json
import os
from pathlib import Path
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add the ml directory to the path
ml_dir = Path(__file__).parent
sys.path.insert(0, str(ml_dir))

from predict_user_file import EEGFeatureExtractor

def generate_pdf_report(result, file_name, report_path, analysis_type="seizure"):
    """Generate a PDF report with visualizations"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        from datetime import datetime
        import tempfile
        
        doc = SimpleDocTemplate(report_path, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title
        title_text = "Seizure Detection Analysis Report" if analysis_type == "seizure" else "Sleep Stage Classification Report"
        story.append(Paragraph(title_text, title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Date and file info
        date_text = f"<b>Analysis Date:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
        story.append(Paragraph(date_text, styles['Normal']))
        story.append(Paragraph(f"<b>File Name:</b> {file_name}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Results
        story.append(Paragraph("Analysis Results", heading_style))
        
        prediction = result.get('prediction', 'Unknown')
        confidence = result.get('confidence', 0)
        
        result_data = [
            ['Prediction', prediction],
            ['Confidence Score', f"{confidence:.2f}%"],
            ['Features Extracted', str(result.get('features_extracted', 'N/A'))],
        ]
        
        result_table = Table(result_data, colWidths=[2.5*inch, 3.5*inch])
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e0e7ff')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        
        story.append(result_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Pie chart
        probabilities = result.get('probabilities', {})
        if probabilities:
            story.append(Paragraph("Probability Distribution", heading_style))
            
            fig, ax = plt.subplots(figsize=(6, 6))
            
            labels = []
            sizes = []
            colors_list = []
            
            if analysis_type == "seizure":
                if 'normal' in probabilities:
                    labels.append('Normal')
                    sizes.append(probabilities['normal'])
                    colors_list.append('#10b981')
                if 'seizure' in probabilities:
                    labels.append('Seizure')
                    sizes.append(probabilities['seizure'])
                    colors_list.append('#ef4444')
            else:
                for stage, prob in probabilities.items():
                    labels.append(stage.upper())
                    sizes.append(prob)
                colors_list = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981'][:len(labels)]
            
            if sizes:
                wedges, texts, autotexts = ax.pie(
                    sizes, 
                    labels=labels, 
                    autopct='%1.1f%%',
                    colors=colors_list,
                    startangle=90,
                    textprops={'fontsize': 12, 'weight': 'bold'}
                )
                
                ax.axis('equal')
                plt.title('Prediction Probabilities', fontsize=14, weight='bold', pad=20)
                
                temp_chart = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                plt.savefig(temp_chart.name, bbox_inches='tight', dpi=150)
                plt.close()
                
                story.append(Image(temp_chart.name, width=4*inch, height=4*inch))
                story.append(Spacer(1, 0.3*inch))
                
                try:
                    os.unlink(temp_chart.name)
                except:
                    pass
        
        # Interpretation
        story.append(Paragraph("Interpretation", heading_style))
        
        if analysis_type == "seizure":
            if result.get('prediction_value') == 1:
                interpretation = """
                <b>Seizure Detected:</b> The analysis indicates the presence of epileptic seizure patterns 
                in the EEG data. Further clinical evaluation is recommended.
                """
            else:
                interpretation = """
                <b>Normal Reading:</b> The analysis indicates normal EEG patterns with no evidence 
                of seizure activity detected.
                """
        else:
            interpretation = f"<b>Sleep Stage: {prediction}</b><br/>The analysis has classified the sleep stage based on EEG patterns."
        
        story.append(Paragraph(interpretation, styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Disclaimer
        story.append(Paragraph("Important Notice", heading_style))
        disclaimer = """
        <b>DISCLAIMER:</b> This report is for research and educational purposes only. 
        Not for medical diagnosis. Consult a healthcare provider for medical concerns.
        """
        story.append(Paragraph(disclaimer, styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 0.5*inch))
        footer_text = f"Generated by NeuroDetect AI | {datetime.now().strftime('%Y')}"
        footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=10, textColor=colors.grey, alignment=TA_CENTER)
        story.append(Paragraph(footer_text, footer_style))
        
        doc.build(story)
        return True
        
    except Exception as e:
        print(f"Error generating PDF: {str(e)}", file=sys.stderr)
        return False

def load_file_data(file_path, file_format):
    """Load data from various file formats"""
    try:
        if file_format == 'npz':
            data = np.load(file_path)
            # Try common key names
            loaded_data = None
            for key in ['X', 'train_signals', 'val_signals', 'test_signals', 'data', 'signals', 'eeg']:
                if key in data:
                    loaded_data = data[key]
                    break
            
            # Return first array if no standard key found
            if loaded_data is None:
                loaded_data = data[list(data.keys())[0]]
            
            return loaded_data
        
        elif file_format == 'edf':
            import mne
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            edf_data = raw.get_data()
            # EDF usually has shape (n_channels, n_timepoints)
            # Reshape to (1, n_channels, n_timepoints) for consistency
            if edf_data.ndim == 2:
                edf_data = edf_data.reshape(1, *edf_data.shape)
            return edf_data
        
        elif file_format == 'csv':
            import pandas as pd
            df = pd.read_csv(file_path)
            csv_data = df.values
            # Reshape to 3D if needed
            if csv_data.ndim == 2:
                csv_data = csv_data.reshape(1, *csv_data.shape)
            return csv_data
        
        elif file_format == 'pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            loaded_data = None
            if isinstance(data, dict):
                for key in ['X', 'data', 'signals', 'eeg']:
                    if key in data:
                        loaded_data = data[key]
                        break
                if loaded_data is None:
                    loaded_data = list(data.values())[0]
            else:
                loaded_data = data
            
            return loaded_data
        
        else:
            return None
            
    except Exception as e:
        print(f"Error loading file: {str(e)}", file=sys.stderr)
        return None

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No file path provided", "success": False}))
        sys.exit(1)
    
    file_path = sys.argv[1]
    file_format = None
    report_path = None
    analysis_type = "seizure"
    
    # Parse arguments
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
    
    if not os.path.exists(file_path):
        print(json.dumps({"error": "File not found", "success": False}))
        sys.exit(1)
    
    try:
        # Load model
        model_path = ml_dir / "best_model.pkl"
        feature_extractor_path = ml_dir / "feature_extractor.pkl"
        
        if not model_path.exists():
            print(json.dumps({"error": "Model file not found. Please train the model first.", "success": False}))
            sys.exit(1)
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        if feature_extractor_path.exists():
            with open(feature_extractor_path, 'rb') as f:
                feature_extractor = pickle.load(f)
        else:
            feature_extractor = EEGFeatureExtractor()
        
        # Determine format
        if not file_format:
            file_format = Path(file_path).suffix[1:]
        
        # Load data
        data = load_file_data(file_path, file_format)
        
        if data is None:
            print(json.dumps({"error": f"Failed to load file in {file_format} format", "success": False}))
            sys.exit(1)
        
        # Process data - handle both single and multiple samples
        if data.ndim == 3:
            # Multiple samples - process all and average
            print(f"Processing {data.shape[0]} samples from file", file=sys.stderr)
            features = feature_extractor.transform(data)
            
            # Predict on all samples
            predictions = model.predict(features)
            prediction_probas = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
            
            # Calculate statistics
            seizure_count = int(np.sum(predictions == 1))
            total_samples = int(len(predictions))
            seizure_percentage = float((seizure_count / total_samples) * 100)
            
            # Average confidence
            if prediction_probas is not None:
                avg_confidence = float(np.mean(np.max(prediction_probas, axis=1)) * 100)
                avg_seizure_prob = float(np.mean(prediction_probas[:, 1]) * 100) if prediction_probas.shape[1] > 1 else 0
                avg_normal_prob = float(np.mean(prediction_probas[:, 0]) * 100)
            else:
                avg_confidence = None
                avg_seizure_prob = None
                avg_normal_prob = None
            
            # Overall prediction based on majority
            final_prediction = 1 if seizure_percentage > 50 else 0
            
            result = {
                "success": True,
                "prediction": "Seizure Detected" if final_prediction == 1 else "Normal (No Seizure)",
                "prediction_value": int(final_prediction),
                "confidence": avg_confidence,
                "file_format": file_format,
                "features_extracted": int(features.shape[1]),
                "total_samples": total_samples,
                "seizure_samples": seizure_count,
                "seizure_percentage": seizure_percentage,
                "message": f"Analyzed {total_samples} samples. Found {seizure_count} seizure events ({seizure_percentage:.1f}%)"
            }
            
            if prediction_probas is not None:
                result["probabilities"] = {
                    "normal": avg_normal_prob,
                    "seizure": avg_seizure_prob
                }
        
        else:
            # Single sample
            features = feature_extractor.extract_all_features(data)
            features = features.reshape(1, -1)
            
            # Predict
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
            
            if prediction_proba is not None:
                result["probabilities"] = {
                    "normal": float(prediction_proba[0] * 100),
                    "seizure": float(prediction_proba[1] * 100) if len(prediction_proba) > 1 else 0
                }
        
        # Generate PDF report if requested
        if report_path:
            pdf_generated = generate_pdf_report(result, Path(file_path).name, report_path, analysis_type)
            result["report_generated"] = pdf_generated
        
        print(json.dumps(result))
        
    except ImportError as e:
        print(json.dumps({
            "error": f"Missing required Python package: {str(e)}",
            "success": False,
            "suggestion": "Please install: pip install numpy pandas scikit-learn mne scipy matplotlib reportlab"
        }))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Analysis failed: {str(e)}", "success": False}))
        sys.exit(1)

if __name__ == "__main__":
    main()
