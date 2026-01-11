#!/usr/bin/env python3
"""
CSV Multi-Reading Processor
==========================

Simple script to process CSV files containing multiple MAX30102 readings
from the same person and get accurate hemoglobin predictions.

CSV Format Expected:
Red,IR
115000,105000
116000,106000
...

Or with headers:
Red (a.u),Infra Red (a.u)
115000,105000
116000,106000
...
"""

import joblib
import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, Tuple

VERBOSE_DEFAULT = True


def _find_signal_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Identify likely RED and IR columns regardless of header naming."""

    red_col = None
    ir_col = None

    for col in df.columns:
        col_lower = str(col).lower()
        if 'red' in col_lower and red_col is None:
            red_col = col
        elif any(keyword in col_lower for keyword in ['ir', 'infrared', 'infra']):
            ir_col = col

        if red_col is not None and ir_col is not None:
            break

    return red_col, ir_col


def _prepare_predictions(
    df: pd.DataFrame,
    gender: str,
    age: int,
    model_data,
    red_col: str,
    ir_col: str,
    zscore_threshold: float = 2.0
):
    """Generate predictions array and metadata for summary statistics."""

    predictions = []
    gender_encoded = model_data['label_encoder'].transform([gender])[0]

    for idx, row in df.iterrows():
        try:
            input_data = np.array([[row[red_col], row[ir_col], gender_encoded, age]])
            pred = model_data['model'].predict(input_data)[0]
            predictions.append(pred)
        except Exception:
            continue

    if not predictions:
        return None, 0

    predictions = np.array(predictions)

    if len(predictions) < 2:
        clean_predictions = predictions
    else:
        z_scores = np.abs(stats.zscore(predictions))
        clean_predictions = predictions[z_scores < zscore_threshold]
    outliers_removed = int(len(predictions) - len(clean_predictions))

    if len(clean_predictions) == 0:
        return None, outliers_removed

    predictions = clean_predictions

    results = {
        'total_readings': len(df),
        'valid_predictions': len(predictions),
        'outliers_removed': outliers_removed,
        'mean_hemoglobin': np.mean(predictions),
        'median_hemoglobin': np.median(predictions),
        'std_deviation': np.std(predictions),
        'min_hemoglobin': np.min(predictions),
        'max_hemoglobin': np.max(predictions),
        'confidence_95_lower': np.percentile(predictions, 2.5),
        'confidence_95_upper': np.percentile(predictions, 97.5),
        'coefficient_variation': (np.std(predictions) / np.mean(predictions)) * 100,
        'all_predictions': predictions.tolist()
    }

    return results, outliers_removed


def process_readings_dataframe(
    df: pd.DataFrame,
    gender: str,
    age: int,
    model_path: str = 'hemoglobin_svr_model.pkl',
    *,
    zscore_threshold: float = 2.0,
    verbose: bool = VERBOSE_DEFAULT,
    model_data=None
):
    """Process readings provided as a DataFrame containing RED and IR columns."""

    if model_data is None:
        if verbose:
            print("Loading trained model...")

        model_data = joblib.load(model_path)

    if verbose:
        print(f"✅ Model loaded (Test R²: {model_data['results']['test_r2']:.4f})")

    red_col, ir_col = _find_signal_columns(df)

    if red_col is None or ir_col is None:
        if verbose:
            print("❌ Could not find Red and IR columns in provided readings")
            print(f"Available columns: {list(df.columns)}")
        return None

    if verbose:
        print(f"Using columns: '{red_col}' and '{ir_col}'")
        print("Making predictions...")

    results, outliers_removed = _prepare_predictions(
        df,
        gender,
        age,
        model_data,
        red_col,
        ir_col,
        zscore_threshold=zscore_threshold
    )

    if results is None:
        if verbose:
            print("❌ No valid predictions made")
        return None

    if verbose and outliers_removed > 0:
        print(f"🧹 Removed {outliers_removed} outlier readings")

    return results

def process_csv_readings(csv_file, gender, age, model_path='hemoglobin_svr_model.pkl'):
    """
    Process multiple readings from a CSV file
    
    Args:
        csv_file (str): Path to CSV file with Red and IR columns
        gender (str): 'Male' or 'Female'  
        age (int): Age in years
        model_path (str): Path to trained model
        
    Returns:
        dict: Prediction results
    """

    # Load CSV data
    print(f"Loading readings from {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} readings")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None
    
    return process_readings_dataframe(
        df,
        gender,
        age,
        model_path=model_path,
        verbose=True
    )

def format_summary(results, gender, age):
    """Return a formatted summary string for prediction results."""

    if results is None:
        return ""

    lines = []
    lines.append("=" * 50)
    lines.append("🩸 HEMOGLOBIN PREDICTION SUMMARY")
    lines.append("=" * 50)
    lines.append(f"Patient: {gender}, {age} years old")
    lines.append(f"Total readings: {results['total_readings']}")
    lines.append(f"Valid predictions: {results['valid_predictions']}")
    lines.append(f"Outliers removed: {results['outliers_removed']}")

    lines.append("")
    lines.append("📊 RESULTS:")
    lines.append(f"Mean Hemoglobin:    {results['mean_hemoglobin']:.2f} g/dL")
    lines.append(f"Median Hemoglobin:  {results['median_hemoglobin']:.2f} g/dL")
    lines.append(f"Standard Deviation: {results['std_deviation']:.3f} g/dL")
    lines.append(f"Range:              {results['min_hemoglobin']:.2f} - {results['max_hemoglobin']:.2f} g/dL")

    lines.append("")
    lines.append("🎯 95% CONFIDENCE INTERVAL:")
    lines.append(f"{results['confidence_95_lower']:.2f} - {results['confidence_95_upper']:.2f} g/dL")

    cv = results['coefficient_variation']
    if cv < 2:
        quality = "Excellent (very consistent)"
    elif cv < 5:
        quality = "Good (consistent)"
    elif cv < 10:
        quality = "Fair (moderate variation)"
    else:
        quality = "Poor (high variation)"

    lines.append("")
    lines.append(f"⭐ MEASUREMENT QUALITY: {quality}")
    lines.append(f"Coefficient of Variation: {cv:.1f}%")

    if cv < 5:
        recommended = results['mean_hemoglobin']
        method = "Use Mean value"
    else:
        recommended = results['median_hemoglobin']
        method = "Use Median value (reduces outlier impact)"

    lines.append("")
    lines.append(f"🏥 RECOMMENDED VALUE: {recommended:.2f} g/dL")
    lines.append(f"Method: {method}")

    if gender == 'Male':
        normal_range = "13.8 - 17.2 g/dL"
        if 13.8 <= recommended <= 17.2:
            status = "✅ Normal"
        elif recommended < 13.8:
            status = "⚠️ Below normal"
        else:
            status = "⚠️ Above normal"
    else:
        normal_range = "12.1 - 15.1 g/dL"
        if 12.1 <= recommended <= 15.1:
            status = "✅ Normal"
        elif recommended < 12.1:
            status = "⚠️ Below normal"
        else:
            status = "⚠️ Above normal"

    lines.append(f"Clinical Status: {status}")
    lines.append(f"Normal range for {gender.lower()}s: {normal_range}")
    lines.append("=" * 50)

    return "\n".join(lines)


def print_summary(results, gender, age):
    """Print summary of results using formatted text."""
    summary = format_summary(results, gender, age)
    if summary:
        print(f"\n{summary}")

def create_sample_csv():
    """Create a sample CSV file for testing"""
    np.random.seed(42)
    n_readings = 120
    
    # Simulate realistic readings with noise
    base_red = 115000 
    base_ir = 105000
    
    red_readings = base_red + np.random.normal(0, 1500, n_readings)
    ir_readings = base_ir + np.random.normal(0, 1200, n_readings)
    
    # Add some drift over time
    drift_red = np.linspace(0, 2000, n_readings)
    drift_ir = np.linspace(0, 1500, n_readings)
    
    red_readings += drift_red
    ir_readings += drift_ir
    
    # Create DataFrame and save
    df = pd.DataFrame({
        'Red': red_readings.astype(int),
        'IR': ir_readings.astype(int)
    })
    
    filename = 'sample_multiple_readings.csv'
    df.to_csv(filename, index=False)
    print(f"✅ Created sample CSV: {filename}")
    return filename

def main():
    """Main function"""
    print("🩸 CSV MULTI-READING HEMOGLOBIN PREDICTOR")
    print("="*45)
    
    print("\nOptions:")
    print("1. Process your CSV file")
    print("2. Create and process sample CSV")
    
    choice = input("Select option (1-2): ").strip()
    
    if choice == '1':
        csv_file = input("Enter path to CSV file: ").strip()
        gender = input("Enter gender (Male/Female): ").strip().capitalize()
        
        try:
            age = int(input("Enter age: "))
        except ValueError:
            print("Invalid age")
            return
            
    elif choice == '2':
        csv_file = create_sample_csv()
        gender = 'Male'
        age = 28
        print(f"Using sample data: {gender}, {age} years old")
    else:
        print("Invalid choice")
        return
    
    # Process the readings
    results = process_csv_readings(csv_file, gender, age)
    
    # Print summary
    print_summary(results, gender, age)
    
    # Save detailed results
    if results:
        output_file = f"hemoglobin_results_{gender.lower()}_{age}.txt"
        with open(output_file, 'w') as f:
            f.write(f"Hemoglobin Prediction Results\n")
            f.write(f"Patient: {gender}, {age} years\n")
            f.write(f"Recommended Hemoglobin: {results['mean_hemoglobin']:.2f} g/dL\n")
            f.write(f"95% Confidence Interval: {results['confidence_95_lower']:.2f} - {results['confidence_95_upper']:.2f} g/dL\n")
            f.write(f"All predictions: {results['all_predictions']}\n")
        
        print(f"\n💾 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()