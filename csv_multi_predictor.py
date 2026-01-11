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
    
    # Load model
    print("Loading trained model...")
    model_data = joblib.load(model_path)
    print(f"✅ Model loaded (Test R²: {model_data['results']['test_r2']:.4f})")
    
    # Load CSV data
    print(f"Loading readings from {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} readings")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None
    
    # Handle different column name formats
    red_col = None
    ir_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'red' in col_lower and red_col is None:
            red_col = col
        elif 'ir' in col_lower or 'infrared' in col_lower or 'infra' in col_lower:
            ir_col = col
    
    if red_col is None or ir_col is None:
        print("❌ Could not find Red and IR columns in CSV")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    print(f"Using columns: '{red_col}' and '{ir_col}'")
    
    # Make predictions for each reading
    predictions = []
    gender_encoded = model_data['label_encoder'].transform([gender])[0]
    
    print("Making predictions...")
    for idx, row in df.iterrows():
        try:
            input_data = np.array([[row[red_col], row[ir_col], gender_encoded, age]])
            pred = model_data['model'].predict(input_data)[0]
            predictions.append(pred)
        except Exception as e:
            print(f"⚠️ Skipping row {idx}: {e}")
    
    if len(predictions) == 0:
        print("❌ No valid predictions made")
        return None
    
    predictions = np.array(predictions)
    
    # Remove outliers (optional)
    z_scores = np.abs(stats.zscore(predictions))
    clean_predictions = predictions[z_scores < 2.0]  # Remove readings > 2 std dev
    outliers_removed = len(predictions) - len(clean_predictions)
    
    if outliers_removed > 0:
        print(f"🧹 Removed {outliers_removed} outlier readings")
        predictions = clean_predictions
    
    # Calculate statistics
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
    
    return results

def print_summary(results, gender, age):
    """Print summary of results"""
    if results is None:
        return
    
    print(f"\n" + "="*50)
    print(f"🩸 HEMOGLOBIN PREDICTION SUMMARY")
    print(f"="*50)
    print(f"Patient: {gender}, {age} years old")
    print(f"Total readings: {results['total_readings']}")
    print(f"Valid predictions: {results['valid_predictions']}")
    print(f"Outliers removed: {results['outliers_removed']}")
    
    print(f"\n📊 RESULTS:")
    print(f"Mean Hemoglobin:    {results['mean_hemoglobin']:.2f} g/dL")
    print(f"Median Hemoglobin:  {results['median_hemoglobin']:.2f} g/dL") 
    print(f"Standard Deviation: {results['std_deviation']:.3f} g/dL")
    print(f"Range:              {results['min_hemoglobin']:.2f} - {results['max_hemoglobin']:.2f} g/dL")
    
    print(f"\n🎯 95% CONFIDENCE INTERVAL:")
    print(f"{results['confidence_95_lower']:.2f} - {results['confidence_95_upper']:.2f} g/dL")
    
    # Quality assessment
    cv = results['coefficient_variation']
    if cv < 2:
        quality = "Excellent (very consistent)"
    elif cv < 5:
        quality = "Good (consistent)"  
    elif cv < 10:
        quality = "Fair (moderate variation)"
    else:
        quality = "Poor (high variation)"
    
    print(f"\n⭐ MEASUREMENT QUALITY: {quality}")
    print(f"Coefficient of Variation: {cv:.1f}%")
    
    # Recommendation
    if cv < 5:
        recommended = results['mean_hemoglobin']
        method = "Use Mean value"
    else:
        recommended = results['median_hemoglobin'] 
        method = "Use Median value (reduces outlier impact)"
    
    print(f"\n🏥 RECOMMENDED VALUE: {recommended:.2f} g/dL")
    print(f"Method: {method}")
    
    # Clinical interpretation
    if gender == 'Male':
        normal_range = "13.8 - 17.2 g/dL"
        if 13.8 <= recommended <= 17.2:
            status = "✅ Normal"
        elif recommended < 13.8:
            status = "⚠️ Below normal"
        else:
            status = "⚠️ Above normal"
    else:  # Female
        normal_range = "12.1 - 15.1 g/dL"
        if 12.1 <= recommended <= 15.1:
            status = "✅ Normal"
        elif recommended < 12.1:
            status = "⚠️ Below normal"
        else:
            status = "⚠️ Above normal"
    
    print(f"Clinical Status: {status}")
    print(f"Normal range for {gender.lower()}s: {normal_range}")
    print(f"="*50)

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