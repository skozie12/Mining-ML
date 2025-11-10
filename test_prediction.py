"""
Test the trained model with sample mining permit records.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

print("="*70)
print("🔮 MINING PERMIT APPROVAL PREDICTION - TEST")
print("="*70)

# Load the trained model and preprocessors
print("\n📦 Loading trained model and preprocessors...")
model_dir = Path("models")

model = joblib.load(model_dir / "best_model_gradient_boosting.joblib")
scaler = joblib.load(model_dir / "scaler.joblib")
label_encoders = joblib.load(model_dir / "label_encoders.joblib")

with open(model_dir / "feature_names.txt", 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

print("✅ Model loaded successfully!")

# Define a sample test record
print("\n" + "="*70)
print("📋 TEST RECORD #1: Hypothetical Gold Mine in Ontario")
print("="*70)

test_record_1 = {
    'province': 'Ontario',
    'mining_type': 'Open-pit',
    'mineral_type': 'Au',  # Gold
    'company_size': 'Large',
    'project_area': 250.5,  # hectares
    'estimated_duration': 15,  # years
    'distance_to_water': 2.5,  # km
    'distance_to_protected_area': 8.0,  # km
    'distance_to_indigenous_land': 5.0,  # km
    'expected_employment': 350,  # jobs
    'environmental_assessment_score': 85.0,  # out of 100
    'public_comments_received': 45,
    'public_opposition_percentage': 15.0,  # %
    'company_compliance_history': 92.0,  # out of 100
    'previous_permits': 5
}

# Define another test record
test_record_2 = {
    'province': 'British Columbia',
    'mining_type': 'Underground',
    'mineral_type': 'Cu',  # Copper
    'company_size': 'Medium',
    'project_area': 150.0,
    'estimated_duration': 10,
    'distance_to_water': 0.5,  # Very close to water
    'distance_to_protected_area': 1.0,  # Very close to protected area
    'distance_to_indigenous_land': 0.8,  # Very close to indigenous land
    'expected_employment': 150,
    'environmental_assessment_score': 55.0,  # Lower score
    'public_comments_received': 120,
    'public_opposition_percentage': 45.0,  # High opposition
    'company_compliance_history': 65.0,  # Lower compliance
    'previous_permits': 1
}

def predict_approval(record, record_name="Test Record"):
    """Make a prediction for a single record."""
    
    print(f"\n{record_name}:")
    print("-" * 70)
    
    # Display record details
    for key, value in record.items():
        print(f"  {key:.<40} {value}")
    
    # Prepare the data
    df_test = pd.DataFrame([record])
    
    # Encode categorical features
    categorical_features = ['province', 'mining_type', 'mineral_type', 'company_size']
    
    for col in categorical_features:
        if col in label_encoders:
            try:
                df_test[col + '_encoded'] = label_encoders[col].transform(df_test[col].astype(str))
            except ValueError as e:
                print(f"\n⚠️  Warning: Unknown value for '{col}': {df_test[col].values[0]}")
                print(f"   Using default encoding...")
                df_test[col + '_encoded'] = 0
    
    # Create feature vector in the correct order
    X_test = df_test[feature_names]
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Make prediction
    prediction = model.predict(X_test_scaled)[0]
    prediction_proba = model.predict_proba(X_test_scaled)[0]
    
    # Get class labels
    classes = model.classes_
    
    # Display results
    print("\n" + "🎯 PREDICTION RESULTS".center(70, "-"))
    print(f"\n  ✨ Predicted Approval Confidence: {prediction}")
    print(f"\n  📊 Confidence Probabilities:")
    for cls, prob in zip(classes, prediction_proba):
        bar = "█" * int(prob * 50)
        print(f"     {cls:.<15} {prob:>6.2%} {bar}")
    
    return prediction, prediction_proba


# Test with both records
print("\n" + "="*70)
print("🧪 TESTING MODEL WITH SAMPLE RECORDS")
print("="*70)

prediction_1, proba_1 = predict_approval(test_record_1, "TEST RECORD #1")
print("\n" + "="*70)
prediction_2, proba_2 = predict_approval(test_record_2, "TEST RECORD #2")

# Summary
print("\n" + "="*70)
print("📊 SUMMARY")
print("="*70)
print(f"\nRecord #1 (Strong application): {prediction_1}")
print(f"Record #2 (Weaker application): {prediction_2}")

print("\n💡 INTERPRETATION:")
print("  • HIGH confidence = Strong approval likelihood")
print("  • MEDIUM confidence = Moderate approval likelihood")
print("  • LOW confidence = Weak approval likelihood or likely rejection")

print("\n" + "="*70)
print("✅ Testing Complete!")
print("="*70)

# Interactive mode
print("\n" + "="*70)
print("🔧 CUSTOM PREDICTION")
print("="*70)
print("\nYou can also test with a record from the dataset:")
print("\nTo test with a specific record, modify test_record_1 or test_record_2")
print("in the script, or load a record from FORMATTED_NRCAN_NATIONAL.csv")

# Example: Load a real record from the dataset
print("\n" + "="*70)
print("📋 TESTING WITH REAL RECORD FROM DATASET")
print("="*70)

df = pd.read_csv("data/raw/FORMATTED_NRCAN_NATIONAL.csv")

# Pick a random record
sample_record = df.sample(1).iloc[0]

real_record = {
    'province': sample_record['province'],
    'mining_type': sample_record['mining_type'],
    'mineral_type': sample_record['mineral_type'],
    'company_size': sample_record['company_size'],
    'project_area': sample_record['project_area'],
    'estimated_duration': sample_record['estimated_duration'],
    'distance_to_water': sample_record['distance_to_water'],
    'distance_to_protected_area': sample_record['distance_to_protected_area'],
    'distance_to_indigenous_land': sample_record['distance_to_indigenous_land'],
    'expected_employment': sample_record['expected_employment'],
    'environmental_assessment_score': sample_record['environmental_assessment_score'],
    'public_comments_received': sample_record['public_comments_received'],
    'public_opposition_percentage': sample_record['public_opposition_percentage'],
    'company_compliance_history': sample_record['company_compliance_history'],
    'previous_permits': sample_record['previous_permits']
}

actual_confidence = sample_record['approval_confidence']

prediction_3, proba_3 = predict_approval(real_record, "REAL RECORD FROM DATASET")
print(f"\n  📌 Actual Confidence from Dataset: {actual_confidence}")
print(f"  🎯 Model Prediction: {prediction_3}")
print(f"  {'✅ CORRECT!' if prediction_3 == actual_confidence else '❌ INCORRECT'}")

print("\n" + "="*70)
