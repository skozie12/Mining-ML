"""
Simple interactive script to test mining permit predictions.
"""

import pandas as pd
import joblib
from pathlib import Path

# Load model and preprocessors
model = joblib.load("models/best_model_gradient_boosting.joblib")
scaler = joblib.load("models/scaler.joblib")
label_encoders = joblib.load("models/label_encoders.joblib")

with open("models/feature_names.txt", 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

def predict_permit(
    province='Ontario',
    mining_type='Open-pit',
    mineral_type='Au',
    company_size='Large',
    project_area=200.0,
    estimated_duration=12,
    distance_to_water=3.0,
    distance_to_protected_area=10.0,
    distance_to_indigenous_land=8.0,
    expected_employment=250,
    environmental_assessment_score=80.0,
    public_comments_received=30,
    public_opposition_percentage=20.0,
    company_compliance_history=85.0,
    previous_permits=3
):
    """
    Predict mining permit approval confidence.
    
    Args:
        province: Province/territory (e.g., 'Ontario', 'British Columbia')
        mining_type: 'Open-pit' or 'Underground'
        mineral_type: Type of mineral (e.g., 'Au', 'Cu', 'Fe', 'Ni', 'Zn')
        company_size: 'Small', 'Medium', 'Large', or 'Very Large'
        project_area: Project area in hectares (e.g., 200.0)
        estimated_duration: Project duration in years (e.g., 12)
        distance_to_water: Distance to water in km (e.g., 3.0)
        distance_to_protected_area: Distance to protected area in km (e.g., 10.0)
        distance_to_indigenous_land: Distance to indigenous land in km (e.g., 8.0)
        expected_employment: Number of jobs (e.g., 250)
        environmental_assessment_score: Score 0-100 (e.g., 80.0)
        public_comments_received: Number of public comments (e.g., 30)
        public_opposition_percentage: Opposition percentage 0-100 (e.g., 20.0)
        company_compliance_history: Score 0-100 (e.g., 85.0)
        previous_permits: Number of previous permits (e.g., 3)
    
    Returns:
        prediction: 'High', 'Medium', or 'Low'
        probabilities: Dictionary of probabilities for each class
    """
    
    # Create record
    record = {
        'province': province,
        'mining_type': mining_type,
        'mineral_type': mineral_type,
        'company_size': company_size,
        'project_area': project_area,
        'estimated_duration': estimated_duration,
        'distance_to_water': distance_to_water,
        'distance_to_protected_area': distance_to_protected_area,
        'distance_to_indigenous_land': distance_to_indigenous_land,
        'expected_employment': expected_employment,
        'environmental_assessment_score': environmental_assessment_score,
        'public_comments_received': public_comments_received,
        'public_opposition_percentage': public_opposition_percentage,
        'company_compliance_history': company_compliance_history,
        'previous_permits': previous_permits
    }
    
    # Prepare data
    df_test = pd.DataFrame([record])
    
    # Encode categorical features
    for col in ['province', 'mining_type', 'mineral_type', 'company_size']:
        if col in label_encoders:
            try:
                df_test[col + '_encoded'] = label_encoders[col].transform(df_test[col].astype(str))
            except ValueError:
                df_test[col + '_encoded'] = 0
    
    # Create feature vector
    X_test = df_test[feature_names]
    X_test_scaled = scaler.transform(X_test)
    
    # Predict
    prediction = model.predict(X_test_scaled)[0]
    prediction_proba = model.predict_proba(X_test_scaled)[0]
    
    # Format results
    classes = model.classes_
    probabilities = dict(zip(classes, prediction_proba))
    
    # Print results
    print("\n" + "="*70)
    print("🔮 MINING PERMIT APPROVAL PREDICTION")
    print("="*70)
    print(f"\n📍 Location: {province}")
    print(f"⛏️  Mining Type: {mining_type}")
    print(f"💎 Mineral: {mineral_type}")
    print(f"🏢 Company Size: {company_size}")
    print(f"📏 Project Area: {project_area} hectares")
    print(f"⏱️  Duration: {estimated_duration} years")
    print(f"💧 Distance to Water: {distance_to_water} km")
    print(f"🌳 Distance to Protected Area: {distance_to_protected_area} km")
    print(f"🏘️  Distance to Indigenous Land: {distance_to_indigenous_land} km")
    print(f"👥 Expected Employment: {expected_employment} jobs")
    print(f"🌍 Environmental Score: {environmental_assessment_score}/100")
    print(f"💬 Public Comments: {public_comments_received}")
    print(f"📊 Public Opposition: {public_opposition_percentage}%")
    print(f"✅ Company Compliance: {company_compliance_history}/100")
    print(f"📋 Previous Permits: {previous_permits}")
    
    print("\n" + "-"*70)
    print(f"🎯 PREDICTED APPROVAL CONFIDENCE: {prediction}")
    print("-"*70)
    print("\n📊 Confidence Breakdown:")
    for cls in ['High', 'Medium', 'Low']:
        if cls in probabilities:
            prob = probabilities[cls]
            bar = "█" * int(prob * 40)
            print(f"  {cls:.<10} {prob:>6.1%} {bar}")
    
    print("\n💡 Interpretation:")
    if prediction == 'High':
        print("  ✅ Strong likelihood of approval - favorable conditions")
    elif prediction == 'Medium':
        print("  ⚠️  Moderate likelihood - some concerns present")
    else:
        print("  ❌ Low likelihood - significant concerns present")
    
    print("\n" + "="*70)
    
    return prediction, probabilities


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 MINING PERMIT APPROVAL PREDICTION TOOL")
    print("="*70)
    print("\nExample 1: Ideal Gold Mine Project")
    
    # Example 1: Ideal project
    predict_permit(
        province='Ontario',
        mining_type='Underground',
        mineral_type='Au',
        company_size='Large',
        project_area=100.0,
        estimated_duration=10,
        distance_to_water=5.0,
        distance_to_protected_area=15.0,
        distance_to_indigenous_land=20.0,
        expected_employment=400,
        environmental_assessment_score=95.0,
        public_comments_received=20,
        public_opposition_percentage=5.0,
        company_compliance_history=98.0,
        previous_permits=8
    )
    
    print("\n\n" + "="*70)
    print("Example 2: Challenging Copper Mine Project")
    
    # Example 2: Challenging project
    predict_permit(
        province='British Columbia',
        mining_type='Open-pit',
        mineral_type='Cu',
        company_size='Small',
        project_area=500.0,
        estimated_duration=25,
        distance_to_water=0.3,
        distance_to_protected_area=0.5,
        distance_to_indigenous_land=0.8,
        expected_employment=50,
        environmental_assessment_score=45.0,
        public_comments_received=200,
        public_opposition_percentage=75.0,
        company_compliance_history=55.0,
        previous_permits=0
    )
    
    print("\n\n" + "="*70)
    print("📚 HOW TO USE THIS TOOL")
    print("="*70)
    print("""
You can use this as a Python module:

    from predict_model import predict_permit
    
    prediction, probabilities = predict_permit(
        province='Quebec',
        mining_type='Underground',
        mineral_type='Ni',
        company_size='Medium',
        project_area=150.0,
        environmental_assessment_score=70.0,
        public_opposition_percentage=30.0,
        # ... other parameters
    )
    
    print(f"Prediction: {prediction}")
    print(f"Probabilities: {probabilities}")

Available provinces: Ontario, British Columbia, Nova Scotia, Saskatchewan,
                    Yukon, Quebec, Newfoundland and Labrador, Nunavut
                    
Available minerals: Au (Gold), Cu (Copper), Ni (Nickel), Zn (Zinc), 
                   Ag (Silver), Fe (Iron), and many more...
    """)
