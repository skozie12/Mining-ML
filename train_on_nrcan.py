"""
Train ML models on the FORMATTED_NRCAN_NATIONAL dataset.
This script loads the real Canadian mining data and trains approval prediction models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

print("="*70)
print("🎯 TRAINING MINING PERMIT APPROVAL PREDICTION MODELS")
print("="*70)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load data
print("\n📊 Loading data...")
data_path = Path("data/raw/FORMATTED_NRCAN_NATIONAL.csv")
df = pd.read_csv(data_path)

print(f"✅ Loaded {len(df)} records from {data_path}")
print(f"\n📋 Columns: {list(df.columns)}")
print(f"\n📊 Province distribution:")
print(df['province'].value_counts())

# Prepare features and target
print("\n🔧 Preparing features and target variable...")

# Define features to use
numeric_features = [
    'project_area', 'estimated_duration', 'distance_to_water',
    'distance_to_protected_area', 'distance_to_indigenous_land',
    'expected_employment', 'environmental_assessment_score',
    'public_comments_received', 'public_opposition_percentage',
    'company_compliance_history', 'previous_permits'
]

categorical_features = [
    'province', 'mining_type', 'mineral_type', 'company_size'
]

# Target variable
target = 'approval_confidence'  # High/Medium/Low

# Check target distribution
print(f"\n📊 Target variable ('{target}') distribution:")
print(df[target].value_counts())

# Encode categorical features
df_encoded = df.copy()
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    df_encoded[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"  Encoded '{col}': {len(le.classes_)} unique values")

# Prepare feature matrix
feature_columns = numeric_features + [f"{col}_encoded" for col in categorical_features]
X = df_encoded[feature_columns]
y = df_encoded[target]

print(f"\n✅ Feature matrix shape: {X.shape}")
print(f"✅ Target shape: {y.shape}")

# Split data
print("\n✂️  Splitting data into train/test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Test set: {X_test.shape[0]} samples")

# Scale features
print("\n⚖️  Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
print("\n" + "="*70)
print("🤖 TRAINING MODELS")
print("="*70)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for model_name, model in models.items():
    print(f"\n🔄 Training {model_name}...")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"  ✅ Accuracy: {accuracy:.4f}")
    print(f"  ✅ Precision: {precision:.4f}")
    print(f"  ✅ Recall: {recall:.4f}")
    print(f"  ✅ F1 Score: {f1:.4f}")
    
    results[model_name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred
    }
    
    # Feature importance for tree-based models
    if hasattr(model, 'feature_importances_'):
        importances = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        print(f"\n  📊 Top 10 Most Important Features:")
        for idx, row in importances.iterrows():
            print(f"     {row['feature']}: {row['importance']:.4f}")

# Find best model
print("\n" + "="*70)
print("🏆 BEST MODEL")
print("="*70)

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
best_accuracy = results[best_model_name]['accuracy']

print(f"\n✨ Best Model: {best_model_name}")
print(f"✨ Accuracy: {best_accuracy:.4f}")

# Detailed classification report
print(f"\n📋 Detailed Classification Report for {best_model_name}:")
print(classification_report(y_test, results[best_model_name]['predictions']))

# Confusion Matrix
print(f"\n📊 Confusion Matrix for {best_model_name}:")
cm = confusion_matrix(y_test, results[best_model_name]['predictions'])
print(cm)

# Save best model
print("\n💾 Saving best model...")
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

import joblib
model_file = model_dir / f"best_model_{best_model_name.replace(' ', '_').lower()}.joblib"
joblib.dump(best_model, model_file)
print(f"✅ Saved model to: {model_file}")

# Save scaler and encoders
scaler_file = model_dir / "scaler.joblib"
encoders_file = model_dir / "label_encoders.joblib"
joblib.dump(scaler, scaler_file)
joblib.dump(label_encoders, encoders_file)
print(f"✅ Saved scaler to: {scaler_file}")
print(f"✅ Saved encoders to: {encoders_file}")

# Save feature names
feature_names_file = model_dir / "feature_names.txt"
with open(feature_names_file, 'w') as f:
    f.write('\n'.join(feature_columns))
print(f"✅ Saved feature names to: {feature_names_file}")

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nTrained on {len(df)} real Canadian mining records")
print(f"Best model: {best_model_name} with {best_accuracy:.2%} accuracy")
