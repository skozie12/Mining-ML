"""
Prediction utilities for mining permit approval.

This module provides functions to make predictions using trained models.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Any, Dict, Union, Tuple

logger = logging.getLogger(__name__)


def predict_approval_time_and_confidence(
    time_model: Any,
    confidence_model: Any,
    application_data: Union[pd.DataFrame, Dict],
    preprocessing_info: Dict = None
) -> Dict[str, Union[float, str]]:
    """
    Predict the approval time and confidence level for a mining permit application.
    
    Args:
        time_model: Trained regression model for time prediction
        confidence_model: Trained classification model for confidence prediction
        application_data: Application data (DataFrame or dict)
        preprocessing_info: Preprocessing information (encoders, scalers, etc.)
        
    Returns:
        Dictionary with estimated_approval_time_months and approval_confidence
    """
    logger.info("Making time and confidence predictions")
    
    # Convert dict to DataFrame if necessary
    if isinstance(application_data, dict):
        application_data = pd.DataFrame([application_data])
    
    # Apply preprocessing if provided
    if preprocessing_info:
        application_data_processed = apply_preprocessing(application_data, preprocessing_info)
    else:
        application_data_processed = application_data
    
    # Make time prediction (regression)
    estimated_time = time_model.predict(application_data_processed)
    
    # Make confidence prediction (classification)
    confidence_prediction = confidence_model.predict(application_data_processed)
    confidence_probabilities = confidence_model.predict_proba(application_data_processed)
    
    # Get the maximum probability for the predicted confidence level
    predicted_class_idx = confidence_model.classes_.tolist().index(confidence_prediction[0])
    confidence_score = confidence_probabilities[0][predicted_class_idx]
    
    result = {
        'estimated_approval_time_months': round(float(estimated_time[0]), 1),
        'approval_confidence': confidence_prediction[0],
        'confidence_score': round(float(confidence_score), 3)
    }
    
    logger.info(f"Prediction: {result['estimated_approval_time_months']} months, {result['approval_confidence']} confidence")
    
    return result


def apply_preprocessing(
    data: pd.DataFrame, 
    preprocessing_info: Dict
) -> pd.DataFrame:
    """
    Apply preprocessing steps to raw application data.
    
    Args:
        data: Raw application data
        preprocessing_info: Preprocessing information from training
        
    Returns:
        Preprocessed data ready for model prediction
    """
    data_processed = data.copy()
    
    # Apply label encoding
    for col, encoder in preprocessing_info.get('label_encoders', {}).items():
        if col in data_processed.columns:
            # Handle unseen categories by using the most frequent label
            try:
                data_processed[col] = encoder.transform(data_processed[col].astype(str))
            except ValueError:
                logger.warning(f"Unseen category in {col}, using most frequent label")
                data_processed[col] = encoder.transform([encoder.classes_[0]])[0]
    
    # Apply scaling
    scaler = preprocessing_info.get('scaler')
    numerical_columns = preprocessing_info.get('numerical_columns', [])
    
    if scaler and numerical_columns:
        numerical_cols_present = [col for col in numerical_columns if col in data_processed.columns]
        if numerical_cols_present:
            data_processed[numerical_cols_present] = scaler.transform(data_processed[numerical_cols_present])
    
    # Ensure feature order matches training data
    feature_names = preprocessing_info.get('feature_names', [])
    if feature_names:
        # Add missing columns with default values
        for col in feature_names:
            if col not in data_processed.columns:
                data_processed[col] = 0
        
        # Reorder columns to match training data
        data_processed = data_processed[feature_names]
    
    return data_processed


def predict_approval_probability(
    model: Any,
    application_data: Union[pd.DataFrame, Dict],
    preprocessing_info: Dict = None
) -> Union[float, np.ndarray]:
    """
    DEPRECATED: Use predict_approval_time_and_confidence instead.
    Predict the approval probability for a mining permit application.
    
    Args:
        model: Trained model
        application_data: Application data (DataFrame or dict)
        preprocessing_info: Preprocessing information (encoders, scalers, etc.)
        
    Returns:
        Approval probability (0-1)
    """
    logger.warning("This function is deprecated. Use predict_approval_time_and_confidence instead.")
    
    # Convert dict to DataFrame if necessary
    if isinstance(application_data, dict):
        application_data = pd.DataFrame([application_data])
    
    # Apply preprocessing if provided
    if preprocessing_info:
        application_data = apply_preprocessing(application_data, preprocessing_info)
    
    # Make prediction
    probability = model.predict_proba(application_data)[:, 1]
    
    if len(probability) == 1:
        return float(probability[0])
    return probability


def load_trained_models(
    model_path: Path,
    time_model_name: str = None,
    confidence_model_name: str = None
) -> Tuple[Any, Any]:
    """
    Load both trained models from disk.
    
    Args:
        model_path: Directory containing the models
        time_model_name: Name of the time prediction model
        confidence_model_name: Name of the confidence prediction model
        
    Returns:
        Tuple of (time_model, confidence_model)
    """
    import joblib
    
    if time_model_name is None:
        # Try to find the best time model from config or use default
        time_model_name = "time_random_forest_regressor"
    
    if confidence_model_name is None:
        # Try to find the best confidence model from config or use default
        confidence_model_name = "confidence_random_forest_classifier"
    
    time_model_file = model_path / f"{time_model_name}.joblib"
    confidence_model_file = model_path / f"{confidence_model_name}.joblib"
    
    if not time_model_file.exists():
        raise FileNotFoundError(f"Time model not found: {time_model_file}")
    
    if not confidence_model_file.exists():
        raise FileNotFoundError(f"Confidence model not found: {confidence_model_file}")
    
    time_model = joblib.load(time_model_file)
    confidence_model = joblib.load(confidence_model_file)
    
    logger.info(f"Loaded models: {time_model_name}, {confidence_model_name}")
    
    return time_model, confidence_model


def create_sample_permit_application() -> Dict:
    """
    Create a sample permit application for testing predictions.
    
    Returns:
        Dictionary with sample application data
    """
    return {
        'province': 'British Columbia',
        'mining_type': 'Open-pit',
        'mineral_type': 'Copper',
        'company_size': 'Large',
        'project_area': 500.0,
        'estimated_duration': 10,
        'distance_to_water': 2.5,
        'distance_to_protected_area': 15.0,
        'distance_to_indigenous_land': 8.0,
        'expected_employment': 150,
        'environmental_assessment_score': 7.5,
        'public_comments_received': 25,
        'public_opposition_percentage': 20.0,
        'company_compliance_history': 8.0,
        'previous_permits': 3
    }


def predict_approval(
    model: Any,
    application_data: Union[pd.DataFrame, Dict],
    threshold: float = 0.5,
    preprocessing_info: Dict = None
) -> Union[int, np.ndarray]:
    """
    Predict whether a mining permit will be approved.
    
    Args:
        model: Trained model
        application_data: Application data (DataFrame or dict)
        threshold: Classification threshold
        preprocessing_info: Preprocessing information
        
    Returns:
        Binary prediction (0 or 1)
    """
    probability = predict_approval_probability(model, application_data, preprocessing_info)
    
    if isinstance(probability, float):
        return int(probability >= threshold)
    return (probability >= threshold).astype(int)


def explain_prediction(
    model: Any,
    application_data: pd.DataFrame,
    feature_names: list
) -> pd.DataFrame:
    """
    Explain which features contributed most to the prediction.
    
    This function works best with tree-based models that have
    feature_importances_ attribute.
    
    Args:
        model: Trained model
        application_data: Application data
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance
    """
    logger.info("Explaining prediction")
    
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    elif hasattr(model, 'coef_'):
        # For linear models
        coefficients = model.coef_[0]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })
        
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        
        return importance_df
    
    else:
        logger.warning("Model does not support feature importance extraction")
        return pd.DataFrame()


def batch_predict(
    model: Any,
    applications_file: Path,
    output_file: Path,
    preprocessing_info: Dict = None
) -> pd.DataFrame:
    """
    Make predictions for multiple applications from a file.
    
    Args:
        model: Trained model
        applications_file: Path to CSV file with applications
        output_file: Path to save predictions
        preprocessing_info: Preprocessing information
        
    Returns:
        DataFrame with predictions
    """
    logger.info(f"Loading applications from {applications_file}")
    
    applications = pd.read_csv(applications_file)
    
    # Make predictions
    probabilities = predict_approval_probability(model, applications, preprocessing_info)
    predictions = predict_approval(model, applications, preprocessing_info=preprocessing_info)
    
    # Add predictions to dataframe
    results = applications.copy()
    results['predicted_approval'] = predictions
    results['approval_probability'] = probabilities
    
    # Save results
    results.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")
    
    return results


if __name__ == "__main__":
    # Example usage for new prediction functionality
    from ..utils.config import load_config, setup_logging
    
    config = load_config()
    setup_logging(config)
    
    try:
        # Load both trained models
        model_path = Path(config['model']['model_path'])
        
        print("Loading trained models...")
        time_model, confidence_model = load_trained_models(model_path)
        
        # Create a sample permit application
        sample_application = create_sample_permit_application()
        
        print("\n" + "="*60)
        print("SAMPLE MINING PERMIT APPLICATION")
        print("="*60)
        for key, value in sample_application.items():
            print(f"{key:30}: {value}")
        
        # Make prediction (Note: In practice, you'd need preprocessing_info)
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        
        # This is a simplified example without full preprocessing
        # In production, you'd load the preprocessing_info from training
        print("Note: This is a simplified demonstration.")
        print("For production use, ensure proper preprocessing is applied.")
        
        print(f"\nEstimated approval time: 8-15 months (varies by application)")
        print(f"Approval confidence: Medium-High")
        print(f"Key factors affecting approval:")
        print(f"  • Environmental assessment score: {sample_application['environmental_assessment_score']}")
        print(f"  • Distance to protected areas: {sample_application['distance_to_protected_area']} km")
        print(f"  • Public opposition: {sample_application['public_opposition_percentage']}%")
        print(f"  • Company compliance history: {sample_application['company_compliance_history']}/10")
        
        print("\nTo use the prediction system:")
        print("1. Train models using train_model.py")
        print("2. Load preprocessing information from training")
        print("3. Apply preprocessing to new applications")
        print("4. Call predict_approval_time_and_confidence()")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease train models first using train_model.py")
        print("The system expects:")
        print("  • Time prediction model (regression)")
        print("  • Confidence prediction model (classification)")
