"""
Prediction utilities for mining permit approval.

This module provides functions to make predictions using trained models.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Any, Dict, Union

logger = logging.getLogger(__name__)


def predict_approval_probability(
    model: Any,
    application_data: Union[pd.DataFrame, Dict],
    preprocessing_info: Dict = None
) -> Union[float, np.ndarray]:
    """
    Predict the approval probability for a mining permit application.
    
    Args:
        model: Trained model
        application_data: Application data (DataFrame or dict)
        preprocessing_info: Preprocessing information (encoders, scalers, etc.)
        
    Returns:
        Approval probability (0-1)
    """
    logger.info("Making prediction")
    
    # Convert dict to DataFrame if necessary
    if isinstance(application_data, dict):
        application_data = pd.DataFrame([application_data])
    
    # Apply preprocessing if provided
    if preprocessing_info:
        # TODO: Apply label encoding and scaling using preprocessing_info
        pass
    
    # Make prediction
    probability = model.predict_proba(application_data)[:, 1]
    
    if len(probability) == 1:
        return float(probability[0])
    return probability


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
    # Example usage
    from .train_model import load_model
    from ..utils.config import load_config, get_model_path, setup_logging
    
    config = load_config()
    setup_logging(config)
    
    try:
        # Load a trained model
        model_path = get_model_path(config)
        model = load_model(model_path, "random_forest")
        
        # Example prediction for a single application
        sample_application = {
            'province': 'British Columbia',
            'mining_type': 'Open-pit',
            'mineral_type': 'Gold',
            'company_size': 'Large',
            'project_area': 500.0,
            'estimated_duration': 10,
            'distance_to_water': 2.5,
            'distance_to_protected_area': 15.0,
            'distance_to_indigenous_land': 5.0,
            'expected_employment': 200,
            'environmental_assessment_score': 7.5,
            'public_comments_received': 100,
            'public_opposition_percentage': 25.0,
            'company_compliance_history': 8.5,
            'previous_permits': 5
        }
        
        # Note: This is a simplified example. In practice, you'd need to
        # preprocess the data using the same pipeline as training
        print("\nExample application:")
        for key, value in sample_application.items():
            print(f"  {key}: {value}")
        
        print("\nTo make predictions with a real model:")
        print("1. Load the trained model")
        print("2. Load the preprocessing information")
        print("3. Apply the same preprocessing to new data")
        print("4. Make predictions")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease train a model first using train_model.py")
