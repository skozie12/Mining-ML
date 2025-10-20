"""
Data preprocessing utilities for mining permit data.

This module handles data cleaning, feature engineering, and preparation
for machine learning models.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate mining permit data.
    
    Args:
        df (pd.DataFrame): Raw permit data
        
    Returns:
        pd.DataFrame: Cleaned data
    """
    logger.info("Starting data cleaning")
    initial_shape = df.shape
    
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates(subset=['permit_id'], keep='first')
    logger.info(f"Removed {initial_shape[0] - len(df_clean)} duplicate records")
    
    # Handle missing values
    missing_before = df_clean.isnull().sum().sum()
    
    # For numerical columns, fill with median
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_clean[col].isnull().any():
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # For categorical columns, fill with mode
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    missing_after = df_clean.isnull().sum().sum()
    logger.info(f"Handled {missing_before - missing_after} missing values")
    
    # Convert date columns
    date_columns = [col for col in df_clean.columns if 'date' in col.lower()]
    for col in date_columns:
        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    # Remove outliers (optional - can be commented out)
    # Use IQR method for numerical columns
    for col in ['project_area', 'expected_employment']:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            if outliers > 0:
                df_clean = df_clean[
                    (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                ]
                logger.info(f"Removed {outliers} outliers from {col}")
    
    logger.info(f"Data cleaning complete. Final shape: {df_clean.shape}")
    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features from existing data.
    
    Args:
        df (pd.DataFrame): Cleaned permit data
        
    Returns:
        pd.DataFrame: Data with engineered features
    """
    logger.info("Starting feature engineering")
    df_features = df.copy()
    
    # Time-based features
    if 'application_date' in df_features.columns:
        df_features['application_year'] = pd.to_datetime(df_features['application_date']).dt.year
        df_features['application_month'] = pd.to_datetime(df_features['application_date']).dt.month
        df_features['application_quarter'] = pd.to_datetime(df_features['application_date']).dt.quarter
    
    if 'decision_date' in df_features.columns and 'application_date' in df_features.columns:
        df_features['processing_days'] = (
            pd.to_datetime(df_features['decision_date']) - 
            pd.to_datetime(df_features['application_date'])
        ).dt.days
    
    # Environmental risk score
    if all(col in df_features.columns for col in ['distance_to_water', 'distance_to_protected_area']):
        df_features['environmental_risk'] = (
            (1 / (df_features['distance_to_water'] + 1)) * 0.5 +
            (1 / (df_features['distance_to_protected_area'] + 1)) * 0.5
        )
    
    # Project scale indicator
    if all(col in df_features.columns for col in ['project_area', 'estimated_duration', 'expected_employment']):
        df_features['project_scale'] = (
            df_features['project_area'] * 
            df_features['estimated_duration'] * 
            df_features['expected_employment']
        ) ** (1/3)  # Geometric mean
    
    # Company experience indicator
    if 'previous_permits' in df_features.columns:
        df_features['is_experienced_company'] = (df_features['previous_permits'] >= 3).astype(int)
    
    # Public sentiment
    if all(col in df_features.columns for col in ['public_comments_received', 'public_opposition_percentage']):
        df_features['public_engagement'] = df_features['public_comments_received']
        df_features['public_support_ratio'] = (
            100 - df_features['public_opposition_percentage']
        ) / 100
    
    # Indigenous land proximity flag
    if 'distance_to_indigenous_land' in df_features.columns:
        df_features['near_indigenous_land'] = (df_features['distance_to_indigenous_land'] < 10).astype(int)
    
    logger.info(f"Feature engineering complete. New shape: {df_features.shape}")
    return df_features


def prepare_for_modeling(
    df: pd.DataFrame,
    time_target: str = 'approval_time_months',
    confidence_target: str = 'approval_confidence',
    categorical_columns: Optional[List[str]] = None,
    numerical_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, dict]:
    """
    Prepare data for multi-target machine learning modeling.
    
    Args:
        df (pd.DataFrame): Feature-engineered data
        time_target (str): Name of the time prediction target variable
        confidence_target (str): Name of the confidence prediction target variable
        categorical_columns (List[str]): List of categorical column names
        numerical_columns (List[str]): List of numerical column names
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple containing X_train, X_test, y_time_train, y_time_test, y_conf_train, y_conf_test, and preprocessing info
    """
    logger.info("Preparing data for multi-target modeling")
    
    # Separate features and targets
    if time_target not in df.columns:
        raise ValueError(f"Time target column '{time_target}' not found in data")
    if confidence_target not in df.columns:
        raise ValueError(f"Confidence target column '{confidence_target}' not found in data")
    
    y_time = df[time_target]
    y_confidence = df[confidence_target]
    
    # Drop columns not needed for modeling
    columns_to_drop = [time_target, confidence_target, 'permit_id', 'application_date', 'decision_date', 'approval_probability']
    X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Identify categorical and numerical columns if not provided
    if categorical_columns is None:
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    
    if numerical_columns is None:
        numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    
    logger.info(f"Categorical features: {len(categorical_columns)}")
    logger.info(f"Numerical features: {len(numerical_columns)}")
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_columns:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Split data (stratify by confidence level)
    X_train, X_test, y_time_train, y_time_test, y_conf_train, y_conf_test = train_test_split(
        X, y_time, y_confidence, test_size=test_size, random_state=random_state, stratify=y_confidence
    )
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    logger.info(f"Training set avg time: {y_time_train.mean():.1f} months")
    logger.info(f"Test set avg time: {y_time_test.mean():.1f} months")
    logger.info(f"Training confidence distribution: {y_conf_train.value_counts().to_dict()}")
    logger.info(f"Test confidence distribution: {y_conf_test.value_counts().to_dict()}")
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])
    
    # Store preprocessing information
    preprocessing_info = {
        'label_encoders': label_encoders,
        'scaler': scaler,
        'categorical_columns': categorical_columns,
        'numerical_columns': numerical_columns,
        'feature_names': X.columns.tolist()
    }
    
    logger.info("Data preparation complete")
    
    return X_train, X_test, y_time_train, y_time_test, y_conf_train, y_conf_test, preprocessing_info


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_time_train: pd.Series,
    y_time_test: pd.Series,
    y_conf_train: pd.Series,
    y_conf_test: pd.Series,
    output_path: Path
) -> None:
    """
    Save processed data to files.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_time_train: Training time target
        y_time_test: Test time target
        y_conf_train: Training confidence target
        y_conf_test: Test confidence target
        output_path: Directory to save processed data
    """
    logger.info(f"Saving processed data to {output_path}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV files
    X_train.to_csv(output_path / "X_train.csv", index=False)
    X_test.to_csv(output_path / "X_test.csv", index=False)
    y_time_train.to_csv(output_path / "y_time_train.csv", index=False, header=['approval_time_months'])
    y_time_test.to_csv(output_path / "y_time_test.csv", index=False, header=['approval_time_months'])
    y_conf_train.to_csv(output_path / "y_conf_train.csv", index=False, header=['approval_confidence'])
    y_conf_test.to_csv(output_path / "y_conf_test.csv", index=False, header=['approval_confidence'])
    
    logger.info("Processed data saved successfully")


def load_and_clean_data(file_path: Path) -> pd.DataFrame:
    """
    Convenience function to load and clean data in one step.
    
    Args:
        file_path (Path): Path to raw data file
        
    Returns:
        pd.DataFrame: Cleaned and feature-engineered data
    """
    from .data_collection import load_permit_data
    
    # Load data
    df = load_permit_data(file_path)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Engineer features
    df_features = engineer_features(df_clean)
    
    return df_features


if __name__ == "__main__":
    # Example usage
    from ..utils.config import load_config, get_data_path, setup_logging
    
    config = load_config()
    setup_logging(config)
    
    # Load sample data
    raw_data_path = get_data_path(config, "raw")
    sample_file = raw_data_path / "sample_permits.csv"
    
    if sample_file.exists():
        # Load and process data
        df = load_and_clean_data(sample_file)
        
        # Prepare for modeling
        X_train, X_test, y_time_train, y_time_test, y_conf_train, y_conf_test, prep_info = prepare_for_modeling(df)
        
        # Save processed data
        processed_path = get_data_path(config, "processed")
        save_processed_data(X_train, X_test, y_time_train, y_time_test, y_conf_train, y_conf_test, processed_path)
        
        print("\nData preprocessing complete!")
        print(f"Training features shape: {X_train.shape}")
        print(f"Test features shape: {X_test.shape}")
    else:
        print(f"Sample data not found at {sample_file}")
        print("Run data_collection.py first to generate sample data")
