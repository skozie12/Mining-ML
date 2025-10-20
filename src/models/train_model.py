"""
Model training utilities for mining permit approval prediction.

This module provides functions to train various machine learning models
and evaluate their performance.
"""

import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import cross_val_score, GridSearchCV

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")


# ============================================================================
# REGRESSION MODELS FOR TIME PREDICTION
# ============================================================================

def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> LinearRegression:
    """
    Train a Linear Regression model for time prediction.
    
    Args:
        X_train: Training features
        y_train: Training target (approval time)
        random_state: Random seed
        
    Returns:
        Trained LinearRegression model
    """
    logger.info("Training Linear Regression model")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    logger.info("Linear Regression training complete")
    return model


def train_random_forest_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = 10,
    random_state: int = 42
) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor for time prediction.
    
    Args:
        X_train: Training features
        y_train: Training target (approval time)
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        random_state: Random seed
        
    Returns:
        Trained RandomForestRegressor model
    """
    logger.info("Training Random Forest Regressor model")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    logger.info("Random Forest Regressor training complete")
    return model


def train_xgboost_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Any:
    """
    Train an XGBoost Regressor for time prediction.
    
    Args:
        X_train: Training features
        y_train: Training target (approval time)
        random_state: Random seed
        
    Returns:
        Trained XGBRegressor model
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not available. Install with: pip install xgboost")
    
    logger.info("Training XGBoost Regressor model")
    
    model = XGBRegressor(
        random_state=random_state,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    model.fit(X_train, y_train)
    
    logger.info("XGBoost Regressor training complete")
    return model


def train_lightgbm_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Any:
    """
    Train a LightGBM Regressor for time prediction.
    
    Args:
        X_train: Training features
        y_train: Training target (approval time)
        random_state: Random seed
        
    Returns:
        Trained LGBMRegressor model
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is not available. Install with: pip install lightgbm")
    
    logger.info("Training LightGBM Regressor model")
    
    model = LGBMRegressor(
        random_state=random_state,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    logger.info("LightGBM Regressor training complete")
    return model


# ============================================================================
# CLASSIFICATION MODELS FOR CONFIDENCE PREDICTION
# ============================================================================

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> LogisticRegression:
    """
    Train a Logistic Regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed
        
    Returns:
        Trained LogisticRegression model
    """
    logger.info("Training Logistic Regression model")
    
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    logger.info("Logistic Regression training complete")
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = 10,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Train a Random Forest model.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        random_state: Random seed
        
    Returns:
        Trained RandomForestClassifier model
    """
    logger.info("Training Random Forest model")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    logger.info("Random Forest training complete")
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42
) -> Any:
    """
    Train an XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        random_state: Random seed
        
    Returns:
        Trained XGBoost model
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
    
    logger.info("Training XGBoost model")
    
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    logger.info("XGBoost training complete")
    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = -1,
    learning_rate: float = 0.1,
    random_state: int = 42
) -> Any:
    """
    Train a LightGBM model.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth (-1 for no limit)
        learning_rate: Learning rate
        random_state: Random seed
        
    Returns:
        Trained LightGBM model
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")
    
    logger.info("Training LightGBM model")
    
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        class_weight='balanced',
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    logger.info("LightGBM training complete")
    return model


def evaluate_regression_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Regression Model"
) -> Dict[str, float]:
    """
    Evaluate a trained regression model on test data.
    
    Args:
        model: Trained regression model
        X_test: Test features
        y_test: Test target (approval time)
        model_name: Name of the model for logging
        
    Returns:
        Dictionary of regression evaluation metrics
    """
    logger.info(f"Evaluating {model_name}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }
    
    # Log metrics
    logger.info(f"\n{model_name} Performance:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
    
    return metrics


def evaluate_classification_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Classification Model"
) -> Dict[str, float]:
    """
    Evaluate a trained classification model on test data.
    
    Args:
        model: Trained classification model
        X_test: Test features
        y_test: Test target (confidence level)
        model_name: Name of the model for logging
        
    Returns:
        Dictionary of classification evaluation metrics
    """
    logger.info(f"Evaluating {model_name}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0)
    }
    
    # Log metrics
    logger.info(f"\n{model_name} Performance:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
    
    # Print classification report
    logger.info(f"\nClassification Report for {model_name}:")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    return metrics


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model for logging
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating {model_name}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Log metrics
    logger.info(f"\n{model_name} Performance:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
    
    # Print classification report
    logger.info(f"\nClassification Report for {model_name}:")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    return metrics


def train_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'random_forest_regressor',
    config: Optional[Dict] = None
) -> Any:
    """
    Train a regressor based on specified model type.
    
    Args:
        X_train: Training features
        y_train: Training target (approval time)
        model_type: Type of model to train
        config: Configuration dictionary
        
    Returns:
        Trained regression model
    """
    random_state = config.get('model', {}).get('random_state', 42) if config else 42
    
    if model_type == 'linear_regression':
        return train_linear_regression(X_train, y_train, random_state)
    
    elif model_type == 'random_forest_regressor':
        params = config.get('model', {}).get('random_forest', {}) if config else {}
        return train_random_forest_regressor(
            X_train, y_train,
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10),
            random_state=random_state
        )
    
    elif model_type == 'xgboost_regressor':
        return train_xgboost_regressor(X_train, y_train, random_state)
    
    elif model_type == 'lightgbm_regressor':
        return train_lightgbm_regressor(X_train, y_train, random_state)
    
    else:
        raise ValueError(f"Unknown regressor type: {model_type}")


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'random_forest',
    config: Optional[Dict] = None
) -> Any:
    """
    Train a classifier based on specified model type.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model to train
        config: Configuration dictionary
        
    Returns:
        Trained model
    """
    random_state = config.get('model', {}).get('random_state', 42) if config else 42
    
    if model_type == 'logistic_regression':
        return train_logistic_regression(X_train, y_train, random_state)
    
    elif model_type == 'random_forest':
        params = config.get('model', {}).get('random_forest', {}) if config else {}
        return train_random_forest(
            X_train, y_train,
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10),
            random_state=random_state
        )
    
    elif model_type == 'xgboost':
        params = config.get('model', {}).get('xgboost', {}) if config else {}
        return train_xgboost(
            X_train, y_train,
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            random_state=random_state
        )
    
    elif model_type == 'lightgbm':
        params = config.get('model', {}).get('lightgbm', {}) if config else {}
        return train_lightgbm(
            X_train, y_train,
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', -1),
            learning_rate=params.get('learning_rate', 0.1),
            random_state=random_state
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def save_model(model: Any, model_path: Path, model_name: str) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model
        model_path: Directory to save the model
        model_name: Name for the saved model file
    """
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / f"{model_name}.joblib"
    
    joblib.dump(model, model_file)
    logger.info(f"Model saved to {model_file}")


def load_model(model_path: Path, model_name: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Directory containing the model
        model_name: Name of the model file (without extension)
        
    Returns:
        Loaded model
    """
    model_file = model_path / f"{model_name}.joblib"
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    model = joblib.load(model_file)
    logger.info(f"Model loaded from {model_file}")
    
    return model


if __name__ == "__main__":
    # Example usage
    from ..utils.config import load_config, get_data_path, get_model_path, setup_logging
    
    config = load_config()
    setup_logging(config)
    
    # Load processed data
    processed_path = get_data_path(config, "processed")
    
    try:
        X_train = pd.read_csv(processed_path / "X_train.csv")
        X_test = pd.read_csv(processed_path / "X_test.csv")
        y_time_train = pd.read_csv(processed_path / "y_time_train.csv")['approval_time_months']
        y_time_test = pd.read_csv(processed_path / "y_time_test.csv")['approval_time_months']
        y_conf_train = pd.read_csv(processed_path / "y_conf_train.csv")['approval_confidence']
        y_conf_test = pd.read_csv(processed_path / "y_conf_test.csv")['approval_confidence']
        
        print("Training multi-target models...\n")
        
        # Train and evaluate time prediction models
        time_models = {}
        time_results = {}
        model_path = Path(config['model']['model_path'])
        
        print("="*70)
        print("TRAINING TIME PREDICTION MODELS")
        print("="*70)
        
        for model_type in config['model']['time_prediction_algorithms']:
            print(f"\nTraining {model_type} for time prediction...")
            
            model = train_regressor(X_train, y_time_train, model_type, config)
            metrics = evaluate_regression_model(model, X_test, y_time_test, model_type)
            
            time_models[model_type] = model
            time_results[model_type] = metrics
            
            # Save model
            save_model(model, model_path, f"time_{model_type}")
        
        # Find best time model
        best_time_model = min(time_results, key=lambda x: time_results[x]['rmse'])
        print(f"\nBest time prediction model: {best_time_model}")
        print(f"RMSE: {time_results[best_time_model]['rmse']:.2f} months")
        
        print("\n" + "="*70)
        print("TRAINING CONFIDENCE PREDICTION MODELS")
        print("="*70)
        
        # Train and evaluate confidence prediction models
        conf_models = {}
        conf_results = {}
        
        for model_type in config['model']['confidence_prediction_algorithms']:
            print(f"\nTraining {model_type} for confidence prediction...")
            
            model = train_classifier(X_train, y_conf_train, model_type, config)
            metrics = evaluate_classification_model(model, X_test, y_conf_test, model_type)
            
            conf_models[model_type] = model
            conf_results[model_type] = metrics
            
            # Save model
            save_model(model, model_path, f"confidence_{model_type}")
        
        # Find best confidence model
        best_conf_model = max(conf_results, key=lambda x: conf_results[x]['accuracy'])
        print(f"\nBest confidence prediction model: {best_conf_model}")
        print(f"Accuracy: {conf_results[best_conf_model]['accuracy']:.4f}")
        
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        print(f"Best Time Model: {best_time_model} (RMSE: {time_results[best_time_model]['rmse']:.2f} months)")
        print(f"Best Confidence Model: {best_conf_model} (Accuracy: {conf_results[best_conf_model]['accuracy']:.4f})")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run preprocessing.py first to prepare the data.")
