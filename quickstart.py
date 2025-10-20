#!/usr/bin/env python3
"""
Quick Start Script for Mining ML Project

This script provides a guided walkthrough to get started with the
Canadian Mining Permits ML project.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def main():
    """Main quick start function."""
    
    print_header("Canadian Mining Permits ML - Quick Start")
    
    print("Welcome to the Mining ML project!")
    print("This script will guide you through the initial setup and usage.\n")
    
    print("What would you like to do?\n")
    print("1. Generate sample data")
    print("2. Preprocess data for modeling")
    print("3. Train models")
    print("4. Make predictions")
    print("5. View project structure")
    print("6. Exit")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == "1":
        generate_sample_data()
    elif choice == "2":
        preprocess_data()
    elif choice == "3":
        train_models()
    elif choice == "4":
        make_predictions()
    elif choice == "5":
        show_structure()
    elif choice == "6":
        print("\nGoodbye!")
        return
    else:
        print("\nInvalid choice. Please run the script again.")


def generate_sample_data():
    """Generate sample mining permit data."""
    print_header("Generate Sample Data")
    
    try:
        from data.data_collection import create_sample_data
        from utils.config import load_config, get_data_path, setup_logging
        
        config = load_config()
        setup_logging(config)
        raw_data_path = get_data_path(config, "raw")
        
        n_samples = input("How many sample records? (default: 1000): ").strip()
        n_samples = int(n_samples) if n_samples else 1000
        
        print(f"\nGenerating {n_samples} sample records...")
        df = create_sample_data(raw_data_path, n_samples=n_samples)
        
        print(f"\n✓ Successfully generated {len(df)} sample records!")
        print(f"✓ Data saved to: {raw_data_path / 'sample_permits.csv'}")
        print(f"✓ Approval rate: {df['approved'].mean():.2%}")
        
    except Exception as e:
        print(f"\n✗ Error generating sample data: {e}")


def preprocess_data():
    """Preprocess data for modeling."""
    print_header("Preprocess Data")
    
    try:
        from data.preprocessing import load_and_clean_data, prepare_for_modeling, save_processed_data
        from utils.config import load_config, get_data_path, setup_logging
        
        config = load_config()
        setup_logging(config)
        
        raw_data_path = get_data_path(config, "raw")
        processed_path = get_data_path(config, "processed")
        
        sample_file = raw_data_path / "sample_permits.csv"
        
        if not sample_file.exists():
            print(f"\n✗ Sample data not found at {sample_file}")
            print("Please run option 1 to generate sample data first.")
            return
        
        print("Loading and preprocessing data...")
        df = load_and_clean_data(sample_file)
        
        print("Preparing data for modeling...")
        X_train, X_test, y_train, y_test, prep_info = prepare_for_modeling(df)
        
        print("Saving processed data...")
        save_processed_data(X_train, X_test, y_train, y_test, processed_path)
        
        print(f"\n✓ Data preprocessing complete!")
        print(f"✓ Training set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        print(f"✓ Number of features: {len(X_train.columns)}")
        
    except Exception as e:
        print(f"\n✗ Error preprocessing data: {e}")


def train_models():
    """Train ML models."""
    print_header("Train Models")
    
    try:
        import pandas as pd
        from models.train_model import train_classifier, evaluate_model, save_model
        from utils.config import load_config, get_data_path, get_model_path, setup_logging
        
        config = load_config()
        setup_logging(config)
        
        processed_path = get_data_path(config, "processed")
        model_path = get_model_path(config)
        
        # Load processed data
        X_train = pd.read_csv(processed_path / "X_train.csv")
        X_test = pd.read_csv(processed_path / "X_test.csv")
        y_train = pd.read_csv(processed_path / "y_train.csv")['approved']
        y_test = pd.read_csv(processed_path / "y_test.csv")['approved']
        
        print("Available models:")
        print("1. Logistic Regression")
        print("2. Random Forest")
        print("3. XGBoost (if installed)")
        print("4. LightGBM (if installed)")
        print("5. Train all models")
        
        choice = input("\nSelect model to train (1-5): ").strip()
        
        model_map = {
            '1': 'logistic_regression',
            '2': 'random_forest',
            '3': 'xgboost',
            '4': 'lightgbm'
        }
        
        if choice in model_map:
            model_types = [model_map[choice]]
        elif choice == '5':
            model_types = ['logistic_regression', 'random_forest']
        else:
            print("Invalid choice.")
            return
        
        results = {}
        for model_type in model_types:
            print(f"\n{'='*50}")
            print(f"Training {model_type}...")
            print('='*50)
            
            try:
                model = train_classifier(X_train, y_train, model_type, config)
                metrics = evaluate_model(model, X_test, y_test, model_type)
                save_model(model, model_path, model_type)
                results[model_type] = metrics
                
                print(f"\n✓ {model_type} trained successfully!")
                print(f"  - Accuracy: {metrics['accuracy']:.4f}")
                print(f"  - ROC-AUC: {metrics['roc_auc']:.4f}")
                
            except Exception as e:
                print(f"\n✗ Error training {model_type}: {e}")
        
        if results:
            print("\n" + "="*50)
            print("TRAINING SUMMARY")
            print("="*50)
            for model_type, metrics in results.items():
                print(f"\n{model_type}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
        
    except FileNotFoundError:
        print("\n✗ Processed data not found. Please run option 2 first.")
    except Exception as e:
        print(f"\n✗ Error training models: {e}")


def make_predictions():
    """Make predictions with trained model."""
    print_header("Make Predictions")
    
    print("This is a template. To make predictions:")
    print("\n1. Load a trained model using:")
    print("   from models.train_model import load_model")
    print("   model = load_model(model_path, 'random_forest')")
    print("\n2. Prepare your application data with the same features")
    print("\n3. Use predict_approval_probability() from models.predict")
    print("\nSee src/models/predict.py for examples.")


def show_structure():
    """Show project structure."""
    print_header("Project Structure")
    
    structure = """
Mining-ML/
├── data/
│   ├── raw/              # Raw data files
│   ├── processed/        # Cleaned data
│   └── external/         # External data sources
├── notebooks/
│   └── 01_data_exploration.ipynb  # EDA notebook
├── src/
│   ├── data/             # Data processing modules
│   ├── features/         # Feature engineering
│   ├── models/           # Model training & prediction
│   └── utils/            # Utilities and config
├── models/               # Saved model files
├── reports/              # Analysis reports
├── config.yaml           # Project configuration
├── requirements.txt      # Python dependencies
└── quickstart.py         # This script
    """
    
    print(structure)
    print("\nKey Files:")
    print("- README.md: Project documentation and getting started guide")
    print("- config.yaml: Configuration settings")
    print("- requirements.txt: Python package dependencies")
    print("\nKey Scripts:")
    print("- src/data/data_collection.py: Generate/collect data")
    print("- src/data/preprocessing.py: Clean and prepare data")
    print("- src/models/train_model.py: Train ML models")
    print("- src/models/predict.py: Make predictions")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
