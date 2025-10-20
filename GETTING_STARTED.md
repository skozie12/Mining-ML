# Getting Started with Mining ML

This guide will walk you through the complete process of building a machine learning model to predict Canadian mining permit approvals.

## Step 1: Setup Environment

### Install Dependencies

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## Step 2: Generate or Collect Data

### Option A: Use Sample Data (Quick Start)

Generate synthetic sample data for testing:

```bash
python src/data/data_collection.py
```

Or use the interactive quickstart script:

```bash
python quickstart.py
# Select option 1: Generate sample data
```

This will create `data/raw/sample_permits.csv` with 1000 synthetic mining permit applications.

### Option B: Use Real Data (Production)

1. Collect real mining permit data from sources listed in README.md
2. Save the data as CSV in `data/raw/` directory
3. Ensure your data has the required columns (see `data/raw/README.md`)

## Step 3: Explore the Data

Open the Jupyter notebook for exploratory data analysis:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

This notebook will help you:
- Understand data distributions
- Identify patterns and correlations
- Visualize approval rates by province, mining type, etc.
- Discover which features are most important

## Step 4: Preprocess the Data

Clean and prepare the data for machine learning:

```bash
python src/data/preprocessing.py
```

Or use the quickstart script:

```bash
python quickstart.py
# Select option 2: Preprocess data for modeling
```

This will:
- Clean the data (handle missing values, remove duplicates)
- Engineer new features
- Split data into train/test sets
- Scale numerical features
- Encode categorical variables
- Save processed data to `data/processed/`

## Step 5: Train Models

Train machine learning models:

```bash
python src/models/train_model.py
```

Or use the quickstart script:

```bash
python quickstart.py
# Select option 3: Train models
```

This will:
- Train multiple model types (Logistic Regression, Random Forest, etc.)
- Evaluate each model on the test set
- Display performance metrics
- Save trained models to `models/` directory

### Available Models:

1. **Logistic Regression**: Simple, interpretable baseline
2. **Random Forest**: Handles non-linear relationships, provides feature importance
3. **XGBoost**: High-performance gradient boosting (requires `pip install xgboost`)
4. **LightGBM**: Fast gradient boosting (requires `pip install lightgbm`)

## Step 6: Make Predictions

Use a trained model to predict permit approval probability:

```python
from pathlib import Path
import pandas as pd
from src.models.train_model import load_model
from src.models.predict import predict_approval_probability
from src.utils.config import load_config, get_model_path

# Load configuration
config = load_config()
model_path = get_model_path(config)

# Load trained model
model = load_model(model_path, "random_forest")

# Prepare application data
new_application = pd.DataFrame([{
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
}])

# Make prediction
# Note: In production, you need to preprocess new_application 
# using the same pipeline as training data
probability = predict_approval_probability(model, new_application)
print(f"Approval probability: {probability:.2%}")
```

## Step 7: Interpret Results

### Feature Importance

Understand which features drive predictions:

```python
from src.models.predict import explain_prediction

# Get feature importance
importance_df = explain_prediction(model, X_test, feature_names)
print(importance_df.head(10))
```

### SHAP Values (Advanced)

For detailed feature contribution analysis:

```python
import shap

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualize
shap.summary_plot(shap_values, X_test)
```

## Project Workflow Summary

```
1. Data Collection → 2. Data Exploration → 3. Data Preprocessing
                ↓                                    ↓
        4. Model Training ← ← ← ← ← ← ← ← ← ← ← ← ← 
                ↓
        5. Model Evaluation
                ↓
        6. Make Predictions
                ↓
        7. Interpret & Deploy
```

## Configuration

Edit `config.yaml` to customize:
- Model hyperparameters
- Data paths
- Feature engineering options
- Evaluation metrics
- And more

## Tips for Success

1. **Start Simple**: Begin with logistic regression before trying complex models
2. **Validate Assumptions**: Ensure your sample data reflects real patterns
3. **Feature Engineering**: Good features are more important than complex models
4. **Cross-Validation**: Use CV to get robust performance estimates
5. **Interpretability**: Always understand why your model makes predictions
6. **Iterate**: ML is iterative - continuously improve your pipeline

## Common Issues

### Import Errors

If you get import errors, make sure you're running scripts from the project root:

```bash
cd /path/to/Mining-ML
python src/data/data_collection.py
```

### Missing Dependencies

Install optional dependencies as needed:

```bash
pip install xgboost
pip install lightgbm
pip install shap
```

### Data Not Found

Make sure to generate sample data or add your own data before preprocessing:

```bash
python quickstart.py  # Select option 1
```

## Next Steps

After completing this guide:

1. **Collect Real Data**: Replace sample data with actual mining permit records
2. **Advanced Features**: Engineer domain-specific features
3. **Hyperparameter Tuning**: Optimize model parameters using GridSearchCV
4. **Ensemble Methods**: Combine multiple models for better predictions
5. **Deploy Model**: Create a web API or dashboard for predictions
6. **Monitor Performance**: Track model performance over time

## Resources

- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)

## Getting Help

- Review the code comments and docstrings
- Check the main README.md for data sources and methodology
- Open an issue on GitHub for bugs or questions

Good luck with your mining permit prediction project! 🚀
