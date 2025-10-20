# Mining-ML: Canadian Mining Permits Approval Prediction

A machine learning project to predict the likelihood of Canadian mining permits getting approved based on historical data and various features.

## 🎯 Project Overview

This project aims to build a predictive model that can estimate the probability of mining permit approval in Canada. The model analyzes historical permit applications, environmental factors, regulatory requirements, and other relevant features to provide insights into approval likelihood.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment tool (recommended)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/skozie12/Mining-ML.git
cd Mining-ML
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
Mining-ML/
├── data/
│   ├── raw/                    # Raw data files
│   ├── processed/              # Cleaned and preprocessed data
│   └── external/               # External data sources
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_collection.py  # Scripts to collect data
│   │   └── preprocessing.py    # Data cleaning and preprocessing
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py   # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py      # Model training scripts
│   │   └── predict.py          # Prediction scripts
│   └── utils/
│       ├── __init__.py
│       └── config.py           # Configuration and constants
├── models/                     # Saved model files
├── reports/                    # Analysis reports and figures
├── requirements.txt
├── config.yaml
└── README.md
```

## 📊 Data Sources

To build an effective model, you'll need data from various sources:

### Primary Data Sources:
1. **Natural Resources Canada (NRCan)**
   - Mining regulations and permits database
   - URL: https://www.nrcan.gc.ca/

2. **Provincial Mining Regulatory Bodies**
   - British Columbia: Ministry of Energy, Mines and Low Carbon Innovation
   - Ontario: Ministry of Mines
   - Quebec: Ministère de l'Énergie et des Ressources naturelles
   - Saskatchewan: Ministry of Energy and Resources
   - Other provincial authorities

3. **Canadian Environmental Assessment Agency**
   - Environmental assessment reports
   - Impact assessment decisions

### Key Data Fields to Collect:
- **Application Details**: Date, type of permit, company name, project location
- **Environmental Factors**: Proximity to protected areas, water bodies, Indigenous lands
- **Project Characteristics**: Mining type (surface/underground), mineral type, project scale
- **Company Information**: History of compliance, previous permits, financial status
- **Public Consultation**: Number of public comments, objections, support
- **Regulatory Context**: Province, municipality, Indigenous territory
- **Outcome**: Approved/Rejected, conditions attached, timeline

## 🔬 Methodology

### 1. Data Collection and Preparation
- Gather historical mining permit applications from various Canadian jurisdictions
- Clean and standardize data formats across different sources
- Handle missing values and outliers
- Create a unified dataset

### 2. Exploratory Data Analysis (EDA)
- Analyze approval rates across provinces, years, and mining types
- Identify key factors correlating with approval/rejection
- Visualize trends and patterns
- Understand class imbalance (if any)

### 3. Feature Engineering
- Create meaningful features from raw data
- Encode categorical variables
- Generate interaction features
- Scale numerical features

### 4. Model Development
Start with baseline models and iterate:
- **Logistic Regression**: Simple, interpretable baseline
- **Random Forest**: Handle non-linear relationships
- **Gradient Boosting** (XGBoost/LightGBM): High performance
- **Neural Networks**: For complex patterns (if sufficient data)

### 5. Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curve
- Cross-validation
- Feature importance analysis
- Confusion matrix

### 6. Model Interpretation
- SHAP values for feature importance
- Partial dependence plots
- Individual prediction explanations

## 📈 Quick Start Example

```python
# Load and preprocess data
from src.data.preprocessing import load_and_clean_data
data = load_and_clean_data('data/raw/permits.csv')

# Train a model
from src.models.train_model import train_classifier
model = train_classifier(data, model_type='random_forest')

# Make predictions
from src.models.predict import predict_approval_probability
probability = predict_approval_probability(model, new_application)
print(f"Approval probability: {probability:.2%}")
```

## 🛠️ Key Features to Consider

1. **Geographic Features**
   - Province/Territory
   - Distance to urban areas
   - Proximity to protected lands
   - Indigenous territory involvement

2. **Environmental Features**
   - Environmental sensitivity of location
   - Water body proximity
   - Wildlife habitat presence
   - Previous environmental incidents

3. **Project Features**
   - Mining method (open-pit, underground)
   - Mineral/resource type
   - Project size and duration
   - Expected employment

4. **Company Features**
   - Company size and financial health
   - Previous permit history
   - Compliance record
   - Years in operation

5. **Regulatory Features**
   - Application completeness
   - Time period (year, quarter)
   - Current regulatory climate
   - Recent policy changes

6. **Stakeholder Features**
   - Public consultation results
   - Indigenous consultation status
   - Community support/opposition
   - NGO involvement

## 📝 Next Steps

1. **Data Collection**: Start gathering historical permit data from provincial databases
2. **Data Cleaning**: Standardize and clean the collected data
3. **Baseline Model**: Build a simple logistic regression model
4. **Iterate**: Gradually improve with better features and algorithms
5. **Validate**: Test predictions against recent decisions
6. **Deploy**: Create a user-friendly interface for predictions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## ⚠️ Disclaimer

This model is for research and educational purposes only. Predictions should not be used as the sole basis for business decisions. Always consult with legal and regulatory experts for official guidance on mining permit applications.

## 📚 Resources

- [Natural Resources Canada](https://www.nrcan.gc.ca/)
- [Canadian Environmental Assessment Agency](https://www.canada.ca/en/impact-assessment-agency.html)
- [Provincial Mining Associations](https://mining.ca/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.