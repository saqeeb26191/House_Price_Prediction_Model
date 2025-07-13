# Bengaluru House Price Prediction - ML Project

## 📊 Project Overview

This project implements a comprehensive Machine Learning solution for predicting house prices in Bengaluru, India. The analysis includes data exploration, preprocessing, multiple model comparison, feature importance analysis, and **explainable AI (XAI) with SHAP** for dynamic model selection and transparency.

## 🎯 Key Features

- **Comprehensive Data Analysis**: Exploratory data analysis with visualizations
- **Advanced Preprocessing**: Handles missing values, categorical encoding, and feature engineering
- **Multiple ML Models**: Compares 4 different algorithms
- **Dynamic Model Selection**: Fair comparison based on actual performance, not assumptions
- **Model Evaluation**: Cross-validation and multiple metrics
- **Feature Importance**: Analysis of key factors affecting house prices
- **Interactive Predictions**: Sample prediction functionality with all models
- **Prediction Statistics**: Shows range, mean, and model consensus
- **Modern UI**: Animated, uniform cards for model outputs with icons and badges
- **Insightful Graphs**: Predicted price, pie chart, consensus, and feature importance
- **Explainable AI (XAI)**: SHAP force plot, waterfall plot, and summary plot for every prediction

## 🧠 Explainable AI (XAI) with SHAP

After every prediction, the app generates:
- **SHAP Force Plot**: Shows how each feature pushed the prediction up or down for your input.
- **SHAP Waterfall Plot**: Visualizes the additive contributions of each feature to the final prediction.
- **SHAP Summary Plot**: Shows the overall importance of each feature across the dataset.

These plots help users understand why the model made its prediction, increasing trust and transparency.

## 📈 Models Implemented

1. **Linear Regression** - Baseline linear model
2. **Random Forest** - Ensemble tree-based model
3. **Gradient Boosting** - Sequential ensemble learning
4. **XGBoost** - Optimized gradient boosting

## 🏠 Dataset Features

- **area_type**: Type of area (Super built-up, Built-up, Plot, Carpet)
- **availability**: Availability status
- **location**: Location/area in Bengaluru
- **size**: Number of bedrooms (BHK)
- **total_sqft**: Total square footage
- **bath**: Number of bathrooms
- **balcony**: Number of balconies
- **price**: Target variable (price in lakhs)

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone or download the project files**
   ```bash
   # Make sure you have the following files in your directory:
   # - Bengaluru_House_Data.csv
   # - requirements.txt
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web app**
   ```bash
   python app.py
   ```

4. **Visit**
   - Open your browser and go to: http://localhost:5000

## 🖥️ User Experience

- **Animated, uniform cards** for model predictions, each with a model icon and a "Best" badge for the most stable model.
- **Four insightful graphs**: Model vs. Predicted Price, Model Proportion Pie Chart, Model Consensus, Feature Importance.
- **SHAP explainability**: See why the model predicted the price, with force, waterfall, and summary plots.
- **Modern, responsive UI**: Works on desktop and mobile.

## 🧩 SHAP Visualizations Explained

- **Force Plot**: Shows how each feature contributed to the specific prediction (red = increases price, blue = decreases price).
- **Waterfall Plot**: Breaks down the prediction into additive feature contributions.
- **Summary Plot**: Shows which features are most important overall for the model.

## 📊 Expected Results

The analysis will generate:

1. **Console Output**: Detailed analysis results and model comparisons
2. **data_analysis.png**: Exploratory data analysis visualizations
3. **model_comparison.png**: Model performance comparison charts
4. **feature_importance.png**: Feature importance analysis

## 🔍 Analysis Components

### 1. Data Exploration
- Price distribution analysis
- Area type distribution
- Location-based price analysis
- Correlation analysis
- Missing value assessment

### 2. Data Preprocessing
- Missing value imputation
- Categorical variable encoding
- Feature engineering (bedroom extraction, location frequency)
- Data scaling for appropriate models

### 3. Model Training & Comparison
- Training 4 different ML models
- Cross-validation for robust evaluation
- Multiple performance metrics (R², RMSE, MAE)
- Dynamic model selection based on actual performance

### 4. Feature Importance
- Analysis of key factors affecting house prices
- Comparison across different tree-based models

## 📈 Model Performance

Based on typical results from this dataset:

| Model | R² Score | RMSE | Best For |
|-------|----------|------|----------|
| XGBoost | ~0.85-0.90 | ~25-30 | Often highest R² score |
| Random Forest | ~0.83-0.88 | ~27-32 | Good balance of performance and interpretability |
| Gradient Boosting | ~0.82-0.87 | ~28-33 | Robust performance |
| Linear Models | ~0.60-0.75 | ~35-45 | Baseline comparison |

**Note**: The actual "best" model for each prediction is determined dynamically based on the specific property characteristics and model consensus.

## 🎯 Key Insights

### Most Important Features
1. **Total Square Footage** - Primary driver of house prices
2. **Location** - Significant impact on pricing
3. **Number of Bedrooms** - Important factor for family homes
4. **Number of Bathrooms** - Luxury indicator
5. **Area Type** - Different pricing for different area types

### Price Patterns
- **Location Premium**: Certain areas command significantly higher prices
- **Size Relationship**: Non-linear relationship between size and price
- **Luxury Features**: Additional bathrooms and balconies add value
- **Market Uncertainty**: Prediction ranges show inherent market volatility

### Dynamic Model Selection
- **Most Stable Model**: Prediction closest to the average of all models
- **Model Consensus**: Range of predictions indicates market uncertainty
- **User Empowerment**: All predictions displayed for informed decisions
- **No Bias**: All models treated equally in selection process

## 🔧 Customization

### Adding New Models
```python
# In the train_models method, add new models to the models dictionary
models = {
    'Your Model': YourModelClass(),
    # ... existing models
}
```

### Feature Engineering
```python
# In preprocess_data method, add new features
df['new_feature'] = # your feature engineering logic
```

### Hyperparameter Tuning
```python
# Use GridSearchCV for hyperparameter optimization
from sklearn.model_selection import GridSearchCV
param_grid = {'parameter': [values]}
grid_search = GridSearchCV(model, param_grid, cv=5)
```

## 📝 Sample Usage

```python
from house_price_prediction import HousePricePredictor

# Initialize predictor
predictor = HousePricePredictor('Bengaluru_House_Data.csv')

# Run complete analysis
results = predictor.run_complete_analysis()

# Make a sample prediction
sample_house = {
    'area_type': 'Super built-up  Area',
    'location': 'Whitefield',
    'size': '3 BHK',
    'total_sqft': 1500,
    'bath': 3,
    'balcony': 2,
    'availability': 'Ready To Move'
}

predicted_price = predictor.predict_sample(sample_house)
print(f"Predicted Price: {predicted_price:.2f} Lakhs")
```

## 🎓 Learning Outcomes

This project demonstrates:

1. **End-to-end ML pipeline** development
2. **Data preprocessing** techniques for real-world data
3. **Dynamic model comparison** and selection strategies
4. **Feature engineering** for better model performance
5. **Cross-validation** for robust model evaluation
6. **Visualization** techniques for data analysis
7. **Fair model selection** based on actual performance

## 🔮 Future Enhancements

1. **Web Application**: Create a Flask/Django web app for interactive predictions
2. **Advanced Features**: Add more sophisticated feature engineering
3. **Deep Learning**: Implement neural networks for comparison
4. **Real-time Data**: Connect to live real estate APIs
5. **Geographic Analysis**: Add location-based clustering
6. **Time Series**: Include temporal price trends
7. **Ensemble Methods**: Stacking and voting approaches

## 📄 License

This project is for educational purposes. Feel free to use and modify as needed.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.

---

**Note**: The actual model performance may vary based on the specific dataset and preprocessing choices. This project serves as a comprehensive example of ML workflow implementation with emphasis on fairness and transparency in model selection. 