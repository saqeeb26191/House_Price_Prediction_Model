import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
import re

def test_prediction():
    print("🧪 Testing House Price Prediction System (All Models)...")
    print("=" * 60)
    
    try:
        # Load data
        print("📊 Loading dataset...")
        data = pd.read_csv('Bengaluru_House_Data.csv')
        print(f"✅ Dataset loaded: {data.shape[0]} properties")
        
        # Preprocess data
        print("🔧 Preprocessing data...")
        df = data.copy()
        
        # Handle missing values
        df.loc[:, 'society'] = df['society'].fillna('Unknown')
        df.loc[:, 'balcony'] = df['balcony'].fillna(df['balcony'].median())
        df.loc[:, 'bath'] = df['bath'].fillna(df['bath'].median())
        
        # Clean total_sqft
        def clean_sqft(x):
            if pd.isna(x):
                return np.nan
            
            if isinstance(x, str):
                x = re.sub(r'[^\d.\-]', '', x)
                
                if '-' in x:
                    parts = x.split('-')
                    try:
                        return (float(parts[0].strip()) + float(parts[1].strip())) / 2
                    except:
                        return np.nan
                else:
                    try:
                        return float(x)
                    except:
                        return np.nan
            elif isinstance(x, (int, float)):
                return float(x)
            else:
                return np.nan
        
        df['total_sqft'] = df['total_sqft'].apply(clean_sqft)
        df = df.dropna(subset=['total_sqft'])
        
        # Extract bedrooms
        def extract_bedrooms(size):
            if pd.isna(size):
                return 2
            
            if isinstance(size, str):
                if 'BHK' in size:
                    try:
                        return int(size.split()[0])
                    except:
                        return 2
                elif 'Bedroom' in size:
                    try:
                        return int(size.split()[0])
                    except:
                        return 2
                elif 'RK' in size:
                    return 1
            return 2
        
        df['bedrooms'] = df['size'].apply(extract_bedrooms)
        df['is_ready_to_move'] = (df['availability'] == 'Ready To Move').astype(int)
        
        # Location features
        location_counts = df['location'].value_counts()
        df['location_frequency'] = df['location'].map(location_counts)
        
        # Encode categorical variables
        le_area = LabelEncoder()
        le_location = LabelEncoder()
        df['area_type_encoded'] = le_area.fit_transform(df['area_type'].astype(str))
        df['location_encoded'] = le_location.fit_transform(df['location'].astype(str))
        
        # Select features
        feature_cols = [
            'total_sqft', 'bath', 'balcony', 'bedrooms', 'is_ready_to_move',
            'location_frequency', 'area_type_encoded', 'location_encoded'
        ]
        
        df_clean = df[feature_cols + ['price']].dropna()
        print(f"✅ Cleaned dataset: {df_clean.shape[0]} properties")
        
        # Train all models
        print("🤖 Training all models...")
        X = df_clean[feature_cols]
        y = df_clean['price']
        
        # Create scaler for Linear Regression
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        if XGBRegressor is not None:
            models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42)
        
        trained_models = {}
        for name, model in models.items():
            print(f"   Training {name}...")
            if name == 'Linear Regression':
                model.fit(X_scaled, y)
            else:
                model.fit(X, y)
            trained_models[name] = model
        
        print("✅ All models trained successfully!")
        
        # Test predictions
        print("\n🏠 Testing sample predictions with all models...")
        
        test_cases = [
            {
                'name': 'Premium Property (Whitefield)',
                'area_type': 'Super built-up  Area',
                'location': 'Whitefield',
                'size': '3 BHK',
                'total_sqft': 1500,
                'bath': 3,
                'balcony': 2,
                'availability': 'Ready To Move'
            },
            {
                'name': 'Mid-range Property (Electronic City)',
                'area_type': 'Super built-up  Area',
                'location': 'Electronic City',
                'size': '2 BHK',
                'total_sqft': 1000,
                'bath': 2,
                'balcony': 1,
                'availability': 'Ready To Move'
            },
            {
                'name': 'Budget Property (Kengeri)',
                'area_type': 'Built-up  Area',
                'location': 'Kengeri',
                'size': '1 BHK',
                'total_sqft': 600,
                'bath': 1,
                'balcony': 1,
                'availability': 'Ready To Move'
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. {test_case['name']}")
            print("   Details:")
            for key, value in test_case.items():
                if key != 'name':
                    print(f"   - {key}: {value}")
            
            # Prepare features
            bedrooms = extract_bedrooms(test_case['size'])
            is_ready = 1 if test_case['availability'] == 'Ready To Move' else 0
            
            area_encoded = le_area.transform([test_case['area_type']])[0]
            location_encoded = le_location.transform([test_case['location']])[0]
            location_freq = location_counts.get(test_case['location'], 1)
            
            features = np.array([[
                test_case['total_sqft'], test_case['bath'], test_case['balcony'],
                bedrooms, is_ready, location_freq, area_encoded, location_encoded
            ]])
            
            features_scaled = scaler.transform(features)
            
            # Get predictions from all models
            predictions = {}
            print("   Predictions from all models:")
            print("   " + "-" * 40)
            
            for name, model in trained_models.items():
                if name == 'Linear Regression':
                    pred = model.predict(features_scaled)[0]
                else:
                    pred = model.predict(features)[0]
                predictions[name] = pred
                print(f"   {name:<20}: ₹{pred:.2f} Lakhs")
            
            # Calculate prediction statistics
            pred_values = list(predictions.values())
            mean_pred = np.mean(pred_values)
            min_pred = min(pred_values)
            max_pred = max(pred_values)
            range_pred = max_pred - min_pred
            
            # Find most stable model (closest to mean)
            most_stable_model = min(predictions.keys(), key=lambda k: abs(predictions[k] - mean_pred))
            
            print("   " + "-" * 40)
            print(f"   📊 Prediction Summary:")
            print(f"      Average: ₹{mean_pred:.2f} Lakhs")
            print(f"      Range: ₹{range_pred:.2f} Lakhs (₹{min_pred:.2f} - ₹{max_pred:.2f})")
            print(f"      Most Stable: {most_stable_model} (₹{predictions[most_stable_model]:.2f} Lakhs)")
        
        print("\n🎉 All tests completed successfully!")
        print("\n📊 System Performance Summary:")
        print(f"   - Training samples: {len(X)}")
        print(f"   - Features used: {len(feature_cols)}")
        print(f"   - Models tested: {len(trained_models)}")
        print(f"   - Feature names: {feature_cols}")
        
        # Feature importance from Random Forest
        rf_model = trained_models['Random Forest']
        importances = rf_model.feature_importances_
        feature_importance = list(zip(feature_cols, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("\n🔍 Top 5 Most Important Features (Random Forest):")
        for i, (feature, importance) in enumerate(feature_importance[:5], 1):
            print(f"   {i}. {feature}: {importance:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_prediction()
    if success:
        print("\n✅ System is working correctly with dynamic model selection!")
        print("🌐 You can now run 'python app.py' to start the web application")
    else:
        print("\n❌ System needs fixing") 