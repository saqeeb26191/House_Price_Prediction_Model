from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
import pickle
import os
import re
import json
from datetime import datetime
import matplotlib.pyplot as plt
import shap
import matplotlib
matplotlib.use('Agg')
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production
# Database configuration (SQLite)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///house_prediction.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Prediction model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    input_data = db.Column(db.Text, nullable=False)
    prediction_result = db.Column(db.Text, nullable=False)
    explanation = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Global variables for the models
models = {}
scaler = None
label_encoders = {}
data = None



def load_model_and_data():
    """Load the trained models and data"""
    global models, scaler, label_encoders, data
    
    # Load the dataset
    data = pd.read_csv('Bengaluru_House_Data.csv')
    
    # Preprocess data (same as in main script)
    df = data.copy()
    
    # Handle missing values - Fix pandas warnings
    df.loc[:, 'society'] = df['society'].fillna('Unknown')
    df.loc[:, 'balcony'] = df['balcony'].fillna(df['balcony'].median())
    df.loc[:, 'bath'] = df['bath'].fillna(df['bath'].median())
    
    # Clean total_sqft column - Improved function
    def clean_sqft(x):
        if pd.isna(x):
            return np.nan
        
        if isinstance(x, str):
            # Remove any text like "Sq. Meter", "sqft", etc.
            x = re.sub(r'[^\d.\-]', '', x)
            
            if '-' in x:
                # Take average of range
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
    
    # Remove rows with invalid total_sqft
    df = df.dropna(subset=['total_sqft'])
    
    # Extract bedrooms
    def extract_bedrooms(size):
        if pd.isna(size):
            return 2  # Default value
        
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
        return 2  # Default value
    
    df['bedrooms'] = df['size'].apply(extract_bedrooms)
    
    # Handle availability
    df['is_ready_to_move'] = (df['availability'] == 'Ready To Move').astype(int)
    
    # Create location features
    location_counts = df['location'].value_counts()
    location_counts_dict = location_counts.to_dict() if hasattr(location_counts, 'to_dict') else dict(location_counts)
    df['location_frequency'] = df['location'].apply(lambda x: location_counts_dict.get(x, 1))
    
    # Encode categorical variables
    categorical_cols = ['area_type', 'location']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Select features
    feature_cols = [
        'total_sqft', 'bath', 'balcony', 'bedrooms', 'is_ready_to_move',
        'location_frequency', 'area_type_encoded', 'location_encoded'
    ]
    
    # Remove rows with missing values
    df_clean = df[feature_cols + ['price']].dropna()
    
    # Prepare training data
    X = df_clean[feature_cols]
    y = df_clean['price']

    # Split into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create scaler (fit only on training data)
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Train all models on training data
    models.clear()

    # Linear Regression (use scaled features)
    lr = LinearRegression()
    lr.fit(scaler.transform(X_train), y_train)
    models['Linear Regression'] = lr

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb

    if XGBRegressor is not None:
        xgb = XGBRegressor(n_estimators=100, random_state=42)
        xgb.fit(X_train, y_train)
        models['XGBoost'] = xgb

    # Store test data for later evaluation (optional)
    global global_X_test, global_y_test
    global_X_test = X_test
    global_y_test = y_test

    print("All models loaded successfully!")
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Features used: {feature_cols}")

def get_sample_properties():
    """Get sample properties for display on home page - Exactly 9 cards"""
    global data
    if data is None:
        return []
    
    # Initialize with exactly 9 slots
    sample_properties = []
    target_count = 9
    
    # Try to get specific locations first
    try:
        # Vijayanagar properties (up to 2)
        vijayanagar_props = data[data['location'].str.contains('Vijayanagar', case=False, na=False)]
        if len(vijayanagar_props) > 0:
            vijayanagar_sample = vijayanagar_props.sample(min(2, len(vijayanagar_props)))
            for _, row in vijayanagar_sample.iterrows():
                if len(sample_properties) < target_count:
                    sample_properties.append({
                        'location': row['location'],
                        'area_type': row['area_type'],
                        'size': row['size'],
                        'total_sqft': row['total_sqft'],
                        'price': row['price'],
                        'bath': row['bath'],
                        'balcony': row['balcony'],
                        'category': 'Premium' if row['price'] > 100 else 'Mid-Range' if row['price'] >= 50 else 'Budget'
                    })
        
        # Dodda Nekkundi properties (up to 2)
        dodda_nekkundi_props = data[data['location'].str.contains('Dodda Nekkundi', case=False, na=False)]
        if len(dodda_nekkundi_props) > 0:
            dodda_nekkundi_sample = dodda_nekkundi_props.sample(min(2, len(dodda_nekkundi_props)))
            for _, row in dodda_nekkundi_sample.iterrows():
                if len(sample_properties) < target_count:
                    sample_properties.append({
                        'location': row['location'],
                        'area_type': row['area_type'],
                        'size': row['size'],
                        'total_sqft': row['total_sqft'],
                        'price': row['price'],
                        'bath': row['bath'],
                        'balcony': row['balcony'],
                        'category': 'Premium' if row['price'] > 100 else 'Mid-Range' if row['price'] >= 50 else 'Budget'
                    })
    except:
        pass
    
    # Fill remaining slots to get exactly 9 properties
    remaining_slots = target_count - len(sample_properties)
    
    if remaining_slots > 0:
        # Calculate how many of each category to add
        premium_count = max(1, remaining_slots // 3)
        mid_range_count = max(1, remaining_slots // 3)
        budget_count = remaining_slots - premium_count - mid_range_count
        
        # Add Premium properties
        premium_data = data[data['price'] > 100]
        if len(premium_data) > 0:
            premium_sample = premium_data.sample(min(premium_count, len(premium_data)))
            for _, row in premium_sample.iterrows():
                if len(sample_properties) < target_count:
                    sample_properties.append({
                        'location': row['location'],
                        'area_type': row['area_type'],
                        'size': row['size'],
                        'total_sqft': row['total_sqft'],
                        'price': row['price'],
                        'bath': row['bath'],
                        'balcony': row['balcony'],
                        'category': 'Premium'
                    })
        
        # Add Mid-range properties
        mid_range_data = data[(data['price'] >= 50) & (data['price'] <= 100)]
        if len(mid_range_data) > 0:
            mid_range_sample = mid_range_data.sample(min(mid_range_count, len(mid_range_data)))
            for _, row in mid_range_sample.iterrows():
                if len(sample_properties) < target_count:
                    sample_properties.append({
                        'location': row['location'],
                        'area_type': row['area_type'],
                        'size': row['size'],
                        'total_sqft': row['total_sqft'],
                        'price': row['price'],
                        'bath': row['bath'],
                        'balcony': row['balcony'],
                        'category': 'Mid-Range'
                    })
        
        # Add Budget properties
        budget_data = data[data['price'] < 50]
        if len(budget_data) > 0:
            budget_sample = budget_data.sample(min(budget_count, len(budget_data)))
            for _, row in budget_sample.iterrows():
                if len(sample_properties) < target_count:
                    sample_properties.append({
                        'location': row['location'],
                        'area_type': row['area_type'],
                        'size': row['size'],
                        'total_sqft': row['total_sqft'],
                        'price': row['price'],
                        'bath': row['bath'],
                        'balcony': row['balcony'],
                        'category': 'Budget'
                    })
    
    # If we still don't have 9 properties, fill with any available data
    while len(sample_properties) < target_count:
        random_prop = data.sample(1).iloc[0]
        sample_properties.append({
            'location': random_prop['location'],
            'area_type': random_prop['area_type'],
            'size': random_prop['size'],
            'total_sqft': random_prop['total_sqft'],
            'price': random_prop['price'],
            'bath': random_prop['bath'],
            'balcony': random_prop['balcony'],
            'category': 'Premium' if random_prop['price'] > 100 else 'Mid-Range' if random_prop['price'] >= 50 else 'Budget'
        })
    
    # Ensure exactly 9 properties and return
    return sample_properties[:target_count]

@app.route('/')
def index():
    """Landing page with animated title"""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and user.password == password:
            session['user'] = user.email
            session['username'] = user.name
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password!', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Signup page"""
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
        elif User.query.filter_by(email=email).first():
            flash('Email already registered!', 'error')
        else:
            new_user = User(name=name, email=email, password=password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    """Logout user"""
    session.pop('user', None)
    session.pop('username', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/home')
def home():
    """Home page with sample properties"""
    if 'user' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))
    
    sample_properties = get_sample_properties()
    return render_template('home.html', 
                         username=session.get('username', 'User'),
                         properties=sample_properties)

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    """Prediction page"""
    if 'user' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'GET':
        # Defensive: Ensure data is loaded
        if data is None:
            flash('Data not loaded. Please restart the app.', 'error')
            return redirect(url_for('login'))
        # Get unique values for dropdowns - Fix sorting issue with mixed data types
        area_types = sorted(data['area_type'].unique().tolist())
        
        # Handle location sorting with mixed data types
        locations = data['location'].unique().tolist()
        # Filter out any non-string values and sort
        locations = sorted([str(loc) for loc in locations if pd.notna(loc) and str(loc).strip()])
        
        return render_template('predict.html', 
                             username=session.get('username', 'User'),
                             area_types=area_types, 
                             locations=locations,
                             prediction_stats=None)
    
    # Handle POST request for prediction
    try:
        # Defensive: Ensure data is loaded
        if data is None:
            flash('Data not loaded. Please restart the app.', 'error')
            return redirect(url_for('login'))
        # Get form data
        area_type = request.form['area_type']
        location = request.form['location']
        size = request.form['size']
        total_sqft = float(request.form['total_sqft'])
        bath = int(request.form['bath'])
        balcony = int(request.form['balcony'])
        availability = request.form['availability']
        
        # Preprocess the input
        def extract_bedrooms(size):
            if 'BHK' in size:
                return int(size.split()[0])
            elif 'Bedroom' in size:
                return int(size.split()[0])
            elif 'RK' in size:
                return 1
            return 2
        
        bedrooms = extract_bedrooms(size)
        is_ready_to_move = 1 if availability == 'Ready To Move' else 0
        
        # Encode categorical variables
        if not label_encoders or 'area_type' not in label_encoders or 'location' not in label_encoders:
            flash('Model not loaded properly. Please restart the app.', 'error')
            return redirect(url_for('predict_page'))
        area_type_encoded = label_encoders['area_type'].transform([area_type])[0]
        location_encoded = label_encoders['location'].transform([location])[0]
        
        # Get location frequency
        location_counts = data['location'].value_counts()
        location_frequency = location_counts.get(location, 1)
        
        # Create feature vector
        features = np.array([[
            total_sqft, bath, balcony, bedrooms, is_ready_to_move,
            location_frequency, area_type_encoded, location_encoded
        ]])
        
        # Predict with all models
        results = {}
        model_metrics = {}
        r2_scores = []
        rmse_scores = []
        # Always use numpy array for features
        features_np = np.array(features)
        for name, mdl in models.items():
            if name == 'Linear Regression':
                if scaler is not None:
                    pred = mdl.predict(scaler.transform(features_np))[0]
                else:
                    flash('Scaler not loaded. Please restart the app.', 'error')
                    return redirect(url_for('predict_page'))
            elif name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
                pred = mdl.predict(features_np)[0]
            else:
                pred = mdl.predict(features_np)[0]
            results[name] = round(pred, 2)
            # Calculate metrics for each model using training data
            # Use the same features as in training
            if name == 'Linear Regression' and scaler is not None:
                X_train = scaler.transform(models[name].X) if hasattr(models[name], 'X') else scaler.transform(features)
                y_train = models[name].y if hasattr(models[name], 'y') else [pred]
                r2 = mdl.score(X_train, y_train)
                y_pred = mdl.predict(X_train)
            else:
                X_train = models[name].X if hasattr(models[name], 'X') else features
                y_train = models[name].y if hasattr(models[name], 'y') else [pred]
                r2 = mdl.score(X_train, y_train)
                y_pred = mdl.predict(X_train)
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))
            r2_scores.append(r2)
            rmse_scores.append(rmse)
            if hasattr(mdl, 'feature_importances_'):
                fi = mdl.feature_importances_
            else:
                fi = None
            model_metrics[name] = {'r2': r2, 'rmse': rmse, 'feature_importance': fi}
        mean_prediction = np.mean(list(results.values()))
        best_model = min(results, key=lambda k: abs(results[k] - mean_prediction))
        # --- Generate Graphs ---
        import os
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        # 1. Model vs. Predicted Price
        plt.figure(figsize=(7,5))
        bars = plt.bar(list(results.keys()), list(results.values()), color=[('green' if k==best_model else 'skyblue') for k in results.keys()])
        plt.title('Model vs. Predicted Price')
        plt.ylabel('Predicted Price (Lakhs)')
        plt.xlabel('Model')
        plt.xticks(rotation=15)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        price_chart = os.path.join('static', 'model_pred_price.png')
        plt.savefig(price_chart)
        plt.close()
       
        # 4. Feature Importance (for best tree-based model)
        tree_model = None
        for m in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
            if m in models and hasattr(models[m], 'feature_importances_'):
                tree_model = models[m]
                break
        feature_chart = None
        if tree_model is not None:
            feature_names = ['total_sqft', 'bath', 'balcony', 'bedrooms', 'is_ready_to_move', 'location_frequency', 'area_type_encoded', 'location_encoded']
            importances = tree_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(8,5))
            plt.bar([feature_names[i] for i in indices], importances[indices], color='teal')
            plt.title('Feature Importance (Best Tree Model)')
            plt.ylabel('Importance')
            plt.xlabel('Feature')
            plt.xticks(rotation=30)
            plt.tight_layout()
            feature_chart = os.path.join('static', 'feature_importance.png')
            plt.savefig(feature_chart)
            plt.close()
        # 5. Pie chart: Proportion of each model's predicted price
        plt.figure(figsize=(6,6))
        prices = list(results.values())
        labels = list(results.keys())
        pie_colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']
        plt.pie(prices, labels=labels, autopct='%1.1f%%', startangle=140, colors=pie_colors[:len(prices)])
        plt.title('Proportion of Each Model\'s Predicted Price')
        pie_chart = os.path.join('static', 'model_pie.png')
        plt.savefig(pie_chart)
        plt.close()
        # 6. Consensus bar: Absolute difference from mean prediction
        plt.figure(figsize=(7,5))
        mean_pred = mean_prediction
        diffs = [abs(p - mean_pred) for p in prices]
        plt.barh(labels, diffs, color='coral')
        plt.xlabel('Absolute Difference from Mean (Lakhs)')
        plt.title('Model Consensus (Lower is More Stable)')
        for i, v in enumerate(diffs):
            plt.text(v + 1, i, f'{v:.2f}', va='center', fontsize=10)
        plt.tight_layout()
        consensus_chart = os.path.join('static', 'model_consensus.png')
        plt.savefig(consensus_chart)
        plt.close()
        # Pass image filenames to template (show only the 2 main, feature importance, and new 2 charts)
        chart_files = [os.path.basename(price_chart), os.path.basename(pie_chart), os.path.basename(consensus_chart)]
        if feature_chart:
            chart_files.append(os.path.basename(feature_chart))
        # SHAP explainability for all models
        shap_charts_by_model = {}
        # Dynamic explanation for each prediction
        best_model_features = None
        best_model_shap = None
        best_model_importances = None
        best_model_name = best_model
        best_model_pred = results[best_model]
        best_model_diff = abs(best_model_pred - mean_prediction)
        # Find SHAP values and feature importances for best model
        if best_model_name in models:
            mdl = models[best_model_name]
            feature_names = ['total_sqft', 'bath', 'balcony', 'bedrooms', 'is_ready_to_move', 'location_frequency', 'area_type_encoded', 'location_encoded']
            if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
                explainer = shap.TreeExplainer(mdl)
                shap_values = explainer.shap_values(features_np, check_additivity=False)
                best_model_shap = shap_values[0] if isinstance(shap_values, np.ndarray) else shap_values
                best_model_features = dict(zip(feature_names, best_model_shap))
                if hasattr(mdl, 'feature_importances_'):
                    best_model_importances = dict(zip(feature_names, mdl.feature_importances_))
            elif best_model_name == 'Linear Regression':
                explainer = shap.KernelExplainer(mdl.predict, features_np)
                shap_values = explainer.shap_values(features_np)
                best_model_shap = shap_values[0] if isinstance(shap_values, np.ndarray) else shap_values
                best_model_features = dict(zip(feature_names, best_model_shap))
        # Get top impacting feature for SHAP and importance
        top_shap_feature = None
        top_shap_value = None
        if best_model_features:
            top_shap_feature = max(best_model_features, key=lambda k: abs(best_model_features[k]))
            top_shap_value = best_model_features[top_shap_feature]
        top_importance_feature = None
        if best_model_importances:
            top_importance_feature = max(best_model_importances, key=lambda k: best_model_importances[k])
        # Build dynamic explanation
        dynamic_explanation = f"The best model for your input is '{best_model_name}' because its prediction (₹ {best_model_pred} Lakhs) is closest to the mean of all models (₹ {mean_prediction:.2f} Lakhs), with a difference of {best_model_diff:.2f} Lakhs. "
        if top_shap_feature:
            dynamic_explanation += f"For this prediction, the feature with the highest impact was '{top_shap_feature}' (SHAP value: {top_shap_value:.2f}), meaning it contributed most to the predicted price. "
        if top_importance_feature:
            dynamic_explanation += f"Overall, '{top_importance_feature}' is the most important feature for this model. "
        dynamic_explanation += "SHAP charts below show how each feature affected your prediction. The consensus chart shows how close each model's prediction is to the mean, and feature importance shows which features matter most for the best model."
        for model_name, mdl in models.items():
            if model_name not in ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost']:
                continue  # Only process model objects
            try:
                explainer = None
                shap_values = None
                feature_names = ['total_sqft', 'bath', 'balcony', 'bedrooms', 'is_ready_to_move', 'location_frequency', 'area_type_encoded', 'location_encoded']
                model_charts = []
                # Use only tree-based models for TreeExplainer
                if model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
                    explainer = shap.TreeExplainer(mdl)
                    shap_values = explainer.shap_values(features_np, check_additivity=False)
                    # 1. SHAP force plot
                    shap.initjs()
                    plt.figure(figsize=(8, 3))
                    shap.force_plot(explainer.expected_value, shap_values[0], features_np[0], feature_names=feature_names, matplotlib=True, show=False)
                    shap_chart = os.path.join('static', f'shap_force_{model_name.replace(' ', '_').lower()}.png')
                    plt.savefig(shap_chart, bbox_inches='tight')
                    plt.close()
                    model_charts.append(os.path.basename(shap_chart))
                    # 2. SHAP waterfall plot
                    plt.figure(figsize=(8, 5))
                    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], features_np[0], feature_names=feature_names, show=False)
                    waterfall_chart = os.path.join('static', f'shap_waterfall_{model_name.replace(' ', '_').lower()}.png')
                    plt.savefig(waterfall_chart, bbox_inches='tight')
                    plt.close()
                    model_charts.append(os.path.basename(waterfall_chart))
                    # 3. SHAP summary plot (using training data if available)
                    if hasattr(mdl, 'feature_importances_') and 'X' in mdl.__dict__:
                        X_train = mdl.X
                        shap_values_train = explainer.shap_values(X_train, check_additivity=False)
                        plt.figure(figsize=(8, 5))
                        shap.summary_plot(shap_values_train, X_train, feature_names=feature_names, show=False)
                        summary_chart = os.path.join('static', f'shap_summary_{model_name.replace(' ', '_').lower()}.png')
                        plt.savefig(summary_chart, bbox_inches='tight')
                        plt.close()
                        model_charts.append(os.path.basename(summary_chart))
                elif model_name == 'Linear Regression':
                    explainer = shap.KernelExplainer(mdl.predict, features_np)
                    shap_values = explainer.shap_values(features_np)
                    shap.initjs()
                    # 1. SHAP force plot
                    plt.figure(figsize=(8, 3))
                    shap.force_plot(explainer.expected_value, shap_values[0], features_np[0], feature_names=feature_names, matplotlib=True, show=False)
                    shap_chart = os.path.join('static', f'shap_force_{model_name.replace(' ', '_').lower()}.png')
                    plt.savefig(shap_chart, bbox_inches='tight')
                    plt.close()
                    model_charts.append(os.path.basename(shap_chart))
                    # 2. SHAP waterfall plot
                    plt.figure(figsize=(8, 5))
                    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], features_np[0], feature_names=feature_names, show=False)
                    waterfall_chart = os.path.join('static', f'shap_waterfall_{model_name.replace(' ', '_').lower()}.png')
                    plt.savefig(waterfall_chart, bbox_inches='tight')
                    plt.close()
                    model_charts.append(os.path.basename(waterfall_chart))
                    # 3. SHAP summary plot (using input features as a fallback)
                    plt.figure(figsize=(8, 5))
                    shap.summary_plot(shap_values, features_np, feature_names=feature_names, show=False)
                    summary_chart = os.path.join('static', f'shap_summary_{model_name.replace(' ', '_').lower()}.png')
                    plt.savefig(summary_chart, bbox_inches='tight')
                    plt.close()
                    model_charts.append(os.path.basename(summary_chart))
                shap_charts_by_model[model_name] = model_charts
            except Exception as shap_ex:
                print(f'SHAP explanation error for {model_name}: {shap_ex}')
        # Save prediction and explanation to database
        user_email = session.get('user')
        user = User.query.filter_by(email=user_email).first() if user_email else None
        if user:
            input_data = json.dumps({
                'area_type': area_type,
                'location': location,
                'size': size,
                'total_sqft': float(total_sqft),
                'bath': int(bath),
                'balcony': int(balcony),
                'availability': availability
            })
            # Convert all prediction results to float before serializing
            prediction_results = json.dumps({k: float(v) for k, v in results.items()})
            explanation = dynamic_explanation
            new_pred = Prediction(user_id=user.id, input_data=input_data, prediction_result=prediction_results, explanation=explanation)
            db.session.add(new_pred)
            db.session.commit()
        return render_template('predict.html', 
                             username=session.get('username', 'User'),
                             area_types=sorted(data['area_type'].unique().tolist()),
                             locations=sorted([str(loc) for loc in data['location'].unique().tolist() if pd.notna(loc) and str(loc).strip()]),
                             prediction_stats={
                                 'results': results,
                                 'mean_prediction': mean_prediction,
                                 'best_model': best_model,
                                 'chart_files': [os.path.basename(f) for f in chart_files],
                                 'shap_charts_by_model': shap_charts_by_model,
                                 'explainable_ai_text': dynamic_explanation
                             })
    except Exception as e:
        flash(f'Prediction error: {str(e)}', 'error')
        return redirect(url_for('predict_page'))

@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')

if __name__ == '__main__':
    # Initialize database tables
    with app.app_context():
        db.create_all()
    # Load model when starting the app
    load_model_and_data()
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)