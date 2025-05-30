import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 🚗 Define max possible range for your EV
MAX_RANGE_KM = 300  # <-- Change this as per your EV specs

# Load dataset
df = pd.read_csv('ev_fleet_telemetry_synthetic.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['Minute'] = df['Timestamp'].dt.minute
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

df = df.dropna()
df = df.sort_values(by=['Vehicle_ID', 'Timestamp'])

# Define target and features (without Prev_SoC)
target = 'SoC (%)'
categorical_cols = ['Terrain', 'Weather']
features = ['Speed (Km/h)', 'Braking (m/s²)', 'Acceleration (m/s²)', 'Terrain', 'Weather']
numerical_cols = ['Speed (Km/h)', 'Braking (m/s²)', 'Acceleration (m/s²)']

X = df[features]
y = df[target]

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [5, 10, 15, None],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__max_features': [None, 'sqrt', 'log2']
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Predict SoC
y_pred_soc = best_model.predict(X_test)

# Calculate estimated range (based on predicted SoC)
y_pred_range = (y_pred_soc / 100) * MAX_RANGE_KM

# Evaluation
mae = mean_absolute_error(y_test, y_pred_soc)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_soc))
print(f"\n📊 MAE (SoC Prediction): {mae:.2f}")
print(f"📉 RMSE (SoC Prediction): {rmse:.2f}")

# Show a few predictions
results_df = pd.DataFrame({
    'Actual_SoC': y_test.values,
    'Predicted_SoC': y_pred_soc,
    'Estimated_Range_km': y_pred_range
})
print("\n🔍 Sample Predictions:")
print(results_df.head(10))

# Save model
joblib.dump(best_model, 'ev_range_predictor_current_state.pkl')
print("\n✅ Model saved as ev_range_predictor_current_state.pkl")
