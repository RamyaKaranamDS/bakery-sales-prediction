import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import sys
import joblib
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

def create_features(df):
    df['dayofweek'] = df['DATE'].dt.dayofweek
    df['month'] = df['DATE'].dt.month
    df['day'] = df['DATE'].dt.day
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df['Days'] = (df['DATE'] - df['DATE'].min()).dt.days
    return df

def main(n_days):
    # Load dataset
    df = pd.read_csv("dataset.csv")
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE')
    df = create_features(df)

    # Features and target
    feature_cols = ['Days', 'dayofweek', 'month', 'day', 'is_weekend']
    X = df[feature_cols]
    y = df['SALES']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Training
    lr_model = LinearRegression()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    lr_model.fit(X_train_scaled, y_train)
    rf_model.fit(X_train_scaled, y_train)

    # Save best model and scaler
    joblib.dump(rf_model, 'rf_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # Evaluation
    linear_pred = lr_model.predict(X_test_scaled)
    rf_pred = rf_model.predict(X_test_scaled)

    results = {
        "Model": ["Linear Regression", "Random Forest Regressor"],
        "R2 Score": [r2_score(y_test, linear_pred), r2_score(y_test, rf_pred)],
        "MSE": [mean_squared_error(y_test, linear_pred), mean_squared_error(y_test, rf_pred)],
        "MAE": [mean_absolute_error(y_test, linear_pred), mean_absolute_error(y_test, rf_pred)]
    }

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv("model_comparison.csv", index=False)

 

    # Predict future sales using Random Forest
    last_date = df['DATE'].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=n_days)
    future_df = pd.DataFrame({'DATE': future_dates})
    future_df = create_features(future_df)
    X_future = future_df[feature_cols]
    X_future_scaled = scaler.transform(X_future)
    rf_future = rf_model.predict(X_future_scaled)

    future_sales = pd.DataFrame({
        'DATE': future_dates,
        'PREDICTED_SALES': rf_future.round().astype(int)
    })

    future_sales.to_csv("future_predictions.csv", index=False)

    print("\nFuture Sales Predictions (Random Forest):")
    print(future_sales)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_sales.py <N_days>")
    else:
        try:
            n_days = int(sys.argv[1])
            main(n_days)
        except ValueError:
            print("Please enter a valid integer for number of days.")