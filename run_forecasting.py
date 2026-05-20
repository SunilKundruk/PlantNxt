import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import holidays
import os

# Set style for professional charts
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.figsize': (12, 6)
})

C_PRIMARY = "#3A6B35"
C_SECONDARY = "#CBD18F"
C_ACCENT = "#E3B448"
C_DARK = "#1A1A1D"

# MAPE calculation helper
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero by filtering out zero actual revenues
    mask = y_true > 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def create_features(df):
    df = df.copy()
    # Sort by date
    df = df.sort_values('ds').reset_index(drop=True)
    
    # 1. Calendar features
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['quarter'] = df['ds'].dt.quarter
    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year
    df['dayofyear'] = df['ds'].dt.dayofyear
    df['dayofmonth'] = df['ds'].dt.day
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # 2. Lag features
    df['y_lag1'] = df['y'].shift(1)
    df['y_lag7'] = df['y'].shift(7)
    df['y_lag30'] = df['y'].shift(30)
    
    # 3. Rolling window statistics
    df['y_roll7_mean'] = df['y'].shift(1).rolling(window=7, min_periods=1).mean()
    df['y_roll7_std'] = df['y'].shift(1).rolling(window=7, min_periods=1).std()
    df['y_roll30_mean'] = df['y'].shift(1).rolling(window=30, min_periods=1).mean()
    df['y_roll30_std'] = df['y'].shift(1).rolling(window=30, min_periods=1).std()
    
    # 4. Indian Holidays
    in_holidays = holidays.country_holidays('IN')
    df['is_holiday'] = df['ds'].apply(lambda d: 1 if d in in_holidays else 0)
    
    # Fill remaining NaNs from rolling stats
    df = df.fillna(method='bfill')
    return df

def train_eval_xgboost(df, train_idx, test_idx, feature_cols):
    X = df[feature_cols]
    y = df['y']
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    
    # Train model
    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predict
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None) # Revenue cannot be negative
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    mape = calculate_mape(y_test, preds)
    
    return preds, rmse, mae, mape, model

def train_eval_prophet(df, train_idx, test_idx):
    train_df = df.iloc[train_idx][['ds', 'y']].copy()
    test_df = df.iloc[test_idx][['ds', 'y']].copy()
    
    # Initialize and configure Prophet
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive'
    )
    # Add Indian holidays natively
    model.add_country_holidays(country_name='IN')
    
    model.fit(train_df)
    
    # Predict on test period
    future = model.make_future_dataframe(periods=len(test_df), freq='D')
    forecast = model.predict(future)
    
    # Extract test predictions
    test_forecast = forecast.iloc[test_idx]
    preds = test_forecast['yhat'].values
    preds = np.clip(preds, 0, None) # Revenue cannot be negative
    
    # Metrics
    y_test = test_df['y'].values
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    mape = calculate_mape(y_test, preds)
    
    return preds, rmse, mae, mape, model, forecast

def main():
    print("Starting Forecasting Pipeline...")
    os.makedirs('plots', exist_ok=True)
    
    results = {}
    
    for name, filename in [("Account A", "daily_account_a.csv"), ("Account B", "daily_account_b.csv")]:
        print(f"\n==================== Modelling {name} ====================")
        df_raw = pd.read_csv(filename)
        df_raw['ds'] = pd.to_datetime(df_raw['ds'])
        
        # Build features for ML
        df_feats = create_features(df_raw)
        
        # Chronological Split (last 30 days as test)
        n_rows = len(df_feats)
        test_size = 30
        train_idx = list(range(0, n_rows - test_size))
        test_idx = list(range(n_rows - test_size, n_rows))
        
        test_dates = df_feats['ds'].iloc[test_idx]
        y_test = df_feats['y'].iloc[test_idx].values
        
        print(f"Total days: {n_rows}")
        print(f"Training days: {len(train_idx)} (up to {df_feats['ds'].iloc[train_idx[-1]].date()})")
        print(f"Testing days: {len(test_idx)} (from {test_dates.min().date()} to {test_dates.max().date()})")
        
        # 1. Train & Eval Prophet
        print("Training Prophet...")
        p_preds, p_rmse, p_mae, p_mape, p_model, p_forecast = train_eval_prophet(df_feats, train_idx, test_idx)
        print(f"Prophet Metrics - RMSE: {p_rmse:,.2f}, MAE: {p_mae:,.2f}, MAPE: {p_mape:.2f}%")
        
        # 2. Train & Eval XGBoost
        print("Training XGBoost...")
        feature_cols = [
            'dayofweek', 'quarter', 'month', 'dayofmonth', 'dayofyear', 'is_weekend', 'is_holiday',
            'y_lag1', 'y_lag7', 'y_lag30', 'y_roll7_mean', 'y_roll7_std', 'y_roll30_mean', 'y_roll30_std'
        ]
        x_preds, x_rmse, x_mae, x_mape, x_model = train_eval_xgboost(df_feats, train_idx, test_idx, feature_cols)
        print(f"XGBoost Metrics - RMSE: {x_rmse:,.2f}, MAE: {x_mae:,.2f}, MAPE: {x_mape:.2f}%")
        
        # Compare and Choose Best
        best_model = "Prophet" if p_mape < x_mape else "XGBoost"
        print(f"--> Best Model for {name} based on MAPE: {best_model}")
        
        # Save evaluation metrics
        results[name] = {
            'prophet': {'rmse': p_rmse, 'mae': p_mae, 'mape': p_mape},
            'xgboost': {'rmse': x_rmse, 'mae': x_mae, 'mape': x_mape},
            'best': best_model
        }
        
        # 3. Plot Actual vs Predicted on Test Set
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, y_test / 1e5, 'o-', color=C_DARK, label='Actual Daily Revenue')
        plt.plot(test_dates, p_preds / 1e5, 's--', color=C_PRIMARY, label=f'Prophet (MAPE: {p_mape:.1f}%)')
        plt.plot(test_dates, x_preds / 1e5, 'd--', color=C_ACCENT, label=f'XGBoost (MAPE: {x_mape:.1f}%)')
        plt.title(f"{name} Model Evaluation: Actual vs. Forecasted on Test Set", fontweight='bold')
        plt.xlabel("Date")
        plt.ylabel("Revenue (Lakhs INR)")
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(f"plots/{name.lower().replace(' ', '_')}_val_comparison.png", dpi=150)
        plt.close()

        # 4. Generate 30-Day Forward Forecast (using Prophet for uncertainty intervals)
        print("Generating 30-Day Forward Forecast with Prophet...")
        # Fit Prophet on full data for the actual forward forecast
        final_prophet = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='additive'
        )
        final_prophet.add_country_holidays(country_name='IN')
        final_prophet.fit(df_feats[['ds', 'y']])
        
        future_df = final_prophet.make_future_dataframe(periods=30, freq='D')
        forecast_df = final_prophet.predict(future_df)
        
        # Extract forward forecast
        forward_forecast = forecast_df.tail(30).copy()
        forward_forecast['yhat'] = np.clip(forward_forecast['yhat'], 0, None)
        forward_forecast['yhat_lower'] = np.clip(forward_forecast['yhat_lower'], 0, None)
        forward_forecast['yhat_upper'] = np.clip(forward_forecast['yhat_upper'], 0, None)
        
        # Save forecast to CSV
        forward_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(
            f"plots/{name.lower().replace(' ', '_')}_30day_forecast.csv", index=False
        )
        
        # Plot 30-Day Forward Forecast
        plt.figure(figsize=(14, 6))
        # Plot recent actuals (last 90 days) for clarity
        recent_actuals = df_feats.tail(90)
        plt.plot(recent_actuals['ds'], recent_actuals['y'] / 1e5, color=C_DARK, label='Historical Daily Revenue')
        plt.plot(forward_forecast['ds'], forward_forecast['yhat'] / 1e5, color=C_PRIMARY, linewidth=2.5, label='Projected Forecast (yhat)')
        plt.fill_between(
            forward_forecast['ds'], 
            forward_forecast['yhat_lower'] / 1e5, 
            forward_forecast['yhat_upper'] / 1e5, 
            color=C_PRIMARY, alpha=0.2, label='Confidence Interval (80%)'
        )
        plt.title(f"{name} 30-Day Forward Sales Revenue Projection", fontweight='bold')
        plt.xlabel("Date")
        plt.ylabel("Revenue (Lakhs INR)")
        plt.legend(loc='upper left', frameon=True)
        plt.tight_layout()
        plt.savefig(f"plots/{name.lower().replace(' ', '_')}_forward_forecast.png", dpi=150)
        plt.close()
        
        print(f"30-Day Forward Forecast for {name} saved successfully!")
        print(f"  Projected Total Revenue: INR {forward_forecast['yhat'].sum():,.2f}")

    print("\n==================== Validation Summary ====================")
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Prophet -> RMSE: {metrics['prophet']['rmse']:,.2f}, MAE: {metrics['prophet']['mae']:,.2f}, MAPE: {metrics['prophet']['mape']:.2f}%")
        print(f"  XGBoost -> RMSE: {metrics['xgboost']['rmse']:,.2f}, MAE: {metrics['xgboost']['mae']:,.2f}, MAPE: {metrics['xgboost']['mape']:.2f}%")
        print(f"  Best Choice: {metrics['best']}")

if __name__ == '__main__':
    main()
