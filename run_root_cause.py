import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import os

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})

C_PRIMARY = "#3A6B35"
C_SECONDARY = "#CBD18F"
C_ACCENT = "#E3B448"
C_DARK = "#1A1A1D"

def main():
    print("Starting Root Cause & What-If Analysis...")
    os.makedirs('plots', exist_ok=True)
    
    # Load cleaned transaction ledger
    df_trans = pd.read_csv('cleaned_transactions.csv.gz', compression='gzip')
    df_trans['Inv Date'] = pd.to_datetime(df_trans['Inv Date'])
    
    # Load daily Account A time series
    df_a = pd.read_csv('daily_account_a.csv')
    df_a['ds'] = pd.to_datetime(df_a['ds'])
    
    # Focus on the anomaly: May 1st, 2024
    target_date = pd.to_datetime('2024-05-01')
    
    print("\n--- 1. Anomaly Overview ---")
    print(f"Target Anomaly Date: {target_date.date()} (International Workers' Day / May Day)")
    
    # Get actual revenue on the anomaly date
    actual_rev = df_a[df_a['ds'] == target_date]['y'].values[0]
    print(f"Actual Daily Revenue on {target_date.date()}: INR {actual_rev:,.2f} ({actual_rev/1e5:.2f} Lakhs)")
    
    # Look at adjacent dates (April 25 to May 7, 2024)
    adjacent = df_a[(df_a['ds'] >= '2024-04-25') & (df_a['ds'] <= '2024-05-07')].copy()
    print("\nAdjacent Days Revenue (Lakhs INR):")
    for idx, row in adjacent.iterrows():
        print(f"  {row['ds'].date()} ({row['ds'].day_name()}): INR {row['y']:,.2f} ({row['y']/1e5:.2f} Lakhs)")
        
    # Transaction analysis for the anomaly day vs. normal day
    # Let's compare May 1st, 2024 (Wednesday) with May 8th, 2024 (Wednesday)
    comp_date = pd.to_datetime('2024-05-08')
    trans_anomaly = df_trans[(df_trans['Account'] == 'Account A') & (df_trans['Inv Date'] == target_date)]
    trans_normal = df_trans[(df_trans['Account'] == 'Account A') & (df_trans['Inv Date'] == comp_date)]
    
    print("\n--- 2. Transactional Evidence ---")
    print(f"May 1, 2024 (Anomaly): Transactions count: {len(trans_anomaly)}, Total Quantity: {trans_anomaly['Quantity'].sum()}")
    print(f"May 8, 2024 (Normal):  Transactions count: {len(trans_normal)}, Total Quantity: {trans_normal['Quantity'].sum()}")
    
    # 3. What-If Counterfactual Estimation
    # Train a Prophet model excluding the anomaly day, and predict its value!
    print("\n--- 3. What-If Counterfactual Estimation ---")
    train_df = df_a[df_a['ds'] != target_date][['ds', 'y']].copy()
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    # We do NOT add the May 1 holiday to let the model forecast what a "normal" Wednesday would have looked like!
    model.fit(train_df)
    
    forecast = model.predict(pd.DataFrame({'ds': [target_date]}))
    counterfactual_rev = forecast['yhat'].values[0]
    lower_bound = forecast['yhat_lower'].values[0]
    upper_bound = forecast['yhat_upper'].values[0]
    
    print(f"Counterfactual Estimated Revenue: INR {counterfactual_rev:,.2f} ({counterfactual_rev/1e5:.2f} Lakhs)")
    print(f"  Confidence Interval: INR {lower_bound:,.2f} to {upper_bound:,.2f}")
    
    revenue_loss = counterfactual_rev - actual_rev
    print(f"Estimated Revenue Impact (Loss): INR {revenue_loss:,.2f} ({revenue_loss/1e5:.2f} Lakhs)")
    
    # 4. Generate Anomaly Chart
    plt.figure(figsize=(12, 6))
    
    # Plot surrounding actuals
    surrounding = df_a[(df_a['ds'] >= '2024-04-15') & (df_a['ds'] <= '2024-05-15')]
    plt.plot(surrounding['ds'], surrounding['y'] / 1e5, 'o-', color=C_DARK, label='Actual Daily Revenue')
    
    # Highlight the anomaly
    plt.scatter([target_date], [actual_rev / 1e5], color='red', s=150, zorder=5, label='Actual Anomaly (May Day Shut Down)')
    
    # Plot counterfactual estimate
    plt.scatter([target_date], [counterfactual_rev / 1e5], color=C_PRIMARY, s=150, zorder=5, label='What-If Counterfactual (Model Estimate)')
    plt.errorbar([target_date], [counterfactual_rev / 1e5], 
                 yerr=[[(counterfactual_rev - lower_bound) / 1e5], [(upper_bound - counterfactual_rev) / 1e5]], 
                 fmt='none', ecolor=C_PRIMARY, elinewidth=2, capsize=5, label='Estimated Range (80%)')
    
    plt.title("Account A: Root Cause & What-If Analysis for May 1, 2024 Dip", fontweight='bold')
    plt.xlabel("Invoice Date")
    plt.ylabel("Revenue (Lakhs INR)")
    plt.legend(frameon=True, loc='lower left')
    plt.tight_layout()
    plt.savefig('plots/account_a_may_day_anomaly.png', dpi=150)
    plt.close()
    
    print("\nRoot Cause Analysis Completed! Chart saved as 'plots/account_a_may_day_anomaly.png'.")

if __name__ == '__main__':
    main()
