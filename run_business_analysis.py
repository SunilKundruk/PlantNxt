import pandas as pd
import numpy as np

def main():
    print("Starting Business Risk Analysis...")
    
    # 1. Load the 30-day forecasts
    fc_a = pd.read_csv('plots/account_a_30day_forecast.csv')
    fc_b = pd.read_csv('plots/account_b_30day_forecast.csv')
    
    fc_a['ds'] = pd.to_datetime(fc_a['ds'])
    fc_b['ds'] = pd.to_datetime(fc_b['ds'])
    
    # Define weeks (7-day buckets starting from the beginning of the forecast)
    for name, df in [("Account A", fc_a), ("Account B", fc_b)]:
        df['Week'] = ((df['ds'] - df['ds'].min()).dt.days // 7) + 1
        weekly_rev = df.groupby('Week')['yhat'].sum().reset_index()
        weekly_rev['yhat_Lakhs'] = weekly_rev['yhat'] / 1e5
        print(f"\n{name} 30-Day Forecast Weekly breakdown:")
        for idx, row in weekly_rev.iterrows():
            print(f"  Week {int(row['Week'])}: INR {row['yhat']:,.2f} ({row['yhat_Lakhs']:.2f} Lakhs)")
        
        # Identify highest risk week (lowest revenue week)
        min_week = weekly_rev.sort_values(by='yhat').iloc[0]
        print(f"  --> Highest Risk Week: Week {int(min_week['Week'])} (Projected Revenue: INR {min_week['yhat']:,.2f})")

    # 2. Analyze customer concentration in cleaned transactions
    df_trans = pd.read_csv('cleaned_transactions.csv.gz', compression='gzip')
    
    print("\n--- Customer Concentration Analysis ---")
    for name in ["Account A", "Account B"]:
        df_act = df_trans[df_trans['Account'] == name].copy()
        cust_rev = df_act.groupby('Cust Code')['Revenue'].sum().reset_index()
        cust_rev = cust_rev.sort_values(by='Revenue', ascending=False).reset_index(drop=True)
        total_rev = cust_rev['Revenue'].sum()
        cust_rev['Percentage'] = (cust_rev['Revenue'] / total_rev) * 100
        cust_rev['Cumulative_Percentage'] = cust_rev['Percentage'].cumsum()
        
        print(f"\n{name}:")
        print(f"  Total Customers: {len(cust_rev)}")
        print(f"  Top 5 Customers by Revenue:")
        for idx, row in cust_rev.head(5).iterrows():
            print(f"    {row['Cust Code']}: INR {row['Revenue']:,.2f} ({row['Percentage']:.2f}%)")
            
        # Top customers driving 80%
        cust_80 = cust_rev[cust_rev['Cumulative_Percentage'] <= 80.5]
        print(f"  Customers driving 80% revenue: {len(cust_80) + 1} out of {len(cust_rev)} ({(len(cust_80)+1)/len(cust_rev)*100:.2f}%)")
        
        # 15% Underperformance Exposure Shock
        shock_val = total_rev * 0.15
        print(f"  15% Underperformance Shock impact: INR {shock_val:,.2f}")
        print(f"  This underperformance is equivalent to completely losing their top {len(cust_rev[cust_rev['Cumulative_Percentage'] <= 15.0]) + 1} customer(s).")

if __name__ == '__main__':
    main()
