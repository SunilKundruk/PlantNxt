import pandas as pd
import numpy as np
import os

def clean_column(val):
    if pd.isna(val):
        return np.nan
    s = str(val).replace(',', '').strip()
    try:
        return float(s)
    except ValueError:
        return np.nan

def main():
    print("Starting Data Preparation...")
    csv_path = 'revenue_ledger.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return

    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} transactions.")

    # Clean numeric columns
    for col in ['Price', 'Tax', 'Others']:
        df[col] = df[col].apply(clean_column)
        print(f"Cleaned {col} column. Missing now: {df[col].isnull().sum()}")

    # Parse dates
    df['Inv Date'] = pd.to_datetime(df['Inv Date'], errors='coerce')
    print(f"Parsed Inv Date. Missing dates: {df['Inv Date'].isnull().sum()}")
    df = df.dropna(subset=['Inv Date'])
    df = df.sort_values(by='Inv Date').reset_index(drop=True)

    # Inspect quantities
    neg_qty = (df['Quantity'] < 0).sum()
    zero_qty = (df['Quantity'] == 0).sum()
    pos_qty = (df['Quantity'] > 0).sum()
    print(f"Quantity breakdown: Positive={pos_qty}, Negative (returns)={neg_qty}, Zero={zero_qty}")

    # Impute missing values for Tax and Others
    # We can compute standard tax rate per part code if available
    print("Imputing missing Tax and Others...")
    
    # 1. Tax imputation
    # Let's calculate typical tax rate (Tax / (Quantity * Price)) for each part
    df['Base_Val'] = df['Quantity'] * df['Price']
    
    # Avoid division by zero
    non_zero_base = (df['Base_Val'] != 0)
    df['Tax_Rate'] = np.nan
    df.loc[non_zero_base & df['Tax'].notnull(), 'Tax_Rate'] = (
        df.loc[non_zero_base & df['Tax'].notnull(), 'Tax'] / df.loc[non_zero_base & df['Tax'].notnull(), 'Base_Val']
    )
    
    # Get median tax rate per Part Code
    part_tax_rates = df.groupby('Part Code')['Tax_Rate'].transform('median')
    # Fill missing Tax Rate with account-specific median tax rate
    account_tax_rates = df.groupby('Account')['Tax_Rate'].transform('median')
    df['Tax_Rate'] = df['Tax_Rate'].fillna(part_tax_rates).fillna(account_tax_rates).fillna(0)
    
    # Impute missing Tax using (Base_Val * Tax_Rate)
    missing_tax_mask = df['Tax'].isnull()
    df.loc[missing_tax_mask, 'Tax'] = df.loc[missing_tax_mask, 'Base_Val'] * df.loc[missing_tax_mask, 'Tax_Rate']

    # 2. Others imputation
    # Let's calculate typical Others per Unit (Others / Quantity) for each part
    df['Others_Rate'] = np.nan
    non_zero_qty = (df['Quantity'] != 0)
    df.loc[non_zero_qty & df['Others'].notnull(), 'Others_Rate'] = (
        df.loc[non_zero_qty & df['Others'].notnull(), 'Others'] / df.loc[non_zero_qty & df['Others'].notnull(), 'Quantity']
    )
    
    # Get median Others rate per Part Code
    part_others_rates = df.groupby('Part Code')['Others_Rate'].transform('median')
    account_others_rates = df.groupby('Account')['Others_Rate'].transform('median')
    df['Others_Rate'] = df['Others_Rate'].fillna(part_others_rates).fillna(account_others_rates).fillna(0)
    
    # Impute missing Others using (Quantity * Others_Rate)
    missing_others_mask = df['Others'].isnull()
    df.loc[missing_others_mask, 'Others'] = df.loc[missing_others_mask, 'Quantity'] * df.loc[missing_others_mask, 'Others_Rate']

    print(f"After imputation: Tax missing = {df['Tax'].isnull().sum()}, Others missing = {df['Others'].isnull().sum()}")

    # Compute Total Revenue for each transaction
    # Total Revenue = Quantity * Price + Tax + Others
    df['Revenue'] = df['Quantity'] * df['Price'] + df['Tax'] + df['Others']

    # Summary statistics
    print(f"Total Revenue across dataset: INR {df['Revenue'].sum():,.2f}")

    # Group by Inv Date and Account to create daily time-series
    daily_df = df.groupby(['Inv Date', 'Account']).agg(
        Revenue=('Revenue', 'sum'),
        Quantity=('Quantity', 'sum'),
        Transactions=('Part Code', 'count')
    ).reset_index()

    # Split into Account A and Account B
    df_a = daily_df[daily_df['Account'] == 'Account A'].copy().rename(columns={'Inv Date': 'ds', 'Revenue': 'y'}).set_index('ds')
    df_b = daily_df[daily_df['Account'] == 'Account B'].copy().rename(columns={'Inv Date': 'ds', 'Revenue': 'y'}).set_index('ds')

    # Fill missing dates in the daily time-series to ensure continuous index
    all_dates_a = pd.date_range(start=df_a.index.min(), end=df_a.index.max(), freq='D')
    df_a = df_a.reindex(all_dates_a)
    df_a['y'] = df_a['y'].fillna(0)
    df_a['Quantity'] = df_a['Quantity'].fillna(0)
    df_a['Transactions'] = df_a['Transactions'].fillna(0)
    df_a['Account'] = 'Account A'
    df_a = df_a.reset_index().rename(columns={'index': 'ds'})

    all_dates_b = pd.date_range(start=df_b.index.min(), end=df_b.index.max(), freq='D')
    df_b = df_b.reindex(all_dates_b)
    df_b['y'] = df_b['y'].fillna(0)
    df_b['Quantity'] = df_b['Quantity'].fillna(0)
    df_b['Transactions'] = df_b['Transactions'].fillna(0)
    df_b['Account'] = 'Account B'
    df_b = df_b.reset_index().rename(columns={'index': 'ds'})

    print(f"\nAccount A daily data shape: {df_a.shape} (from {df_a['ds'].min().date()} to {df_a['ds'].max().date()})")
    print(f"Account B daily data shape: {df_b.shape} (from {df_b['ds'].min().date()} to {df_b['ds'].max().date()})")

    # Save cleaned transactions and daily data
    df.to_csv('cleaned_transactions.csv.gz', index=False, compression='gzip')
    df_a.to_csv('daily_account_a.csv', index=False)
    df_b.to_csv('daily_account_b.csv', index=False)
    print("\nSaved CSV files successfully!")

if __name__ == '__main__':
    main()
