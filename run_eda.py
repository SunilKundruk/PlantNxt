import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# Custom premium palette colors
C_PRIMARY = "#3A6B35"   # Forest Green
C_SECONDARY = "#CBD18F" # Muted Sage Green
C_ACCENT = "#E3B448"    # Amber gold
C_DARK = "#1A1A1D"      # Dark Charcoal

def main():
    print("Starting Exploratory Data Analysis...")
    os.makedirs('plots', exist_ok=True)

    # Load cleaned data
    df_trans = pd.read_csv('cleaned_transactions.csv.gz', compression='gzip')
    df_a = pd.read_csv('daily_account_a.csv')
    df_b = pd.read_csv('daily_account_b.csv')

    df_trans['Inv Date'] = pd.to_datetime(df_trans['Inv Date'])
    df_a['ds'] = pd.to_datetime(df_a['ds'])
    df_b['ds'] = pd.to_datetime(df_b['ds'])

    # 1. Macro Trends Analysis & Visualization
    print("\n--- 1. Macro Trends ---")
    for name, df in [("Account A", df_a), ("Account B", df_b)]:
        print(f"\n{name} Revenue Summary:")
        print(f"  Total Days: {len(df)}")
        print(f"  Total Revenue: INR {df['y'].sum():,.2f}")
        print(f"  Daily Mean: INR {df['y'].mean():,.2f}")
        print(f"  Daily Median: INR {df['y'].median():,.2f}")
        print(f"  Daily Max: INR {df['y'].max():,.2f}")
        print(f"  Daily Min: INR {df['y'].min():,.2f}")

        # Compute rolling 30-day average
        df['y_roll30'] = df['y'].rolling(window=30, min_periods=1).mean()

        plt.figure(figsize=(14, 6))
        plt.plot(df['ds'], df['y'] / 1e5, color=C_SECONDARY, alpha=0.5, label='Daily Revenue (Lakhs INR)')
        plt.plot(df['ds'], df['y_roll30'] / 1e5, color=C_PRIMARY, linewidth=2.5, label='30-Day Moving Average')
        plt.title(f"{name} Daily Sales Revenue Trend (2022 - 2026)", fontweight='bold', color=C_DARK)
        plt.xlabel("Invoice Date")
        plt.ylabel("Revenue (Lakhs INR)")
        plt.legend(loc='upper left', frameon=True)
        plt.tight_layout()
        plt.savefig(f"plots/{name.lower().replace(' ', '_')}_revenue_trend.png", dpi=150)
        plt.close()

    # 2. Seasonality & Day-of-Week / Month-of-Year Effects
    print("\n--- 2. Seasonality ---")
    for name, df in [("Account A", df_a), ("Account B", df_b)]:
        # Extract date features
        df['DayOfWeek'] = df['ds'].dt.day_name()
        df['Month'] = df['ds'].dt.month_name()
        df['MonthNum'] = df['ds'].dt.month
        df['DayOfWeekNum'] = df['ds'].dt.dayofweek # Monday=0, Sunday=6

        # Day of Week Seasonality
        dow_avg = df.groupby(['DayOfWeekNum', 'DayOfWeek'])['y'].mean().reset_index()
        plt.figure(figsize=(10, 5))
        sns.barplot(data=dow_avg, x='DayOfWeek', y=df['y']/1e5, palette='viridis', hue='DayOfWeek', legend=False)
        plt.title(f"{name} Average Daily Revenue by Day of Week", fontweight='bold', color=C_DARK)
        plt.xlabel("Day of Week")
        plt.ylabel("Average Revenue (Lakhs INR)")
        plt.tight_layout()
        plt.savefig(f"plots/{name.lower().replace(' ', '_')}_dow_seasonality.png", dpi=150)
        plt.close()

        # Month Seasonality
        month_avg = df.groupby(['MonthNum', 'Month'])['y'].mean().reset_index()
        plt.figure(figsize=(12, 5))
        sns.barplot(data=month_avg, x='Month', y=df['y']/1e5, palette='magma', hue='Month', legend=False)
        plt.title(f"{name} Average Daily Revenue by Month", fontweight='bold', color=C_DARK)
        plt.xlabel("Month")
        plt.ylabel("Average Revenue (Lakhs INR)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"plots/{name.lower().replace(' ', '_')}_month_seasonality.png", dpi=150)
        plt.close()

        print(f"{name} Top 3 Sales Days of Week (by average revenue):")
        top_dows = dow_avg.sort_values(by='y', ascending=False).head(3)
        for _, row in top_dows.iterrows():
            print(f"  {row['DayOfWeek']}: INR {row['y']:,.2f}")

    # 3. Product Pareto (80/20 Rule) Analysis
    print("\n--- 3. Product Pareto Analysis ---")
    for name in ["Account A", "Account B"]:
        df_act = df_trans[df_trans['Account'] == name].copy()
        part_rev = df_act.groupby('Part Code')['Revenue'].sum().reset_index()
        part_rev = part_rev.sort_values(by='Revenue', ascending=False).reset_index(drop=True)
        part_rev['Cumulative_Rev'] = part_rev['Revenue'].cumsum()
        total_rev = part_rev['Revenue'].sum()
        part_rev['Percentage'] = (part_rev['Revenue'] / total_rev) * 100
        part_rev['Cumulative_Percentage'] = (part_rev['Cumulative_Rev'] / total_rev) * 100

        print(f"\n{name} Parts statistics:")
        print(f"  Unique Parts: {len(part_rev)}")
        
        # How many parts make up 80%?
        parts_80 = part_rev[part_rev['Cumulative_Percentage'] <= 80.5]
        num_parts_80 = len(parts_80) + 1 # Include transition part
        pct_parts_80 = (num_parts_80 / len(part_rev)) * 100
        print(f"  Parts driving 80% revenue: {num_parts_80} out of {len(part_rev)} ({pct_parts_80:.2f}%)")

        print(f"  Top 5 parts by revenue:")
        for idx, row in part_rev.head(5).iterrows():
            print(f"    {row['Part Code']}: INR {row['Revenue']:,.2f} ({row['Percentage']:.2f}%)")

        # Visualize Pareto Chart
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()
        
        top_25_parts = part_rev.head(25)
        ax1.bar(top_25_parts['Part Code'], top_25_parts['Revenue'] / 1e5, color=C_PRIMARY, alpha=0.8)
        ax2.plot(top_25_parts['Part Code'], top_25_parts['Cumulative_Percentage'], color=C_ACCENT, marker="o", linewidth=2)
        
        ax1.set_xlabel('Part Code')
        ax1.set_ylabel('Revenue (Lakhs INR)', color=C_PRIMARY)
        ax2.set_ylabel('Cumulative Percentage (%)', color=C_ACCENT)
        ax1.tick_params(axis='x', rotation=90)
        
        plt.title(f"{name} Revenue Pareto Chart (Top 25 Parts)", fontweight='bold', color=C_DARK)
        plt.tight_layout()
        plt.savefig(f"plots/{name.lower().replace(' ', '_')}_pareto.png", dpi=150)
        plt.close()

    # 4. Outliers & Significant Dips/Spikes
    print("\n--- 4. Historical Anomaly & Event Detection ---")
    for name, df in [("Account A", df_a), ("Account B", df_b)]:
        # Define anomalies using rolling statistics
        # We can calculate daily Z-score based on 30-day rolling window
        df['y_roll_mean'] = df['y'].rolling(window=30, min_periods=7, center=True).mean()
        df['y_roll_std'] = df['y'].rolling(window=30, min_periods=7, center=True).std()
        df['z_score'] = (df['y'] - df['y_roll_mean']) / (df['y_roll_std'] + 1e-5)

        spikes = df[df['z_score'] > 3.0]
        dips = df[(df['z_score'] < -2.5) & (df['y'] < df['y_roll_mean'] * 0.3)] # severe dip

        print(f"\n{name} Anomalies Detected:")
        print(f"  Significant Spikes count: {len(spikes)}")
        print(f"  Significant Dips count: {len(dips)}")

        if len(dips) > 0:
            print("  Top 3 Dips:")
            for idx, row in dips.sort_values(by='z_score').head(3).iterrows():
                print(f"    Date: {row['ds'].date()}, Revenue: INR {row['y']:,.2f}, Z-score: {row['z_score']:.2f}")
        if len(spikes) > 0:
            print("  Top 3 Spikes:")
            for idx, row in spikes.sort_values(by='z_score', ascending=False).head(3).iterrows():
                print(f"    Date: {row['ds'].date()}, Revenue: INR {row['y']:,.2f}, Z-score: {row['z_score']:.2f}")

        # Let's plot anomalies on the trend chart
        plt.figure(figsize=(14, 6))
        plt.plot(df['ds'], df['y'] / 1e5, color=C_SECONDARY, alpha=0.6, label='Daily Revenue')
        plt.plot(df['ds'], df['y_roll_mean'] / 1e5, color=C_PRIMARY, linestyle='--', alpha=0.8, label='Rolling Mean')
        
        if len(spikes) > 0:
            plt.scatter(spikes['ds'], spikes['y'] / 1e5, color='red', marker='^', s=80, label='Revenue Spike')
        if len(dips) > 0:
            plt.scatter(dips['ds'], dips['y'] / 1e5, color='blue', marker='v', s=80, label='Revenue Dip')
            
        plt.title(f"{name} Daily Revenue & Historical Anomalies", fontweight='bold', color=C_DARK)
        plt.xlabel("Invoice Date")
        plt.ylabel("Revenue (Lakhs INR)")
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(f"plots/{name.lower().replace(' ', '_')}_anomalies.png", dpi=150)
        plt.close()

    print("\nEDA Completed successfully! Charts generated in 'plots/' directory.")

if __name__ == '__main__':
    main()
