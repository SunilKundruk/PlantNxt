import pandas as pd

def main():
    print("--- Product Mix (Top Parts Share) Changes Over Time ---")
    df = pd.read_csv('cleaned_transactions.csv.gz', compression='gzip')
    df['Inv Date'] = pd.to_datetime(df['Inv Date'])
    df['Year'] = df['Inv Date'].dt.year

    for name in ['Account A', 'Account B']:
        df_act = df[(df['Account'] == name) & (df['Year'].isin([2023, 2024, 2025]))].copy()
        if len(df_act) == 0:
            # Maybe Account B only has data for 2025/2026? Let's check 2025 and 2026
            df_act = df[(df['Account'] == name) & (df['Year'].isin([2025, 2026]))].copy()
            
        yearly_totals = df_act.groupby('Year')['Revenue'].sum().reset_index().rename(columns={'Revenue': 'Total_Revenue'})
        part_yearly = df_act.groupby(['Year', 'Part Code'])['Revenue'].sum().reset_index()
        part_yearly = part_yearly.merge(yearly_totals, on='Year')
        part_yearly['Share_Pct'] = (part_yearly['Revenue'] / part_yearly['Total_Revenue']) * 100
        
        top_parts = df_act.groupby('Part Code')['Revenue'].sum().sort_values(ascending=False).head(3).index.tolist()
        
        print(f"\n{name} Top Part Shares by Year:")
        for part in top_parts:
            part_data = part_yearly[part_yearly['Part Code'] == part].sort_values('Year')
            print(f"  {part}:")
            for idx, row in part_data.iterrows():
                print(f"    Year {int(row['Year'])}: INR {row['Revenue']:,.2f} ({row['Share_Pct']:.2f}%)")

if __name__ == '__main__':
    main()
