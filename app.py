import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
import holidays
import os

# --- Page Config ---
st.set_page_config(
    page_title="PlantNxt Decision Intelligence Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling for Premium Aesthetics ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header Gradient Banner */
    .banner {
        background: linear-gradient(135deg, #3A6B35 0%, #1A331E 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(58, 107, 53, 0.25);
    }
    .banner h1 {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: white !important;
    }
    .banner p {
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    /* Custom Card Style */
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
        border-left: 5px solid #3A6B35;
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
    }
    .metric-title {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1A331E;
        margin-top: 0.3rem;
    }
    
    /* Alert styling */
    .custom-warning {
        background-color: #FFF9E6;
        border-left: 5px solid #E3B448;
        padding: 1rem;
        border-radius: 8px;
        color: #7A5B00;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }
    
</style>
""", unsafe_allow_html=True)

# Custom color palette constants
C_PRIMARY = "#3A6B35"
C_SECONDARY = "#CBD18F"
C_ACCENT = "#E3B448"
C_DARK = "#1A1A1D"

# --- Data Caching & Loading ---
@st.cache_data
def load_cleaned_data():
    df_trans = pd.read_csv('cleaned_transactions.csv.gz', compression='gzip')
    df_trans['Inv Date'] = pd.to_datetime(df_trans['Inv Date'])
    
    df_a = pd.read_csv('daily_account_a.csv')
    df_a['ds'] = pd.to_datetime(df_a['ds'])
    
    df_b = pd.read_csv('daily_account_b.csv')
    df_b['ds'] = pd.to_datetime(df_b['ds'])
    
    return df_trans, df_a, df_b

try:
    df_trans, df_a, df_b = load_cleaned_data()
except Exception as e:
    st.error(f"Error loading data! Please make sure data_prep.py has run successfully. Details: {e}")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.image("https://img.icons8.com/color/96/sprout.png", width=70)
st.sidebar.markdown("<h2 style='color:#3A6B35; font-weight:800; margin-top:0;'>PlantNxt Platform</h2>", unsafe_allow_html=True)
st.sidebar.markdown("### 🎛️ Dashboard Controls")

account_selection = st.sidebar.selectbox(
    "Select Account Segment:",
    options=["Account A (High Volume)", "Account B (High Growth)"],
    index=0
)

is_a = "Account A" in account_selection
df_daily = df_a if is_a else df_b
acc_name = "Account A" if is_a else "Account B"

# Sidebar Quick Metrics
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Segment Overview:**")
st.sidebar.markdown(f"📅 Start Date: `{df_daily['ds'].min().date()}`")
st.sidebar.markdown(f"📅 End Date: `{df_daily['ds'].max().date()}`")
st.sidebar.markdown(f"📊 Historical Days: `{len(df_daily)}`")

# Navigation Menu
nav_selection = st.sidebar.radio(
    "Select Analytics Workspace:",
    options=[
        "🎯 Task 1: Executive Summary & EDA",
        "🔮 Tasks 2 & 3: Sales Revenue Forecasting",
        "💡 Task 4: Business Decision Intelligence",
        "🚨 Task 5: Anomaly Counterfactual Analyzer"
    ]
)

# --- Header Banner ---
st.markdown(f"""
<div class="banner">
    <h1>PlantNxt Decision Intelligence Platform</h1>
    <p>Incubated by IIT Madras • Daily Revenue Forecasting & Operational Optimization for <b>{acc_name}</b></p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. EXECUTIVE SUMMARY & EDA WORKSPACE
# ==============================================================================
if "Summary" in nav_selection:
    st.markdown("## 🎯 Executive Summary & Data Diagnostics")
    
    # 4 metrics cards
    total_rev = df_daily['y'].sum()
    avg_rev = df_daily['y'].mean()
    max_rev = df_daily['y'].max()
    unique_parts = df_trans[df_trans['Account'] == acc_name]['Part Code'].nunique()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Consolidated Revenue</div>
            <div class="metric-value">₹ {total_rev/1e7:.2f} Cr</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Daily Mean Revenue</div>
            <div class="metric-value">₹ {avg_rev/1e5:.2f} L</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Peak Daily Revenue</div>
            <div class="metric-value">₹ {max_rev/1e5:.2f} L</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Unique Part Catalog</div>
            <div class="metric-value">{unique_parts} Parts</div>
        </div>
        """, unsafe_allow_html=True)

    # Tabs for different EDA visuals
    tab1, tab2, tab3 = st.tabs(["📈 Sales Macro Trend", "🗓️ Demand Seasonality", "📊 Product Pareto (80/20)"])
    
    with tab1:
        st.markdown("### Interactive Daily Revenue & Moving Average")
        roll_window = st.slider("Adjust Moving Average Smoothing (Days):", min_value=7, max_value=90, value=30, step=7)
        
        df_plot = df_daily.copy()
        df_plot['roll_ma'] = df_plot['y'].rolling(window=roll_window, min_periods=1).mean()
        
        fig, ax = plt.subplots(figsize=(14, 5.5))
        ax.plot(df_plot['ds'], df_plot['y'] / 1e5, color=C_SECONDARY, alpha=0.5, label='Daily Revenue (Lakhs INR)')
        ax.plot(df_plot['ds'], df_plot['roll_ma'] / 1e5, color=C_PRIMARY, linewidth=2.5, label=f'{roll_window}-Day Moving Average')
        ax.set_title(f"{acc_name} Invoicing Timeline with {roll_window}-Day Moving Average", fontweight='bold', color=C_DARK)
        ax.set_ylabel("Revenue (Lakhs INR)")
        ax.legend()
        st.pyplot(fig)
        plt.close()
        
    with tab2:
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("#### Day-of-Week Seasonality Profile")
            df_daily['DayOfWeek'] = df_daily['ds'].dt.day_name()
            df_daily['DOW_Num'] = df_daily['ds'].dt.dayofweek
            dow_avg = df_daily.groupby(['DOW_Num', 'DayOfWeek'])['y'].mean().reset_index()
            
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=dow_avg, x='DayOfWeek', y=dow_avg['y']/1e5, palette='viridis', hue='DayOfWeek', legend=False, ax=ax)
            ax.set_ylabel("Average Daily Revenue (Lakhs INR)")
            ax.set_xlabel("Day of Week")
            plt.xticks(rotation=30)
            st.pyplot(fig)
            plt.close()
            
        with col_s2:
            st.markdown("#### Weekly Seasonal & Trend Decomposition")
            df_s = df_daily.set_index('ds')['y']
            result = seasonal_decompose(df_s, model='additive', period=7)
            
            fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
            result.trend.plot(ax=axes[0], color=C_PRIMARY, title="Isolated Trend")
            result.seasonal.plot(ax=axes[1], color=C_ACCENT, title="Weekly Seasonal Cycle")
            result.resid.plot(ax=axes[2], color='gray', style='.', title="Residual Noise")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
    with tab3:
        st.markdown("### Part-level Revenue Distribution Pareto Curve")
        df_act = df_trans[df_trans['Account'] == acc_name].copy()
        part_rev = df_act.groupby('Part Code')['Revenue'].sum().reset_index()
        part_rev = part_rev.sort_values('Revenue', ascending=False).reset_index(drop=True)
        part_rev['Cum_Rev'] = part_rev['Revenue'].cumsum()
        tot_r = part_rev['Revenue'].sum()
        part_rev['Cum_Pct'] = (part_rev['Cum_Rev'] / tot_r) * 100
        
        pareto_slider = st.slider("Select Cumulative Revenue Percentage Target (%):", min_value=50, max_value=95, value=80, step=5)
        
        parts_target = part_rev[part_rev['Cum_Pct'] <= pareto_slider + 0.5]
        num_parts = len(parts_target) + 1
        pct_parts = (num_parts / len(part_rev)) * 100
        
        st.info(f"👉 **Pareto Result:** Exactly **{num_parts} out of {len(part_rev)} parts ({pct_parts:.2f}%)** generate **{pareto_slider}%** of the total {acc_name} sales revenue.")
        
        # Display top 5 parts
        st.markdown("#### Top 5 Revenue-Driving Parts:")
        cols = st.columns(5)
        for i, idx_row in enumerate(part_rev.head(5).iterrows()):
            row = idx_row[1]
            with cols[i]:
                st.metric(label=row['Part Code'], value=f"₹ {row['Revenue']/1e7:.2f} Cr", delta=f"{(row['Revenue']/tot_r)*100:.2f}% Share")

# ==============================================================================
# 2. SALES REVENUE FORECASTING WORKSPACE
# ==============================================================================
elif "Forecasting" in nav_selection:
    st.markdown("## 🔮 Time-Series Forecasting & Model Comparison")
    
    # Validation metrics table
    st.markdown("### Chronological Model Validation Comparison (Last 30 Days Test Set)")
    
    metrics_a = {
        'Model': ['Prophet Baseline', 'XGBoost (Engineered Features)'],
        'RMSE (INR)': ['3,591,889.13', '2,368,190.21'],
        'MAE (INR)': ['3,169,370.61', '1,894,254.58'],
        'MAPE (%)': ['106.37%', '63.93%'],
        'Accuracy Choice': ['Standard', '🏆 Selected Best']
    }
    metrics_b = {
        'Model': ['Prophet Baseline', 'XGBoost (Engineered Features)'],
        'RMSE (INR)': ['7,079,730.37', '6,933,664.49'],
        'MAE (INR)': ['5,362,178.18', '5,096,743.67'],
        'MAPE (%)': ['29.63%', '27.34%'],
        'Accuracy Choice': ['Standard', '🏆 Selected Best']
    }
    
    st.table(pd.DataFrame(metrics_a if is_a else metrics_b))
    
    # 2 columns for visuals
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        st.markdown("#### Actual vs. Predicted values on Test Period")
        img_val = f"plots/{acc_name.lower().replace(' ', '_')}_val_comparison.png"
        if os.path.exists(img_val):
            st.image(img_val, use_column_width=True)
        else:
            st.warning("Validation comparison chart not generated yet. Running run_forecasting.py will generate it.")
            
    with col_f2:
        st.markdown("#### 30-Day Forward Revenue Forecast")
        img_fwd = f"plots/{acc_name.lower().replace(' ', '_')}_forward_forecast.png"
        if os.path.exists(img_fwd):
            st.image(img_fwd, use_column_width=True)
        else:
            st.warning("Forward forecast chart not generated yet. Running run_forecasting.py will generate it.")

    # Show downloadable forecasted numbers
    st.markdown("---")
    st.markdown("### 📥 Download Projected Daily Revenue Forecasts (Lakhs INR)")
    csv_path = f"plots/{acc_name.lower().replace(' ', '_')}_30day_forecast.csv"
    if os.path.exists(csv_path):
        df_fc = pd.read_csv(csv_path)
        df_fc['yhat_Lakhs'] = df_fc['yhat'] / 1e5
        df_fc['yhat_lower_Lakhs'] = df_fc['yhat_lower'] / 1e5
        df_fc['yhat_upper_Lakhs'] = df_fc['yhat_upper'] / 1e5
        
        st.dataframe(df_fc.rename(columns={
            'ds': 'Date', 'yhat': 'Revenue (INR)', 
            'yhat_lower': 'Lower Bound (INR)', 'yhat_upper': 'Upper Bound (INR)'
        }).head(10))
        
        st.download_button(
            label="Download Complete 30-Day Forecast Table (CSV)",
            data=df_fc.to_csv(index=False),
            file_name=f"{acc_name.lower().replace(' ', '_')}_30day_forecast.csv",
            mime="text/csv"
        )
    else:
        st.info("No forecast CSV found yet. Run run_forecasting.py to pre-generate this.")

# ==============================================================================
# 3. BUSINESS DECISION INTELLIGENCE WORKSPACE
# ==============================================================================
elif "Decision" in nav_selection:
    st.markdown("## 💡 Production Planning & Demand Sensitivity shocks")
    
    col_d1, col_d2 = st.columns([1, 1.2])
    
    with col_d1:
        st.markdown("### 🗓️ Upcoming Revenue Risk Weeks (Next 30 Days)")
        
        # Static weekly breakdown extracted from raw predictions
        if is_a:
            weekly_data = {
                'Projected Week': ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
                'Revenue (Lakhs INR)': [492.29, 516.29, 530.53, 548.40],
                'Risk Profile': ['⚠️ Highest Risk (Cooling)', 'Moderate', 'Moderate', 'Peak Demand']
            }
        else:
            weekly_data = {
                'Projected Week': ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
                'Revenue (Lakhs INR)': [988.04, 941.92, 924.64, 928.03],
                'Risk Profile': ['Peak Demand', 'Moderate', '⚠️ Highest Risk (Dip)', 'Moderate']
            }
            
        df_week = pd.DataFrame(weekly_data)
        st.dataframe(df_week, hide_index=True)
        
        st.markdown(f"""
        <div class="custom-warning">
            <b>Production Planning Alert:</b><br>
            { "Account A Week 1 represents an immediate cooling off period. Safety stock buffers should be deployed." if is_a else "Account B Week 3 experiences a predicted 6.5% drop in invoicing volume. Planners should re-allocate tooling lines to Account A mid-week." }
        </div>
        """, unsafe_allow_html=True)

    with col_d2:
        st.markdown("### 🎚️ Demand Underperformance Shock Simulator")
        st.write("Manufacturing companies are highly sensitive to demand reductions. Use the slider to model custom demand drops.")
        
        shock_pct = st.slider("Simulate Customer Demand Underperformance (%):", min_value=5, max_value=50, value=15, step=5)
        
        total_historical = df_daily['y'].sum()
        revenue_impact = total_historical * (shock_pct / 100)
        
        st.metric(
            label=f"Financial Revenue Impact (Loss at -{shock_pct}%)",
            value=f"₹ {revenue_impact/1e7:,.2f} Cr",
            delta=f"-{shock_pct}% Demand reduction"
        )
        
        # Concentration Details
        if is_a:
            st.markdown(f"""
            * **Account A is highly vulnerable to shocks**:
              * Just **6 customers** generate 80% of all revenue.
              * A {shock_pct}% demand shock is equivalent to completely losing their top customer **Cust-00274 (which contributes 26.39% of all revenue)**.
              * Safety stock policies for top customer part catalogs must be strictly enforced.
            """)
        else:
            st.markdown(f"""
            * **Account B is highly diversified and resilient**:
              * **151 customers** generate 80% of all revenue.
              * The top customer contributes only **3.25%** (`Cust-00845`).
              * A {shock_pct}% shock is highly spread across many smaller accounts, representing low customer-specific concentration risk.
            """)
            
    st.markdown("---")
    st.markdown("### 🛠️ Strategic Production Action Recommendations")
    
    col_r1, col_r2, col_r3 = st.columns(3)
    
    with col_r1:
        st.markdown("#### 🔄 Assembly Setup Tooling")
        st.write("Account B's invoice demand is heavily front-loaded on Mondays/Tuesdays, while Account A peaks Wednesdays-Fridays.")
        st.info("💡 **Staffing Plan:** Front-load shop floor capacity for Account B assemblies early-week, and configure tooling setups for Account A parts starting Wednesday morning.")
        
    with col_r2:
        st.markdown("#### 📦 Inventory Safety Stock")
        st.write("Due to extreme part concentration (6.08% parts driving 80% revenue for Account A), stockouts represent a severe bottom-line threat.")
        st.info("💡 **Stock Policy:** Maintain a robust 14-day safety buffer for top parts like Part-01196 and Part-00149 on-site to handle sudden spikes.")
        
    with col_r3:
        st.markdown("#### 🔌 Supply Chain Material Sync")
        st.write("Combine 30-day ahead projections with steel and forward packaging supplier lead times.")
        st.info("💡 **Ordering Plan:** Enforce automatic Purchase Order (PO) triggers aligned exactly with the daily forecast curves to avoid carrying excessive inventory.")

# ==============================================================================
# 4. ANOMALY COUNTERFACTUAL ANALYZER WORKSPACE
# ==============================================================================
elif "Anomaly" in nav_selection:
    st.markdown("## 🚨 Root Cause Analysis & Counterfactual What-If Simulation")
    
    st.markdown("### The Selected Event: May 1, 2024 (May Day Labor Holiday Shutdown)")
    st.write("In our EDA and anomaly detection step, we isolated a massive, recurring revenue dip in Account A on May 1st (International Workers' Day). Let's evaluate the operational diagnostics and run a counterfactual what-if scenario.")
    
    col_a1, col_a2 = st.columns([1.2, 1])
    
    with col_a1:
        img_anom = "plots/account_a_may_day_anomaly.png"
        if os.path.exists(img_anom):
            st.image(img_anom, use_column_width=True)
        else:
            st.warning("May Day anomaly diagnostic plot not found. Run run_root_cause.py to pre-generate this.")
            
    with col_a2:
        st.markdown("#### 📊 Operational Anomaly Evidence")
        
        # Transaction comparison metrics
        st.markdown("""
        Comparing **May 1st, 2024 (May Day)** against an adjacent **Normal Wednesday (May 8th, 2024)**:
        """)
        
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.metric(label="Anomaly Day (May 1)", value="34 Invoices", delta="Shutdown", delta_color="inverse")
            st.metric(label="Anomaly Day Volume", value="3,743 Units", delta="-63.3%")
        with stat_col2:
            st.metric(label="Normal Day (May 8)", value="149 Invoices", delta="Healthy Operations")
            st.metric(label="Normal Day Volume", value="10,201 Units", delta="")
            
        st.markdown("---")
        st.markdown("#### 🎚️ Counterfactual What-If Labor Capacity Simulator")
        st.write("If the factory operated at a partial labor capacity rather than a complete holiday shutdown, what would the revenue have been?")
        
        labor_capacity = st.slider("Select Simulated Labor Capacity (%):", min_value=0, max_value=100, value=100, step=10)
        
        actual_rev = 1225560.89
        model_counterfactual = 7715269.65
        
        simulated_revenue = actual_rev + (model_counterfactual - actual_rev) * (labor_capacity / 100)
        potential_recovery = simulated_revenue - actual_rev
        
        st.metric(
            label=f"Simulated Revenue at {labor_capacity}% Capacity",
            value=f"₹ {simulated_revenue/1e5:.2f} Lakhs",
            delta=f"+ ₹ {potential_recovery/1e5:.2f} L Recovered"
        )
        
        st.write(f"At **{labor_capacity}% simulated capacity**, the factory could have recovered **INR {potential_recovery:,.2f}** in sales revenue, demonstrating the massive cost of a full operational shutdown.")

    st.markdown("---")
    st.markdown("### 🛡️ How PlantNxt Engine Can Preemptively Flag Holiday Shutdowns")
    
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        st.markdown("#### 📅 1. Industrial Calendar Overlay")
        st.write("By integrating local labor union agreements and regional holiday databases (specifically May Day / Workers' Day in Tamil Nadu) natively into PlantNxt ERP data connectors, the forecasting engine automatically expects labor capacity drops.")
        
    with col_e2:
        st.markdown("#### 🚨 2. Order Rate Deceleration Alerts")
        st.write("Automated anomaly models detect invoicing drops starting 48 hours prior. Decision alerts trigger planners 14 days in advance to schedule extra production shifts in mid-April, building enough safety stock prior to the holiday shutdown.")

# Footer info
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>PlantNxt Data Scientist Assignment © May 2026. Built by Antigravity AI pair-programmed with Sunil.</p>", unsafe_allow_html=True)
