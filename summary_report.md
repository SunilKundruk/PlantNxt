# PlantNxt Executive Summary: Revenue Forecasting & Decision Intelligence

**Prepared for**: Production Planning & Decision Intelligence Stakeholders  
**Focus**: Sales Revenue Forecasting & Operational Optimization  
**Data Scientist**: Antigravity AI  

---

## 1. Executive Summary
PlantNxt is a decision intelligence platform Incubated at IIT Madras that translates manufacturing data into financially aligned, prescriptive decisions. This report details the analysis of sales invoice transaction data from two distinct accounts (**Account A** and **Account B**) containing **234,516 rows** spanning **2022-04-01 to 2026-05-16**.

By developing a robust, automated data-preparation and machine learning pipeline, we successfully cleaned raw transactional issues, designed time-series forecasting engines using **Prophet** and **XGBoost Regressors**, identified critical operational seasonality and customer concentrations, and simulated the counterfactual impact of historical factory shutdowns.

---

## 2. Key Findings & Data Understanding (Task 1)

### Segment Profiling
* **Account A** (222,078 transactions, ~4.1 years of continuous daily data) represents a mature, high-volume segment with a total revenue of **INR 1,088.9 Crore** (Daily Average: INR 72.3 Lakhs).
* **Account B** (12,438 transactions, ~1.1 years of daily data) represents a rapidly growing, high-value segment with a total revenue of **INR 567.0 Crore** (Daily Average: INR 1.38 Crore).

### Day-of-Week Seasonality
We discovered highly distinct cyclical demand patterns between the two accounts:
* **Account A peaks late-week**: Wednesday (INR 81.9 Lakhs), Thursday (INR 80.8 Lakhs), and Friday (INR 83.8 Lakhs) drive the highest invoicing activity.
* **Account B peaks early-week**: Monday (INR 1.96 Crore) and Tuesday (INR 1.74 Crore) are heavily front-loaded.

### Product concentration (Pareto 80/20 Rule)
Both accounts exhibit extreme product-line concentration, far exceeding the standard 80/20 rule:
* **Account A**: Just **127 out of 2,088 unique parts (6.08%)** drive **80% of all revenue**. Top product is `Part-01196` (4.22% of revenue, INR 45.9 Crore).
* **Account B**: Just **64 out of 431 unique parts (14.85%)** drive **80% of all revenue**. Top product is `Part-00149` (5.66% of revenue, INR 32.1 Crore).

---

## 3. Forecasting Models & Validation Results (Tasks 2 & 3)
We built a chronological time-series split using the final 30 days of historical data as our test set (zero data-leakage validation). We compared a generalized additive model (**Prophet**) against an engineered machine learning model (**XGBoost Regressor**) built on lag variables ($t-1$, $t-7$, $t-30$), rolling averages, rolling standard deviations, calendar indicators, and Indian national holiday calendars.

| Account | Model | RMSE (INR) | MAE (INR) | MAPE (%) | Selected Best Model |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Account A** | Prophet | 3,591,889.13 | 3,169,370.61 | 106.37% | **XGBoost** |
| | **XGBoost** | **2,368,190.21** | **1,894,254.58** | **63.93%** | *(Selected)* |
| **Account B** | Prophet | 7,079,730.37 | 5,362,178.18 | 29.63% | **XGBoost** |
| | **XGBoost** | **6,933,664.49** | **5,096,743.67** | **27.34%** | *(Selected)* |

### Technical Rationale
XGBoost significantly outperformed Prophet on daily forecasts. Because daily manufacturing invoicing exhibits high volatility and rapid non-linear shifts, XGBoost leverages **rolling standard deviation (volatility)** and **short-term lags (momentum)** to capture high-frequency patterns that smooth curve-fitting models (like Prophet) tend to wash out. Prophet is maintained for long-range forward projections due to its robust native confidence bands.

---

## 4. Business Interpretation & Sensitivity Shock (Task 4)

### Q1: Highest Revenue Risk Weeks (Next 30 Days)
* **Account A**: Week 1 of the projection (INR 4.92 Crores) represents the lowest active full week, indicating an immediate cooling period.
* **Account B**: Week 3 represents the highest risk period (INR 9.25 Crores), reflecting a ~6.5% dip compared to Week 1.

### Q2: 15% Underperformance Exposure & Customer Concentration
* **Account A is extremely vulnerable**: It has only 136 customers. Just **6 customers (4.41%)** drive **80% of total revenue**. The top customer `Cust-00274` alone represents **26.39% (INR 287.4 Crore)**. A 15% demand shock (INR 163.3 Crore) is equivalent to completely losing their top customer. 
* **Account B is highly diversified**: It has 712 customers. **151 customers (21.21%)** drive 80% of revenue. The top customer contributes only **3.25%**. A 15% drop (INR 85.0 Crore) is equivalent to losing their top 6 customers.

### Q3: Actionable Recommendations for the Production Planning Team
1. **Differentiated Scheduling Shifts**: Staff and set up machinery for Account B's complex assemblies early in the week (Monday/Tuesday). Transition line configurations to Account A's high-volume part setups starting mid-week to align exactly with historical delivery timing.
2. **Safety Stock Policy**: Given that just 6 customers generate 80% of Account A's revenue, enforce a high safety stock buffer (e.g., 14 days of average demand) for their top-selling parts (`Part-01196`, `Part-01626`) to avoid stockouts.
3. **Contractual Procurement Locking**: Integrate the 30-day forecast directly with steel/parts suppliers to trigger automatic material delivery based on predicted sales spikes, lowering inventory carrying costs.

### Q4: High-Value Additional Data Streams
* **Sales Opportunity Pipeline**: Pending contract quotes and customer-confirmed Purchase Orders (POs) would replace speculative statistical forecasts with locked-in delivery schedules.
* **MES/IoT Machine Health logs**: Real-time sensor logs from the factory floor would let us match upcoming demand spikes against active machine capacity and schedule machine maintenance during predicted low-risk weeks (e.g., Week 3 of Account B).

---

## 5. Anomaly Root-Cause & Counterfactual "What-If" Analysis (Task 5)

### The Selected Anomaly
We isolated a severe revenue dip in Account A on **May 1st, 2024 (International Workers' Day / May Day)**. Revenue dropped to just **INR 12.26 Lakhs** (compared to a typical adjacent Wednesday on May 8th, which brought in **INR 1.07 Crores**).

### Transactional Evidence
* **May 1st, 2024 (Anomaly Day)**: Only **34 transactions** and **3,743 units** were processed.
* **May 8th, 2024 (Normal Day)**: **149 transactions** and **10,201 units** were processed.
This stark decline in transaction and volume count indicates a complete operational and factory floor shutdown for the May Day holiday.

### Counterfactual What-If Simulation
We trained our time-series forecasting model *excluding* the May 1st date to predict what the revenue *would have been* under normal operating conditions.
* **Actual Revenue**: INR 12,25,560.89 (12.26 Lakhs)
* **What-If Counterfactual Revenue**: **INR 77,15,269.65** (77.15 Lakhs)
* **Estimated Revenue Impact**: **INR 64,89,708.76 (64.90 Lakhs lost/deferred)**.

### Proactive Warnings on PlantNxt
PlantNxt's decision intelligence engine can preemptively flag these occurrences by:
1. **Natively incorporating industrial calendar overlays** (local labor union laws, state-level holidays) to automatically depress predicted daily target shipments.
2. **Alerting planners 14 days in advance** to schedule extra manufacturing shifts in mid-April to build up inventory, ensuring monthly shipment targets are met prior to the May 1st holiday.
