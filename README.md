# 🍞 CrustIQ — Bakery Inventory & Operations Intelligence Platform

> An end-to-end data engineering and machine learning project that helps
> bakeries make smarter decisions about inventory, staffing, and demand
> forecasting — reducing waste, cutting labor costs, and maximizing profit.

---

## View the Dashboard here - 

https://huggingface.co/spaces/sritikaa/crustiq-bakery

---

## 🧠 Problem Statement

Bakeries face a unique and complex operational challenge:

- **Perishable inventory** — products expire in hours, not days
- **Highly variable demand** — weekends spike 40%, holidays spike up to 180%
- **Labor cost control** — overstaffing on slow days wastes thousands per year
- **No data-driven decisions** — most bakeries rely on gut feeling

**CrustIQ solves this** by combining data engineering, machine learning, and
optimization to give bakery managers a single intelligent platform for all
operational decisions.

---

## 🏗️ Architecture
```
Raw Data Generation
        ↓
    Python (Faker + Pandas)
        ↓
Snowflake Data Warehouse
┌─────────────────────────────┐
│  BRONZE  ←  Raw tables      │
│     ↓                       │
│  SILVER  ←  Enriched views  │
│     ↓                       │
│  GOLD    ←  ML-ready tables │
└─────────────────────────────┘
        ↓
ML Forecasting (XGBoost · LightGBM · Random Forest · Linear Regression)
        ↓
Optimization Engines (Inventory · Staff Scheduling)
        ↓
Streamlit Dashboard
```

---

## 🚀 Key Features

### 📊 Business Intelligence Dashboard
- 18 months of sales analytics
- Revenue by product, category, day of week, hour
- Holiday demand spike visualization
- Interactive Plotly charts

### 🔮 Demand Forecasting (ML)
- 4 models compared — XGBoost, LightGBM, Random Forest, Linear Regression
- 15 engineered features including lag features, rolling averages, holiday signals
- Time-based train/test split (last 60 days as test set)
- Best model: Linear Regression (R²: 0.63, RMSE: 5.58 units)

### 📦 Inventory Optimization Engine
- Daily baking recommendations per product
- Safety stock calculation based on shelf life
- Freshness risk flagging (🔴 HIGH / 🟡 MEDIUM / 🟢 LOW)
- Waste cost and revenue projections

### 👨‍🍳 Staff Scheduling Optimizer
- Demand-linked staff recommendations per role
- Weekend and holiday multipliers
- 7-day labor cost forecast
- Estimated $5,000+ annual savings vs manual scheduling

### ❄️ Snowflake Medallion Architecture
- **Bronze** — 5 raw tables (45,664 sales transactions)
- **Silver** — 3 enriched views with business logic
- **Gold** — 5 analytical tables ready for ML and dashboarding

---

## 📁 Project Structure
```
bakery-analytics/
│
├── data/
│   ├── raw/              ← Generated CSVs (products, sales, staff, etc.)
│   ├── processed/        ← Cleaned & feature-engineered data
│   └── exports/          ← Optimization outputs (inventory plan, schedule)
│
├── notebooks/
│   └── 01_eda.ipynb      ← Exploratory Data Analysis
│
├── src/
│   ├── ingestion/
│   │   ├── generate_data.py      ← Synthetic data generation
│   │   ├── transform_data.py     ← Cleaning & feature engineering
│   │   └── load_to_snowflake.py  ← Snowflake data loader
│   ├── models/
│   │   ├── demand_forecast.py    ← XGBoost demand forecasting
│   │   ├── model_comparison.py   ← Multi-model comparative analysis
│   │   └── artifacts/            ← Saved models & encoders
│   ├── optimization/
│   │   ├── inventory_optimizer.py ← Baking recommendations
│   │   └── staff_optimizer.py     ← Staff scheduling engine
│   └── utils/
│       └── snowflake_conn.py      ← Snowflake connection utility
│
├── sql/
│   ├── bronze/setup.sql           ← Raw table DDL
│   ├── silver/silver_transforms.sql ← Enriched views
│   └── gold/gold_aggregates.sql   ← Analytical tables
│
├── dashboard/
│   └── app.py            ← Streamlit dashboard
│
├── .env                  ← Snowflake credentials (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data Warehouse | Snowflake |
| Data Processing | Pandas, NumPy |
| Machine Learning | XGBoost, LightGBM, Scikit-learn |
| Visualization | Plotly, Streamlit |
| Data Generation | Faker |
| Environment | Virtual Environment (venv) |
| IDE | VS Code |

---

## ⚙️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/bakery-analytics.git
cd bakery-analytics
```

**2. Create virtual environment**
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Set up environment variables**
```bash
cp .env.example .env
# Fill in your Snowflake credentials
```

**4. Generate data**
```bash
python src/ingestion/generate_data.py
python src/ingestion/transform_data.py
```

**5. Load to Snowflake**
```bash
python src/ingestion/load_to_snowflake.py
```

**6. Train ML models**
```bash
python src/models/model_comparison.py
```

**7. Run optimization engines**
```bash
python src/optimization/inventory_optimizer.py
python src/optimization/staff_optimizer.py
```

**8. Launch dashboard**
```bash
streamlit run dashboard/app.py
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Total Sales Transactions | 45,664 |
| Data Range | Jan 2024 — Jun 2025 |
| Best ML Model | Linear Regression |
| Model R² Score | 0.63 |
| Model RMSE | 5.58 units |
| Estimated Annual Labor Savings | $5,009 |
| Holiday Demand Spike (Mother's Day) | +180% |
| Weekend Demand Spike | +40% |

---

## 💡 Key Business Insights

- **Saturday is the highest revenue day** — 40% above weekday average
- **9AM–11AM is peak demand window** — staff accordingly
- **Custom Cake has highest margin** — prioritize during holiday seasons
- **Short shelf-life products** (croissants, danishes) carry highest waste risk
- **Overstaffing on weekdays** costs an average of $13.73/day — $5,009/year

---

## 🔮 Future Improvements

- [ ] Connect live POS data via Snowflake Streams
- [ ] Add weather API signals to forecasting features
- [ ] Build Prophet model for long-range seasonal forecasting
- [ ] Implement automated retraining pipeline
- [ ] Add customer segmentation analysis
- [ ] Deploy dashboard to Streamlit Cloud

---

## 👩‍💻 Author

**Ritika Sisodiya**
- LinkedIn: https://www.linkedin.com/in/ritikaa-s
- GitHub: ritika-s05

---

> *Built with Python · Snowflake · XGBoost · Streamlit*

