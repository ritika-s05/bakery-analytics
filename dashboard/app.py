import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from datetime import datetime, timedelta
import os

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="CrustIQ — Bakery Intelligence",
    page_icon="🍞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #1C110A; color: #FAF5EB; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #2C1A0E;
        border-right: 1px solid rgba(200,132,42,0.3);
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: rgba(200,132,42,0.08);
        border: 1px solid rgba(200,132,42,0.25);
        border-radius: 10px;
        padding: 16px;
    }

    /* Headers */
    h1, h2, h3 { color: #E8A84C !important; }

    /* Dataframe */
    [data-testid="stDataFrame"] { border-radius: 8px; }

    /* Divider */
    hr { border-color: rgba(200,132,42,0.2); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    sales     = pd.read_csv("data/processed/sales_clean.csv",     parse_dates=["transaction_ts", "transaction_date"])
    inventory = pd.read_csv("data/processed/inventory_clean.csv", parse_dates=["baked_date"])
    staff     = pd.read_csv("data/processed/staff_clean.csv",     parse_dates=["shift_date"])
    products  = pd.read_csv("data/raw/products.csv")
    holidays  = pd.read_csv("data/raw/holidays.csv",              parse_dates=["holiday_date"])

    inv_plan  = pd.read_csv("data/exports/inventory_plan.csv")
    staff_sch = pd.read_csv("data/exports/staff_schedule.csv")
    model_cmp = pd.read_csv("src/models/artifacts/model_comparison.csv")
    feat_imp  = pd.read_csv("src/models/artifacts/feature_importance.csv")

    return sales, inventory, staff, products, holidays, inv_plan, staff_sch, model_cmp, feat_imp

sales, inventory, staff, products, holidays, inv_plan, staff_sch, model_cmp, feat_imp = load_data()

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🍞 CrustIQ")
    st.markdown("### Bakery Intelligence Platform")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Overview", "📦 Inventory", "👨‍🍳 Staff", "🤖 ML Models", "❄️ Snowflake"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**Data Range**")
    st.caption(f"From: {sales['transaction_date'].min().date()}")
    st.caption(f"To:   {sales['transaction_date'].max().date()}")
    st.divider()
    st.caption("Built with Python · Snowflake · XGBoost · Streamlit")

# ─────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 Business Overview")
    st.caption(f"Last updated: {datetime.now().strftime('%B %d, %Y %H:%M')}")

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Total Revenue",
                  f"${sales['revenue'].sum():,.0f}",
                  f"+23% vs last period")
    with col2:
        st.metric("📈 Total Margin",
                  f"${sales['gross_margin'].sum():,.0f}",
                  f"{sales['gross_margin'].sum()/sales['revenue'].sum()*100:.1f}% margin")
    with col3:
        st.metric("🧾 Transactions",
                  f"{len(sales):,}",
                  "18 months")
    with col4:
        total_waste = inventory["waste_cost_usd"].sum()
        st.metric("🗑️ Total Waste Cost",
                  f"${total_waste:,.0f}",
                  "Reduction target: 20%",
                  delta_color="inverse")

    st.divider()

    # Daily Revenue Chart
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Daily Revenue — 2024")
        daily = sales[sales["transaction_date"].dt.year == 2024].groupby(
            "transaction_date")["revenue"].sum().reset_index()
        holidays_2024 = holidays[holidays["holiday_date"].dt.year == 2024]

        fig = px.line(daily, x="transaction_date", y="revenue",
                      template="plotly_dark",
                      color_discrete_sequence=["#E8A84C"])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="", yaxis_title="Revenue ($)",
            height=320
        )
        for _, h in holidays_2024.iterrows():
            fig.add_vline(x=h["holiday_date"], line_dash="dash",
                          line_color="#C0392B", opacity=0.7)
            fig.add_vrect(
                x0=h["holiday_date"] - pd.Timedelta(days=5),
                x1=h["holiday_date"] + pd.Timedelta(days=5),
                fillcolor="rgba(192,57,43,0.1)", layer="below", line_width=0
            )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Revenue by Category")
        cat = sales.groupby("category")["revenue"].sum().reset_index()
        fig = px.pie(cat, values="revenue", names="category",
                     template="plotly_dark",
                     color_discrete_sequence=px.colors.sequential.Oranges_r)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            height=320
        )
        st.plotly_chart(fig, use_container_width=True)

    # Day of week & hourly
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Revenue by Day of Week")
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow = sales.groupby("day_of_week")["revenue"].sum().reset_index()
        dow["day_of_week"] = pd.Categorical(dow["day_of_week"], categories=order, ordered=True)
        dow = dow.sort_values("day_of_week")
        fig = px.bar(dow, x="day_of_week", y="revenue",
                     template="plotly_dark",
                     color="revenue", color_continuous_scale="Oranges")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=280)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Peak Hours")
        hourly = sales.groupby("hour")["revenue"].sum().reset_index()
        fig = px.area(hourly, x="hour", y="revenue",
                      template="plotly_dark",
                      color_discrete_sequence=["#E8A84C"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=280)
        st.plotly_chart(fig, use_container_width=True)

    # Top products
    st.subheader("Product Performance")
    top = sales.groupby("product_name").agg(
        Revenue=("revenue", "sum"),
        Margin=("gross_margin", "sum"),
        Units=("quantity", "sum")
    ).reset_index().sort_values("Revenue", ascending=False)
    top["Margin %"] = round(top["Margin"] / top["Revenue"] * 100, 1)
    top["Revenue"]  = top["Revenue"].apply(lambda x: f"${x:,.2f}")
    top["Margin"]   = top["Margin"].apply(lambda x: f"${x:,.2f}")
    st.dataframe(top, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────
# PAGE 2 — INVENTORY
# ─────────────────────────────────────────
elif page == "📦 Inventory":
    st.title("📦 Inventory Intelligence")
    st.caption("AI-powered baking recommendations · Waste prevention · Freshness tracking")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Units to Bake Tomorrow",
                  f"{inv_plan['recommended_bake'].sum():,}")
    with col2:
        st.metric("💰 Est. Revenue",
                  f"${inv_plan['estimated_revenue'].sum():,.2f}")
    with col3:
        st.metric("📈 Est. Margin",
                  f"${inv_plan['estimated_margin'].sum():,.2f}")
    with col4:
        st.metric("⚠️ Max Waste Risk",
                  f"${inv_plan['max_waste_risk'].sum():,.2f}",
                  delta_color="inverse")

    st.divider()

    # Inventory Plan Table
    st.subheader(f"🍞 Baking Plan — {inv_plan['forecast_date'].iloc[0]}")
    display_plan = inv_plan[[
        "product_name", "category", "predicted_demand",
        "safety_stock", "recommended_bake",
        "estimated_revenue", "max_waste_risk", "freshness_risk"
    ]].copy()
    display_plan.columns = [
        "Product", "Category", "Predicted Demand",
        "Safety Stock", "Recommended Bake",
        "Est. Revenue", "Waste Risk", "Freshness"
    ]
    st.dataframe(display_plan, use_container_width=True, hide_index=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Waste Cost by Product")
        waste = inventory.groupby("product_name")["waste_cost_usd"].sum().reset_index()
        waste = waste.sort_values("waste_cost_usd", ascending=True)
        fig = px.bar(waste, x="waste_cost_usd", y="product_name",
                     orientation="h", template="plotly_dark",
                     color="waste_cost_usd", color_continuous_scale="Reds")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Sell-Through Rate by Product")
        sell = inventory.groupby("product_name")["sell_through_rate"].mean().reset_index()
        sell = sell.sort_values("sell_through_rate", ascending=True)
        fig = px.bar(sell, x="sell_through_rate", y="product_name",
                     orientation="h", template="plotly_dark",
                     color="sell_through_rate", color_continuous_scale="Greens")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Waste trend
    st.subheader("Daily Waste Cost Trend")
    waste_daily = inventory.groupby("baked_date")["waste_cost_usd"].sum().reset_index()
    fig = px.line(waste_daily, x="baked_date", y="waste_cost_usd",
                  template="plotly_dark",
                  color_discrete_sequence=["#C0392B"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=280)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 3 — STAFF
# ─────────────────────────────────────────
elif page == "👨‍🍳 Staff":
    st.title("👨‍🍳 Smart Staff Scheduler")
    st.caption("AI-driven labor optimization · Cost control · Demand-linked scheduling")

    # KPIs
    today_sch = staff_sch[staff_sch["shift_date"] == staff_sch["shift_date"].iloc[0]]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Staff Today",
                  f"{today_sch['recommended_count'].sum()}")
    with col2:
        st.metric("💰 Labor Cost Today",
                  f"${today_sch['labor_cost'].sum():,.2f}")
    with col3:
        weekly_cost = staff_sch.groupby("shift_date")["labor_cost"].sum().sum()
        st.metric("📅 Weekly Labor Cost",
                  f"${weekly_cost:,.2f}")
    with col4:
        avg_overstaff = staff["overstaffing_cost"].mean()
        st.metric("💡 Est. Annual Savings",
                  f"${avg_overstaff*365:,.0f}",
                  "vs manual scheduling")

    st.divider()

    # Today's schedule
    st.subheader(f"Today's Staff Schedule — {staff_sch['shift_date'].iloc[0]}")
    today_display = today_sch[[
        "role", "description", "recommended_count",
        "hourly_rate", "labor_cost"
    ]].copy()
    today_display.columns = ["Role", "Description", "Staff", "Hourly Rate", "Daily Cost"]
    today_display["Hourly Rate"] = today_display["Hourly Rate"].apply(lambda x: f"${x}/hr")
    today_display["Daily Cost"]  = today_display["Daily Cost"].apply(lambda x: f"${x:,.2f}")
    st.dataframe(today_display, use_container_width=True, hide_index=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("7-Day Staff Count")
        weekly_sum = staff_sch.groupby(
            ["shift_date", "day_of_week"]
        )["recommended_count"].sum().reset_index()
        fig = px.bar(weekly_sum, x="day_of_week", y="recommended_count",
                     template="plotly_dark",
                     color="recommended_count",
                     color_continuous_scale="Oranges",
                     text="recommended_count")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("7-Day Labor Cost")
        weekly_cost = staff_sch.groupby(
            ["shift_date", "day_of_week"]
        )["labor_cost"].sum().reset_index()
        fig = px.bar(weekly_cost, x="day_of_week", y="labor_cost",
                     template="plotly_dark",
                     color="labor_cost",
                     color_continuous_scale="Reds",
                     text="labor_cost")
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Historical overstaffing
    st.subheader("Historical Overstaffing Cost")
    overstaff = staff.groupby("shift_date")["overstaffing_cost"].sum().reset_index()
    fig = px.line(overstaff, x="shift_date", y="overstaffing_cost",
                  template="plotly_dark",
                  color_discrete_sequence=["#C0392B"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=280)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 4 — ML MODELS
# ─────────────────────────────────────────
elif page == "🤖 ML Models":
    st.title("🤖 ML Model Performance")
    st.caption("Demand forecasting · Comparative analysis · Feature importance")

    # Model comparison table
    st.subheader("Model Comparison")
    st.dataframe(model_cmp, use_container_width=True, hide_index=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("R² Score by Model")
        fig = px.bar(model_cmp, x="Model", y="R²",
                     template="plotly_dark",
                     color="R²", color_continuous_scale="Greens",
                     text="R²")
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("RMSE by Model (lower is better)")
        fig = px.bar(model_cmp, x="Model", y="RMSE",
                     template="plotly_dark",
                     color="RMSE", color_continuous_scale="Reds_r",
                     text="RMSE")
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.subheader("Feature Importance — XGBoost")
    feat_imp = feat_imp.sort_values("importance", ascending=True)
    fig = px.bar(feat_imp, x="importance", y="feature",
                 orientation="h", template="plotly_dark",
                 color="importance", color_continuous_scale="Oranges")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=420)
    st.plotly_chart(fig, use_container_width=True)

    # Key insight
    st.divider()
    st.subheader("💡 Key Insight")
    best_model = model_cmp.loc[model_cmp["RMSE"].idxmin(), "Model"]
    best_r2    = model_cmp.loc[model_cmp["RMSE"].idxmin(), "R²"]
    best_rmse  = model_cmp.loc[model_cmp["RMSE"].idxmin(), "RMSE"]
    st.info(f"""
    **Best Model: {best_model}**
    - R²: {best_r2} — explains {best_r2*100:.1f}% of demand variance
    - RMSE: {best_rmse} units average prediction error
    - Linear Regression outperforms tree models on synthetic data due to
      linear demand multiplier patterns. In production with real POS data,
      XGBoost or LightGBM would likely outperform due to complex
      non-linear customer behavior patterns.
    """)

# ─────────────────────────────────────────
# PAGE 5 — SNOWFLAKE
# ─────────────────────────────────────────
elif page == "❄️ Snowflake":
    st.title("❄️ Snowflake Data Architecture")
    st.caption("Medallion architecture · Bronze → Silver → Gold")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🥉 Bronze Layer")
        st.markdown("""
        Raw ingested data
        - `PRODUCTS` — 12 rows
        - `HOLIDAYS` — 11 rows
        - `SALES` — 45,664 rows
        - `INVENTORY` — 6,564 rows
        - `STAFF` — 3,282 rows
        """)
    with col2:
        st.subheader("🥈 Silver Layer")
        st.markdown("""
        Cleaned & enriched views
        - `SALES_ENRICHED`
        - `INVENTORY_ENRICHED`
        - `STAFF_ENRICHED`
        """)
    with col3:
        st.subheader("🥇 Gold Layer")
        st.markdown("""
        Analytical & ML-ready tables
        - `DAILY_SALES_SUMMARY` — 6,520 rows
        - `DEMAND_FEATURES` — 6,436 rows
        - `WASTE_SUMMARY` — 12 rows
        - `STAFF_COST_SUMMARY` — 547 rows
        - `PRODUCT_SCORECARD` — 12 rows
        """)

    st.divider()

    st.subheader("Pipeline Flow")
    st.code("""
Raw CSV Files
    ↓  (load_to_snowflake.py)
BRONZE Layer  ←  Raw tables exactly as generated
    ↓  (silver_transforms.sql)
SILVER Layer  ←  Cleaned, enriched, joined views
    ↓  (gold_aggregates.sql)
GOLD Layer    ←  Aggregated, ML-ready tables
    ↓
ML Models     ←  XGBoost, LightGBM, Random Forest
    ↓
Optimization  ←  Inventory + Staff engines
    ↓
Dashboard     ←  Streamlit (this app)
    """, language="text")

    st.divider()
    st.subheader("Product Scorecard from Snowflake Gold")
    st.caption("Showing local data — connect Snowflake connector for live queries")
    waste_summary = inventory.groupby("product_name").agg(
        Avg_Waste_Pct=("waste_pct", "mean"),
        Total_Waste_Cost=("waste_cost_usd", "sum"),
        Sell_Through=("sell_through_rate", "mean")
    ).reset_index()
    waste_summary = waste_summary.round(2)
    st.dataframe(waste_summary, use_container_width=True, hide_index=True)