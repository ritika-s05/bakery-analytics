#libraries

import pandas as pd
import numpy as np
import os

#loading RAW DATA
print("📂 Loading raw data...")
products  = pd.read_csv("data/raw/products.csv")
holidays  = pd.read_csv("data/raw/holidays.csv", parse_dates=["holiday_date"])
sales     = pd.read_csv("data/raw/sales.csv", parse_dates=["transaction_ts"])
inventory = pd.read_csv("data/raw/inventory.csv", parse_dates=["baked_at", "baked_date"])
staff     = pd.read_csv("data/raw/staff.csv", parse_dates=["shift_date"])

# 2. CLEAN SALES DATA
print("🧹 Cleaning sales data...")

# Fix dtypes
sales["is_weekend"]      = sales["is_weekend"].astype(bool)
sales["transaction_date"] = sales["transaction_ts"].dt.date
sales["transaction_date"] = pd.to_datetime(sales["transaction_date"])
sales["month"]            = sales["transaction_ts"].dt.month
sales["week"]             = sales["transaction_ts"].dt.isocalendar().week.astype(int)
sales["year"]             = sales["transaction_ts"].dt.year
sales["quarter"]          = sales["transaction_ts"].dt.quarter

# Remove any duplicates
sales = sales.drop_duplicates(subset=["transaction_id"])

#clean inventory data

print("🧹 Cleaning inventory data...")

inventory["baked_date"]  = pd.to_datetime(inventory["baked_date"])
inventory["waste_pct"]   = round(
    inventory["units_wasted"] / inventory["units_baked"] * 100, 2
)
inventory["sell_through_rate"] = round(
    inventory["units_sold"] / inventory["units_baked"] * 100, 2
)

#clean staff data
print("🧹 Cleaning staff data...")

staff["is_weekend"]    = staff["is_weekend"].astype(bool)
staff["month"]         = staff["shift_date"].dt.month
staff["day_of_week"]   = staff["shift_date"].dt.day_name()
staff["understaffing"] = (staff["staff_delta"] < 0).astype(int)
staff["overstaffing"]  = (staff["staff_delta"] > 0).astype(int)

#building daily sales average
print("⚙️  Building daily aggregates...")

daily_sales = sales.groupby(
    ["transaction_date", "product_id", "product_name", "category"]
).agg(
    units_sold=("quantity", "sum"),
    revenue=("revenue", "sum"),
    cogs=("cogs", "sum"),
    gross_margin=("gross_margin", "sum"),
    transactions=("transaction_id", "count")
).reset_index()

daily_sales["margin_pct"] = round(
    daily_sales["gross_margin"] / daily_sales["revenue"] * 100, 2
)

#feature engineering

print("Engineering ML features...")

daily_sales = daily_sales.sort_values(["product_id", "transaction_date"])

# Lag features — what sold yesterday and last week
daily_sales["lag_1d"]  = daily_sales.groupby("product_id")["units_sold"].shift(1)
daily_sales["lag_7d"]  = daily_sales.groupby("product_id")["units_sold"].shift(7)
daily_sales["lag_14d"] = daily_sales.groupby("product_id")["units_sold"].shift(14)

# Rolling averages
daily_sales["rolling_7d_avg"]  = daily_sales.groupby("product_id")["units_sold"].transform(
    lambda x: x.shift(1).rolling(7, min_periods=1).mean()
)
daily_sales["rolling_30d_avg"] = daily_sales.groupby("product_id")["units_sold"].transform(
    lambda x: x.shift(1).rolling(30, min_periods=1).mean()
)

# Date-based features
daily_sales["day_of_week"]    = daily_sales["transaction_date"].dt.day_name()
daily_sales["day_of_week_num"] = daily_sales["transaction_date"].dt.dayofweek
daily_sales["is_weekend"]     = daily_sales["transaction_date"].dt.dayofweek >= 5
daily_sales["month"]          = daily_sales["transaction_date"].dt.month
daily_sales["week"]           = daily_sales["transaction_date"].dt.isocalendar().week.astype(int)
daily_sales["year"]           = daily_sales["transaction_date"].dt.year
daily_sales["quarter"]        = daily_sales["transaction_date"].dt.quarter

# Holiday features
print(" Adding holiday features...")

def get_holiday_features(date, holidays):
    # Is this date within 3 days of a holiday?
    is_holiday_window = False
    holiday_name      = None
    demand_multiplier = 1.0
    days_to_holiday   = 999

    for _, h in holidays.iterrows():
        diff = (h["holiday_date"] - date).days
        abs_diff = abs(diff)
        if abs_diff <= 3:
            is_holiday_window = True
            holiday_name      = h["holiday_name"]
            demand_multiplier = h["demand_multiplier"]
        if 0 <= diff < days_to_holiday:
            days_to_holiday = diff

    return pd.Series({
        "is_holiday_window": is_holiday_window,
        "holiday_name":      holiday_name,
        "demand_multiplier": demand_multiplier,
        "days_to_holiday":   min(days_to_holiday, 30)
    })

holiday_features = daily_sales["transaction_date"].apply(
    lambda d: get_holiday_features(d, holidays)
)
daily_sales = pd.concat([daily_sales, holiday_features], axis=1)

#dropping the rows with null values
daily_sales = daily_sales.dropna(subset=["lag_7d"])

#saved processed data
print("💾 Saving processed data...")
os.makedirs("data/processed", exist_ok=True)

daily_sales.to_csv("data/processed/daily_sales_features.csv", index=False)
inventory.to_csv("data/processed/inventory_clean.csv",        index=False)
staff.to_csv("data/processed/staff_clean.csv",                index=False)
sales.to_csv("data/processed/sales_clean.csv",                index=False)

print("\n✅ Transformation complete!")
print(f"   daily_sales_features: {daily_sales.shape[0]:,} rows x {daily_sales.shape[1]} columns")
print(f"   inventory_clean:      {inventory.shape[0]:,} rows")
print(f"   staff_clean:          {staff.shape[0]:,} rows")
print(f"   sales_clean:          {sales.shape[0]:,} rows")
print(f"\n🔧 ML Features created:")
print(f"   lag_1d, lag_7d, lag_14d")
print(f"   rolling_7d_avg, rolling_30d_avg")
print(f"   day_of_week, is_weekend, month, quarter")
print(f"   is_holiday_window, days_to_holiday, demand_multiplier")
