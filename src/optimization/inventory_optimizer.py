import pandas as pd
import numpy as np
import pickle
import os

# Load the trained demand forecasting model
df       = pd.read_csv("data/processed/daily_sales_features.csv", parse_dates=["transaction_date"])
inventory = pd.read_csv("data/processed/inventory_clean.csv", parse_dates=["baked_date"])
products  = pd.read_csv("data/raw/products.csv")

with open("src/models/artifacts/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("src/models/artifacts/label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

#predicting demand for tomorrow

tomorrow = df["transaction_date"].max() + pd.Timedelta(days=1)
is_weekend = tomorrow.weekday() >= 5

predictions = []

for _, product in products.iterrows():
    # Get last known data for this product
    prod_df = df[df["product_name"] == product["product_name"]].sort_values("transaction_date")

    if len(prod_df) == 0:
        continue

    last = prod_df.iloc[-1]

    # Build feature row
    try:
        product_encoded  = encoders["product"].transform([product["product_name"]])[0]
        category_encoded = encoders["category"].transform([product["category"]])[0]
        dow_encoded      = encoders["dow"].transform([tomorrow.strftime("%A")])[0]
    except:
        continue

    # Interaction features
    demand_multiplier      = last["demand_multiplier"] if not pd.isna(last["demand_multiplier"]) else 1.0
    weekend_x_multiplier   = int(is_weekend) * demand_multiplier
    lag_x_multiplier       = last["lag_7d"] * demand_multiplier

    features = pd.DataFrame([{
        "product_encoded":      product_encoded,
        "category_encoded":     category_encoded,
        "dow_encoded":          dow_encoded,
        "is_weekend":           int(is_weekend),
        "month":                tomorrow.month,
        "quarter":              (tomorrow.month - 1) // 3 + 1,
        "week":                 tomorrow.isocalendar()[1],
        "lag_1d":               last["units_sold"],
        "lag_7d":               last["lag_7d"] if not pd.isna(last["lag_7d"]) else last["units_sold"],
        "lag_14d":              last["lag_14d"] if not pd.isna(last["lag_14d"]) else last["units_sold"],
        "rolling_7d_avg":       last["rolling_7d_avg"] if not pd.isna(last["rolling_7d_avg"]) else last["units_sold"],
        "rolling_30d_avg":      last["rolling_30d_avg"] if not pd.isna(last["rolling_30d_avg"]) else last["units_sold"],
        "is_holiday_window":    int(last["is_holiday_window"]) if not pd.isna(last["is_holiday_window"]) else 0,
        "demand_multiplier":    demand_multiplier,
        "days_to_holiday":      last["days_to_holiday"] if not pd.isna(last["days_to_holiday"]) else 30,
        "weekend_x_multiplier": weekend_x_multiplier,
        "lag_x_multiplier":     lag_x_multiplier,
    }])

    predicted_units = max(0, round(model.predict(features)[0]))

    predictions.append({
        "product_id":       product["product_id"],
        "product_name":     product["product_name"],
        "category":         product["category"],
        "shelf_life_hours": product["shelf_life_hours"],
        "unit_cost":        product["unit_cost"],
        "unit_price":       product["unit_price"],
        "prep_hours":       product["prep_hours"],
        "predicted_demand": predicted_units,
        "is_weekend":       is_weekend,
        "forecast_date":    tomorrow.date(),
    })

pred_df = pd.DataFrame(predictions)

#inventory optimization logic
print("Running inventory optimization.")

# Safety stock — buffer based on shelf life
# Short shelf life = lower buffer (can't hold excess)
# Long shelf life  = higher buffer (can hold some extra)

def get_safety_stock(shelf_life_hours, predicted_demand):
    if shelf_life_hours <= 12:
        buffer = 0.05   # 5% buffer — very perishable
    elif shelf_life_hours <= 24:
        buffer = 0.10   # 10% buffer
    elif shelf_life_hours <= 48:
        buffer = 0.15   # 15% buffer
    else:
        buffer = 0.20   # 20% buffer — longer shelf life
    return max(1, round(predicted_demand * buffer))

pred_df["safety_stock"]       = pred_df.apply(
    lambda r: get_safety_stock(r["shelf_life_hours"], r["predicted_demand"]), axis=1
)
pred_df["recommended_bake"]   = pred_df["predicted_demand"] + pred_df["safety_stock"]
pred_df["estimated_revenue"]  = round(pred_df["recommended_bake"] * pred_df["unit_price"], 2)
pred_df["estimated_cost"]     = round(pred_df["recommended_bake"] * pred_df["unit_cost"], 2)
pred_df["estimated_margin"]   = round(pred_df["estimated_revenue"] - pred_df["estimated_cost"], 2)
pred_df["max_waste_risk"]     = round(pred_df["safety_stock"] * pred_df["unit_cost"], 2)

#freshness risk flag
pred_df["freshness_risk"] = pred_df["shelf_life_hours"].apply(
    lambda h: "🔴 HIGH"   if h <= 12
    else      "🟡 MEDIUM" if h <= 24
    else      "🟢 LOW"
)

#print recommendations
print(f"\n{'='*75}")
print(f"  🍞 BAKERY INVENTORY PLAN — {tomorrow.strftime('%A, %B %d %Y')}")
print(f"  {'Weekend' if is_weekend else 'Weekday'} Schedule")
print(f"{'='*75}")
print(f"  {'Product':<22} {'Pred':>5} {'Buffer':>7} {'Bake':>6} {'Revenue':>9} {'Waste Risk':>11} {'Freshness':>10}")
print(f"  {'-'*75}")

for _, row in pred_df.sort_values("estimated_margin", ascending=False).iterrows():
    print(f"  {row['product_name']:<22} "
          f"{row['predicted_demand']:>5} "
          f"{row['safety_stock']:>7} "
          f"{row['recommended_bake']:>6} "
          f"${row['estimated_revenue']:>8.2f} "
          f"${row['max_waste_risk']:>10.2f} "
          f"{row['freshness_risk']:>10}")

print(f"\n  {'TOTALS':<22} "
      f"{'':>5} {'':>7} "
      f"{pred_df['recommended_bake'].sum():>6} "
      f"${pred_df['estimated_revenue'].sum():>8.2f} "
      f"${pred_df['max_waste_risk'].sum():>10.2f}")

print(f"\n  💰 Estimated Revenue:  ${pred_df['estimated_revenue'].sum():,.2f}")
print(f"  📦 Total Units to Bake: {pred_df['recommended_bake'].sum():,}")
print(f"  ⚠️  Max Waste Risk:     ${pred_df['max_waste_risk'].sum():,.2f}")
print(f"  📈 Estimated Margin:   ${pred_df['estimated_margin'].sum():,.2f}")


#save recommendations
os.makedirs("data/exports", exist_ok=True)
pred_df.to_csv("data/exports/inventory_plan.csv", index=False)
print(f"\n Saved → data/exports/inventory_plan.csv")
print(f" Inventory optimization complete!")
