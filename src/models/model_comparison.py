import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import pickle
import time
import os

# 1. LOAD & PREPARE DATA
print("Loading feature data...")
df = pd.read_csv("data/processed/daily_sales_features.csv", parse_dates=["transaction_date"])

# Encode categoricals
le_product  = LabelEncoder()
le_category = LabelEncoder()
le_dow      = LabelEncoder()

df["product_encoded"]  = le_product.fit_transform(df["product_name"])
df["category_encoded"] = le_category.fit_transform(df["category"])
df["dow_encoded"]      = le_dow.fit_transform(df["day_of_week"])

# Fill nulls
df["is_holiday_window"] = df["is_holiday_window"].fillna(False).astype(int)
df["demand_multiplier"] = df["demand_multiplier"].fillna(1.0)
df["days_to_holiday"]   = df["days_to_holiday"].fillna(30)
df["lag_14d"]           = df["lag_14d"].fillna(df["lag_7d"])
df["is_weekend"]        = df["is_weekend"].astype(int)

FEATURES = [
    "product_encoded",
    "category_encoded",
    "dow_encoded",
    "is_weekend",
    "month",
    "quarter",
    "week",
    "lag_1d",
    "lag_7d",
    "lag_14d",
    "rolling_7d_avg",
    "rolling_30d_avg",
    "is_holiday_window",
    "demand_multiplier",
    "days_to_holiday",
    # New interaction features
    "weekend_x_multiplier",
    "lag_x_multiplier",
]

# Add interaction features
df["weekend_x_multiplier"] = df["is_weekend"].astype(int) * df["demand_multiplier"]
df["lag_x_multiplier"]     = df["lag_7d"] * df["demand_multiplier"]

TARGET = "units_sold"

df = df.dropna(subset=FEATURES + [TARGET])

X = df[FEATURES]
y = df[TARGET]

# Time-based split
cutoff     = df["transaction_date"].max() - pd.Timedelta(days=60)
train_mask = df["transaction_date"] <= cutoff
test_mask  = df["transaction_date"] > cutoff

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

# 2. EVALUATION FUNCTION
def evaluate(name, model, X_train, y_train, X_test, y_test):
    start = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - start, 2)

    y_pred = np.maximum(model.predict(X_test), 0)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  MAE:          {mae:.2f} units")
    print(f"  RMSE:         {rmse:.2f} units")
    print(f"  R²:           {r2:.4f}")
    print(f"  MAPE:         {mape:.2f}%")
    print(f"  Train time:   {train_time}s")

    return {
        "model_name":  name,
        "mae":         round(mae, 3),
        "rmse":        round(rmse, 3),
        "r2":          round(r2, 4),
        "mape":        round(mape, 2),
        "train_time":  train_time,
        "model":       model
    }

# 3. DEFINE & TRAIN ALL MODELS
print("Training all models...\n")

results = []

# Model 1 — Linear Regression (baseline)
results.append(evaluate(
    "Linear Regression",
    LinearRegression(),
    X_train, y_train, X_test, y_test
))

# Model 2 — Random Forest
results.append(evaluate(
    "Random Forest",
    RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ),
    X_train, y_train, X_test, y_test
))

# Model 3 — XGBoost
results.append(evaluate(
    "XGBoost",
    xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ),
    X_train, y_train, X_test, y_test
))
# Model 4 — LightGBM
results.append(evaluate(
    "LightGBM",
    lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ),
    X_train, y_train, X_test, y_test
))

# 4. COMPARISON TABLE

print(f"\n\n{'='*65}")
print("  📊 MODEL COMPARISON SUMMARY")
print(f"{'='*65}")

comparison = pd.DataFrame([{
    "Model":       r["model_name"],
    "MAE":         r["mae"],
    "RMSE":        r["rmse"],
    "R²":          r["r2"],
    "MAPE %":      r["mape"],
    "Train Time":  f"{r['train_time']}s"
} for r in results])

print(comparison.to_string(index=False))


# 5. PICK BEST MODEL

best = min(results, key=lambda x: x["rmse"])
print(f"\n🏆 Best Model: {best['model_name']}")
print(f"   RMSE: {best['rmse']} | R²: {best['r2']} | MAPE: {best['mape']}%")

# 6. SAVE BEST MODEL & COMPARISON

os.makedirs("src/models/artifacts", exist_ok=True)

with open("src/models/artifacts/best_model.pkl", "wb") as f:
    pickle.dump(best["model"], f)

with open("src/models/artifacts/label_encoders.pkl", "wb") as f:
    pickle.dump({
        "product":  le_product,
        "category": le_category,
        "dow":      le_dow
    }, f)

comparison.to_csv("src/models/artifacts/model_comparison.csv", index=False)

print(f" Saved best model → src/models/artifacts/best_model.pkl")
print(f"Saved comparison  → src/models/artifacts/model_comparison.csv")
print(f"Model comparison complete!")

