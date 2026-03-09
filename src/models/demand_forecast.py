import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pickle
import os

#loading the features
df = pd.read_csv("data/processed/daily_sales_features.csv", parse_dates=["transaction_date"])
print(f"   Shape: {df.shape}")
print(f"   Date range: {df['transaction_date'].min()} → {df['transaction_date'].max()}")

#preprocessing and encoding
print("Preprocessing data..")

le_product  = LabelEncoder()
le_category = LabelEncoder()
le_dow      = LabelEncoder()

df['product_encoded'] = le_product.fit_transform(df['product_name'])
df['category_encoded'] = le_category.fit_transform(df['category'])
df['dow_encoded'] = le_dow.fit_transform(df['day_of_week'])

#filling remaining nulls
df["is_holiday_window"]  = df["is_holiday_window"].fillna(False).astype(int)
df["demand_multiplier"]  = df["demand_multiplier"].fillna(1.0)
df["days_to_holiday"]    = df["days_to_holiday"].fillna(30)
df["lag_14d"]            = df["lag_14d"].fillna(df["lag_7d"])

#defining features and target
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
]

TARGET = "units_sold"

#droping rows with nulls in target
df = df.dropna(subset = FEATURES + [TARGET])
x = df[FEATURES]
y = df[TARGET]

print(f"   Features: {len(FEATURES)}")
print(f"   Training samples: {len(x):,}")

#splitting data into train and test sets
print('Splitting data into train and test sets...')

#use last 60 days as test set
cutoff = df["transaction_date"].max() - pd.Timedelta(days=60)
train_mask = df['transaction_date'] <= cutoff
test_mask = df["transaction_date"] > cutoff

x_train, y_train = x[train_mask], y[train_mask]
x_test, y_test   = x[test_mask], y[test_mask]   

print(f"   Training samples: {len(x_train):,}")
print(f"   Testing samples: {len(x_test):,}")

#training the model

model1 = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.5, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, random_state=42, n_jobs=-1, verbosity=0)
model1.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose = False)

print(" Training complete!")

#evaluating the model
print("\nEvaluating model")
y_pred = model1.predict(x_test)
y_pred = np.maximum(y_pred, 0)  # no negative predictions

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100

print(f"\n   {'Metric':<10} {'Value':<10}")
print(f"   {'-'*20}")
print(f"   {'MAE':<10} {mae:.2f} units")
print(f"   {'RMSE':<10} {rmse:.2f} units")
print(f"   {'R²':<10} {r2:.4f}")
print(f"   {'MAPE':<10} {mape:.2f}%")

#feature importance 
importance = pd.DataFrame({
    "feature":    FEATURES,
    "importance": model1.feature_importances_
}).sort_values("importance", ascending=False)

for _, row in importance.iterrows():
    bar = "█" * int(row["importance"] * 100)
    print(f"   {row['feature']:<25} {bar} {row['importance']:.4f}")

#save models and encoders
os.makedirs("src/models/artifacts", exist_ok=True)

with open("src/models/artifacts/demand_model.pkl", "wb") as f:
    pickle.dump(model1, f)

with open("src/models/artifacts/label_encoders.pkl", "wb") as f:
    pickle.dump({
        "product":  le_product,
        "category": le_category,
        "dow":      le_dow
    }, f)

importance.to_csv("src/models/artifacts/feature_importance.csv", index=False)

print("   Saved: src/models/artifacts/demand_model.pkl")
print("   Saved: src/models/artifacts/label_encoders.pkl")
print("   Saved: src/models/artifacts/feature_importance.csv")
print("\n Demand forecasting model complete!")


