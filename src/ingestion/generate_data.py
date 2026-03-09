import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import os

fake = Faker()
np .random.seed(42)
random.seed(42)

#Product catalog

products = pd.DataFrame([
    {"product_id": "P001", "product_name": "Butter Croissant",  "category": "Pastry",     "unit_cost": 1.20, "unit_price": 3.50, "shelf_life_hours": 12, "prep_hours": 3},
    {"product_id": "P002", "product_name": "Sourdough Loaf",    "category": "Bread",      "unit_cost": 1.80, "unit_price": 7.00, "shelf_life_hours": 48, "prep_hours": 8},
    {"product_id": "P003", "product_name": "Almond Danish",     "category": "Pastry",     "unit_cost": 1.10, "unit_price": 3.00, "shelf_life_hours": 10, "prep_hours": 2},
    {"product_id": "P004", "product_name": "Chocolate Cake",    "category": "Cake",       "unit_cost": 8.00, "unit_price": 32.0, "shelf_life_hours": 72, "prep_hours": 4},
    {"product_id": "P005", "product_name": "Cinnamon Roll",     "category": "Pastry",     "unit_cost": 0.90, "unit_price": 4.00, "shelf_life_hours": 8,  "prep_hours": 2},
    {"product_id": "P006", "product_name": "Bagel Plain",       "category": "Bread",      "unit_cost": 0.50, "unit_price": 2.00, "shelf_life_hours": 24, "prep_hours": 2},
    {"product_id": "P007", "product_name": "Macaron Box",       "category": "Confection", "unit_cost": 4.00, "unit_price": 18.0, "shelf_life_hours": 72, "prep_hours": 3},
    {"product_id": "P008", "product_name": "Cheese Danish",     "category": "Pastry",     "unit_cost": 1.20, "unit_price": 3.50, "shelf_life_hours": 10, "prep_hours": 2},
    {"product_id": "P009", "product_name": "Baguette",          "category": "Bread",      "unit_cost": 0.80, "unit_price": 3.00, "shelf_life_hours": 24, "prep_hours": 4},
    {"product_id": "P010", "product_name": "Custom Cake",       "category": "Cake",       "unit_cost": 18.0, "unit_price": 65.0, "shelf_life_hours": 48, "prep_hours": 6},
    {"product_id": "P011", "product_name": "Blueberry Muffin",  "category": "Muffin",     "unit_cost": 0.70, "unit_price": 3.00, "shelf_life_hours": 24, "prep_hours": 1},
    {"product_id": "P012", "product_name": "Brioche Loaf",      "category": "Bread",      "unit_cost": 2.00, "unit_price": 8.00, "shelf_life_hours": 48, "prep_hours": 5},
])

#holidays and special events
holidays = pd.DataFrame([
    {"holiday_name": "Valentine's Day",  "holiday_date": "2024-02-14", "demand_multiplier": 2.2},
    {"holiday_name": "Mother's Day",     "holiday_date": "2024-05-12", "demand_multiplier": 2.8},
    {"holiday_name": "Father's Day",     "holiday_date": "2024-06-16", "demand_multiplier": 1.8},
    {"holiday_name": "4th of July",      "holiday_date": "2024-07-04", "demand_multiplier": 1.6},
    {"holiday_name": "Thanksgiving",     "holiday_date": "2024-11-28", "demand_multiplier": 2.0},
    {"holiday_name": "Christmas Eve",    "holiday_date": "2024-12-24", "demand_multiplier": 2.5},
    {"holiday_name": "Christmas Day",    "holiday_date": "2024-12-25", "demand_multiplier": 2.3},
    {"holiday_name": "New Year's Eve",   "holiday_date": "2024-12-31", "demand_multiplier": 1.9},
    {"holiday_name": "Valentine's Day",  "holiday_date": "2025-02-14", "demand_multiplier": 2.2},
    {"holiday_name": "Mother's Day",     "holiday_date": "2025-05-11", "demand_multiplier": 2.8},
    {"holiday_name": "Easter",           "holiday_date": "2025-04-20", "demand_multiplier": 1.7},
])
holidays["holiday_date"] = pd.to_datetime(holidays["holiday_date"])

#sales transactions for 18 months

start_date = datetime(2024, 1, 1)
end_date   = datetime(2025, 6, 30)

def get_demand_multiplier(date):
    # Weekend boost
    multiplier = 1.8 if date.weekday() >= 5 else 1.0
    # Holiday boost — within 3 days of a holiday
    for _, h in holidays.iterrows():
        diff = abs((date.date() - h["holiday_date"].date()).days)
        if diff <= 3:
            multiplier *= h["demand_multiplier"]
            break
    return multiplier

records = []
current = start_date
while current <= end_date:
    multiplier = get_demand_multiplier(current)
    daily_transactions = int(random.randint(40, 80) * multiplier)

    for _ in range(daily_transactions):
        product = products.sample(1).iloc[0]
        hour    = random.choices(
            range(6, 20),
            weights=[5,10,15,20,15,12,10,8,7,5,4,3,3,2],
            k=1
        )[0]
        minute  = random.randint(0, 59)
        ts      = current.replace(hour=hour, minute=minute)

        quantity = random.choices([1, 2, 3, 4], weights=[50, 30, 15, 5], k=1)[0]

        records.append({
            "transaction_id": fake.uuid4(),
            "product_id":     product["product_id"],
            "product_name":   product["product_name"],
            "category":       product["category"],
            "quantity":       quantity,
            "unit_price":     product["unit_price"],
            "unit_cost":      product["unit_cost"],
            "revenue":        round(quantity * product["unit_price"], 2),
            "cogs":           round(quantity * product["unit_cost"], 2),
            "gross_margin":   round(quantity * (product["unit_price"] - product["unit_cost"]), 2),
            "transaction_ts": ts,
            "day_of_week":    ts.strftime("%A"),
            "is_weekend":     ts.weekday() >= 5,
            "hour":           hour,
            "channel":        random.choices(["in_store", "online", "catering"], weights=[70, 20, 10], k=1)[0],
            "staff_id":       f"S{random.randint(1, 12):03d}",
        })

    current += timedelta(days=1)

sales = pd.DataFrame(records)

#inventory batches

batches = []
batch_date = start_date
while batch_date <= end_date:
    multiplier = get_demand_multiplier(batch_date)
    for _, product in products.iterrows():
        base_units = random.randint(20, 60)
        units_baked = int(base_units * multiplier)
        daily_sales = sales[
            (sales["product_id"] == product["product_id"]) &
            (sales["transaction_ts"].dt.date == batch_date.date())
        ]["quantity"].sum()
        units_sold      = min(int(daily_sales), units_baked)
        units_wasted    = max(0, units_baked - units_sold)
        waste_cost      = round(units_wasted * product["unit_cost"], 2)

        batches.append({
            "batch_id":       fake.uuid4(),
            "product_id":     product["product_id"],
            "product_name":   product["product_name"],
            "baked_date":     batch_date.date(),
            "baked_at":       batch_date.replace(hour=random.randint(4, 7)),
            "units_baked":    units_baked,
            "units_sold":     units_sold,
            "units_wasted":   units_wasted,
            "waste_cost_usd": waste_cost,
            "shelf_life_hours": product["shelf_life_hours"],
        })
    batch_date += timedelta(days=1)

inventory = pd.DataFrame(batches)

#staff schedule 

roles = ["Head Baker", "Pastry Chef", "Kitchen Assistant", "Store Associate", "Decorator", "Delivery"]
staff_records = []
sched_date = start_date
while sched_date <= end_date:
    is_weekend  = sched_date.weekday() >= 5
    multiplier  = get_demand_multiplier(sched_date)
    for role in roles:
        base = {"Head Baker": 2, "Pastry Chef": 2, "Kitchen Assistant": 3,
                "Store Associate": 4, "Decorator": 1, "Delivery": 2}[role]
        scheduled   = int(base * (1.5 if is_weekend else 1.0)) + random.choices([-1, 0, 0, 1], weights=[10, 50, 30, 10], k=1)[0]
        recommended = max(1, int(base * min(multiplier, 2.0)))
        scheduled   = max(1, scheduled)

        hourly_rate = {"Head Baker": 28, "Pastry Chef": 24, "Kitchen Assistant": 18,
                       "Store Associate": 16, "Decorator": 22, "Delivery": 17}[role]
        delta       = scheduled - recommended
        staff_records.append({
            "shift_date":         sched_date.date(),
            "role":               role,
            "scheduled_count":    scheduled,
            "recommended_count":  recommended,
            "staff_delta":        delta,
            "hourly_rate":        hourly_rate,
            "shift_hours":        8,
            "overstaffing_cost":  round(max(0, delta) * hourly_rate * 8, 2),
            "is_weekend":         is_weekend,
        })
    sched_date += timedelta(days=1)

staff = pd.DataFrame(staff_records)

#save to csv

os.makedirs("data/raw", exist_ok=True)

products.to_csv("data/raw/products.csv",   index=False)
holidays.to_csv("data/raw/holidays.csv",   index=False)
sales.to_csv("data/raw/sales.csv",         index=False)
inventory.to_csv("data/raw/inventory.csv", index=False)
staff.to_csv("data/raw/staff.csv",         index=False)

print("✅ Data generation complete!")
print(f"   Products:     {len(products):,} rows")
print(f"   Holidays:     {len(holidays):,} rows")
print(f"   Sales:        {len(sales):,} rows")
print(f"   Inventory:    {len(inventory):,} rows")
print(f"   Staff:        {len(staff):,} rows")
