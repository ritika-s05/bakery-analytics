import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta

#load the datasets
inventory_plan = pd.read_csv("data/exports/inventory_plan.csv")
staff_df       = pd.read_csv("data/processed/staff_clean.csv", parse_dates=["shift_date"])
products       = pd.read_csv("data/raw/products.csv")

#roles defination
# Each role has:
# - base_units_per_shift: how many units they can handle per 8hr shift
# - hourly_rate: cost per hour
# - min_count: always need at least this many
# - weekend_multiplier: extra staff needed on weekends

ROLES = {
    "Head Baker": { "units_per_shift": 40, "hourly_rate":     28, "min_count":       1, "weekend_extra":   1, "description":     "Manages baking operations"
    }, "Pastry Chef": {
        "units_per_shift": 35,
        "hourly_rate":     24,
        "min_count":       1,
        "weekend_extra":   1,
        "description":     "Handles pastries & cakes"
    }, "Kitchen Assistant": {
        "units_per_shift": 50,
        "hourly_rate":     18,
        "min_count":       1,
        "weekend_extra":   2,
        "description":     "Supports baking & prep"
    }, "Store Associate": {
        "units_per_shift": 60,
        "hourly_rate":     16,
        "min_count":       2,
        "weekend_extra":   2,
        "description":     "Customer service & sales"
    }, "Decorator": {
        "units_per_shift": 20,
        "hourly_rate":     22,
        "min_count":       0,
        "weekend_extra":   1,
        "description":     "Cake & pastry decoration"
    },
    "Delivery": {
        "units_per_shift": 80,
        "hourly_rate":     17,
        "min_count":       0,
        "weekend_extra":   1,
        "description":     "Order fulfillment & delivery"
    },}

SHIFT_HOURS = 8

#shift optimization logic 

def optimize_staff(total_units, is_weekend, is_holiday=False):
    schedule = []
    holiday_multiplier = 1.3 if is_holiday else 1.0

    for role, config in ROLES.items():
        # Base requirement from demand
        units_adjusted    = total_units * holiday_multiplier
        raw_needed        = units_adjusted / config["units_per_shift"]
        recommended_count = max(
            config["min_count"],
            int(np.ceil(raw_needed))
        )

        # Add weekend extra
        if is_weekend:
            recommended_count += config["weekend_extra"]

        # Cost calculations
        optimal_cost     = recommended_count * config["hourly_rate"] * SHIFT_HOURS
        min_cost         = config["min_count"] * config["hourly_rate"] * SHIFT_HOURS

        schedule.append({
            "role":              role,
            "description":       config["description"],
            "recommended_count": recommended_count,
            "hourly_rate":       config["hourly_rate"],
            "shift_hours":       SHIFT_HOURS,
            "labor_cost":        round(optimal_cost, 2),
            "cost_per_unit":     round(optimal_cost / max(total_units, 1), 2),
        })

    return pd.DataFrame(schedule)

#generating a 7 day schedule
print("7 days schedule of the staff")

forecast_date = pd.to_datetime(inventory_plan["forecast_date"].iloc[0])
total_units   = inventory_plan["recommended_bake"].sum()

weekly_schedule = []
for i in range(7):
    day         = forecast_date + timedelta(days=i)
    is_weekend  = day.weekday() >= 5
    # Weekends have higher demand
    day_units   = int(total_units * (1.4 if is_weekend else 1.0))
    schedule    = optimize_staff(day_units, is_weekend)
    schedule["shift_date"]  = day.date()
    schedule["day_of_week"] = day.strftime("%A")
    schedule["is_weekend"]  = is_weekend
    schedule["total_units"] = day_units
    weekly_schedule.append(schedule)

weekly_df = pd.concat(weekly_schedule, ignore_index=True)
print(weekly_df[["shift_date", "day_of_week", "role", "recommended_count", "labor_cost"]])

#print today's schedule
today_schedule = weekly_df[weekly_df["shift_date"] == forecast_date.date()]

print(f"{'='*70}")
print(f"  👨‍🍳 STAFF SCHEDULE — {forecast_date.strftime('%A, %B %d %Y')}")
print(f"  Total Units to Produce: {total_units} | "
      f"{'Weekend' if forecast_date.weekday() >= 5 else 'Weekday'}")
print(f"{'='*70}")
print(f"  {'Role':<22} {'Desc':<30} {'Staff':>5} {'Rate':>6} {'Cost':>8}")
print(f"  {'-'*70}")

for _, row in today_schedule.iterrows():
    print(f"  {row['role']:<22} "
          f"{row['description']:<30} "
          f"{row['recommended_count']:>5} "
          f"${row['hourly_rate']:>5}/hr "
          f"${row['labor_cost']:>7.2f}")

total_cost  = today_schedule["labor_cost"].sum()
total_staff = today_schedule["recommended_count"].sum()

print(f"\n  {'TOTALS':<22} {'':>30} {total_staff:>5} {'':>6} ${total_cost:>7.2f}")
print(f"\n  💰 Total Daily Labor Cost: ${total_cost:,.2f}")
print(f"  👥 Total Staff Scheduled:  {total_staff}")
print(f"  📦 Units per Staff Member: {round(total_units/total_staff, 1)}")

#printing the summary for 7 days

print(f"\n\n{'='*60}")
print(f" 7-DAY LABOR COST FORECAST")
print(f"{'='*60}")
print(f"  {'Day':<12} {'Date':<12} {'Staff':>6} {'Units':>7} {'Cost':>10}")
print(f"  {'-'*55}")

weekly_summary = weekly_df.groupby(
    ["shift_date", "day_of_week", "is_weekend", "total_units"]
).agg(
    total_staff=("recommended_count", "sum"),
    total_cost=("labor_cost", "sum")
).reset_index()

total_week_cost  = 0
total_week_staff = 0

for _, row in weekly_summary.iterrows():
    flag = "🌟 " if row["shift_date"] == forecast_date.date() else "   "
    print(f"  {flag}{row['day_of_week']:<10} "
          f"{str(row['shift_date']):<12} "
          f"{row['total_staff']:>6} "
          f"{row['total_units']:>7} "
          f"${row['total_cost']:>9.2f}")
    total_week_cost  += row["total_cost"]
    total_week_staff += row["total_staff"]

print(f"\n  💰 Total Weekly Labor Cost: ${total_week_cost:,.2f}")
print(f"  👥 Total Staff-Days:        {total_week_staff}")


#cost analysis for the week
print(f"\n\n{'='*60}")
print(f"  💡 AI SCHEDULING vs MANUAL SCHEDULING")
print(f"{'='*60}")

# Compare with historical overstaffing
avg_overstaffing = staff_df["overstaffing_cost"].mean()
weekly_saving    = avg_overstaffing * 7
monthly_saving   = avg_overstaffing * 30

print(f"  Avg daily overstaffing cost (historical): ${avg_overstaffing:,.2f}")
print(f"  Estimated weekly savings with AI:         ${weekly_saving:,.2f}")
print(f"  Estimated monthly savings with AI:        ${monthly_saving:,.2f}")
print(f"  Estimated annual savings with AI:         ${avg_overstaffing*365:,.2f}")


#save outputs
os.makedirs("data/exports", exist_ok=True)
weekly_df.to_csv("data/exports/staff_schedule.csv", index=False)

print(f"\n Saved → data/exports/staff_schedule.csv")
print(f" Staff optimization complete!")
