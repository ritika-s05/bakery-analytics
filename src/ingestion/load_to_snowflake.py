import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from dotenv import load_dotenv
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.snowflake_conn import get_connection

load_dotenv()

def run_sql_file(conn, filepath):
    with open(filepath, "r") as f:
        sql = f.read()
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    cursor = conn.cursor()
    for statement in statements:
        cursor.execute(statement)
        print(f"✅ Executed: {statement[:60]}...")
    cursor.close()

def load_dataframe(conn, df, table_name, schema="BRONZE"):
    # Uppercase all column names for Snowflake
    df.columns = [c.upper() for c in df.columns]
    # Rename reserved keywords
    reserved = {"MONTH": "MONTH_NUM", "HOUR": "HOUR_OF_DAY", "WEEK": "WEEK_NUM", "YEAR": "YEAR_NUM"}
    df.columns = [reserved.get(c, c) for c in df.columns]
    success, nchunks, nrows, _ = write_pandas(
        conn, df, table_name,
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=schema,
        overwrite=True
    )
    print(f"✅ Loaded {nrows:,} rows into {schema}.{table_name}")

def main():
    print("❄️  Connecting to Snowflake...")
    conn = get_connection()

    # Step 1 — Run setup SQL
    print("\n📦 Creating database, schemas and tables...")
    run_sql_file(conn, "sql/bronze/setup.sql")

    # Step 2 — Load all processed CSVs
    print("\n📤 Loading data into Snowflake Bronze layer...")

    products  = pd.read_csv("data/raw/products.csv")
    holidays  = pd.read_csv("data/raw/holidays.csv")
    sales     = pd.read_csv("data/processed/sales_clean.csv")
    sales     = sales[[
    'transaction_id', 'product_id', 'product_name', 'category',
    'quantity', 'unit_price', 'unit_cost', 'revenue', 'cogs',
    'gross_margin', 'transaction_ts', 'transaction_date', 'day_of_week',
    'is_weekend', 'hour', 'channel', 'staff_id']]
    inventory = pd.read_csv("data/processed/inventory_clean.csv")
    staff     = pd.read_csv("data/processed/staff_clean.csv")

    load_dataframe(conn, products,  "PRODUCTS")
    load_dataframe(conn, holidays,  "HOLIDAYS")
    load_dataframe(conn, sales,     "SALES")
    load_dataframe(conn, inventory, "INVENTORY")
    load_dataframe(conn, staff,     "STAFF")

    conn.close()
    print("\n🎉 All data loaded into Snowflake successfully!")

if __name__ == "__main__":
    main()