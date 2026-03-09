#importing libraries
import snowflake.connector
from dotenv import load_dotenv
import os

load_dotenv()

def get_connection():
    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role=os.getenv("SNOWFLAKE_ROLE")
    )
    return conn

if __name__ == "__main__":
    try:
        conn = get_connection()
        print("✅ Snowflake connection successful!")
        conn.close()
    except Exception as e:
        print(f"❌ Connection failed: {e}")