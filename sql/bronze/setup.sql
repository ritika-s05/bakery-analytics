-- Creating database and schemas
CREATE DATABASE IF NOT EXISTS BAKERY_DB;

USE DATABASE BAKERY_DB;

CREATE SCHEMA IF NOT EXISTS BRONZE;
CREATE SCHEMA IF NOT EXISTS SILVER;
CREATE SCHEMA IF NOT EXISTS GOLD;

USE SCHEMA BRONZE;

CREATE OR REPLACE TABLE PRODUCTS (
    product_id        VARCHAR(10),
    product_name      VARCHAR(100),
    category          VARCHAR(50),
    unit_cost         FLOAT,
    unit_price        FLOAT,
    shelf_life_hours  INT,
    prep_hours        INT
);

CREATE OR REPLACE TABLE HOLIDAYS (
    holiday_name       VARCHAR(100),
    holiday_date       DATE,
    demand_multiplier  FLOAT
);

CREATE OR REPLACE TABLE SALES (
    transaction_id    VARCHAR(36),
    product_id        VARCHAR(10),
    product_name      VARCHAR(100),
    category          VARCHAR(50),
    quantity          INT,
    unit_price        FLOAT,
    unit_cost         FLOAT,
    revenue           FLOAT,
    cogs              FLOAT,
    gross_margin      FLOAT,
    transaction_ts    TIMESTAMP,
    transaction_date  DATE,
    day_of_week       VARCHAR(20),
    is_weekend        BOOLEAN,
    hour_of_day       INT,
    channel           VARCHAR(20),
    staff_id          VARCHAR(10)
);

CREATE OR REPLACE TABLE INVENTORY (
    batch_id           VARCHAR(36),
    product_id         VARCHAR(10),
    product_name       VARCHAR(100),
    baked_date         DATE,
    baked_at           TIMESTAMP,
    units_baked        INT,
    units_sold         INT,
    units_wasted       INT,
    waste_cost_usd     FLOAT,
    shelf_life_hours   INT,
    waste_pct          FLOAT,
    sell_through_rate  FLOAT
);

CREATE OR REPLACE TABLE STAFF (
    shift_date          DATE,
    role                VARCHAR(50),
    scheduled_count     INT,
    recommended_count   INT,
    staff_delta         INT,
    hourly_rate         FLOAT,
    shift_hours         INT,
    overstaffing_cost   FLOAT,
    is_weekend          BOOLEAN,
    month_num           INT,
    day_of_week         VARCHAR(20),
    understaffing       INT,
    overstaffing        INT
);
