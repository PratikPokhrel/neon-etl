# Databricks notebook source
# Silver layer: standardization and data quality ONLY - one silver table
# per bronze table, no joins. Joins/business logic belong in the
# integration or gold layer, which will read multiple silver tables
# together via dlt.read()/dlt.read_stream().

import dlt
from pyspark.sql import functions as F
from pyspark.sql import Window

# COMMAND ----------


@dlt.view(name="customers_cleaned")
def customers_cleaned():
    # No customer_sk here - surrogate keys are a GOLD/dimensional concern
    # (they exist to identify "this SCD2 version" for BI joins), not a
    # silver concern. Silver stops at: clean columns, correct types, and
    # SCD2 history via __START_AT/__END_AT (added by apply_changes below).
    # customer_id (the business key) is what silver carries forward.
    df = dlt.read_stream("bronze_customers")
    return df.select(
        F.col("customer_id"),
        F.trim(F.col("name")).alias("customer_name"),
        F.lower(F.trim(F.col("email"))).alias("email"),
        F.col("signup_date").cast("date"),
        F.col("phone"),
        F.col("date_of_birth").cast("date"),
        F.coalesce(F.col("loyalty_tier"), F.lit("Bronze")).alias("loyalty_tier"),
        F.col("updated_at").cast("timestamp"),
    )


dlt.create_streaming_table(
    name="silver_customers",
    comment="Standardized customers with SCD2 history (business key + __START_AT/__END_AT). "
            "Surrogate key (customer_sk) is generated downstream in dim_customers, not here.",
    expect_all_or_drop={"has_email": "email IS NOT NULL"},
    expect_all={"valid_signup_date": "signup_date <= current_date()"},
)

dlt.apply_changes(
    target="silver_customers",
    source="customers_cleaned",
    keys=["customer_id"],
    sequence_by=F.col("updated_at"),
    stored_as_scd_type=2,
    track_history_column_list=["loyalty_tier", "customer_name"],
)

# COMMAND ----------


@dlt.view(name="products_cleaned")
def products_cleaned():
    # Same reasoning as customers_cleaned - no product_sk here.
    df = dlt.read_stream("bronze_products")
    return df.select(
        F.col("product_id"),
        F.col("sku"),
        F.trim(F.col("name")).alias("product_name"),
        F.col("category"),
        F.col("category_id"),
        F.col("price").cast("decimal(10,2)"),
        F.col("updated_at").cast("timestamp"),
    )


dlt.create_streaming_table(
    name="silver_products",
    comment="Standardized products with SCD2 history (business key + __START_AT/__END_AT). "
            "Surrogate key (product_sk) is generated downstream in dim_products, not here.",
    expect_all_or_drop={"has_sku": "sku IS NOT NULL"},
    expect_all={"valid_price": "price > 0"},
)

dlt.apply_changes(
    target="silver_products",
    source="products_cleaned",
    keys=["product_id"],
    sequence_by=F.col("updated_at"),
    stored_as_scd_type=2,
    track_history_column_list=["price"],
)

# COMMAND ----------


@dlt.view(name="orders_cleaned")
def orders_cleaned():
    df = dlt.read_stream("bronze_orders")
    return df.select(
        F.col("order_id"),
        F.col("customer_id"),
        F.col("channel_id"),
        F.col("product"),
        F.col("amount").cast("decimal(10,2)"),
        F.col("status").alias("order_status"),
        F.col("created_at").cast("timestamp"),
        F.col("updated_at").cast("timestamp"),
    )


dlt.create_streaming_table(
    name="silver_orders",
    comment="Standardized orders - latest version per order_id via apply_changes.",
    expect_all_or_drop={"valid_order_id": "order_id IS NOT NULL"},
    expect_all={"valid_amount": "amount > 0", "has_customer": "customer_id IS NOT NULL"},
)

dlt.apply_changes(
    target="silver_orders",
    source="orders_cleaned",
    keys=["order_id"],
    sequence_by=F.col("updated_at"),
    stored_as_scd_type=1,
)

# COMMAND ----------


@dlt.table(
    name="silver_order_items",
    comment="Standardized order line items - typed, quality-checked. Insert-only, no dedup needed.",
)
@dlt.expect_or_drop("valid_order_item_id", "order_item_id IS NOT NULL")
@dlt.expect("valid_quantity", "quantity > 0")
@dlt.expect("valid_unit_price", "unit_price > 0")
def silver_order_items():
    df = dlt.read_stream("bronze_order_items")
    return df.select(
        F.col("order_item_id"),
        F.col("order_id"),
        F.col("product_id"),
        F.col("quantity"),
        F.col("unit_price").cast("decimal(10,2)"),
        F.coalesce(F.col("discount_pct"), F.lit(0)).alias("discount_pct"),
    )

# COMMAND ----------


@dlt.table(
    name="silver_addresses",
    comment="Standardized addresses - typed, quality-checked. Insert-only, no dedup needed.",
)
@dlt.expect_or_drop("has_customer", "customer_id IS NOT NULL")
def silver_addresses():
    df = dlt.read_stream("bronze_addresses")
    return df.select(
        F.col("address_id"),
        F.col("customer_id"),
        F.coalesce(F.col("address_type"), F.lit("shipping")).alias("address_type"),
        F.col("street"),
        F.col("city"),
        F.col("state"),
        F.col("zip_code"),
        F.col("country"),
        F.col("created_at").cast("timestamp"),
    )

# COMMAND ----------


@dlt.table(
    name="silver_order_status_history",
    comment="Standardized order status history - typed. Insert-only, no dedup needed.",
)
@dlt.expect_or_drop("has_order_id", "order_id IS NOT NULL")
def silver_order_status_history():
    df = dlt.read_stream("bronze_order_status_history")
    return df.select(
        F.col("history_id"),
        F.col("order_id"),
        F.col("old_status"),
        F.col("new_status"),
        F.col("changed_at").cast("timestamp"),
    )

# COMMAND ----------


@dlt.table(
    name="silver_reviews",
    comment="Standardized reviews - typed, quality-checked. Insert-only, no dedup needed.",
)
@dlt.expect("valid_rating", "rating BETWEEN 1 AND 5")
def silver_reviews():
    df = dlt.read_stream("bronze_reviews")
    return df.select(
        F.col("review_id"),
        F.col("customer_id"),
        F.col("product_id"),
        F.col("rating").cast("int"),
        F.col("review_text"),
        F.col("created_at").cast("timestamp"),
    )

# COMMAND ----------


@dlt.table(
    name="silver_payments",
    comment="Standardized payments - typed, quality-checked.",
)
@dlt.expect_or_drop("has_order_id", "order_id IS NOT NULL")
@dlt.expect("valid_amount", "amount > 0")
def silver_payments():
    df = dlt.read("bronze_payments")
    return df.select(
        F.col("payment_id"),
        F.col("order_id"),
        F.col("amount").cast("decimal(10,2)"),
        F.col("method"),
        F.coalesce(F.col("status"), F.lit("completed")).alias("payment_status"),
        F.col("paid_at").cast("timestamp"),
    )

# COMMAND ----------


@dlt.table(
    name="silver_shipments",
    comment="Standardized shipments - typed, quality-checked.",
)
@dlt.expect_or_drop("has_order_id", "order_id IS NOT NULL")
def silver_shipments():
    df = dlt.read("bronze_shipments")
    return df.select(
        F.col("shipment_id"),
        F.col("order_id"),
        F.col("carrier"),
        F.col("tracking_number"),
        F.col("shipped_at").cast("timestamp"),
        F.col("delivered_at").cast("timestamp"),
        F.col("ship_status"),
    )

# COMMAND ----------


@dlt.table(
    name="silver_categories",
    comment="Standardized categories - typed.",
)
def silver_categories():
    df = dlt.read("bronze_categories")
    return df.select(
        F.col("category_id"),
        F.col("category_name"),
        F.col("parent_category_id"),
    )

# COMMAND ----------


@dlt.table(
    name="silver_channels",
    comment="Standardized channels - typed.",
)
def silver_channels():
    df = dlt.read("bronze_channels")
    return df.select(
        F.col("channel_id"),
        F.col("channel_name"),
        F.col("channel_type"),
    )