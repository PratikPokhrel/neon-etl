# Databricks notebook source
# Silver layer: standardization and data quality ONLY - one silver table
# per bronze table, no joins. Joins/business logic belong in the
# integration or gold layer, which will read multiple silver tables
# together via dlt.read()/dlt.read_stream().
#
# What "standardization" means at this layer:
# - consistent column naming (snake_case, clear names)
# - correct types (cast, not just inferred)
# - deduplication to the latest version of each record
# - data quality expectations (drop or flag bad rows)
# - NOT: joins, aggregations, business-specific derived columns - those
#   are integration/gold's job, where multiple entities come together.

import dlt
from pyspark.sql import functions as F
from pyspark.sql import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ### Why not ROW_NUMBER() for dedup here
# MAGIC Structured Streaming processes data in micro-batches and never sees
# MAGIC the full table history at once, so it can't rank "all rows for this
# MAGIC key" against each other the way ROW_NUMBER() needs to - that's the
# MAGIC NON_TIME_WINDOW_NOT_SUPPORTED_IN_STREAMING error you hit. Streaming
# MAGIC only supports windowing based on TIME, not arbitrary per-key ranking.
# MAGIC
# MAGIC The correct DLT tool for "keep the latest version per key from a
# MAGIC stream of changes" is dlt.apply_changes - built exactly for this.
# MAGIC It takes a source of raw changes (inserts/updates), a set of keys,
# MAGIC and a column to sequence by, and maintains an always-current
# MAGIC "latest per key" table without needing any window function at all.
# MAGIC
# MAGIC Pattern: a @dlt.view does the casting/cleaning (no dedup logic in
# MAGIC it), then apply_changes reads FROM that view and writes the
# MAGIC deduplicated result into a streaming table target.

# COMMAND ----------


@dlt.view(name="customers_cleaned")
def customers_cleaned():
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
    ).withColumn(
        # customer_sk computed BEFORE apply_changes, from customer_id + the
        # SAME column that drives versioning (loyalty_tier, via
        # track_history_column_list below). Because a new SCD2 version is
        # only ever created when loyalty_tier changes, this hash changes at
        # exactly the same moments a new version is created - no earlier,
        # no later. That means it can flow straight through apply_changes
        # as an ordinary column, with __START_AT/__END_AT added alongside
        # it - no second table, no extra step, no full recompute needed.
        #
        # Known limitation: if a customer's loyalty_tier ever returns to a
        # PREVIOUS value (Gold -> Silver -> Gold again), the second "Gold"
        # period would hash to the same customer_sk as the first - a
        # genuine edge case worth knowing, though rare for a tier field
        # that mostly moves in one direction. A monotonically increasing
        # per-customer version counter would avoid this if it ever matters.
        "customer_sk",
        F.sha2(F.concat_ws("||", F.col("customer_id"), F.col("loyalty_tier")), 256),
    )


dlt.create_streaming_table(
    name="silver_customers",
    comment="Standardized customers with SCD2 history and a surrogate key, all in one table. "
            "loyalty_tier changes over time, so historical joins (e.g. against orders) need to see "
            "the tier that was active AT THE TIME of the order, not just the current one.",
    expect_all_or_drop={"has_email": "email IS NOT NULL"},
    expect_all={"valid_signup_date": "signup_date <= current_date()"},
)

dlt.apply_changes(
    target="silver_customers",
    source="customers_cleaned",
    keys=["customer_id"],
    sequence_by=F.col("updated_at"),
    stored_as_scd_type=2,  # keep full history - adds __START_AT / __END_AT columns
    # Only create a new SCD2 version when loyalty_tier itself changes. Without
    # this, a bare updated_at bump with no real business change (or an
    # incidental correction to phone/name) would still spawn a whole new
    # historical row, since apply_changes otherwise compares ALL non-key
    # columns to decide if a row "changed". customer_sk rides along as a
    # normal column - it isn't listed here because it's not itself meant
    # to drive versioning, just to label each version once it's decided.
    track_history_column_list=["loyalty_tier", "customer_name"],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### silver_products

# COMMAND ----------


@dlt.view(name="products_cleaned")
def products_cleaned():
    df = dlt.read_stream("bronze_products")
    return df.select(
        F.col("product_id"),
        F.col("sku"),
        F.trim(F.col("name")).alias("product_name"),
        F.col("category"),
        F.col("category_id"),
        F.col("price").cast("decimal(10,2)"),
        F.col("updated_at").cast("timestamp"),
    ).withColumn(
        # Same technique as customer_sk above: computed from the tracked
        # column (price) before apply_changes, so it changes exactly when
        # a new SCD2 version is created - no second table needed.
        "product_sk",
        F.sha2(F.concat_ws("||", F.col("product_id"), F.col("price").cast("string")), 256),
    )


dlt.create_streaming_table(
    name="silver_products",
    comment="Standardized products with SCD2 history and a surrogate key, all in one table. "
            "price changes over time, so historical revenue/order analysis needs the price that "
            "was active AT THE TIME of the order, not just the current price.",
    expect_all_or_drop={"has_sku": "sku IS NOT NULL"},
    expect_all={"valid_price": "price > 0"},
)

dlt.apply_changes(
    target="silver_products",
    source="products_cleaned",
    keys=["product_id"],
    sequence_by=F.col("updated_at"),
    stored_as_scd_type=2,  # keep full history - adds __START_AT / __END_AT columns
    # Only create a new SCD2 version when price itself changes - same
    # reasoning as silver_customers above.
    track_history_column_list=["price"],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### silver_orders

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

# MAGIC %md
# MAGIC ### silver_order_items

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

# MAGIC %md
# MAGIC ### silver_addresses

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

# MAGIC %md
# MAGIC ### silver_order_status_history

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

# MAGIC %md
# MAGIC ### silver_reviews

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

# MAGIC %md
# MAGIC ### silver_payments, silver_shipments, silver_categories, silver_channels
# MAGIC These read from materialized-view bronze tables (full re-load each
# MAGIC run), so we use dlt.read (batch), not dlt.read_stream, and skip
# MAGIC dedup - a full reload has no historical duplicates to remove.

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
