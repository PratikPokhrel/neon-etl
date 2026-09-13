# Databricks notebook source
# MAGIC %md
# MAGIC # Gold layer: dimensions + business aggregates
# MAGIC
# MAGIC Reconstructed from the expectations baked into the gold test suite,
# MAGIC since the original notebook was lost. Reads multiple silver tables
# MAGIC together (joins/aggregations live here, never in silver).
# MAGIC
# MAGIC Two different kinds of joins happen in this file, and they are NOT
# MAGIC interchangeable:
# MAGIC - **Point-in-time joins** (gold_orders_enriched, gold_order_items_enriched)
# MAGIC   match a fact to the dimension version that was ACTIVE AT THE TIME
# MAGIC   the fact happened - using `valid_from <= event_time < valid_to`
# MAGIC   against SCD2 history. This preserves "what was true then."
# MAGIC - **Current-state joins** (gold_customer_order_summary) intentionally
# MAGIC   use only `is_current = true` rows - this is a snapshot as of NOW,
# MAGIC   not history.
# MAGIC
# MAGIC Mixing these two up is the most common bug in this kind of model -
# MAGIC always check which one a given gold table is supposed to be doing.

import dlt
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dimension tables

# COMMAND ----------


@dlt.table(
    name="dim_customers",
    comment="Customer dimension - full SCD2 history from silver_customers, "
            "with friendlier column names (valid_from/valid_to) and an "
            "is_current flag for convenience.",
)
def dim_customers():
    df = dlt.read("main.bronze.silver_customers")
    return (
        df.withColumnRenamed("__START_AT", "valid_from")
        .withColumnRenamed("__END_AT", "valid_to")
        .withColumn("is_current", F.col("valid_to").isNull())
    )


@dlt.table(
    name="dim_products",
    comment="Product dimension - full SCD2 history from silver_products, "
            "with friendlier column names and an is_current flag.",
)
def dim_products():
    df = dlt.read("main.bronze.silver_products")
    return (
        df.withColumnRenamed("__START_AT", "valid_from")
        .withColumnRenamed("__END_AT", "valid_to")
        .withColumn("is_current", F.col("valid_to").isNull())
    )


@dlt.table(
    name="dim_channels",
    comment="Channel dimension - simple pass-through, channels have no history to track.",
)
def dim_channels():
    return dlt.read("main.bronze.silver_channels")


@dlt.table(
    name="dim_date",
    comment="Date dimension spanning the observed order date range, "
            "with calendar attributes for grouping/filtering in BI tools.",
)
def dim_date():
    orders = dlt.read("main.bronze.silver_orders")
    bounds = orders.select(
        F.min(F.to_date("created_at")).alias("min_date"),
        F.max(F.to_date("created_at")).alias("max_date"),
    )

    return (
        bounds.select(
            F.explode(
                F.sequence(F.col("min_date"), F.col("max_date"), F.expr("interval 1 day"))
            ).alias("full_date")
        )
        .withColumn("year", F.year("full_date"))
        .withColumn("month", F.month("full_date"))
        .withColumn("day", F.dayofmonth("full_date"))
        .withColumn("day_of_week", F.dayofweek("full_date"))  # Spark: Sunday=1 ... Saturday=7
        .withColumn("day_name", F.date_format("full_date", "EEEE"))
        .withColumn("is_weekend", F.col("day_of_week").isin(1, 7))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enriched fact tables (point-in-time joins)

# COMMAND ----------


@dlt.table(
    name="gold_orders_enriched",
    comment="Orders enriched with the customer profile and loyalty tier "
            "that was ACTIVE AT THE TIME the order was placed - a "
            "point-in-time join against customer SCD2 history, NOT the "
            "customer's current tier.",
)
def gold_orders_enriched():
    orders = dlt.read("main.bronze.silver_orders")
    customers = dlt.read("dim_customers")
    channels = dlt.read("dim_channels")

    return (
        orders.alias("o")
        .join(
            customers.alias("c"),
            (F.col("o.customer_id") == F.col("c.customer_id"))
            & (F.col("o.created_at") >= F.col("c.valid_from"))
            & (F.col("c.valid_to").isNull() | (F.col("o.created_at") < F.col("c.valid_to"))),
            "left",
        )
        .join(channels.alias("ch"), F.col("o.channel_id") == F.col("ch.channel_id"), "left")
        .select(
            F.col("o.order_id"),
            F.col("o.customer_id"),
            F.col("c.customer_name"),
            F.col("c.loyalty_tier"),
            F.col("o.product"),
            F.col("o.amount"),
            F.col("o.order_status"),
            F.col("ch.channel_name"),
            F.col("o.created_at").alias("order_created_at"),
            F.to_date("o.created_at").alias("order_date"),
        )
    )


@dlt.table(
    name="gold_order_items_enriched",
    comment="Order line items enriched with the product price that was "
            "ACTIVE AT THE TIME of the order - point-in-time join against "
            "product SCD2 price history, NOT the product's current price.",
)
def gold_order_items_enriched():
    order_items = dlt.read("main.bronze.silver_order_items")
    orders = dlt.read("main.bronze.silver_orders")
    products = dlt.read("dim_products")

    return (
        order_items.alias("oi")
        .join(orders.alias("o"), F.col("oi.order_id") == F.col("o.order_id"), "left")
        .join(
            products.alias("p"),
            (F.col("oi.product_id") == F.col("p.product_id"))
            & (F.col("o.created_at") >= F.col("p.valid_from"))
            & (F.col("p.valid_to").isNull() | (F.col("o.created_at") < F.col("p.valid_to"))),
            "left",
        )
        .select(
            F.col("oi.order_item_id"),
            F.col("oi.order_id"),
            F.col("oi.product_id"),
            F.col("p.price").alias("price_at_time_of_order"),
            F.col("oi.quantity"),
            F.col("oi.discount_pct"),
            (
                F.col("oi.quantity")
                * F.col("p.price")
                * (F.lit(1) - F.coalesce(F.col("oi.discount_pct"), F.lit(0)))
            ).alias("line_total"),
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business aggregates (current-state joins - NOT point-in-time)

# COMMAND ----------


@dlt.table(
    name="gold_customer_order_summary",
    comment="Per-customer order totals, using the customer's CURRENT "
            "profile/tier - deliberately NOT point-in-time, since this "
            "answers 'who is this customer today', not history.",
)
def gold_customer_order_summary():
    orders = dlt.read("gold_orders_enriched")
    customers_current = dlt.read("dim_customers").filter("is_current = true")

    agg = orders.groupBy("customer_id").agg(
        F.count("order_id").alias("total_orders"),
        F.sum("amount").alias("total_spend"),
        F.avg("amount").alias("avg_order_value"),
    )

    return (
        agg.alias("a")
        .join(customers_current.alias("c"), "customer_id", "left")
        .select(
            F.col("a.customer_id"),
            F.col("c.customer_name"),
            F.col("c.loyalty_tier"),
            F.col("a.total_orders"),
            F.col("a.total_spend"),
            F.col("a.avg_order_value"),
        )
    )


@dlt.table(
    name="gold_daily_revenue",
    comment="Daily revenue aggregated by order date and channel - a "
            "time-series rollup for dashboards.",
)
def gold_daily_revenue():
    orders = dlt.read("gold_orders_enriched")
    return orders.groupBy("order_date", "channel_name").agg(
        F.count("order_id").alias("num_orders"),
        F.sum("amount").alias("total_revenue"),
        F.avg("amount").alias("avg_order_value"),
    )