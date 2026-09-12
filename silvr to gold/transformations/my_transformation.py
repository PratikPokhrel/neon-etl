# Databricks notebook source
# Gold layer: joins across silver tables to answer real business questions.
# This is where joins finally happen - silver stayed single-entity on
# purpose so gold could combine them cleanly, without re-deriving cleaning
# logic that already lives upstream.
#
# The one genuinely new technique here: resolving the CORRECT historical
# version of an SCD2 dimension (silver_customers, silver_products) for
# each fact row, instead of always joining to whatever is "current" today.
# A plain equi-join on customer_id would silently give every past order
# the customer's CURRENT loyalty_tier - wrong for historical reporting.
# The fix is a point-in-time range join: match the order's timestamp
# against the customer version whose __START_AT/__END_AT window contains it.

import dlt
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dimension tables
# MAGIC Thin, renamed pass-throughs of the silver dimension tables you
# MAGIC already built. No new logic here - dim_ tables exist so BI tools
# MAGIC (Power BI, etc.) see a clean, conventionally-named star schema
# MAGIC instead of "silver_customers" sitting alongside fact tables.

# COMMAND ----------


@dlt.table(
    name="dim_customers",
    comment="Customer dimension - full SCD2 history, customer_sk is the surrogate key.",
)
def dim_customers():
    return dlt.read("silver_customers").select(
        F.col("customer_sk"),
        F.col("customer_id"),
        F.col("customer_name"),
        F.col("email"),
        F.col("phone"),
        F.col("signup_date"),
        F.col("date_of_birth"),
        F.col("loyalty_tier"),
        F.col("__START_AT").alias("valid_from"),
        F.col("__END_AT").alias("valid_to"),
        F.col("__END_AT").isNull().alias("is_current"),
    )

# COMMAND ----------


@dlt.table(
    name="dim_products",
    comment="Product dimension - full SCD2 history, product_sk is the surrogate key.",
)
def dim_products():
    return dlt.read("silver_products").select(
        F.col("product_sk"),
        F.col("product_id"),
        F.col("sku"),
        F.col("product_name"),
        F.col("category"),
        F.col("category_id"),
        F.col("price"),
        F.col("__START_AT").alias("valid_from"),
        F.col("__END_AT").alias("valid_to"),
        F.col("__END_AT").isNull().alias("is_current"),
    )

# COMMAND ----------


@dlt.table(
    name="dim_channels",
    comment="Channel dimension - no history tracking, channels rarely change.",
)
def dim_channels():
    return dlt.read("silver_channels").select(
        F.col("channel_id"),
        F.col("channel_name"),
        F.col("channel_type"),
    )

# COMMAND ----------


@dlt.table(
    name="dim_date",
    comment="Standard date dimension, generated once, covering the range your fact data spans.",
)
def dim_date():
    orders = dlt.read("silver_orders")
    date_range = orders.select(
        F.min(F.to_date("created_at")).alias("min_date"),
        F.max(F.to_date("created_at")).alias("max_date"),
    ).collect()[0]

    return (
        spark.sql(
            f"SELECT explode(sequence(to_date('{date_range['min_date']}'), "
            f"to_date('{date_range['max_date']}'), interval 1 day)) AS full_date"
        )
        .withColumn("date_sk", F.date_format("full_date", "yyyyMMdd").cast("int"))
        .withColumn("year", F.year("full_date"))
        .withColumn("month", F.month("full_date"))
        .withColumn("day", F.dayofmonth("full_date"))
        .withColumn("day_of_week", F.dayofweek("full_date"))
        .withColumn("day_name", F.date_format("full_date", "EEEE"))
        .withColumn("month_name", F.date_format("full_date", "MMMM"))
        .withColumn("quarter", F.quarter("full_date"))
        .withColumn("is_weekend", F.dayofweek("full_date").isin([1, 7]))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### gold_orders_enriched
# MAGIC Orders + the customer version that was active AT THE TIME of the
# MAGIC order + channel details. This is a point-in-time join against the
# MAGIC SCD2 silver_customers table.

# COMMAND ----------


@dlt.table(
    name="gold_orders_enriched",
    comment="Orders enriched with the customer version active at order time, and channel details.",
)
def gold_orders_enriched():
    orders = dlt.read("silver_orders")
    customers = dlt.read("silver_customers")
    channels = dlt.read("silver_channels")

    # Point-in-time join: match each order to the customer row whose
    # __START_AT/__END_AT window actually contains the order's timestamp.
    # __END_AT IS NULL means "still the current version, no end yet" -
    # coalesce it to a far-future date so the BETWEEN-style comparison
    # below still works for currently-active versions.
    orders_with_customer = orders.join(
        customers,
        on=(
            (orders.customer_id == customers.customer_id)
            & (orders.created_at >= customers.__START_AT)
            & (orders.created_at < F.coalesce(customers.__END_AT, F.lit("9999-12-31")))
        ),
        how="left",
    )

    return (
        orders_with_customer.join(channels, orders.channel_id == channels.channel_id, "left")
        .select(
            orders.order_id,
            orders.customer_id,
            customers.customer_sk,
            customers.customer_name,
            customers.email,
            customers.loyalty_tier,  # the tier AT THE TIME of this order, not today's
            orders.product,
            orders.amount,
            orders.order_status,
            channels.channel_name,
            channels.channel_type,
            orders.created_at.alias("order_created_at"),
            orders.updated_at.alias("order_updated_at"),
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### gold_order_items_enriched
# MAGIC Order line items + the product version active at order time +
# MAGIC parent order context. Same point-in-time join technique, this time
# MAGIC against silver_products.

# COMMAND ----------


@dlt.table(
    name="gold_order_items_enriched",
    comment="Order line items enriched with the product version active at order time.",
)
def gold_order_items_enriched():
    order_items = dlt.read("silver_order_items")
    orders = dlt.read("silver_orders").select(
        F.col("order_id"), F.col("created_at").alias("order_created_at")
    )
    products = dlt.read("silver_products")

    # order_items has no timestamp of its own - borrow the parent order's
    # created_at to know WHEN this line item's product price should be
    # resolved from, then do the same point-in-time join as above.
    items_with_order_time = order_items.join(orders, "order_id", "left")

    items_with_product = items_with_order_time.join(
        products,
        on=(
            (items_with_order_time.product_id == products.product_id)
            & (items_with_order_time.order_created_at >= products.__START_AT)
            & (
                items_with_order_time.order_created_at
                < F.coalesce(products.__END_AT, F.lit("9999-12-31"))
            )
        ),
        how="left",
    )

    return items_with_product.select(
        items_with_order_time.order_item_id,
        items_with_order_time.order_id,
        items_with_order_time.product_id,
        products.product_sk,
        products.sku,
        products.product_name,
        products.category,
        items_with_order_time.quantity,
        products.price.alias("price_at_time_of_order"),
        items_with_order_time.discount_pct,
        (
            items_with_order_time.quantity
            * products.price
            * (1 - F.coalesce(items_with_order_time.discount_pct, F.lit(0)))
        ).alias("line_total"),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### gold_customer_order_summary
# MAGIC A business-ready aggregate: total orders and spend per customer,
# MAGIC using their CURRENT loyalty tier (this table answers "who are our
# MAGIC customers today", not a historical question, so it deliberately
# MAGIC reads only the currently-active SCD2 version, not a point-in-time
# MAGIC join - a different, equally valid choice depending on the question
# MAGIC being asked).

# COMMAND ----------


@dlt.table(
    name="gold_customer_order_summary",
    comment="Total orders and spend per customer, using each customer's CURRENT profile.",
)
def gold_customer_order_summary():
    orders = dlt.read("gold_orders_enriched")
    current_customers = dlt.read("silver_customers").filter(F.col("__END_AT").isNull())

    return (
        orders.groupBy("customer_id")
        .agg(
            F.count("*").alias("total_orders"),
            F.sum("amount").alias("total_spend"),
            F.avg("amount").alias("avg_order_value"),
            F.max("order_created_at").alias("last_order_at"),
        )
        .join(
            current_customers.select("customer_id", "customer_name", "loyalty_tier"),
            "customer_id",
            "left",
        )
        .select(
            "customer_id",
            "customer_name",
            "loyalty_tier",
            "total_orders",
            "total_spend",
            "avg_order_value",
            "last_order_at",
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### gold_daily_revenue
# MAGIC A time-series aggregate: daily order volume and revenue by channel -
# MAGIC the kind of table a BI dashboard would query directly.

# COMMAND ----------


@dlt.table(
    name="gold_daily_revenue",
    comment="Daily order count and revenue, broken down by channel.",
)
def gold_daily_revenue():
    orders = dlt.read("gold_orders_enriched")

    return (
        orders.withColumn("order_date", F.to_date("order_created_at"))
        .groupBy("order_date", "channel_name", "channel_type")
        .agg(
            F.count("*").alias("num_orders"),
            F.sum("amount").alias("total_revenue"),
            F.avg("amount").alias("avg_order_value"),
        )
        .orderBy("order_date")
    )
