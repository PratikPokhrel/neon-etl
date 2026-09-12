# Databricks notebook source
# Extract job: pulls each OLTP table out of the Neon Postgres source via JDBC
# and lands it as Parquet files in a Unity Catalog Volume.
#
# Fixes from the first version:
# 1. shipments/payments switched to "full" mode - they have no updated_at
#    column, so incremental-by-id would silently miss status changes
#    (shipped_at, delivered_at, refund status, etc.)
# 2. df.cache() - avoids re-running the same JDBC query 3 times per table
# 3. try/except per table - one bad table no longer kills the whole job
# 4. Known limitation still not solved here: deletes are invisible to this
#    approach entirely, on every table, regardless of mode. Watermark-based
#    extraction can only ever see inserts/updates, never deletes. That's
#    the core reason real CDC exists - worth remembering even though we're
#    sticking with this approach for now.

# COMMAND ----------

from pyspark.sql import functions as F
from urllib.parse import urlparse

connection_url = dbutils.secrets.get(scope="neon_oltp", key="connection_url")
parsed = urlparse(connection_url)
jdbc_url = f"jdbc:postgresql://{parsed.hostname}:{parsed.port or 5432}{parsed.path}?sslmode=require"
connection_properties = {
    "user": parsed.username,
    "password": parsed.password,
    "driver": "org.postgresql.Driver",
}

LANDING_VOLUME = "/Volumes/main/landing/raw_files"
WATERMARK_TABLE = "main.landing.extract_watermarks"

# mode: "updated_at" (incremental by timestamp column - safe only if EVERY
# mutation touches that column), "id" (incremental by monotonically
# increasing PK - safe ONLY for tables that are pure insert-only, never
# updated after creation), or "full" (always fully re-landed - the safe
# default whenever you're not 100% sure a table's changes are all
# captured by one column).
TABLE_CONFIG = {
    "customers": {"mode": "updated_at", "id_col": "customer_id", "watermark_col": "updated_at"},
    "products": {"mode": "updated_at", "id_col": "product_id", "watermark_col": "updated_at"},
    "orders": {"mode": "updated_at", "id_col": "order_id", "watermark_col": "updated_at"},
    "addresses": {"mode": "id", "id_col": "address_id"},          # insert-only, safe
    "order_items": {"mode": "id", "id_col": "order_item_id"},     # insert-only, safe
    "order_status_history": {"mode": "id", "id_col": "history_id"},  # insert-only, safe
    "reviews": {"mode": "id", "id_col": "review_id"},              # insert-only, safe
    "payments": {"mode": "full"},     # FIXED: status can change, no updated_at
    "shipments": {"mode": "full"},    # FIXED: ship_status/delivered_at change, no updated_at
    "categories": {"mode": "full"},
    "channels": {"mode": "full"},
}

# COMMAND ----------

spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {WATERMARK_TABLE} (
        table_name STRING,
        watermark_value STRING,
        extracted_at TIMESTAMP
    ) USING DELTA
    """
)


def get_watermark(table_name):
    row = (
        spark.table(WATERMARK_TABLE)
        .filter(F.col("table_name") == table_name)
        .select("watermark_value")
        .collect()
    )
    return row[0]["watermark_value"] if row else None


def set_watermark(table_name, value):
    spark.sql(
        f"""
        MERGE INTO {WATERMARK_TABLE} t
        USING (SELECT '{table_name}' AS table_name, '{value}' AS watermark_value, current_timestamp() AS extracted_at) s
        ON t.table_name = s.table_name
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        """
    )


# COMMAND ----------


def extract_table(table_name, config):
    # NOTE: .cache()/.persist() are not supported on serverless compute
    # (PERSIST TABLE not supported error). Instead of caching in memory,
    # we write straight to Parquet first, then read that same written
    # data back to compute counts/watermarks. This still avoids re-running
    # the JDBC query against Postgres multiple times - the expensive part -
    # it just uses disk as the "cache" instead of RAM.
    query = f"(SELECT * FROM {table_name}) AS src"
    out_path = f"{LANDING_VOLUME}/{table_name}"

    if config["mode"] == "full":
        df = spark.read.jdbc(url=jdbc_url, table=query, properties=connection_properties)
        df.write.mode("overwrite").parquet(out_path)

        written = spark.read.parquet(out_path)
        row_count = written.count()
        print(f"{table_name}: full re-land, {row_count} rows")
        return

    watermark_col = config.get("watermark_col", config["id_col"])
    watermark = get_watermark(table_name)

    if watermark is not None:
        predicate = (
            f"{watermark_col} > '{watermark}'"
            if config["mode"] == "updated_at"
            else f"{watermark_col} > {watermark}"
        )
        query = f"(SELECT * FROM {table_name} WHERE {predicate}) AS src"

    df = spark.read.jdbc(url=jdbc_url, table=query, properties=connection_properties)

    # Write to a staging path first so we can count rows without a second
    # JDBC read - if it's empty, clean up and treat as "nothing new".
    staging_path = f"{out_path}/_staging_{table_name}"
    df.withColumn("_extracted_at", F.current_timestamp()).write.mode("overwrite").parquet(staging_path)

    staged = spark.read.parquet(staging_path)
    row_count = staged.count()

    if row_count == 0:
        dbutils.fs.rm(staging_path, recurse=True)
        print(f"{table_name}: no new rows since watermark={watermark}")
        return

    staged.write.mode("append").parquet(out_path)
    dbutils.fs.rm(staging_path, recurse=True)

    new_watermark = staged.agg(F.max(watermark_col)).collect()[0][0]
    set_watermark(table_name, str(new_watermark))
    print(f"{table_name}: landed {row_count} rows, watermark now {new_watermark}")


# COMMAND ----------

results = {"success": [], "failed": []}

for table_name, config in TABLE_CONFIG.items():
    try:
        extract_table(table_name, config)
        results["success"].append(table_name)
    except Exception as e:
        print(f"FAILED: {table_name} - {e}")
        results["failed"].append((table_name, str(e)))

print(f"\nDone. {len(results['success'])} succeeded, {len(results['failed'])} failed.")
if results["failed"]:
    for table_name, err in results["failed"]:
        print(f"  - {table_name}: {err}")