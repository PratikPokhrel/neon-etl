# Databricks notebook source
# This notebook is a Declarative Pipeline (DLT) source file - it is NOT run
# directly like a normal notebook. It must be attached to a Pipeline
# (Workflows > Pipelines > Create Pipeline > point it at this notebook),
# and the pipeline itself is what executes it.
#
# Compared to the hand-rolled Structured Streaming version:
# - No manual checkpointLocation / schemaLocation - DLT manages both
#   internally per table.
# - No manual try/except loop - if one table's query fails, DLT reports it
#   in the pipeline UI without needing hand-written error tracking.
# - No manual "for table in TABLES: run it" orchestration code needed for
#   execution order - DLT infers the dependency graph from your functions
#   (not needed here since these are independent leaf tables, but matters
#   once silver tables read from these bronze tables).
# - Auto Loader (cloudFiles) is used identically underneath - same source,
#   same mechanism, just wrapped in @dlt.table instead of writeStream.

import dlt
from pyspark.sql import functions as F

LANDING_VOLUME = "/Volumes/main/landing/raw_files"

# Tables landed incrementally (append-only files in the landing volume) -
# a plain streaming bronze table is the right fit for these.
INCREMENTAL_TABLES = [
    "customers", "products", "orders", "addresses", "order_items",
    "order_status_history", "reviews",
]

# Tables the extraction job fully overwrites each run (payments, shipments,
# categories, channels - no reliable updated_at column to go incremental
# against). A streaming source would get confused by files disappearing
# and reappearing on overwrite, so these use dlt.table as a BATCH read
# instead of readStream - DLT will just re-read and replace the table's
# contents each pipeline run, which matches how the source data behaves.
FULL_REFRESH_TABLES = ["payments", "shipments", "categories", "channels"]

# COMMAND ----------

# Incremental bronze tables - one @dlt.table per source table, generated
# in a loop. Note the default-argument trick (table_name=table_name) -
# without it, every closure would capture the SAME final loop variable
# value instead of the value at the time each function was defined.


def make_incremental_bronze_table(table_name):
    @dlt.table(
        name=f"bronze_{table_name}",
        comment=f"Incremental bronze table for {table_name}, loaded via Auto Loader.",
    )
    def _bronze_table():
        return (
            spark.readStream.format("cloudFiles")
            .option("cloudFiles.format", "parquet")
            .option("cloudFiles.inferColumnTypes", "true")
            .load(f"{LANDING_VOLUME}/{table_name}")
            .withColumn("_bronze_loaded_at", F.current_timestamp())
        )
    return _bronze_table


for table_name in INCREMENTAL_TABLES:
    make_incremental_bronze_table(table_name)

# COMMAND ----------

# Full-refresh bronze tables - plain batch read, re-landed in full each run.
def make_full_refresh_bronze_table(table_name):
    @dlt.table(
        name=f"bronze_{table_name}",
        comment=f"Full-refresh bronze table for {table_name} (source has no reliable change-tracking column).",
    )
    def _bronze_table():
        return (
            spark.read.format("parquet")
            .load(f"{LANDING_VOLUME}/{table_name}")
            .withColumn("_bronze_loaded_at", F.current_timestamp())
        )
    return _bronze_table

for table_name in FULL_REFRESH_TABLES:
    make_full_refresh_bronze_table(table_name)