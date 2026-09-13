# db-neon-etl

A Databricks ETL pipeline implementation using the Medallion Architecture (Bronze, Silver, Gold) for data processing and transformation.

## 📋 Overview

This project implements a multi-layer data pipeline that processes data through three distinct quality zones:
- **Landing → Bronze**: Raw data ingestion
- **Bronze → Silver**: Data cleansing and validation
- **Silver → Gold**: Business-level aggregations and dimensional modeling

## 🏗️ Project Structure

```
db-neon-etl/
├── landing to bronze pip/     # Raw data ingestion layer
│   └── transformations/
├── bronze-to-silver/          # Data cleansing layer
│   └── transformations/
├── bronze to silver pip/      # Alternative bronze-to-silver pipeline
│   └── transformations/
├── Silver to gold/            # Business logic layer
│   └── transformations/
├── neon-pipeline/             # Pipeline configurations
├── custom SCD 2 notebook      # SCD Type 2 implementation for customer dimension
└── README.md
```

## 🔄 Data Pipeline Layers

### Bronze Layer (Raw Data)
- Ingests raw data from source systems
- Minimal transformations
- Preserves original data with metadata (`_extracted_at`, `_bronze_loaded_at`)
- Stored in Unity Catalog: `main.bronze.*`

### Silver Layer (Cleansed Data)
- Data cleansing and standardization
- Deduplication and data quality checks
- Schema enforcement
- Stored in Unity Catalog: `main.bronze.silver_*` (transitional tables)

### Gold Layer (Business Data)
- Business-level aggregations
- Dimensional modeling (SCD Type 2)
- Optimized for analytics and reporting
- Stored in Unity Catalog: `main.gold.*`

## 📊 Key Features

### Slowly Changing Dimension (SCD) Type 2
Implemented in `custom SCD 2 notebook` for customer dimension:
- Tracks historical changes in customer attributes
- Maintains effective dates (`__START_AT`, `__END_AT`)
- Captures changes in customer information (name, email, phone, loyalty tier)
- Generates surrogate keys using SHA-256 hashing

### Data Processing
- **PySpark** for distributed data processing
- **Delta Lake** for ACID transactions and time travel
- **Unity Catalog** for data governance

## 🚀 Getting Started

### Prerequisites
- Databricks workspace
- Access to Unity Catalog (`main` catalog)
- Appropriate permissions on bronze, silver, and gold schemas

### Running the Pipeline

1. **Landing to Bronze**
   - Navigate to `landing to bronze pip/transformations/`
   - Execute notebooks to ingest raw data

2. **Bronze to Silver**
   - Navigate to `bronze-to-silver/transformations/`
   - Run cleansing and validation notebooks

3. **Silver to Gold**
   - Navigate to `Silver to gold/transformations/`
   - Execute dimensional modeling notebooks
   - Run `custom SCD 2 notebook` for customer dimension updates

## 📦 Tables

### Bronze Tables
- `main.bronze.bronze_customers` - Raw customer data

### Silver Tables
- `main.bronze.silver_customers` - Cleansed customer data with SCD Type 2

### Gold Tables
- `main.gold.dim_customers` - Customer dimension for analytics

## 🛠️ Technologies

- **Databricks**: Unified analytics platform
- **Apache Spark**: Distributed computing engine
- **Delta Lake**: ACID-compliant storage layer
- **Unity Catalog**: Data governance and management
- **Python/PySpark**: Primary development language

## 📝 Notes

- All transformations are idempotent and can be re-run safely
- The pipeline uses Delta Lake's MERGE operations for upserts
- SCD Type 2 implementation uses business timestamps (`updated_at`) for temporal tracking
- Data quality is maintained through deduplication using window functions

## 🤝 Contributing

When adding new transformations:
1. Follow the existing folder structure
2. Place transformation logic in the appropriate layer's `transformations/` folder
3. Maintain consistent naming conventions
4. Document any new tables in this README

## 📄 License

[Add your license information here]

## 👥 Contact

[Add contact information here]

