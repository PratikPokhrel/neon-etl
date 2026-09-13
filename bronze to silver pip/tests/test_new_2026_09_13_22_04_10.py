import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    DecimalType, DateType, TimestampType
)
from datetime import datetime, date
from decimal import Decimal


@pytest.fixture(scope="session")
def spark():
    """Create a Spark session for testing."""
    return SparkSession.builder \
        .appName("pipeline_tests") \
        .getOrCreate()


class TestCustomersTransformation:
    """Test suite for customers transformation logic."""
    
    def test_customers_cleaning_and_standardization(self, spark):
        """
        Test that customer data is properly cleaned:
        - name is trimmed
        - email is lowercased and trimmed
        - dates are cast correctly
        - loyalty_tier defaults to 'Bronze' when null
        - customer_sk surrogate key is generated correctly
        """
        # Input data with messy values
        input_data = [
            (1, "  John Doe  ", "  JOHN@EXAMPLE.COM  ", "2024-01-01", "555-1234", 
             "1990-05-15", "Gold", "2024-01-01 10:00:00"),
            (2, "Jane Smith", "jane@example.com", "2024-01-02", "555-5678", 
             "1985-08-20", None, "2024-01-02 11:00:00"),  # null loyalty_tier
            (3, "Bob Wilson", "BOB@TEST.COM", "2024-01-03", "555-9999", 
             "1995-03-10", "Silver", "2024-01-03 12:00:00"),
        ]
        
        input_schema = StructType([
            StructField("customer_id", IntegerType(), True),
            StructField("name", StringType(), True),
            StructField("email", StringType(), True),
            StructField("signup_date", StringType(), True),
            StructField("phone", StringType(), True),
            StructField("date_of_birth", StringType(), True),
            StructField("loyalty_tier", StringType(), True),
            StructField("updated_at", StringType(), True),
        ])
        
        input_df = spark.createDataFrame(input_data, input_schema)
        
        # Apply the transformation logic (extracted from customers_cleaned view)
        result_df = input_df.select(
            F.col("customer_id"),
            F.trim(F.col("name")).alias("customer_name"),
            F.lower(F.trim(F.col("email"))).alias("email"),
            F.col("signup_date").cast("date"),
            F.col("phone"),
            F.col("date_of_birth").cast("date"),
            F.coalesce(F.col("loyalty_tier"), F.lit("Bronze")).alias("loyalty_tier"),
            F.col("updated_at").cast("timestamp"),
        ).withColumn(
            "customer_sk",
            F.sha2(F.concat_ws("||", F.col("customer_id"), F.col("loyalty_tier")), 256),
        )
        
        result = result_df.collect()
        
        # Assertions
        assert len(result) == 3
        
        # Test customer 1 - name trimmed, email lowercased
        assert result[0]["customer_name"] == "John Doe"
        assert result[0]["email"] == "john@example.com"
        assert result[0]["loyalty_tier"] == "Gold"
        assert result[0]["customer_sk"] is not None
        
        # Test customer 2 - null loyalty_tier defaults to Bronze
        assert result[1]["customer_name"] == "Jane Smith"
        assert result[1]["loyalty_tier"] == "Bronze"
        
        # Test customer 3 - email lowercased
        assert result[2]["email"] == "bob@test.com"
        assert result[2]["loyalty_tier"] == "Silver"
        
        # Verify surrogate keys are unique and deterministic
        sk_1 = result[0]["customer_sk"]
        sk_2 = result[1]["customer_sk"]
        sk_3 = result[2]["customer_sk"]
        assert len({sk_1, sk_2, sk_3}) == 3  # All three are unique
        
    def test_customer_surrogate_key_consistency(self, spark):
        """
        Test that customer_sk is deterministic - same customer_id + loyalty_tier 
        always produces the same surrogate key.
        """
        input_data = [
            (1, "John", "Gold", "2024-01-01 10:00:00"),
            (1, "John Updated", "Gold", "2024-01-02 10:00:00"),  # Same tier
            (1, "John Final", "Platinum", "2024-01-03 10:00:00"),  # Different tier
        ]
        
        input_schema = StructType([
            StructField("customer_id", IntegerType(), True),
            StructField("name", StringType(), True),
            StructField("loyalty_tier", StringType(), True),
            StructField("updated_at", StringType(), True),
        ])
        
        input_df = spark.createDataFrame(input_data, input_schema)
        
        result_df = input_df.withColumn(
            "customer_sk",
            F.sha2(F.concat_ws("||", F.col("customer_id"), F.col("loyalty_tier")), 256),
        )
        
        result = result_df.collect()
        
        # First two records have same tier -> same surrogate key
        assert result[0]["customer_sk"] == result[1]["customer_sk"]
        
        # Third record has different tier -> different surrogate key
        assert result[0]["customer_sk"] != result[2]["customer_sk"]


class TestProductsTransformation:
    """Test suite for products transformation logic."""
    
    def test_products_cleaning_and_standardization(self, spark):
        """
        Test that product data is properly cleaned:
        - product name is trimmed
        - price is cast to decimal(10,2)
        - product_sk surrogate key is generated correctly
        """
        input_data = [
            (101, "SKU001", "  Laptop  ", "Electronics", 1, "999.99", "2024-01-01 10:00:00"),
            (102, "SKU002", "Mouse", "Accessories", 2, "25.50", "2024-01-02 11:00:00"),
            (103, "SKU003", "Keyboard", "Accessories", 2, "75.00", "2024-01-03 12:00:00"),
        ]
        
        input_schema = StructType([
            StructField("product_id", IntegerType(), True),
            StructField("sku", StringType(), True),
            StructField("name", StringType(), True),
            StructField("category", StringType(), True),
            StructField("category_id", IntegerType(), True),
            StructField("price", StringType(), True),
            StructField("updated_at", StringType(), True),
        ])
        
        input_df = spark.createDataFrame(input_data, input_schema)
        
        # Apply the transformation logic (extracted from products_cleaned view)
        result_df = input_df.select(
            F.col("product_id"),
            F.col("sku"),
            F.trim(F.col("name")).alias("product_name"),
            F.col("category"),
            F.col("category_id"),
            F.col("price").cast("decimal(10,2)"),
            F.col("updated_at").cast("timestamp"),
        ).withColumn(
            "product_sk",
            F.sha2(F.concat_ws("||", F.col("product_id"), F.col("price").cast("string")), 256),
        )
        
        result = result_df.collect()
        
        # Assertions
        assert len(result) == 3
        
        # Test product name trimming
        assert result[0]["product_name"] == "Laptop"
        
        # Test price casting to decimal
        assert result[0]["price"] == Decimal("999.99")
        assert result[1]["price"] == Decimal("25.50")
        assert result[2]["price"] == Decimal("75.00")
        
        # Verify surrogate keys are generated
        assert result[0]["product_sk"] is not None
        assert result[1]["product_sk"] is not None
        assert result[2]["product_sk"] is not None
        
    def test_product_price_change_generates_new_sk(self, spark):
        """
        Test that when a product's price changes, it generates a different 
        surrogate key (for SCD2 tracking).
        """
        input_data = [
            (101, "999.99"),
            (101, "899.99"),  # Price changed
        ]
        
        input_schema = StructType([
            StructField("product_id", IntegerType(), True),
            StructField("price", StringType(), True),
        ])
        
        input_df = spark.createDataFrame(input_data, input_schema)
        
        result_df = input_df.withColumn(
            "product_sk",
            F.sha2(F.concat_ws("||", F.col("product_id"), F.col("price")), 256),
        )
        
        result = result_df.collect()
        
        # Different prices should generate different surrogate keys
        assert result[0]["product_sk"] != result[1]["product_sk"]


class TestOrderItemsTransformation:
    """Test suite for order_items transformation logic."""
    
    def test_order_items_standardization(self, spark):
        """
        Test that order items are properly standardized:
        - unit_price is cast to decimal(10,2)
        - discount_pct defaults to 0 when null
        """
        input_data = [
            (1, 100, 101, 2, "99.99", "10"),
            (2, 100, 102, 1, "25.50", None),  # null discount
            (3, 101, 103, 3, "75.00", "5"),
        ]
        
        input_schema = StructType([
            StructField("order_item_id", IntegerType(), True),
            StructField("order_id", IntegerType(), True),
            StructField("product_id", IntegerType(), True),
            StructField("quantity", IntegerType(), True),
            StructField("unit_price", StringType(), True),
            StructField("discount_pct", StringType(), True),
        ])
        
        input_df = spark.createDataFrame(input_data, input_schema)
        
        # Apply the transformation logic
        result_df = input_df.select(
            F.col("order_item_id"),
            F.col("order_id"),
            F.col("product_id"),
            F.col("quantity"),
            F.col("unit_price").cast("decimal(10,2)"),
            F.coalesce(F.col("discount_pct"), F.lit(0)).alias("discount_pct"),
        )
        
        result = result_df.collect()
        
        # Assertions
        assert len(result) == 3
        
        # Test unit_price casting
        assert result[0]["unit_price"] == Decimal("99.99")
        assert result[1]["unit_price"] == Decimal("25.50")
        
        # Test discount_pct default value
        # Note: coalesce with F.lit(0) converts string column to integer
        assert result[0]["discount_pct"] == 10
        assert result[1]["discount_pct"] == 0  # null defaulted to 0
        assert result[2]["discount_pct"] == 5


class TestAddressesTransformation:
    """Test suite for addresses transformation logic."""
    
    def test_addresses_standardization_with_defaults(self, spark):
        """
        Test that addresses are properly standardized:
        - address_type defaults to 'shipping' when null
        - created_at is cast to timestamp
        """
        input_data = [
            (1, 100, "billing", "123 Main St", "New York", "NY", "10001", "USA", "2024-01-01 10:00:00"),
            (2, 101, None, "456 Oak Ave", "Los Angeles", "CA", "90001", "USA", "2024-01-02 11:00:00"),  # null type
            (3, 102, "shipping", "789 Pine Rd", "Chicago", "IL", "60601", "USA", "2024-01-03 12:00:00"),
        ]
        
        input_schema = StructType([
            StructField("address_id", IntegerType(), True),
            StructField("customer_id", IntegerType(), True),
            StructField("address_type", StringType(), True),
            StructField("street", StringType(), True),
            StructField("city", StringType(), True),
            StructField("state", StringType(), True),
            StructField("zip_code", StringType(), True),
            StructField("country", StringType(), True),
            StructField("created_at", StringType(), True),
        ])
        
        input_df = spark.createDataFrame(input_data, input_schema)
        
        # Apply the transformation logic
        result_df = input_df.select(
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
        
        result = result_df.collect()
        
        # Assertions
        assert len(result) == 3
        
        # Test address_type default
        assert result[0]["address_type"] == "billing"
        assert result[1]["address_type"] == "shipping"  # null defaulted to 'shipping'
        assert result[2]["address_type"] == "shipping"
        
        # Test timestamp casting
        assert isinstance(result[0]["created_at"], datetime)
        assert isinstance(result[1]["created_at"], datetime)


class TestPaymentsTransformation:
    """Test suite for payments transformation logic."""
    
    def test_payments_standardization_with_defaults(self, spark):
        """
        Test that payments are properly standardized:
        - amount is cast to decimal(10,2)
        - payment_status defaults to 'completed' when null
        - paid_at is cast to timestamp
        """
        input_data = [
            (1, 100, "150.00", "credit_card", "completed", "2024-01-01 10:00:00"),
            (2, 101, "250.50", "paypal", None, "2024-01-02 11:00:00"),  # null status
            (3, 102, "99.99", "debit_card", "pending", "2024-01-03 12:00:00"),
        ]
        
        input_schema = StructType([
            StructField("payment_id", IntegerType(), True),
            StructField("order_id", IntegerType(), True),
            StructField("amount", StringType(), True),
            StructField("method", StringType(), True),
            StructField("status", StringType(), True),
            StructField("paid_at", StringType(), True),
        ])
        
        input_df = spark.createDataFrame(input_data, input_schema)
        
        # Apply the transformation logic
        result_df = input_df.select(
            F.col("payment_id"),
            F.col("order_id"),
            F.col("amount").cast("decimal(10,2)"),
            F.col("method"),
            F.coalesce(F.col("status"), F.lit("completed")).alias("payment_status"),
            F.col("paid_at").cast("timestamp"),
        )
        
        result = result_df.collect()
        
        # Assertions
        assert len(result) == 3
        
        # Test amount casting
        assert result[0]["amount"] == Decimal("150.00")
        assert result[1]["amount"] == Decimal("250.50")
        assert result[2]["amount"] == Decimal("99.99")
        
        # Test payment_status default
        assert result[0]["payment_status"] == "completed"
        assert result[1]["payment_status"] == "completed"  # null defaulted to 'completed'
        assert result[2]["payment_status"] == "pending"
        
        # Test timestamp casting
        assert isinstance(result[0]["paid_at"], datetime)
        assert isinstance(result[1]["paid_at"], datetime)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])