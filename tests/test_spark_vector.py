import pytest
from pyspark.sql import SparkSession, functions as F
from utils.interface import dot_product


@pytest.fixture(scope="module")
def spark():
    spark_session = (
        SparkSession.builder
        .appName("DotProductTest")
        .master("local[*]")
        .getOrCreate()
    )
    yield spark_session
    spark_session.stop()


def test_dot_product_spark(spark):
    data = [
        ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
        ([0.0, 1.0, 0.0], [1.0, 2.0, 3.0]),
        ([2.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
    ]

    df = spark.createDataFrame(data, schema=["vec1", "vec2"])

    df = df.withColumn(
        "dot_prod",
        dot_product(F.col("vec1"), F.col("vec2"))
    )

    results = [row.dot_prod for row in df.select("dot_prod").collect()]
    expected = [32.0, 2.0, 0.0]

    assert results == expected
