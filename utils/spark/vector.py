from pyspark.sql import functions as F
from pyspark.sql import Column

def dot_product(col1: Column, 
                col2: Column) -> Column:
    """
    Compute dot product between two vector columns
    """
    return F.aggregate(
        F.arrays_zip(col1.alias("v1"), col2.alias("v2")),
        F.lit(0.0),
        lambda acc, x: acc + x.v1 * x.v2
    )