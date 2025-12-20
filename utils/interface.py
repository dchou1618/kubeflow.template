from pyspark.sql import Column
from typing import Union
from pandas import Series

def dot_product(col1: Union[Column, Series], 
                col2: Union[Column, Series]) -> Union[Column, Series]:
    """
    Compute dot product between two vector columns
    """
    if isinstance(col1, Column) and isinstance(col2, Column):
        from utils.spark.vector import dot_product as spark_dot_product
        return spark_dot_product(col1, col2)
    elif isinstance(col1, Series) and isinstance(col2, Series):
        from utils.tabular.vector import dot_product as pandas_dot_product
        return pandas_dot_product(col1, col2)
    else:
        raise TypeError("Both arguments must be either Spark Columns or Pandas Series")