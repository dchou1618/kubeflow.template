from pandas import Series
from numpy import dot

def dot_product(col1: Series, 
                col2: Series) -> Series:
    """
    Compute dot product between two vector columns
    """
    return col1.combine(col2, lambda v1, v2: float(dot(v1, v2)))