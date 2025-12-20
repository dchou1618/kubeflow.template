# tests/test_dot_product_pandas.py

import pandas as pd
from utils.interface import dot_product


def test_dot_product_pandas():
    df = pd.DataFrame({
        "vec1": [
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
        ],
        "vec2": [
            [4.0, 5.0, 6.0],
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 0.0],
        ]
    })

    result = dot_product(df["vec1"], df["vec2"])

    expected = pd.Series([32.0, 2.0, 0.0])
    pd.testing.assert_series_equal(result, expected)
