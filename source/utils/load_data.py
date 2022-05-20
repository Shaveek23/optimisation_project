import pandas as pd
from pandas import DataFrame
from sklearn.datasets import make_regression


def get_high_dimensional_artificial_ds(n_samples: int = 10_000, n_features: int = 5_000, n_informative: int = 1_000,
                                       bias: float = 0.0) -> tuple[DataFrame, DataFrame]:

    X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative, bias=bias)
    return DataFrame(X), DataFrame(y)
