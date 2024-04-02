import pandas as pd
from statsmodels.tsa.stattools import acf, pacf


def autocorrelation(series: pd.Series) -> dict:
    return {
        "lags": {
            i: val
            for i, val in enumerate(
                list(
                    acf(
                        series,
                        adjusted=False,
                        fft=True,
                        missing="none",
                    )
                )
            )
        },
    }


def partial_autocorrelation(series: pd.Series) -> dict:
    return {
        "lags": {i: val for i, val in enumerate(list(pacf(series)))},
    }
