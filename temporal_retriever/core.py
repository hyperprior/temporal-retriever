from typing import Literal
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


def reset_time_index(
    *,
    series: pd.Series,
    format: Literal["ISO8601", "mixed"] = "ISO8601",
    grain: Literal["D", "W", "M", "H", "m"] | None = None,
):
    if not grain:
        return pd.to_datetime(series, format=format, utc=True)

    match grain:
        case "D":
            return pd.to_datetime(series, format=format, utc=True).dt.date
        case "W":
            return (
                pd.to_datetime(series, format=format, utc=True)
                .dt.to_period("W")
                .dt.end_time
            )
        case "M":
            return (
                pd.to_datetime(series, format=format, utc=True)
                .dt.to_period("M")
                .dt.end_time
            )
        case "H":
            return pd.to_datetime(series, format=format, utc=True).dt.floor("H")
        case "m":
            return pd.to_datetime(series, format=format, utc=True).dt.floor("T")
        case _:
            raise ValueError(f"Unsupported granularity: {grain}")
