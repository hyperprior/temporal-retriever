from darts.metrics import mape
from darts import TimeSeries
from darts.models import Prophet


import pandas as pd
import numpy as np
from fastapi import FastAPI, status

from temporal_retriever.requests import AnalysisRequest

from darts.utils.statistics import granger_causality_tests, remove_trend
from icecream import ic


app: FastAPI = FastAPI()


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    return


def predict(
    series: TimeSeries,
    quantiles: list[tuple] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
    prediction_horizon: int = 14,
    num_samples: int = 1000,
) -> dict:
    model = Prophet(
        add_seasonalities={
            "name": "quarterly_seasonality",
            "seasonal_periods": 4,
            "fourier_order": 5,
        },
        add_encoders={
            "cyclic": {"future": ["month", "week"]},
            "datetime_attribute": {"future": ["dayofweek"]},
            "position": {"future": ["relative"]},
        },
    )

    model.fit(series)

    forecast = model.predict(n=prediction_horizon,
                             num_samples=num_samples).all_values()
    predictions = pd.DataFrame(
        np.quantile(forecast, quantiles, axis=-1)
        .reshape(len(quantiles), prediction_horizon)
        .T
    )

    predictions.columns = [f"q{quant}" for quant in quantiles]

    return predictions.to_dict(orient="records")

    # backtests = model.historical_forecasts(
    # series, start=0.5, forecast_horizon=prediction_horizon, overlap_end=True
    # )
    # backtest_mape = mape(backtests, series)
    # if backtest_mape is np.nan:
    # backtest_mape = None


def correlate(from_series: TimeSeries, to_series: TimeSeries, max_lag: int = 14):
    return granger_causality_tests(
        remove_trend(from_series), remove_trend(to_series), maxlag=max_lag
    )


@app.post("/analyze")
async def analyze_datasets(request: AnalysisRequest):
    correlations = request.get("analyticsOptions").get("correlations")

    output = {}

    for correlation in correlations:
        covariate_name: str = correlation.get("fromIndex")
        target_name: str = correlation.get("toIndex")

        covariates = pd.DataFrame(request.get(correlation["fromData"])["data"])[
            ["date", covariate_name]
        ]
        covariates["date"] = pd.to_datetime(covariates["date"])

        targets = pd.DataFrame(request.get(correlation["toData"], {}).get("data", {}))[
            ["date", target_name]
        ]
        targets["date"] = pd.to_datetime(targets["date"])

        covariates: TimeSeries = TimeSeries.from_dataframe(
            covariates,
            time_col="date",
            value_cols=covariate_name,
            fill_missing_dates=True,
        )
        targets: TimeSeries = TimeSeries.from_dataframe(
            targets, time_col="date", value_cols=target_name, fill_missing_dates=True
        )

        output[correlation.get("id")] = {
            "correlations": {
                "granger_causality": {
                    "description": "Statistical hypothesis test to determine if one time series causes X has a predictive relationship in the future to time series Y.",
                    "values": correlate(covariates, targets, max_lag=2),
                }
            },
            "predictions": {
                "from": {
                    "fromData": correlation.get("fromData"),
                    "fromIndex": correlation.get("fromIndex"),
                    "values": predict(covariates, prediction_horizon=1),
                },
                "to": {
                    "toData": correlation.get("toData"),
                    "toIndex": correlation.get("toIndex"),
                    "values": predict(targets, prediction_horizon=1),
                },
            },
        }

        # forecast = forecast.rename(columns=cols)[list(cols.values())]
        # print(dir(model))
        # print(model.seasonalities)
        # print(model.changepoints)

    return output
