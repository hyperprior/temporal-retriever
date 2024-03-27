from typing import Literal
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.utils.statistics import granger_causality_tests, remove_trend
from fastapi import FastAPI, status
from pydash import get
from prophet import Prophet
from prophet.utilities import regressor_coefficients
from pydantic import BaseModel, Field, conint


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

    forecast = model.predict(n=prediction_horizon, num_samples=num_samples).all_values()
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
    if not (len(from_series) >= 14 and len(to_series) >= 14):
        return None
    return granger_causality_tests(
        remove_trend(from_series), remove_trend(to_series), maxlag=max_lag
    )


class Correlation(BaseModel):
    id: str
    type: Literal["prophet", "granger"] = "prophet"  # TODO make literals
    from_data: str = Field(..., alias="fromData")
    from_index: str = Field(..., alias="fromIndex")
    to_data: str = Field(..., alias="toData")
    to_index: str = Field(..., alias="toIndex")
    frequency: Literal["D", "W", "H", "m"] = Field(
        "D",
        description="granularity of the dataset, will be used for forecasting and aggregating raw data so that there are no overlaps in the time index",
    )
    aggregate: Literal["sum", "min", "max", "mean", "meadian"] = Field(
        "sum",
        description="to avoid duplicates in the time index, will aggregate the values by the `freq` field using the supplied operation",
    )
    prediction_horizon: conint(ge=1) = 7
    quantiles: list[tuple] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


class AnalyticsOptions(BaseModel):
    correlations: list[Correlation]


class AnalyticsRequest(BaseModel):
    documents: dict
    analytics_options: AnalyticsOptions = Field(..., alias="analyticsOptions")


@app.post("/analyze")
async def analyze_datasets(request: AnalyticsRequest):
    correlations = request.analytics_options.correlations

    output = {"correlations": {}}

    explanations = {
        "ds": "The datestamp or the date for which the prediction is made.",
        "trend": "The overall trend component of the time series at the specified date, capturing the long-term growth or decline.",
        "yhat_lower": "The lower bound of the prediction interval for the specified date.",
        "yhat_upper": "The upper bound of the prediction interval for the specified date.",
        "trend_lower": "The lower bound of the trend component.",
        "trend_upper": "The upper bound of the trend component.",
        "additive_terms": "The sum of all the additive components (such as seasonality and holidays) for the specified date.",
        "additive_terms_lower": "The lower bound of the additive terms.",
        "additive_terms_upper": "The upper bound of the additive terms.",
        "multiplicative_terms": "The multiplicative components (if any) of the model.",
        "multiplicative_terms_lower": "The lower bound of the multiplicative terms.",
        "multiplicative_terms_upper": "The upper bound of the multiplicative terms.",
        "yhat": "The final predicted value for the specified date, which is the sum of the trend, additive terms, and any extra regressors.",
        "extra_regressors_additive": "The sum of the contributions from all the extra regressors (if any) to the prediction.",
        "extra_regressors_additive_lower": "The lower bound of the extra regressors' additive contribution.",
        "extra_regressors_additive_upper": "The upper bound of the extra regressors' additive contribution.",
        "weekly": "The contribution of the weekly seasonality component to the prediction.",
        "weekly_lower": "The lower bound of the weekly seasonality component.",
        "weekly_upper": "The upper bound of the weekly seasonality component.",
    }

    for correlation in correlations:
        quantiles = correlation.quantiles
        prediction_horizon = correlation.prediction_horizon
        covariate_name: str = correlation.from_index

        from_data = [
            {"ds": data.get("date"), "y": get(data, covariate_name)}
            for data in request.documents.get(correlation.from_data).get("data")
        ]

        covariates = pd.DataFrame(from_data)
        covariates["ds"] = pd.to_datetime(covariates["ds"])
        covariates = (
            covariates.groupby("ds").agg({"y": correlation.aggregate}).reset_index()
        )

        covariate_model = Prophet()
        covariate_model.fit(covariates)
        covariate_future = covariate_model.make_future_dataframe(
            periods=prediction_horizon
        )

        covariate_predictions = covariate_model.predict(covariate_future)[
            ["ds", "yhat"]
        ]

        covariate_predictions = covariate_predictions.merge(
            covariates, how="left", on="ds"
        )
        covariate_predictions[covariate_name] = covariate_predictions[
            "y"
        ].combine_first(covariate_predictions["yhat"])

        covariate_predictions = covariate_predictions[["ds", covariate_name]]

        # predict(covariates, prediction_horizon=14, num_samples=1)

        target_name: str = correlation.to_index

        to_data = [
            {"ds": data.get("date"), "y": get(data, target_name)}
            for data in request.documents.get(correlation.to_data)["data"]
        ]

        targets = pd.DataFrame(to_data)
        targets = targets.groupby("ds").agg({"y": correlation.aggregate}).reset_index()
        targets["ds"] = pd.to_datetime(targets["ds"])

        targets = targets.merge(covariate_predictions, how="left", on="ds")

        target_model = Prophet()
        target_model.add_regressor(covariate_name)
        target_model.fit(targets)

        future = target_model.make_future_dataframe(periods=prediction_horizon).merge(
            covariate_predictions, on="ds"
        )  # .tail(prediction_horizon)

        target_forecast = target_model.predict(future)

        output["correlations"][correlation.id] = {
            "type": "prophet",
            "regressor_coefficients": regressor_coefficients(target_model).to_dict(
                orient="records"
            ),
            "predictions": target_forecast.to_dict(orient="records"),
        }

    return output
