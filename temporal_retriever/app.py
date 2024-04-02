from datetime import datetime
from typing import Literal

import pandas as pd
from fastapi import FastAPI, status
from loguru import logger
from prophet import Prophet
from prophet.utilities import regressor_coefficients
from pydantic import BaseModel, Field, conint
from pydash import get

from temporal_retriever.core import (
    autocorrelation,
    partial_autocorrelation,
    reset_time_index,
)

from temporal_retriever.responses import AnalyticsResponse

app: FastAPI = FastAPI()


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    return


class Correlation(BaseModel):
    id: str
    type: Literal["prophet", "granger", "univariateStatistics"] = "prophet"
    from_data: str = Field(..., alias="fromData")
    from_index: str = Field(..., alias="fromIndex")
    to_data: str = Field(..., alias="toData")
    to_index: str = Field(..., alias="toIndex")
    changepoint_prior_scale: float = Field(0.8, alias="ChangePointPriorScale")
    grain: Literal["D", "W", "M", "H", "min"] = Field(
        "D",
        description="granularity of the dataset, will be used for forecasting and aggregating raw data so that there are no overlaps in the time index",
        alias="dataSetGranularity",
    )
    aggregation: Literal["sum", "min", "max", "mean", "meadian"] = Field(
        "sum",
        description="to avoid duplicates in the time index, will aggregate the values by the `freq` field using the supplied operation",
        alias="dataAggregationType",
    )
    prediction_horizon: conint(ge=1) | None = Field(
        None,
        description="How far into the future should we predict?",
        alias="unitsToForecast",
    )
    quantiles: list[tuple] = Field(
        (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
        description="The quantiles to use for interval predictions",
        alias="predictionQuantiles",
    )


class AnalyticsOptions(BaseModel):
    correlations: list[Correlation]


class AnalyticsRequest(BaseModel):
    documents: dict
    analytics_options: AnalyticsOptions = Field(..., alias="analyticsOptions")


def prepare_dataset(
    *,
    dataset: list[dict],
    time_column: str = "ds",
    aggregation: str = "sum",
    grain: Literal["D", "W", "M", "H", "m"] | None = None,
    prediction_horizon: int | None = None,
):
    dataframe = pd.DataFrame(dataset)
    try:
        dataframe[time_column] = reset_time_index(
            series=dataframe[time_column], grain=grain
        )
    except ValueError:
        logger.info("falling back to mixed date format")
        series = reset_time_index(
            series=dataframe[time_column], format="mixed", grain=grain
        )

    dataframe = dataframe.groupby(time_column).agg({"y": aggregation}).reset_index()

    prediction_horizon = prediction_horizon or len(dataframe["ds"])

    return dataframe, prediction_horizon


@app.post("/analyze")
async def analyze_datasets(request: AnalyticsRequest) -> AnalyticsResponse:
    correlations = request.analytics_options.correlations

    output = {"correlations": {}}

    for correlation in correlations:
        output["correlations"][correlation.id] = {"type": correlation.type}

        grain = correlation.grain
        aggregation = correlation.aggregation
        quantiles = correlation.quantiles
        covariate_name: str = correlation.from_index
        target_name: str = correlation.to_index

        from_data = [
            {"ds": data.get("date"), "y": get(data, covariate_name)}
            for data in request.documents.get(correlation.from_data).get("data")
        ]

        covariates, covariates_prediction_horizon = prepare_dataset(
            dataset=from_data,
            time_column="ds",
            aggregation=aggregation,
            grain=grain,
        )

        covariate_date_bounds = covariates["ds"].min(), covariates["ds"].max()

        covariate_model = Prophet(
            uncertainty_samples=1000,
            changepoint_prior_scale=correlation.changepoint_prior_scale,
            # seasonality_prior_scale=10.0,
            # holidays_prior_scale=10.0,
        )

        covariate_model.fit(covariates)
        covariate_future = covariate_model.make_future_dataframe(
            periods=covariates_prediction_horizon, freq=grain
        )

        covariate_future_dates = covariate_future["ds"].to_list()

        covariate_predictions = covariate_model.predict(covariate_future)[
            ["ds", "yhat"]
        ]

        covariate_predictions["ds"] = pd.to_datetime(covariate_predictions["ds"])

        covariates["ds"] = pd.to_datetime(covariates["ds"])

        covariate_predictions = covariate_predictions.merge(
            covariates, how="left", on="ds"
        )
        covariate_predictions[covariate_name] = covariate_predictions[
            "y"
        ].combine_first(covariate_predictions["yhat"])

        covariate_predictions = covariate_predictions[["ds", covariate_name]]

        to_data = [
            {"ds": data.get("date"), "y": get(data, target_name)}
            for data in request.documents.get(correlation.to_data)["data"]
        ]

        targets, targets_prediction_horizon = prepare_dataset(
            dataset=to_data,
            time_column="ds",
            aggregation=aggregation,
            grain=grain,
        )

        targets["ds"] = pd.to_datetime(targets["ds"])

        target_date_bounds = targets["ds"].min(), targets["ds"].max()

        targets = targets.merge(covariate_predictions, how="left", on="ds")

        target_model = Prophet(
            uncertainty_samples=1000,
            changepoint_prior_scale=correlation.changepoint_prior_scale,
            # seasonality_prior_scale=10.0,
            # holidays_prior_scale=10.0,
        )
        target_model.add_regressor(covariate_name)
        target_model.fit(targets)

        future = target_model.make_future_dataframe(
            periods=targets_prediction_horizon, freq=grain
        )

        future["ds"] = pd.to_datetime(future["ds"])

        future_dates = future["ds"].to_list()

        future = future.merge(covariate_predictions, on="ds")

        target_forecast = target_model.predict(future).rename(
            columns={
                "ds": "date",
                "yhat": "prediction",
                "yhat_lower": "prediction_lower_bound",
                "yhat_upper": "prediction_upper_bound",
                "trend_lower": "trend_lower_bound",
                "trend_upper": "trend_upper_bound",
            }
        )

        historical_forecast_dates = future_dates[:targets_prediction_horizon]
        future_forecast_dates = future_dates[targets_prediction_horizon:]

        historical_forecast = target_forecast[
            target_forecast["date"].isin(historical_forecast_dates)
        ]
        future_forecast = target_forecast[
            target_forecast["date"].isin(future_forecast_dates)
        ]

        output["correlations"][correlation.id]["diagnostics"] = {
            "units": grain,
            "from": {
                "data": correlation.from_data,
                "index": correlation.from_index,
                "minDate": covariate_date_bounds[0],
                "maxDate": covariate_date_bounds[1],
                "unitsForecasted": covariates_prediction_horizon,
            },
            "to": {
                "data": correlation.to_data,
                "index": correlation.to_index,
                "minDate": target_date_bounds[0],
                "maxDate": target_date_bounds[1],
                "unitsForecasted": targets_prediction_horizon,
            },
        }

        output["correlations"][correlation.id]["autocorrelations"] = {
            "description": "Autocorrelation measures the correlation between a time series and its lagged values. It shows the degree of similarity between a time series and a lagged version of itself over successive time intervals. The autocorrelation coefficient ranges from -1 to +1, with values close to +1 indicating a strong positive correlation and values close to -1 indicating a strong negative correlation. However, autocorrelation does not distinguish between direct and indirect dependencies. It can be influenced by intermediate lags.",
            "from": autocorrelation(covariates["y"]),
            "to": autocorrelation(targets["y"]),
        }

        output["correlations"][correlation.id]["partialAutocorrelations"] = {
            "description": "Partial autocorrelation measures the correlation between a time series and its lagged values, while removing the effect of the intermediate lags. It shows the direct relationship between a time series and a specific lagged value, excluding the influence of other lags in between. Partial autocorrelation helps identify the direct influence of a lagged value on the current value of the series.",
            "from": partial_autocorrelation(covariates["y"]),
            "to": partial_autocorrelation(targets["y"]),
        }

        output["correlations"][correlation.id]["regressorCoefficients"] = (
            regressor_coefficients(target_model).to_dict(orient="records")
        )

        output["correlations"][correlation.id]["predictions"] = {
            "historicalForecasts": historical_forecast.to_dict(orient="records"),
            "futureForecasts": future_forecast.to_dict(orient="records"),
        }

        return output
