from typing import Literal
from datetime import datetime
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.utils.statistics import granger_causality_tests, remove_trend
from fastapi import FastAPI, status
from pydash import get
from prophet import Prophet
from prophet.utilities import regressor_coefficients
from pydantic import BaseModel, Field, conint, AliasChoices

from loguru import logger


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
        validation_alias=AliasChoices("predictionHorizon", "unitsToForecast"),
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


class IndexResponse(BaseModel):
    index: str
    minDate: datetime
    maxDate: datetime
    unitsForecasted: conint(ge=1)
    historicalForecastDates: list[datetime]
    futureForecastDates: list[datetime]


class DiagnosticsResponse(BaseModel):
    units: Literal["D", "W", "M", "H", "m"]
    from_: IndexResponse = Field(..., alias="from")
    to: IndexResponse


class CorrelationResponse(BaseModel):
    type: Literal["prophet", "granger", "autocorrelation"] = "prophet"
    diagnostics: DiagnosticsResponse


class AnalyticsResponse(BaseModel):
    correlations: dict


@app.post("/analyze")
async def analyze_datasets(request: AnalyticsRequest) -> AnalyticsResponse:
    correlations = request.analytics_options.correlations

    output = {"correlations": {}}

    for correlation in correlations:
        grain = correlation.grain
        aggregation = correlation.aggregation
        quantiles = correlation.quantiles
        covariate_name: str = correlation.from_index
        target_name: str = correlation.to_index

        from_data = [
            {"ds": data.get("date"), "y": data.get("covariate_name")}
            for data in request.documents.get(correlation.from_data).get("data")
        ]

        covariates, covariates_prediction_horizon = prepare_dataset(
            dataset=from_data,
            time_column="ds",
            aggregation=aggregation,
            grain=grain,
        )

        covariate_date_bounds = covariates["ds"].min(), covariates["ds"].max()

        covariate_model = Prophet()
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

        target_model = Prophet()
        target_model.add_regressor(covariate_name)
        target_model.fit(targets)

        future = target_model.make_future_dataframe(
            periods=targets_prediction_horizon, freq=grain
        )

        future["ds"] = pd.to_datetime(future["ds"])

        future_dates = future["ds"].to_list()

        future = future.merge(covariate_predictions, on="ds")

        target_forecast = target_model.predict(future)

        output["correlations"][correlation.id] = {
            "type": "prophet",
            "diagnostics": {
                "units": grain,
                "from": {
                    "index": correlation.from_index,
                    "minDate": covariate_date_bounds[0],
                    "maxDate": covariate_date_bounds[1],
                    "unitsForecasted": covariates_prediction_horizon,
                    "historicalForecastDates": covariate_future_dates[
                        :covariates_prediction_horizon
                    ],
                    "futureForecastDates": covariate_future_dates[
                        covariates_prediction_horizon:
                    ],
                },
                "to": {
                    "index": correlation.to_index,
                    "minDate": target_date_bounds[0],
                    "maxDate": target_date_bounds[1],
                    "unitsForecasted": targets_prediction_horizon,
                    "historicalForecastDates": future_dates[
                        :targets_prediction_horizon
                    ],
                    "futureForecastDates": future_dates[targets_prediction_horizon:],
                },
            },
            "regressor_coefficients": regressor_coefficients(target_model).to_dict(
                orient="records"
            ),
            "predictions": target_forecast.rename(
                columns={
                    "ds": "date",
                    "yhat": "prediction",
                    "yhat_lower": "prediction_lower_bound",
                    "yhat_upper": "prediction_upper_bound",
                    "trend_lower": "trend_lower_bound",
                    "trend_upper": "trend_upper_bound",
                }
            ).to_dict(orient="records"),
        }

        return output
