from typing import Literal
from types import SimpleNamespace

from functools import cached_property

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
        dataframe[time_column] = reset_time_index(
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

        future = future.merge(covariate_predictions, on="ds", how="left").dropna()

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


class Cap(BaseModel):
    floor: float | None = 0
    ceiling: float | None = None


class Caps(BaseModel):
    from_index: Cap = Field(Cap(), alias="fromIndex")
    to_index: Cap = Field(Cap(), alias="toIndex")


class ForecastingOption(BaseModel):
    uncertainty_samples: conint(ge=1) = Field(1000, alias="uncertaintySamples")
    changepoint_prior_scale: float = Field(0.5, alias="changepointPriorScale")
    growth: Literal["linear", "logistic"] = "logistic"
    caps: Caps = Caps()


class ForecastingOptions(BaseModel):
    from_index: ForecastingOption = Field(ForecastingOption(), alias="fromIndex")
    to_index: ForecastingOption = Field(ForecastingOption(), alias="toIndex")


class SaturatingGrowthCorrelation(BaseModel):
    id: str
    type: Literal["prophet"] = "prophet"
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
        alias="unitsToForecast",
    )
    forecasting_options: ForecastingOptions = Field(
        ForecastingOptions(), alias="ForecastingOptions"
    )


class SaturatingGrowthAnalyticsOptions(BaseModel):
    correlations: list[SaturatingGrowthCorrelation]


class SaturatingGrowthRequest(BaseModel):
    analytics_options: SaturatingGrowthAnalyticsOptions = Field(
        ..., alias="analyticsOptions"
    )
    documents: dict


class SaturatingGrowthResponse(BaseModel):
    pass


class UnivariateTimeSeriesDataBundle:
    def __init__(
        self,
        *,
        dataset: list[dict[str, str | float]],
        value_column: str,
        prediction_horizon: int | None,
        time_column: str = "ds",
        aggregation: str = "sum",
        grain: str | None = None,
    ):
        self.dataset = dataset
        self.time_column = time_column
        self.value_column = value_column
        self.aggregation = aggregation
        self.grain = grain
        self.prediction_horizon = prediction_horizon or len(dataset)
        self.value_column = value_column

    @property
    def output_columns(self):
        return {
            "ds": "date",
            "yhat_lower": "prediction_lower_bound",
            "yhat": "prediction",
            "yhat_upper": "prediction_upper_bound",
            "trend_lower": "trend_lower_bound",
            "trend": "trend",
            "trend_upper": "trend_upper_bound",
            "additive_terms_lower": "additive_terms_lower",
            "additive_terms": "additive_terms",
            "additive_terms_upper": "additive_terms_upper",
            "multiplicative_terms_lower": "multiplicative_terms_lower",
            "multiplicative_terms": "multiplicative_terms",
            "multiplicative_terms_upper": "multiplicative_terms_upper",
        }

    @cached_property
    def floor(self):
        return min(self.value_column.floor, self.dataframe["y"].min())

    @cached_property
    def ceiling(self):
        return max(
            self.value_column.ceiling
            or (self.dataframe["y"].max() + self.dataframe["y"].std() * 3),
            self.dataframe["y"].max(),
        )

    @cached_property
    def date_bounds(self):
        return SimpleNamespace(
            min=self.dataframe["ds"].min(), max=self.dataframe["ds"].max()
        )

    @cached_property
    def raw_dataframe(self):
        dataframe = pd.DataFrame(self.dataset)
        dataframe["y"] = dataframe[self.value_column.name]
        try:
            dataframe[self.time_column] = self._reset_time_index(
                series=dataframe[self.time_column], grain=self.grain
            )
        except ValueError:
            logger.info("falling back to mixed date format")

            dataframe[self.time_column] = self._reset_time_index(
                series=dataframe[self.time_column], grain=self.grain, format="mixed"
            )

        return dataframe

    @cached_property
    def dataframe(self):
        return (
            self.raw_dataframe.groupby(self.time_column)
            .agg({"y": self.aggregation, self.value_column.name: self.aggregation})
            .reset_index()
        )

    @property
    def historical_forecasts(self):
        return (
            self.predictions[self.predictions["ds"] <= self.date_bounds.max][
                list(self.output_columns.keys())
            ]
            .rename(columns=self.output_columns)
            .to_dict(orient="records")
        )

    @property
    def future_forecasts(self):
        return (
            self.predictions[self.predictions["ds"] > self.date_bounds.max][
                list(self.output_columns.keys())
            ]
            .rename(columns=self.output_columns)
            .to_dict(orient="records")
        )

    def _reset_time_index(
        self,
        *,
        series: pd.Series,
        format: Literal["ISO8601", "mixed"] = "ISO8601",
        grain: Literal["D", "W", "M", "H", "m"] | None = None,
    ):
        series = pd.to_datetime(series, format=format, utc=True)

        match grain:
            case None:
                return series.dt.tz_localize(None)
            case "D":
                return series.dt.date.dt.tz_localize(None)
            case "W":
                return series.dt.to_period("W").dt.end_time
            case "M":
                return series.dt.to_period("M").dt.end_time
            case "H":
                return series.dt.floor("H")
            case "m":
                return series.dt.floor("T")
            case _:
                raise ValueError(f"Unsupported granularity: {grain}")

    def predict_prophet(self, options, covariates=None, is_target: bool = False):
        data = self.dataframe

        if options.growth == "logistic":
            data["floor"] = self.floor
            data["cap"] = self.ceiling

        model = Prophet(
            uncertainty_samples=options.uncertainty_samples,
            changepoint_prior_scale=options.changepoint_prior_scale,
            growth=options.growth,
        )

        if covariates:
            data = data[["ds", "y", "floor", "cap"]].merge(
                covariates.predictions, how="left", on="ds"
            )
            model.add_regressor(covariates.value_column.name)

        model.fit(data)

        future = model.make_future_dataframe(
            periods=self.prediction_horizon, freq=self.grain
        )

        if covariates:
            future = future.merge(covariates.predictions, on="ds")

        if options.growth == "logistic":
            future["floor"] = self.floor
            future["cap"] = self.ceiling

        predictions = model.predict(future)  # [["ds", "yhat"]]

        predictions["ds"] = pd.to_datetime(predictions["ds"])

        if not is_target:
            predictions = predictions.merge(self.dataframe, how="left", on="ds")
            predictions[self.value_column.name] = predictions["y"].combine_first(
                predictions["yhat"]
            )
            predictions = predictions[["ds", self.value_column.name]]

        self.model = model
        self.future = future
        self.predictions = predictions


@app.post("/saturating-growth")
async def saturating_growth(request: SaturatingGrowthRequest):
    correlations = request.analytics_options.correlations

    output = {"correlations": {}}

    for correlation in correlations:
        grain = correlation.grain
        aggregation = correlation.aggregation
        covariate_name: str = correlation.from_index
        target_name: str = correlation.to_index
        prediction_horizon = correlation.prediction_horizon

        covariate_forecasting_options = correlation.forecasting_options.from_index
        target_forecasting_options = correlation.forecasting_options.to_index

        from_data = [
            {"ds": data.get("date"), covariate_name: get(data, covariate_name)}
            for data in request.documents.get(correlation.from_data).get("data")
        ]

        covariates = UnivariateTimeSeriesDataBundle(
            dataset=from_data,
            prediction_horizon=prediction_horizon,
            value_column=SimpleNamespace(
                name=covariate_name,
                floor=covariate_forecasting_options.caps.from_index.floor,
                ceiling=covariate_forecasting_options.caps.from_index.ceiling,
            ),
        )

        covariates.predict_prophet(
            options=covariate_forecasting_options, is_target=False
        )

        to_data = [
            {"ds": data.get("date"), target_name: get(data, target_name)}
            for data in request.documents.get(correlation.to_data)["data"]
        ]

        targets = UnivariateTimeSeriesDataBundle(
            dataset=to_data,
            prediction_horizon=prediction_horizon,
            value_column=SimpleNamespace(
                name=target_name,
                floor=target_forecasting_options.caps.to_index.floor,
                ceiling=target_forecasting_options.caps.to_index.ceiling,
            ),
        )

        targets.predict_prophet(
            options=target_forecasting_options, covariates=covariates, is_target=True
        )

        output["correlations"][correlation.id] = {
            "type": {
                "model": correlation.type,
                "growth": target_forecasting_options.growth,
                "bounds": {
                    "min": targets.date_bounds.min,
                    "max": targets.date_bounds.max,
                },
            },
            "predictions": {
                "historicalForecasts": targets.historical_forecasts,
                "futureForecasts": targets.future_forecasts,
            },
        }

    return output


@app.post("/saturating-growth/single")
async def saturating_growth(request: SaturatingGrowthRequest):
    correlations = request.analytics_options.correlations

    output = {"correlations": {}}

    for correlation in correlations:
        grain = correlation.grain
        aggregation = correlation.aggregation
        target_name: str = correlation.to_index
        target_forecasting_options = correlation.forecasting_options.to_index

        to_data = [
            {"ds": data.get("date"), target_name: get(data, target_name)}
            for data in request.documents.get(correlation.to_data)["data"]
        ]

        targets = UnivariateTimeSeriesDataBundle(
            dataset=to_data,
            prediction_horizon=correlation.prediction_horizon,
            value_column=SimpleNamespace(
                name=target_name,
                floor=target_forecasting_options.caps.to_index.floor,
                ceiling=target_forecasting_options.caps.to_index.ceiling,
            ),
        )

        print(f"{correlation.prediction_horizon=}")
        print(f"{targets.prediction_horizon=}")

        targets.predict_prophet(options=target_forecasting_options, is_target=True)

        output["correlations"][correlation.id] = {
            "type": {
                "model": correlation.type,
                "growth": target_forecasting_options.growth,
                "bounds": {
                    "min": targets.date_bounds.min,
                    "max": targets.date_bounds.max,
                },
            },
            "predictions": {
                "historicalForecasts": targets.historical_forecasts,
                "futureForecasts": targets.future_forecasts,
            },
        }

    return output
