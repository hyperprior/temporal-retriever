from typing import Literal
from datetime import datetime
from pydantic import BaseModel, Field, conint


class IndexResponse(BaseModel):
    index: str
    minDate: datetime
    maxDate: datetime
    unitsForecasted: conint(ge=1)


class DiagnosticsResponse(BaseModel):
    units: Literal["D", "W", "M", "H", "m"]
    from_: IndexResponse = Field(..., alias="from")
    to: IndexResponse


class Prediction(BaseModel):
    date: datetime
    trend: float
    prediction_lower_bound: float
    prediction: float
    prediction_upper_bound: float
    trend_lower_bound: float
    trend: float
    trend_upper_bound: float
    additive_terms_lower: float
    additive_terms: float
    additive_terms_upper: float
    multiplicative_terms_lower: float
    multiplicative_terms: float
    multiplicative_terms_upper: float


class Predictions(BaseModel):
    historicalForecasts: list[Prediction]
    futureForecasts: list[Prediction]


class RegressorCoefficient(BaseModel):
    regressor: str
    regressor_mode: Literal["additive", "multiplicative"]
    center: float
    coef_lower: float
    coef: float
    coef_upper: float


class CorrelationResponse(BaseModel):
    type: Literal["prophet", "granger", "autocorrelation"] = "prophet"
    diagnostics: DiagnosticsResponse
    regressorCoefficients: list[RegressorCoefficient]
    predictions: Predictions


class AnalyticsResponse(BaseModel):
    correlations: dict[str, CorrelationResponse]
