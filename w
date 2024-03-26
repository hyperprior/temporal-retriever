from darts.dataprocessing import dtw
from darts.metrics import dtw_metric, mae, mape
from darts import TimeSeries
from darts.models import Prophet, TFTModel

from darts.utils.statistics import (
    check_seasonality,
    extract_trend_and_seasonality,
    stationarity_tests,
    granger_causality_tests,
    remove_trend,
)

from darts.metrics import rmse

import pandas as pd
import numpy as np
from fastapi import FastAPI, status

from temporal_retriever.requests import AnalysisRequest


app: FastAPI = FastAPI()


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    return


def interval_predictions(model, quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]):
    forecast = model.predict(n=14, num_samples=1000).all_values()
    predictions = pd.DataFrame(np.quantile(forecast, quantiles, axis=-1).reshape(7,14).T)

    predictions.columns = [f"q{quant}" for quant in quantiles]

    return predictions




@app.post("/analyze")
async def analyze_datasets(request: AnalysisRequest):
    prediction_horizon = 14
    correlations = request.get("analyticsOptions").get("correlations")

    output = {}
    
    for correlation in correlations:
        covariate_name = correlation.get("fromIndex")
        target_name = correlation.get("toIndex")

        covariates = pd.DataFrame(request.get(correlation["fromData"])["data"])[["date", covariate_name ]]
        covariates["date"] = pd.to_datetime(covariates["date"])

        targets = pd.DataFrame(request.get(correlation["toData"], {}).get("data", {}))[["date", target_name]]
        targets["date"] = pd.to_datetime(targets["date"])

        dataset.ffill(inplace=True)

        

        # dataset = TimeSeries.from_dataframe(dataset, time_col="date", fill_missing_dates=True)
        # alignment = dtw.dtw(covariates, targets)
        # warped_covariates, warped_target = 
        # targets = pd.DataFrame(request.get(correlation["toData"]).get("data")) #["data"]) #[["date", correlation.get("toIndex")]]
        # print(covariates)
        # print(targets)

        # targets = targets.rename(columns={"date": "ds", target_name: "y"})

        model = Prophet(
        # add_seasonalities={
        # 'name':"quarterly_seasonality",
        # 'seasonal_periods':4,
        # 'fourier_order':5
        # },
        #     add_encoders={
        # 'cyclic': {'future': ['month', 'week']},
        # 'datetime_attribute': {'future': ['dayofweek', 'dayofmonth', 'monthofyear']},
        # 'position': {'future': ['relative']},
        # """  """}
        )

        targets = TimeSeries.from_dataframe(targets, time_col="date", value_cols=target_name)

        model.fit(targets)

        predictions = interval_predictions(model)
        print(predictions)
        
        backtests = model.historical_forecasts(targets, start=0.5, forecast_horizon=prediction_horizon, overlap_end=True)
        backtest_score = mape(backtests, targets)

        print(backtest_score)
    
        

        # forecast = forecast.rename(columns=cols)[list(cols.values())]
        # print(dir(model))
        # print(model.seasonalities)
        # print(model.changepoints)

    # return forecast# .to_dict(orient="records")
