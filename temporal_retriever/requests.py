from re import L
from typing import List

from pydantic import Basemodel

Observation = dict


Dataset = List[Observation]


class DataSeries(BaseModel):


TimeSeries = List[Observation]


class TimeSeries(BaseModel):
    data: List[float]


[
    {
        "name": "purchaseOrders",
        "data": [
            {
                "date": asdfa
                "value": 234,

            }
        ]
    },
    {
        "name": "salesOrders",
        "data": [{"date": ..., "value": ...}]
    }

    {
        "name": "purchaseOrdersTotalWithTax"
        "target": "purchaseOrders",
    }
]

j


class Correlation(BaseModel):
    fromData: str
    fromIndex:


class AnalysisRequest(BaseModel):
    data: Dict[str, TimeSeries]
    targets: Optional[]
    covariates:
    correlations: Correlation
    correlations:
    forecast: bool = True

    def time_warp(self):
        pass

    def preprocess(self):
        self.data = pd.DataFrame(self.data).sort_values("date")
        self.data.agg("date")[]
