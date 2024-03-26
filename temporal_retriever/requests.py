"""Request models for the API."""

from functools import cached_property

import pandas as pd
from pydantic import BaseModel, Field


class Correlation(BaseModel):
    """Correlations for data in a nested, i.e., Document-->Field format."""

    from_data: str = Field(..., alias="fromData")
    from_index: str = Field(..., alias="fromIndex")
    to_data: str = Field(..., alias="toData")
    to_index: str = Field(..., alias="toIndex")


Observation = dict[str, float]


class Document(BaseModel):
    """A document here is data from a document database
    so that a document has time series data in it."""

    description: str | None = None
    data: list[Observation]

    # @validator("data")
    # def validate_observations_have_date_field(
    #     self, v: list[Observation]
    # ) -> list[Observation]:
    #     """All temporal datasets must have at least a date field,
    #     else they would not be temporal."""
    #     for observation in v:
    #         if not observation.get("date"):
    #             raise ValueError("missing field `date` for observation")
    #     return v
    #


class AnalysisRequest(BaseModel):
    data: dict[str, Document]
    # targets: Optional[]
    # covariates:
    analyticsOptions: dict
    # correlations:
    forecast: bool = True
    aggregations: dict[str, str] | None = None

    def time_warp(self):
        pass

    def preprocess(self):
        self.data = pd.DataFrame(self.data).sort_values("date")
        # self.data.agg("date")

    @cached_property
    def dataframe(self):
        data = pd.DataFrame(self.data)
        if self.aggregations:
            data.group_by("date").agg(self.aggregations).reset_index()

        return data.sort_values("date", ascending=True)


AnalysisRequest = dict
