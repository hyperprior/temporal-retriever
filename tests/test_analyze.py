import pytest

from temporal_retriever.requests import AnalysisRequest, Document, Observation


@pytest.mark.unit
def test_observations_has_datetime():
    data = [{"date": "2024-01-01"}]
    document = Document(data=data)

    assert document.data == data
    assert document.data[0]["date"] == "2024-01-01"

    with pytest.raises(ValueError):
        data = [{"not_date": "2024-01-01"}]
        document = Document(data=data)


def test_dataset_is_converted_to_dataframe():
    gdp = [{"date": "2024-01-01", "us_gdp": 100}, {"2024-02-01": "us_gdp": 200}]
    document = Document(data=data)

    assert document.data


def test_observation_has_at_least_one_variate():
    with pytest.raises():
        observation = Observation({"date": "2024-01-01"})


# const x = {
#     salesOrders: {
#         description: `sales orders ...`,
#         data: [
#             {
#                 date: '2014-09-21',
#                 totalWithTax: 23,
#                 totalTax: 23,
#                 totalShipping: 12
#             }
#         ]
#     },
#     purchaseOrders: {
#         name: 'Purchase orders',
#         description: `purchase orders ....`,
#         data: [
#             {
#                 date: '2014-09-22',
#                 totalWithTax: 55,
#                 totalTax: 1,
#                 totalShipping: 3
#             }
#         ]
#     },
#     analyticsOptions: {
#         correlations: [
#             {id: 'x1', type: 'prophet',
#              fromData: 'salesOrder',
#              fromIndex: 'totalWithTax',
#                 toData: 'purchaseOrder', toIndex: 'totalWithTax'},
#             {id: 'x2', type: 'prophet', fromData: 'salesOrder', fromIndex: 'totalWithTax',
#              toData: 'inventoryDemandInbound', toIndex: 'totalWithTax'},
#         ]
#     }
# }
