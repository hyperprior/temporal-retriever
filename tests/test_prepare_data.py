import pandas as pd
from pandas.testing import assert_series_equal


def test_iso8601_format():
    series = pd.Series(["2023-06-01T10:00:00Z", "2023-06-02T12:30:00Z"])
    expected_index = pd.to_datetime(
        pd.Series(
            ["2023-06-01 10:00:00", "2023-06-02 12:30:00"],
        )
    )
    assert_series_equal(reset_time_index(series=series), expected_index)


# def test_mixed_format():
#     series = pd.Series(["2023-06-01T10:00:00Z", "2023-06-02 12:30:00"])
#     expected_index = pd.DatetimeIndex(
#         ["2023-06-01 10:00:00", "2023-06-02 12:30:00"], tz="UTC"
#     )
#     assert_index_equal(reset_time_index(series=series, format="mixed"), expected_index)
#
#
# def test_grain_none():
#     series = pd.Series(["2023-06-01T10:00:00Z", "2023-06-02T12:30:00Z"])
#     expected_index = pd.DatetimeIndex(
#         ["2023-06-01 10:00:00", "2023-06-02 12:30:00"], tz="UTC"
#     )
#     assert_index_equal(reset_time_index(series=series, grain=None), expected_index)
#
#
# def test_grain_day():
#     series = pd.Series(["2023-06-01T10:00:00Z", "2023-06-02T12:30:00Z"])
#     expected_index = pd.Index(
#         [pd.Timestamp("2023-06-01").date(), pd.Timestamp("2023-06-02").date()]
#     )
#     assert_index_equal(reset_time_index(series=series, grain="D"), expected_index)
#
#
# def test_empty_series():
#     series = pd.Series([])
#     expected_index = pd.DatetimeIndex([], tz="UTC")
#     assert_index_equal(reset_time_index(series=series), expected_index)
#
#
# def test_missing_values():
#     series = pd.Series(["2023-06-01T10:00:00Z", None, "2023-06-02T12:30:00Z"])
#     expected_index = pd.DatetimeIndex(
#         ["2023-06-01 10:00:00", pd.NaT, "2023-06-02 12:30:00"], tz="UTC"
#     )
#     assert_index_equal(reset_time_index(series=series), expected_index)
#
#
# def test_timezone_handling():
#     series = pd.Series(["2023-06-01T10:00:00+02:00", "2023-06-02T12:30:00+05:00"])
#     expected_index = pd.DatetimeIndex(
#         ["2023-06-01 08:00:00", "2023-06-02 07:30:00"], tz="UTC"
#     )
#     assert_index_equal(reset_time_index(series=series), expected_index)
#
#
# def test_invalid_input_type():
#     with pytest.raises(AttributeError):
#         reset_time_index(series=[1, 2, 3])
#
#
# def test_unsupported_format():
#     series = pd.Series(["2023-06-01T10:00:00Z", "2023-06-02T12:30:00Z"])
#     with pytest.raises(ValueError):
#         reset_time_index(series=series, format="unsupported")
