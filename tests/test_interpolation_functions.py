import numpy as np
import pandas as pd
import pytest

from ticoi.interpolation_functions import reconstruct_common_ref


def _baseline_reconstruct_common_ref(result, second_date_list=None):
    if result.empty:
        length = 1 if second_date_list is None else len(second_date_list)
        nan_list = np.full(length, np.nan)
        second_dates = [np.nan] if second_date_list is None else second_date_list
        return pd.DataFrame(
            {
                "Ref_date": nan_list,
                "Second_date": second_dates,
                "dx": nan_list,
                "dy": nan_list,
                "xcount_x": nan_list,
                "xcount_y": nan_list,
            }
        )

    data = pd.DataFrame(
        {"Ref_date": result["date1"][0], "Second_date": result["date2"]}
    )
    for var in result.columns.difference(["date1", "date2"]):
        if var in [
            "result_dx", "result_dy", "xcount_x", "xcount_y",
            "error_x", "error_y", "xcount_z",
        ]:
            data[var] = result[var].values.cumsum()
    data = data.rename(columns={"result_dx": "dx", "result_dy": "dy"})

    if second_date_list is not None:
        tmp = pd.DataFrame(
            {
                "Ref_date": pd.NaT,
                "Second_date": second_date_list,
                **{
                    var: np.nan
                    for var in data.columns.difference(["Ref_date", "Second_date"])
                },
            }
        )
        positions = np.searchsorted(second_date_list, data["Second_date"].values)
        tmp.iloc[positions] = data.values
        return tmp
    return data


def _result_fixture():
    return pd.DataFrame(
        {
            "date1": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-06"]),
            "date2": pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-10"]),
            "xcount_y": np.array([1, 2, 4], dtype=np.int16),
            "result_dy": np.array([0.5, np.nan, 1.5], dtype=np.float32),
            "ignored": ["a", "b", "c"],
            "result_dx": np.array([1.0, 2.0, 4.0], dtype=np.float32),
            "xcount_x": np.array([3, 5, 7], dtype=np.int16),
            "error_x": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        },
        index=[4, 0, 7],
    )


@pytest.mark.parametrize("with_target_dates", [False, True])
def test_reconstruct_common_ref_matches_incremental_dataframe_build(with_target_dates):
    result = _result_fixture()
    target_dates = None
    if with_target_dates:
        target_dates = np.arange(
            np.datetime64("2020-01-03"),
            np.datetime64("2020-01-11"),
            np.timedelta64(1, "D"),
        )

    expected = _baseline_reconstruct_common_ref(result, target_dates)
    actual = reconstruct_common_ref(result, target_dates)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_reconstruct_common_ref_empty_result_unchanged():
    result = _result_fixture().iloc[:0]
    expected = _baseline_reconstruct_common_ref(result)
    actual = reconstruct_common_ref(result)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
