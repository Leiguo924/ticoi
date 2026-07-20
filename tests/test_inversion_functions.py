import numpy as np
import pytest
import scipy.sparse as sp

from ticoi.core import mu_regularisation
from ticoi.inversion_functions import (
    construction_a_lf,
    construction_dates_range_np,
    inversion_one_component,
    inversion_two_components,
)


class Test_inversion:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        # This method will run before each test
        self.dates = np.array(
            [
                ["2013-03-14", "2013-03-30"],
                ["2013-03-14", "2013-03-30"],
                ["2013-03-14", "2013-04-15"],
                ["2013-03-30", "2013-04-15"],
                ["2013-03-30", "2013-04-15"],
                ["2013-03-14", "2013-08-13"],
                ["2013-03-14", "2013-10-16"],
                ["2013-06-19", "2013-07-13"],
                ["2013-03-14", "2013-10-24"],
                ["2013-03-14", "2013-11-01"],
            ]
        ).astype("datetime64[D]")

        self.dates_range = np.array(
            [
                "2013-03-14",
                "2013-03-30",
                "2013-04-15",
                "2013-06-19",
                "2013-07-13",
                "2013-08-13",
                "2013-10-16",
                "2013-10-24",
                "2013-11-01",
            ]
        ).astype("datetime64[D]")

        self.A = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )

        self.data = np.array(
            [
                [-0.69107729, -8.73340321],
                [2.40452456, -13.41930866],
                [-3.96273065, -9.17936611],
                [3.73120785, -14.85955429],
                [-2.19656491, -9.20514107],
                [10.38781738, -28.12755966],
                [3.23966694, -17.77642059],
                [368.12982178, -118.80034637],
                [1.13138795, -11.47720432],
                [2.95655584, -19.49642754],
            ]
        )

        self.mu1accelnotnull = np.array(
            [
                [-0.0625, 0.0625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -0.0625, 0.01538462, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -0.01538462, 0.04166667, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -0.04166667, 0.03225806, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -0.03225806, 0.015625, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -0.015625, 0.125, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.125, 0.125],
            ]
        ).astype("float32")

    def test_construct_dates_range(self):
        """Test construction of Dates_range for a small subset of values"""
        expected_dates_range = self.dates_range
        result = construction_dates_range_np(self.dates)
        np.testing.assert_array_equal(result, expected_dates_range)

    def test_construction_a_lf(self):
        """Test construction of A for a small subset of values"""

        expected = self.A
        actual = construction_a_lf(self.dates, self.dates_range)
        np.testing.assert_array_equal(actual, expected, err_msg="Construction A LP does not give the correct result")

    @pytest.mark.parametrize("regu", ["1", "1accelnotnull"])
    def test_first_order_regularization_matches_full_matrix_baseline_exactly(self, regu):
        n_columns = self.A.shape[1]
        expected = np.diag(np.full(n_columns, -1, dtype="float32"))
        expected[np.arange(n_columns - 1), np.arange(n_columns - 1) + 1] = 1
        expected /= np.diff(self.dates_range) / np.timedelta64(1, "D")
        expected = np.delete(expected, -1, axis=0)

        actual = mu_regularisation(regu, self.A, self.dates_range)

        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize("n_ini", [2, 4])
    def test_direction_regularization_matches_loop_baseline_exactly(self, n_ini):
        n_columns = self.A.shape[1]
        x = np.linspace(1.0, 3.0, n_columns)
        y = np.linspace(2.0, 4.0, n_columns)
        ini = [x, y] if n_ini == 2 else [x, y, x + 1, y + 2]
        delta = [
            (self.dates_range[k + 1] - self.dates_range[k]) / np.timedelta64(1, "D")
            for k in range(len(self.dates_range) - 1)
        ]
        expected = np.zeros((n_columns, 2 * n_columns), dtype="float64")
        if n_ini == 2:
            vv = np.array(ini[0]) ** 2 + np.array(ini[1]) ** 2
        else:
            vv = np.sqrt(ini[0] ** 2 + ini[1] ** 2) / 365 * np.sqrt(ini[2] ** 2 + ini[3] ** 2) / delta
        for k in range(n_columns):
            scale = 1 if n_ini == 2 else 365
            expected[k, k] = ini[0][k] / scale / int(delta[k]) / vv[k]
            expected[k, k + n_columns] = ini[1][k] / scale / int(delta[k]) / vv[k]

        actual = mu_regularisation("directionxy", self.A, self.dates_range, ini=ini)

        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize(
        "regu, expected",
        [
            (
                "1",
                np.array(
                    [
                        [-0.0625, 0.0625, 0, 0, 0, 0, 0, 0],
                        [0, -0.0625, 0.01538462, 0, 0, 0, 0, 0],
                        [0, 0, -0.01538462, 0.04166667, 0, 0, 0, 0],
                        [0, 0, 0, -0.04166667, 0.03225806, 0, 0, 0],
                        [0, 0, 0, 0, -0.03225806, 0.015625, 0, 0],
                        [0, 0, 0, 0, 0, -0.015625, 0.125, 0],
                        [0, 0, 0, 0, 0, 0, -0.125, 0.125],
                    ]
                ).astype("float32"),
            ),  # Shortened for brevity
            (
                "2",
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0625, -0.125, 0.01538462, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0625, -0.03076923, 0.04166667, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.01538462, -0.08333333, 0.03225806, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.04166667, -0.06451613, 0.015625, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.03225806, -0.03125, 0.125, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.015625, -0.25, 0.125],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ).astype("float32"),
            ),
            (
                "1accelnotnull",
                np.array(
                    [
                        [-0.0625, 0.0625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, -0.0625, 0.01538462, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, -0.01538462, 0.04166667, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, -0.04166667, 0.03225806, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, -0.03225806, 0.015625, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, -0.015625, 0.125, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.125, 0.125],
                    ]
                ).astype("float32"),
            ),
        ],
    )
    def test_mu_regularization(self, regu, expected):
        """Test construction of mu for a three different regularization"""
        actual = mu_regularisation(regu, self.A, self.dates_range)
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-6,
            atol=1,
            err_msg=f"mu_regularisation does not give the correct result for regu={regu}",
        )

    @pytest.mark.parametrize(
        "solver, expected, ini",
        [
            (
                "LSMR",
                np.array(
                    [-7.578118, -8.461816, 108.818729, -118.183104, -2.871212, 10.768036, 5.558181, -7.348526]
                ).astype("float64"),
                None,
            ),
            (
                "LS",
                np.array(
                    [-7.5791097, -8.460544, 113.54745, -118.18973, -7.585056, 10.759924, 5.560593, -7.3482523]
                ).astype("float64"),
                None,
            ),
            # (
            #     "LSMR_ini",
            #     np.array(
            #         [[  -7.576902,   -8.461586,  106.760633, -118.179535,   -0.81826 ,10.772076,    5.557501,   -7.34847 ]]
            #     ).astype("float64"),
            #     np.array([-7, -8.0, 100.0, -110.0, -7.0, 10.0, 5.0, -10.0]).astype("float64"),
            # ),
        ],
    )
    def test_inversion_one_component(self, solver, expected, ini):
        actual = inversion_one_component(
            self.A, self.dates_range, 1, self.data, solver=solver, Weight=1, mu=self.mu1accelnotnull, ini=ini
        )[0]
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-6,
            atol=1,
        )

    def test_norm_residual_matches_weighted_boundary_baseline_exactly(self):
        weight = np.array([1.0, 0.5, 0.0, 0.8, 1.0, 0.0, 0.3, 1.0, 0.6, 1.0])
        x, actual = inversion_one_component(
            self.A,
            self.dates_range,
            1,
            self.data,
            solver="LSMR",
            Weight=weight,
            mu=self.mu1accelnotnull,
            coef=100,
            result_quality=["Norm_residual"],
        )
        keep = weight != 0
        f_regu = 100 * self.mu1accelnotnull
        f = sp.csc_matrix(np.vstack([weight[keep, None] * self.A[keep], f_regu]).astype("float64"))
        d = np.hstack([weight[keep] * self.data[keep, 1], np.zeros(f_regu.shape[0])]).astype("float64")
        residual = f.dot(x) - d
        old_boundary = np.multiply(weight[keep], self.data[keep, 1]).shape[0]
        expected = [
            np.linalg.norm(residual[:old_boundary], ord=2),
            np.linalg.norm(residual[old_boundary:] / 100, ord=2),
        ]

        np.testing.assert_array_equal(actual, expected)

    def test_two_component_sparse_system_matches_dense_baseline_exactly(self, monkeypatch):
        weight = np.linspace(0.2, 1.0, 2 * self.A.shape[0])
        weight[::4] = 0
        mu = np.zeros((self.A.shape[1], 2 * self.A.shape[1]), dtype="float64")
        rows = np.arange(self.A.shape[1])
        mu[rows, rows] = 0.25
        mu[rows, rows + self.A.shape[1]] = 0.75
        block_a = np.block(
            [[self.A, np.zeros_like(self.A)], [np.zeros_like(self.A), self.A]]
        )
        keep = weight != 0
        expected_f = sp.csc_matrix(
            np.vstack([weight[keep, None] * block_a[keep], 3 * mu]).astype("float64")
        )
        velocity = np.concatenate([self.data[:, 0], self.data[:, 1]])
        expected_d = np.hstack([weight[keep] * velocity[keep], np.ones(mu.shape[0]) * 3]).astype("float64")

        def verify_lsmr(actual_f, actual_d, **kwargs):
            np.testing.assert_array_equal(actual_f.data, expected_f.data)
            np.testing.assert_array_equal(actual_f.indices, expected_f.indices)
            np.testing.assert_array_equal(actual_f.indptr, expected_f.indptr)
            np.testing.assert_array_equal(actual_d, expected_d)
            return (np.zeros(2 * self.A.shape[1]),)

        monkeypatch.setattr(sp.linalg, "lsmr", verify_lsmr)
        direction_data = np.column_stack(
            [np.zeros((len(self.data), 2)), self.data[:, 0], self.data[:, 1]]
        )
        inversion_two_components(
            self.A,
            self.dates_range,
            0,
            direction_data,
            weight,
            mu,
            solver="LSMR",
            coef=3,
            show_L_curve=True,
        )
