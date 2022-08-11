import math
import numpy as np
import pytest
import sklearn.metrics as skl_mt

from lib.model_scorers import rmsle_no_negative, mape, mae, mape_no_negative, mae_no_negative, mape_total_no_negative,\
    mae_total_no_negative, rmse, mse, make_evaluation_scorers

y_true_dict = {
    'case_1': np.array([0, 1, 2, 3, 4, 5]),
    'case_2': np.array([0, 1, 2, 3, 4, 5]),
    'case_3': np.array([0, 0, 0, 0, 0, 0]),
    'case_4': np.array([0, 0, 0, 0, 0, 0]),
    'case_5': np.array([1, 2, 3, 4, 5, 6]),
    'case_6': np.array([1, 2, 3, 4, 5, 6]),
    'case_7': np.array([-1, -2, -3, -4, -5, -6]),
    'case_8': np.array([-1, -2, -3, -4, -5, -6])
}

y_pred_dict = {
    'case_1': np.array([0, 1, 2, 3, 4, 5]),
    'case_2': np.array([1, 2, 3, 4, 5, 6]),
    'case_3': np.array([0, 0, 0, 0, 0, 0]),
    'case_4': np.array([0, 1, 2, 3, 4, 5]),
    'case_5': np.array([0, 0, 0, 0, 0, 0]),
    'case_6': np.array([-1, -2, -3, -4, -5, -6]),
    'case_7': np.array([1, 2, 3, 4, 5, 6]),
    'case_8': np.array([-1, -2, -3, -4, -5, -6])
}

expected_errors_dict = {
    'rmsle_no_negative': [0.0, 0.139, 0.0, 1.568, 2.199, 2.199, 'error',  'error'],
    'mape': [0.0, 'very_large', 0.0, 'very_large', 1.0, 2.0, 2.0, 0.0],
    'mae': [0.0, 1.0, 0.0, 2.5, 3.5, 7.0, 7.0, 0.0],
    'mape_no_negative': [0.0, 'very_large', 0.0, 'very_large', 1.0, 1.0, 2.0, 1.0],
    'mae_no_negative': [0.0, 1.0, 0.0, 2.5, 3.5, 3.5, 7.0, 3.5],
    'mape_total_no_negative': [0.0, 0.4, 'very_large', 'very_large', 1.0, 1.0, -2.0, -1.0],
    'mae_total_no_negative': [0, 6, 0, 15, 21, 21, 42, 21],
    'rmse': [0.0, 1.0, 0.0, 9.167, 15.167, 60.667, 60.667, 0.0],
    'mse': [0.0, 1.0, 0.0, 3.028, 3.894, 7.789, 7.789, 0.0]
}


def test_rmsle_no_negative():

    for ind, case in enumerate(y_true_dict.keys()):
        if expected_errors_dict['rmsle_no_negative'][ind] == 'error':
            # EXERCISE
            with pytest.raises(ValueError):
                rmsle_no_negative(
                    y_true=y_true_dict[case],
                    y_pred=y_pred_dict[case].copy()
                )
        else:
            # EXERCISE
            error = rmsle_no_negative(
                y_true=y_true_dict[case],
                y_pred=y_pred_dict[case].copy()
            )

            assert round(error, 3) == expected_errors_dict['rmsle_no_negative'][ind]


def test_mape():
    for ind, case in enumerate(y_true_dict.keys()):
        if expected_errors_dict['mape'][ind] == 'very_large':
            # EXERCISE
            error = mape(
                y_true=y_true_dict[case],
                y_pred=y_pred_dict[case].copy()
            )

            assert error > 10000
        else:
            # EXERCISE
            error = mape(
                y_true=y_true_dict[case],
                y_pred=y_pred_dict[case].copy()
            )

            assert round(error, 3) == expected_errors_dict['mape'][ind]


def test_mae():
    for ind, case in enumerate(y_true_dict.keys()):
        # EXERCISE
        error = mae(
            y_true=y_true_dict[case],
            y_pred=y_pred_dict[case].copy()
        )
        assert (error) == expected_errors_dict['mae'][ind]


def test_mae_no_negative():
    for ind, case in enumerate(y_true_dict.keys()):
        # EXERCISE
        error = mae_no_negative(
            y_true=y_true_dict[case],
            y_pred=y_pred_dict[case].copy()
        )
        assert (error) == expected_errors_dict['mae_no_negative'][ind]


def test_mae_total_no_negative():
    for ind, case in enumerate(y_true_dict.keys()):
        # EXERCISE
        error = mae_total_no_negative(
            y_true=y_true_dict[case],
            y_pred=y_pred_dict[case].copy()
        )
        assert (error) == expected_errors_dict['mae_total_no_negative'][ind]


def test_mape_no_negative():
    for ind, case in enumerate(y_true_dict.keys()):
        if expected_errors_dict['mape_no_negative'][ind] == 'very_large':
            # EXERCISE
            error = mape_no_negative(
                y_true=y_true_dict[case],
                y_pred=y_pred_dict[case].copy()
            )

            assert error > 10000
        else:
            # EXERCISE
            error = mape_no_negative(
                y_true=y_true_dict[case],
                y_pred=y_pred_dict[case].copy()
            )

            assert round(error, 3) == expected_errors_dict['mape_no_negative'][ind]


def test_mape_total_no_negative():
    for ind, case in enumerate(y_true_dict.keys()):
        if expected_errors_dict['mape_total_no_negative'][ind] == 'very_large':
            # EXERCISE
            error = mape_total_no_negative(
                y_true=y_true_dict[case],
                y_pred=y_pred_dict[case].copy()
            )

            assert (error > 10000) | math.isnan(error) | math.isinf(error)
        else:
            # EXERCISE
            error = mape_total_no_negative(
                y_true=y_true_dict[case],
                y_pred=y_pred_dict[case].copy()
            )

            assert round(error, 3) == expected_errors_dict['mape_total_no_negative'][ind]


def test_rmse():
    for ind, case in enumerate(y_true_dict.keys()):
        # EXERCISE
        error = rmse(
            y_true=y_true_dict[case],
            y_pred=y_pred_dict[case].copy()
        )
        assert round(error, 3) == expected_errors_dict['rmse'][ind]


def test_mse():
    for ind, case in enumerate(y_true_dict.keys()):
        # EXERCISE
        error = mse(
            y_true=y_true_dict[case],
            y_pred=y_pred_dict[case].copy()
        )
        assert round(error, 3) == expected_errors_dict['mse'][ind]


def test_fleetfly_make_scorers():
    scorers_names=['rmse', 'rmsle_no_negative', 'mape_total_no_negative']
    scorers = make_evaluation_scorers(
        scorers_names
    )
    for ind, scorer in enumerate(scorers_names):
        assert type(scorers[scorers_names[ind]]) == skl_mt._scorer._PredictScorer
