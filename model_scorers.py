import numpy as np
import pandas as pd
import sklearn.metrics as skl_mt

"""
Set of functions used to create ML model scorers. 
Note: example of how scorers are created and passed to evaluation function: 
scorer = skl_mt.make_scorer(rmsle_no_negative, greater_is_better=False)
grid = ms.GridSearchCV(estimator=XGBRegressor(random_state=10, n_jobs=-1),
                       param_grid=space,
                       cv=wfv,
                       scoring=scorer
                       )
Multiscorers also can be created (e.g. as dictionaries)
Note: when we fit using any of the create metrics X_train and y_train (or test) are also passed and can be called
inside of the error scorers functions. They can have indexes identifying e.g. locations or time, this can be used
to calculate aggregated errors (e.g. on hourly level or cluster level). Other way to do it would be using
multioutput = 'raw_values' param.
"""


def calculate_combined_scorers_metrics(
    evaluation_results: pd.DataFrame,
    scorers_names: list
):
    """
    Function that calculates combination metric used to choose best model hyperparameters during evaluation
    The combination metric is obtained by normalization of all scorers. The combined metric is added as a new column
    to evaluation_results.
    Args:
        evaluation_results: evaluation results created by create_evaluation_metrics_df() function.
        scorers_names: list with scorers names used in evaluation.
    Returns:
        evaluation_results: evaluation results with combined metric column added.
    """

    # Obtain standardized metrics
    for scorer in scorers_names:
        best_mean_test_results_scorer = \
            evaluation_results.loc[evaluation_results[f"rank_test_{scorer}"] == 1, f"mean_test_{scorer}"].values[0]
        evaluation_results[f"mean_test_{scorer}_normalized"] = evaluation_results[
                                                                   f"mean_test_{scorer}"] / best_mean_test_results_scorer

    evaluation_results['mean_test_combined_normalized'] = evaluation_results[
                                                              [f"mean_test_{scorer}_normalized" for scorer in
                                                               scorers_names]].sum(axis=1) / len(scorers_names)
    evaluation_results['rank_test_combined_normalized'] = evaluation_results['mean_test_combined_normalized'] \
        .rank(method='min')

    return evaluation_results


def calculate_error_metrics(
        results: pd.DataFrame,
        col_target: str,
        col_predict: str,
        conf: dict
) -> dict:
    """
    Calculates error metrics (using scorers available in this file) for col_target (real values) and col_predict
     (predicted values) columns.
    TODO:
     - add possibility of calculating errors on different aggregation levels (e.g. hour or cluster)
     - available_scorers (training) should be part of the config (separated from eval scorers)
    Args:
        results: df with results
        col_target: column with real values
        col_predict: column with predicted values
        conf: config as dict
    Returns:
        errors_dict: dictionary with predicted errors

    """
    available_scorers = []
    if "rmsle_no_negative" in conf['evaluation_model_scorers']:
        available_scorers.append(rmsle_no_negative)
    if "mape" in conf['evaluation_model_scorers']:
        available_scorers.append(mape)
    if "mae" in conf['evaluation_model_scorers']:
        available_scorers.append(mae)
    if "mape_no_negative" in conf['evaluation_model_scorers']:
        available_scorers.append(mape_no_negative)
    if "mape_total" in conf['evaluation_model_scorers']:
        available_scorers.append(mape_total)
    if "mae_no_negative" in conf['evaluation_model_scorers']:
        available_scorers.append(mae_no_negative)
    if "mape_total_no_negative" in conf['evaluation_model_scorers']:
        available_scorers.append(mape_total_no_negative)
    if "mae_total_no_negative" in conf['evaluation_model_scorers']:
        available_scorers.append(mae_total_no_negative)
    if "rmse" in conf['evaluation_model_scorers']:
        available_scorers.append(rmse)
    if "mse" in conf['evaluation_model_scorers']:
        available_scorers.append(mse)

    errors_dict = {}
    for scorer_function in available_scorers:
        error = scorer_function(
            y_true=results.loc[:, col_target].copy(),
            y_pred=results.loc[:, col_predict].copy()
        )
        errors_dict[scorer_function.__name__] = error

    return errors_dict


def mae(
    y_true: np.ndarray,
    y_pred: list
):
    """
    Mean Absolute Error scorer.
    Args:
        y_true: vector of true values of target variable
        y_pred: vector of predicted values of target variable

    Returns:
        error: error
    """
    error = skl_mt.mean_absolute_error(y_true,
                                       y_pred,
                                       multioutput='uniform_average')
    return error


def mae_no_negative(
    y_true: np.ndarray,
    y_pred: list
):
    """
    Mean Absolute Error scorer. Values predicted below zero are set to zeroes.
    Args:
        y_true: vector of true values of target variable
        y_pred: vector of predicted values of target variable

    Returns:
        error: error
    """
    y_pred[y_pred < 0] = 0
    error = skl_mt.mean_absolute_error(y_true,
                                       y_pred,
                                       multioutput='uniform_average')
    return error


def mae_total_no_negative(
    y_true: np.ndarray,
    y_pred: list
):
    """
    Total Mean Absolute Error scorer. Values predicted below zero are set to zeroes.
    Args:
        y_true: vector of true values of target variable
        y_pred: vector of predicted values of target variable

    Returns:
        error: error
    """
    y_pred[y_pred < 0] = 0
    sum_true = sum(y_true)
    sum_pred = sum(y_pred)

    error = abs(sum_true - sum_pred)

    return error


def make_evaluation_scorers(
    scorers_names: [list, None]
) -> [dict, list]:
    """
    Create scorers list for evaluation.
    Args:
        scorers_names: scorers names, a list with scorers used for evaluation, have to be strings with names
        of the scorers functions available in this .py file
    Returns:
        scorers: dict with scorers
        scorers_names: list with scorers names
    TODO: available_scorers can be obtained automatically reading all available functions in this module
    """

    if not scorers_names:
        scorers_names = ['rmse']

    available_scorers = [rmsle_no_negative, mape, mae, mape_no_negative, mape_total,
                         mae_no_negative, mape_total_no_negative, mae_total_no_negative, rmse, mse]
    scorers_list = []
    for available_scorer in available_scorers:
        if available_scorer.__name__ in scorers_names:
            scorers_list = scorers_list + [available_scorer]

    scorers = {}
    for ind, scorer in enumerate(scorers_list):
        scorers[scorers_names[ind]] = skl_mt.make_scorer(scorer, greater_is_better=False)

    return scorers


def mape(
    y_true: np.ndarray,
    y_pred: list
):
    """
    Mean Absolute Percentage Error scorer.
    Note: will produce very large values if any of the y_true values = 0 (unless the same y_pred value = 0),
     maybe new scorer where we add very small value for these cases, or we ignore these cases)
    Args:
        y_true: vector of true values of target variable
        y_pred: vector of predicted values of target variable

    Note: no negative y_true values
    Returns:
        error: error
    """

    error = skl_mt.mean_absolute_percentage_error(y_true,
                                                  y_pred,
                                                  multioutput='uniform_average')
    return error


def mape_no_negative(
    y_true: np.ndarray,
    y_pred: list
):
    """
    Mean Absolute Percentage Error scorer. Values predicted below zero are set to zeroes.
    Note: will produce very large values if any of the y_true values = 0 (unless the same y_pred value = 0),
     maybe new scorer where we add very small value for these cases, or we ignore these cases)
    Args:
        y_true: vector of true values of target variable
        y_pred: vector of predicted values of target variable

    Returns:
        error: error
    """

    y_pred[y_pred < 0] = 0
    error = skl_mt.mean_absolute_percentage_error(y_true,
                                                  y_pred,
                                                  multioutput='uniform_average')
    return error


def mape_total_no_negative(
    y_true: np.ndarray,
    y_pred: list
):
    """
    Total Mean Absolute Percentage Error scorer. Values predicted below zero are set to zeroes.
    Args:
        y_true: vector of true values of target variable
        y_pred: vector of predicted values of target variable

    Returns:
        error: error
    """
    y_pred[y_pred < 0] = 0

    sum_true = sum(y_true)
    sum_pred = sum(y_pred)

    error = abs(sum_true - sum_pred) / sum_true

    return error


def mape_total(
    y_true: np.ndarray,
    y_pred: list
):
    """
    Total Mean Absolute Percentage Error scorer.
    Args:
        y_true: vector of true values of target variable
        y_pred: vector of predicted values of target variable

    Returns:
        error: error
    """

    sum_true = sum(y_true)
    sum_pred = sum(y_pred)

    error = abs(sum_true - sum_pred) / sum_true

    return error

def mse(
    y_true: np.ndarray,
    y_pred: list
):
    """
    Mean Squared Error scorer.
    Args:
        y_true: vector of true values of target variable
        y_pred: vector of predicted values of target variable

    Returns:
        error: error
    """
    error = skl_mt.mean_squared_error(y_true,
                                      y_pred,
                                      squared=False,
                                      multioutput='uniform_average')

    return error


def rmse(
    y_true: np.ndarray,
    y_pred: list
):
    """
    Root Mean Squared Error scorer.
    Args:
        y_true: vector of true values of target variable
        y_pred: vector of predicted values of target variable

    Returns:
        error: error
    """
    error = skl_mt.mean_squared_error(y_true,
                                      y_pred,
                                      squared=True,
                                      multioutput='uniform_average')

    return error


def rmsle_no_negative(
    y_true: np.ndarray,
    y_pred: list
):
    """
    Root Mean Squared Logged Error scorer.
    Note: will produce very large values if any of the y_true values = 0(?). Can be used if y_true or y_pred have zero
    values.
    Args:
        y_true: vector of true values of target variable
        y_pred: vector of predicted values of target variable

    Returns:
        error: error
    """
    y_pred[y_pred < 0] = 0
    error = skl_mt.mean_squared_log_error(y_true,
                                          y_pred)

    return error
