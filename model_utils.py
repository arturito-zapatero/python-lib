import datetime
import datetime as dt
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from typing import Generator

import lib.model_scorers as md_scr
import lib.data_transformations as dt_trf


def calculate_model_data_time_period(
    ml_framework_step: str,
    conf: dict
) -> [[pd.Timestamp, datetime.date], [pd.Timestamp, datetime.date], [pd.Timestamp, datetime.date, None]]:
    """
    Calculate first and last day of data and first prediction day of data, depending on the config parameters.
    First day of data is calculated with regard to config running_metric_days and lags days variables,
    in data engineering part when calculating this features, we will have NaN for lags_days first days and
    for running_metric_days plus days from the shift for prediction_days
    For prediction last day of data is last prediction day as well

    TODO:
     - possibility to find the first day of data automatically (e.g. if conf['first_training_day']='auto'
    Args:
        ml_framework_step: evaluation, training or predictions
        conf: config as dict

    Returns:
        first_day_data: first day of data
        last_day_data: last day of data (or last day of predictions)
        first_prediction_day_data: first day of predictions
    """
    # Calculate shift to include running means and lag days into data
    if 'running_metric_days' in conf.keys() or 'lags_days' in conf.keys():
        running_lag_shift_days = max([conf['prediction_days'] + max([day-1 for day in conf['running_metric_days']])] +
                                      conf['lags_days'])
    else:
        running_lag_shift_days = 0

    if ml_framework_step == 'evaluation':
        if conf['last_evaluation_day'] == 'yesterday':
            first_day_data = pd.to_datetime(pd.Timestamp.today().date() -
                                            pd.Timedelta(days=(conf['evaluation_days'] + running_lag_shift_days + 1)))
            last_day_data = pd.to_datetime(pd.Timestamp.today().date() -
                                           pd.Timedelta(days=1))
        else:
            last_day_data = pd.to_datetime(conf['last_evaluation_day'])
            first_day_data = last_day_data - dt.timedelta(days=(conf['evaluation_days'] + running_lag_shift_days))

        first_prediction_day_data = None

    elif ml_framework_step == 'training':
        if conf['last_training_day'] == 'yesterday':
            first_day_data = pd.to_datetime(pd.Timestamp.today().date() -
                                            pd.Timedelta(days=(conf['training_days'] + running_lag_shift_days + 1)))
            last_day_data = pd.to_datetime(pd.Timestamp.today().date() -
                                           pd.Timedelta(days=1))
        else:
            last_day_data = pd.to_datetime(conf['last_training_day'])
            first_day_data = last_day_data - dt.timedelta(days=(conf['training_days'] + running_lag_shift_days))

        first_prediction_day_data = None

    elif ml_framework_step == 'predictions':

        if conf['first_prediction_day'] == 'today':
            first_prediction_day_data = pd.Timestamp.today().date()
        else:
            first_prediction_day_data = pd.Timestamp(conf['first_prediction_day'])
        first_day_data = pd.to_datetime(first_prediction_day_data - pd.Timedelta(days=running_lag_shift_days))
        last_day_data = first_prediction_day_data + dt.timedelta(days=conf['prediction_days'])

    else:
        raise Exception(f"ML framework step {ml_framework_step} not known.")

    return first_day_data, last_day_data, first_prediction_day_data


def create_evaluation_metrics_df(
    cv_results,
    scorers_names: list
) -> pd.DataFrame:

    """
    Creates dataframe with all the metrics obtained in evaluation process for all scorers used in the evaluation
    (defined in scorers_names). Metrics provided include mean errors, standard deviations, ranks for each error,
    fitting and scoring times.
    Note: the actual scores obtained are negative, so we need to negate them, this is described here:
    https://stackoverflow.com/questions/21050110/sklearn-gridsearchcv-with-pipeline
    Args:
        cv_results: results of CV (e.g.GridSearchCV - grid_result.cv_results_)
        scorers_names: list with scorer names used in evaluation

    Returns:
        eval_results_df: evaluation results
    """

    # Get the list with parameters used in evaluation process
    cols_eval_results = list(cv_results['params'][0].keys())

    # Add columns we want to save for each scorer
    for scorer_name in scorers_names:
        cols_eval_results.append(f"mean_test_{scorer_name}")
        cols_eval_results.append(f"rank_test_{scorer_name}")
        cols_eval_results.append(f"std_test_{scorer_name}")

    # Add time metrics
    cols_eval_results.append(f"mean_fit_time")
    cols_eval_results.append(f"mean_score_time")

    # Iterate over every hyperparams set, save results
    eval_results = []
    for ind, params in enumerate(cv_results['params']):
        eval_result = []
        eval_result.append(list(params.values()))

        # Iterate over every scorer to get its metrics, multiscorers use different naming convention
        for scorer_name in scorers_names:
            eval_result[0] = eval_result[0] + [(-1)*cv_results[f"mean_test_{scorer_name}"][ind]]
            eval_result[0] = eval_result[0] + [cv_results[f"rank_test_{scorer_name}"][ind]]
            eval_result[0] = eval_result[0] + [cv_results[f"std_test_{scorer_name}"][ind]]

        eval_result[0] = eval_result[0] + [cv_results[f"mean_fit_time"][ind]]
        eval_result[0] = eval_result[0] + [cv_results[f"mean_score_time"][ind]]

        eval_results.append(eval_result[0])

    # Create df for saving evaluation results
    eval_results_df = pd.DataFrame(eval_results, columns=cols_eval_results)

    return eval_results_df


def get_wfv_indexes(
    eval_data: pd.DataFrame,
    col_wfv: str,
    validation_perc: [float, bool],
    validation_wfv_units: [int, bool]
) -> list:

    """
    Creates a Walk Forward Validation (WFV) indexes. Each of the lists inside this list contains the
    indexes of eval_data to be used for validation (ie. other will be used for data training) in each step of
    WFV process. Each next of the wfv_indexes lists is shorter by one col_wfv unit (e.g. day).
    This indexes will be used to yield a generator object to be passed on to GridSearchCV.

    Example: for data with 10 unique days (01.Jan to 10.Jan), and validation_perc=0.3, wfv_indexes will contain
    3 lists:
        - with indexes of all the eval_data from 8th to 10th Jan
        - with indexes of all the eval_data from 9th to 10th Jan
        - with indexes of all the eval_data from 10th Jan
    Args:
        eval_data: data for the Walk Forward Evaluation split
        col_wfv: column on which WFV will be based
        validation_perc: part of unique_wfv_units to use for WFV, only if validation_wfv_units = None
        validation_wfv_units: how many unique_wfv_units (max) to use for WFV

    Returns:
        wfv_indexes: (WFV) indexes  list with lists of (pandas.core.indexes.numeric.Int64Index) type.
    """

    # Find unique values of the col_wfv (e.g. days), sorted ascending
    unique_wfv_units = list(eval_data[col_wfv].unique())
    unique_wfv_units.sort()

    wfv_indexes = []

    # How many unique wfv units (e.g. days) for validation and training
    if validation_wfv_units:
        validation_max_size = validation_wfv_units
    else:
        validation_max_size = round(validation_perc * len(unique_wfv_units))
    train_min_size = len(unique_wfv_units) - validation_max_size

    for ind in range(validation_max_size):

        # Create a list with unique_wfv_units to use in this iteration
        validation_indexes = unique_wfv_units[:(train_min_size + ind)]
        validation_rows = eval_data[eval_data[col_wfv].isin(validation_indexes)].index

        wfv_indexes += [validation_rows]

    # Drop empty lists
    wfv_indexes = [x for x in wfv_indexes if len(x) > 0]

    return wfv_indexes


def wf_evaluate_rf_model(
    X_eval: pd.DataFrame,
    y_eval: pd.DataFrame,
    wfv_generator,
    scorers: dict,
    scorers_names: list,
    conf: dict
) -> pd.DataFrame:

    """
    Evaluates Random Forest model using Walk Forward validation (suitable for Time Series). param_grid is created
    using config dict.
    Args:
        X_eval: features data
        y_eval: target variable data
        wfv_generator: a generator object to be passed to GridSearchCV
        scorers: dict with scorers
        scorers_names: list with scorer names
        conf: config as dict

    Returns:
        evaluation_results: evaluation results df
    """

    param_grid = {
        'n_estimators': conf['rf_n_estimators'],
        'max_depth': conf['rf_max_depth'],
        'min_samples_leaf': conf['rf_min_samples_leaf'],
        'max_features': conf['rf_max_features']
    }

    rf_eval_model_pipeline = GridSearchCV(
        estimator=RandomForestRegressor(random_state=23, n_jobs=-1),
        param_grid=param_grid,
        cv=wfv_generator,
        scoring=scorers,
        refit=False
    )

    grid_result = rf_eval_model_pipeline.fit(
        X_eval,
        y_eval.values.ravel()
    )

    evaluation_results = create_evaluation_metrics_df(
        cv_results=grid_result.cv_results_,
        scorers_names=scorers_names
    )

    if len(scorers_names) > 1:
        evaluation_results = md_scr.calculate_combined_scorers_metrics(
            evaluation_results,
            scorers_names
        )

    return evaluation_results


def wf_train_rf_model(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        best_hyperparams_grid: dict,
        conf: dict
) -> [pd.DataFrame, Pipeline]:

    """
    Trains model pipeline and predicts on training data.
    Args:
        X_train: training features
        y_train: training target
        best_hyperparams_grid: the best RF model hyperparameters
        conf: config as dict

    Returns:
        y_hats: predictions on training data(dataframe with one column: conf['col_predict']) and the same index as
        X_pred.
        model_pipeline: trained model pipeline
    """

    model_definition = RandomForestRegressor(
        max_depth=best_hyperparams_grid['max_depth'],
        max_features=best_hyperparams_grid['max_features'],
        n_estimators=best_hyperparams_grid['n_estimators'],
        min_samples_leaf=best_hyperparams_grid['min_samples_leaf'],
        random_state=23,
        n_jobs=10
    )
    model_pipeline = Pipeline([('regression', model_definition)])
    model_pipeline.fit(X_train, y_train.values.ravel())

    # Predict on training data
    y_hats = model_pipeline.predict(X_train)

    # Create dataframe with results indexed with cols_ids
    y_hats = pd.DataFrame(y_hats, columns=[conf['col_predict']])
    y_hats.index = y_train.index

    # For negative values we predict 0
    y_hats.loc[y_hats[conf['col_predict']] < 0, conf['col_predict']] = 0

    return y_hats, model_pipeline


def wf_predict_rf_model(
    X_pred: pd.DataFrame,
    model_pipeline: Pipeline,
    conf: dict
) -> pd.DataFrame:

    """
    Creates prediction for X_pred dataframe using trained model_pipeline (trained with the same features as in
    X_pred).
    Args:
        X_pred: prediction features
        model_pipeline: trained model pipeline
        conf: config as dict

    Returns:
        y_hats: predictions (dataframe with one column: conf['col_predict']) and the same index as X_pred.
    """

    y_hats = model_pipeline.predict(X_pred)
    y_hats = pd.DataFrame(y_hats,
                          columns=[conf['col_predict']],
                          index=X_pred.index)

    return y_hats


def wf_split_evaluation_data(
    data: pd.DataFrame,
    cols_features_full: list,
    col_target: str,
    conf: dict,
    logger: logging.Logger
) -> [pd.DataFrame, pd.DataFrame, Generator]:

    """
    Splits data for WFV evaluation.
    Args:
        data: data to be split
        cols_features_full: all the features list
        col_target: target variable
        conf: config as dict
        logger: logger object

    Returns:
        X_eval: features data
        y_eval: target variable data
        wfv_generator: a generator object to be passed to GridSearchCV
    """

    if not isinstance(data, pd.DataFrame):
        raise Exception("Input 'data' must be a pandas DataFrame")
    if not set(cols_features_full).issubset(set(data.columns)):
        raise Exception("Feature columns provided unavailable in input DataFrame")
    if conf['col_date'] not in data.columns:
        raise Exception("Date column missing in input DataFrame")
    if col_target not in data.columns:
        raise Exception("Target column missing in input DataFrame")

    # Split data
    X_eval = (data[cols_features_full + conf['cols_id']]
              .sample(frac=conf['evaluation_clustered_sample_size'], random_state=23)
              .dropna(how='any'))

    dt_trf.low_days_warning(
        data=X_eval,
        col_date=conf['col_date'],
        min_day_filter=conf["evaluation_days"],
        logger=logger
    )

    x_index = X_eval.index
    y_eval = data.loc[x_index, [col_target] + conf['cols_id']]
    X_eval = X_eval.reset_index(drop=True)

    # Get WFV indexes
    wfv_indexes = get_wfv_indexes(
        eval_data=X_eval,
        col_wfv=conf['col_date'],
        validation_perc=False,
        validation_wfv_units=conf['validation_wfv_units']
    )

    wfv_generator = yield_wfv(wfv_indexes)

    X_eval = X_eval.set_index(conf['cols_id']).sort_index()
    y_eval = y_eval.set_index(conf['cols_id']).sort_index()

    return X_eval, y_eval, wfv_generator


def wf_split_training_data(
    data: pd.DataFrame,
    cols_features_full: list,
    col_target: str,
    conf: dict,
    logger: logging.Logger
) -> [pd.DataFrame, pd.DataFrame]:

    """
    Splits data for WFV training.
    Args:
        data: data to be split
        cols_features_full: all the features list
        col_target: target variable
        conf: config as dict
        logger: logger object
    Returns:
        X_train: training features data
        y_train: training target data
    """

    if not isinstance(data, pd.DataFrame):
        raise Exception("Input 'data' must be a pandas DataFrame")
    if not set(cols_features_full).issubset(set(data.columns)):
        raise Exception("Feature columns provided unavailable in input DataFrame")
    if conf['col_date'] not in data.columns:
        raise Exception("Date column missing in input DataFrame")
    if col_target not in data.columns:
        raise Exception("Target column missing in input DataFrame")

    X_train = (data[cols_features_full + conf['cols_id']]
               .sample(frac=conf['training_clustered_sample_size'], random_state=23)
               .dropna(how='any'))

    dt_trf.low_days_warning(
        data=X_train,
        col_date=conf['col_date'],
        min_day_filter=conf["training_days"],
        logger=logger
    )

    x_index = X_train.index
    y_train = data.loc[x_index, [col_target] + conf['cols_id']]

    X_train = X_train.set_index(conf['cols_id']).sort_index()
    y_train = y_train.set_index(conf['cols_id']).sort_index()

    return X_train, y_train


def wf_split_prediction_data(
    data: pd.DataFrame,
    cols_features_full: list,
    col_target: str,
    first_prediction_day_data,
    conf: dict,
    logger: logging.Logger
) -> [pd.DataFrame, pd.DataFrame]:

    """
    Splits data for WFV predictions.
    Args:
        data: data to be split
        cols_features_full: all the features list
        col_target: target variable
        first_prediction_day_data: first prediction day
        conf: config as dict
        logger: logger object
    Returns:
        X_pred: prediction features data
    """

    if not isinstance(data, pd.DataFrame):
        raise Exception("Input 'data' must be a pandas DataFrame")
    if not set(cols_features_full).issubset(set(data.columns)):
        raise Exception("Feature columns provided unavailable in input DataFrame")
    if conf['col_date'] not in data.columns:
        raise Exception("Date column missing in input DataFrame")
    if col_target not in data.columns:
        raise Exception("Target column missing in input DataFrame")

    # Fill NA values
    data = dt_trf.fill_time_series_data_nans(
        data=data,
        cols_to_fillna=conf['cols_to_fillna'],
        col_date=conf['col_date'],
        n_days_for_fillna=conf['n_days_for_fillna']
    )

    # Create test data
    time_filter = data[conf['col_date']] >= first_prediction_day_data
    X_pred = data.loc[time_filter, cols_features_full + conf['cols_id']]

    dt_trf.low_days_warning(
        data=X_pred.copy().dropna(how='any'),
        col_date=conf['col_date'],
        min_day_filter=conf["prediction_days"],
        logger=logger
    )

    X_pred = X_pred.set_index(conf['cols_id']).sort_index()
    X_pred = X_pred.dropna(how='any')

    return X_pred
