import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from lib.model_utils import calculate_model_data_time_period, get_wfv_indexes, create_evaluation_metrics_df, \
wf_split_evaluation_data, wf_split_training_data, wf_split_prediction_data, wf_evaluate_rf_model

import create_dataframes_blueprints as blp
import lib.model_scorers as md_scr
import lib.model_utils as md_uts

col_lon_data = 'event_longitude'
col_lat_data = 'event_latitude'
col_date_data = 'event_date'
col_timestamp_data = 'event_timestamp'
col_lon_clusters = 'cluster_longitude'
col_lat_clusters = 'cluster_latitude'
cols_coords_clusters = [col_lon_clusters, col_lat_clusters]
cols_coords_data = [col_lon_data, col_lat_data]
col_cluster_id = 'cluster_id'
col_cluster_id_prefix = 'Dynamic-DP-'
col_demand = 'event_count'
col_count_clusters = 'cluster_size'

col_default_numerical = 'col_numeric_data'
col_default_numerical_1 = 'col_numeric_data_1'
col_default_numerical_2 = 'col_numeric_data_2'
col_default_string = 'col_string'
col_default_int = 'col_int'
col_default_float = 'col_float'
col_default_date = 'col_date'
col_default_another_1 = 'col_another_1'
col_default_another_2 = 'col_another_2'
col_default_target = 'col_target'
col_months = "months"
col_dow = 'dow'


def test_calculate_model_data_time_period():

    # SETUP
    test_conf_1 = {
        'evaluation_days': 365,
        'training_days': 365,
        'prediction_days': 7,
        'running_metric_days': [28],
        'lags_days': [7],
        'last_evaluation_day': "today",
        'last_training_day': "today",
        'first_prediction_day': "today"
    }
    running_lag_shift_days_1 = max(test_conf_1['running_metric_days'] + test_conf_1['lags_days'])

    test_conf_2 = {
        'evaluation_days': 30,
        'training_days': 30,
        'prediction_days': 14,
        'running_metric_days': [7],
        'lags_days': [28],
        'last_evaluation_day': "today",
        'last_training_day': "today",
        'first_prediction_day': "today"
    }
    running_lag_shift_days_2 = max(test_conf_2['running_metric_days'] + test_conf_2['lags_days'])

    test_conf_3 = {
        'evaluation_days': 395,
        'training_days': 395,
        'prediction_days': 31,
        'running_metric_days': [31],
        'lags_days': [31],
        'last_evaluation_day': "today",
        'last_training_day': "today",
        'first_prediction_day': "today"
    }
    running_lag_shift_days_3 = max(test_conf_3['running_metric_days'] + test_conf_3['lags_days'])

    expected_first_day_data_evaluation_1 = pd.to_datetime(pd.Timestamp.today().date() - pd.Timedelta(days=(365 + 28)))
    expected_last_day_data_evaluation_1 = pd.to_datetime(pd.Timestamp.today().date())
    expected_first_day_data_evaluation_2 = pd.to_datetime(pd.Timestamp.today().date() - pd.Timedelta(days=(30 + 28)))
    expected_last_day_data_evaluation_2 = pd.to_datetime(pd.Timestamp.today().date())

    expected_first_day_data_predictions_1 = pd.to_datetime(pd.Timestamp.today().date() -
                                    pd.Timedelta(days=(test_conf_1['training_days'] + running_lag_shift_days_1)))
    expected_last_day_data_predictions_1 = pd.to_datetime(pd.Timestamp.today().date() + pd.Timedelta(days=(7)))
    expected_first_day_data_predictions_3 = pd.to_datetime(pd.Timestamp.today().date() -
                                    pd.Timedelta(days=(test_conf_3['training_days'] + running_lag_shift_days_3)))
    expected_last_day_data_predictions_3 = pd.to_datetime(pd.Timestamp.today().date() + pd.Timedelta(days=(31)))
    expected_first_prediction_day_data = pd.to_datetime(pd.Timestamp.today().date())

    # VERIFY
    first_day_data_evaluation_1, last_day_data_evaluation_1, _ = calculate_model_data_time_period(
        ml_framework_step='evaluation',
        conf=test_conf_1
    )

    first_day_data_evaluation_2, last_day_data_evaluation_2, _ = calculate_model_data_time_period(
        ml_framework_step='evaluation',
        conf=test_conf_2
    )

    first_day_data_training_1, last_day_data_training_1, _ = calculate_model_data_time_period(
        ml_framework_step='training',
        conf=test_conf_1
    )

    first_day_data_training_2, last_day_data_training_2, _ = calculate_model_data_time_period(
        ml_framework_step='training',
        conf=test_conf_2
    )

    first_day_data_predictions_1, last_day_data_predictions_1, first_prediction_day_data_1 =\
        calculate_model_data_time_period(
        ml_framework_step='predictions',
        conf=test_conf_1
    )

    first_day_data_predictions_3, last_day_data_predictions_3, first_prediction_day_data_3 =\
        calculate_model_data_time_period(
        ml_framework_step='predictions',
        conf=test_conf_3
    )

    # ASSERT
    assert first_day_data_evaluation_1 == expected_first_day_data_evaluation_1
    assert last_day_data_evaluation_1 == expected_last_day_data_evaluation_1
    assert first_day_data_evaluation_2 == expected_first_day_data_evaluation_2
    assert last_day_data_evaluation_2 == expected_last_day_data_evaluation_2
    assert first_day_data_training_1 == expected_first_day_data_evaluation_1
    assert last_day_data_training_1 == expected_last_day_data_evaluation_1
    assert first_day_data_training_2 == expected_first_day_data_evaluation_2
    assert last_day_data_training_2 == expected_last_day_data_evaluation_2

    assert first_day_data_predictions_1 == expected_first_day_data_predictions_1
    assert last_day_data_predictions_1 == expected_last_day_data_predictions_1
    assert first_prediction_day_data_1 == expected_first_prediction_day_data
    assert first_day_data_predictions_3 == expected_first_day_data_predictions_3
    assert last_day_data_predictions_3 == expected_last_day_data_predictions_3
    assert first_prediction_day_data_3 == expected_first_prediction_day_data


def test_get_wfv_indexes():

    # SETUP
    # 10 rows, 10 days, ordered
    eval_data, _ = blp.create_test_dataframe(
        case=6,
        n_rows=10
    )

    # 25 rows, 5 days, ordered
    eval_data_2, _ = blp.create_test_dataframe(
        case=7,
        n_rows=5
    )

    # EXECUTE
    wfv_indexes_1 = get_wfv_indexes(
        eval_data,
        col_wfv=col_date_data,
        validation_perc=None,
        validation_wfv_units=5
    )

    wfv_indexes_2 = get_wfv_indexes(
        eval_data,
        col_wfv=col_date_data,
        validation_perc=0.5,
        validation_wfv_units=None
    )

    wfv_indexes_3 = get_wfv_indexes(
        eval_data,
        col_wfv=col_date_data,
        validation_perc=None,
        validation_wfv_units=1
    )

    wfv_indexes_4 = get_wfv_indexes(
        eval_data,
        col_wfv=col_date_data,
        validation_perc=0.1,
        validation_wfv_units=None
    )

    wfv_indexes_5 = get_wfv_indexes(
        eval_data_2,
        col_wfv=col_date_data,
        validation_perc=None,
        validation_wfv_units=2
    )

    # 0.4 of 5 days are 2 days
    wfv_indexes_6 = get_wfv_indexes(
        eval_data_2,
        col_wfv=col_date_data,
        validation_perc=0.4,
        validation_wfv_units=None
    )

    wfv_indexes_7 = get_wfv_indexes(
        eval_data_2,
        col_wfv=col_date_data,
        validation_perc=None,
        validation_wfv_units=1
    )

    wfv_indexes_8 = get_wfv_indexes(
        eval_data_2,
        col_wfv=col_date_data,
        validation_perc=0.2,
        validation_wfv_units=None
    )

    # VERIFY:
    for wfv_index in wfv_indexes_1:
        assert type(wfv_index) == pd.core.indexes.numeric.Int64Index

    assert list(wfv_indexes_1[0]) == [0, 1, 2, 3, 4]
    assert list(wfv_indexes_1[1]) == [0, 1, 2, 3, 4, 5]
    assert list(wfv_indexes_1[2]) == [0, 1, 2, 3, 4, 5, 6]
    assert list(wfv_indexes_1[3]) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert list(wfv_indexes_1[4]) == [0, 1, 2, 3, 4, 5, 6, 7, 8]

    for wfv_index in wfv_indexes_2:
        assert type(wfv_index) == pd.core.indexes.numeric.Int64Index

    assert list(wfv_indexes_2[0]) == [0, 1, 2, 3, 4]
    assert list(wfv_indexes_2[1]) == [0, 1, 2, 3, 4, 5]
    assert list(wfv_indexes_2[2]) == [0, 1, 2, 3, 4, 5, 6]
    assert list(wfv_indexes_2[3]) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert list(wfv_indexes_2[4]) == [0, 1, 2, 3, 4, 5, 6, 7, 8]

    for wfv_index in wfv_indexes_3:
        assert type(wfv_index) == pd.core.indexes.numeric.Int64Index
    assert list(wfv_indexes_3[0]) == [0, 1, 2, 3, 4, 5, 6, 7, 8]

    for wfv_index in wfv_indexes_2:
        assert type(wfv_index) == pd.core.indexes.numeric.Int64Index
    assert list(wfv_indexes_4[0]) == [0, 1, 2, 3, 4, 5, 6, 7, 8]

    for wfv_index in wfv_indexes_5:
        assert type(wfv_index) == pd.core.indexes.numeric.Int64Index

    assert list(wfv_indexes_5[0]) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    assert list(wfv_indexes_5[1]) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    for wfv_index in wfv_indexes_6:
        assert type(wfv_index) == pd.core.indexes.numeric.Int64Index

    assert list(wfv_indexes_6[0]) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    assert list(wfv_indexes_6[1]) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    for wfv_index in wfv_indexes_7:
        assert type(wfv_index) == pd.core.indexes.numeric.Int64Index

    assert list(wfv_indexes_7[0]) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    for wfv_index in wfv_indexes_8:
        assert type(wfv_index) == pd.core.indexes.numeric.Int64Index

    assert list(wfv_indexes_8[0]) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


def test_wf_evaluate_rf_model():

    # SETUP
    conf_1 = {
        'col_date': col_date_data,
        'cols_id': [col_cluster_id, col_date_data],
        'evaluation_clustered_sample_size': 1.0,
        'validation_wfv_units': 2,
        'rf_n_estimators': [100],
        'rf_max_depth': [10],
        'rf_min_samples_leaf': [1],
        'rf_max_features': [1.0]
    }

    conf_2 = {
        'col_date': col_date_data,
        'cols_id': [col_cluster_id, col_date_data],
        'evaluation_clustered_sample_size': 1.0,
        'validation_wfv_units': 2,
        'rf_n_estimators': [10, 100],
        'rf_max_depth': [5, 10],
        'rf_min_samples_leaf': [1],
        'rf_max_features': [1.0]
    }

    scorers_names_1 = ['rmse']
    scorers_names_2 = ['rmse', 'mae']
    model_data, _ = blp.create_test_dataframe(
        case='full_model_data',
        n_rows=5
    )

    scorers_1 = md_scr.make_evaluation_scorers(
        scorers_names=scorers_names_1
    )
    scorers_2 = md_scr.make_evaluation_scorers(
        scorers_names=scorers_names_2
    )
    cols_features_full = [col_default_numerical_1, col_default_numerical_2]
    col_target = col_default_target

    expected_columns_1 = {'max_depth', 'max_features', 'min_samples_leaf', 'n_estimators', 'mean_test_rmse',
                          'rank_test_rmse', 'std_test_rmse', 'mean_fit_time', 'mean_score_time'}
    expected_columns_2 = {'max_depth', 'max_features', 'min_samples_leaf', 'n_estimators', 'mean_test_rmse',
                          'rank_test_rmse', 'std_test_rmse', 'mean_test_mae', 'rank_test_mae', 'std_test_mae',
                          'mean_fit_time', 'mean_score_time', 'mean_test_rmse_normalized', 'mean_test_mae_normalized',
                          'mean_test_combined_normalized', 'rank_test_combined_normalized'}

    # EXECUTE
    X_eval, y_eval, wfv_generator = md_uts.wf_split_evaluation_data(
        data=model_data,
        cols_features_full=cols_features_full,
        col_target=col_target,
        conf=conf_1
    )

    evaluation_results_1 = wf_evaluate_rf_model(
        X_eval.copy(),
        y_eval.copy(),
        wfv_generator,
        scorers_1,
        scorers_names_1,
        conf_1
    )

    X_eval, y_eval, wfv_generator = md_uts.wf_split_evaluation_data(
        data=model_data,
        cols_features_full=cols_features_full,
        col_target=col_target,
        conf=conf_2
    )
    evaluation_results_2 = wf_evaluate_rf_model(
        X_eval.copy(),
        y_eval.copy(),
        wfv_generator,
        scorers_1,
        scorers_names_1,
        conf_2
    )

    X_eval, y_eval, wfv_generator = md_uts.wf_split_evaluation_data(
        data=model_data,
        cols_features_full=cols_features_full,
        col_target=col_target,
        conf=conf_1
    )

    evaluation_results_3 = wf_evaluate_rf_model(
        X_eval.copy(),
        y_eval.copy(),
        wfv_generator,
        scorers_2,
        scorers_names_2,
        conf_1
    )

    X_eval, y_eval, wfv_generator = md_uts.wf_split_evaluation_data(
        data=model_data,
        cols_features_full=cols_features_full,
        col_target=col_target,
        conf=conf_2
    )

    evaluation_results_4 = wf_evaluate_rf_model(
        X_eval.copy(),
        y_eval.copy(),
        wfv_generator,
        scorers_2,
        scorers_names_2,
        conf_2
    )

    # VERIFY
    assert evaluation_results_1.shape[0] == 1
    assert expected_columns_1.issubset(evaluation_results_1.columns)
    assert (evaluation_results_1['mean_test_rmse'] >= 0).all()
    assert (evaluation_results_1['rank_test_rmse'] >= 1).all()

    assert evaluation_results_2.shape[0] == 4
    assert expected_columns_1.issubset(evaluation_results_2.columns)
    assert (evaluation_results_2['mean_test_rmse'] >= 0).all()
    assert (evaluation_results_2['rank_test_rmse'] >= 1).all()

    assert evaluation_results_3.shape[0] == 1
    assert expected_columns_2.issubset(evaluation_results_3.columns)
    assert (evaluation_results_3['mean_test_rmse'] >= 0).all()
    assert (evaluation_results_3['rank_test_rmse'] >= 1).all()
    assert (evaluation_results_3['mean_test_mae'] >= 0).all()
    assert (evaluation_results_3['rank_test_mae'] >= 1).all()

    assert evaluation_results_4.shape[0] == 4
    assert expected_columns_2.issubset(evaluation_results_4.columns)
    assert (evaluation_results_4['mean_test_rmse'] >= 0).all()
    assert (evaluation_results_4['rank_test_rmse'] >= 1).all()
    assert (evaluation_results_4['mean_test_mae'] >= 0).all()
    assert (evaluation_results_4['rank_test_mae'] >= 1).all()


def test_wf_split_evaluation_data():

    conf = {
        'col_date': col_date_data,
        'cols_id': [col_cluster_id, col_date_data],
        'evaluation_clustered_sample_size': 1.0,
        'validation_wfv_units': 2,
        'rf_n_estimators': [10, 100],
        'rf_max_depth': [5, 10],
        'rf_min_samples_leaf': [1],
        'rf_max_features': [1.0]
    }

    model_data, _ = blp.create_test_dataframe(
        case='full_model_data',
        n_rows=5
    )
    cols_features_full = [col_default_numerical_1, col_default_numerical_2]
    col_target = col_default_target

    X_eval, y_eval, wfv_generator = wf_split_evaluation_data(
        model_data,
        cols_features_full=cols_features_full,
        col_target=col_target,
        conf=conf
    )

    assert X_eval.equals(model_data[cols_features_full + conf['cols_id']].set_index(conf['cols_id']).sort_index())
    assert set(X_eval.index.names) == set(conf['cols_id'])
    assert set(X_eval.columns) == set(cols_features_full)

    assert y_eval.equals(model_data[[col_target] + conf['cols_id']].set_index(conf['cols_id']).sort_index())
    assert set(y_eval.index.names) == set(conf['cols_id'])
    assert set(y_eval.columns) == {col_target}


def test_wf_split_training_data():

    conf = {
        'col_date': col_date_data,
        'cols_id': [col_cluster_id, col_date_data],
        'training_clustered_sample_size': 1.0,
        'validation_wfv_units': 2,
        'rf_n_estimators': [10, 100],
        'rf_max_depth': [5, 10],
        'rf_min_samples_leaf': [1],
        'rf_max_features': [1.0]
    }

    model_data, _ = blp.create_test_dataframe(
        case='full_model_data',
        n_rows=5
    )
    cols_features_full = [col_default_numerical_1, col_default_numerical_2]
    col_target = col_default_target

    X_train, y_train = wf_split_training_data(
        model_data,
        cols_features_full=cols_features_full,
        col_target=col_target,
        conf=conf
    )

    assert X_train.equals(model_data[cols_features_full + conf['cols_id']].set_index(conf['cols_id']).sort_index())
    assert set(X_train.index.names) == set(conf['cols_id'])
    assert set(X_train.columns) == set(cols_features_full)

    assert y_train.equals(model_data[[col_target] + conf['cols_id']].set_index(conf['cols_id']).sort_index())
    assert set(y_train.index.names) == set(conf['cols_id'])
    assert set(y_train.columns) == {col_target}


def test_wf_split_prediction_data():

    cols_features_full = [col_default_numerical_1, col_default_numerical_2]
    col_target = col_default_target
    conf = {
        'col_date': col_date_data,
        'cols_id': [col_cluster_id, col_date_data],
        'training_clustered_sample_size': 1.0,
        'validation_wfv_units': 2,
        'rf_n_estimators': [10, 100],
        'rf_max_depth': [5, 10],
        'rf_min_samples_leaf': [1],
        'rf_max_features': [1.0],
        'cols_to_fillna': cols_features_full,
        'n_days_for_fillna': 2
    }

    model_data, _ = blp.create_test_dataframe(
        case='full_model_data',
        n_rows=5
    )

    # Third out of 5 days
    first_prediction_day_data = model_data[conf['col_date']].unique()[2]

    X_test = wf_split_prediction_data(
        model_data,
        cols_features_full=cols_features_full,
        col_target=col_target,
        first_prediction_day_data=first_prediction_day_data,
        conf=conf
    )

    assert X_test.shape[0] == 3 * 5
    assert set(X_test.index.names) == set(conf['cols_id'])
    assert set(X_test.columns) == set(cols_features_full)


def test_create_evaluation_metrics_df():

    # Note: this test and test_wf_evaluate_rf_model are very similar (these two functions are dependent)
    scorers_names = ['rmse', 'mae']
    model_data, _ = blp.create_test_dataframe(
        case='full_model_data',
        n_rows=5
    )

    scorers = md_scr.make_evaluation_scorers(
        scorers_names=scorers_names
    )
    cols_features_full = [col_default_numerical_1, col_default_numerical_2]
    col_target = col_default_target

    expected_columns = {'max_depth', 'max_features', 'min_samples_leaf', 'n_estimators', 'mean_test_rmse',
                        'rank_test_rmse', 'std_test_rmse', 'mean_fit_time', 'mean_score_time'}

    conf = {
        'col_date': col_date_data,
        'cols_id': [col_cluster_id, col_date_data],
        'training_clustered_sample_size': 1.0,
        'validation_wfv_units': 2,
        'evaluation_clustered_sample_size': 1.0,
        'rf_n_estimators': [10, 100],
        'rf_max_depth': [5, 10],
        'rf_min_samples_leaf': [1],
        'rf_max_features': [1.0]
    }

    param_grid = {
        'n_estimators': conf['rf_n_estimators'],
        'max_depth': conf['rf_max_depth'],
        'min_samples_leaf': conf['rf_min_samples_leaf'],
        'max_features': conf['rf_max_features']
    }

    # EXECUTE
    X_eval, y_eval, wfv_generator = md_uts.wf_split_evaluation_data(
        data=model_data,
        cols_features_full=cols_features_full,
        col_target=col_target,
        conf=conf
    )

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

    assert evaluation_results.shape[0] == 4
    assert expected_columns.issubset(evaluation_results.columns)
    assert (evaluation_results['mean_test_rmse'] >= 0).all()
    assert (evaluation_results['rank_test_rmse'] >= 1).all()