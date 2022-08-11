
import numpy as np
import pandas as pd

import create_dataframes_blueprints as blp
import lib.base_utils as bs_uts

from lib.data_transformations import add_calendar_features, add_day_lag_variables, add_day_running_metrics,\
    add_day_trend_variable, fill_time_series_data_nans, amplify_variable, calc_rides_per_scoot_in_period

number_of_days = 14
col_lon_data = 'event_longitude'
col_lat_data = 'event_latitude'
col_lon_clusters = 'cluster_longitude'
col_lat_clusters = 'cluster_latitude'
cols_coords_clusters = [col_lon_clusters, col_lat_clusters]
cols_coords_data = [col_lon_data, col_lat_data]
col_cluster_id = 'cluster_id'
col_cluster_id_prefix = 'Dynamic-DP-'
col_distance = 'distance'
col_count_data = 'event_count'
col_count_clusters = 'cluster_size'
col_date_data = 'event_date'
col_default_numerical = 'col_numeric_data'
col_default_numerical_1 = 'col_numeric_data_1'
col_default_numerical_2 = 'col_numeric_data_2'

col_timestamp_data = 'event_timestamp'
col_demand = 'event_count'
ol_default_string = 'col_string'
col_default_int = 'col_int'
col_default_float = 'col_float'
col_default_date = 'col_date'
col_default_another_1 = 'col_another_1'
col_default_another_2 = 'col_another_2'
col_default_target = 'col_target'
col_months = "months"
col_dow = 'dow'

# Setup logger
logger = bs_uts.logger_setup(
    log_file='log_file.log'
)

def test_add_calendar_features():

    # SETUP:
    input_data, _ = blp.create_test_dataframe(
        case=6,
        n_rows=number_of_days
    )
    
    expected_columns_data_1 = {
        col_date_data,
        col_default_numerical,
        col_date_data + '_iso_day_of_week',
        col_date_data + '_iso_calendar_week',
        col_date_data + '_iso_month'
    }
    expected_columns_data_2 = {
        col_date_data,
        col_default_numerical,
        col_date_data + '_iso_day_of_week'
    }
    
    # EXECUTE:
    output_data_1, _, _, _ = add_calendar_features(
        input_data,
        col_date_data,
        calendar_features=['iso_day_of_week', 'iso_calendar_week', 'iso_month'],
        cols_numerical=[],
        cols_categorical=[],
        cols_cyclical_in=[]
    )

    output_data_2, _, _, _ = add_calendar_features(
        input_data,
        col_date_data,
        calendar_features=['iso_day_of_week', 'day_of_month'],
        cols_numerical=[],
        cols_categorical=[],
        cols_cyclical_in=[]
    )

    print(output_data_1[col_date_data + '_iso_calendar_week']
          )

    # VERIFY
    # Check expected values
    assert output_data_1.loc[output_data_1[col_date_data] == '1981-05-24',
                             col_date_data + '_iso_day_of_week'].values[0] == 7
    assert output_data_1.loc[output_data_1[col_date_data] == '1981-05-25',
                             col_date_data + '_iso_day_of_week'].values[0] == 1
    assert output_data_1.loc[output_data_1[col_date_data] == '1981-05-24',
                             col_date_data + '_iso_calendar_week'].values[0] == 21
    assert output_data_1.loc[output_data_1[col_date_data] == '1981-06-01',
                             col_date_data + '_iso_calendar_week'].values[0] == 23
    assert output_data_1.loc[output_data_1[col_date_data] == '1981-05-24',
                             col_date_data + '_iso_month'].values[0] == 5
    assert output_data_1.loc[output_data_1[col_date_data] == '1981-06-01',
                             col_date_data + '_iso_month'].values[0] == 6

    # Check row numbers
    assert input_data.shape[0] == output_data_1.shape[0]
    assert input_data.shape[0] == output_data_2.shape[0]

    # Check columns names and types
    assert expected_columns_data_1 == set(output_data_1[expected_columns_data_1].columns)
    assert expected_columns_data_2 == set(output_data_1[expected_columns_data_2].columns)
    assert output_data_1.dtypes[col_date_data + '_iso_day_of_week'] == 'int64'
    assert output_data_1.dtypes[col_date_data + '_iso_calendar_week'] == ('int64') or (pd.UInt32Dtype)
    assert output_data_1.dtypes[col_date_data + '_iso_month'] == 'int64'
    assert output_data_1.dtypes[col_date_data] == '<M8[ns]'



def test_add_day_lag_variables_no_input_cols():

    # PREPARE:
    col_lag = col_date_data
    cols_lags_vars = [f'{col_default_numerical_1}',
                      f'{col_default_numerical_2}']
    cols_lags_agg = [col_cluster_id]
    lags_days = [1, 3]
    cols_numerical = []
    cols_plot = []

    input_data, expected_output = blp.create_test_dataframe(
        case=7,
        n_rows=number_of_days
    )

    expected_columns_out = (list(input_data.columns) +
                           [f'{col_default_numerical_1}_lag_1',
                            f'{col_default_numerical_1}_lag_3',
                            f'{col_default_numerical_2}_lag_1',
                            f'{col_default_numerical_2}_lag_3'])
    expected_cols_numerical_out = [f'{col_default_numerical_1}_lag_1',
                                   f'{col_default_numerical_1}_lag_3',
                                   f'{col_default_numerical_2}_lag_1',
                                   f'{col_default_numerical_2}_lag_3']
    expected_cols_plot_out = [f'{col_default_numerical_1}_lag_1',
                              f'{col_default_numerical_1}_lag_3',
                              f'{col_default_numerical_2}_lag_1',
                              f'{col_default_numerical_2}_lag_3']

    # EXECUTE:
    output_data, cols_numerical_out, cols_plot_out, cols_lags_out = add_day_lag_variables(
        input_data,
        col_lag,
        cols_lags_vars,
        cols_lags_agg,
        lags_days,
        cols_numerical,
        cols_plot
    )
    assert output_data.loc[output_data[col_date_data].isin(['2020-01-01']),
                           f'{col_default_numerical_1}_lag_1'].isnull().all()
    assert output_data.loc[output_data[col_date_data].isin(['2020-01-01', '2020-01-02', '2020-01-03']),
                           f'{col_default_numerical_1}_lag_3'].isnull().all()

    assert list(output_data.groupby(col_cluster_id)[f'{col_default_numerical_1}_lag_1'].mean()) == \
           [int(x) for x in expected_output[f'{col_default_numerical_1}']]
    assert list(output_data.dropna(subset=[f'{col_default_numerical_1}_lag_1'])
                .groupby(col_lag)[f'{col_default_numerical_1}_lag_1'].mean()) == \
           [3.0] * (expected_output['number_of_days'] - 1)
    assert list(output_data.groupby(col_cluster_id)[f'{col_default_numerical_1}_lag_1'].mean()) == \
           expected_output[f'{col_default_numerical_1}']
    assert list(output_data.dropna(subset=[f'{col_default_numerical_1}_lag_3'])
                .groupby(col_lag)[f'{col_default_numerical_1}_lag_3'].mean()) == \
           [3.0] * (expected_output['number_of_days'] - 3)
    assert list(output_data.groupby(col_cluster_id)[f'{col_default_numerical_2}_lag_1'].mean()) == \
           expected_output[f'{col_default_numerical_2}']
    assert list(output_data.dropna(subset=[f'{col_default_numerical_2}_lag_1'])
                .groupby(col_lag)[f'{col_default_numerical_2}_lag_1'].mean()) == \
           [-6.0] * (expected_output['number_of_days'] - 1)
    assert list(output_data.groupby(col_cluster_id)[f'{col_default_numerical_2}_lag_3'].mean()) == \
           expected_output[f'{col_default_numerical_2}']
    assert list(output_data.dropna(subset=[f'{col_default_numerical_2}_lag_3'])
                .groupby(col_lag)[f'{col_default_numerical_2}_lag_3'].mean()) == \
           [-6.0] * (expected_output['number_of_days'] - 3)
    assert expected_columns_out == list(output_data.columns)

    assert expected_columns_out == list(output_data.columns)
    assert expected_cols_numerical_out == cols_numerical_out
    assert expected_cols_plot_out == cols_plot_out


def test_add_day_lag_variables_2vars_1agg():

    # PREPARE:
    col_lag = col_date_data
    cols_lags_vars = [f'{col_default_numerical_1}', f'{col_default_numerical_2}']
    cols_lags_agg = [col_cluster_id]
    lags_days = [1, 3]
    cols_numerical = ['col_numerical_1', 'col_numerical_2']
    cols_plot = ['col_plot_1']
    input_data, expected_output = blp.create_test_dataframe(
        case=7,
        n_rows=number_of_days
    )

    expected_columns_out = list(input_data.columns) +\
                           [f'{col_default_numerical_1}_lag_1',
                            f'{col_default_numerical_1}_lag_3',
                            f'{col_default_numerical_2}_lag_1',
                            f'{col_default_numerical_2}_lag_3']
    expected_cols_numerical_out = cols_numerical + \
                                  [f'{col_default_numerical_1}_lag_1',
                                   f'{col_default_numerical_1}_lag_3',
                                   f'{col_default_numerical_2}_lag_1',
                                   f'{col_default_numerical_2}_lag_3']
    expected_cols_plot_out = cols_plot + \
                             [f'{col_default_numerical_1}_lag_1',
                              f'{col_default_numerical_1}_lag_3',
                              f'{col_default_numerical_2}_lag_1',
                              f'{col_default_numerical_2}_lag_3']

    # EXECUTE:
    output_data, cols_numerical_out, cols_plot_out, cols_lags_out = add_day_lag_variables(
        input_data,
        col_lag,
        cols_lags_vars,
        cols_lags_agg,
        lags_days,
        cols_numerical,
        cols_plot
    )

    
    assert output_data.loc[output_data[col_date_data].isin(['2020-01-01']),
                           f'{col_default_numerical_1}_lag_1'].isnull().all()
    assert output_data.loc[output_data[col_date_data].isin(['2020-01-01', '2020-01-02', '2020-01-03']),
                           f'{col_default_numerical_1}_lag_3'].isnull().all()
    assert list(output_data.groupby(col_cluster_id)[f'{col_default_numerical_1}_lag_1'].mean()) == \
           [int(x) for x in expected_output[f'{col_default_numerical_1}']]
    assert list(output_data.dropna(subset=[f'{col_default_numerical_1}_lag_1'])
                .groupby(col_lag)[f'{col_default_numerical_1}_lag_1'].mean()) == \
           [3.0] * (expected_output['number_of_days'] - 1)
    assert list(output_data.groupby(col_cluster_id)[f'{col_default_numerical_1}_lag_1'].mean()) == \
           expected_output[f'{col_default_numerical_1}']
    assert list(output_data.dropna(subset=[f'{col_default_numerical_1}_lag_3'])
                .groupby(col_lag)[f'{col_default_numerical_1}_lag_3'].mean()) == \
           [3.0] * (expected_output['number_of_days'] - 3)
    assert list(output_data.groupby(col_cluster_id)[f'{col_default_numerical_2}_lag_1'].mean()) == \
           expected_output[f'{col_default_numerical_2}']
    assert list(output_data.dropna(subset=[f'{col_default_numerical_2}_lag_1'])
                .groupby(col_lag)[f'{col_default_numerical_2}_lag_1'].mean()) == \
           [-6.0] * (expected_output['number_of_days'] - 1)
    assert list(output_data.groupby(col_cluster_id)[f'{col_default_numerical_2}_lag_3'].mean()) == \
           expected_output[f'{col_default_numerical_2}']
    assert list(output_data.dropna(subset=[f'{col_default_numerical_2}_lag_3'])
                .groupby(col_lag)[f'{col_default_numerical_2}_lag_3'].mean()) == \
           [-6.0] * (expected_output['number_of_days'] - 3)
    assert expected_columns_out == list(output_data.columns)

    assert expected_columns_out == list(output_data.columns)
    assert expected_columns_out == list(output_data.columns)
    assert expected_cols_numerical_out == cols_numerical_out
    assert expected_cols_plot_out == cols_plot_out


def test_add_day_lag_variables_2vars_no_agg():

    # PREPARE:
    col_lag = col_date_data
    cols_lags_vars = [f'{col_default_numerical_1}', f'{col_default_numerical_2}']
    cols_lags_agg = []
    lags_days = [1,3]
    cols_numerical = ['col_numerical_1', 'col_numerical_2']
    cols_plot = ['col_plot_1']
    input_data, expected_output = blp.create_test_dataframe(
        case=7,
        n_rows=number_of_days
    )

    expected_columns_out = list(input_data.columns) +\
                           [f'{col_default_numerical_1}_lag_1',
                            f'{col_default_numerical_1}_lag_3',
                            f'{col_default_numerical_2}_lag_1',
                            f'{col_default_numerical_2}_lag_3']

    # EXECUTE:
    output_data, cols_numerical_out, cols_plot_out, cols_lags_out = add_day_lag_variables(
        input_data,
        col_lag,
        cols_lags_vars,
        cols_lags_agg,
        lags_days,
        cols_numerical,
        cols_plot
    )

    # VERIFY
    assert output_data.loc[output_data[col_date_data].isin(['2020-01-01']),
                           f'{col_default_numerical_1}_lag_1'].isnull().all()
    assert output_data.loc[output_data[col_date_data].isin(['2020-01-01', '2020-01-02', '2020-01-03']),
                        f'{col_default_numerical_1}_lag_3'].isnull().all()
    assert list(output_data.groupby(col_cluster_id)[f'{col_default_numerical_1}_lag_1'].mean()) == \
           [3.0] * expected_output['number_of_clusters']
    assert list(output_data.dropna(subset=[f'{col_default_numerical_1}_lag_1'])
                .groupby(col_lag)[f'{col_default_numerical_1}_lag_1'].mean()) == \
           [3.0] * (expected_output['number_of_days'] - 1)
    assert list(output_data.groupby(col_cluster_id)[f'{col_default_numerical_1}_lag_3'].mean()) == \
           [3.0] * expected_output['number_of_clusters']
    assert list(output_data.dropna(subset=[f'{col_default_numerical_1}_lag_3'])
                .groupby(col_lag)[f'{col_default_numerical_1}_lag_3'].mean()) == \
           [3.0] * (expected_output['number_of_days'] - 3)
    assert list(output_data.groupby(col_cluster_id)[f'{col_default_numerical_2}_lag_1'].mean()) == \
           [-6.0] * expected_output['number_of_clusters']
    assert list(output_data.dropna(subset=[f'{col_default_numerical_2}_lag_1'])
                .groupby(col_lag)[f'{col_default_numerical_2}_lag_1'].mean()) == \
           [-6.0] * (expected_output['number_of_days'] - 1)
    assert list(output_data.groupby(col_cluster_id)[f'{col_default_numerical_2}_lag_3'].mean()) == \
           [-6.0] * expected_output['number_of_clusters']
    assert list(output_data.dropna(subset=[f'{col_default_numerical_2}_lag_3'])
                .groupby(col_lag)[f'{col_default_numerical_2}_lag_3'].mean()) == \
           [-6.0] * (expected_output['number_of_days'] - 3)
    assert expected_columns_out == list(output_data.columns)


def test_add_day_lag_variables_1vars_no_agg():

    # PREPARE:
    col_lag = col_date_data
    cols_lags_vars = [f'{col_default_numerical_1}']
    cols_lags_agg = []
    lags_days = [1,3]
    cols_numerical = ['col_numerical_1', 'col_numerical_2']
    cols_plot = []
    input_data, expected_output = blp.create_test_dataframe(
        case=7,
        n_rows=number_of_days
    )

    expected_columns_out = list(input_data.columns) +\
                           [f'{col_default_numerical_1}_lag_1',
                            f'{col_default_numerical_1}_lag_3']

    # EXECUTE:
    output_data, cols_numerical_out, cols_plot_out, cols_lags_out = add_day_lag_variables(
        input_data,
        col_lag,
        cols_lags_vars,
        cols_lags_agg,
        lags_days,
        cols_numerical,
        cols_plot
    )

    # VERIFY
    assert output_data.loc[output_data[col_date_data].isin(['2020-01-01']),
                           f'{col_default_numerical_1}_lag_1'].isnull().all()
    assert output_data.loc[output_data[col_date_data].isin(['2020-01-01', '2020-01-02', '2020-01-03']),
                        f'{col_default_numerical_1}_lag_3'].isnull().all()
    assert list(output_data.groupby(col_cluster_id)[f'{col_default_numerical_1}_lag_1'].mean()) ==\
           [3.0] * expected_output['number_of_clusters']
    assert list(output_data.dropna(subset=[f'{col_default_numerical_1}_lag_1'])
                .groupby(col_lag)[f'{col_default_numerical_1}_lag_1'].mean()) ==\
           [3.0] * (expected_output['number_of_days'] - 1)
    assert list(output_data.groupby(col_cluster_id)[f'{col_default_numerical_1}_lag_3'].mean()) ==\
           [3.0] * expected_output['number_of_clusters']
    assert list(output_data.dropna(subset=[f'{col_default_numerical_1}_lag_3'])
                .groupby(col_lag)[f'{col_default_numerical_1}_lag_3'].mean()) ==\
           [3.0] * (expected_output['number_of_days'] - 3)
    assert expected_columns_out == list(output_data.columns)


def test_add_day_running_metrics_no_input_cols():

    # PREPARE:
    conf = {}
    conf['col_date'] = col_date_data
    conf['use_dow_as_running_agg'] = False
    col_running_metric = conf['col_date']
    cols_running_metric_vars = [col_default_numerical_1, col_default_numerical_2]
    cols_running_metric_agg = [col_cluster_id]
    conf['running_metric_days'] = [1, 3]
    running_metric = 'mean'
    cols_numerical = []
    cols_categorical = [f"{conf['col_date']}_iso_day_of_week"]
    cols_plot = []
    shift = 0
    conf['prediction_days'] = shift

    input_data, expected_output = blp.create_test_dataframe(
        case=7,
        n_rows=number_of_days
    )

    expected_columns_out = list(input_data.columns) + \
                           [f'{col_default_numerical_1}_running_mean_lag_1',
                           f'{col_default_numerical_1}_running_mean_lag_3',
                           f'{col_default_numerical_2}_running_mean_lag_1',
                           f'{col_default_numerical_2}_running_mean_lag_3']
    expected_cols_numerical_out = [f'{col_default_numerical_1}_running_mean_lag_1',
                                   f'{col_default_numerical_1}_running_mean_lag_3',
                                   f'{col_default_numerical_2}_running_mean_lag_1',
                                   f'{col_default_numerical_2}_running_mean_lag_3']
    expected_cols_plot_out = [f'{col_default_numerical_1}_running_mean_lag_1',
                              f'{col_default_numerical_1}_running_mean_lag_3',
                              f'{col_default_numerical_2}_running_mean_lag_1',
                              f'{col_default_numerical_2}_running_mean_lag_3']
    use_dow_as_running_agg = False

    # EXECUTE:
    output_data, cols_numerical_out, cols_plot_out, cols_running_metrics_out = add_day_running_metrics(
        input_data,
        col_running_metric,
        cols_running_metric_vars,
        cols_running_metric_agg,
        use_dow_as_running_agg,
        running_metric,
        cols_numerical,
        cols_categorical,
        cols_plot,
        conf
    )

    assert output_data.loc[
        output_data['event_date'].isin(['2020-01-01', '2020-01-02']),
        f'{col_default_numerical_1}_running_mean_lag_3'].isnull().all()
    assert list(output_data.groupby(cols_running_metric_agg)[f'{col_default_numerical_1}'].mean()) ==\
           expected_output[f'{col_default_numerical_1}']
    assert list(output_data.groupby(cols_running_metric_agg)[f'{col_default_numerical_2}'].mean()) ==\
           expected_output[f'{col_default_numerical_2}']
    assert list(output_data.groupby(col_running_metric)[f'{col_default_numerical_1}'].mean()) ==\
           [3] * expected_output['number_of_days']
    assert list(output_data.groupby(col_running_metric)[f'{col_default_numerical_2}'].mean()) ==\
           [-6] * expected_output['number_of_days']

    assert expected_columns_out == list(output_data.columns)
    assert expected_cols_numerical_out == cols_numerical_out
    assert expected_cols_plot_out == cols_plot_out


def test_add_day_running_metrics_2vars_1agg():

    # PREPARE:
    conf = {}
    conf['col_date'] = col_date_data
    conf['use_dow_as_running_agg'] = False
    col_running_metric = conf['col_date']
    cols_running_metric_vars = [col_default_numerical_1, col_default_numerical_2]
    cols_running_metric_agg = [col_cluster_id]
    conf['running_metric_days'] = [1, 3]
    running_metric = 'mean'
    cols_numerical = []
    cols_categorical = [f"{conf['col_date']}_iso_day_of_week"]
    cols_plot = ['col_plot_1']
    shift = 0
    conf['prediction_days'] = shift

    input_data, expected_output = blp.create_test_dataframe(
        case=7,
        n_rows=number_of_days
    )

    expected_columns_out = list(input_data.columns) + \
                           [f'{col_default_numerical_1}_running_mean_lag_1',
                            f'{col_default_numerical_1}_running_mean_lag_3',
                            f'{col_default_numerical_2}_running_mean_lag_1',
                            f'{col_default_numerical_2}_running_mean_lag_3']
    expected_cols_numerical_out = cols_numerical +\
                                  [f'{col_default_numerical_1}_running_mean_lag_1',
                                   f'{col_default_numerical_1}_running_mean_lag_3',
                                   f'{col_default_numerical_2}_running_mean_lag_1',
                                   f'{col_default_numerical_2}_running_mean_lag_3']
    expected_cols_plot_out = cols_plot +\
                             [f'{col_default_numerical_1}_running_mean_lag_1',
                              f'{col_default_numerical_1}_running_mean_lag_3',
                              f'{col_default_numerical_2}_running_mean_lag_1',
                              f'{col_default_numerical_2}_running_mean_lag_3']
    use_dow_as_running_agg = False

    # EXECUTE:
    output_data, cols_numerical_out, cols_plot_out, cols_running_metrics_out = add_day_running_metrics(
        input_data,
        col_running_metric,
        cols_running_metric_vars,
        cols_running_metric_agg,
        use_dow_as_running_agg,
        running_metric,
        cols_numerical,
        cols_categorical,
        cols_plot,
        conf
    )
    assert output_data.loc[output_data['event_date'].isin(['2020-01-01', '2020-01-02']),
                           f'{col_default_numerical_1}_running_mean_lag_3'].isnull().all()
    assert list(output_data.groupby(cols_running_metric_agg)[f'{col_default_numerical_1}'].mean()) ==\
           expected_output[f'{col_default_numerical_1}']
    assert list(output_data.groupby(cols_running_metric_agg)[f'{col_default_numerical_2}'].mean()) ==\
           expected_output[f'{col_default_numerical_2}']
    assert list(output_data.groupby(col_running_metric)[f'{col_default_numerical_1}'].mean()) ==\
           [3]*expected_output['number_of_days']
    assert list(output_data.groupby(col_running_metric)[f'{col_default_numerical_2}'].mean()) ==\
           [-6]*expected_output['number_of_days']

    assert expected_columns_out == list(output_data.columns)
    assert expected_columns_out == list(output_data.columns)
    assert expected_cols_numerical_out == cols_numerical_out
    assert expected_cols_plot_out == cols_plot_out


def test_add_day_running_metrics_2vars_no_agg():

    # PREPARE:
    conf = {}
    conf['col_date'] = col_date_data
    conf['use_dow_as_running_agg'] = False
    col_running_metric = conf['col_date']
    cols_running_metric_vars = [col_default_numerical_1, col_default_numerical_2]
    cols_running_metric_agg = []
    conf['running_metric_days'] = [1, 3]
    running_metric = 'mean'
    cols_numerical = []
    cols_categorical = [f"{conf['col_date']}_iso_day_of_week"]
    cols_plot = ['col_plot_1']
    shift = 0
    conf['prediction_days'] = shift

    input_data, expected_output = blp.create_test_dataframe(
        case=7,
        n_rows=number_of_days
    )
    expected_columns_out = list(input_data.columns) +\
                           [f'{col_default_numerical_1}_running_mean_lag_1',
                            f'{col_default_numerical_1}_running_mean_lag_3',
                            f'{col_default_numerical_2}_running_mean_lag_1',
                            f'{col_default_numerical_2}_running_mean_lag_3']
    use_dow_as_running_agg = False

    # EXECUTE:
    output_data, cols_numerical_out, cols_plot_out, cols_running_metrics_out = add_day_running_metrics(
        input_data,
        col_running_metric,
        cols_running_metric_vars,
        cols_running_metric_agg,
        use_dow_as_running_agg,
        running_metric,
        cols_numerical,
        cols_categorical,
        cols_plot,
        conf
    )
    assert output_data.loc[output_data['event_date'].isin(['2020-01-01', '2020-01-02']),
                           f'{col_default_numerical_1}_running_mean_lag_3'].isnull().all()
    assert list(output_data.groupby('cluster_id')[f'{col_default_numerical_1}'].mean()) ==\
           expected_output[f'{col_default_numerical_1}']
    assert list(output_data.groupby('cluster_id')[f'{col_default_numerical_2}'].mean()) ==\
           expected_output[f'{col_default_numerical_2}']
    assert list(output_data.groupby(col_running_metric)[f'{col_default_numerical_1}'].mean()) ==\
           [3]*expected_output['number_of_days']
    assert list(output_data.groupby(col_running_metric)[f'{col_default_numerical_2}'].mean()) ==\
           [-6]*expected_output['number_of_days']
    assert expected_columns_out == list(output_data.columns)


def test_add_day_running_metrics_1vars_no_agg():

    # PREPARE:
    conf = {}
    conf['col_date'] = col_date_data
    conf['use_dow_as_running_agg'] = False
    col_running_metric = conf['col_date']
    cols_running_metric_vars = [col_default_numerical_1]
    cols_running_metric_agg = []
    conf['running_metric_days'] = [1, 3]
    running_metric = 'mean'
    cols_numerical = []
    cols_categorical = [f"{conf['col_date']}_iso_day_of_week"]
    cols_plot = ['col_plot_1']
    shift = 0
    conf['prediction_days'] = shift

    input_data, expected_output = blp.create_test_dataframe(
        case=7,
        n_rows=number_of_days
    )

    expected_columns_out = list(input_data.columns) + \
                           [f'{col_default_numerical_1}_running_mean_lag_1',
                            f'{col_default_numerical_1}_running_mean_lag_3']
    use_dow_as_running_agg = False

    # EXECUTE:
    output_data, cols_numerical_out, cols_plot_out, cols_running_metrics_out = add_day_running_metrics(
        input_data,
        col_running_metric,
        cols_running_metric_vars,
        cols_running_metric_agg,
        use_dow_as_running_agg,
        running_metric,
        cols_numerical,
        cols_categorical,
        cols_plot,
        conf
    )
    assert output_data.loc[output_data['event_date'].isin(['2020-01-01', '2020-01-02']),
                           f'{col_default_numerical_1}_running_mean_lag_3'].isnull().all()
    assert list(output_data.groupby('cluster_id')[f'{col_default_numerical_1}'].mean()) ==\
           expected_output[f'{col_default_numerical_1}']
    assert list(output_data.groupby(col_running_metric)[f'{col_default_numerical_1}'].mean()) ==\
           [3]*expected_output['number_of_days']
    assert expected_columns_out == list(output_data.columns)


def test_add_day_running_metrics_shift_negative():

    # PREPARE:
    conf = {}
    conf['col_date'] = col_date_data
    conf['use_dow_as_running_agg'] = False
    col_running_metric = conf['col_date']
    cols_running_metric_vars = [col_default_numerical_1, col_default_numerical_2]
    cols_running_metric_agg = [col_cluster_id]
    conf['running_metric_days'] = [1, 3]
    running_metric = 'mean'
    cols_numerical = []
    cols_categorical = [f"{conf['col_date']}_iso_day_of_week"]
    cols_plot = ['col_plot_1']
    shift = -2
    conf['prediction_days'] = shift


    input_data, expected_output = blp.create_test_dataframe(
        case=7,
        n_rows=number_of_days
    )

    expected_columns_out = list(input_data.columns) + \
                           [f'{col_default_numerical_1}_running_mean_lag_1',
                            f'{col_default_numerical_1}_running_mean_lag_3',
                            f'{col_default_numerical_2}_running_mean_lag_1',
                            f'{col_default_numerical_2}_running_mean_lag_3']
    use_dow_as_running_agg = False

    # EXECUTE:
    output_data, cols_numerical_out, cols_plot_out, cols_running_metrics_out = add_day_running_metrics(
        input_data,
        col_running_metric,
        cols_running_metric_vars,
        cols_running_metric_agg,
        use_dow_as_running_agg,
        running_metric,
        cols_numerical,
        cols_categorical,
        cols_plot,
        conf
    )
    assert output_data.loc[output_data['event_date'].isin(['2020-01-01', '2020-01-02']),
                           f'{col_default_numerical_1}_running_mean_lag_1'].isnull().all()
    assert output_data.loc[output_data['event_date'].isin(['2020-01-01', '2020-01-02',
                                                     '2020-01-03', '2020-01-04']),
                           f'{col_default_numerical_1}_running_mean_lag_3'].isnull().all()
    assert list(output_data.groupby(cols_running_metric_agg)[f'{col_default_numerical_1}'].mean()) \
           == expected_output[f'{col_default_numerical_1}']
    assert list(output_data.groupby(cols_running_metric_agg)[f'{col_default_numerical_2}'].mean()) \
           == expected_output[f'{col_default_numerical_2}']
    assert list(output_data.groupby(col_running_metric)[f'{col_default_numerical_1}'].mean()) \
           == [3]*expected_output['number_of_days']
    assert list(output_data.groupby(col_running_metric)[f'{col_default_numerical_2}'].mean()) \
           == [-6]*expected_output['number_of_days']

    assert expected_columns_out == list(output_data.columns)


def test_add_day_running_metrics_shift_positive():

    conf = {}
    conf['col_date'] = col_date_data
    conf['use_dow_as_running_agg'] = False
    col_running_metric = conf['col_date']
    cols_running_metric_vars = [col_default_numerical_1, col_default_numerical_2]
    cols_running_metric_agg = [col_cluster_id]
    conf['running_metric_days'] = [1, 3]
    running_metric = 'mean'
    cols_numerical = []
    cols_categorical = [f"{conf['col_date']}_iso_day_of_week"]
    cols_plot = ['col_plot_1']
    shift = 2
    conf['prediction_days'] = shift

    # PREPARE:
    input_data, expected_output = blp.create_test_dataframe(
        case=7,
        n_rows=number_of_days
    )

    expected_columns_out = list(input_data.columns) + \
                           [f'{col_default_numerical_1}_running_mean_lag_1',
                            f'{col_default_numerical_1}_running_mean_lag_3',
                            f'{col_default_numerical_2}_running_mean_lag_1',
                            f'{col_default_numerical_2}_running_mean_lag_3']
    use_dow_as_running_agg = False

    # EXECUTE:
    output_data, cols_numerical_out, cols_plot_out, cols_running_metrics_out = add_day_running_metrics(
        input_data,
        col_running_metric,
        cols_running_metric_vars,
        cols_running_metric_agg,
        use_dow_as_running_agg,
        running_metric,
        cols_numerical,
        cols_categorical,
        cols_plot,
        conf
    )
    assert output_data.loc[output_data['event_date'].isin(['2020-01-01', '2020-01-02']),
                           f'{col_default_numerical_1}_running_mean_lag_1'].isnull().all()
    assert output_data.loc[output_data['event_date'].isin(['2020-01-01', '2020-01-02',
                                                     '2020-01-03', '2020-01-04']),
                           f'{col_default_numerical_1}_running_mean_lag_3'].isnull().all()
    assert list(output_data.groupby(cols_running_metric_agg)[f'{col_default_numerical_1}'].mean()) ==\
           expected_output[f'{col_default_numerical_1}']
    assert list(output_data.groupby(cols_running_metric_agg)[f'{col_default_numerical_2}'].mean()) ==\
           expected_output[f'{col_default_numerical_2}']
    assert list(output_data.groupby(col_running_metric)[f'{col_default_numerical_1}'].mean()) ==\
           [3]*expected_output['number_of_days']
    assert list(output_data.groupby(col_running_metric)[f'{col_default_numerical_2}'].mean()) ==\
           [-6]*expected_output['number_of_days']

    assert expected_columns_out == list(output_data.columns)


def test_add_day_trend_variable():

    # Create list with days, 10 days, each day repeated 5 times, 50 elements in total
    number_of_days = 10
    trend_variable = col_date_data
    cols_numerical = [col_default_numerical_1, col_default_numerical_2]
    expected_cols_numerical_out = [col_default_numerical_1, col_default_numerical_2, 'day_trend']

    # PREPARE:
    input_data, expected_output = blp.create_test_dataframe(
        case=7,
        n_rows=number_of_days
    )

    # EXECUTE
    data_out, cols_numerical_out = add_day_trend_variable(
        input_data,
        trend_variable,
        cols_numerical
    )

    # VERIFY:
    assert cols_numerical_out == expected_cols_numerical_out
    assert list(data_out.loc[data_out[trend_variable] == data_out[trend_variable].min(), 'day_trend'].unique()) == [0]
    assert list(data_out.loc[data_out[trend_variable] == data_out[trend_variable].max(), 'day_trend'].unique()) == [9]
    assert list(data_out['day_trend'].unique()) == [0, 1, 2, 3, 4, 5, 6 ,7 ,8 ,9]


def test_fill_time_series_data_nans():
    # Create list with days, 10 days, each day repeated 5 times, 50 elements in total
    cols_numerical = [col_default_numerical_1, col_default_numerical_2]
    expected_cols_numerical_out = [col_default_numerical_1, col_default_numerical_2, 'day_trend']

    # PREPARE:
    input_data, _ = blp.create_test_dataframe(
        case='full_model_data',
        n_rows=5
    )

    input_data.loc[input_data[col_date_data].isin(input_data[col_date_data].unique()[-2:]), cols_numerical +
                   [col_default_target]] = np.nan

    n_days_for_fillna = cols_numerical

    output_data = fill_time_series_data_nans(
        data=input_data,
        cols_to_fillna=n_days_for_fillna,
        col_date=col_date_data,
        n_days_for_fillna=2
    )
    assert output_data.shape[0] == output_data.dropna(subset=n_days_for_fillna).shape[0]
    assert output_data.loc[output_data[col_date_data].isin(
        input_data[col_date_data].unique()[-2:]), col_default_numerical_1].mean() == 3
    assert output_data.loc[output_data[col_date_data].isin(
        input_data[col_date_data].unique()[-2:]), col_default_numerical_2].mean() == -6
    assert output_data.shape[0] != output_data.dropna(subset=[col_default_target]).shape[0]


def test_amplify_variable():

    col_default_another_2 = 'col_another_2'
    col_default_another_2_values = [123, 456, -123, 123.0, -999.0]

    input_data, _ = blp.create_test_dataframe(
        case=8
    )

    positive_amplifier_1 = 1
    negative_amplifier_1 = 1

    positive_amplifier_2 = 1
    negative_amplifier_2 = 1.5

    positive_amplifier_3 = 2.5
    negative_amplifier_3 = 0.25

    positive_amplifier_4 = 3
    negative_amplifier_4 = 5

    positive_amplifier_5 = 0.25
    negative_amplifier_5 = 0.75

    output_data_1 = amplify_variable(
        data=input_data,
        col_variable=col_default_another_2,
        positive_amplifier=positive_amplifier_1,
        negative_amplifier=negative_amplifier_1
    )
    output_data_2 = amplify_variable(
        data=input_data,
        col_variable=col_default_another_2,
        positive_amplifier=positive_amplifier_2,
        negative_amplifier=negative_amplifier_2
    )
    output_data_3 = amplify_variable(
        data=input_data,
        col_variable=col_default_another_2,
        positive_amplifier=positive_amplifier_3,
        negative_amplifier=negative_amplifier_3
    )
    output_data_4 = amplify_variable(
        data=input_data,
        col_variable=col_default_another_2,
        positive_amplifier=positive_amplifier_4,
        negative_amplifier=negative_amplifier_4
    )
    output_data_5 = amplify_variable(
        data=input_data,
        col_variable=col_default_another_2,
        positive_amplifier=positive_amplifier_5,
        negative_amplifier=negative_amplifier_5
    )

    sum_positive = sum([x for x in col_default_another_2_values if x > 0])
    sum_negative = sum([x for x in col_default_another_2_values if x < 0])

    assert output_data_1.loc[output_data_1[col_default_another_2]>0, col_default_another_2].sum() ==\
           sum_positive * positive_amplifier_1
    assert output_data_1.loc[output_data_1[col_default_another_2] < 0, col_default_another_2].sum() == \
           sum_negative * negative_amplifier_1
    assert output_data_2.loc[output_data_2[col_default_another_2]>0, col_default_another_2].sum() ==\
           sum_positive * positive_amplifier_2
    assert output_data_2.loc[output_data_2[col_default_another_2] < 0, col_default_another_2].sum() == \
           sum_negative * negative_amplifier_2
    assert output_data_3.loc[output_data_3[col_default_another_2]>0, col_default_another_2].sum() ==\
           sum_positive * positive_amplifier_3
    assert output_data_3.loc[output_data_3[col_default_another_2] < 0, col_default_another_2].sum() == \
           sum_negative * negative_amplifier_3
    assert output_data_4.loc[output_data_4[col_default_another_2]>0, col_default_another_2].sum() ==\
           sum_positive * positive_amplifier_4
    assert output_data_4.loc[output_data_4[col_default_another_2] < 0, col_default_another_2].sum() == \
           sum_negative * negative_amplifier_4
    assert output_data_5.loc[output_data_5[col_default_another_2]>0, col_default_another_2].sum() ==\
           sum_positive * positive_amplifier_5
    assert output_data_5.loc[output_data_5[col_default_another_2] < 0, col_default_another_2].sum() == \
           sum_negative * negative_amplifier_5


def test_calc_rides_per_scoot_in_period():

    col_vehicle_id = "vehicle_id"
    col_rides_in_period = "rides_in_period"
    col_timestamp_data = "timestamp"

    tests_dict = {}

    test_1 = {}
    test_1["rides_per_scoot_calc_period"] = 3
    test_1["expected_result"] = [1,1,1,1,1,
                                 3,2,1,3,1,
                                 4,3,2,1,1
                                 ]

    test_2 = {}
    test_2["rides_per_scoot_calc_period"] = 24
    test_2["expected_result"] = [4,3,3,2,1,
                                 5,3,2,5,1,
                                 5,4,3,2,1
                                 ]

    test_3 = {}
    test_3["rides_per_scoot_calc_period"] = 48
    test_3["expected_result"] = [5,4,3,2,1,
                                 5,3,2,5,1,
                                 5,4,3,2,1
                                 ]

    tests_dict["test_1"] = test_1
    tests_dict["test_2"] = test_2
    tests_dict["test_3"] = test_3

    expected_columns = [col_vehicle_id, col_rides_in_period, col_timestamp_data]



    for test in tests_dict:

        input_data, expected_df = blp.create_test_dataframe(
            case="data_rides_3_vehs_in_48_hours_fxd"
        )


        output_data = calc_rides_per_scoot_in_period(
            rides_data=input_data,
            rides_per_scoot_calc_period=tests_dict[test]["rides_per_scoot_calc_period"]
        )

        assert set(expected_columns) == set(output_data[expected_columns].columns)
        assert list(output_data["rides_in_period"]) == tests_dict[test]["expected_result"]