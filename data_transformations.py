import datetime
import logging

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime

import lib.data_utils as dt_uts
from lib.print_out_info import print_out_info


@print_out_info
def add_calendar_features(
    data: pd.DataFrame,
    col_date: str,
    calendar_features: list,
    cols_numerical: list,
    cols_categorical: list,
    cols_cyclical_in: list
) -> [pd.DataFrame, list, list, list]:

    """
    Adds columns with selected calendar features created based on col_date for given dataframe. Possibilities are:
    - hour (if column of pd.Timestamp or dt.datetime type)
    - iso_day_of_week
    - day_of_month
    - day_of_year
    - iso_calendar_week
    - iso_month
    - year
    - time (if column of pd.Timestamp or dt.datetime type)
    - date
    Args:
        data: Pandas df for which cols with calendar features will be added
        col_date: string with name of date column (based on this column calendar features will be created)
        calendar_features: list of calendar features (strings) to be created. As for now iso_day_of_week,
        iso_calendar_week, _iso_month and year are supported
        cols_numerical: list of numerical features names with new features added
        cols_categorical: list of categorical features names with new features added
        cols_cyclical_in: list of cyclical features names with new features added
    Returns:
        DataFrame: input pandas df with added calendar features columns
        conf: config dictionary with calendar variables added
    """

    if not isinstance(data, pd.DataFrame):
        raise Exception("Input 'data' should be a DataFrame")
    if not isinstance(col_date, object):
        raise Exception("Input 'col' should be a string")
    if not is_datetime(data[col_date]):
        raise Exception(f"Column {col_date} in input 'data' should be of type datetime")
    if col_date not in data.columns:
        raise Exception(f"Column {col_date} unavailable in input 'data'")

    if 'hour' in calendar_features:
        data[f"{col_date}_hour"] = data[col_date].dt.hour
        cols_cyclical_in.append(f"{col_date}_hour")
    if 'iso_day_of_week' in calendar_features:
        data[f"{col_date}_iso_day_of_week"] = data[col_date].dt.weekday + 1
        cols_categorical.append(f"{col_date}_iso_day_of_week")
    if 'day_of_month' in calendar_features:
        data[f"{col_date}_day_of_month"] = data[col_date].dt.day
        cols_categorical.append(f"{col_date}_day_of_month")
    if 'day_of_year' in calendar_features:
        data[f"{col_date}_day_of_year"] = data[col_date].dt.dayofyear
        cols_categorical.append(f"{col_date}_day_of_year")
    if 'iso_calendar_week' in calendar_features:
        data[f"{col_date}_iso_calendar_week"] = data[col_date].dt.isocalendar().week
        cols_cyclical_in.append(f"{col_date}_iso_calendar_week")
    if 'iso_month' in calendar_features:
        data[f"{col_date}_iso_month"] = data[col_date].dt.month
        cols_cyclical_in.append(f"{col_date}_iso_month")
    if 'year' in calendar_features:
        data[f"{col_date}_year"] = data[col_date].dt.year
        cols_numerical.append(f"{col_date}_year")
    if 'time' in calendar_features:
        data[f"{col_date}_time"] = data[col_date].dt.time
        cols_numerical.append(f"{col_date}_time")
    if 'date' in calendar_features:
        data[f"{col_date}_date"] = data[col_date].dt.date
        cols_numerical.append(f"{col_date}_date")

    return data, cols_numerical, cols_categorical, cols_cyclical_in


@print_out_info
def add_day_lag_variables(
    data: pd.DataFrame,
    col_lag: str,
    cols_lags_vars: list,
    cols_lags_agg: list,
    lags_days: list,
    cols_numerical: list,
    cols_plot: list,
    prediction_days: [int, float] = 0,
    logger: logging.Logger = None
) -> [pd.DataFrame, list, list]:

    """
    Adds day lag variables to the data.
    Date column is defined by col_lag.
    Variables for which lags vars are created are defined by cols_lags_vars.
    Aggregation levels are defined in cols_lags_agg.
    Returns df with added lag cols and config with numerical and plot columns list extended by new cols

    Example: for cols_lags_agg=['hour', cluster'], cols_lags_vars=['sales'] and lags_days=[1,5] we get 2 new lag
    columns: sales 1 and 5 days ago at the same hour in the same cluster

    Note: we group by [col_lag] + cols_lags_agg level, so data doesn't necessary need to have unique
    [col_lag] + cols_lags_agg level combinations (although it makes more sense from data perspective)

    Args:
        data: data with col_lag, cols_lags_vars and cols_lags_agg
        col_lag: columns containing day data in datetime64[ns] format
        cols_lags_vars: variables for which lag variables will be created
        cols_lags_agg: aggregation level, lag variables will be created on this level (n days ago) only
        lags_days: days (in the past) for lags variables creation
        cols_numerical: list of numerical features names
        cols_plot: lists with features names for plotting, each lag vars are in separate list
        prediction_days: prediction days
        logger: logger object
    Returns:
        data: data with lag cols added
        cols_numerical: list of numerical features names with new features added
        cols_plot: list features names for plotting with new features added
    TODO: this function is meant to be on day level (for col_lag), should be on any level, e.g. hour for col_lag?

    """

    if prediction_days > min(lags_days):
        logger.warning(f"Prediction period ({prediction_days} e.g. days) is larger than minimum lag variable "
                       f"({min(lags_days)} e.g. days). Predictions are only possible for ({min(lags_days)}) days")

    cols_lags = []

    # Iterate over lag variables and lag values
    for col_lag_var in cols_lags_vars:
        cols_subplot = []
        for lag in lags_days:
            col_lag_name = f"{col_lag}_lag_{lag}"
            col_lag_var_name = f"{col_lag_var}_lag_{lag}"

            # Create date col for given lag
            data[col_lag_name] = data[col_lag] - pd.to_timedelta(lag, unit='d')

            # Grouping (we want one value per day-cols_lags_agg) - to avoid duplicates when merging
            lag_data = (data
                        .groupby([col_lag] + cols_lags_agg)
                        .mean()
                        .reset_index()
                        .loc[:, [col_lag_var, col_lag] + cols_lags_agg]
                        .rename(columns={col_lag_var: col_lag_var_name}, inplace=False))

            # Join the same df on the created date column to obtain data lag days before. Rename new lag column
            data = (data.merge(lag_data,
                               left_on=[col_lag_name] + cols_lags_agg,
                               right_on=[col_lag] + cols_lags_agg,
                               how="left")
                        .drop([f"{col_lag}_y", col_lag_name],
                              axis=1)
                        .rename(columns={f"{col_lag}_x": col_lag}))

            # Append to columns
            cols_numerical.append(col_lag_var_name)
            cols_subplot.append(col_lag_var_name)
        cols_plot.append(cols_subplot)

    return data, cols_numerical, cols_plot, cols_lags


@print_out_info
def add_day_running_metrics(
    data: pd.DataFrame,
    col_running_metric: str,
    cols_running_metric_vars: list,
    cols_running_metric_agg: list,
    use_dow_as_running_agg: bool,
    running_metric: str,
    running_metric_days: list,
    running_metrics_shift: [int, float],
    cols_numerical: list,
    cols_categorical: list,
    cols_plot: list,
    conf: dict,
    logger: logging.Logger
) -> [pd.DataFrame, list, list]:

    """
    Add daily running metrics to data. Date column is defined by col_running_metric.
    Variables for which running metrics vars are created are defined by cols_running_metric_vars.
    Aggregation levels are defined in cols_lags_agg.
    Type of running metric is defined by running_metric.
    Running windows are defined in running_metric_days.
    Additionally data is shifted by running_metrics_shift (this is usually imposed by period of predictions -
    we need to shift the data by (days of predictions - 1) days to obtain the data for all prediction days)
    Returns df with added running metrics cols and config with numerical and plot columns list extended by new cols.
    Running metrics are shifted regarding the original data by shift days.

    Example: for cols_running_metric_agg=['hour', cluster'], cols_running_metric_vars=['sales'], running_metric='median'
    and running_metric_days=[2,5] we get 2 new running metric cols: running median of sales using data at the same hour
    in the same cluster in the last 2 and 5 days.

    Note:
        - data need to have unique [col_lag] + cols_lags_agg level combinations if cols_lags_agg is not empty.
        - if we use DOW as aggregator (or any other column that uses col_running_metric (ie. days), like months),
        we need to take it into account in running_metric_days, e.g. if we want to have a mean value for given hour
        at given DOW from last 28 days,  running_metric_days = 28/7
        - running_metrics_shift var - by how many days calculated running metrics will be shifted regarding the original
         data (for positive: 'to the future'), this is useful for prediction data, where we don't have values of target
         variable for the prediction dates. E.g. for shift=0 and running_metric_days = [3], we get NaNs for running
         metrics for first two days, with shift=-2 the same for last two days.
    Args:
        data: data with col_running_metric, cols_running_metric_agg and cols_running_metric_vars
        col_running_metric: columns containing day data in datetime64[ns] format
        cols_running_metric_vars: variables for which running metric variables will be created
        cols_running_metric_agg: aggregation level, running metric variables will be created on this, for running metric
        use_dow_as_running_agg: if the day of week is used as aggregator
        calculated only for col_running_metric (no other aggs) empty list should be passed
        running_metric: which running metric to use, possible options are: 'mean', 'median', 'min' and 'max'
        running_metric_days: list with windows for running metrics
        running_metrics_shift: days for which we need to shift the running metrics, usually number of prediction days
        (e.g. if we predict for next 5 days, we need to shift the data by (5-1) days to get the moving average for the
        5th day)
        cols_numerical: list of numerical features names, or empty list
        cols_categorical: list of categorical features names, or empty list
        cols_plot: list features names for plotting, or empty list
        conf: config as dict
        logger: logging object

    Returns:
        data: data with lag cols added
        cols_numerical: list of numerical features names with new features added
        cols_plot: list features names for plotting with new features added
    TODO: - is running lag on week level necessary? Is it not the same as lag variable - no, it can take median of last
     x weeks? Also it is moved by 7 days + prediction days - 1, which doesn't seem correct.
     - as for now running metrics lag col is considered to be mainly days, would it work for other lag (e.g. hours)?

    """
    if running_metrics_shift > min(running_metric_days):
        logger.warning(f"Prediction period/running metrics shift ({running_metrics_shift} e.g. days) is larger than"
                       f" minimum lag variable ({min(running_metric_days)} e.g. days). Predictions are only possible "
                       f"for ({min(running_metric_days)}) days")

    if (f"{conf['col_date']}_iso_day_of_week" in cols_categorical) & (conf['use_dow_as_running_agg']):
        conf['cols_lags_agg'].append(f"{conf['col_date']}_iso_day_of_week")

        # Check if running windows values are divisible by 7 for weekly running window
        weekly_lag = [lag%7 for lag in running_metric_days]
        weekly_lags_correct = all(x == 0 for x in weekly_lag)
        if not weekly_lags_correct:
            logger.warning(f"You are using DOW for running metric window, but some or all of your running metrics"
                           f"day lags are not divisible by 7")

        # If running metric consists of DOW, we need to join on minus n*7 days
        running_metrics_shift = np.ceil(running_metrics_shift / 7) * 7
        running_metric_days = [int(np.floor(day / 7)) for day in running_metric_days]
    else:
        conf['use_dow_as_running_agg'] = False

    # To rename the index agg variable after group_by
    nr_agg_cols = len(cols_running_metric_agg)

    # Create date col with minus shift days to join running metric columns
    col_lag_name = f"{col_running_metric}_lag_{running_metrics_shift}"
    data[col_lag_name] = data[col_running_metric] - pd.to_timedelta(running_metrics_shift, unit='d')

    cols_running_metrics = []
    # Iterate over running metric variables and running metric day values
    for var in cols_running_metric_vars:
        cols_subplot = []
        for lag in running_metric_days:

            # Create date col for given running metric-day
            if use_dow_as_running_agg:
                lag_name = f"{lag}_weeks"
            else:
                lag_name = lag
            col_running_metric_name = f"{var}_running_{running_metric}_lag_{lag_name}"

            # Sort by days
            running_metric_data = data.sort_values(col_running_metric)

            # Grouping, for no agg columns (running metric is aggregated by whole day) we use mean value of
            # col_running_metric that day (also for other metrics as median etc.)
            if cols_running_metric_agg:
                running_metric_data = (running_metric_data
                                       .groupby(cols_running_metric_agg))
            else:
                running_metric_data = (running_metric_data
                                       .groupby(col_running_metric)
                                       .mean())

            # Calculate running metric
            running_metric_data = running_metric_data[var].rolling(lag, center=False)
            if running_metric == 'mean':
                running_metric_data = running_metric_data.mean()
            elif running_metric == 'median':
                running_metric_data = running_metric_data.median()
            elif running_metric == 'min':
                running_metric_data = running_metric_data.min()
            elif running_metric == 'max':
                running_metric_data = running_metric_data.max()

            running_metric_data = (running_metric_data
                                   .reset_index()
                                   .rename(columns={f"level_{nr_agg_cols}": 'index',
                                    var: col_running_metric_name}))

            if cols_running_metric_agg:
                running_metric_data = (running_metric_data.set_index('index')
                                                          .merge(data[col_running_metric],
                                                                 left_index=True,
                                                                 right_index=True))

            # Join the same df on the created lagged date column to obtain running metric. Rename new column
            data = (data.merge(running_metric_data,
                               left_on=cols_running_metric_agg + [col_lag_name],
                               right_on=cols_running_metric_agg + [col_running_metric],
                               how='left')
                        .rename(columns={col_running_metric + '_x': col_running_metric})
                        .drop([col_running_metric + '_y'], axis=1))

            cols_numerical.append(col_running_metric_name)
            cols_subplot.append(col_running_metric_name)
            cols_running_metrics.append(col_running_metric_name)

        cols_plot.append(cols_subplot)

    data = data.drop(col_lag_name, axis=1)

    return data, cols_numerical, cols_plot, cols_running_metrics


@print_out_info
def add_day_trend_variable(
    data: pd.DataFrame,
    trend_variable: str,
    cols_numerical: list,
    day_zero: [pd.Timestamp, datetime.date] = pd.Timestamp('2000-01-01')
) -> [pd.DataFrame, list]:

    """
    Adds day trend variable, which equals to number of days since day zero, and increases by 1 for each following day.
    This variable represents the general trend of the growth or decay of target variable.
    Args:
        data: data with trend_variable col
        trend_variable: string denoting column representing day
        cols_numerical: list of numerical features names
        day_zero: which date is the trend variable counted for

    Returns:
        data: data with trend variable added

    TODO:
     - trend variable is assumed to be linear. Add non-linearity term as input.
     - only day trend is calculated now, add more options.
    """

    date_range = pd.date_range(data[trend_variable].min(),
                               data[trend_variable].max())

    day_trend_variable_list = []
    for ind, day in enumerate(date_range):
        days_since_day_zero = pd.Timedelta(date_range[ind] - day_zero).days
        day_trend_variable_list.append([date_range[ind], days_since_day_zero])

    day_trend_variable = pd.DataFrame(
        day_trend_variable_list,
        columns=[trend_variable, 'day_trend']
    )
    data = data.merge(day_trend_variable,
                      on=trend_variable,
                      how='left')

    cols_numerical.append('day_trend')

    return data, cols_numerical


def add_weather_variables(
    data: pd.DataFrame,
    weather_data: pd.DataFrame,
    cols_numerical: list,
    cols_categorical: list,
    conf: dict
) -> [pd.DataFrame, list, list]:
    """
    Adds weather data to model data. Forecast and historical forecast weather data is predicted for next 5 days,
     3 hours interval.
    Args:
        data: model data
        weather_data: weather data
        cols_numerical: numerical columns
        cols_categorical: categorical columns
        conf: config as dictionary

    Returns:
         data: model data with weather data added
         cols_numerical: numerical columns with weather columns added
         cols_categorical: categorical columns with weather columns added
    """

    weather_data = weather_data[[conf["col_weather_hour"]] + [conf["col_weather_date"]] +
                                conf["cols_weather_categorical"] + conf["cols_weather_numerical"]]
    data = data.merge(
        weather_data,
        how='left',
        left_on=[conf["col_hour"], conf["col_date"]],
        right_on=[conf["col_weather_hour"], conf["col_weather_date"]]
    )
    cols_numerical.extend(conf['cols_weather_numerical'])
    cols_categorical.extend(conf['cols_weather_categorical'])

    return data, cols_numerical, cols_categorical


def fill_time_series_data_nans(
    data: pd.DataFrame,
    cols_to_fillna: list,
    col_date: str,
    n_days_for_fillna: [int, float]
) -> pd.DataFrame:

    """
    Fills NaNs found in time series data, only in cols_to_fillna columns. It uses mean of n_days_for_fillna last days
    for this.
    Args:
        data: data to fill NaNs
        cols_to_fillna: columns to fill NaNs
        col_date: date column,
        n_days_for_fillna: how many last days used to create a mean to fill NaNs

    Returns:
        data: data with specified columns having NaNs filled
    """

    for col in cols_to_fillna:
        value_to_fill_na = data.dropna(how='any',
                                       subset=[col],
                                       axis='index') \
                               .groupby(col_date) \
                               .mean() \
                               .sort_values(col_date, ascending=False) \
                               .reset_index() \
                               .loc[0:n_days_for_fillna + 1, col] \
            .mean()
        data = data.fillna(value={col: value_to_fill_na})

    return data


@print_out_info
def level_variable(
    data: pd.DataFrame,
    col_variable: str
) -> pd.DataFrame:

    """
    Levels a variable that has positive and the negative values in the way that the sum of both positive and negative
    values is equal.
    Example:
    Args:
        data: df with col_variable to level
        col_variable: variable name
    Returns:
        output_data: df with col_variable levelled
    """

    output_data = data.copy()
    sum_positive = data.loc[data[col_variable] > 0, col_variable].sum()
    sum_negative = data.loc[data[col_variable] < 0, col_variable].sum() * (-1)

    positive_multiplier = (sum_positive + sum_negative) / (sum_positive * 2)
    negative_multiplier = (sum_positive + sum_negative) / (sum_negative * 2)

    output_data.loc[output_data[col_variable] > 0, col_variable] = (
                output_data.loc[output_data[col_variable] > 0, col_variable] * positive_multiplier)
    output_data.loc[output_data[col_variable] < 0, col_variable] = (
                output_data.loc[output_data[col_variable] < 0, col_variable] * negative_multiplier)

    return output_data


def low_days_warning(
        data: pd.DataFrame,
        col_date: str,
        min_day_filter: int,
        logger: logging.Logger
) -> None:
    """
    Counts number of days (defined by col_date) in data (not necessary consecutive) and prints logger warning if the
    amount lower than the min_day_filter threshold.
    Args:
        data: data df
        col_date: column defining date
        min_day_filter: minimum days below a warning is printed
        logger: logger object

    Returns:
        None
    """

    number_of_days = len(data[col_date].unique())
    number_of_days_warning_filter = number_of_days < min_day_filter
    if number_of_days_warning_filter:
        logger.warning(f"Only {number_of_days} days of data are available")


@print_out_info
def amplify_variable(
    data: pd.DataFrame,
    col_variable: str,
    positive_amplifier: [int, float],
    negative_amplifier: [int, float]
) -> pd.DataFrame:

    """
    Amplifies positive and negative values of the col_variable by positive_amplifier and negative_amplifier
    respectively.
    Args:
        data: df with col_variable
        col_variable: name of the column to be amplified
        positive_amplifier: multiplier of positive values of the col_variable
        negative_amplifier: multiplier of negative values of the col_variable

    Returns:
        output_data: df with col_variable amplified
    """

    if positive_amplifier <= 0 or negative_amplifier <= 0:
        raise Warning("Both positive and negative amplifiers should be a positive number")

    output_data = data.copy()

    output_data.loc[output_data[col_variable] > 0, col_variable] = (
                output_data.loc[output_data[col_variable] > 0, col_variable] * positive_amplifier)
    output_data.loc[output_data[col_variable] < 0, col_variable] = (
                output_data.loc[output_data[col_variable] < 0, col_variable] * negative_amplifier)

    return output_data


@print_out_info
def wf_feature_engineering(
    data,
    weather_data: pd.DataFrame,
    cols_numerical,
    cols_categorical,
    cols_cyclical_in,
    cols_plot,
    conf,
    logger: logging.Logger
):
    """
    Does feature engineering for time series data. These features can be used later in WFV evaluation, training and
    predictions. Each transformation in this function is described in conf dictionary.
    Following transformations can be executed this function:
    - external data is merged to data
    - date features are added
    - mean variables are added
    - lag variables are added
    - running metrics variables are added
    - trend variables are added
    - cyclical features are created
    Args:
        data: pre-processed data for given project
        weather_data: weather data
        cols_numerical: list of numerical features names
        cols_categorical: list of categorical features names
        cols_cyclical_in: list of cyclical features names
        cols_plot: list features names for plotting
        conf: config as dict
        logger: logger object

    Returns:
        clustered_data: enriched with features
        conf: config dict extended with new variables
        cols_numerical: list of numerical features names with new features added
        cols_categorical: list of categorical features names with new features added
        cols_cyclical_in: list of cyclical features names with new features added
        cols_plot: list features names for plotting with new features added
        conf: updated config dictionary
    """

    # Add date variables
    if conf['add_date_variables_step']:
        data, cols_numerical, cols_categorical, cols_cyclical_in = add_calendar_features(
            data=data,
            col_date=conf['col_date'],
            calendar_features=conf['date_vars'],
            cols_numerical=cols_numerical,
            cols_categorical=cols_categorical,
            cols_cyclical_in=cols_cyclical_in
        )

    # Add mean count for target variable on cluster level after aggregating clusters and creating cartesian product
    if conf['add_mean_vars_step']:
        for ind, var in enumerate(conf['cols_mean_vars']):
            var_mean = (data
                        .groupby(conf['cols_mean_aggs'][ind])[var]
                        .mean()
                        .reset_index()
                        .rename(columns={var: f"mean_{var}"}))
            data = data.merge(
                var_mean,
                on=conf['cols_mean_aggs'][ind],
                how='left'
            )
            cols_numerical.append(f"mean_{var}")

    # Add lag variables
    if conf['add_lags_variables_step']:
        data, cols_numerical, cols_plot, cols_lags = add_day_lag_variables(
            data=data,
            col_lag=conf['col_lag'],
            cols_lags_vars=conf['cols_lags_vars'],
            cols_lags_agg=conf['cols_lags_agg'],
            lags_days=conf['lags_days'],
            cols_numerical=cols_numerical,
            cols_plot=cols_plot,
            prediction_days=conf['prediction_days'],
            logger=logger
        )
    else:
        cols_lags = []

    # Add running metrics variables
    if conf['add_running_metrics_step']:
        data, cols_numerical, cols_plot, cols_running_metrics = add_day_running_metrics(
            data=data,
            col_running_metric=conf['col_running_metric'],
            cols_running_metric_vars=conf['cols_running_metric_vars'],
            cols_running_metric_agg=conf['cols_lags_agg'],
            use_dow_as_running_agg=conf['use_dow_as_running_agg'],
            running_metric=conf['running_metric'],
            running_metric_days = conf['running_metric_days'],
            running_metrics_shift = conf['prediction_days'],
            cols_numerical=cols_numerical,
            cols_categorical=cols_categorical,
            cols_plot=cols_plot,
            conf=conf,
            logger=logger
        )
    else:
        cols_running_metrics = []

    # Add trend variable
    if conf['add_trend_variable_step']:
        data, cols_numerical = add_day_trend_variable(
            data=data,
            trend_variable=conf['col_date'],
            cols_numerical=cols_numerical,
            day_zero=pd.Timestamp('2000-01-01')
        )

    # Add weather variables
    if conf['add_weather_variables_step']:
        data, cols_numerical, cols_categorical = add_weather_variables(
            data=data,
            weather_data=weather_data,
            cols_numerical=cols_numerical,
            cols_categorical=cols_categorical,
            conf=conf
        )

    # Encode cyclical columns
    if cols_cyclical_in:
        data, cols_cyclical_out = dt_uts.encode_cyclical_columns(
            data=data,
            cols_cyclical=cols_cyclical_in,
            drop_orig_cols=False
        )
        cols_numerical = cols_numerical + cols_cyclical_out

    return data, cols_numerical, cols_categorical, cols_cyclical_in, cols_plot, cols_running_metrics, cols_lags, conf
