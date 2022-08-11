import lib.data_utils as dt_uts

import pandas as pd


def aggregate_weather_data(
    weather_data: pd.DataFrame,
    cols_weather: list,
    cols_weather_id: list,
    conf: dict
) -> pd.DataFrame:

    """
    Create weather data with hourly granularity from weather data with granularity of x hours (x is integer)
    Args:
        weather_data: weather data with granularity of more than one hour
        cols_weather: numerical and categorical weather columns
        cols_weather_id: list with date and hour column in weather_data
        conf: config dictionary

    Returns:
        weather_data: weather data with hourly granularity
    """

    weather_data = weather_data.sort_values(cols_weather_id)

    # First historical data, create all possible combination of data
    _, full_weather_data = \
        dt_uts.create_cartesian_product(
            weather_data,
            cols_names_list=cols_weather_id,
            min_values_list=[None,
                             conf['first_operation_hour']
                             ],
            max_values_list=[None,
                             conf['last_operation_hour']
                             ],
            step_values_list=[None,
                              1],
            fillna=None
        )
    full_weather_data = (full_weather_data
                         .sort_values(cols_weather_id)
                         .loc[:, cols_weather_id])
    full_weather_data = full_weather_data.merge(weather_data,
                                                on=cols_weather_id,
                                                how='left')

    full_weather_data[cols_weather] = full_weather_data[cols_weather].fillna(method='ffill')
    weather_data = full_weather_data

    return weather_data
