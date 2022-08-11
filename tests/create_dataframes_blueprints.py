import numpy as np
import pandas as pd
from random import randrange, choice, randint
from string import ascii_uppercase
import datetime as dt

from tests.create_polygon_blueprints import create_regions_polygons

def flatten(xss):
    return [x for xs in xss for x in xs]

col_lon_data = 'event_longitude'
col_lat_data = 'event_latitude'
col_date_data = 'event_date'
col_date_hour = 'event_hour'
col_timestamp_data = 'timestamp'
col_lon_clusters = 'cluster_longitude'
col_lat_clusters = 'cluster_latitude'
cols_coords_clusters = [col_lon_clusters, col_lat_clusters]
cols_coords_data = [col_lon_data, col_lat_data]
col_cluster_id = 'cluster_id'
col_vehicle_id = 'vehicle_id'
col_cluster_id_prefix = 'Dynamic-DP-'
col_demand = 'event_count'
col_predict_rides = 'rides_count'
col_predict_arrivals = 'arrivals_count'
col_live_scoots = 'live_scoots_count'
col_pred_rides_per_scoot_cluster = "pred_rides_per_scoot_cluster"
col_sm_delta = "sm_delta"
col_deploy = 'deploy'

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


def random_timestamp(
    start_ts,
    end_ts
):
    """
    This function will return a random datetime between two datetime objects.
    """
    delta = end_ts - start_ts
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start_ts + dt.timedelta(seconds=random_second)


def create_test_dataframe(
        case,
        n_rows=5,
        **kwargs
):
    np.random.seed(42)

    # Case 1: clustering input data, random lon, lat data for one day at one timestamp
    if case==1:
        event_date = '2021-06-01'
        long_min, long_max = 2, 3
        lat_min, lat_max = 41, 42
        event_timestamp = '2021-06-01 00:00:00'

        df = pd.DataFrame({
            col_lon_data: np.random.uniform(long_min, long_max, n_rows),
            col_lat_data: np.random.uniform(lat_min, lat_max, n_rows),
            col_date_data: event_date,
            col_timestamp_data: event_timestamp
        })

        # Expected data for n_rows=5
        expected_df = {
            'event_longitude': {
                0: 2.374540,
                1: 2.950714,
                2: 2.731994,
                3: 2.598658,
                4: 2.156019
            },
            'event_latitude': {
                0: 41.155995,
                1: 41.058084,
                2: 41.866176,
                3: 41.601115,
                4: 41.708073
            },
            'event_date': {
                0: pd.Timestamp('2021-06-01 00:00:00'),
                1: pd.Timestamp('2021-06-01 00:00:00'),
                2: pd.Timestamp('2021-06-01 00:00:00'),
                3: pd.Timestamp('2021-06-01 00:00:00'),
                4: pd.Timestamp('2021-06-01 00:00:00')
            },
            'timestamp': {
                0: pd.Timestamp('2021-06-01 00:00:00'),
                1: pd.Timestamp('2021-06-01 00:00:00'),
                2: pd.Timestamp('2021-06-01 00:00:00'),
                3: pd.Timestamp('2021-06-01 00:00:00'),
                4: pd.Timestamp('2021-06-01 00:00:00')
            }
        }
        expected_df = pd.DataFrame(expected_df)

        df = df.round({
            col_lon_data: 3,
            col_lat_data: 3}
        )
        expected_df = expected_df.round({
            col_lon_data: 3,
            col_lat_data: 3}
        )

    # input clustering data, df with only lon and lat (random) and event count
    if case==2:
        np.random.seed(42)
        long_min, long_max = 2, 3
        lat_min, lat_max = 41, 42
        df = pd.DataFrame({
            col_lon_data: np.random.uniform(long_min, long_max, n_rows),
            col_lat_data: np.random.uniform(lat_min, lat_max, n_rows),
            col_demand: 1
        })
        df = df.round({
            col_lon_data: 3,
            col_lat_data: 3}
        )
        expected_df = None

    # output of clustering algorythm
    if case==3:
        np.random.seed(33)
        long_min, long_max = 2, 3
        lat_min, lat_max = 41, 42
        df = pd.DataFrame({
            col_cluster_id: [f'{col_cluster_id_prefix}{i}' for i in range(n_rows)],
            col_lon_clusters: np.random.uniform(long_min, long_max, n_rows),
            col_lat_clusters: np.random.uniform(lat_min, lat_max, n_rows),
            col_count_clusters: [n * 100 for n in range(n_rows)]
        })
        df = df.round({
            col_lon_clusters: 3,
            col_lat_clusters: 3}
        )
        expected_df = None

    # As case 2 but with col_cluster (clustering input data with assigned clusters), data for 5 different clusters
    if case==4:
        np.random.seed(42)
        long_min, long_max = 2, 3
        lat_min, lat_max = 41, 42
        cluster_ids = [f'{col_cluster_id_prefix}{i}' for i in range(kwargs['number_of_clusters'])]

        df = pd.DataFrame({
            col_lon_data: np.random.uniform(long_min, long_max, n_rows),
            col_lat_data: np.random.uniform(lat_min, lat_max, n_rows),
            col_demand: 1,
            col_cluster_id: np.random.choice(cluster_ids, n_rows),
        })
        df = df.round({
            col_lon_data: 3,
            col_lat_data: 3}
        )
        expected_df = None

    # df with two random lon lat columns, in this case data and clusters, but data not assigned to closest cluster
    if case==5:
        np.random.seed(42)
        long_min, long_max = 2, 3
        lat_min, lat_max = 41, 42

        df = pd.DataFrame({
            col_lon_data: np.random.uniform(long_min, long_max, n_rows),
            col_lat_data: np.random.uniform(lat_min, lat_max, n_rows),
            col_lon_clusters: np.random.uniform(long_min, long_max, n_rows),
            col_lat_clusters: np.random.uniform(lat_min, lat_max, n_rows),
        })
        df = df.round({
            col_lon_data: 3,
            col_lat_data: 3,
            col_lon_clusters: 3,
            col_lat_clusters: 3
        }
        )
        expected_df = [70333.443, 126609.919, 98252.416, 85875.305,
            93275.340, 98552.737, 15790.092, 61493.073, 28010.275, 75449.493]

    # df with two columns, one with consecutive dates ordered, second with random numerical data
    if case == 6:

        days_list = pd.date_range(
            pd.Timestamp('1981-05-24'),
            periods=n_rows
        ).tolist()
        random_list = [float(x) for x in np.random.uniform(-1000, 1000, n_rows)]

        df = pd.DataFrame({
            col_date_data: days_list,
            col_default_numerical: random_list
        })

        expected_df = None

    # df with days, n_rows days, each day repeated 5 times, n_rows*5 elements in total
    if case == 7:
        
        number_of_events = n_rows * 5
        number_of_days = n_rows
        events_per_day = int(number_of_events / number_of_days)

        days_list = pd.date_range(pd.Timestamp('2021-01-01'), periods=number_of_days).tolist()
        days_list = [day for day in days_list for repetitions in range(events_per_day)]

        cluster_id_list = ['Cluster_1', 'Cluster_2', 'Cluster_3', 'Cluster_4', 'Cluster_5'] * number_of_days

        col_values_1 = [1, 2, 3, 4, 5]
        col_values_1_list = col_values_1 * number_of_days
        col_values_2 = [-10, -8, -6, -4, -2]
        col_values_2_list = col_values_2 * number_of_days

        df = pd.DataFrame({
            col_date_data: days_list,
            col_cluster_id: cluster_id_list,
            col_default_numerical_1: col_values_1_list,
            col_default_numerical_2: col_values_2_list
        })
        expected_df = {
            col_default_numerical_1: col_values_1,
            col_default_numerical_2: col_values_2,
            'number_of_days': number_of_days,
            'number_of_clusters': int(number_of_events / number_of_days)
        }

    if case == 'full_model_data':
        number_of_events = n_rows * 5
        number_of_days = n_rows
        events_per_day = int(number_of_events / number_of_days)

        days_list = pd.date_range(pd.Timestamp('2021-01-01'), periods=number_of_days).tolist()
        days_list = [day for day in days_list for repetitions in range(events_per_day)]

        cluster_id_list = ['id_1', 'id_2', 'id_3', 'id_4', 'id_5'] * number_of_days

        col_values_1 = [1, 2, 3, 4, 5]
        col_values_1_list = col_values_1 * number_of_days
        col_values_2 = [-10, -8, -6, -4, -2]
        col_values_2_list = col_values_2 * number_of_days
        col_values_target = [-1, 1, 5, -5, 10]
        col_values_target_list = col_values_target * number_of_days

        df = pd.DataFrame({
            col_date_data: days_list,
            col_cluster_id: cluster_id_list,
            col_default_numerical_1: col_values_1_list,
            col_default_numerical_2: col_values_2_list,
            col_default_target: col_values_target_list
        })
        expected_df = None

    if case == 'x_model_data':
        number_of_events = n_rows * 5
        number_of_days = n_rows
        events_per_day = int(number_of_events / number_of_days)

        days_list = pd.date_range(pd.Timestamp('2021-01-01'), periods=number_of_days).tolist()
        days_list = [day for day in days_list for repetitions in range(events_per_day)]

        cluster_id_list = ['id_1', 'id_2', 'id_3', 'id_4', 'id_5'] * number_of_days

        col_values_1 = [1, 2, 3, 4, 5]
        col_values_1_list = col_values_1 * number_of_days
        col_values_2 = [-10, -8, -6, -4, -2]
        col_values_2_list = col_values_2 * number_of_days

        df = pd.DataFrame({
            col_date_data: days_list,
            col_cluster_id: cluster_id_list,
            col_default_numerical_1: col_values_1_list,
            col_default_numerical_2: col_values_2_list
        })
        expected_df = None

    if case == 'y_model_data':
        number_of_events = n_rows * 5
        number_of_days = n_rows
        events_per_day = int(number_of_events / number_of_days)

        days_list = pd.date_range(pd.Timestamp('2021-01-01'), periods=number_of_days).tolist()
        days_list = [day for day in days_list for repetitions in range(events_per_day)]

        cluster_id_list = ['id_1', 'id_2', 'id_3', 'id_4', 'id_5'] * number_of_days

        col_values_target = [-1, 1, 5, -5, 10]
        col_values_target_list = col_values_target * number_of_days

        df = pd.DataFrame({
            col_date_data: days_list,
            col_cluster_id: cluster_id_list,
            col_default_target: col_values_target_list
        })
        expected_df = None

    # DF with several columns of different data types, e.g. for cartesian product function testing
    if case == 8:

        df = pd.DataFrame({
            col_default_string: ['a', 'bb', 'a', 'c', 'a'],
            col_default_int: [1, 1, 3, 1, 3],
            col_default_float: [0.5, 0.9, 0.9, 0.9, 0.9],
            col_default_date: pd.date_range(start='1/1/2020', periods=5, freq='D'),
            col_default_another_1: ['lo', 'la', 'lo', 'li', 'li'],
            col_default_another_2: [123, 456, -123, 123.0, -999.0]
        })

        expected_df = pd.DataFrame({
            col_default_int: [1, 1, 3, 1, 3],
            col_default_date: pd.date_range(start='1/1/2020', periods=5, freq='D'),
            f"{col_default_string}_bb": [0, 1, 0, 0, 0],
            f"{col_default_string}_c": [0, 0, 0, 1, 0],
            col_default_another_1: ['lo', 'la', 'lo', 'li', 'li'],
            col_default_float: [0.5, 0.9, 0.9, 0.9, 0.9]
        })

    # One column with values from 0 to 9
    if case == 9:
        df = pd.DataFrame({
            'vector': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        })
        expected_df = pd.DataFrame({
            'sin': [0.0, 0.64, 0.98, 0.87, 0.34, -0.34, -0.87, -0.98, -0.64, -0.0],
            'cos': [1.0, 0.77, 0.17, -0.5, -0.94, -0.94, -0.5, 0.17, 0.77, 1.0]
        })

    # One column with months values from 0 to 11
    if case == 10:
        data = {
            col_months: [month for month in range(0, 12)]
        }
        df = pd.DataFrame(data=data)
        expected_df = None

    # Two columns, with months values from 0 to 11 and various days of week
    if case == 11:
        data = {
            col_months: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            col_dow: [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4]
        }
        df = pd.DataFrame(data=data)
        expected_df = None

    # data for col_predict_arrivals, col_predict_rides and col_live_scoots, for several hours
    # (kwargs['rebalance_hours']) in several clusters. col_live_scoots is not zero only in first hour
    if case == 20:
        start_date = pd.Timestamp('2021-01-01')
        rebalance_hours = 8

        cluster_id_vector = [''.join(choice(ascii_uppercase) for i in range(8)) for x in range(n_rows)]
        data = {
            col_cluster_id: cluster_id_vector,
            col_predict_arrivals: [x for x in np.random.uniform(0, 20, n_rows)],
            col_predict_rides: [x for x in np.random.uniform(0, 20, n_rows)],
            col_live_scoots: [x for x in np.random.uniform(0, 20, n_rows)],
        }

        df = pd.DataFrame(data=data)
        # Add data-hour data
        event_date_vector = [start_date.date().strftime('%Y-%m-%d') for x in range(rebalance_hours)]
        event_hour_vector = [int(round(x)) for x in range(rebalance_hours)]
        event_hour_vector = [x % 24 for x in event_hour_vector]

        timestamp_data = pd.DataFrame({
            col_date_data: event_date_vector,
            col_date_hour: event_hour_vector})

        timestamp_data[col_timestamp_data] = timestamp_data.apply(
            lambda x: dt.datetime.combine(dt.datetime.strptime(x[col_date_data], "%Y-%m-%d"),
                                          dt.datetime.strptime(str(x[col_date_hour]), '%H')
                                          .time()),
            axis=1)

        df['key'] = 0
        timestamp_data['key'] = 0

        df = df.merge(timestamp_data, how='outer')

        df = df.drop(['event_date', 'event_hour', 'key'], axis=1)

        expected_df = None

    # as case==20, but different data values
    if case == 21:
        col_cluster_id_vector = [f"cluster_id_nr_{x}" for x in range(n_rows)]

        data = pd.DataFrame({
            col_cluster_id: col_cluster_id_vector,
            col_predict_arrivals: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
            col_live_scoots: [10, 5, 0, 10, 5, 0, 10, 5, 0, 20],
            col_predict_rides: [10, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        })

        # Add data-hour data
        event_date_vector = [kwargs['start_date'].date().strftime('%Y-%m-%d') for x in range(kwargs['rebalance_hours'])]
        event_hour_vector = [int(round(x)) for x in range(kwargs['rebalance_hours'])]
        event_hour_vector = [x % 24 for x in event_hour_vector]

        temporal_data = pd.DataFrame({
            col_date_data: event_date_vector,
            col_date_hour: event_hour_vector})

        temporal_data[col_timestamp_data] = temporal_data.apply(
            lambda x: dt.datetime.combine(dt.datetime.strptime(x[col_date_data], "%Y-%m-%d"),
                                          dt.datetime.strptime(str(x[col_date_hour]), '%H')
                                          .time()),
            axis=1)
        data['key'] = 0
        temporal_data['key'] = 0

        data = data.merge(temporal_data, how='outer')
        data.loc[data[col_timestamp_data] != (data[col_timestamp_data].min()), col_live_scoots] = 0
        df = data.drop([col_date_data, col_date_hour, 'key'], axis=1)
        expected_df = None

    if case=="data_w_rnd_date_and_hour_cols":

        start_ts = dt.datetime.strptime("1981-12-01", "%Y-%m-%d")
        end_ts = dt.datetime.strptime("1982-03-01", "%Y-%m-%d")

        event_date_vector = [random_timestamp(start_ts, end_ts).date() for x in range(n_rows)]
        event_hour_vector = [randint(0, 23) for x in range(n_rows)]
        col_values_1_vector = [randrange(-1000, 1000, 1) for x in range(n_rows)]
        df = pd.DataFrame(
            {
                col_date_data: event_date_vector,
                col_date_hour: event_hour_vector,
                col_default_numerical: col_values_1_vector
            }
        )

        df[col_date_data] = pd.to_datetime(df[col_date_data])


        expected_df=None

    if case == "data_w_clusters_and_2_values_cols":

        if "col_numerical_1" in kwargs:
            col_numerical_1 = kwargs.get("col_numerical_1")
        else:
            col_numerical_1 = col_default_numerical_1
        if "col_numerical_2" in kwargs:
            col_numerical_2 = kwargs.get("col_numerical_2")
        else:
            col_numerical_2 = col_default_numerical_2
        n_rows = 10
        col_cluster_id_vector = [f"cluster_id_nr_{x}" for x in range(n_rows)]
        print(kwargs)
        df = pd.DataFrame({
            col_cluster_id: col_cluster_id_vector,
            col_numerical_1: [1, 2, 3, 5, 6.5, 1, 4, 7, 10, 20],
            col_numerical_2: [3, 2, 2, 10, 15, 1, 10, 4, 10, 10]
        })

        expected_df = None

    if case == "data_w_clusters_and_values_cols":

        if "col_numerical" in kwargs:
            col_numerical = kwargs.get("col_numerical")
        else:
            col_numerical = col_default_numerical
        n_rows = 10
        col_cluster_id_vector = [f"cluster_id_nr_{x}" for x in range(n_rows)]

        df = pd.DataFrame({
            col_cluster_id: col_cluster_id_vector,
            col_numerical: [0.5, 0, 10.5, 7.5, 4.5, 10.5, 12.75, 1.5, 2, 20.5]
        })

        expected_df = None

    if case == "data_rides_n_vehs_m_hours_rnd":
        start_date = pd.Timestamp('2021-01-01')
        n_vehicles = kwargs.get("n_vehicles")
        hours = kwargs.get("hours")
        sampling = kwargs.get("sampling")
        start_ts = kwargs.get("start_ts")
        rides_per_scoot_calc_period = kwargs.get("rides_per_scoot_calc_period")
        end_ts = start_ts + pd.to_timedelta(rides_per_scoot_calc_period, unit='h')


        vehicle_id_vector = [''.join(choice(ascii_uppercase) for i in range(8)) for x in range(n_vehicles)]
        col_values_1_vector = [randrange(-1000, 1000, 1) for x in range(n_vehicles)]


        data = {
            col_vehicle_id: vehicle_id_vector,
            col_predict_rides: [1 for ride in range(n_vehicles)],
            f"{col_default_numerical}_1": col_values_1_vector,
            f"{col_default_numerical}_2": col_values_1_vector,
            f"{col_default_numerical}_3": col_values_1_vector
        }
        df = pd.DataFrame(data=data)

        # Add data-hour data
        event_date_vector = [start_date.date().strftime('%Y-%m-%d') for x in range(hours)]
        event_hour_vector = [int(round(x)) for x in range(hours)]
        event_hour_vector = [x % 24 for x in event_hour_vector]

        timestamp_data = pd.DataFrame({
            col_date_data: event_date_vector,
            col_date_hour: event_hour_vector})

        timestamp_data[col_timestamp_data] = timestamp_data.apply(
            lambda x: dt.datetime.combine(dt.datetime.strptime(x[col_date_data], "%Y-%m-%d"),
                                          dt.datetime.strptime(str(x[col_date_hour]), '%H')
                                          .time()),
            axis=1)

        df['key'] = 0
        timestamp_data['key'] = 0
        df = df.merge(timestamp_data, how='outer')
        df = df.drop(['event_date', 'event_hour', 'key'], axis=1)

        # Sample and order data
        df = df.sample(frac=sampling)
        df = df[[f"{col_default_numerical}_1", f"{col_default_numerical}_2", col_timestamp_data,
                 f"{col_default_numerical}_3", col_vehicle_id, col_predict_rides]]

        expected_df = df.copy()
        ts_filter = (expected_df[col_timestamp_data]>=start_ts) & (expected_df[col_timestamp_data]<=end_ts)
        expected_df = expected_df[ts_filter]
        expected_df = (expected_df
                       .groupby(col_vehicle_id)[col_predict_rides]
                       .sum()
                       .reset_index(drop=False))

    if case == "data_rides_3_vehs_in_48_hours_fxd":

        vehicle_id_1 = "scoot_1"
        vehicle_id_2 = "scoot_2"
        vehicle_id_3 = "scoot_3"

        n_rows = 15
        n_vehicles = 3
        vehicle_id_list = [vehicle_id for vehicle_id in [vehicle_id_1, vehicle_id_2, vehicle_id_3]
                           for repetitions in range(int(n_rows/n_vehicles))]

        col_values_1_vector = [randrange(-1000, 1000, 1) for x in range(n_rows)]
        col_values_2_vector = [randrange(-1000, 1000, 1) for x in range(n_rows)]
        col_values_3_vector = [randrange(-1000, 1000, 1) for x in range(n_rows)]

        data = {
            f"{col_default_numerical}_1": col_values_1_vector,
            f"{col_default_numerical}_2": col_values_2_vector,
            col_timestamp_data: [
             pd.Timestamp('2021-06-01 00:00:00'),
             pd.Timestamp('2021-06-01 06:00:00'),
             pd.Timestamp('2021-06-01 12:00:00'),
             pd.Timestamp('2021-06-02 00:00:00'),
             pd.Timestamp('2021-06-02 12:00:00'),

             pd.Timestamp('2021-06-02 03:00:00'),
             pd.Timestamp('2021-06-02 03:45:00'),
             pd.Timestamp('2021-06-02 06:15:59'),
             pd.Timestamp('2021-06-02 03:00:00'),
             pd.Timestamp('2021-06-03 02:12:00'),

             pd.Timestamp('2021-06-01 01:15:00'),
             pd.Timestamp('2021-06-01 02:59:59'),
             pd.Timestamp('2021-06-01 03:01:01'),
             pd.Timestamp('2021-06-01 04:00:00'),
             pd.Timestamp('2021-06-02 00:00:00'),
            ],
            f"{col_default_numerical}_3": col_values_3_vector,
            "vehicle_id": vehicle_id_list
        }

        df = pd.DataFrame(data=data)
        expected_df = None

    if case == "data_w_clusters_predicted_rides_and_live_scoots":

        n_rows = 10
        col_cluster_id_vector = [f"cluster_id_nr_{x}" for x in range(n_rows)]

        df = pd.DataFrame({
            col_cluster_id: col_cluster_id_vector,
            col_live_scoots: [randrange(0, 10, 1) for x in range(n_rows)],
            col_pred_rides_per_scoot_cluster: [0, 1, 2, 10, 20, 30, 0, 1, 2, 50]
        })

        expected_df = None

    if case == "data_w_clusters_predicted_rides_and_live_scoots_2":

        n_rows = 10
        col_cluster_id_vector = [f"cluster_id_nr_{x}" for x in range(n_rows)]

        df = pd.DataFrame({
            col_cluster_id: col_cluster_id_vector,
            col_live_scoots: [0, 5, 15, 10, 5, 30, 0, 10, 20, 5],
            col_pred_rides_per_scoot_cluster: [x for x in range(n_rows)],
            col_sm_delta: [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 100.0]
        })

        expected_df = None

    if case == "data_w_clusters_predicted_rides_and_live_scoots_3":

        n_rows = 5
        col_cluster_id_vector = [f"cluster_id_nr_{x}" for x in range(n_rows)]

        df = pd.DataFrame({
            col_cluster_id: col_cluster_id_vector,
            col_live_scoots: [0, 5, 15, 10, 5],
            col_pred_rides_per_scoot_cluster: [x for x in range(n_rows)],
            col_sm_delta: [0, -5.0, -10.0, 10.0, 25.0]
        })

        expected_df = None

    if case == "data_w_surplus_of_scoots":
        n_rows = 5

        # Surplus of 30 scoots removed over added
        df = pd.DataFrame({
            col_cluster_id: [x for x in range(n_rows)],
            col_live_scoots: [0, 50.0, 10.0, 10.0, 25.0],
            col_sm_delta: [0, -50.0, -10.0, 10.0, 20.0]
        })
        expected_df = None

    if case == "data_w_clusters_predicted_rides_and_live_scoots_4":

        n_rows = 5
        col_cluster_id_vector = [f"cluster_id_nr_{x}" for x in range(n_rows)]

        df = pd.DataFrame({
            col_cluster_id: col_cluster_id_vector,
            col_live_scoots: [10, 5, 15, 0, 5],
            col_pred_rides_per_scoot_cluster: [x for x in range(n_rows)],
            col_sm_delta: [0, -5.0, -10.0, 5.0, 5.0]
        })

        expected_df = None

    if case == "data_w_clusters_predicted_rides_and_live_scoots_5":

        n_rows = 10
        col_cluster_id_vector = [f"cluster_id_nr_{x}" for x in range(n_rows)]

        df = pd.DataFrame({
            col_cluster_id: col_cluster_id_vector,
            col_live_scoots: [5, 10, 8, 15, 20, 30, 0, 5, 2, 5],
            col_pred_rides_per_scoot_cluster: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        expected_df = None

    if case == "data_w_clusters_and_sm_delta":

        n_rows = 10
        col_cluster_id_vector = [f"cluster_id_nr_{x}" for x in range(n_rows)]

        df = pd.DataFrame({
            col_cluster_id: col_cluster_id_vector,
            col_sm_delta: [0, 5, 15, -10, 5, 30, 0, 10, -20, 5]
        })

        expected_df = None

    if case == 'scoots_data_in_4_regions':

        n_vehicles = 100
        n_regions = 4
        n_vehicles_in_region = int(n_vehicles/n_regions)
        col_lon_region_1 = [randrange(1, 49, 1) for x in range(n_vehicles_in_region)]
        col_lon_region_1 = [x/10 for x in col_lon_region_1]
        col_lon_region_2 = [randrange(51, 99, 1) for x in range(n_vehicles_in_region)]
        col_lon_region_2 = [x/10 for x in col_lon_region_2]
        col_lon_region_3 = [randrange(101, 149, 1) for x in range(n_vehicles_in_region)]
        col_lon_region_3 = [x/10 for x in col_lon_region_3]
        col_lon_region_4 = [randrange(151, 199, 1) for x in range(n_vehicles_in_region)]
        col_lon_region_4 = [x/10 for x in col_lon_region_4]


        col_lon = [col_lon_region_1, col_lon_region_2, col_lon_region_3, col_lon_region_4]
        col_lon_values = flatten(col_lon)
        col_lat = [randrange(1, 99, 1) for x in range(n_vehicles)]
        col_lat_values = [x/10 for x in col_lat]

        col_sm_delta_values = [randrange(0, 100, 1) for x in range(n_vehicles)]
        col_pred_rides_per_scoot_cluster_values = [randrange(0, 100, 1) for x in range(n_vehicles)]
        col_live_scoots_values = [1 for x in range(n_vehicles)]

        data = {
            col_lon_clusters: col_lon_values,
            col_lat_clusters: col_lat_values,
            col_sm_delta: col_sm_delta_values,
            f"adj_{col_live_scoots}": col_live_scoots_values,
            col_pred_rides_per_scoot_cluster: col_pred_rides_per_scoot_cluster_values
        }

        df = pd.DataFrame(data=data)
        expected_df = None

    if case == 'scoots_data_in_4_regions_2':

        n_clusters = 12
        n_regions = 4
        n_vehicles_in_region = int(n_clusters/n_regions)
        col_live_scoots_values = [1 for x in range(n_clusters)]

        col_cluster_id_values = [f"cluster_{x}" for x in range(n_clusters)]
        col_regions_values = [f"Region1"]*3 + [f"Region2"]*3 + [f"Region3"]*3 + [f"Region4"]*3

        col_lon_region_1 = [randrange(1, 49, 1) for x in range(n_vehicles_in_region)]
        col_lon_region_1 = [x/10 for x in col_lon_region_1]
        col_lon_region_2 = [randrange(51, 99, 1) for x in range(n_vehicles_in_region)]
        col_lon_region_2 = [x/10 for x in col_lon_region_2]
        col_lon_region_3 = [randrange(101, 149, 1) for x in range(n_vehicles_in_region)]
        col_lon_region_3 = [x/10 for x in col_lon_region_3]
        col_lon_region_4 = [randrange(151, 199, 1) for x in range(n_vehicles_in_region)]
        col_lon_region_4 = [x/10 for x in col_lon_region_4]


        col_lon = [col_lon_region_1, col_lon_region_2, col_lon_region_3, col_lon_region_4]
        col_lon_values = flatten(col_lon)
        col_lat = [randrange(1, 99, 1) for x in range(n_clusters)]
        col_lat_values = [x/10 for x in col_lat]

        data = {
            col_lon_clusters: col_lon_values,
            col_lat_clusters: col_lat_values,
            col_cluster_id: col_cluster_id_values,
            'regions_name': col_regions_values,
            f"adj_{col_live_scoots}": col_live_scoots_values,
            col_pred_rides_per_scoot_cluster: [1, 2, 3, 3, 2, 1, 1, 1, 1, 3, 1, 5],
            col_deploy: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'adj_capped_sm_delta': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1]
        }

        df = pd.DataFrame(data=data)
        expected_df = None

    return df, expected_df
