import haversine
import pandas as pd
import create_dataframes_blueprints as blp

import lib.clustering_funcs as clst_funcs

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


def test_add_average_distance_per_cluster_col():
    # SETUP
    number_of_events = 50
    number_of_clusters = 5

    input_data, _ = blp.create_test_dataframe(
        case=2,
        n_rows=number_of_events
    )

    input_clusters, _ = blp.create_test_dataframe(
        case=3,
        n_rows=number_of_clusters
    )

    input_clusters_cols = input_clusters.columns
    input_data = clst_funcs.assign_data_to_nearest_clusters(
        data=input_data,
        clusters=input_clusters,
        col_cluster_clusters=col_cluster_id,
        col_cluster_data=col_cluster_id,
        cols_coords_clusters=cols_coords_clusters,
        cols_coords_data=cols_coords_data
    )

    input_data = input_data.merge(
        input_clusters,
        how='left',
        on='cluster_id'
    )
    input_data_columns = input_data.columns
    expected_first_5_distances = pd.Series({
        0: 4426.201,
        1: 33226.895,
        2: 35727.484,
        3: 21875.091,
        4: 14722.337
    })
    expected_avg_distances = pd.Series({
        0: 18748.498,
        1: 12238.714,
        2: 22698.489,
        3: 18022.975,
        4: 29179.872
    })

    # EXCERCISE
    output_data, output_clusters = clst_funcs.add_average_distance_per_cluster_col(
        data=input_data,
        clusters=input_clusters,
        cols_coords_clusters=cols_coords_clusters,
        cols_coords_data=cols_coords_data,
        col_cluster_id=col_cluster_id,
        col_distance=col_distance
    )

    output_data = output_data.round({
        col_distance: 3}
    )
    output_clusters = output_clusters.round({
        'avg_dist_to_scooter': 3}
    )
    output_first_5_distances = output_data[col_distance].head()
    avg_distances = output_clusters['avg_dist_to_scooter']

    # TEST
    assert output_first_5_distances.equals(expected_first_5_distances)
    assert avg_distances.equals(expected_avg_distances)
    assert output_clusters[input_clusters_cols].equals(input_clusters)
    assert output_data[input_data_columns].equals(input_data[input_data_columns])
    assert 'avg_dist_to_scooter' in output_clusters.columns
    assert col_distance in output_data.columns


def test_add_clusters_count_col():
    # SETUP
    number_of_clusters = 5
    number_of_events = 50

    input_data, _ = blp.create_test_dataframe(
        case=4,
        n_rows=number_of_events,
        number_of_clusters=5
    )

    input_clusters, _ = blp.create_test_dataframe(
        case=3,
        n_rows=number_of_clusters
    )
    input_clusters = input_clusters[[col_cluster_id, col_lat_clusters, col_lon_clusters]]
    expected_output = pd.DataFrame({
        'cluster_id': {
            0: f'{col_cluster_id_prefix}0',
            1: f'{col_cluster_id_prefix}1',
            2: f'{col_cluster_id_prefix}2',
            3: f'{col_cluster_id_prefix}3',
            4: f'{col_cluster_id_prefix}4'
        },
        col_count_clusters: {0: 13, 1: 5, 2: 9, 3: 14, 4: 9}})

    # EXERCISE
    output_clusters = clst_funcs.add_clusters_count_col(
        clusters=input_clusters,
        data=input_data,
        col_cluster=col_cluster_id,
        col_count_data=col_count_data,
        col_count_clusters=col_count_clusters
    )

    output_clusters = output_clusters[[col_cluster_id, col_count_clusters]]

    # TEST
    assert expected_output.equals(output_clusters)
    assert output_clusters[col_count_clusters].sum()==number_of_events


def test_assign_data_to_nearest_clusters():
    # SETUP
    number_of_events = 100
    number_of_clusters = 10

    input_data, _ = blp.create_test_dataframe(
        case=4,
        n_rows=number_of_events,
        number_of_clusters=number_of_clusters
    )

    input_clusters, _ = blp.create_test_dataframe(
        case=3,
        n_rows=number_of_clusters
    )
    expected_first_ten_results = pd.Series({
        0: f'{col_cluster_id_prefix}2',
        1: f'{col_cluster_id_prefix}7',
        2: f'{col_cluster_id_prefix}8',
        3: f'{col_cluster_id_prefix}9',
        4: f'{col_cluster_id_prefix}5',
        5: f'{col_cluster_id_prefix}3',
        6: f'{col_cluster_id_prefix}3',
        7: f'{col_cluster_id_prefix}7',
        8: f'{col_cluster_id_prefix}8',
        9: f'{col_cluster_id_prefix}4'
    })
    expected_columns = {col_cluster_id, col_lat_data, col_lon_data}

    # EXERCISE
    output_data = clst_funcs.assign_data_to_nearest_clusters(
        data=input_data,
        clusters=input_clusters,
        col_cluster_clusters=col_cluster_id,
        col_cluster_data=col_cluster_id,
        cols_coords_clusters=cols_coords_clusters,
        cols_coords_data=cols_coords_data
    )
    first_ten_results = output_data[col_cluster_id].head(10)

    # TEST
    assert expected_first_ten_results.equals(first_ten_results)
    assert len(output_data) == number_of_events
    assert expected_columns.issubset(set(output_data.columns))


def test_calc_shortest_dist_between_clusters_existence():
    # SETUP
    number_of_clusters = 5

    input_clusters, _ = blp.create_test_dataframe(
        case=3,
        n_rows=number_of_clusters
    )
    expected_centers_columns = {
        col_cluster_id,
        col_lat_clusters,
        col_lon_clusters,
        'distance_to_closest_cluster'
    }
    expected_combs_columns = {
        f'{col_cluster_id}_1',
        f'{col_cluster_id}_2',
        f'{col_lat_clusters}_1',
        f'{col_lat_clusters}_2',
        f'{col_lon_clusters}_1',
        f'{col_lon_clusters}_2',
        col_distance
    }
    expected_agg_combs_columns = {f'{col_cluster_id}_1', col_distance}

    # EXERCISE
    output_clusters, output_cart_prod, output_min_distances = clst_funcs.calc_shortest_dist_between_clusters(
        clusters=input_clusters,
        col_cluster=col_cluster_id,
        col_longitude=col_lon_clusters,
        col_latitude=col_lat_clusters,
        col_distance=col_distance
    )

    assert expected_centers_columns.issubset(set(output_clusters.columns))
    assert expected_combs_columns.issubset(set(output_cart_prod))
    assert expected_agg_combs_columns.issubset(set(output_min_distances.columns))
    assert type(output_clusters) == pd.DataFrame
    assert type(output_cart_prod) == pd.DataFrame
    assert type(output_min_distances) == pd.DataFrame
    assert not output_clusters.empty
    assert not output_cart_prod.empty
    assert not output_min_distances.empty


def test_calc_shortest_dist_between_clusters_contents():
    # SETUP
    number_of_clusters = 5

    input_clusters, _ = blp.create_test_dataframe(
        case=3,
        n_rows=number_of_clusters
    )
    # with the power of the seed - works only with number_of_clusters = 5
    expected_output_min_distances = pd.DataFrame({
        f'{col_cluster_id}_1': {
            0: f'{col_cluster_id_prefix}0',
            1: f'{col_cluster_id_prefix}1',
            2: f'{col_cluster_id_prefix}2',
            3: f'{col_cluster_id_prefix}3',
            4: f'{col_cluster_id_prefix}4'
        },
        col_distance: {
            0: 24905.021,
            1: 24905.021,
            2: 32834.326,
            3: 32834.326,
            4: 55087.364
        }
    })

    # EXERCISE
    output_clusters, output_cart_prod, output_min_distances = clst_funcs.calc_shortest_dist_between_clusters(
        clusters=input_clusters,
        col_cluster=col_cluster_id,
        col_longitude=col_lon_clusters,
        col_latitude=col_lat_clusters,
        col_distance=col_distance
    )
    output_min_distances = output_min_distances.round(
        {col_distance: 3}
    )

    assert output_min_distances.equals(expected_output_min_distances)
    assert len(output_clusters) == number_of_clusters
    assert len(output_cart_prod) == number_of_clusters * number_of_clusters - number_of_clusters
    assert len(output_min_distances) == number_of_clusters


def test_distribute_scooters_by_clusters():

    # SETUP
    number_of_clusters = 5
    n_scooters = 100

    input_clusters, _ = blp.create_test_dataframe(
        case=3,
        n_rows=number_of_clusters
    )
    n_events = input_clusters[col_count_clusters].sum()
    expected_output = pd.DataFrame({
        col_count_clusters: {0: 0, 1: 100, 2: 200, 3: 300, 4: 400},
        'n_scooters': {0: 0.0, 1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0}
    })

    # EXERCISE
    output_clusters = clst_funcs.distribute_scooters_by_clusters(
        input_clusters,
        n_scooters,
        n_events
    )

    output_clusters = output_clusters[[col_count_clusters, 'n_scooters']]

    # TEST
    assert output_clusters.equals(expected_output)


def test_obtain_n_clusters_range_no_hybrid():
    # SETUP
    oa_area = 100
    static_oa_area = 0
    min_dps_per_km2 = 10
    max_dps_per_km2 = 20
    clusters_steps = 10
    expected_output = range(1000, 2001, 100)

    # EXERCISE
    output = clst_funcs.obtain_n_clusters_range(
        oa_area,
        static_oa_area,
        min_dps_per_km2,
        max_dps_per_km2,
        clusters_steps=clusters_steps
    )

    # ASSERT
    assert output == expected_output


def test_obtain_n_clusters_range_hybrid():
    # SETUP
    oa_area = 100
    static_oa_area = 10
    min_dps_per_km2 = 10
    max_dps_per_km2 = 20
    clusters_steps = 10
    expected_output = range(900, 1801, 90)

    # EXERCISE
    output = clst_funcs.obtain_n_clusters_range(
        oa_area,
        static_oa_area,
        min_dps_per_km2,
        max_dps_per_km2,
        clusters_steps=clusters_steps
    )

    # ASSERT
    assert output == expected_output
