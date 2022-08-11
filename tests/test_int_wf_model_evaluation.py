import json
import os
import pandas as pd
from unittest.mock import patch
from pathlib import Path
import sys

# Add submodules to path
cwd = Path(os.getcwd())
parent_cwd = str(cwd.parent.absolute())
sys.path.append(os.path.join(parent_cwd, 'routing_engine'))
sys.path.append(os.path.join(parent_cwd, 'fleetyfly_api_python_client'))

import lib.base_utils as bs_uts
from lib_main.wf_model_evaluation import wf_model_evaluation
from lib.dynamo_utils import mock_vdo_config
import lib.map_utils as map_uts
from lib.map_utils import extract_map_features_from_json as _extract_map_features_from_json

"""
TODO: Create weather variables for testing
"""

city = 'Wakanda'
client = "fleetyfly"
project = "vdo"

logger = bs_uts.logger_setup(
        log_file='log_file.log'
    )
os.environ['environment'] = "test"


# Read config from template function
conf = mock_vdo_config(
            city
        )
conf['input_data_format'] = 'parquet'

events_data = pd.read_csv('../../../data/rides_data_template.csv')
cluster_data = pd.read_csv('../../../data/clustering_template.csv')

first_day_data = events_data[conf['col_date']].min()
last_day_data = events_data[conf['col_date']].max()


# Read map from template json file
with open('../../../data/map_template.json') as f:
    map_dict = json.load(f)

# Copy functions to be used outside mocking environment:
def copy_extract_map_features_from_json(
    oa_poly_layer_name,
    forbidden_poly_layer_name,
    static_poly_layer_name,
    static_dps_layer_name,
    regions_layer_name,
    map_dict,
    city_type,
    col_static_dps_lon,
    col_static_dps_lat,
    col_static_dps_name
):

    return _extract_map_features_from_json(
        oa_poly_layer_name,
        forbidden_poly_layer_name,
        static_poly_layer_name,
        static_dps_layer_name,
        regions_layer_name,
        map_dict,
        city_type,
        col_static_dps_lon,
        col_static_dps_lat,
        col_static_dps_name
    )


@patch('lib.model_utils.calculate_model_data_time_period')
@patch('lib.load_and_store_data_utils.load_cluster_data_from_cloud')
@patch('lib.map_utils.load_map_features_from_cloud')
@patch('lib.base_utils.s3_store_df')
@patch('lib.map_utils.snap_to_roads')
@patch('lib.load_and_store_data_utils.load_eval_train_pred_data')
@patch('lib.api_utils.get_config_via_api_gw')
def test_wf_model_evaluation(
        mocked_get_config_via_api_gw,
        mocked_load_eval_train_pred_data,
        mocked_snap_to_roads,
        mocked_s3_store_df,
        mocked_load_map_features_from_cloud,
        mocked_load_cluster_data_from_cloud,
        mocked_calculate_model_data_time_period
):
    city_type = 'hybrid'
    cloud_environment='AWS'
    client = 'fleetfly'

    mocked_get_config_via_api_gw.return_value = None

    def my_snap_to_roads(locations):
        return locations

    mocked_snap_to_roads.side_effect = my_snap_to_roads

    mocked_s3_store_df.return_value = None

    mocked_city_config = mock_vdo_config(city=city)
    mocked_city_config['city_type'] = city_type
    mocked_city_config["add_weather_variables_step"] = False

    mocked_get_config_via_api_gw.return_value = mocked_city_config
    mocked_load_cluster_data_from_cloud.return_value = cluster_data

    mocked_calculate_model_data_time_period.return_value = [first_day_data, last_day_data, None]

    oa_polygons, forbidden_area_polygons, static_oa_polygons, regions_polygons, static_dps, oa_area, static_dp_area = \
        copy_extract_map_features_from_json(
            oa_poly_layer_name='OA',
            forbidden_poly_layer_name='FA',
            static_poly_layer_name='SA',
            static_dps_layer_name='StaticDPs',
            regions_layer_name='Regions',
            map_dict=map_dict,
            city_type=city_type,
            col_static_dps_lon='cluster_longitude',
            col_static_dps_lat='cluster_latitude',
            col_static_dps_name='cluster_id'
        )

    input_data = {
        "events_data": events_data,
        "cluster_data": cluster_data,
        "oa_polygons": oa_polygons
    }

    mocked_load_eval_train_pred_data.return_value = [input_data, None, mocked_city_config]

    mocked_load_map_features_from_cloud.return_value = [
    oa_polygons,
        map_uts.extract_map_features_from_json(
            oa_poly_layer_name='OA',
            forbidden_poly_layer_name='FA',
            static_poly_layer_name='SA',
            static_dps_layer_name='StaticDPs',
            map_dict=map_dict,
            city_type=city_type,
            col_static_dps_lon='cluster_longitude',
            col_static_dps_lat='cluster_latitude',
            col_static_dps_name='cluster_id'
        )[1],
        static_oa_polygons,
        static_dps,
        oa_area,
        static_dp_area
    ]

    target='rides'
    config_type='test'

    param_grid = {
        'n_estimators': conf['rf_n_estimators'],
        'max_depth': conf['rf_max_depth'],
        'min_samples_leaf': conf['rf_min_samples_leaf'],
        'max_features': conf['rf_max_features']
    }

    number_of_params_combinations = 1
    for key, value in param_grid.items():
        number_of_params_combinations = number_of_params_combinations * len(value)

    # ASSERT
    evaluation_results = wf_model_evaluation(
        cloud_environment=cloud_environment,
        project=project,
        city=city,
        target=target,
        client=client,
        logger=logger
    )

    # VERIFY
    assert evaluation_results.shape[0] == number_of_params_combinations
    assert evaluation_results.dropna().shape[0] == number_of_params_combinations
    assert (evaluation_results['mean_test_rmsle_no_negative'] >= 0).all()
    assert (evaluation_results['rank_test_rmsle_no_negative'] >= 1).all()
    assert (evaluation_results['mean_test_mape_total_no_negative'] >= 0).all()
    assert (evaluation_results['rank_test_mape_total_no_negative'] >= 1).all()

