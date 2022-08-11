import numpy as np
import pandas as pd
import datetime as dt


def create_optimization_test_dictionary(
    case: str,
    conf: dict
) -> dict:

    if case == 'test_optimization_minimize_levelled_exp_function_general':
        number_of_clusters_test_1 = 6
        col_cluster_id_vector = [f"cluster_id_nr_{x}" for x in range(number_of_clusters_test_1)]

        test_dict = {'test_1': {
                'data': pd.DataFrame({
                    conf['col_predict_arrivals']: [1, 1, 1, 1, 1, 1],
                    conf['col_predict_rides']: [1, 1, 1, 1, 1, 1],
                    conf['col_live_scoots']: [1, 5, 0, 0, 5, 0],
                    conf['col_date']: ['2020-01-01'] * number_of_clusters_test_1,
                    conf['col_hour']: [1 for x in range(number_of_clusters_test_1)],
                    conf['col_cluster_id']: col_cluster_id_vector
                }),
                'available_fleet': 50,
                'max_scooters_per_cluster': 5
            },
                'test_2': {
                    'data': pd.DataFrame({
                        conf['col_predict_arrivals']: [10, 5, 1, 10, 15, 1],
                        conf['col_predict_rides']: [1, 1, 1, 1, 1, 1],
                        conf['col_live_scoots']: [1, 5, 0, 10, 5, 0],
                        conf['col_date']: ['2020-01-01'] * number_of_clusters_test_1,
                        conf['col_hour']: [1 for x in range(number_of_clusters_test_1)],
                        conf['col_cluster_id']: col_cluster_id_vector
                    }),
                    'available_fleet': 50,
                    'max_scooters_per_cluster': 5
                },
                'test_3': {
                    'data': pd.DataFrame({
                        conf['col_predict_arrivals']: [1, 1, 1, 1, 1, 1],
                        conf['col_predict_rides']: [10, 15, 10, 5, 1, 20],
                        conf['col_live_scoots']: [1, 0, 0, 0, 1, 0],
                        conf['col_date']: ['2020-01-01'] * number_of_clusters_test_1,
                        conf['col_hour']: [1 for x in range(number_of_clusters_test_1)],
                        conf['col_cluster_id']: col_cluster_id_vector
                    }),
                    'available_fleet': 50,
                    'max_scooters_per_cluster': 5
                },
                'test_4': {
                    'data': pd.DataFrame({
                        conf['col_predict_arrivals']: [1, 1, 1, 1, 1, 1],
                        conf['col_predict_rides']: [10, 15, 10, 5, 1, 20],
                        conf['col_live_scoots']: [1, 0, 0, 0, 1, 0],
                        conf['col_date']: ['2020-01-01'] * number_of_clusters_test_1,
                        conf['col_hour']: [1 for x in range(number_of_clusters_test_1)],
                        conf['col_cluster_id']: col_cluster_id_vector
                    }),
                    'available_fleet': 250,
                    'max_scooters_per_cluster': 50
                },
                'test_5': {
                    'data': pd.DataFrame({
                        conf['col_predict_arrivals']: [1, 1, 1, 1, 1, 1],
                        conf['col_predict_rides']: [1, 1, 1, 1, 1, 1],
                        conf['col_live_scoots']: [10, 15, 20, 30, 10, 40],
                        conf['col_date']: ['2020-01-01'] * number_of_clusters_test_1,
                        conf['col_hour']: [1 for x in range(number_of_clusters_test_1)],
                        conf['col_cluster_id']: col_cluster_id_vector
                    }),
                    'available_fleet': 250,
                    'max_scooters_per_cluster': 50
                }
            }

    if case == 'test_optimization_minimize_levelled_exp_function_specific':
        number_of_clusters = 8
        col_cluster_id_vector = [f"cluster_id_nr_{int(np.floor(0.5 * x))}" for x in range(number_of_clusters)]
        test_dict = {
            'test_1': {
                'data': pd.DataFrame({
                    conf['col_predict_arrivals']: [0, 0, 0, 0, 0, 0, 0, 0],
                    conf['col_predict_rides']: [2, 6, 0, 2, 0, 0, 4, 2],
                    conf['col_live_scoots']: [0, 0, 0, 0, 0, 0, 0, 0],
                    conf['col_date']: ['2020-01-01'] * number_of_clusters,
                    conf['col_hour']: [x for x in range(int(number_of_clusters / 4))] * 4,
                    conf['col_cluster_id']: col_cluster_id_vector}),
                'maximize_rides_coeff': -1,
                'minimize_rebalances_coeff': 1.9,
                'maximize_rides_per_scoot_coeff': 0,
                'minimize_rebalances_per_cluster_coeff': 0,
                'expected_deployment_vector': [4, 0, 0, 3],
                'test_name': 'only_rides_beta_1_9'},
            'test_2': {
                'data': pd.DataFrame({
                    conf['col_predict_arrivals']: [0, 0, 0, 0, 0, 0, 0, 0],
                    conf['col_predict_rides']: [2, 6, 0, 2, 0, 0, 4, 2],
                    conf['col_live_scoots']: [0, 0, 0, 0, 0, 0, 0, 0],
                    conf['col_date']: ['2020-01-01'] * number_of_clusters,
                    conf['col_hour']: [x for x in range(int(number_of_clusters / 4))] * 4,
                    conf['col_cluster_id']: col_cluster_id_vector}),
                'maximize_rides_coeff': -1,
                'minimize_rebalances_coeff': 1,
                'maximize_rides_per_scoot_coeff': 0,
                'minimize_rebalances_per_cluster_coeff': 0,
                'expected_deployment_vector': [4, 1, 0, 3],
                'test_name': 'only_rides_beta_1_0'},
            'test_3': {
                'data': pd.DataFrame({
                    conf['col_predict_arrivals']: [0, 0, 0, 0, 0, 0, 0, 0],
                    conf['col_predict_rides']: [2, 6, 0, 2, 0, 0, 4, 2],
                    conf['col_live_scoots']: [1, 0, 1, 0, 0, 0, 1, 0],
                    conf['col_date']: ['2020-01-01'] * number_of_clusters,
                    conf['col_hour']: [x for x in range(int(number_of_clusters / 4))] * 4,
                    conf['col_cluster_id']: col_cluster_id_vector}),
                'maximize_rides_coeff': -1,
                'minimize_rebalances_coeff': 1.9,
                'maximize_rides_per_scoot_coeff': 0,
                'minimize_rebalances_per_cluster_coeff': 0,
                'expected_deployment_vector': [3, 0, 0, 2],
                'test_name': 'rides_with_current_beta_1_9'},
            'test_4': {
                'data': pd.DataFrame({
                    conf['col_predict_arrivals']: [0, 0, 0, 0, 0, 0, 0, 0],
                    conf['col_predict_rides']: [2, 6, 0, 2, 0, 0, 4, 2],
                    conf['col_live_scoots']: [3, 0, 1, 0, 0, 0, 1, 0],
                    conf['col_date']: ['2020-01-01'] * number_of_clusters,
                    conf['col_hour']: [x for x in range(int(number_of_clusters / 4))] * 4,
                    conf['col_cluster_id']: col_cluster_id_vector}),
                'maximize_rides_coeff': -1,
                'minimize_rebalances_coeff': 1.9,
                'maximize_rides_per_scoot_coeff': 0,
                'minimize_rebalances_per_cluster_coeff': 0,
                'expected_deployment_vector': [0, 0, 0, 1],  # should be [0, 0, 0, 2]
                'test_name': 'rides_with_current_beta_1_9_v2'},
            'test_5': {
                'data': pd.DataFrame({
                    conf['col_predict_arrivals']: [2, 6, 0, 2, 0, 0, 4, 2],
                    conf['col_predict_rides']: [0, 0, 0, 0, 0, 0, 0, 0],
                    conf['col_live_scoots']: [0, 0, 0, 0, 0, 0, 0, 0],
                    conf['col_date']: ['2020-01-01'] * number_of_clusters,
                    conf['col_hour']: [x for x in range(int(number_of_clusters / 4))] * 4,
                    conf['col_cluster_id']: col_cluster_id_vector}),
                'maximize_rides_coeff': -1,
                'minimize_rebalances_coeff': 1.9,
                'maximize_rides_per_scoot_coeff': 0,
                'minimize_rebalances_per_cluster_coeff': 0,
                'expected_deployment_vector': [0, 0, 0, 0],
                'test_name': 'only_arrivals_beta_1_9'},
            'test_6': {
                'data': pd.DataFrame({
                    conf['col_predict_arrivals']: [2, 6, 0, 2, 0, 0, 4, 2],
                    conf['col_predict_rides']: [0, 0, 0, 0, 0, 0, 0, 0],
                    conf['col_live_scoots']: [1, 0, 0, 0, 0, 0, 0, 0],
                    conf['col_date']: ['2020-01-01'] * number_of_clusters,
                    conf['col_hour']: [x for x in range(int(number_of_clusters / 4))] * 4,
                    conf['col_cluster_id']: col_cluster_id_vector}),
                'maximize_rides_coeff': 0,
                'minimize_rebalances_coeff': 0,
                'maximize_rides_per_scoot_coeff': -1,
                'minimize_rebalances_per_cluster_coeff': 1.9,
                'expected_deployment_vector': [-1, 0, 0, 0],
                'test_name': 'only_arrivals_phi_1.9'},
            'test_7': {
                'data': pd.DataFrame({
                    conf['col_predict_arrivals']: [2, 6, 0, 2, 0, 0, 4, 2],
                    conf['col_predict_rides']: [0, 0, 0, 0, 0, 0, 0, 0],
                    conf['col_live_scoots']: [1, 0, 0, 0, 0, 0, 0, 0],
                    conf['col_date']: ['2020-01-01'] * number_of_clusters,
                    conf['col_hour']: [x for x in range(int(number_of_clusters / 4))] * 4,
                    conf['col_cluster_id']: col_cluster_id_vector}),
                'maximize_rides_coeff': 0,
                'minimize_rebalances_coeff': 0,
                'maximize_rides_per_scoot_coeff': -1,
                'minimize_rebalances_per_cluster_coeff': 1,
                'expected_deployment_vector': [-1, 0, 0, 0],
                'test_name': 'only_arrivals_phi_1'},
            'test_8': {
                'data': pd.DataFrame({
                    conf['col_predict_arrivals']: [2, 6, 0, 2, 0, 0, 4, 2],
                    conf['col_predict_rides']: [0, 0, 0, 0, 0, 0, 0, 0],
                    conf['col_live_scoots']: [1, 0, 0, 0, 0, 0, 0, 0],
                    conf['col_date']: ['2020-01-01'] * number_of_clusters,
                    conf['col_hour']: [x for x in range(int(number_of_clusters / 4))] * 4,
                    conf['col_cluster_id']: col_cluster_id_vector}),
                'maximize_rides_coeff': 0,
                'minimize_rebalances_coeff': 0,
                'maximize_rides_per_scoot_coeff': 0,
                'minimize_rebalances_per_cluster_coeff': 1,
                'expected_deployment_vector': [-1, 0, 0, 0],
                'test_name': 'only_arrivals_phi_1_no_theta'},
            'test_9': {
                'data': pd.DataFrame({
                    conf['col_predict_arrivals']: [2, 6, 0, 2, 0, 0, 4, 2],
                    conf['col_predict_rides']: [0, 0, 0, 0, 0, 0, 0, 0],
                    conf['col_live_scoots']: [0, 0, 0, 0, 0, 0, 0, 0],
                    conf['col_date']: ['2020-01-01'] * number_of_clusters,
                    conf['col_hour']: [x for x in range(int(number_of_clusters / 4))] * 4,
                    conf['col_cluster_id']: col_cluster_id_vector}),
                'maximize_rides_coeff': 0,
                'minimize_rebalances_coeff': 0,
                'maximize_rides_per_scoot_coeff': 0,
                'minimize_rebalances_per_cluster_coeff': 1,
                'expected_deployment_vector': [0, 0, 0, 0],  # will not remove if there is no current
                'test_name': 'only_arrivals_phi_theta_no_current'},
            'test_10': {
                'data': pd.DataFrame({
                    conf['col_predict_arrivals']: [2, 6, 0, 2, 0, 0, 4, 2],
                    conf['col_predict_rides']: [2, 6, 0, 2, 0, 0, 4, 2],
                    conf['col_live_scoots']: [0, 0, 0, 0, 0, 0, 0, 0],
                    conf['col_date']: ['2020-01-01'] * number_of_clusters,
                    conf['col_hour']: [x for x in range(int(number_of_clusters / 4))] * 4,
                    conf['col_cluster_id']: col_cluster_id_vector}),
                'maximize_rides_coeff': -1,
                'minimize_rebalances_coeff': 1.9,
                'maximize_rides_per_scoot_coeff': 0,
                'minimize_rebalances_per_cluster_coeff': 0,
                'expected_deployment_vector': [0, 0, 0, 0],
                'test_name': 'arrivals_equals_rides'},
            'test_11': {
                'data': pd.DataFrame({
                    conf['col_predict_arrivals']: [2, 6, 0, 2, 0, 0, 4, 2],
                    conf['col_predict_rides']: [2, 6, 0, 2, 0, 0, 4, 2],
                    conf['col_live_scoots']: [1, 0, 0, 0, 0, 0, 0, 0],
                    conf['col_date']: ['2020-01-01'] * number_of_clusters,
                    conf['col_hour']: [x for x in range(int(number_of_clusters / 4))] * 4,
                    conf['col_cluster_id']: col_cluster_id_vector}),
                'maximize_rides_coeff': -1,
                'minimize_rebalances_coeff': 1.9,
                'maximize_rides_per_scoot_coeff': 0,
                'minimize_rebalances_per_cluster_coeff': 0,
                'expected_deployment_vector': [-1, 0, 0, 0],  # removes one scooter because of the init guess vector
                'test_name': 'arrivals_equals_rides_with_current'},
            'test_12': {
                'data': pd.DataFrame({
                    conf['col_predict_arrivals']: [2, 6, 0, 2, 0, 0, 4, 2],
                    conf['col_predict_rides']: [2, 6, 0, 2, 0, 0, 4, 2],
                    conf['col_live_scoots']: [2, 0, 0, 0, 0, 0, 0, 0],
                    conf['col_date']: ['2020-01-01'] * number_of_clusters,
                    conf['col_hour']: [x for x in range(int(number_of_clusters / 4))] * 4,
                    conf['col_cluster_id']: col_cluster_id_vector}),
                'maximize_rides_coeff': -1,
                'minimize_rebalances_coeff': 1.9,
                'maximize_rides_per_scoot_coeff': 0,
                'minimize_rebalances_per_cluster_coeff': 0,
                'expected_deployment_vector': [-2, 0, 0, 0],
                'test_name': 'arrivals_equals_rides_with_current_v2'},
            'test_13': {
                'data': pd.DataFrame({
                    conf['col_predict_arrivals']: [2, 6, 0, 2, 0, 0, 4, 2],
                    conf['col_predict_rides']: [2, 6, 0, 2, 0, 0, 4, 2],
                    conf['col_live_scoots']: [2, 0, 0, 0, 0, 0, 0, 0],
                    conf['col_date']: ['2020-01-01'] * number_of_clusters,
                    conf['col_hour']: [x for x in range(int(number_of_clusters / 4))] * 4,
                    conf['col_cluster_id']: col_cluster_id_vector}),
                'maximize_rides_coeff': -1,
                'minimize_rebalances_coeff': 1.9,
                'maximize_rides_per_scoot_coeff': 0,
                'minimize_rebalances_per_cluster_coeff': 0,
                'expected_deployment_vector': [-2, 0, 0, 0],
                'test_name': 'arrivals_equals_rides_with_current_v3'},
            'test_14': {
                'data': pd.DataFrame({
                    conf['col_predict_arrivals']: [2, 6, 0, 2, 0, 0, 4, 2],
                    conf['col_predict_rides']: [2, 6, 0, 2, 0, 0, 4, 2],
                    conf['col_live_scoots']: [2, 0, 0, 0, 0, 0, 0, 0],
                    conf['col_date']: ['2020-01-01'] * number_of_clusters,
                    conf['col_hour']: [x for x in range(int(number_of_clusters / 4))] * 4,
                    conf['col_cluster_id']: col_cluster_id_vector}),
                'maximize_rides_coeff': -1,
                'minimize_rebalances_coeff': 1.0,
                'maximize_rides_per_scoot_coeff': 0,
                'minimize_rebalances_per_cluster_coeff': 0,
                'expected_deployment_vector': [-2, 0, 0, 0],
                'test_name': 'arrivals_equals_rides_with_current_beta_1_0'},
            'test_15': {
                'data': pd.DataFrame({
                    conf['col_predict_arrivals']: [1, 3, 0, 1, 0, 0, 2, 1],
                    conf['col_predict_rides']: [2, 6, 0, 2, 0, 0, 4, 2],
                    conf['col_live_scoots']: [0, 0, 0, 0, 0, 0, 0, 0],
                    conf['col_date']: ['2020-01-01'] * number_of_clusters,
                    conf['col_hour']: [x for x in range(int(number_of_clusters / 4))] * 4,
                    conf['col_cluster_id']: col_cluster_id_vector}),
                'maximize_rides_coeff': -1.9,
                'minimize_rebalances_coeff': 1.0,
                'maximize_rides_per_scoot_coeff': 0,
                'minimize_rebalances_per_cluster_coeff': 0,
                'expected_deployment_vector': [0, 0, 0, 0],  # we level rides and arrivals so will be equal
                'test_name': 'arrivals_half_rides'},
            'test_16': {
                'data': pd.DataFrame({
                    conf['col_predict_arrivals']: [1, 1, 1, 1, 1, 1, 1, 1],
                    conf['col_predict_rides']: [2, 1, 1, 1, 1, 1, 1, 1],
                    conf['col_live_scoots']: [0, 0, 0, 0, 0, 0, 0, 0],
                    conf['col_date']: ['2020-01-01'] * number_of_clusters,
                    conf['col_hour']: [x for x in range(int(number_of_clusters / 4))] * 4,
                    conf['col_cluster_id']: col_cluster_id_vector}),
                'maximize_rides_coeff': -1.9,
                'minimize_rebalances_coeff': 1.0,
                'maximize_rides_per_scoot_coeff': 0,
                'minimize_rebalances_per_cluster_coeff': 0,
                'expected_deployment_vector': [1, 0, 0, 0],  # deployed because of the init guess vector
                'test_name': 'arrivals_and_rides_v1'},
            'test_17': {
                'data': pd.DataFrame({
                    conf['col_predict_arrivals']: [2, 1, 1, 1, 1, 1, 1, 1],
                    conf['col_predict_rides']: [1, 1, 1, 1, 1, 1, 1, 1],
                    conf['col_live_scoots']: [0, 0, 0, 0, 0, 0, 0, 0],
                    conf['col_date']: ['2020-01-01'] * number_of_clusters,
                    conf['col_hour']: [x for x in range(int(number_of_clusters / 4))] * 4,
                    conf['col_cluster_id']: col_cluster_id_vector}),
                'maximize_rides_coeff': -1.9,
                'minimize_rebalances_coeff': 1.0,
                'maximize_rides_per_scoot_coeff': 0,
                'minimize_rebalances_per_cluster_coeff': 0,
                'expected_deployment_vector': [0, 0, 0, 0],
                'test_name': 'arrivals_and_rides_v2'}
        }
    if case == 'test_calculate_first_hours_wo_scooters_metric_1':
        number_of_clusters = 6
        day_date = '2020-01-01'
        test_dict = {'test_1': {
            'data': pd.DataFrame({
                conf['col_predict_arrivals']: [1, 1, 1, 1, 0, 0],
                conf['col_predict_rides']: [1, 1, 1, 1, 1, 1],
                conf['col_live_scoots']: [0, 0, 0, 0, 0, 0],
                conf['col_date']: [day_date] * number_of_clusters,
                conf['col_hour']: [x for x in range(number_of_clusters)],
                conf['col_cluster_id']: ['cluster_id_123'] * number_of_clusters
            }),
            'first_hour_wo_scooters': 4,
            'hours_to_no_scooters': 5.0
        },
            'test_2': {
                'data': pd.DataFrame({
                    conf['col_predict_arrivals']: [0, 0, 5, 0, 0, 0],
                    conf['col_predict_rides']: [1, 1, 1, 20, 1, 1],
                    conf['col_live_scoots']: [5, 0, 0, 0, 0, 0],
                    conf['col_date']: [day_date] * number_of_clusters,
                    conf['col_hour']: [x for x in range(number_of_clusters)],
                    conf['col_cluster_id']: ['cluster_id_123'] * number_of_clusters
                }),
                'first_hour_wo_scooters': 3,
                'hours_to_no_scooters': 4.0
            }
            ,
            'test_3': {
                'data': pd.DataFrame({
                    conf['col_predict_arrivals']: [1, 1, 1, 1, 0, 0],
                    conf['col_predict_rides']: [10, 1, 1, 1, 1, 1],
                    conf['col_live_scoots']: [0, 0, 0, 0, 0, 0],
                    conf['col_date']: [day_date] * number_of_clusters,
                    conf['col_hour']: [x for x in range(number_of_clusters)],
                    conf['col_cluster_id']: ['cluster_id_123'] * number_of_clusters
                }),
                'first_hour_wo_scooters': 0,
                'hours_to_no_scooters': 1.0
            }
            ,
            'test_4': {
                'data': pd.DataFrame({
                    conf['col_predict_arrivals']: [1, 1, 1, 1, 0, 0],
                    conf['col_predict_rides']: [1, 1, 1, 1, 1, 1],
                    conf['col_live_scoots']: [100, 0, 0, 0, 0, 0],
                    conf['col_date']: [day_date] * number_of_clusters,
                    conf['col_hour']: [x for x in range(number_of_clusters)],
                    conf['col_cluster_id']: ['cluster_id_123'] * number_of_clusters
                }),
                'first_hour_wo_scooters': 4,
                'hours_to_no_scooters': 5.0
            }
        }
    if case == 'test_calculate_first_hours_wo_scooters_metric_2':
        number_of_clusters = 6
        day_date = '2020-01-01'
        test_dict = {
            'test_1': {
                'data': pd.DataFrame({
                    conf['col_predict_arrivals']: [1, 1, 1, 1, 0, 0],
                    conf['col_predict_rides']: [1, 1, 1, 10, 1, 1],
                    conf['col_live_scoots']: [0, 0, 0, 0, 0, 0],
                    conf['col_deploy']: [1, 0, 0, 0, 0, 0],
                    conf['col_date']: [day_date] * number_of_clusters,
                    conf['col_hour']: [x for x in range(number_of_clusters)],
                    conf['col_cluster_id']: ['cluster_id_123'] * number_of_clusters
                }),
                'exp_first_hour_wo_scooters': '2020-01-01 03:00',
                'exp_rel_time_to_no_scooters': 0.25,
                'exp_hours_to_no_scooters': 4
            },
            'test_2': {
                'data': pd.DataFrame({
                    conf['col_predict_arrivals']: [1, 1, 1, 1, 0, 0],
                    conf['col_predict_rides']: [1, 1, 1, 10, 1, 1],
                    conf['col_live_scoots']: [0, 0, 0, 0, 0, 0],
                    conf['col_deploy']: [-1, 0, 0, 0, 0, 0],
                    conf['col_date']: [day_date] * number_of_clusters,
                    conf['col_hour']: [x for x in range(number_of_clusters)],
                    conf['col_cluster_id']: ['cluster_id_123'] * number_of_clusters
                }),
                'exp_first_hour_wo_scooters': 'NA',
                'exp_rel_time_to_no_scooters': 0,
                'exp_hours_to_no_scooters': 'NA'
            },
            'test_3': {
                'data': pd.DataFrame({
                    conf['col_predict_arrivals']: [1, 1, 1, 1, 0, 0],
                    conf['col_predict_rides']: [1, 1, 1, 10, 1, 1],
                    conf['col_live_scoots']: [0, 0, 0, 0, 0, 0],
                    conf['col_deploy']: [0, 0, 0, 0, 0, 0],
                    conf['col_date']: [day_date] * number_of_clusters,
                    conf['col_hour']: [x for x in range(number_of_clusters)],
                    conf['col_cluster_id']: ['cluster_id_123'] * number_of_clusters
                }),
                'exp_first_hour_wo_scooters': 'NA',
                'exp_rel_time_to_no_scooters': 0,
                'exp_hours_to_no_scooters': 'NA'
            }
        }
    if case == 'test_calculate_optimization_performance_metrics_1':
        test_dict = {'test_1': {
            'data': pd.DataFrame({
                conf['col_deploy']: [5, 0, 0, 0, 0, 0]
            }),
            'scoots_deploy_penalty_offset': 1,
            'scoots_remove_penalty_offset': 1,
            'exp_out_data_1': 4,
            'exp_out_data_2': 0
        },
            'test_2': {
                'data': pd.DataFrame({
                    conf['col_deploy']: [0, 0, 0, 0, 0, 0]
                }),
                'scoots_deploy_penalty_offset': 1,
                'scoots_remove_penalty_offset': 1,
                'exp_out_data_1': 0,
                'exp_out_data_2': 0
            },
            'test_3': {
                'data': pd.DataFrame({
                    conf['col_deploy']: [-5, 0, 0, 0, 0, 0]
                }),
                'scoots_deploy_penalty_offset': 1,
                'scoots_remove_penalty_offset': 1,
                'exp_out_data_1': 0,
                'exp_out_data_2': 4
            },
            'test_4': {
                'data': pd.DataFrame({
                    conf['col_deploy']: [-1, 0, 0, 0, 0, 0]
                }),
                'scoots_deploy_penalty_offset': 1,
                'scoots_remove_penalty_offset': 1,
                'exp_out_data_1': 0,
                'exp_out_data_2': 0
            }
        }

    if case == 'test_calculate_optimization_performance_metrics_2':
        number_of_clusters = 6
        day_date = '2020-01-01'
        test_dict = {'test_1': {
            'data': pd.DataFrame({
                conf['col_predict_rides']: [5, 1, 1, 1, 1, 1],
                conf['col_live_scoots']: [0, 0, 0, 0, 0, 0],
                conf['col_date']: [day_date] * number_of_clusters,
                conf['col_hour']: [x for x in range(number_of_clusters)]
            }),
            'exp_next_hour_penalty': 5
        },
            'test_2': {
                'data': pd.DataFrame({
                    conf['col_predict_rides']: [5, 10, 10, 10, 10, 10],
                    conf['col_live_scoots']: [3, 0, 0, 0, 0, 0],
                    conf['col_date']: [day_date] * number_of_clusters,
                    conf['col_hour']: [x for x in range(number_of_clusters)]
                }),
                'exp_next_hour_penalty': 2
            },
            'test_3': {
                'data': pd.DataFrame({
                    conf['col_predict_rides']: [0, 1, 1, 1, 1, 1],
                    conf['col_live_scoots']: [0, 0, 0, 0, 0, 0],
                    conf['col_date']: [day_date] * number_of_clusters,
                    conf['col_hour']: [x for x in range(number_of_clusters)]
                }),
                'exp_next_hour_penalty': 0
            }
        }

    if case == 'test_get_optimization_time_range':
        test_dict = {
            'test_1': {'rebalance_period_hours': 4,
                       'start_time': dt.datetime(2020, 1, 1, 2, 25, 00, 000000),
                       'optim_start_time': None,
                       'optim_end_time': None,
                       'exp_optim_start_time': dt.datetime(2020, 1, 1, 2, 00, 00, 000000),
                       'exp_optim_end_time': dt.datetime(2020, 1, 1, 2 + 4, 00, 00, 000000)
                       },
            'test_2': {'rebalance_period_hours': 4,
                       'start_time': dt.datetime(2020, 1, 1, 2, 35, 00, 000000),
                       'optim_start_time': None,
                       'optim_end_time': None,
                       'exp_optim_start_time': dt.datetime(2020, 1, 1, 2 + 1, 00, 00, 000000),
                       'exp_optim_end_time': dt.datetime(2020, 1, 1, 2 + 1 + 4, 00, 00, 000000)
                       },
            'test_3': {'rebalance_period_hours': 24,
                       'start_time': dt.datetime(2020, 1, 1, 2, 25, 00, 000000),
                       'optim_start_time': None,
                       'optim_end_time': None,
                       'exp_optim_start_time': dt.datetime(2020, 1, 1, 2, 00, 00, 000000),
                       'exp_optim_end_time': dt.datetime(2020, 1, 2, 2, 00, 00, 000000)
                       },
            'test_4': {'rebalance_period_hours': 28,
                       'start_time': dt.datetime(2020, 1, 1, 2, 25, 00, 000000),
                       'optim_start_time': None,
                       'optim_end_time': None,
                       'exp_optim_start_time': dt.datetime(2020, 1, 1, 2, 00, 00, 000000),
                       'exp_optim_end_time': dt.datetime(2020, 1, 2, 2 + 4, 00, 00, 000000)
                       },
            'test_5': {'rebalance_period_hours': 24,
                       'start_time': dt.datetime(2020, 12, 31, 9, 35, 00, 000000),
                       'optim_start_time': None,
                       'optim_end_time': None,
                       'exp_optim_start_time': dt.datetime(2020, 12, 31, 10, 00, 00, 000000),
                       'exp_optim_end_time': dt.datetime(2021, 1, 1, 10, 00, 00, 000000)
                       }

        }
    if case == 'test_filter_optimization_data':
        test_dict = test_filter_optim_data_time_ranges_dict = {
            'test_1': {'optim_start_time': dt.datetime(2020, 12, 18, 11, 0),
                        'optim_end_time': dt.datetime(2020, 12, 18, 18, 0),
                        'arrival_results': None,
                        'rides_results': None},
            'test_2': {'optim_start_time': dt.datetime(2020, 12, 31, 11, 0),
                       'optim_end_time': dt.datetime(2021, 1, 1, 13, 0),
                       'arrival_results': None,
                       'rides_results': None},
            'test_3': {'optim_start_time': dt.datetime(2020, 12, 18, 11, 0),
                       'optim_end_time': dt.datetime(2020, 12, 22, 18, 0),
                       'arrival_results': None,
                       'rides_results': None},
            'test_4': {'optim_start_time': pd.Timestamp('2020-12-31 11:00:00'),
                       'optim_end_time': pd.Timestamp('2021-01-01 13:00:00'),
                       'arrival_results': None,
                       'rides_results': None}
        }

    if case == 'test_merge_optimization_data':
        test_dict = {
            'test_1': {'results': None},
            'test_2': {'results': None},
            'test_3': {'results': None},
            'test_4': {'input_pred_df_results': None,
                       'current_results': None}
        }
    if case == 'test_adjust_optimization_data_by_hours':
        # Because of the bounds we can only deploy between (-current, max(max_scoot_number-current, 0))
        test_dict = {
            'test_1':
            {'data': pd.DataFrame({
                conf['col_predict_arrivals']: [1, 1, 1, 1, 1, 1],
                conf['col_predict_rides']: [1, 1, 1, 1, 1, 1],
                conf['col_live_scoots']: [1, 0, 0, 0, 0, 0],
                conf['col_date']: ['2020-01-01'] * 6,
                conf['col_hour']: [x for x in range(6)]}),
                'deployment_vector': [1],
                'objective_id': 'levelled_exp_function',
                'expected_available_scooters': [3, 3, 3, 3, 3, 3],
                'expected_actual_rides': [1, 1, 1, 1, 1, 1]
            },
            'test_2': {'data': pd.DataFrame({
                conf['col_predict_arrivals']: [1, 1, 1, 1, 1, 1],
                conf['col_predict_rides']: [1, 1, 1, 1, 1, 1],
                conf['col_live_scoots']: [1, 0, 0, 0, 0, 0],
                conf['col_date']: ['2020-01-01'] * 6,
                conf['col_hour']: [x for x in range(6)]}),
                'deployment_vector': [-1],
                'objective_id': 'levelled_exp_function',
                'expected_available_scooters': [1, 1, 1, 1, 1, 1],
                'expected_actual_rides': [1, 1, 1, 1, 1, 1]
            },
            'test_3': {'data': pd.DataFrame({
                conf['col_predict_arrivals']: [1, 1, 1, 1, 1, 1],
                conf['col_predict_rides']: [1, 1, 1, 1, 1, 1],
                conf['col_live_scoots']: [0, 0, 0, 0, 0, 0],
                conf['col_date']: ['2020-01-01'] * 6,
                conf['col_hour']: [x for x in range(6)]}),
                'deployment_vector': [-1],
                'objective_id': 'levelled_exp_function',
                'expected_available_scooters': [0, 1, 1, 1, 1, 1],
                'expected_actual_rides': [0, 1, 1, 1, 1, 1]
            },
            'test_4': {'data': pd.DataFrame({
                conf['col_predict_arrivals']: [1, 1, 1, 1, 1, 1],
                conf['col_predict_rides']: [2, 2, 2, 3, 3, 3],
                conf['col_live_scoots']: [0, 0, 0, 0, 0, 0],
                conf['col_date']: ['2020-01-01'] * 6,
                conf['col_hour']: [x for x in range(6)]}),
                'deployment_vector': [0],
                'objective_id': 'levelled_exp_function',
                'expected_available_scooters': [1, 1, 1, 1, 1, 1],
                'expected_actual_rides': [1, 1, 1, 1, 1, 1]
            },
            'test_5': {'data': pd.DataFrame({
                conf['col_predict_arrivals']: [1, 1, 1, 1, 1, 1],
                conf['col_predict_rides']: [2, 2, 2, 3, 3, 3],
                conf['col_live_scoots']: [0, 0, 0, 0, 0, 0],
                conf['col_date']: ['2020-01-01'] * 6,
                conf['col_hour']: [x for x in range(6)]}),
                'deployment_vector': [7],
                'objective_id': 'levelled_exp_function',
                'expected_available_scooters': [8, 7, 6, 5, 3, 1],
                'expected_actual_rides': [2, 2, 2, 3, 3, 1]
            },
            'test_6': {'data': pd.DataFrame({
                conf['col_predict_arrivals']: [1, 1, 1, 1, 1, 1],
                conf['col_predict_rides']: [2, 2, 2, 3, 3, 3],
                conf['col_live_scoots']: [2, 0, 0, 0, 0, 0],
                conf['col_date']: ['2020-01-01'] * 6,
                conf['col_hour']: [x for x in range(6)]}),
                'deployment_vector': [7],
                'objective_id': 'levelled_exp_function',
                'expected_available_scooters': [10, 9, 8, 7, 5, 3],
                'expected_actual_rides': [2, 2, 2, 3, 3, 3]
            },
            'test_7': {'data': pd.DataFrame({
                conf['col_predict_arrivals']: [1, 1, 1, 1, 1, 1],
                conf['col_predict_rides']: [2, 2, 2, 3, 3, 3],
                conf['col_live_scoots']: [7, 0, 0, 0, 0, 0],
                conf['col_date']: ['2020-01-01'] * 6,
                conf['col_hour']: [x for x in range(6)]}),
                'deployment_vector': [2],
                'objective_id': 'levelled_exp_function',
                'expected_available_scooters': [10, 9, 8, 7, 5, 3],
                'expected_actual_rides': [2, 2, 2, 3, 3, 3]
            }
        }
    if case == 'test_level_rides_arrivals':
        test_dict = {
            'test_1': {'data': pd.DataFrame({
                conf['col_predict_arrivals']: [1, 1, 1, 1],
                conf['col_predict_rides']: [1, 1, 1, 1]
            }),
                'level_coeff_1': 0.5,
                'level_coeff_2': 0.5,
                'level_coeff_3': 0.5,
                'level_coeff_4': 0.5,
                'expected_rides': 4,
                'expected_arrivals': 4
            },
            'test_2': {'data': pd.DataFrame({
                conf['col_predict_arrivals']: [1, 1, 1, 1],
                conf['col_predict_rides']: [1, 1, 1, 1]
            }),
                'level_coeff_1': 1,
                'level_coeff_2': 1,
                'level_coeff_3': 1,
                'level_coeff_4': 1,
                'expected_rides': 8,
                'expected_arrivals': 8
            },
            'test_3': {'data': pd.DataFrame({
                conf['col_predict_arrivals']: [1, 1, 1, 1],
                conf['col_predict_rides']: [1, 1, 1, 1]
            }),
                'level_coeff_1': 1.5,
                'level_coeff_2': 1.5,
                'level_coeff_3': 1.5,
                'level_coeff_4': 1.5,
                'expected_rides': 12,
                'expected_arrivals': 12
            },
            'test_4': {'data': pd.DataFrame({
                conf['col_predict_arrivals']: [1, 1, 1, 1],
                conf['col_predict_rides']: [1, 1, 1, 1]
            }),
                'level_coeff_1': 0,
                'level_coeff_2': 0,
                'level_coeff_3': 0,
                'level_coeff_4': 0,
                'expected_rides': 0,
                'expected_arrivals': 0
            },
            'test_5': {'data': pd.DataFrame({
                conf['col_predict_arrivals']: [1, 1, 1, 1],
                conf['col_predict_rides']: [1, 1, 1, 1]
            }),
                'level_coeff_1': 0,
                'level_coeff_2': 1.2,
                'level_coeff_3': 0,
                'level_coeff_4': 1,
                'expected_rides': 4.8,
                'expected_arrivals': 4.0
            },
            'test_6': {'data': pd.DataFrame({
                conf['col_predict_arrivals']: [1, 1, 1, 1],
                conf['col_predict_rides']: [1, 1, 1, 1]
            }),
                'level_coeff_1': 0.5,
                'level_coeff_2': 1,
                'level_coeff_3': 0,
                'level_coeff_4': 1,
                'expected_rides': 6.0,
                'expected_arrivals': 4.0
            },
            'test_7': {'data': pd.DataFrame({
                conf['col_predict_arrivals']: [2, 2, 2, 2],
                conf['col_predict_rides']: [1, 1, 1, 1]
            }),
                'level_coeff_1': 0.5,
                'level_coeff_2': 0.5,
                'level_coeff_3': 0.5,
                'level_coeff_4': 0.5,
                'expected_rides': 6,
                'expected_arrivals': 6
            }
            ,
            'test_8': {'data': pd.DataFrame({
                conf['col_predict_arrivals']: [1, 1, 1, 1],
                conf['col_predict_rides']: [2, 2, 2, 2]
            }),
                'level_coeff_1': 0.5,
                'level_coeff_2': 0.5,
                'level_coeff_3': 0.5,
                'level_coeff_4': 0.5,
                'expected_rides': 6,
                'expected_arrivals': 6
            },
            'test_9': {'data': pd.DataFrame({
                conf['col_predict_arrivals']: [2, 2, 2, 2],
                conf['col_predict_rides']: [1, 1, 1, 1]
            }),
                'level_coeff_1': 0.5,
                'level_coeff_2': 0.5,
                'level_coeff_3': 0.5,
                'level_coeff_4': 0.5,
                'expected_rides': 6,
                'expected_arrivals': 6
            },
            'test_10': {'data': pd.DataFrame({
                conf['col_predict_arrivals']: [0, 0, 0, 8],
                conf['col_predict_rides']: [6, 8, 8, 10]
            }),
                'level_coeff_1': 0.5,
                'level_coeff_2': 0.5,
                'level_coeff_3': 0.5,
                'level_coeff_4': 0.5,
                'expected_rides': 20,
                'expected_arrivals': 20
            }
        }


    return test_dict
