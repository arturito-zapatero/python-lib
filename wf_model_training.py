import logging
import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'fleetyfly_api_python_client'))

import lib.api_utils as api_uts
import lib.base_utils as bs_uts
import lib.data_transformations as dt_trf
import lib.data_utils as dt_uts
import lib.load_and_store_data_utils as ld_uts
import lib.model_scorers as md_scr
import lib.model_utils as md_uts
from lib.print_out_info import print_out_info


# TODO: which metrics are used for training?

@print_out_info
def wf_model_training(
    city: [str, int, float],
    cloud_environment: str,
    project: str,
    target: str,
    client: str,
    logger: logging.Logger
):
    """
    Trains the model using Walk Forward Validation (WFV) and stores produced sklearn pipelines in cloud.

    At the moment trains only Random Forest model.

    Args:
        city: city name
        cloud_environment: cloud environment
        target: target, currently rides and arrivals
        client: client name
        logger: logger object
    Returns:
        Random Forest and OHE (?) model pipelines.
    """

    logger.info(f"Start model training for city: {city}, client: {client}")
    logger.info('Load config')
    conf = api_uts.get_config_via_api_gw(
        city_name=city,
        client=client,
        project=project
    )

    # Calculate training time period
    first_day_data, last_day_data, _ = md_uts.calculate_model_data_time_period(
        ml_framework_step='training',
        conf=conf
    )

    logger.info('Load data')
    input_data, conf = ld_uts.load_eval_train_pred_data(
        cloud_environment=cloud_environment,
        project=project,
        client=client,
        city=city,
        target=target,
        first_day_data=first_day_data,
        last_day_data=last_day_data,
        conf=conf,
        logger=logger
    )

    # Initialize model variables
    cols_numerical = conf['cols_numerical_basic']
    cols_categorical = conf['cols_categorical_basic']
    cols_cyclical_in = conf['cols_cyclical_in_basic']
    cols_plot = conf['cols_plot_basic']
    col_target = conf['col_target']
    col_predict = conf['col_predict']

    logger.info('Prepare data')
    model_data = dt_trf.data_prep_train_eval_pred(
        input_data=input_data,
        project=project,
        first_day_data=first_day_data,
        last_day_data=last_day_data,
        ml_framework_step='training',
        sample_size=conf["training_sample_size"],
        conf=conf,
        logger=logger
    )

    logger.info('Add features to data')
    model_data_features, cols_numerical, cols_categorical, cols_cyclical_in, cols_plot, _, _, conf = \
        dt_trf.wf_feature_engineering(
            data=model_data,
            weather_data=input_data['weather_data'],
            cols_numerical=cols_numerical,
            cols_categorical=cols_categorical,
            cols_cyclical_in=cols_cyclical_in,
            cols_plot=cols_plot,
            conf=conf,
            logger=logger
        )

    logger.info('Data OHE')
    data, _, cols_features_full = dt_uts.ohe(
        input_data=model_data_features,
        pipeline_ohe=None,
        cols_categorical=cols_categorical,
        cols_numerical=cols_numerical,
        id_columns=conf['cols_id'],
        target_column=conf['col_target']
    )

    logger.info('Split data')
    X_train, y_train = md_uts.wf_split_training_data(
        data=data,
        cols_features_full=cols_features_full,
        col_target=col_target,
        conf=conf,
        logger=logger
    )

    best_hyperparams_grid = ld_uts.load_best_model_hyperparams_from_cloud(
        cloud_environment=cloud_environment,
        project=project,
        city=city,
        target=target,
        conf=conf
    )

    # Train pipeline, predict on training data
    logger.info('Train model')
    y_hats, trained_model_pipeline = md_uts.wf_train_rf_model(
        X_train=X_train,
        y_train=y_train,
        best_hyperparams_grid=best_hyperparams_grid,
        conf=conf
    )

    # Merge predictions with original data
    data = data.set_index(conf['cols_id'])
    training_results = data.merge(y_hats, left_index=True, right_index=True)

    training_errors_dict = md_scr.calculate_error_metrics(
        results=training_results,
        col_target=col_target,
        col_predict=col_predict,
        conf=conf
    )

    ld_uts.store_training_results_to_cloud(
        training_results=training_results,
        training_errors=training_errors_dict,
        trained_model_pipeline=trained_model_pipeline,
        cloud_environment=cloud_environment,
        city=city,
        project=project,
        target=target,
        conf=conf
    )

    return trained_model_pipeline, training_results, training_errors_dict


if __name__ == "__main__":
    # Setup logger
    logger = bs_uts.logger_setup(
        log_file='log_file.log'
    )
    args = bs_uts.parse_ecs_arguments([{'name': 'target', 'type': 'str', 'is_list': False},
                                       {'name': 'city', 'type': 'str', 'is_list': False},
                                       {'name': 'client', 'type': 'str', 'is_list': False},
                                       {'name': 'cloud_environment', 'type': 'str', 'is_list': False},
                                       {'name': 'project', 'type': 'str', 'is_list': False}])
    city = args['city']
    cloud_environment = args['cloud_environment']
    client = args['client']
    target = args['target']
    project = args['project']

    final_dps = wf_model_training(
        city=city,
        cloud_environment=cloud_environment,
        project=project,
        target=target,
        client=client,
        logger=logger
    )
