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


@print_out_info
def wf_model_evaluation(
    cloud_environment: str,
    project: str,
    city: [str, int, float],
    target: str,
    client: str,
    logger: logging.Logger
):
    """
    Obtains the the best model hyperparameters using Walk Forward Validation (WFV) and stores them in Cloud.

    At the moment evaluates only Random Forest model.

    Note: conf['col_target'] and target are two different things:
        - target is the name of the predicted quantity used to create folders for laoding and saving data/results
        - conf['col_target'] is the column name containing target variable in the input data

     Args:
        cloud_environment: cloud environment
        city: city name
        project: project name
        target: target
        client: client name
        logger: logger object
    Returns:
        Random Forest model best hyperparameters
    """

    # Setup logger
    logger.info(f"Start Walk Forward evaluation for city: {city}, client: {client}")
    logger.info('Load config')
    conf = api_uts.get_config_via_api_gw(
        city_name=city,
        client=client,
        project=project
    )

    # Calculate evaluation time period
    first_day_data, last_day_data, _ = md_uts.calculate_model_data_time_period(
        ml_framework_step='evaluation',
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

    # this step as separate, project dependent functions, for AVDO it will also calculate delta
    logger.info('Prepare data')
    model_data = dt_trf.data_prep_train_eval_pred(
        input_data=input_data,
        project=project,
        first_day_data=first_day_data,
        last_day_data=last_day_data,
        ml_framework_step='evaluation',
        sample_size=conf["evaluation_sample_size"],
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

    scorers = md_scr.make_evaluation_scorers(
        scorers_names=conf['evaluation_model_scorers']
    )

    logger.info('Split data')
    X_eval, y_eval, wfv_generator = md_uts.wf_split_evaluation_data(
        data=data,
        cols_features_full=cols_features_full,
        col_target=col_target,
        conf=conf,
        logger=logger
    )

    logger.info('Evaluate model')
    evaluation_results = md_uts.wf_evaluate_rf_model(
        X_eval=X_eval,
        y_eval=y_eval,
        wfv_generator=wfv_generator,
        scorers=scorers,
        scorers_names=conf['evaluation_model_scorers'],
        conf=conf
    )

    logger.info('Save best parameters in S3 config file for given city')
    ld_uts.store_evaluation_results_to_cloud(
        evaluation_results=evaluation_results,
        cloud_environment=cloud_environment,
        project=project,
        city=city,
        target=target,
        conf=conf
    )

    return evaluation_results


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

    final_dps = wf_model_evaluation(
        city=city,
        cloud_environment=cloud_environment,
        project=project,
        target=target,
        client=client,
        logger=logger
    )
