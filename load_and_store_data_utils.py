import boto3
import datetime
import logging
import os
import pandas as pd
import tempfile
import warnings

import lib.base_utils as bs_uts
from lib.print_out_info import print_out_info


@print_out_info
def load_data_from_local(
    data_format: str,
    path: str,
    data_fn: str
) -> pd.DataFrame:

    """
    Loads data from local.
    Args:
        data_format: .csv or .parquet
        path: local folder where the data is stored
        data_fn: data file name
    Returns:
        data: loaded data
    """

    data_destination = os.path.join(path, data_fn, data_format)

    if data_format == 'csv':
        data = pd.read_csv(data_destination)
    elif data_format == 'parquet':
        data = pd.read_parquet(data_destination)
    else:
        raise Exception(f"Data format not known: {data_format}")

    return data


@print_out_info
def load_latest_data_from_s3(
    s3_bucket: str,
    path: str,
    data_format: str = 'parquet'
) -> pd.DataFrame:

    """
    Finds the latest file with data in given S3 bucket/path and loads it as pandas df.

    Args:
        s3_bucket: S3 bucket
        path: AWS key where the data is stored
        data_format: S3 data format
    Returns:
        data: loaded data df
    """

    s3 = boto3.client('s3')
    tmp_fn = f'{tempfile.gettempdir()}/data.{data_format}'
    if bs_uts.find_latest_file_in_s3_path(
            s3_bucket=s3_bucket,
            path=path
    ):

        most_recent_key = bs_uts.find_latest_file_in_s3_path(
            s3_bucket=s3_bucket,
            path=path
        )

        s3.download_file(
            Bucket=s3_bucket,
            Key=most_recent_key,
            Filename=tmp_fn
        )
    else:
        raise Exception(f"No files not present in S3 bucket and path: {s3_bucket}/{path}")

    data_format = data_format[1:] if data_format[0] == '.' else data_format
    if data_format == 'csv':
        data = pd.read_csv(tmp_fn)
    elif data_format == 'parquet':
        data = pd.read_parquet(tmp_fn)
    else:
        raise Exception(f"Data format not known: {data_format}")

    return data


@print_out_info
def filter_and_sample_model_data(
    data: pd.DataFrame,
    first_day_data: datetime.date,
    last_day_data: datetime.date,
    sampling: bool,
    sample_size: [int, float, None],
    conf: dict,
    logger: logging.Logger
) -> pd.DataFrame:

    """
    Filters data for given dates and hours. Samples data.
    Args:
        data: model input data
        first_day_data: first day of data filter
        last_day_data: last day of data filter
        sampling: if sample data
        sample_size: sample size
        conf: config as dict
        logger: logger object
    Returns:
        data: filtered and sampled data

    Notes:
        - at this point, prediction data for AVDO and VDO contains only historical records, and can be sampled for VDO
        project
        - sample does not work for AVDO (when calculating rides in the next 24h). Is it possible to tackle this?
        Sampling on clustering level later is ok
        - if last day data is in the future (ie. predictions) the future data is not created here, but in cartesian
        product step
        - for AVDO sample would be possible for live data location

    TODO:
     - set max row sampling (e.g. if over 32 gigabytes, limit to 32 gigabytes)?

    """
    # Check the dataframe size (megabytes)
    data_size_gb = data.memory_usage(deep=True).sum()/(1024*1024*1024)
    logger.info(f"Dataframe datasize = {data_size_gb} gigabytes")
    if data_size_gb > 32:
        logger.warning(f"Dataframe datasize exceeds {data_size_gb} gigabytes")

    # Filter data
    if not conf["col_date"] in data.columns:
        data[conf["col_date"]] = data[conf["col_timestamp"]].dt.date
    data[conf['col_date']] = pd.to_datetime(data[conf['col_date']])

    if not conf["col_hour"] in data.columns:
        data[conf["col_hour"]] = data[conf["col_timestamp"]].dt.hour

    if not conf["col_count_data"] in data.columns:
        data[conf["col_count_data"]] = 1

    data = data.loc[(data[conf['col_date']] >= first_day_data) &
                    (data[conf['col_date']] <= last_day_data)]
    data = data.loc[(data[conf['col_hour']] >= conf['first_operation_hour']) &
                    (data[conf['col_hour']] <= conf['last_operation_hour'])]

    if sampling:
        # Sample data
        data = data.sample(
            frac=sample_size,
            random_state=45
        ).reset_index(drop=True)
        data[conf['col_count_data']] = data[conf['col_count_data']] / sample_size

    else:
        data = data

    return data
