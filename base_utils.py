import argparse
import boto3
import botocore
import joblib
import json
import logging
import os
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
from sklearn.pipeline import Pipeline
import tempfile
import time

from lib.print_out_info import print_out_info


"""
General remarks:
- the object key (or key name) uniquely identifies the object in an Amazon S3 bucket (key does not contains bucket name)
- (S3) path defines a path to given object (without the object name)
- key/fkey = path + file name
- a prefix is the complete path in front of the object name, which includes the bucket name 
- folder is similar to folder in Unix/Windows
"""


def check_folder_exists(
    s3_bucket: str,
    path: str
) -> bool:

    """
    Checks if folder exists in given S3 buckets path. Folder can be empty.
    Args:
        s3_bucket: S3 bucket
        path: path to the S3 folder

    Returns:
        True if folder exists, False otherwise
    """

    s3 = boto3.client('s3')
    path = path.rstrip('/')
    resp = s3.list_objects(Bucket=s3_bucket, Prefix=path, Delimiter='/',MaxKeys=1)

    return 'CommonPrefixes' in resp


def get_ssm_secret(parameter_name: str):
    ssm = boto3.client("ssm", region_name="eu-central-1")
    return ssm.get_parameter(
        Name=parameter_name,
        WithDecryption=True
    )


@print_out_info
def connect_boto(
    service_name: str,
    region_name: str = ''
):
    """
    Connects to an AWS service (both locally and in the server)
    Args:
        service_name: AWS service name
        region_name: AWS region

    Returns:
        client: AWS client object
    """

    region_name = region_name if region_name else None
    client = boto3.client(
        service_name,
        region_name=region_name
    )

    return client


@print_out_info
def find_latest_file_in_s3_path(
    s3_bucket: str,
    path: str,
    file_type: [str, bool, None] = None
) -> [str, None]:

    """
    Finds latest (by creation/modification name) file in given AWS folder. Can return only given file types. Also
    works if path have several child folders (finds the latest file from all the child folders).
    Args:
        s3_bucket: S3 bucket
        path: path to the S3 folder
        file_type: returns only files of this type
    Returns:
        most_recent_key: latest file in given S3 buckets key, returns file key (full path and file name, no bucket)
    """

    keys = list_pages_in_s3_path(
        s3_bucket=s3_bucket,
        path=path
    )
    if not keys:
        print(f'No files in {s3_bucket}\{path}')
        return None

    keys_df = pd.DataFrame.from_records(keys)
    keys_df['file_type'] = keys_df['Key'].apply(lambda x: x.split('/')[-1].split('.')[-1])
    keys_df['is_file'] = keys_df['Key'].apply(lambda x: True if x.split('/')[-1] else False)
    keys_df = keys_df.loc[keys_df['is_file'] == True]
    keys_df = keys_df.sort_values(
        by=['LastModified'],
        ascending=False
    )
    if file_type:
        keys_df = keys_df.loc[keys_df['file_type'] == file_type]

    most_recent_key = keys_df.reset_index(drop=True).loc[0, 'Key']

    return most_recent_key


@print_out_info
def list_pages_in_s3_path(
    s3_bucket: str,
    path: str
) -> [list, None]:

    """
    Lists pages for all files in given AWS path. If no files are present returns empty list.
    Args:
        s3_bucket: S3 bucket
        path: path to the S3 folder

    Returns:
        all_keys: list with pages (containing e.g. keys) for every file in given folder

    Example of page dict:
    {'Key': 'avdo/Szczecin/input_data/',
      'LastModified': datetime.datetime(2022, 1, 31, 16, 59, 49, tzinfo=tzutc()),
      'ETag': '"d41d8cd98f00b204e9800998ecf8427e"',
      'Size': 0,
      'StorageClass': 'STANDARD'}
    """

    try:
        s3 = boto3.client(service_name='s3')
        folder_exists = check_folder_exists(
            s3_bucket=s3_bucket,
            path=path
        )
        if not folder_exists:
            raise Exception(f"Folder {path} does not exist in S3 bucket {s3_bucket}")

        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(
            Bucket=s3_bucket,
            Prefix=path
        )

        all_keys = []
        for page in pages:
            if 'Contents' in page:
                all_keys.extend(page['Contents'])

    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "NoSuchBucket":
            raise Exception(f"Bucket {s3_bucket} does not exists")
        else:
            print(f"{e.response['Error']['Code']}")
            raise e

    return all_keys


def parse_ecs_arguments(
    args: list
) -> dict:

    """
    Gets function input parameters from ECS task definition
    Args:
        args: list with dictionaries defining arguments to be parsed, e.g.:

        [{'name': 'city', 'type': 'str', 'is_list': False},
         {'name': 'city_types', 'type': 'str', 'is_list': True},
         {'name': 'client', 'type': 'str', 'is_list': False},
         {'name': 'cloud_environment', 'type': 'str', 'is_list': False},
         {'name': 'project', 'type': 'str', 'is_list': False}]

    Returns:
        Dictionary with arguments
    """

    # Get config param from ECS task
    parser = argparse.ArgumentParser()
    for arg in args:
        if not arg['is_list']:
            parser.add_argument(
                f"--{arg['name']}",
                type=eval(arg['type'])
            )
        else:
            parser.add_argument(
                f"--{arg['name']}",
                nargs="*",
                type=eval(arg['type'])
            )

    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        logging.warning(f"There were some unused arguments passed: {unknown_args}")

    return vars(args)


def s3_list_all_files_dates_in_path(
    s3_bucket: str,
    path: str
) -> [pd.Series, pd.Series]:

    """
    Lists all file creation dates in given S3 bucket path as series of datetime objects. Also returns series with all
    files keys associated with dates (in the same order).
    Args:
        s3_bucket: S3 bucket
        path: path to the S3 folder

    Returns:
        all_files_dates: all creation timestamps of files in given fkey, each element of serie is of type
        datetime64[ns, tzutc()]
        all_files_keys: all files keys associated with dates in all_files_dates, each element of serie is of type string
    """

    response_fkey = list_pages_in_s3_path(
        s3_bucket=s3_bucket,
        path=path
    )
    all_files_fkey = pd.DataFrame.from_records(response_fkey)
    all_files_time_ordered = (all_files_fkey.loc[all_files_fkey['Size'] > 0, :]
                                            .sort_values(by=['LastModified'], ascending=False)
                                            .reset_index()
                                            .loc[:, ['Key', 'LastModified']])
    all_files_keys = all_files_time_ordered['Key']
    all_files_dates = all_files_time_ordered['LastModified']

    # Change from timestamp to date
    all_files_dates = pd.to_datetime(all_files_dates.apply(lambda x: x.date()))

    return all_files_dates, all_files_keys


def logger_setup(
    log_file: str
):
    """
    Creates and setups logger object.
    Args:
        log_file: log file name

    Returns:
        logger: logger object
    """

    logging.basicConfig(
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handle = logging.FileHandler(log_file)
    handle.setFormatter(formatter)

    logger.addHandler(handle)
    logger.info('Logging start')

    return logger


@print_out_info
def s3_download_file(
    s3_bucket: str,
    key: str,
    temporary_fn: [str, None] = None,
    temporary_fs: [str, None] = None,
) -> str:

    """
    Downloads file from AWS S3, saves to local and returns local file path to the file.
    Args:
        s3_bucket: S3 bucket where the .csv is stored
        key: key pointing to file in the bucket
        temporary_fs: temporary local file suffix for random file name
        temporary_fn: temporary local file name
    Returns:
        local_file_path: local file path
    """

    try:
        # Get a tmp file name
        if not temporary_fn:
            temporary_fn = tempfile.NamedTemporaryFile(
                suffix=temporary_fs
            )
            temporary_fn.close()
            temp_file_path = temporary_fn.name
        else:
            temp_file_path = f"{temporary_fn}{temporary_fs}"

        # Download file from S3
        s3 = boto3.client('s3')
        s3.download_file(
            Bucket=s3_bucket,
            Key=key,
            Filename=temp_file_path
        )
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            raise Exception(f"The file requested does not exist in S3 path: {s3_bucket}/{key}")
        else:
            raise e

    return temp_file_path


@print_out_info
def s3_load_file_as_df(
    s3_bucket: str,
    key: str,
    date_columns: [bool, list, dict] = None,
    dtype: dict = None,
    low_memory: bool = True,
    file_type: str = '.csv'
) -> pd.DataFrame:

    """
    Reads a .csv or .parquet file from S3 and returns it as a pandas df.
    Args:
        s3_bucket: S3 bucket where the file is stored
        key: key pointing to file in the bucket
        date_columns: as parse_dates in pd.read_csv
        dtype: as in pd.read_csv
        low_memory: as in pd.read_csv
        file_type: currently '.csv' and '.parquet' are accepted
    Returns:
        data: downloaded .file as pandas dataframe

    """

    # Get a tmp file name
    file_type = file_type[1:] if file_type[0] == '.' else file_type
    temporary_file = tempfile.NamedTemporaryFile(
        suffix=file_type
    )
    temporary_file.close()
    temp_file_path = temporary_file.name

    # Downloads file from S3
    try:
        s3 = boto3.client('s3')
        s3.download_file(
            Bucket=s3_bucket,
            Key=key,
            Filename=temp_file_path
        )
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            raise Exception(f"The file requested does not exist in S3 path: {s3_bucket}/{key}")
        else:
            raise e

    # Read .csv in dataframe
    if file_type == 'csv':
        data = pd.read_csv(
            temp_file_path,
            parse_dates=date_columns,
            dtype=dtype,
            low_memory=low_memory
        )
    elif file_type == 'parquet':
        data = pd.read_parquet(
            temp_file_path
        )
    else:
       raise Exception(f"File type not known: {file_type}")

    try:
        os.remove(temp_file_path)

    except Exception as e:
        pass

    return data


@print_out_info
def s3_read_json(
    s3_bucket: str,
    key: str
) -> dict:
    """
    Function that reads a .json file stored in S3 and returns as a json object.
    Args:
        s3_bucket: bucket name that contains the config file
        key: key pointing to .json file in the bucket

    Returns:
        json object
    """

    # Get S3 client
    s3_client = connect_boto(
        service_name='s3'
    )

    # Get a temporary file name
    temporary_file = tempfile.NamedTemporaryFile()
    temporary_file.close()

    # Download file
    try:
        s3_client.download_file(
            Bucket=s3_bucket,
            Filename=temporary_file.name,
            Key=key
        )
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            raise Exception(f"The json file requested does not exist in S3 path: {s3_bucket}/{key}")
        else:
            raise e
    with open(temporary_file.name) as f:
        return json.load(f)


def s3_read_pipeline(
    s3_bucket: str,
    key: str
) -> Pipeline:
    """
    Function that reads a sklearn Pipeline file (.sav) stored in S3 and returns as a Pipeline object.
    Args:
        s3_bucket: bucket name that contains the config file
        key: key pointing to the pipeline file in the bucket

    Returns:
        Pipeline object
    """

    s3_client = boto3.client(service_name='s3')
    temporary_file = tempfile.NamedTemporaryFile(suffix='.sav')
    temporary_file.close()
    temp_file_path = temporary_file.name

    try:
        s3_client.download_file(
            Bucket=s3_bucket,
            Key=key,
            Filename=temp_file_path
        )
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            raise Exception(f"The pipeline requested does not exist in S3 path: {s3_bucket}/{key}")
        else:
            raise e

    pipeline = joblib.load(temp_file_path)

    return pipeline


@print_out_info
def s3_store_df(
    df: pd.DataFrame,
    s3_bucket: str,
    path: str,
    file_base_name: str,
    file_format: str = 'parquet',
    add_timestamp: bool = True,
    delete_files: bool = True,
    logger: [None, logging.Logger] = None
) -> [str, str]:
    """
    Stores pandas df as file in s3 (storing it first in local and uploading to S3 afterwards). Storage as .csv and
    .parquet files are supported.
    Args:
        df: df to be stored in S3
        s3_bucket: S3 bucket where the df is stored
        path: path pointing to folder where df is stored
        file_base_name: file base name
        file_format: file format
        add_timestamp: if add timestamps
        delete_files: if remove temporary files
        logger: logging logger object

    Returns:
        key: key to the file in the bucket pointing to the stored file
        temporary_file_path: tmp local file path
    """

    # Check for df size
    if df.shape[0] == 0:
        return None

    # Store file locally
    file_format = file_format[1:] if file_format[0] == '.' else file_format

    fn = file_base_name + (time.strftime("%Y%m%d_%H%M%S") if add_timestamp else '') + '.' + file_format
    temp_file_path = tempfile.gettempdir() + '/' + fn
    if file_format == 'parquet':
        arrow_table = pyarrow.Table.from_pandas(
            df=df
        )
        pq.write_table(
            arrow_table,
            temp_file_path,
            flavor='spark'
        )
    elif file_format == 'csv':
        df.to_csv(
            temp_file_path,
            index=False
        )

    # Obtain S3 client
    s3_client = connect_boto(
        service_name='s3'
    )

    # Store file in S3
    key = (path + '/' if path[-1] != '/' else path) + fn
    try:
        with open(temp_file_path, 'rb') as data:
            s3_client.upload_fileobj(
                Fileobj=data,
                Bucket=s3_bucket,
                Key=key
            )
        if logger:
            logger.info(f"Dataframe stored in: {s3_bucket}/{key}")
    except Exception:
        raise Exception(f"Error saving dataframe in: {s3_bucket}/{key}")

    # Delete file
    if delete_files:
        try:
            os.remove(
                temp_file_path
            )
        except Exception as e:
            pass

    return key, temp_file_path


@print_out_info
def s3_store_file(
    local_file_path: str,
    s3_bucket: str,
    key: str
) -> str:
    """
    Uploads local file of any type to S3.
    Args:
        local_file_path: local file path
        s3_bucket: S3 bucket where the file will be stored
        key: key pointing to the stored file in the bucket

    Returns:
        key: key pointing to the stored file in the bucket
    """
    try:
        with open(local_file_path, "rb") as local_file:
            s3_client = boto3.client(
                service_name="s3"
            )
            s3_client.upload_fileobj(
                Fileobj=local_file,
                Bucket=s3_bucket,
                Key=key
            )
    except Exception:
        raise Exception(f"Error uploading file to: {s3_bucket}/{key}")

    return key


@print_out_info
def s3_store_json(
    s3_bucket: str,
    path: str,
    file_name: [str, None],
    json_dict: dict
) -> None:
    """
        Uploads python dict to S3 as .json file.
        Args:
            s3_bucket: S3 bucket where the file will be stored
            path: path to the folder in the bucket
            file_name: file name as stored in S3
            json_dict: dict to be stores as .json file in S3
        Returns:
            key: key to the file in the bucket pointing to storing location
        """

    if not file_name:
        # Get temporary file name
        temporary_file = tempfile.NamedTemporaryFile(suffix='.json')
        temporary_file.close()
        file_name = temporary_file.name

    # Save to local
    with open(file_name, 'w') as outfile:
        json.dump(
            json_dict,
            outfile,
            ensure_ascii=False,
            indent=4
        )

    s3_store_file(
        local_file_path=file_name,
        s3_bucket=s3_bucket,
        key=os.path.join(path, file_name)
    )


def s3_store_model_pipeline(
    pipeline: Pipeline,
    s3_bucket: str,
    path: str,
    file_name: str
) -> str:

    """
    Store sklearn model pipeline in S3
    Args:
        pipeline: sklearn Pipeline object
        s3_bucket: S3 bucket where the file will be stored
        path: path to the folder in the bucket
        file_name: file name

    Returns:
        key: key pointing to S3 location where pipeline is stored
    """
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, file_name)
    joblib.dump(pipeline, temp_file)

    key = s3_store_file(
        local_file_path=temp_file,
        s3_bucket=s3_bucket,
        key=os.path.join(path, file_name)
    )

    try:
        os.remove(temp_file)
    except Exception:
        pass

    return key
