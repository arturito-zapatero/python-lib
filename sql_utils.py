import os
import tempfile

import lib.base_utils as bs_uts


# Function that get query from S3 (as string)
def get_sql_query_from_s3(
    sql_file: str,
    **kwargs
) -> [str, None]:

    """
    Gets sql file from S3 (key: os.environ['main_project'] + '/' + os.environ['environment'] + '/transformations/sql/'
     + sql_file). Replace the sql variables (in {} brackets) with kwargs.
    Args:
        sql_file: sql file name
        **kwargs: vars to replace  sql variables

    Returns:
        sql_query: sql query with sql variables replaced with kwargs
    """

    # Check environment variables present
    required_envvars = ['main_project', 'environment', 'code_bucket']
    if not all(envvar in os.environ for envvar in required_envvars):
        missing_envvars = [var for var in required_envvars if var not in os.environ]
        raise Exception(f"Following environment vars {', '.join(missing_envvars)} are missing")

    # Open S3 client
    s3_client = bs_uts.connect_boto('s3')

    # Download the file, if present
    temporary_file = tempfile.NamedTemporaryFile()
    temporary_file.close()
    path = os.environ['main_project'] + '/' + os.environ['environment'] + '/transformations/sql/'
    fkey = os.path.join(path, sql_file)
    if bs_uts.find_latest_file_in_s3_path(
        s3_bucket=os.environ['code_bucket'],
        path=path
    ) is None:
        return None

    s3_client.download_file(
        Bucket=os.environ['code_bucket'],
        Filename=temporary_file.name,
        Key=fkey
    )

    # Read file
    with open(temporary_file.name, 'r') as f:
        sql_query = f.read()

    # Replace parameters in query
    for key, value in kwargs.items():
        sql_query = sql_query.replace('{' + key + '}', value)

    return sql_query
