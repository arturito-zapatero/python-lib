from aws_requests_auth.aws_auth import AWSRequestsAuth
import json
import os
import requests

import lib.base_utils as bs_uts
import lib.dynamo_utils as dyn_uts
from lib.print_out_info import print_out_info


@print_out_info
def get_config_via_api_gw(
    city_name: str,
    client: str,
    project: str
) -> dict:

    """
    Function that connects to API Gateway and extracts the config from DynamoDB (via API Gateway and lambda function)
    Args:
        project: project name
        client: client for which the data are prepared
        city_name: city_name for which the data are prepared

    Returns:
        config: config for current project as dict
    """

    # Setup AWS access credentials
    os.environ['AWS_ACCESS_KEY_ID'] = bs_uts.get_ssm_secret('/main/aws_access_key_id')['Parameter']['Value']
    os.environ['AWS_SECRET_ACCESS_KEY'] = bs_uts.get_ssm_secret('/main/aws_secret_access_key')['Parameter']['Value']
    os.environ['AWS_SESSION_TOKEN'] = ''
    os.environ['region'] = bs_uts.get_ssm_secret('/main/region')['Parameter']['Value']

    # Setup API parameters
    config_api_id = (bs_uts.get_ssm_secret(f'/machine_learning/{os.environ["environment"]}/config_api_id')
                     ['Parameter']['Value'])
    config_api_method_path = os.environ['main_project']
    config_api_resource_path = os.environ['environment']

    # Get AWS authentication
    auth = AWSRequestsAuth(
        aws_access_key=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        aws_token=os.environ['AWS_SESSION_TOKEN'],
        aws_host=config_api_id + '.execute-api.' + os.environ['region'] + '.amazonaws.com',
        aws_region=os.environ['region'],
        aws_service='execute-api'
    )

    # Create API GW URL
    url = f"https://{config_api_id}.execute-api.{os.environ['region']}.amazonaws.com" \
          f"/{config_api_resource_path}" \
          f"/{config_api_method_path}"

    city_types = dyn_uts.get_dynamodb_table_with_query(
        table_name = f"ml_{client}_city_list_{os.environ['environment']}",
        col_partition_key="city_name",
        partition_key_filter=city_name,
        col_sort_key=False,
        sort_key_filter=False,
        col_value="city_type"
    )

    city_params = {
        'city_name': city_name,
        'client': client,
        'project': project,
        'city_types': city_types
    }

    # Request config via API
    response_config = requests.get(
        url=url,
        auth=auth,
        params=city_params
    )

    # Transform config
    config = response_config.content
    config = json.loads(config.decode("utf-8"))

    return config
