from decimal import Decimal
import json
import boto3
from boto3.dynamodb.conditions import Key
import os


def put_one_item_in_dynamodb(
    table_name: str,
    col_partition_key: str,
    partition_key: str,
    col_sort_key: str,
    sort_key: str,
    col_value: str,
    value: [dict, list, str, int]
):
    """
    Puts an item in DynamoDB table.

    Args:
        table_name: dynamoDB table
        col_partition_key: name of partition key column name
        partition_key: partition key
        col_sort_key: sort key column name
        sort_key: sort key
        col_value: name of value column name
        value: value
    Returns:
        None
    """

    if 'region' in os.environ:
        region_name = os.environ['region']
    else:
        region_name = ''

    dynamodb_client = boto3.resource(
        'dynamodb',
        region_name=region_name
    )

    table = dynamodb_client.Table(
        table_name
    )

    # Change float to decimal
    value = json.loads(json.dumps(value), parse_float=Decimal)

    if col_sort_key:
        db_item = {
                col_partition_key: partition_key,
                col_sort_key: sort_key,
                col_value: value
            }
    else:
        db_item = {
            col_partition_key: partition_key,
            col_value: value
        }

    response = table.put_item(
        Item=db_item
    )

    return response


def scan_table(
    dynamo_client: str,
    *,
    table_name: str,
    **kwargs
):
    """
    Generates all the items in a DynamoDB table.

    dynamo_client: A boto3 client for DynamoDB.
    table_name: The name of the table to scan.

    Other keyword arguments will be passed directly to the Scan operation.
    See https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/dynamodb.html#DynamoDB.Client.scan

    Function from:
    https://alexwlchan.net/2020/05/getting-every-item-from-a-dynamodb-table-with-python/

    """
    paginator = dynamo_client.get_paginator("scan")

    for page in paginator.paginate(TableName=table_name, **kwargs):
        yield from page["Items"]


def get_dynamodb_table(
    table_name: str,
    col_partition_key: str,
    col_value: str
) -> dict:
    """
    Gets the whole table from DynamoDB.

    TODO: include sort key
    Args:
        table_name: table name
        col_partition_key: partition key column name
        col_value: values column name

    Returns:
        dictionary: DynamoDB table as dictionary
    """

    if 'region' in os.environ:
        region_name = os.environ['region']
    else:
        region_name = ''

    dynamo_client = boto3.client(
        'dynamodb',
        region_name=region_name
    )

    dictionary = {}
    for item in scan_table(dynamo_client, table_name=table_name):
        dictionary[[*item[col_partition_key].values()][0]] = [*item[col_value].values()][0]

    dictionary = replace_decimals(dictionary)

    return dictionary


def get_dynamodb_table_with_query(
    table_name: str,
    col_partition_key: [str, bool],
    partition_key_filter: [str, bool],
    col_sort_key: [str, bool],
    sort_key_filter: [str, bool],
    col_value: str,
) -> [dict, list, tuple, set, str, int, float]:

    """
    Gets DynamoDB table using query filters by partition and sort keys.
    Args:
        table_name: dynamoDB table
        col_partition_key: name of partition key column name
        partition_key_filter: partition key filter
        col_sort_key: sort key column name
        sort_key_filter: sort key filter
        col_value: name of value column name

    Returns:
        dictionary: DynamoDB table as dictionary
    """

    if 'region' in os.environ:
        region_name = os.environ['region']
    else:
        region_name = ''

    dynamodb_client1 = boto3.client(
        'dynamodb',
        region_name=region_name
    )

    dynamodb_client = boto3.resource(
        'dynamodb',
        region_name=region_name
    )
    try:
        table = dynamodb_client.Table(table_name)

        if col_sort_key:
            queried_items = table.query(
                Select="ALL_ATTRIBUTES",
                KeyConditionExpression=Key(col_partition_key).eq(partition_key_filter) &
                                       Key(col_sort_key).eq(sort_key_filter))["Items"]

            if queried_items:
                dictionary = queried_items[0][col_value]
            else:
                print(f"No element found for {col_partition_key} = {partition_key_filter} and {col_sort_key} = {sort_key_filter}")
                dictionary = None
        else:
            queried_items = table.query(
                Select="ALL_ATTRIBUTES",
                KeyConditionExpression=Key(col_partition_key).eq(partition_key_filter))["Items"]
            if queried_items:
                dictionary = queried_items[0][col_value]
            else:
                print(f"No element found for {col_partition_key} = {partition_key_filter}")
                dictionary = None
    except dynamodb_client1.exceptions.ResourceNotFoundException as e:
        msg = f"{e}. DynamoDB table/resource: {table} not found"
        raise Exception(msg)

    if dictionary:
        dictionary = replace_decimals(dictionary)

    return dictionary


def put_dictionary_in_dynamodb(
    table_name: str,
    dictionary: dict,
    col_partition_key: str,
    col_sort_key: str,
    sort_key: str,
    col_value: str
):
    """
    Puts a dictionary in DynamoDB table. Puts every key value pair as separate table item (key in col_partition_key
     columns, value in col_value column).

    Args:
        table_name: dynamoDB table
        dictionary: dictionary to be put in DynamoDB
        col_partition_key: partition key column name
        col_sort_key: sort key column name
        sort_key: sort key
        col_value: name of value column name
        col_value: name of value column

    Returns:
        None
    """
    if 'region' in os.environ:
        region_name = os.environ['region']
    else:
        region_name = ''

    dynamodb_client = boto3.resource(
        'dynamodb',
        region_name=region_name
    )

    table = dynamodb_client.Table(
        table_name
    )

    # Change float to decimal
    dictionary = json.loads(json.dumps(dictionary), parse_float=Decimal)

    for key, value in dictionary.items():
        if col_sort_key:
            db_item = {
                    col_partition_key: key,
                    col_sort_key: sort_key,
                    col_value: value
                }
        else:
            db_item = {
                    col_partition_key: key,
                    col_value: value
            }

        response = table.put_item(
            Item=db_item
        )


def replace_decimals(
    obj: dict
) -> [dict, list, dict]:

    """
    Replaces decimal.Decimal types with either python int or float object in the whole input dictionary. Works also
    for deeply nested dictionaries. Main purpose is to be used with AWS DynamoDB which cannot store floats, and saves
    them as Decimals, and while reading from Dynamo DB both floats and integers are read as Decimals
    Args:
        obj: input dictionary with Decimal values

    Returns:
        obj: output dictionary with Decimal values replaced by integers or floats
    """

    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = replace_decimals(obj[i])
        return obj
    elif isinstance(obj, dict):
        for k in obj:
            obj[k] = replace_decimals(obj[k])
        return obj
    elif isinstance(obj, Decimal):
        if obj % 1 == 0:
            return int(obj)
        else:
            return float(obj)
    else:
        return obj
