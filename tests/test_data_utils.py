import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

import create_dataframes_blueprints as blp

from lib.data_utils import add_distance_between_two_points_col, create_cartesian_product, ohe, encode_cyclical_columns,\
    encode_cyclical_vector, min_max_normalization

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

col_default_string = 'col_string'
col_default_int = 'col_int'
col_default_float = 'col_float'
col_default_date = 'col_date'
col_default_another_1 = 'col_another_1'
col_default_another_2 = 'col_another_1'
col_months = "months"
col_dow  = 'dow'


def test_add_distance_between_two_points_col():

    # SETUP
    number_of_events = 10

    input_data, expected_distance_column = blp.create_test_dataframe(
        case=5,
        n_rows=number_of_events
    )
    expected_columns = set(cols_coords_data + cols_coords_clusters + [col_distance])

    # EXECUTE
    output_data = add_distance_between_two_points_col(
        data=input_data,
        cols_coords_1=cols_coords_data,
        cols_coords_2=cols_coords_clusters,
        col_distance=col_distance,
        unit='m'
    )
    output_data = output_data.round({col_distance: 3})
    distance_positive = (output_data[col_distance] >= 0)


    # VERIFY
    assert list(output_data[col_distance]) == expected_distance_column
    assert expected_columns.issubset(set(output_data.columns))
    assert all(distance_positive)
    

def test_create_cartesian_product():

    # SETUP
    input_data, _ = blp.create_test_dataframe(
        case=8,
        n_rows=None
    )

    # EXERCISE
    _, cart_prod_result = create_cartesian_product(input_data,
                                                   [col_default_string, col_default_int,
                                                    col_default_float, col_default_date],
                                                   [None, None, None, None],
                                                   [None, None, None, None],
                                                   [None, None, 0.2, None])

    # VERIFY
    assert cart_prod_result.shape[0] == 3 * 3 * 3 * 5
    assert cart_prod_result[col_default_another_1].count() == 5
    assert cart_prod_result[col_default_another_2].count() == 5
    assert len(cart_prod_result.columns) == len(input_data.columns)
    assert all(coli in cart_prod_result.columns for coli in input_data.columns)


def test_create_cartesian_product_min_max():

    # SETUP
    input_data, _ = blp.create_test_dataframe(
        case=8,
        n_rows=None
    )

    # EXERCISE
    _, cart_prod_result = create_cartesian_product(input_data,
                                                     [col_default_string, col_default_int,
                                                      col_default_float, col_default_date],
                                                     [None, 1, 0.4, None],
                                                     [None, 5, 1, None],
                                                     [None, 2, 0.1, None])

    # VERIFY
    assert cart_prod_result.shape[0] == 3 * 3 * 7 * 5
    assert cart_prod_result[col_default_another_1].count() == 5
    assert cart_prod_result[col_default_another_2].count() == 5
    assert len(cart_prod_result.columns) == len(input_data.columns)
    assert all(coli in cart_prod_result.columns for coli in input_data.columns)


def test_create_cartesian_product_date_min_max():

    # SETUP
    input_data, _ = blp.create_test_dataframe(
        case=8,
        n_rows=None
    )

    # EXERCISE
    # Since we want only the cartesian product in given limits, '2020/01/01' will be excluded from data
    _, cart_prod_result = create_cartesian_product(input_data,
                                                   [col_default_string, col_default_int,
                                                    col_default_float, col_default_date],
                                                   [None, None, None,
                                                    pd.to_datetime('2020/01/02', format='%Y/%m/%d')],
                                                   [None, None, None,
                                                    pd.to_datetime('2020/01/08', format='%Y/%m/%d')],
                                                   [None, None, 0.2, None])

    # VERIFY
    assert cart_prod_result.shape[0] == 3 * 3 * 3 * 7
    assert cart_prod_result[col_default_another_1].count() == 4
    assert cart_prod_result[col_default_another_2].count() == 4
    assert len(cart_prod_result.columns) == len(input_data.columns)
    assert all(coli in cart_prod_result.columns for coli in input_data.columns)


def test_create_cartesian_product_date_wrong_list_length():

    # SETUP
    input_data, _ = blp.create_test_dataframe(
        case=8,
        n_rows=None
    )

    # VERIFY
    with pytest.raises(Exception, match=r".*All lists in function input must contain the same number of values.*"):
        create_cartesian_product(input_data,
                                 [col_default_string, col_default_int,
                                 col_default_float, col_default_date],
                                 [None, None, None, pd.to_datetime('2020/01/02', format='%Y/%m/%d')],
                                 [None, None, None, pd.to_datetime('2020/01/08', format='%Y/%m/%d')],
                                 [None, None, 0.2, None, 0])[1]


def test_create_cartesian_product_date_wrong_columns():

    # SETUP
    input_data, _ = blp.create_test_dataframe(
        case=8,
        n_rows=None
    )

    # VERIFY
    with pytest.raises(Exception, match=r".*Column.* missing in input column list.*"):
        create_cartesian_product(input_data,
                                 [col_default_string, f"wrong_name_{col_default_int}",
                                  col_default_float, col_default_date],
                                 [None, None, None, pd.to_datetime('2020/01/02', format='%Y/%m/%d')],
                                 [None, None, None, pd.to_datetime('2020/01/08', format='%Y/%m/%d')],
                                 [None, None, 0.2, None])[1]


def test_create_cartesian_product_date_wrong_str_arg():

    # SETUP
    input_data, _ = blp.create_test_dataframe(
        case=8,
        n_rows=None
    )

    # VERIFY
    with pytest.raises(Exception, match=r".*Min/Max arguments not used for column type String.*"):
        create_cartesian_product(input_data,
                                 [col_default_string, col_default_int,
                                  col_default_float, col_default_date],
                                 ['val1', None, None, pd.to_datetime('2020/01/02', format='%Y/%m/%d')],
                                 [None, None, None, pd.to_datetime('2020/01/08', format='%Y/%m/%d')],
                                 [None, None, 0.2, None])[1]


def test_create_cartesian_product_date_min_max_check():

    # SETUP
    input_data, _ = blp.create_test_dataframe(
        case=8,
        n_rows=None
    )

    # VERIFY
    with pytest.raises(Exception, match=r".*Min and Max values provided for .* invalid.*"):
        create_cartesian_product(input_data,
                                 [col_default_string, col_default_int,
                                  col_default_float, col_default_date],
                                 [None, 1, 1.5, None],
                                 [None, 5, 1, None],
                                 [None, 2, 0.1, None])[1]


def test_create_cartesian_product_fillna():

    # SETUP
    input_data, _ = blp.create_test_dataframe(
        case=8,
        n_rows=None
    )

    # EXERCISE
    _, cart_prod_result = create_cartesian_product(input_data,
                                                   [col_default_string, col_default_int,
                                                    col_default_float, col_default_date],
                                                   [None, None, None, None],
                                                   [None, None, None, None],
                                                   [None, None, 0.2, None],
                                                   fillna=0)

    # VERIFY
    assert cart_prod_result.shape[0] == 3 * 3 * 3 * 5
    assert cart_prod_result[col_default_another_1].count() == cart_prod_result.shape[0]
    assert cart_prod_result[col_default_another_2].count() == cart_prod_result.shape[0]
    assert len(cart_prod_result.columns) == len(input_data.columns)
    assert all(coli in cart_prod_result.columns for coli in input_data.columns)


def test_create_cartesian_product_fillna_specific_col():

    # SETUP
    input_data, _ = blp.create_test_dataframe(
        case=8,
        n_rows=None
    )

    # EXERCISE
    _, cart_prod_result = create_cartesian_product(input_data,
                                                   [col_default_string, col_default_int,
                                                    col_default_float, col_default_date],
                                                   [None, None, None, None],
                                                   [None, None, None, None],
                                                   [None, None, 0.2, None],
                                                   fillna={col_default_another_1: 'def'})

    # VERIFY
    assert cart_prod_result.shape[0] == 3 * 3 * 3 * 5
    assert cart_prod_result[col_default_another_1].count() == cart_prod_result.shape[0]
    assert cart_prod_result[col_default_another_2].count() == cart_prod_result.shape[0]
    assert len(cart_prod_result.columns) == len(input_data.columns)
    assert all(coli in cart_prod_result.columns for coli in input_data.columns)


def test_ohe():

    # SETUP
    input_data, expected_df = blp.create_test_dataframe(
        case=8,
        n_rows=None
    )

    categorical_columns = [col_default_string]
    numerical_columns = [col_default_int, col_default_date]
    id_columns = [col_default_another_1]
    target_column = col_default_float
    expected_cols = numerical_columns + id_columns + [target_column] + [f"{col_default_string}_bb",
                                                                        f"{col_default_string}_c"]
    expected_feature_cols = numerical_columns + [f"{col_default_string}_bb", f"{col_default_string}_c"]

    # EXERCISE
    output_data_ohe, pip, cols = ohe(
        input_data,
        None,
        categorical_columns,
        numerical_columns,
        id_columns,
        target_column
    )

    # VERIFY
    assert isinstance(output_data_ohe, pd.DataFrame)
    assert output_data_ohe.shape[0] == input_data.shape[0]
    assert set(expected_cols).issubset(output_data_ohe)
    assert output_data_ohe.equals(expected_df)
    assert isinstance(pip, Pipeline)
    assert len(expected_feature_cols) == len(cols) and set(expected_feature_cols).issubset(set(cols))


def test_ohe_missing_col():

    # SETUP
    categorical_columns = [col_default_string, "some_missing_col"]
    numerical_columns = [col_default_int, col_default_date]
    id_columns = [col_default_another_1]
    target_column = col_default_float

    # SETUP
    input_data, _ = blp.create_test_dataframe(
        case=8,
        n_rows=None
    )
    # VERIFY
    with pytest.raises(Exception, match=r".*Not all columns in input list 'categorical_columns' are present in 'input_data'.*"):
        ohe(
            input_data,
            None,
            categorical_columns,
            numerical_columns,
            id_columns,
            target_column
        )


def test_min_max_normalization_shifted():

    # SETUP
    input_vector = pd.Series([1, 2, 3, 4, 5])
    expected_output_vector = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0])

    # EXERCISE
    output_vector = min_max_normalization(input_vector)

    # VERIFY
    assert all(expected_output_vector == output_vector)


def test_min_max_normalization_min_value_max_value():

    # SETUP
    minimum = 1
    maximum = 5
    input_vector = pd.Series([4, 2, 3, 4, 5])
    expected_output_vector = pd.Series([0.75, 0.25, 0.5, 0.75, 1.0])

    # EXERCISE
    output_vector = min_max_normalization(
        input_vector,
        min_value=minimum,
        max_value=maximum
    )

    # VERIFY
    assert all(expected_output_vector == output_vector)


def test_encode_cyclical_vector():

    # SETUP
    input_data, expected_df = blp.create_test_dataframe(
        case=9,
        n_rows=None
    )
    # EXERCISE
    result_sin, result_cos = encode_cyclical_vector(input_data['vector'])
    output_data = pd.DataFrame({
        'vector': input_data['vector'],
        'sin': round(result_sin, 2),
        'cos': round(result_cos, 2)
    })

    # VERIFY
    assert all(output_data[['sin', 'cos']] == expected_df)



def test_encode_cyclical_columns_mono_column_with_dropping():

    # SETUP
    input_data, _ = blp.create_test_dataframe(
        case=10,
        n_rows=None
    )
    columns_to_encode = [col_months]

    # EXERCISE
    output_data, _ = encode_cyclical_columns(
        input_data,
        columns_to_encode,
        drop_orig_cols=True
    )
    output_columns = set(output_data.columns)
    true_output_columns = {f'{col_months}_sin', f'{col_months}_cos'}

    # VERIFY
    assert output_columns == true_output_columns


def test_encode_cyclical_columns_mono_column_without_dropping():
    # SETUP
    input_data, _ = blp.create_test_dataframe(
        case=10,
        n_rows=None
    )
    columns_to_encode = [col_months]

    # EXERCISE
    output_data, _ = encode_cyclical_columns(
        input_data,
        columns_to_encode,
        drop_orig_cols=False
    )
    output_columns = set(output_data.columns)
    true_output_columns = {col_months , f'{col_months}_sin', f'{col_months}_cos'}

    # VERIFY
    assert output_columns == true_output_columns


def test_encode_cyclical_columns_poli_columns_all_encoding_with_dropping():

    # SETUP
    input_data, _ = blp.create_test_dataframe(
        case=11,
        n_rows=None
    )
    columns_to_encode = [col_months, col_dow]

    # EXERCISE
    output_data, _ = encode_cyclical_columns(input_data, columns_to_encode, True)
    output_columns = set(output_data.columns)
    true_output_columns = {f'{col_months}_sin', f'{col_months}_cos',
                           f'{col_dow}_sin', f'{col_dow}_cos'}

    # VERIFY
    assert output_columns == true_output_columns


def test_encode_cyclical_columns_poli_columns_all_encoding_without_dropping():

    # SETUP
    input_data, _ = blp.create_test_dataframe(
        case=11,
        n_rows=None
    )
    columns_to_encode = [col_dow, col_months]

    # EXERCISE
    output_data, _ = encode_cyclical_columns(input_data, columns_to_encode, False)
    output_columns = set(output_data.columns)
    true_output_columns = {col_dow, col_months,
                           f'{col_months}_sin', f'{col_months}_cos',
                           f'{col_dow}_sin', f'{col_dow}_cos'}

    # VERIFY
    assert output_columns == true_output_columns


def test_encode_cyclical_columns_raises_error_when_bad_input_columns_list():

    # SETUP
    input_data, _ = blp.create_test_dataframe(
        case=11,
        n_rows=None
    )
    columns_to_encode = [col_dow, "BAD_BUGGIE_COLUMN_MAN"]

    # EXERCISE
    with pytest.raises(KeyError):
        encode_cyclical_columns(
            input_data,
            columns_to_encode,
            drop_orig_cols=False
        )


def test_encode_cyclical_columns_raises_error_when_bad_input_columns_string():
    # SETUP
    input_data, _ = blp.create_test_dataframe(
        case=11,
        n_rows=None
    )
    columns_to_encode = "dsade"

    # EXERCISE
    with pytest.raises(TypeError):
        encode_cyclical_columns(
            input_data,
            columns_to_encode,
            drop_orig_cols=False
        )
