import category_encoders as ce
import datetime as dt
from haversine import haversine_vector
import itertools
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from sklearn.pipeline import Pipeline

from lib.print_out_info import print_out_info


@print_out_info
def add_distance_between_two_points_col(
    data: pd.DataFrame,
    cols_coords_1: list,
    cols_coords_2: list,
    col_distance: str = 'distance',
    unit: str = 'm'
) -> pd.DataFrame:
    """
    Calculates distance between point specified by cols_coords_1 and cols_coords_2 in each row of a df, adds this
    distance as col_distance columns.
    Args:
        data: data
        cols_coords_1: lon/lat columns for 1st coordinates
        cols_coords_2: lon/lat columns for 2nd coordinates
        col_distance: name of the created distance column
        unit: distance unit

    Returns:
        data: with added col_distance column
    """

    data[col_distance] = haversine_vector(
        [tuple(x) for x in data[cols_coords_1].to_numpy()],
        [tuple(x) for x in data[cols_coords_2].to_numpy()],
        unit=unit
    )

    return data


def combine_date_and_hour_to_timestamp(
    data: pd.DataFrame,
    col_day: str,
    col_hour: str,
    col_ts: str
) -> pd.DataFrame:
    """
    Creates a new column of type datetime64[ns] by combining col_day and col_hour.
    Args:
        data: data with col_day and col_hour
        col_day: date column
        col_hour: hour columns
        col_ts: timestamp column that will be created
    Returns:
        data: data with col_ts added
    """
    if data.dtypes[col_day] == 'datetime64[ns]':
        data.loc[:, col_ts] = data.apply(
            lambda x: dt.datetime.combine(x[col_day],
                                          dt.datetime.strptime(str(int(x[col_hour])), '%H')
                                          .time()),
            axis=1)
    else:
        data.loc[:, col_ts] = data.apply(
            lambda x: dt.datetime.combine(dt.datetime.strptime(x[col_day], "%Y-%m-%d"),
                                          dt.datetime.strptime(str(int(x[col_hour])), '%H')
                                          .time()),
            axis=1)

    return data


@print_out_info
def create_cartesian_product(
    data: pd.DataFrame,
    cols_names_list: list,
    min_values_list: list,
    max_values_list: list,
    step_values_list: list,
    fillna: [dict, None] = None,
    remove_duplicates: bool = True
) -> [pd.DataFrame, pd.DataFrame]:
    """
    Creates the cartesian product of the columns defined, using the min/max/step values defined in the input
    Args:
        data: DataFrame for which to create the cartesian product, has to have cols_names_list columns
        cols_names_list: list of names of columns to use in the cartesian product
        min_values_list: list of min values to use for the cartesian product (only applicable to date/int/float).
            When not applicable the value should be None.
        max_values_list: list of max values to use for the cartesian product (only applicable to date/int/float).
            When not applicable the value should be None.
        step_values_list: list of step values to use for the cartesian product (only applicable to int/float/string).
            When not applicable the value should be None. Step for dates is 1 day. For string column it can be None
            (unique values of string column) are a list/set with values to create cartesian product.
        fillna: value with which to fill the NaN values. Same syntax as pandas.DataFrame.fillna
        remove_duplicates: if True, in full_df cols_names_list values will be unique (using min for all the other cols,
        mean will not work for string cols)

    TODO: Should be possible also w/o input data df, add keep column functionality

    Returns:
        cartesian_df: cartesian product of provided columns
        full_df: cartesian product of provided columns merged with the origin DataFrame
    """

    # Verify inputs
    if len({len(cols_names_list),
            len(min_values_list),
            len(max_values_list),
            len(step_values_list)}) > 1:
        raise Exception("All lists in function input must contain the same number of values")
    if not all(cols in data.columns for cols in cols_names_list):
        missing_cols = [cols for cols in cols_names_list if cols not in data.columns]
        raise Exception(f"Column(s) {', '.join(missing_cols)} missing in input column list")

    # Loop through the columns provided to create the list of values to combine
    val_list = list()
    for col, mn, mx, step in zip(cols_names_list, min_values_list, max_values_list, step_values_list):
        if data[col].dtype == object:
            if step:
                val_list.append(step)
            else:
                data[col] = data[col].astype('str')
                val_list.append(list(sorted(set(data[col].tolist()))))
            if mn is not None or mx is not None:
                raise Warning("Min/Max arguments not used for column type String")
            if step is not None and not isinstance(step, list) and not isinstance(step, set):
                raise Warning("Step arguments must be either list or None for type String")
        elif is_datetime(data[col]):
            mn = mn if mn is not None else data[col].min()
            mx = mx if mx is not None else data[col].max()
            if mx < mn:
                raise Exception(f"Min and Max values provided for {col} invalid")
            if step is not None:
                raise Exception(f"Step parameter cannot be provided for datetime columns ({col})")
            val_list.append(list(pd.date_range(mn, mx)))
        elif data[col].dtype == np.float64:
            mn = mn if mn is not None else data[col].min()
            mx = mx if mx is not None else data[col].max()
            if mx < mn:
                raise Exception(f"Min and Max values provided for {col} invalid")
            if step is None or step == 0:
                raise Exception(f"Invalid step provided for column {col}")
            val_list.append(list(np.round(np.arange(mn, mx + step / 2, step), 3)))
        elif data[col].dtype == np.int64:
            mn = mn if mn is not None else data[col].min()
            mx = mx if mx is not None else data[col].max()
            if mx < mn:
                raise Exception(f"Min and Max values provided for {col} invalid")
            step = step if step is not None else 1
            val_list.append(list(range(mn, mx + step, step)))
        else:
            raise Exception(f"Column type {data[col].dtype} not implemented in create_cartesian_product")

    # Create cartesian product and merge with provided dataset
    cartesian_data = pd.DataFrame.from_records(itertools.product(*val_list),
                                               columns=cols_names_list)
    full_data = cartesian_data.merge(data,
                                     how='left',
                                     on=cols_names_list)

    # To get rid of duplicates
    if remove_duplicates:
        full_data = (full_data
                     .groupby(cols_names_list)
                     .min()
                     .reset_index()
                     )

    # Fill NaN values with provided input
    if fillna is not None:
        full_data = full_data.fillna(value=fillna)

    return cartesian_data, full_data


def ohe(
    input_data: pd.DataFrame,
    pipeline_ohe: [Pipeline, None],
    cols_categorical: list,
    cols_numerical: list,
    id_columns: list,
    target_column: str
) -> [pd.DataFrame, Pipeline, list]:

    """
    Applies one-hot encoding pipeline to an input dataframe, adds OHE columns to data and creates list
    (cols_features_full) with categorical columns names added to existing column names (numerical_columns). Returns
    also created OHE pipeline.
    Args:
        input_data: input data with
        pipeline_ohe: OHE pipeline (if available, else it will be created)
        cols_categorical: list of categorical columns to apply one-hot encoding
        cols_numerical: list of numerical columns
        id_columns: list of id columns
        target_column: target column

    Returns:
        data: data with one-hot-encoded data columns added
        pipeline_ohe: OHE pipeline
        cols_features_full: feature columns, including numerical and categorical ohe
    TODO:
     - create a function for categorical features with lots of unique values (cardinal encoding)
    """

    if pipeline_ohe is not None and not isinstance(pipeline_ohe, Pipeline):
        raise Exception("Argument 'pipeline_ohe' is not of type Pipeline")
    if len(set(cols_categorical).difference(set(input_data.columns))) > 0:
        raise Exception(f"Not all columns in input list 'categorical_columns' are present in 'input_data', column(s) "
                        f"missing: {(set(cols_categorical).difference(set(input_data.columns)))}")
    if len(set(cols_numerical).difference(set(input_data.columns))) > 0:
        raise Exception(f"Not all columns in input list 'numerical_columns' are present in 'input_data', column(s) "
                        f"missing: {(set(cols_numerical).difference(set(input_data.columns)))}")
    if len(set(id_columns).difference(set(input_data.columns))) > 0:
        raise Exception(f"Not all columns in input list 'id_columns' are present in 'input_data', column(s) "
                        f"missing: {(set(id_columns).difference(set(input_data.columns)))}")
    if target_column not in input_data.columns:
        raise Exception(f"Target column {target_column} not present in 'input_data'")

    # Drop NaNs in categorical columns, otherwise we get col_NaN column
    input_data = input_data.dropna(subset=cols_categorical)
    if not pipeline_ohe:
        pipeline_ohe = Pipeline([('ohe_encoder',
                                  ce.OneHotEncoder(handle_unknown='value',
                                                   use_cat_names=True,
                                                   cols=cols_categorical))])
        pipeline_ohe = pipeline_ohe.fit(input_data)

    data_ohe = pipeline_ohe.transform(input_data)
    cols_ohe_out = list(set(list(data_ohe.columns)) - set(list(input_data.columns)))
    cols_ohe_out.sort()

    # To avoid multicolinearity, remove first for each
    for coln in cols_categorical:
        starts_list = list(sorted(set([col for col in cols_ohe_out if col.startswith(coln)])))
        cols_ohe_out.remove(starts_list[0])

    # Produce outputs
    cols_features_full = cols_numerical + cols_ohe_out
    data = data_ohe.loc[:, cols_features_full + id_columns + [target_column]]

    return data, pipeline_ohe, cols_features_full


@print_out_info
def encode_cyclical_columns(
    data: pd.DataFrame,
    cols_cyclical: list,
    drop_orig_cols: bool = True
) -> [pd.DataFrame, list]:

    """
    Function that encodes cyclical features to get rid off jump discontinuities in data. Vector is encoded into a circle
     using sin and cos functions.

    Args:
        data : data which includes cols_cyclical we want to encode
        cols_cyclical: columns to encode
        drop_orig_cols: whether input columns should be dropped
    Returns:
        data: df with added cyclical columns
        cols_cyclical_out: list with added columns names
    """

    if not isinstance(cols_cyclical, list):
        raise TypeError(f"{cols_cyclical} is not a list! Provide *a list* of columns names you want to encode")

    if not set(cols_cyclical).issubset(data.columns):
        raise KeyError(f"Not every column name (of columns to encode) is present in the provided dataframe.")

    for col in cols_cyclical:
        if len(data[col].unique()) <= 1:
            print(f"Column {col} has only one distinct value and will not be taken into consideration as cyclical vars"
                  f"at this time")
            cols_cyclical.remove(col)
        else:
            vector_to_encode = data[col]
            sin_vector, cos_vector = encode_cyclical_vector(vector_to_encode)
            data[f"{col}_sin"] = sin_vector
            data[f"{col}_cos"] = cos_vector
        if drop_orig_cols:
            data = data.drop(columns=[col])

    # Deal with cyclical variables
    cols_variables_sin = [s + '_sin' for s in cols_cyclical]
    cols_variables_cos = [s + '_cos' for s in cols_cyclical]
    cols_cyclical_out = cols_variables_sin + cols_variables_cos

    return data, cols_cyclical_out


def encode_cyclical_vector(
    input_vector: pd.Series,
    start_value: [int, float, None] = None,
    stop_value: [int, float, None] = None
) -> [pd.Series, pd.Series]:

    """
    Function that encodes provided vector (series) into cyclical feature.Uses radian normalisation, then calculates
     sin and cos of the values.

    Args:
        input_vector: vector we want to encode
        start_value: from what value we start counting our feature (eg. in case of encoding months it will be 1 or 0
        for January)
        stop_value: what's the last possible value of our cyclical feature (in the months case it will be
         11 or 12 for December - depends if we count from 0 or 1)
    Returns:
        Tuple of sinusoidal part and cosinusoidal part of the feature.
    """

    normalised_vector = min_max_normalization(
        input_vector,
        min_value=start_value,
        max_value=stop_value
    )
    radian_vector = 2 * np.pi * normalised_vector
    sin_vector = np.sin(radian_vector)
    cos_vector = np.cos(radian_vector)

    return sin_vector, cos_vector


def min_max_normalization(
    input_vector: pd.Series,
    min_value: [int, float, None] = None,
    max_value: [int, float, None] = None
) -> pd.Series:

    """
    Function that normalises the provided input vector of floats into one with min = 0 and max = 1.
    f(R) -> [0,1]

    Args:
        input_vector: vector to normalise
        min_value: minimal value that a series can achieve (if not present in the vector itself)
        max_value: maximal value that a series can achieve (if not present in the vector itself)

    Returns:
        normalised vector
    """

    if not min_value:
        min_value = input_vector.min()
    if not max_value:
        max_value = input_vector.max()
    return (input_vector - min_value) / (max_value - min_value)


# Custom round function using input precision, python round will not round integers, e.g. 13 to 10
def round_to_number(x, base=0.01):
    if base == 0:
        raise Exception("Base cannot be equal 0")
    return base * round(x/base)