from functools import partial
import pandas as pd
import pyproj
import shapely.geometry
import shapely as shp
import shapely.ops as ops
from shapely.geometry import Point, Polygon
from shapely.geos import PredicateError

from lib.print_out_info import print_out_info


@print_out_info
def add_points_col(
    data: pd.DataFrame,
    col_points: str,
    col_lon: str,
    col_lat: str
) -> pd.DataFrame:

    """
    Adds a new column (col_points) to dataframe with shapely Point objects based on col_lon and col_lat columns
    Args:
        data: with lon and lat columns
        col_points: name for new column containing shapely points
        col_lon: name of longitude col
        col_lat: name of latitude col

    Returns:
        data: with col_points added
    """

    data[col_points] = data.apply(lambda x: Point(x[col_lon], x[col_lat]), axis=1)

    return data


@print_out_info
def assign_data_to_polygons(
    polygon_data: pd.DataFrame,
    data_to_assign: pd.DataFrame,
    col_polygon_id: str,
    col_polygon: str,
    col_lon: str,
    col_lat: str
) -> pd.DataFrame:

    """
    Assigns data to shapely polygons. For each row in data_to_assign will assign col_polygon_id based on polygon
    information stored in polygon_data. If given row does not belongs to any of the polygons, "no_region_assigned"
    value is assigned

    Note: Assumption is that the polygons are exclusive.

    Args:
        polygon_data: df with col_polygon and col_polygon_id
        data_to_assign: data to assign with col_lon and col_lat
        col_polygon_id: polygon id in polygon_data, the same will be assigned in data_to_assign
        col_polygon: polygon definition as shapely Polygon
        col_lon: column with longitude information
        col_lat: column with latitude information

    Returns:
        data_to_assign: data with col_polygon_id columnd added

    TODO: add warning if polygons are not exclusive
    """

    data_to_assign[col_polygon_id] = "no_region_assigned"
    for ind, region in polygon_data.iterrows():
        data_in_polygon, _ = get_rows_inside_polygon(
            data=data_to_assign.copy(),
            col_lon=col_lon,
            col_lat=col_lat,
            polygon=region[col_polygon]
        )

        data_in_lat_filter = data_to_assign[col_lat].isin(data_in_polygon[col_lat].unique())
        data_in_lon_filter = data_to_assign[col_lon].isin(data_in_polygon[col_lon].unique())
        data_in_region_filter = data_in_lat_filter & data_in_lon_filter

        data_to_assign.loc[data_in_region_filter, col_polygon_id] = region[col_polygon_id]

    return data_to_assign


@print_out_info
def count_items_in_polygons(
    polygon_data: pd.DataFrame,
    data_to_count: pd.DataFrame,
    col_polygon: str,
    col_lon: str,
    col_lat: str,
    col_count_polygons: str,
    col_count_data: str
) -> pd.DataFrame:

    """
    Assigns data from data_to_count to polygons from polygon_data, counts col_count_data inside each polygon and adds
    to this polygon as col_count_polygons.
    Args:
        polygon_data: df with col_polygon
        data_to_count: data with col_count_data
        col_polygon: polygon definition as shapely Polygon
        col_lon: column with longitude information
        col_lat: column with latitude information
        col_count_polygons: column containing counted data added to polygon_data
        col_count_data: column containing data to count

    Returns:
        polygon_data: data with col_count_polygons added

    """

    for ind, region in polygon_data.iterrows():
        data_in_polygon, _ = get_rows_inside_polygon(
            data=data_to_count.copy(),
            col_lon=col_lon,
            col_lat=col_lat,
            polygon=region[col_polygon]
        )

        polygon_data.loc[ind, col_count_polygons] = data_in_polygon[col_count_data].sum()

    return polygon_data


def calculate_area_in_polygons(
    polygons: list,
    unit: str = 'km2'
) -> float:

    """
    Calculates the sum of areas of polygons contained in a list.
    Args:
        polygons: list with polygons
        unit: in which unit is the calculated area
    Returns:
        area: total area of all the polygons in the list (default in km^2, to get in m2 set unit='m2')

    NOTE: depending on version of functools, lat_1 and lat_2 maybe changed to lat1 and lat2, error
    associated with this is not explanatory
    """

    # Calculate OA area in meters
    area = 0
    for polygon in polygons:
        oa_transformed = ops.transform(
            partial(
                pyproj.transform,
                pyproj.Proj('EPSG:4326'),
                pyproj.Proj(
                    proj='aea',
                    lat_1=polygon.bounds[1],
                    lat_2=polygon.bounds[3])),
            polygon)
        area += oa_transformed.area

    if unit=='m2':
        area = area
    elif unit=='km2':
        area = area / 1000000

    return area


@print_out_info
def subtract_polygons_from_polygons(
        polygons_to_be_subtracted_from: list,
        polygons_to_subtract: list
) -> list:

    """
    Removes all the intersection between polygons in polygons_to_be_substracted_from and polygons_to_subtract.
    Returns
    Args:
        polygons_to_be_subtracted_from: list with shapely polygons to be subtracted from
        polygons_to_subtract: list with shapely polygons to subtract

    Returns:
        clean_forbidden_polygons: cleaned list with shapely polygons containing forbidden areas
    """

    polygons_after_subtraction = []
    for polygon in polygons_to_be_subtracted_from:
        polygon = subtract_polygons_from_polygon(
            polygon,
            polygons_to_subtract=polygons_to_subtract
        )
        polygons_after_subtraction.append(polygon)

    return polygons_after_subtraction


def subtract_polygons_from_polygon(
    polygon: [shp.geometry.polygon.Polygon, shp.geometry.multipolygon.MultiPolygon],
    polygons_to_subtract: list
) -> [shp.geometry.polygon.Polygon, shp.geometry.multipolygon.MultiPolygon]:

    """
    Subtracts from polygon any of the polygons intersecting with it contained in the list_of_polygons_to_subtact
    Args:
        polygon: to subtract from
        polygons_to_subtract: list of polygon to subtract

    Returns:
        polygon: with intersecting polygons subtracted

    """

    for polygon_to_subtract in polygons_to_subtract:
        if polygon.intersects(polygon_to_subtract):
            polygon = (polygon.symmetric_difference(polygon_to_subtract)).difference(polygon_to_subtract)
            try:
                polygon.intersects(polygon_to_subtract)
            except PredicateError:
                print(f"""Intersection of forbidden polygon {str(polygon_to_subtract)} and OA multipolygon 
                     {str(polygon)}  is invalid, invalid polygon will not be removed from OA, the solution is 
                     to manually find the bad polygon in .kml, move it a bit, save the new map and ask the CM to
                     replace the map""")

    return polygon


def get_rows_inside_polygon(
        data: pd.DataFrame,
        col_lon: str,
        col_lat: str,
        polygon: shapely.geometry.Polygon
) -> [pd.DataFrame, pd.DataFrame]:

    """
    Removes rows of df that are located outside given polygon based on lat/lon coordinates. Returns two dfs, one with
    rows inside and one with rows outside polygon.
    Args:
        data: pandas df with data, including lat/lon columns
        col_lon: lon column name
        col_lat: lat column name
        polygon: shapely polygon
    Returns:
        data_inside: resulting df, rows with lon/lat outside polygon dropped
        data_outside: df with rows outside polygon
    """

    inside = data.apply(
        lambda x: polygon.contains(Point(x[col_lon], x[col_lat]))
        , axis=1
    )
    data_inside = data[inside].reset_index(drop=True)
    data_outside = data[~inside].reset_index(drop=True)

    return data_inside, data_outside


@print_out_info
def move_to_nearest_allowed_area(
    data: pd.DataFrame,
    forb_coords_polygons: list,
    col_lon: str,
    col_lat: str
) -> pd.DataFrame:

    """
    Function that moves points inside of forbidden polygons to the closest boundary of given polygons.
    Polygons are passed as list with shapely objects.
    Args:
        data: pandas df with data, including lat/lon columns
        forb_coords_polygons: list with shapely polygon objects
        col_lon: df lon column name
        col_lat: df lat column name
    Returns:
        data_outside_polygon: df with points inside of polygons moved to the nearest boundaries of polygons
    """

    # Get data list with all forbidden rows, remove this rows from data
    data_outside_polygon = data
    in_forb_demand_data_list = []

    for polygon in forb_coords_polygons:
        data_inside_forb, data_outside_forb = get_rows_inside_polygon(data_outside_polygon,
                                                                      col_lon,
                                                                      col_lat,
                                                                      polygon)
        in_forb_demand_data_list.append(data_inside_forb)

        # Take only points outside forbidden (iterate over all polygons)
        data_outside_polygon = data_outside_forb.reset_index().drop('index', axis=1)

    # Move points in the forbidden area to the nearest o.a.
    for polygon_index, polygon in enumerate(forb_coords_polygons):
        in_forb_data = in_forb_demand_data_list[polygon_index]
        for row in in_forb_data.index:
            forb_point = Point(in_forb_data.loc[row, col_lon], in_forb_data.loc[row, col_lat])
            closest_point = ops.nearest_points(polygon.boundary, forb_point)[0]
            point_lon_tmp = closest_point._get_coords()[0][0]
            point_lat_tmp = closest_point._get_coords()[0][1]

            # Update location of a point
            in_forb_data.loc[row, col_lon] = point_lon_tmp
            in_forb_data.loc[row, col_lat] = point_lat_tmp

        data_outside_polygon = pd.concat([data_outside_polygon, in_forb_data], sort=True)

    return data_outside_polygon


@print_out_info
def remove_points_outside_polygons(
    data: pd.DataFrame,
    conf: dict,
    polygons: list
) -> pd.DataFrame:

    """
    Removes point outside oa_polygons (list with shapely polygons).
    Args:
        data: data from which points outside oa_polygons will be removed
        conf: config as dict
        polygons: polygons defining the area outside which data points will be removed

    Returns:
       inside_polygon_data: data inside polygons
    """

    # Remove points outside the polygons
    inside_polygon_data = pd.DataFrame()
    for oa_polygon in polygons:
        inside_polygon_data_pol, _ = get_rows_inside_polygon(
            data,
            col_lon=conf['col_lon_data'],
            col_lat=conf['col_lat_data'],
            polygon=oa_polygon
        )
        inside_polygon_data = inside_polygon_data.append(
            inside_polygon_data_pol,
            ignore_index=True,
            sort=True
        )
    inside_polygon_data = inside_polygon_data.reset_index(drop=True)

    return inside_polygon_data
