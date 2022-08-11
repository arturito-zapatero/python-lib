import pandas as pd
from shapely.geometry.polygon import Polygon
import shapely

from lib.map_utils import extract_kml_points_coords_to_df, extract_kml_polygons_coords_to_polygons
from lib.polygon_utils import get_rows_inside_polygon, move_to_nearest_allowed_area, subtract_polygons_from_polygon,\
    count_items_in_polygons
import lib.map_utils as map_uts
import lib.regions_funcs as reg_fcs

from tests.test_regions_funcs import test_assign_data_to_regions
import tests.create_dataframes_blueprints as blp
import tests.create_polygon_blueprints as poly_blp

"""
Test for functions inside polygon_utils.py. This f. uses UT_map.kml described below. 
It creates shapely polygons, one with standard shape and one with non-standard (ie. can be split into smaller polygons - 
shapely MultiPolygon object) and 14 shapely points. Polygon polygon_with_points_inside_1 contains 6 points inside, 
polygon number_of_points_inside_polygon_2 contains 3 points inside.

Map for testing in .kml format was created using google mymaps, with five layers, each of layers contain polygon
(usually two) or points. Layers', polygons' and points' name exhibit different naming scenarios (e.g. with spaces
or underscores). Below list of layers (marked by numbers) is given, for each layer polygons/points associated with this
 layer are listed. For test purposes some of the points are inside of some polygons, some outside, and one is on the 
 polygon's boundary. The association of this points with polygons is given after the layers list.

1. layer w spaces
- polygon with spaces
- polygon_with_underscores

2. layer_w_underscores
- polygon_w_unusual_chars'$%&' (in .kml file saved as: <![CDATA[polygon_with_unusual_chars'$%&']]>)
- 1 2

3. 1 2 \
- Polygon 1
- Polygon 2  

4. layer_unusual_chars_'$%&' (in .kml file saved as: <![CDATA[layer_unusual_chars_'$%&']]>)
- Polygon 1
- Polygon 2

5. empty_layer

6. layer_w_points

Points inside layer: 'layer w spaces', polygon: 'polygon_w_underscores'
- point_inside_with_underscores_1
- point inside with spaces
- point_inside_unusual_chars_'$%&' (in .kml file saved as: <![CDATA[point_inside_unusual_chars_'$%&']]>)
- point_inside_1
- point_inside_2
- point_inside_3
- point_on_boundary

Points inside layer: '1 2', polygon: 'Polygon 1'
- point_inside_strange_poly_1
- point_inside_strange_poly_2
- point_inside_strange_poly_3
- point_on_boundary

Points outside of every polygon.
- point_outside_1
- point_outside_2
- point_outside_3
- point_outside_4
"""


# SETUP
# Define map location and polygon names:
kml_file_path = "../../../data/UT_map.kml"

layer_with_polygon_with_points_inside_name_1 = "layer w empty spaces"
layer_with_polygon_with_points_inside_name_2 = "1 2"
layer_with_points_name = "layer_w_points"

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
col_scoots = 'n_scooters'
col_description = 'description'
cluster_ids = [f'cluster_{i}' for i in range(1,6)]
descriptions = [f'pt{i}' for  i in range(1,6)]

# Define testing pandas DF
input_dir = {
    col_cluster_id: cluster_ids,
    col_lon_clusters: [2, 4, 6, 8, 10],
    col_lat_clusters: [-2.0, 4.0, -6.0, 7.5, -10.0],
    col_scoots: [1, 2, 3, 4, 5],
    col_description: descriptions,
    }

input_df = pd.DataFrame(data=input_dir)

# Define counts
number_of_points_inside_polygon_1 = 6
number_of_points_outside_polygon_1 = 8
number_of_points_inside_polygon_2 = 3
number_of_points_outside_polygon_2 = 11

# Polygon 1
polygon_with_points_inside_1 = extract_kml_polygons_coords_to_polygons(
        kml_file=kml_file_path,
        layer_name=layer_with_polygon_with_points_inside_name_1
)
# Polygon 2
polygon_with_points_inside_2 = extract_kml_polygons_coords_to_polygons(
        kml_file=kml_file_path,
        layer_name=layer_with_polygon_with_points_inside_name_2
)

# Points DF
points_df = extract_kml_points_coords_to_df(
        kml_file=kml_file_path,
        layer_name=layer_with_points_name,
        col_lon = col_lon_clusters,
        col_lat = col_lat_clusters,
        col_desc = col_cluster_id
    )
point_inside_1 = (points_df
                  .loc[points_df[col_cluster_id] == 'point_inside_1', [col_lon_clusters, col_lat_clusters]]
                  .reset_index()
                  .drop('index', axis=1)
                  )
point_outside_1 = (points_df
                   .loc[points_df[col_cluster_id] == 'point_outside_1', [col_lon_clusters, col_lat_clusters]]
                   .reset_index()
                   .drop('index', axis=1)
                   )

def test_get_rows_inside_standard_polygon_count_and_nan_check():

    # EXERCISE
    data_inside, data_outside = get_rows_inside_polygon(
        data=points_df,
        col_lon=col_lon_clusters,
        col_lat=col_lat_clusters,
        polygon=polygon_with_points_inside_1[0]
    )

    # VERIFY
    assert data_inside.dropna().shape[0] == number_of_points_inside_polygon_1
    assert data_outside.dropna().shape[0] == number_of_points_outside_polygon_1


def test_get_rows_inside_not_standard_polygon_count_and_nan_check():

    # EXERCISE
    data_inside, data_outside = get_rows_inside_polygon(
        data=points_df,
        col_lon=col_lon_clusters,
        col_lat=col_lat_clusters,
        polygon=polygon_with_points_inside_2[0]
    )

    # VERIFY
    assert data_inside.dropna().shape[0] == number_of_points_inside_polygon_2
    assert data_outside.dropna().shape[0] == number_of_points_outside_polygon_2


def test_move_to_nearest_allowed_area_point_inside_moved():


    # EXERCISE
    points_moved_inside = move_to_nearest_allowed_area(
        data=points_df,
        forb_coords_polygons=polygon_with_points_inside_1,
        col_lon=col_lon_clusters,
        col_lat=col_lat_clusters
    )
    point_moved_1 = (points_moved_inside
                     .loc[points_moved_inside[col_cluster_id] == 'point_inside_1', [col_lon_clusters, col_lat_clusters]]
                    .reset_index()
                    .drop('index', axis=1)
                     )

    # VERIFY
    assert float(point_inside_1[col_lon_clusters]) != float(point_moved_1[col_lon_clusters])
    assert float(point_inside_1[col_lat_clusters]) != float(point_moved_1[col_lat_clusters])


def test_move_to_nearest_allowed_area_point_outside_not_moved():

    # EXERCISE
    points_moved_inside = move_to_nearest_allowed_area(
        data=points_df,
        forb_coords_polygons=polygon_with_points_inside_1,
        col_lon=col_lon_clusters,
        col_lat=col_lat_clusters
    )
    point_not_moved_1 = (points_moved_inside
                         .loc[points_moved_inside[col_cluster_id] == 'point_outside_1',
                              [col_lon_clusters, col_lat_clusters]]
                         .reset_index()
                         .drop('index', axis=1)
                         )

    # VERIFY
    assert float(point_outside_1[col_lon_clusters]) == float(point_not_moved_1[col_lon_clusters])
    assert float(point_outside_1[col_lat_clusters]) == float(point_not_moved_1[col_lat_clusters])


def test_subtract_polygons_from_polygon():

    # SETUP
    list_of_polygons_to_subtract = map_uts.extract_kml_polygons_coords_to_polygons(
        kml_file_path,
        'layer_inside_layer_with_spaces'
    )
    list_of_polygon_to_subtract_from = map_uts.extract_kml_polygons_coords_to_polygons(
        kml_file_path,
        'layer with spaces'
    )

    # EXECUTE
    for polygon_to_subtract_from in list_of_polygon_to_subtract_from:
        subtracted_polygon = subtract_polygons_from_polygon(
            polygon_to_subtract_from,
            list_of_polygons_to_subtract
        )

        # VERIFY
        for polygon_to_subtract in list_of_polygons_to_subtract:
            assert not subtracted_polygon.contains(polygon_to_subtract)
        assert (type(subtracted_polygon) == shapely.geometry.point.Point) | \
               (type(subtracted_polygon) == shapely.geometry.multipolygon.MultiPolygon) | (
                           type(subtracted_polygon) == shapely.geometry.polygon.Polygon)


def test_assign_data_to_polygons():
    # Already tested in test_assign_data_to_regions
    test_assign_data_to_regions()
    assert 1==1



col_lon_data = 'event_longitude'
col_lat_data = 'event_latitude'
col_lon_clusters = 'cluster_longitude'
col_lat_clusters = 'cluster_latitude'
col_cluster_id = 'cluster_id'
col_polygon_id = 'regions_name'
col_live_scoots = 'live_scoots_count'
col_pred_rides_per_scoot_cluster = "pred_rides_per_scoot_cluster"
col_sm_delta = "sm_delta"
col_deploy = 'deploy'

conf = {
    'col_lon_data': col_lon_data,
    'col_lat_data': col_lat_data,
    'col_lon_clusters': col_lon_clusters,
    'col_lat_clusters': col_lat_clusters,
    'col_live_scoots': col_live_scoots,
    'col_deploy': col_deploy,
    'col_cluster_id': col_cluster_id
}

regions_dict = poly_blp.create_regions_polygons()

polygon_region_1 = Polygon(regions_dict['Regions']['Region1']['coordinates'][0])
polygon_region_2 = Polygon(regions_dict['Regions']['Region2']['coordinates'][0])
polygon_region_3 = Polygon(regions_dict['Regions']['Region3']['coordinates'][0])
polygon_region_4 = Polygon(regions_dict['Regions']['Region4']['coordinates'][0])
regions_polygons = [polygon_region_1, polygon_region_2, polygon_region_3, polygon_region_4]

regions_names = ['Region1', 'Region2', 'Region3', 'Region4']


def test_count_items_in_polygons():

    input_data, _ = blp.create_test_dataframe(case='scoots_data_in_4_regions')

    expected_columns_data = {
        'scoots_in_polygon',
        'regions_name'
    }

    regions_1 = pd.DataFrame({
        'regions_name': regions_names,
        'regions_polygons': regions_polygons
    })
    input_data = reg_fcs.assign_data_to_regions(
        sm_delta_data=input_data.copy(),
        regions=regions_1,
        col_polygon_id=col_polygon_id,
        col_polygon='regions_polygons',
        conf=conf
    )

    output_data_1 = count_items_in_polygons(
        polygon_data=regions_1,
        data_to_count=input_data,
        col_polygon='regions_polygons',
        col_lon=conf['col_lon_clusters'],
        col_lat=conf['col_lat_clusters'],
        col_count_polygons='scoots_in_polygon',
        col_count_data=f"adj_{conf['col_live_scoots']}"
    )

    assert list(output_data_1['scoots_in_polygon']) == [25.0, 25.0, 25.0, 25.0]
    assert expected_columns_data == set(output_data_1[expected_columns_data].columns)
