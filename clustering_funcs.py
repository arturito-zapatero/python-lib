import warnings

from haversine import haversine_vector
import logging
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.cluster import KMeans

import lib.data_utils as dt_uts
import lib.map_utils as mp_uts
import lib.polygon_utils as pol_uts
from lib.print_out_info import print_out_info


@print_out_info
def add_average_distance_per_cluster_col(
    data: pd.DataFrame,
    clusters: pd.DataFrame,
    cols_coords_clusters: list,
    cols_coords_data: list,
    col_cluster_id: str,
    col_distance: str
):
    """
    Takes data df and clusters df created on this data df, for each data row calculates distance between given
    data point and assigned cluster center. Calculates average distance from all data points assigned to one cluster.
    Args:
        data: data df with assigned cluster information (assigned cluster lon and lat cols)
        clusters: clusters df
        cols_coords_clusters: list with lon and lat column name denoting cluster locations
        cols_coords_data: list with lon and lat column name denoting data locations
        col_cluster_id: cluster id column
        col_distance: distance column

    Returns:
        data: data df with added distance to assigned cluster center column
        clusters: clusters df with average distance from all assigned data points column
    """

    # Obtain avg distance from scooter to cluster center per cluster
    data = dt_uts.add_distance_between_two_points_col(
        data,
        cols_coords_1=cols_coords_data,
        cols_coords_2=cols_coords_clusters,
        col_distance=col_distance
    )
    avg_distance_by_cluster = (data
                               .groupby(col_cluster_id)[col_distance]
                               .mean()
                               .reset_index())
    clusters = clusters.merge(avg_distance_by_cluster,
                              how='left')
    clusters.rename(columns={col_distance: 'avg_dist_to_scooter'},
                    inplace=True
                    )

    return data, clusters


@print_out_info
def add_clusters_count_col(
        clusters: pd.DataFrame,
        data: pd.DataFrame,
        col_cluster: str,
        col_count_data: str,
        col_count_clusters: str
) -> pd.DataFrame:

    """
    Adds information on cluster size (as sum of col_count_data from all the points assigned to given cluster)
    to cluster df.
    Note: both dfs need to have cluster_id column
    Args:
        clusters: clusters df
        data: data df with assigned cluster information (assigned cluster id
        col_cluster: cluster id column name
        col_count_data: data count column name
        col_count_clusters: count column name that will be created in clusters df

    Returns:
        clusters: clusters df with col_count_clusters added

    """

    # Add event count info for each cluster
    clusters_count = (data
                      .groupby(col_cluster)[col_count_data]
                      .sum()
                      .reset_index())

    clusters_count = clusters_count.rename(columns={col_count_data: col_count_clusters})
    clusters = clusters.merge(clusters_count,
                              how='left',
                              on=col_cluster)
    return clusters


@print_out_info
def add_clusters_info(
    data: pd.DataFrame,
    clusters: pd.DataFrame,
    n_scooters: int,
    n_events: int,
    cols_coords_data: list,
    cols_coords_clusters: list,
    col_cluster_id: str,
    col_count_data: str,
    col_distance: str
) -> [pd.DataFrame, pd.DataFrame]:

    """
    Adds info about clusters to clusters and data dfs:
    - add 'cluster_size' (based on sum col_count_data in data df) to clusters df
    - based on 'cluster_size' and n_scooters distributes scoots around clusters
    - merges data df and clusters df with newly added data
    - calculates col_distance to the nearest cluster center for data and avg col_distance from all points in data df
    belonging to given cluster
    Args:
        data: data df with assigned cluster information (assigned cluster lon and lat cols)
        clusters: clusters df
        n_scooters: number of available scooters
        n_events: total number of events in data df
        cols_coords_data: string denoting column
        cols_coords_clusters: string denoting column
        col_cluster_id: string denoting column
        col_count_data: string denoting column
        col_distance: string denoting column

    Returns:
        data: data df with assigned cluster information (assigned cluster lon and lat cols)
        clusters: clusters df with information added
    """

    clusters = add_clusters_count_col(
        clusters=clusters,
        data=data,
        col_cluster=col_cluster_id,
        col_count_data=col_count_data,
        col_count_clusters='cluster_size'
    )

    clusters = distribute_scooters_by_clusters(
        clusters,
        n_scooters,
        n_events
    )

    # Merge cluster and data df
    data = data.merge(
        clusters,
        on=col_cluster_id,
        how='left',
        sort=False
    )

    # Obtain distance to cluster center
    data, clusters = add_average_distance_per_cluster_col(
        data,
        clusters,
        cols_coords_clusters,
        cols_coords_data,
        col_cluster_id,
        col_distance
    )

    clusters, _, _ = calc_shortest_dist_between_clusters(
        clusters=clusters,
        col_cluster='cluster_id',
        col_latitude='cluster_latitude',
        col_longitude='cluster_longitude',
        col_distance='distance'
    )

    return data, clusters


@print_out_info
def assign_data_to_nearest_clusters(
    data: pd.DataFrame,
    clusters: pd.DataFrame,
    col_cluster_clusters: str,
    col_cluster_data: str,
    cols_coords_clusters: list,
    cols_coords_data: list,
    batch_size: [int, None] = 50000,
    metric: str = 'euclidean'
) -> pd.DataFrame:

    """
    Function that assigns points to the nearest cluster using euclidean distance, by adding new column with
    col_cluster_data to the data df. Cluster assignment is made for data split in batches to avoid memory problem.
    Args:
        data: pandas df with data to which we want to assign clusters, needs to have cols_coords_data columns
        clusters: pandas df with centers/clusters data, needs to have cols_coords_centers with coords and
         col_cluster_centers
        col_cluster_clusters:  name of the column containing cluster id, which will be created in centers df
        col_cluster_data:  name of the column containing cluster id in data df
        cols_coords_clusters: strings specifying coordinates column(s) names in centers df
        cols_coords_data: as above but corresponding column names in data df
        batch_size: batch size
        metric: metric to measure distance to the nearest cluster
    Returns:
        data - data df with new column col_cluster denoting assigning to the closest cluster center
    """

    # Only needed columns
    clusters_loc = clusters.loc[:, cols_coords_clusters]
    data_loc = data.loc[:, cols_coords_data]

    if batch_size:
        # Split into chunks
        n_batches = int(data_loc.shape[0] / batch_size) + 1
        data_batches = np.array_split(data_loc, n_batches)

        # Find the closest center for each point
        index_min_value = pd.Series(dtype=int)
        for ind, data_batch in enumerate(data_batches):
            dist_matrix = distance.cdist(
                clusters_loc,
                data_batch,
                metric=metric
            )
            dist_matrix = pd.DataFrame(dist_matrix)
            index_min_value_batch = dist_matrix.idxmin(axis=0)
            index_min_value = index_min_value.append(index_min_value_batch)

    else:

        dist_matrix = distance.cdist(
            clusters_loc,
            data_loc,
            metric=metric
        )
        dist_matrix = pd.DataFrame(dist_matrix)
        index_min_value = dist_matrix.idxmin(axis=0)

    data[col_cluster_data] = clusters.loc[index_min_value, col_cluster_clusters].values

    return data


@print_out_info
def calc_shortest_dist_between_clusters(
    clusters: pd.DataFrame,
    col_cluster: str,
    col_longitude: str,
    col_latitude: str,
    col_distance: str
) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
    Calculates distance to closest cluster for each of the clusters and adds this information as a new column.
    Also returns df with all possible distances between clusters combinations and df with minimum distance for
    each cluster.
    Args:
        clusters: clusters df
        col_cluster: string denoting column
        col_longitude: string denoting column
        col_latitude: string denoting column
        col_distance: string denoting column
    Returns:
        clusters: clusters with with f"{col_distance}_to_closest_cluster" column added
        cart_prod : all possible combination of clusters and distances between them
        min_distances: information about the closest distance for each cluster
    """

    # Calculate all possible distances between clusters using cartesian product
    clusters_1 = clusters.loc[:, [col_cluster,
                                  col_latitude,
                                  col_longitude]]
    clusters_1.rename(columns={col_cluster: f"{col_cluster}_1",
                               col_latitude: f"{col_latitude}_1",
                               col_longitude: f"{col_longitude}_1"},
                      inplace=True)
    clusters_2 = clusters.loc[:, [col_cluster, 'cluster_latitude', 'cluster_longitude']]
    clusters_2.rename(columns={col_cluster: f"{col_cluster}_2",
                               col_latitude: f"{col_latitude}_2",
                               col_longitude: f"{col_longitude}_2"},
                      inplace=True)
    clusters_1['key'] = 0
    clusters_2['key'] = 0

    # Create cartesian product
    cart_prod = clusters_1.merge(
        clusters_2,
        how='outer',
        sort=True
    )
    cart_prod.drop('key', axis=1, inplace=True)
    cart_prod.drop(cart_prod[cart_prod[f"{col_cluster}_1"] == cart_prod[f"{col_cluster}_2"]].index,
                   inplace=True)

    # Calculate distances between every cluster
    cart_prod[col_distance] = haversine_vector(
        [tuple(x) for x in cart_prod[[f"{col_latitude}_1", f"{col_longitude}_1"]].to_numpy()],
        [tuple(x) for x in cart_prod[[f"{col_latitude}_2", f"{col_longitude}_2"]].to_numpy()],
        unit='m'
    )

    # Find min distances for each cluster and merge them to clusters df
    min_distances = cart_prod.groupby(f"{col_cluster}_1")[col_distance].min().reset_index()
    clusters = clusters.drop(
        col_distance,
        axis=1,
        errors='ignore'
    )
    clusters = clusters.merge(
        min_distances,
        left_on=col_cluster,
        right_on=f"{col_cluster}_1",
        how='left',
        sort=True
    )
    clusters.drop(
        f"{col_cluster}_1",
        axis=1,
        inplace=True
    )
    clusters.rename(
        columns={col_distance: f"{col_distance}_to_closest_cluster"},
        inplace=True
    )

    return clusters, cart_prod, min_distances


@print_out_info
def create_clusters(
    data: pd.DataFrame,
    n_clusters: int,
    cols_coords_clusters: list,
    cols_coords_data: list,
    col_cluster_id: str,
    col_cluster_id_prefix: str = ''
) -> [pd.DataFrame, pd.DataFrame]:
    """
    Creates cluster using K-means algorithm based on data df.
    Args:
        data: input data
        n_clusters: number of KMeans centroids
        cols_coords_clusters: denoting lon/lat columns in the created clusters df
        cols_coords_data: lon/lat columns in the input data df
        col_cluster_id: cluster id column in both data and clusters df
        col_cluster_id_prefix: prefix to col_cluster_id

    Returns:
        data: data df with col_cluster_id col added identifying the closest cluster centroid
        clusters: centers df created using KMeans algo
    """

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=45,
        max_iter=75,
        n_init=7
    ).fit(data[cols_coords_data])

    # Get info from kmeans object
    data_clusters_labels = kmeans.labels_

    # Label data with clusters ids
    data.loc[:, col_cluster_id] = data_clusters_labels
    data[col_cluster_id] = data[col_cluster_id].apply(lambda x: col_cluster_id_prefix + str(x))

    # Create df with clusters info, rename column names
    clusters = data.groupby(col_cluster_id)[cols_coords_data].mean().reset_index()

    columns_to_rename = zip(
        cols_coords_data,
        cols_coords_clusters
    )
    clusters = clusters.rename(
        columns=dict(columns_to_rename),
        inplace=False
    )

    return data, clusters


@print_out_info
def obtain_n_clusters_range(
    oa_area: [int, float],
    static_oa_area: [int, float],
    min_dps_per_km2: int = 5,
    max_dps_per_km2: int = 25,
    clusters_steps: int = 5,

) -> range:

    """
    Obtains the optimal number of DPs (n_clusters) based on the OA areas.
    Args:
        oa_area: OA area in km2
        static_oa_area: static OA area in km2
        min_dps_per_km2: lower limit for DPs per km2
        max_dps_per_km2: upper limit for DPs per km2
        clusters_steps: how many range steps

    Returns:
        n_clusters_range: DPs range
    """

    free_area = oa_area - static_oa_area
    min_clusters = int(min_dps_per_km2 * free_area)
    max_clusters = int(max_dps_per_km2 * free_area)
    clusters_step = int((max_clusters - min_clusters) / clusters_steps)

    n_clusters_range = range(min_clusters, max_clusters + 1, clusters_step)

    return n_clusters_range


@print_out_info
def recreate_clusters_from_data(
    data: pd.DataFrame,
    col_cluster_id:str,
    col_lon_clusters: str,
    col_lat_clusters: str
) -> pd.DataFrame:
    """
    Obtain information about clusters from data. Takes data df with col_cluster_id and creates a clusters df with
     col_cluster_id, col_lon_clusters and col_lat_clusters. Assumption is that there each cluster_id has a unique
     location.
    Args:
        data: data df
        col_cluster_id: cluster label column
        col_lon_clusters: longitude column
        col_lat_clusters: latitude column

    Returns:
        cluster: cluster df with clusters location and labels
    """

    clusters = (
        data
        .groupby([col_cluster_id])[[col_lon_clusters, col_lat_clusters]]
        .first()
        .reset_index()
        .sort_values(col_cluster_id)
    )
    return clusters
