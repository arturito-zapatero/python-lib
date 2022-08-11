import numpy as np
import os
import pandas as pd
import requests

import lib.base_utils as bs_uts
from lib.print_out_info import print_out_info


@print_out_info
def snap_to_roads(
    locations: pd.DataFrame,
    location_lon: str = "cluster_longitude",
    location_lat: str = "cluster_latitude"
) -> [pd.DataFrame, None]:

    """
    Takes the locations and moves them to the closest road using google API
    Args:
        locations: df with locations to be moved to the closest road
        location_lon: cluster_longitude
        location_lat: cluster_latitude
    Returns:
        locations_snapped: df with locations moved to the closest roads
    """

    os.environ['google_api_key'] = (bs_uts.get_ssm_secret('/machine_learning/google_api_key')
                                                          ['Parameter']['Value'])
    os.environ['google_endpoint'] = (bs_uts.get_ssm_secret('/machine_learning/google_roads_endpoint')
                                                            ['Parameter']['Value'])

    if locations.shape[0] > 0:
        # Split into a chunks of below 100 points (max number of points that google API accepts)
        number_of_chunks = int(np.ceil(locations.shape[0] / 100))
        locations_chunks_list = np.array_split(
            locations,
            number_of_chunks
        )
        locations_snapped = pd.DataFrame()
        for i in range(number_of_chunks):
            locations_chunk = pd.DataFrame(locations_chunks_list[i])
            locations_chunk = locations_chunk.reset_index(drop=True)

            # Create the input to the API
            points = ''
            for index, row in locations_chunk.iterrows():
                if index > 0:
                    points += '|'
                points = points + str(row[location_lat]) + ',' + str(row[location_lon])

            # Send request to API and obtain response
            payload = {'points': points, 'key': os.environ['google_api_key']}
            r = requests.get(os.environ['google_endpoint'], params=payload).json()

            # Store information into dataframe
            new_locations = locations_chunk.copy()
            if 'snappedPoints' in r and len(r['snappedPoints']) > 0:
                for point in r['snappedPoints']:
                    new_locations.loc[point['originalIndex'], location_lat] = point['location']['latitude']
                    new_locations.loc[point['originalIndex'], location_lon] = point['location']['longitude']
            locations_snapped = locations_snapped.append(new_locations)
    else:
        return None
    return locations_snapped
