import json


def create_regions_polygons():

    regions_dict = {
        "Regions": {
            "Region1": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [
                            0.0,
                            0.0
                        ],
                        [
                            0.0,
                            10
                        ],
                        [
                            5,
                            10
                        ],
                        [
                            5,
                            0.0
                        ],
                        [
                            0.0,
                            0.0
                        ]
                    ]
                ]
            },
            "Region2": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [
                            5,
                            0.0
                        ],
                        [
                            5,
                            10
                        ],
                        [
                            10,
                            10
                        ],
                        [
                            10,
                            0.0
                        ],
                        [
                            5,
                            0.0
                        ]
                    ]
                ]
            },
            "Region3": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [
                            10,
                            0.0
                        ],
                        [
                            10,
                            10
                        ],
                        [
                            15,
                            10
                        ],
                        [
                            15,
                            0.0
                        ],
                        [
                            10,
                            0.0
                        ]
                    ]
                ]
            },
            "Region4": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [
                            15,
                            0.0
                        ],
                        [
                            15,
                            10
                        ],
                        [
                            20,
                            10
                        ],
                        [
                            20,
                            0.0
                        ],
                        [
                            15,
                            0.0
                        ]
                    ]
                ]
            }
        }
    }

    return regions_dict
