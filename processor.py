import random
import time
import re

def extract_station_number(station_name):
    match = re.search(r'\d+', station_name)
    return int(match.group()) if match else None

def process_markers(markers, grid_geojson, incident_geojson, station_data):
    '''
    Process markers and generate data for the frontend. 
    Markers and station_data should have station_id. 
    While grid and incident data should have station_id as a result.
    Args:
        markers (list): GeoJSON coordinates of markers.
        grid_geojson (dict): GeoJSON data for the grid.
        incident_geojson (dict): GeoJSON data for incidents.
        station_data (list): GeoJSON data for stations.
    Returns:
        dict: Processed data including station-cell-incident mapping, float array (travel times or distances), and updated GeoJSON data for colors.
    '''
    # p-median
    # assignments
    '''
    station, cell
    station, incident
    '''
    
    # Use GeoJSONs however needed (e.g., loop through features)
    for feature in grid_geojson.get("features", []):
        feature["properties"]["station_id"] = 5

    for feature in incident_geojson.get("features", []):
        feature["properties"]["station_id"] = 5

    # time.sleep(5)
    
    # Processed marker coords
    # processed_coords = [(lat + random.uniform(-0.01, 0.01), lon + random.uniform(-0.01, 0.01)) for lat, lon in markers]

    # Bar data with random values
    categories = ["A", "B", "C"]
    values = [random.randint(5, 20) for _ in categories]
    bar_data = {"category": categories, "value": values}

    # Float array
    # float_array = [round(random.uniform(0, 1), 3) for _ in markers]

    return {
        "processed_coords": None,
        "bar_data": bar_data,
        "float_array": [0, 1, 2, 3, 4],
        "grid_geojson": grid_geojson,
        "point_geojson": incident_geojson,
    }
