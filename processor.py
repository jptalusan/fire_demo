import random
import time
import numpy as np
import re

def extract_station_number(station_name):
    if isinstance(station_name, int):
        return station_name
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

    station_ids = []
    for feature in station_data.get("features", []):
        station_id = feature["properties"]["station_id"]
        if station_id is not None:
            station_ids.append(station_id)
    for feature in markers.get("features", []):
        station_id = feature["properties"]["station_id"]
        if station_id is not None:
            station_ids.append(station_id)
    station_ids = list(set(station_ids))

    res = {
        "processed_coords": None,
        "station_data": generate_station_data(station_ids),
        "incident_counts":  generate_incident_counts(station_ids, len(incident_geojson["features"])),
        "grid_geojson": grid_geojson,
        "point_geojson": incident_geojson,
    }
    return res

def generate_station_data(station_ids=[], min_samples=5, max_samples=15):
    station_data = {}
    for i in station_ids:
        station_name = f"Station {i:02d}"
        num_values = np.random.randint(min_samples, max_samples + 1)
        values = np.round(np.random.normal(loc=3.0, scale=1.0, size=num_values), 2)
        station_data[station_name] = values.tolist()
    return station_data

def generate_incident_counts(station_ids, total_incidents):
    num_stations = len(station_ids)
    # Randomly partition total_incidents into num_stations non-negative integers summing to total_incidents
    random_parts = np.random.multinomial(total_incidents, np.ones(num_stations)/num_stations)

    # Format station names and assign counts
    incident_counts_dict = {
        f"Station {int(station_id):02d}": count
        for station_id, count in zip(station_ids, random_parts)
    }

    return incident_counts_dict
