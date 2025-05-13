import random
import time

def process_markers(markers, grid_geojson, point_geojson):
    # Use GeoJSONs however needed (e.g., loop through features)
    for feature in grid_geojson.get("features", []):
        feature["properties"]["station_id"] = 5

    for feature in point_geojson.get("features", []):
        feature["properties"]["station_id"] = 5

    time.sleep(5)
    
    # Processed marker coords
    processed_coords = [(lat + random.uniform(-0.01, 0.01), lon + random.uniform(-0.01, 0.01)) for lat, lon in markers]

    # Bar data with random values
    categories = ["A", "B", "C"]
    values = [random.randint(5, 20) for _ in categories]
    bar_data = {"category": categories, "value": values}

    # Float array
    float_array = [round(random.uniform(0, 1), 3) for _ in markers]

    return {
        "processed_coords": processed_coords,
        "bar_data": bar_data,
        "float_array": float_array,
        "grid_geojson": grid_geojson,
        "point_geojson": point_geojson,
    }
