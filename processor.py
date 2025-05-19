import numpy as np
import re
import random
import time
import gurobipy as gp
from gurobipy import GRB
from collections import Counter
import joblib
import geopandas as gpd
import ast
import os
import pandas as pd
import json
import pickle
from shapely.geometry import Point

def extract_station_number(station_name):
    if isinstance(station_name, int):
        return station_name
    match = re.search(r'\d+', station_name)
    return int(match.group()) if match else None

def get_results(station_data, incident_data, grid_data):
    # read the files in output directory
    cell_to_station = pd.read_csv("output/cell_to_station.csv")
    fire_stations = pd.read_csv("output/fire_stations.csv")
    incident_to_station = pd.read_csv("output/incidents_to_station.csv")
    
    return cell_to_station, fire_stations, incident_to_station
    
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
    
    grids    = gpd.GeoDataFrame.from_features(grid_geojson,    crs="EPSG:4326")
    incidents= gpd.GeoDataFrame.from_features(incident_geojson,   crs="EPSG:4326")
    fire_stations = gpd.GeoDataFrame.from_features(station_data, crs="EPSG:4326")

    try:
        gdf_markers = gpd.GeoDataFrame.from_features(markers, crs="EPSG:4326")
        gdf_markers['FacilityName'] = (
            gdf_markers['station_id']
            .apply(lambda x: f"Station {x:02d}")
        )
        fire_stations=pd.concat([fire_stations,gdf_markers[['geometry','FacilityName']]],ignore_index=True)
    except Exception as e:
        print(f"Error processing markers: {e}")
    with open("data/d_distance.json", "r") as f:
        raw = json.load(f)
        d_dist = {tuple(map(int,k.split(","))): v for k,v in raw.items() }

    with open("data/d_time.json", "r") as f:
        raw = json.load(f)
        d_time = { tuple(map(int,k.split(","))): v for k,v in raw.items() }
        
    with open("data/a_historical_density.json", "r") as f:
        raw = json.load(f)
        a_density = { int(k): v for k, v in raw.items() }
        
    # markers={'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'properties': {'type': 'marker', '_leaflet_id': 15795, 'station_id': 2}, 'geometry': {'type': 'Point', 'coordinates': [-86.9121551513672, 36.27638594874375]}}, {'type': 'Feature', 'properties': {'type': 'marker', '_leaflet_id': 15824, 'station_id': 42}, 'geometry': {'type': 'Point', 'coordinates': [-86.75148010253906, 36.348314860643015]}}, {'type': 'Feature', 'properties': {'type': 'marker', '_leaflet_id': 15841, 'station_id': 43}, 'geometry': {'type': 'Point', 'coordinates': [-86.5935516357422, 36.10736936266137]}}]}


    fire_stations= gpd.sjoin(fire_stations, grids, how='inner', predicate='within')
    L=list(grids['cell_id'])
    X_exist= list(fire_stations['cell_id'])

    E = list(range(len(incidents)))
    # a= joblib.load("data/a_sub.pkl")
    # d= joblib.load("data/first_half_by_i.pkl")
    # assignments
    m, X, Y, b = add_p_via_mip_multi(L,L,a_density,d_dist,X_exist,p_add=0, alpha=0)
    sol_X_vals,sol_Y_assign=save_p_median_solution( X, Y, L, L)
    sol_X_vals.reset_index(inplace=True,names='location')
    sol_X = fill_solution_X(sol_X_vals, grids, fire_stations)
    sol_X= gpd.GeoDataFrame(sol_X, geometry='geometry')
    sol_X.crs= 'EPSG:4326'
    df=pd.merge(pd.merge(incidents,sol_Y_assign, left_on='cell_id', right_on='demand', how='inner'),sol_X,on='location', how='inner',suffixes=('','_fs'))
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    # 4) Build incident POINT geometry

    # 5) Compute distances
    gdf['distance'] = gdf.geometry.distance(gdf['geometry_fs'])
    # 6) Keep nearest per demand
    gdf = (
        gdf
        .sort_values('distance')
        .drop_duplicates(subset=['incident_id'], keep='first')
        .sort_values('incident_id')
        .reset_index(drop=True)
    )
    # 7) Drop helpers
    gdf = gdf.drop(columns=['distance'])
    incident_to_station=gdf[['incident_id',	'FacilityName',	'cell_id_fs',	'cell_id']]
    incident_to_station.rename(columns={'cell_id_fs':'assigned_cell_id'}, inplace=True)

    groups=gdf.groupby('FacilityName')
    travel_times={}
    travel_distances={}
    for name, group in groups:
        travel_time=[]
        travel_distance=[]
        for i,row in group.iterrows():
            travel_time.append(min(float(d_time[row['demand'],row['cell_id_fs']])*60,25*60))
            travel_distance.append(float(d_dist[row['demand'],row['cell_id_fs']]))
            
        travel_times[name]=travel_time
        travel_distances[name]=travel_distance
    travel_times_df=pd.Series(travel_times, name='travel_times').to_frame()
    travel_times_df.reset_index(inplace=True,names='FacilityName')
    travel_distances_df=pd.Series(travel_distances, name='travel_distances').to_frame()
    travel_distances_df.reset_index(inplace=True,names='FacilityName')
    fire_stations=pd.merge(fire_stations, travel_times_df, on='FacilityName', how='left')
    fire_stations=pd.merge(fire_stations, travel_distances_df, on='FacilityName', how='left')
    fire_stations.drop(columns=['geometry'], inplace=True)


    
    # cell_to_station, fire_stations, incident_to_station = get_results(station_data, None, None)

    for feature in grid_geojson.get("features", []):
        cell_id = feature["properties"].get("cell_id", None)
        try:
            facility_name = incident_to_station.loc[incident_to_station['cell_id'] == cell_id, 'FacilityName'].iloc[0]
            facility_name = extract_station_number(facility_name)
        except IndexError:
            facility_name = -1

        if cell_id is not None:
            feature["properties"]["station_id"] = facility_name
        else:
            # If no cell_id, assign a default station_id (e.g., 5)
            feature["properties"]["station_id"] = -1

    for feature in incident_geojson.get("features", []):
        incident_id = feature["properties"].get("incident_id", None)
        try:
            facility_name = incident_to_station.loc[incident_to_station['incident_id'] == incident_id, 'FacilityName'].iloc[-1]
            facility_name = extract_station_number(facility_name)
        except IndexError:
            facility_name = -1

        if incident_id is not None:
            # Same as above, if you have an array INCIDENT_COUNT long, then just do station_id = OUTPUT[incident_id]
            feature["properties"]["station_id"] = facility_name
        else:
            feature["properties"]["station_id"] = -1

    station_data = {}
    incident_counts = {}
    for k, v in fire_stations.iterrows():
        station_id = v["FacilityName"]
        travel_times = v["travel_times"]
        if isinstance(travel_times, list):
                station_data[station_id] = travel_times
                incident_counts[station_id] = len(travel_times)

            # Case 2: missing or NaN → treat as empty
        elif pd.isna(travel_times):
                station_data[station_id] = []
                incident_counts[station_id] = 0

            # Case 3: string representation of a list → parse it
        elif isinstance(travel_times, str):
                try:
                    parsed = ast.literal_eval(travel_times)
                    station_data[station_id] = parsed
                    incident_counts[station_id] = len(parsed)
                except (ValueError, SyntaxError):
                    station_data[station_id] = []
                    incident_counts[station_id] = 0

            # Fallback: anything else → empty
        else:
                station_data[station_id] = []
                incident_counts[station_id] = 0

    station_data = dict(sorted(station_data.items(), key=lambda x: x[0]))
    incident_counts = dict(sorted(incident_counts.items(), key=lambda x: x[0]))

    res = {
        "processed_coords": None,
        "station_data": station_data,
        "incident_counts":  incident_counts,
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



def compute_nearest_assignments(sol2_Y_assign: pd.DataFrame,
                                incidents: pd.DataFrame,
                                sol2_X: pd.DataFrame,
                                demand_col: str = 'demand',
                                incident_id_col: str = 'incident_id',
                                lat_col: str = 'lat',
                                lon_col: str = 'lon',
                                location_col: str = 'location',
                                x_col: str = 'x',
                                y_col: str = 'y',
                                geom_col: str = 'geometry',
                                crs: str = "EPSG:4326") :
    """
    For each demand in sol2_Y_assign, find the nearest facility from sol2_X
    by geographic distance to the incident location in incidents.

    Returns a GeoDataFrame with one row per demand, including:
      - lat, lon: incident coords
      - cell_id: facility cell_id (renamed from 'location')
      - centroid_lat, centroid_lon: facility centroid coords (from x,y)
      - geometry: POINT geometry of the incident
      - assigned_station_index: zero-based index for each unique cell_id
    """
    # 1) Merge demand → incident coords
    df = pd.merge(
        sol2_Y_assign,
        incidents[[incident_id_col, lat_col, lon_col]],
        left_on=demand_col, right_on=incident_id_col,
        how='left'
    )
    # 2) Merge in facility centroids & geometries
    df = pd.merge(
        df,
        sol2_X[[location_col, x_col, y_col, geom_col,'FacilityName']],
        on=location_col, how='left'
    )
    # 3) Cast to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geom_col, crs=crs)
    # 4) Build incident POINT geometry
    gdf['incident_geom'] = gpd.points_from_xy(gdf[lon_col], gdf[lat_col], crs=crs)
    # 5) Compute distances
    gdf['distance'] = gdf.geometry.distance(gdf['incident_geom'])
    # 6) Keep nearest per demand
    gdf = (
        gdf
        .sort_values('distance')
        .drop_duplicates(subset=[demand_col], keep='first')
        .sort_values(demand_col)
        .reset_index(drop=True)
    )
    # 7) Drop helpers
    gdf = gdf.drop(columns=[incident_id_col, 'distance'])
    # 8) Rename columns
    gdf = gdf.rename(columns={
        location_col:       'assigned_cell_id',
        lon_col:            'lon',
        lat_col:            'lat',
        x_col:              'centroid_lat',
        y_col:              'centroid_lon'
    })

    # 10) Set geometry to the incident point
    gdf = gdf.set_geometry('incident_geom')
    return gdf


def fill_solution_X(sol2_X_vals: pd.DataFrame,
                    grid_df: pd.DataFrame,
                    fire_stations: pd.DataFrame,
                    count_col: str = 'count',
                    loc_col:   str = 'location') :
    """
    Filters sol2_X_vals to rows where count_col > 0, merges in any existing
    fire_station geometries, and back-fills x, y, cell_id, and geometry
    from grid_df.

    Returns a new DataFrame with all missing values filled.
    """
    # 1) Filter and copy
    sol_X = sol2_X_vals[sol2_X_vals[count_col] > 0].copy()
    
    # 2) Build lookup maps from grid_df
    x_map      = dict(zip(grid_df['cell_id'], grid_df['x']))
    y_map      = dict(zip(grid_df['cell_id'], grid_df['y']))
    cellid_map = dict(zip(grid_df['cell_id'], grid_df['cell_id']))
    
    # 3) Merge any existing fire_station info
    sol_X = pd.merge(
        sol_X,
        fire_stations[['FacilityName','cell_id','x','y','geometry']],
        left_on=loc_col, right_on='cell_id',
        how='left',
        suffixes=('','_fs')
    )
    
    # 4) Fill in x, y, and cell_id from grid_df where still missing
    for col, lookup in [('x', x_map),
                        ('y', y_map),
                        ('cell_id', cellid_map)]:
        mask = sol_X[col].isna()
        sol_X.loc[mask, col] = sol_X.loc[mask, loc_col].map(lookup)
    
    # 5) Construct or keep geometry
    sol_X['geometry'] = sol_X.apply(
        lambda r: r['geometry']
                  if pd.notna(r['geometry'])
                  else Point(r['x'], r['y']),
        axis=1
    )
    
    # 6) Drop the temporary fire_station columns
    fs_cols = [c for c in sol_X.columns if c.endswith('_fs')]
    sol_X.drop(columns=fs_cols, inplace=True)
    
    return sol_X


def save_p_median_solution(
    X, Y, 
    E, L,

    to_csv=True,

):
    """
    Persist a p-median MIP solution (X, Y) in various formats.

    Parameters
    ----------
    X : dict-like of Gurobi vars
        X[j].X gives the integer count at location j.
    Y : dict-like of Gurobi vars
        Y[i,j].X gives the binary assignment of demand i to location j.
    E : iterable
        Demand indices.
    L : iterable
        Candidate‐location indices.
    output_dir : str, default="."
        Directory to write files into (will be created if necessary).
    prefix : str, default="solution"
        Filename prefix (e.g. "solution" → "solution_X.csv", etc.).
    to_csv : bool, default=True
        Whether to write CSV files (`{prefix}_X.csv`, `{prefix}_Y.csv`).
    to_json : bool, default=True
        Whether to write `{prefix}.json`.
    to_pickle : bool, default=True
        Whether to write `{prefix}.pkl`.

    Returns
    -------
    dict
        The Python objects that were saved: 
        {
          "X_vals": {j: int},
          "Y_assign": {i: j}
        }
    """
    # ensure output directory exists
    # os.makedirs(output_dir, exist_ok=True)

    # extract solution values
    X_vals = { j: int(X[j].X) for j in L }
    Y_assign = {
        i: j
        for i in E
        for j in L
        if Y[i,j].X > 0.5
    }

    # CSV output
    if to_csv:
        df_X = pd.DataFrame.from_dict(
            X_vals, orient="index", columns=["count"]
        )
        df_X.index.name = "location"
        # df_X.to_csv(os.path.join(output_dir, f"{prefix}_X.csv"))

        df_Y = pd.DataFrame([
            {"demand": i, "location": j}
            for i, j in Y_assign.items()
        ])
        # df_Y.to_csv(os.path.join(output_dir, f"{prefix}_Y.csv"), index=False)


    return df_X, df_Y

# model2.write("model2.lp")



def add_p_via_mip_multi(E, L, a, d, fire_stations, p_add, alpha=0):
    """
    E: list of demand indices
    L: list of candidate-cell indices
    a: dict i->demand weight
    d: dict (i,j)->distance
    fire_stations: list of cell-indices (may contain duplicates)
    p_add: how many new facilities to add
    alpha: 0 (classic) or 1 (balanced)
    """
    # 1) Count how many existing stations in each cell
    counts = Counter(fire_stations)
    P0 = sum(counts.values())
    p_new = P0 + p_add
    A = sum(a[i] for i in E)

    m = gp.Model("p_median_multi")

    # 2) X[j] integer in [0..p_new]
    X = m.addVars(L, lb=0, ub=p_new, vtype=GRB.INTEGER, name="X")
    Y = m.addVars(E, L, vtype=GRB.BINARY, name="Y")

    # 3) b[j] ∈ [0,1] is the normalized load fraction (only if alpha=1)
    if alpha == 1:
        b = m.addVars(L, lb=0.0, ub=1.0, name="b")

    # 4) Fix existing counts:
    for j, cnt in counts.items():
        m.addConstr(X[j] == cnt, name=f"fix_exist_{j}")

    # 5) Total facilities = old + new
    m.addConstr(X.sum() == p_new, name="total_facilities")

    # 6) Assignment & linkage
    for i in E:
        m.addConstr(Y.sum(i, "*") == 1, name=f"assign_{i}")
        for j in L:
            m.addConstr(Y[i,j] <= X[j], name=f"link_{i}_{j}")

    # 7) Define b[j] = (sum_i a[i]*Y[i,j]) / A
    if alpha == 1:
        for j in L:
            m.addConstr(
                gp.quicksum(a[i]*Y[i,j] for i in E) == b[j] * A,
                name=f"norm_{j}"
            )

    # 8) Objective
    if alpha == 0:
        obj = gp.quicksum(a[i]*d[i,j]*Y[i,j] for i in E for j in L)
        m.setObjective(obj, GRB.MINIMIZE)
    else:
        # now use b[j] directly (in [0,1])
        expr = gp.quicksum(a[i]*d[i,j]*Y[i,j]*b[j] for i in E for j in L)
        m.setObjective(expr, GRB.MINIMIZE)

    m.Params.NonConvex = int(alpha>0)  # allow nonconvex if we used b
    m.optimize()

    return m, X, Y, (b if alpha==1 else None)


def extract_station_number(station_name):
    match = re.search(r'\d+', station_name)
    return int(match.group()) if match else None




