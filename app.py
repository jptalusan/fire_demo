import json
import os
import dash
from dash import html, dcc, Input, Output, State, ctx
import dash_leaflet as dl
import plotly.graph_objs as go
import plotly.express as px
from processor import process_markers
from dash_extensions.javascript import assign

app = dash.Dash(__name__)

# Initial empty figure
initial_fig = go.Figure()
initial_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))


# Example: 10 distinct station IDs
station_ids = list(range(1, 10))  # e.g., 101 to 110

# Use Plotly categorical colors (or pick any 10 hex codes)
colors = px.colors.qualitative.Plotly[:10]

# Build the color map dictionary
color_map = {sid: color for sid, color in zip(station_ids, colors)}



point_to_layer = assign("""function(feature, latlng, context){
    const circleOptions = context.hideout.circleOptions;
    const id = feature.properties.station_id;
    circleOptions.fillColor = context.hideout.color_map[id] || "#ff0000";  // fallback color
    return L.circleMarker(latlng, circleOptions);  // render a simple circle marker
}""")

grid_style_handle = assign("""function(feature, context) {
    const id = feature.properties.station_id;
    const colorMap = context.hideout.colorMap;

    // If station_id is -1, remove the fill color and opacity
    if (id === -1) {
        return {
            color: "#333",    // Outline color
            weight: 0.5,       // Outline weight
            fillOpacity: 0     // No fill
        };
    }

    // If station_id is not -1, use the color map
    return {
        color: "#333",                   // Outline color
        weight: 0.5,                      // Outline weight
        fillColor: colorMap[id] || "#999999",  // fallback color
        fillOpacity: 0.5                  // Fill opacity
    };
}""")


app.layout = html.Div(style={'height': '100vh', 'display': 'flex', 'flexDirection': 'column'}, children=[
    # Main content: map and plots
    html.Div(style={'flex': '1', 'display': 'flex'}, children=[
        # Map Area (2/3)
        html.Div(style={'flex': '2', 'border': '1px solid #ccc', 'position': 'relative'}, children=[
            dl.Map(id="map", center=[36.174465, -86.767960], zoom=12,
                   style={'height': '100%', 'width': '100%'}, children=[
                       dl.LayersControl(id="layer-control", position="topright", children=[
                           dl.BaseLayer(
                                dl.TileLayer(
                                    # url="https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png",
                                    url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
                                    attribution="©OpenStreetMap ©CartoDB",
                                    id="carto-nolabels",
                                    maxZoom=20
                                ),
                                name="Dark Map",
                                checked=False,
                           ),
                           dl.BaseLayer(
                                dl.TileLayer(),
                                name="Base Map",
                                checked=True,
                           ),
                        dl.Overlay(
                            dl.GeoJSON(id="grid-layer", 
                                    options=dict(
                                        #    style=dict(color="blue", weight=0.25, fillOpacity=0.1)
                                            style=grid_style_handle,
                                        ),
                                        hideout=dict(colorMap=color_map),
                                        zoomToBounds=True,
                                    ),
                                name="Grid Layer",
                                checked=True,
                        ),
                        dl.Overlay(
                            dl.GeoJSON(id="point-layer", 
                                    zoomToBounds=True, 
                                    options=dict(pointToLayer=point_to_layer),
                                    hideout=dict(
                                        circleOptions=dict(
                                            radius=5,
                                            fillColor='red',
                                            fillOpacity=1,
                                            stroke=False,
                                        ),
                                            color_map=color_map,
                                    )
                            ),
                            name="Incidents Layer",
                            checked=True,
                        ),
                        dl.Overlay(
                            dl.GeoJSON(id="station-layer",
                                    zoomToBounds=True,
                                    options=dict(
                                        style=dict(color="blue", weight=0.25, fillOpacity=0.1)
                                    ),
                                    hideout=dict(colorMap=color_map),
                            ),
                            name="Stations Layer",
                            checked=True,
                        ),
                        dl.Overlay(
                            dl.LayerGroup(id="drawn-marker-layer"),
                            name="Drawn Markers",
                            checked=True,
                        ),
                        dl.Overlay(
                            dl.FeatureGroup([
                                dl.EditControl(
                                    id="draw-control",
                                    draw={"marker": True, "polyline": False, "polygon": False,
                                            "circle": False, "rectangle": False, "circlemarker": False},
                                    edit=False
                                )
                            ]),
                            name="Draw Control",
                            checked=True,
                        )
                        ]),
                   ])
        ]),

        # Plots (1/3)
        html.Div(style={'flex': '1', 'display': 'flex', 'flexDirection': 'column'}, children=[
            # Top Plot (will show processed points)
            html.Div(style={'flex': '1', 'border': '1px solid #ccc'}, children=[
                dcc.Loading(
                    id="loading-top",
                    type="default",
                    children=[
                        dcc.Graph(id='top-plot', figure=initial_fig)
                    ]
                )
            ]),
            # Bottom Plot (static)
            html.Div(style={'flex': '1', 'border': '1px solid #ccc'}, children=[
                dcc.Loading(
                    id="loading-bottom",
                    type="default",
                    children=[
                        dcc.Graph(id='bottom-plot', figure=initial_fig)
                    ]
                )
            ])
        ])
    ]),

    # Bottom area with buttons
    html.Div(style={
        'height': '60px',
        'display': 'flex',
        'justifyContent': 'center',
        'alignItems': 'center',
        'borderTop': '1px solid #ccc',
        'padding': '10px',
        'backgroundColor': '#f9f9f9'
    }, children=[
        html.Button("Load GeoJSON", id="btn-load-geojson", n_clicks=0, style={'marginRight': '10px'}),
        html.Button("Process", id="btn-process", n_clicks=0),
        dcc.Store(id="all-markers")  # store marker list
    ])
])

# Store all marker coordinates whenever one is added
@app.callback(
    Output("all-markers", "data"),
    Input("draw-control", "geojson"),
)
def update_marker_store(geojson):
    if not geojson or "features" not in geojson:
        return []

    coords = []
    for feature in geojson["features"]:
        if feature["geometry"]["type"] == "Point":
            lon, lat = feature["geometry"]["coordinates"]
            coords.append((lat, lon))
    return coords

# Process all markers and update the top-right plot
@app.callback(
    [
        Output("top-plot", "figure"),
        Output("bottom-plot", "figure"),
        Output("grid-layer", "data"),
        Output("point-layer", "data"),
        Output("station-layer", "data"),
    ],
    Input("btn-process", "n_clicks"),
    Input("btn-load-geojson", "n_clicks"),
    State("all-markers", "data"),
    State("grid-layer", "data"),
    State("point-layer", "data"),
    State("station-layer", "data"),
    prevent_initial_call=True,
    allow_duplicate=True
)
def process_and_plot(process_btn, load_btn, markers, grid_data, point_data, station_data):
    triggered_id = ctx.triggered_id

    if triggered_id == 'btn-process':
        if not markers or not grid_data or not point_data:
            # If no markers or data, return initial empty figures
            return initial_fig, initial_fig, dash.no_update, dash.no_update, dash.no_update

        # Callback to process markers (p-median)
        result = process_markers(markers, grid_data, point_data)
        processed = result["processed_coords"]
        bar_data = result["bar_data"]
        float_array = result["float_array"]
        updated_grid_data = result["grid_geojson"]
        updated_point_data = result["point_geojson"]

        # 1. Top plot from dictionary
        top_fig = go.Figure()
        top_fig.add_trace(go.Bar(x=bar_data["category"], y=bar_data["value"], name="Top Bar"))
        top_fig.update_layout(title="Top Bar Plot (from dict)")

        # 2. Bottom plot from float_array
        bottom_fig = go.Figure()
        bottom_fig.add_trace(go.Bar(x=list(range(len(float_array))), y=float_array, name="Float Values"))
        bottom_fig.update_layout(title="Float Array Plot")

        return top_fig, bottom_fig, updated_grid_data, updated_point_data, station_data
    
    elif triggered_id == 'btn-load-geojson':
        with open(os.path.join("data", "grids_w_id.geojson")) as f:
            grid_data = json.load(f)
        with open(os.path.join("data", "points.geojson")) as f:
            point_data = json.load(f)
        with open(os.path.join("data", "firestations.geojson")) as f:
            station_data = json.load(f)

        # print(point_data)
        return dash.no_update, dash.no_update, grid_data, point_data, station_data
        
@app.callback(
    Output("drawn-marker-layer", "children"),
    Input("draw-control", "geojson"),
)
def update_drawn_markers(geojson):
    if not geojson or "features" not in geojson:
        return []

    markers = []
    for feature in geojson["features"]:
        if feature["geometry"]["type"] == "Point":
            lon, lat = feature["geometry"]["coordinates"]
            # Example: Red marker
            markers.append(dl.CircleMarker(
                center=(lat, lon),
                radius=8,
                color="#000000",
                dashArray="4",
                fill=True,
                fillOpacity=0.3,
                weight=1,
                children=[
                    dl.Tooltip(f"New Station: {lat}, {lon}"),
                    dl.Popup(f"New Station: {lat}, {lon}")
                ]
            ))
    return markers


if __name__ == '__main__':
    app.run(debug=True)
