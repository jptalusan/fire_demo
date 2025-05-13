import json
import os
import dash
from dash import html, dcc, Input, Output, State, ctx
import dash_leaflet as dl
import plotly.graph_objs as go
import plotly.express as px
from processor import process_markers, extract_station_number
from dash_extensions.javascript import assign

app = dash.Dash(__name__)

# Initial empty figure
initial_fig = go.Figure()
initial_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

# Combine multiple qualitative color scales
combined_colors = (
    px.colors.qualitative.Plotly +
    px.colors.qualitative.D3 +
    px.colors.qualitative.T10 +
    px.colors.qualitative.Alphabet +
    px.colors.qualitative.Set3
)

MAX_STATIONS = 50
# Example: 10 distinct station IDs
station_ids = list(range(1, MAX_STATIONS))  # e.g., 101 to 110

# Use Plotly categorical colors (or pick any 10 hex codes)
colors = combined_colors[:MAX_STATIONS]

# Build the color map dictionary
color_map = {sid: color for sid, color in zip(station_ids, colors)}

point_to_layer = assign("""function(feature, latlng, context){
    const circleOptions = { ...context.hideout.circleOptions };  // clone to avoid mutation
    const iid = feature.properties.incident_id;
    const sid = feature.properties.station_id;
    const colorMap = context.hideout.color_map;

    // Assign color based on station_id
    circleOptions.fillColor = colorMap[sid] || "#ff0000";  // fallback color if not in color_map

    const marker = L.circleMarker(latlng, circleOptions);

    // Tooltip content from feature properties
    const tooltipText = "Incident: " + (iid !== undefined ? iid : "N/A") + "<br>Station ID: " + (sid !== undefined ? sid : "N/A");
    marker.bindTooltip(tooltipText);

    return marker;
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

station_tooltip_handle = assign("""function(feature, layer) {
    if (feature.properties && feature.properties.FacilityName) {
        layer.bindTooltip(feature.properties.FacilityName);
    }
}""")

incident_tooltip_handle = assign("""function(feature, layer) {
    if (feature.properties && feature.properties.incident_id) {
        layer.bindTooltip(feature.properties.incident_id);
    }
}""")

grid_tooltip_handle = assign("""function(feature, layer) {
    if (feature.properties && feature.properties.station_id) {
        const cell_id = feature.properties.cell_id;
        const station_id = feature.properties.station_id;
        layer.bindTooltip("Cell ID: " + cell_id + "<br>Station ID: " + station_id);
    }
    else {
        layer.bindTooltip("No Station Assigned");
    }
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
                                        onEachFeature=grid_tooltip_handle,
                                    ),
                                name="Grid Layer",
                                checked=True,
                        ),
                        dl.Overlay(
                            dl.GeoJSON(id="point-layer", 
                                    zoomToBounds=True, 
                                    options=dict(
                                        pointToLayer=point_to_layer,
                                    ),
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
                                        style=dict(color="blue", weight=0.25, fillOpacity=0.1),
                                        onEachFeature=station_tooltip_handle,
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
        html.Div(id="marker-count", style={"marginTop": "10px", "fontWeight": "bold", 'marginRight': '10px'}),
        html.Button("Load GeoJSON", id="btn-load-geojson", n_clicks=0, style={'marginRight': '10px'}),
        html.Button("Process", id="btn-process", n_clicks=0),
        dcc.Store(id="all-markers"),  # store marker list
        dcc.Store(id="station-ids"),
        dcc.Store(id="new-station-ids"),
    ])
])

@app.callback(
    Output("marker-count", "children"),
    Input("all-markers", "data"),
    prevent_initial_call=False
)
def update_marker_count(marker_data):
    if not marker_data:
        return "No markers added."
    
    count = len(marker_data.get("features", []))
    return f"{count} new station{'s' if count != 1 else ''} added."


# Process all markers and update the top-right plot
@app.callback(
    [
        Output("top-plot", "figure"),
        Output("bottom-plot", "figure"),
        Output("grid-layer", "data"),
        Output("point-layer", "data"),
        Output("station-layer", "data"),
        Output("station-ids", "data", allow_duplicate=True),
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
            return initial_fig, initial_fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        # Callback to process markers (p-median)
        result = process_markers(markers, grid_data, point_data, station_data)
        
        station_data = result["station_data"]
        incident_counts = result["incident_counts"]
        updated_grid_data = result["grid_geojson"]
        updated_point_data = result["point_geojson"]

        # 1. Top plot from dictionary
        top_fig = go.Figure()

        box_color = "#1f77b4"  # A Plotly default blue
        # Create a box plot for each station
        for station, values in result["station_data"].items():
            top_fig.add_trace(go.Box(y=values, name=station, boxpoints='outliers', showlegend=False, jitter=0.5, marker_color=box_color))

        top_fig.update_layout(
            title="Travel Distance by Station",
            xaxis_title="Station",
            yaxis_title="Value",
            margin=dict(l=40, r=20, t=40, b=30),
            xaxis=dict(
                tickangle=-90,
                automargin=True  # ensures labels don't get cut off
            ),
        )

        # 2. Bottom plot from float_array
        bottom_fig = make_incident_bar_plot(incident_counts)
        return top_fig, bottom_fig, updated_grid_data, updated_point_data, station_data, dash.no_update
    
    elif triggered_id == 'btn-load-geojson':
        with open(os.path.join("data", "grids.geojson")) as f:
            grid_data = json.load(f)
        with open(os.path.join("data", "incidents.geojson")) as f:
            point_data = json.load(f)
        with open(os.path.join("data", "firestations.geojson")) as f:
            station_data = json.load(f)

        station_ids = []
        for station in station_data["features"]:
            facility_name = station["properties"]["FacilityName"]
            station_id = extract_station_number(facility_name)
            station_ids.append(station_id)
            station["properties"]["station_id"] = station_id
        unique_ids = sorted(set(station_ids))

        return dash.no_update, dash.no_update, grid_data, point_data, station_data, unique_ids
        
@app.callback(
    Output("drawn-marker-layer", "children"),
    Output("all-markers", "data"),
    Output("station-ids", "data", allow_duplicate=True),
    Input("draw-control", "geojson"),
    State("station-ids", "data"),
    prevent_initial_call=True,
)
def update_drawn_markers(geojson, station_ids):
    if not geojson or "features" not in geojson:
        return [], dash.no_update, dash.no_update

    if not station_ids:
        station_ids = []

    markers = []
    new_station_ids = []
    for feature in geojson["features"]:
        if feature["geometry"]["type"] == "Point":
            # Find lowest unused station_id >= 1
            next_id = 1
            while (next_id in station_ids) or (next_id in new_station_ids):
                next_id += 1
            new_station_ids.append(next_id)
            lon, lat = feature["geometry"]["coordinates"]
            feature["properties"]["station_id"] = next_id
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
                    dl.Tooltip(
                        html.Div([
                            html.B(f"New Station {next_id}"),
                            html.Br(),
                            html.Span(f"{lat:.4f}, {lon:.4f}")
                        ])
                    ),
                    dl.Popup(
                        html.Div([
                            html.B(f"New Station {next_id}"),
                            html.Br(),
                            html.Span(f"{lat:.4f}, {lon:.4f}")
                        ])
                    )
                ]
            ))
    
    return markers, geojson, station_ids

def make_incident_bar_plot(incident_counts_dict):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(incident_counts_dict.keys()),
        y=list(incident_counts_dict.values()),
        marker_color='indianred',
        name='Incident Count'
    ))

    fig.update_layout(
        title="Incident Counts per Station",
        xaxis_title="Station",
        yaxis_title="Incident Count",
        template="plotly_white",
        xaxis=dict(
            tickangle=-90,
            automargin=True  # ensures labels don't get cut off
        ),
        margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)
