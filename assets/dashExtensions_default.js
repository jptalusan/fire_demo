window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature, latlng, context) {
            const circleOptions = {
                ...context.hideout.circleOptions
            }; // clone to avoid mutation
            const iid = feature.properties.incident_id;
            const sid = feature.properties.station_id;
            const colorMap = context.hideout.color_map;

            // Assign color based on station_id
            circleOptions.fillColor = colorMap[sid] || "#ff0000"; // fallback color if not in color_map

            const marker = L.circleMarker(latlng, circleOptions);

            // Tooltip content from feature properties
            const tooltipText = "Incident: " + (iid !== undefined ? iid : "N/A") + "<br>Station ID: " + (sid !== undefined ? sid : "N/A");
            marker.bindTooltip(tooltipText);

            return marker;
        },
        function1: function(feature, context) {
            const id = feature.properties.station_id;
            const colorMap = context.hideout.colorMap;

            // If station_id is -1, remove the fill color and opacity
            if (id === -1) {
                return {
                    color: "#333", // Outline color
                    weight: 0.5, // Outline weight
                    fillOpacity: 0 // No fill
                };
            }

            // If station_id is not -1, use the color map
            return {
                color: "#333", // Outline color
                weight: 0.5, // Outline weight
                fillColor: colorMap[id] || "#999999", // fallback color
                fillOpacity: 0.5 // Fill opacity
            };
        },
        function2: function(feature, layer) {
            if (feature.properties && feature.properties.FacilityName) {
                layer.bindTooltip(feature.properties.FacilityName, {
                    permanent: true,
                    direction: "top",
                    className: "station-label"
                }).openTooltip(); // force open in case it's not triggered
            }
        },
        function3: function(feature, layer) {
            if (feature.properties && feature.properties.incident_id) {
                layer.bindTooltip(feature.properties.incident_id);
            }
        },
        function4: function(feature, layer) {
            if (feature.properties && feature.properties.station_id) {
                const cell_id = feature.properties.cell_id;
                const station_id = feature.properties.station_id;
                layer.bindTooltip("Cell ID: " + cell_id + "<br>Station ID: " + station_id);
            } else {
                layer.bindTooltip("No Station Assigned");
            }
        }
    }
});