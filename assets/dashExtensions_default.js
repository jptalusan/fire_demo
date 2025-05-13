window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature, latlng, context) {
            const circleOptions = context.hideout.circleOptions;
            const id = feature.properties.incident_id;
            const marker = L.circleMarker(latlng, circleOptions);

            // Tooltip content from feature properties
            const tooltipText = "Incident: " + (id !== undefined ? id : "N/A");
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
                layer.bindTooltip(feature.properties.FacilityName);
            }
        },
        function3: function(feature, layer) {
            if (feature.properties && feature.properties.incident_id) {
                layer.bindTooltip(feature.properties.incident_id);
            }
        }
    }
});