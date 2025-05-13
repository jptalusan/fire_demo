window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature, latlng, context) {
            const circleOptions = context.hideout.circleOptions;
            const id = feature.properties.station_id;
            circleOptions.fillColor = context.hideout.color_map[id] || "#ff0000"; // fallback color
            return L.circleMarker(latlng, circleOptions); // render a simple circle marker
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
        }
    }
});