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
            return {
                color: "#333",
                weight: 0.5,
                fillColor: colorMap[id] || "#999999", // fallback color
                fillOpacity: 0.5
            };
        }
    }
});