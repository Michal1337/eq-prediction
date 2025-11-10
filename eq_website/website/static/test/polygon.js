var polygon = L.polygon([
    [13, -82],
    [13, -34],
    [-56, -34],
    [-56, -82]
]);

function changeApiBoundary(radio) {
    if (radio.value === "show") {
        polygon.addTo(map);
    }
    else if (radio.value === "hide") {
        map.removeLayer(polygon);
    }
}