var map = L.map('map', {
    zoomControl: false
}).setView([0, 0], 1);

L.control.zoom({
    position: 'bottomright'
}).addTo(map);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
}).addTo(map);

L.grid().addTo(map);

function resetMap() {
    map.setView([0, 0], 1);
}

function changePosition(lat, lon) {
    map.setView([lat, lon], 6);
}

