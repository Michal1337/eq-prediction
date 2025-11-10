// init map

const map = L.map('map', {
    zoomControl: false
}).setView([0, 0], 2);

L.control.zoom({
    position: 'bottomright'
}).addTo(map);

L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
}).addTo(map);

L.grid().addTo(map);

var points = JSON.parse(document.getElementById("points_json").textContent);
