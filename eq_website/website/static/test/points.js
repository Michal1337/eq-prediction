let points = JSON.parse(document.getElementById("points_json").textContent);

points.features.forEach(
    (point) => {
        let marker = L.marker([point.geometry.coordinates[1], point.geometry.coordinates[0]]).addTo(map);
        marker.bindPopup(point.place);
    }
);