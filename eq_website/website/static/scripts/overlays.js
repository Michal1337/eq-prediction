function makePolyline(coords) {
    let points = [];
    for (pt of coords) {
        points.push(new L.LatLng(pt.lat, pt.lon));
    }
    return points;
}

function togglePlates(radio) {
    if (radio.value === "show") {
        plates.addTo(map);
    }
    else if (radio.value === "hide") {
        map.removeLayer(plates);
    }
}

function toggleRegions(radio) {
    if (radio.value === "show") {
        regions.addTo(map);
    }
    else if (radio.value === "hide") {
        map.removeLayer(regions);
    }
}

let platesData = JSON.parse(document.getElementById("plates_json").textContent);
let linesData = [];
for (const [key, value] of Object.entries(platesData)) {
    let style = {
        weight: 2,
        color: 'hotpink' //gray looks cool as well
    }
    let real = new L.Polyline(makePolyline(value.real), style);
    let west = new L.Polyline(makePolyline(value.west), style);
    let east = new L.Polyline(makePolyline(value.east), style);
    linesData.push(real);
    linesData.push(west);
    linesData.push(east);
}
var plates = L.layerGroup(linesData);

let regionsData = JSON.parse(document.getElementById("regions_json").textContent);
let polygonesData = [];
for (rect of regionsData) {
    polygonesData.push(L.rectangle(rect, {color: "red"}).on("click", callPrediction));
}
var regions = L.layerGroup(polygonesData);