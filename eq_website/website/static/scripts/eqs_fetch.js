var mag_min = document.getElementById('magMinSlider');
var mag_max = document.getElementById('magMaxSlider');
var date_min = document.getElementById('date_min');
var date_max = document.getElementById('date_max');
var depth_min = document.getElementById('depthMinSlider');
var depth_max = document.getElementById('depthMaxSlider');
var count_limit = document.getElementById('countMax');

async function getPoints() {
    const url = new URL("http://127.0.0.1:8000/api/eqs");

    var latlon_limits = map.getBounds();
    let lat_min = latlon_limits._southWest.lat
    let lon_min = latlon_limits._southWest.lng
    let lat_max = latlon_limits._northEast.lat
    let lon_max = latlon_limits._northEast.lng

    url.search = new URLSearchParams({
        minlatitude: lat_min,
        maxlatitude: lat_max,
        minlongitude: lon_min,
        maxlongitude: lon_max,
        starttime: date_min.value,
        endtime: date_max.value,
        minmagnitude: mag_min.value,
        maxmagnitude: mag_max.value,
        mindepth: depth_min.value,
        maxdepth: depth_max.value, 
        limcount: (count_limit.value === '' ? 100 :count_limit.value)
    });
    let response = await fetch(url, {});
    return await response.json();
}

// onclick button
async function filterEqs(){
    markersLayer.clearLayers();
    points = await getPoints();
    displayGeoJSONInfo(points);
    points.features.forEach(makeMarkers);
}