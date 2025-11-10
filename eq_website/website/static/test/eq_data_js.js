var points = JSON.parse(document.getElementById("points_json").textContent);
var container = document.getElementById("eq_details_list");

function displayCoordinates(geojsonData, counter) {
  var coordinatesElement = document.getElementById("coordinates" + counter);
  coordinatesElement.textContent =
    "Latitude: " +
    geojsonData.geometry.coordinates[1] +
    ", Longitude: " +
    geojsonData.geometry.coordinates[0];
}

function formatTimestamp(isoTimestamp) {
  var date = new Date(isoTimestamp);
  var options = {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  };
  return date.toLocaleString("en-US", options);
}

function formatLon(lon) {
  var long = lon;
  if (lon > 0) {
    long = lon.toString() + " \u00B0E";
  } else if (lon < 0) {
    lon = -lon;
    long = lon.toString() + "\u00B0W";
  } else {
    long = lon.toString();
  }
  return long;
}

function formatLat(lat) {
  var latt = lat;
  if (lat > 0) {
    latt = lat.toString() + " \u00B0N";
  } else if (lat < 0) {
    lat = -lat;
    latt = lat.toString() + "\u00B0S";
  } else {
    latt = lat.toString();
  }
  return latt;
}

function displayGeoJSONInfo() {
  points.features.forEach(function (feature) {
    var div1 = document.createElement("div");
    div1.classList.add("list-group-item");
    div1.innerHTML =
      '<div class="d-flex justify-content-between"><h5 class="mb-1">' +
      feature.properties.place +
      "</h5><small>" +
      formatTimestamp(feature.properties.time) +
      "</small></div>" +
      '<p class="mb-1">The earthquake had magnitude of ' +
      feature.properties.mag +
      "</p>" +
      '<p class="mb-1">Other features of the earthquake: \n' +
      "<ul> <li>Alert (color-coded): " +
      feature.properties.alert +
      "</li>" +
      "<li>Type: " +
      feature.properties.type +
      "</li>" +
      "<li>Magnitude type: " +
      feature.properties.magType +
      "</li>" +
      "<li>Cdi: " +
      feature.properties.cdi +
      "</li>" +
      "<li>Mmi: " +
      feature.properties.mmi +
      "</li>" +
      "<li>Felt: " +
      feature.properties.felt +
      "</li>" +
      "<li>Significance: " +
      feature.properties.sig +
      "</li>" +
      "</ul>" +
      "</p>" +
      "<small>" +
      formatLon(feature.geometry.coordinates[0]) +
      " " +
      formatLat(feature.geometry.coordinates[1]) +
      " at depth " +
      feature.properties.depth +
      "</small>";

    container.appendChild(div1);
  });
}

document.addEventListener("DOMContentLoaded", displayGeoJSONInfo);
