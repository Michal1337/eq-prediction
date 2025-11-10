function resetMap() {
    map.setView([0, 0], 2);
}

function changePosition(lat, lon) {
    map.setView([lat, lon], 6);
}

function formatTimestamp(isoTimestamp) {
    let date = new Date(isoTimestamp);
    let options = {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    };
    return date.toLocaleString("en-US", options);
}

function formatLon(lon) {
    let long;
    if (lon > 0) {
      long = lon.toString() + " \u00B0E";
    } else if (lon < 0) {
      lon = -lon;
      long = lon.toString() + "\u00B0W";
    } else {
      long = lon.toString() + "\u00B0";
    }
    return long;
}

function formatLat(lat) {
    let latt;
    if (lat > 0) {
      latt = lat.toString() + " \u00B0N";
    } else if (lat < 0) {
      lat = -lat;
      latt = lat.toString() + "\u00B0S";
    } else {
      latt = lat.toString() + "\u00B0";
    }
    return latt;
}

function formatPinData(property) {
  if (property==null){
    property = "no data"
  }

  return property
}