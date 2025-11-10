let pointsVisibility = false;
let layer = L.layerGroup().addTo(map);

function printPoints(points) {
    if (!pointsVisibility) {
        pointsVisibility = true;
        points.forEach((point) => {
            let marker = L.marker([point.lat, point.lon]).addTo(layer);
            marker.bindPopup(point.place);
        })
    }
    else {
        console.log("Warning: points already printed!")
    }
    return points;
}

function clearPoints() {
    if (pointsVisibility) {
        pointsVisibility = false;
        layer.clearLayers();
    }
    else {
        console.log("Warning: points already cleared!")
    }
}

// Stąd w dół dzieje się magia o_O
// Notka do siebie:
// Następnym razem nie tykać rzeczy,
// o których się nie ma pojęcia w ważnej pracy z nieprzesuwalnym terminem
async function getPoints() {
    const url = new URL("http://127.0.0.1:8000/api/eqs");
    url.search = new URLSearchParams({
        minmagnitude: 5
    });
    let response = await fetch(url, {});
    points = await response.json()
    return points;
}

points = [];
getPoints().then(printPoints);

// This happens instantly
// So the result is []
console.log(points);

// This happens after 1 second
// So the result is what we want
setTimeout(() => {console.log(points)}, 1000);