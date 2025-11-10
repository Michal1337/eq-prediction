// create container for eq list
const listContainer = L.DomUtil.create('div', 'list-group list-group-flush');
// styling the container
listContainer.style.overflowY = "scroll";
listContainer.style.maxHeight = "80vh";
listContainer.style.width = "300px";
// making div not click through
L.DomEvent.disableClickPropagation(listContainer);
L.DomEvent.disableScrollPropagation(listContainer);
const listControl = L.control({ position: 'topleft' });


// create container for button to show list
const buttonContainer = L.DomUtil.create('div', 'button-container');
// making div not click through
L.DomEvent.disableScrollPropagation(buttonContainer);
L.DomEvent.disableClickPropagation(buttonContainer);
const button = L.DomUtil.create('button', 'btn btn-dark w-100');
button.innerHTML = 'Earthquakes list';

var isListVisible = false;
listContainer.style.display = 'none';
button.addEventListener('click', function () {
    isListVisible = !isListVisible;
    listContainer.style.display = isListVisible ? 'block' : 'none';
});
buttonContainer.appendChild(button);

// has to be here
listControl.onAdd = function () {
    const container = L.DomUtil.create('div', 'combined-container');
    container.appendChild(buttonContainer);
    container.appendChild(listContainer);
    return container;
};

listControl.addTo(map);


// UI Layer - do not touch!!!
var uiLayer = L.layerGroup().addTo(map);

// main function
function displayGeoJSONInfo(pins) {
    listContainer.innerHTML="";

    if (Object.keys(points.features).length === 0) {
        let div = document.createElement("div");
        div.classList.add("list-group-item");
        div.innerHTML =
            '<h5 class="mb-1">No entries found</h5>' +
            '<p class="mb-1">Try using different filters.</p>'
        listContainer.appendChild(div);
        return;
    }

    pins.features.forEach(function (feature, counter) {
        const prop = feature.properties;
        const geo = feature.geometry.coordinates;

        let div = document.createElement("div");
        div.classList.add("list-group-item");
        div.style.cursor = "pointer";
        div.innerHTML =
            '<h5 class="mb-1">' + prop.place + '</h5>' +
            '<p class="mb-1">' + formatTimestamp(prop.time) + "</p>" +
            '<p class="mb-1">The earthquake had magnitude of ' + prop.mag + "</p>" +
            "<small>" + formatLon(geo[0]) + " " + formatLat(geo[1]) +
              " at depth " + prop.depth +
            "</small>";

        // Zoom when clicking on list item
        div.addEventListener("click", () => changePosition(geo[1], geo[0]));

        listContainer.appendChild(div);
    });
}

// Run main function
displayGeoJSONInfo(points);