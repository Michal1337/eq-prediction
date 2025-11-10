// markers layer
let markersLayer = L.layerGroup().addTo(map);

// create container for details
var detailsContainer = L.DomUtil.create('div', 'list-group list-group-flush');
// details container styling
detailsContainer.style.overflowY = "auto";
detailsContainer.style.maxHeight = "80vh";
detailsContainer.style.width = '300px';
detailsContainer.style.display = 'none';
// making div not click through
L.DomEvent.disableClickPropagation(detailsContainer);
L.DomEvent.disableScrollPropagation(detailsContainer);
const detailsControl = L.control({ position: 'topright' });

// create container for closing button
const closeButtonContainer = L.DomUtil.create('div', 'button-container');
// button container styling
closeButtonContainer.style.backgroundColor = "white"
closeButtonContainer.style.borderColor = "black"
closeButtonContainer.style.display = 'none';
// making div not click through
L.DomEvent.disableScrollPropagation(closeButtonContainer);
L.DomEvent.disableClickPropagation(closeButtonContainer);
const closeButton = L.DomUtil.create('button', 'btn-close');

// UI Layer - do not touch!!!
var uiLayer = L.layerGroup().addTo(map);

// start-up function - default display
function displayInfo() {
    const div = document.createElement("div");
    div.classList.add("list-group-item");
    div.setAttribute("id", "detail-div");
    div.innerHTML =
        '<h5 class="mb-1">Details of earthquake</h5><p class="mb-1">' +
        "</p>";

    detailsContainer.appendChild(div);
    closeButtonContainer.appendChild(closeButton);

    closeButton.addEventListener('click', function () {
        detailsContainer.style.display = 'none';
        closeButtonContainer.style.display = 'none';
    });
    
    // has to be here
    detailsControl.onAdd = function () {
        const container = L.DomUtil.create('div', 'combined-container');
        container.appendChild(closeButtonContainer);
        container.appendChild(detailsContainer);
        return container;
    };

    detailsControl.addTo(map);
}


function makeMarkers(pin) {
    const prop = pin.properties
    const geo = pin.geometry.coordinates

    let marker = L.marker([geo[1], geo[0]]).addTo(markersLayer);

    // add on-click marker - displaying details
    marker.on('click', function() {

        const div = document.getElementById("detail-div")

        div.innerHTML =
            '<h5 class="mb-1">Details of the earthquake</h5><p class="mb-1">' +
            prop.place +
            "</p>" +
            '<p class="mb-1">' + prop.time + "</p>" +
            '<p class="mb-1">' + formatLon(geo[0]) + " " + formatLat(geo[1]) + "</p>" +
            '<p class="mb-1">' +
                "<ul> " +
                    "<li>Magnitude: " + formatPinData(prop.mag) + "</li>" +
                    "<li>Depth: " + formatPinData(prop.depth) + "</li>" +
                    "<li>Alert (color-coded): " + formatPinData(prop.alert) + "</li>" +
                    "<li>Type: " + formatPinData(prop.type) + "</li>" +
                    "<li>Cdi: " + formatPinData(prop.cdi) + "</li>" +
                    "<li>Mmi: " + formatPinData(prop.mmi) + "</li>" +
                    "<li>Felt: " + formatPinData(prop.felt) + "</li>" +
                    "<li>Significance: " + formatPinData(prop.sig) + "</li>" +
                "</ul>" +
            "</p>";

        detailsContainer.style.display = 'block';
        closeButtonContainer.style.display = 'block';
    });

}

// Run main function
displayInfo();
points.features.forEach(makeMarkers);