const offcanvasElementList = document.querySelectorAll('.offcanvas')
const offcanvasList = [...offcanvasElementList].map(
    offcanvasEl => new bootstrap.Offcanvas(offcanvasEl)
)

async function callPrediction(e) {
    offcanvasList.forEach((e) => {e.hide()});

    let rect = e.sourceTarget._bounds
    let x = (rect._northEast.lng + rect._southWest.lng)/2;
    let y = (rect._northEast.lat + rect._southWest.lat)/2;

    const div = document.getElementById("detail-div");

    div.innerHTML =
        '<h5 class="mb-1">Prediction</h5>' +
        '<p class="mb-1">' + formatLon(x) + ' ' + formatLat(y) + '</p>'

    let predictionDiv = document.createElement("div");
    predictionDiv.classList.add("d-flex");
    predictionDiv.classList.add("justify-content-center");
    predictionDiv.innerHTML =
        '<div class="spinner-border text-dark m-3" role="status">\n' +
        '  <span class="visually-hidden">Loading...</span>\n' +
        '</div>'

    div.appendChild(predictionDiv);

    detailsContainer.style.display = 'block';
    closeButtonContainer.style.display = 'block';

    const url = new URL("http://127.0.0.1:8000/api/predict");
    url.search = new URLSearchParams({
        x: x,
        y: y,
        model: document.getElementById("modelSelect").value,
        data: document.querySelector('input[name="predRadio"]:checked').value
    });
    let response = await fetch(url, {});

    if (response.status!==200) {
        if (response.status===206) {
            div.innerHTML =
            '<h5 class="mb-1">Cannot perform a prediction.</h5>' +
            '<p class="mb-1">There were not enough events registered is the last month to perform miningful prediction.</p>'
            return
        }

        div.innerHTML =
        '<h5 class="mb-1">Something went wrong!</h5>' +
        '<p class="mb-1">Please try again or check error messages.</p>'
        return
    }

    let prediction = parseFloat(await response.json());
    predictionDiv.innerHTML =
    '<div class="prediction-result bg-dark mb-1">' +
        (prediction*100).toFixed(2) + "%" +
    '</div>'

    if (prediction>0.25) {
        predictionDiv.classList.add("text-danger");
    }
    else if (prediction>0.15) {
        predictionDiv.classList.add("text-warning");
    }
    else {
        predictionDiv.classList.add("text-light");
    }
}