function controlFromSlider(fromSlider, toSlider) {
  const [from, to] = getParsed(fromSlider, toSlider);
  fillSlider(fromSlider, toSlider, '#C6C6C6', '#25daa5', toSlider);
  if (from > to) {
    fromSlider.value = to;
  }
}

function controlToSlider(fromSlider, toSlider) {
  const [from, to] = getParsed(fromSlider, toSlider);
  fillSlider(fromSlider, toSlider, '#C6C6C6', '#25daa5', toSlider);
  setToggleAccessible(toSlider);
  if (from <= to) {
    toSlider.value = to;
  } else {
    toSlider.value = from;
  }
}

function getParsed(currentFrom, currentTo) {
  const from = parseInt(currentFrom.value, 10);
  const to = parseInt(currentTo.value, 10);
  return [from, to];
}

function fillSlider(from, to, sliderColor, rangeColor, controlSlider) {
    const rangeDistance = to.max-to.min;
    const fromPosition = from.value - to.min;
    const toPosition = to.value - to.min;
    controlSlider.style.background = `linear-gradient(
      to right,
      ${sliderColor} 0%,
      ${sliderColor} ${(fromPosition)/(rangeDistance)*100}%,
      ${rangeColor} ${((fromPosition)/(rangeDistance))*100}%,
      ${rangeColor} ${(toPosition)/(rangeDistance)*100}%, 
      ${sliderColor} ${(toPosition)/(rangeDistance)*100}%, 
      ${sliderColor} 100%)`;
}

function setToggleAccessible(currentTarget) {
  //const toSlider = document.querySelector('#toSlider');
  if (Number(currentTarget.value) <= 0 ) {
    currentTarget.style.zIndex = 2;
  } else {
    currentTarget.style.zIndex = 0;
  }
}

// space = len of number to fill
function createDatalistOptions(datalistDiv, fromSlider, inc, space) {
    const min = parseInt(fromSlider.getAttribute("min"), 10);
    const max = parseInt(fromSlider.getAttribute("max"), 10);
    const wid = datalistDiv.clientWidth/((max+1)/inc);

    const minOption = document.createElement('option');
    minOption.value = min;
    //minOption.label = '<'+min+' ';
    minOption.label = '<'+min;
    minOption.style.width = wid + 'px';
    minOption.style.textAlignLast = 'center';
    datalistDiv.appendChild(minOption);

    for (var i = min+inc; i <= max-inc; i+=inc) {
        const option = document.createElement('option');
        option.value = i;
        //option.label = String(i).padStart(space, " ");
        option.label = i;
        option.style.width = wid + 'px';
        option.style.textAlignLast = 'center';
        datalistDiv.appendChild(option);
    }

    const maxOption = document.createElement('option');
    maxOption.value = max;
    maxOption.label = '>'+max;
    maxOption.style.width = wid + 'px';
    maxOption.style.textAlignLast = 'center';
    datalistDiv.appendChild(maxOption);
}

// magnitude utils
const magMinSlider = document.getElementById('magMinSlider');
const magMaxSlider = document.getElementById('magMaxSlider');
const magDatalist = document.getElementById('magValues');

fillSlider(magMinSlider, magMaxSlider, '#C6C6C6', '#25daa5', magMaxSlider);
setToggleAccessible(magMaxSlider);
createDatalistOptions(magDatalist, magMinSlider, 1, 3);

magMinSlider.oninput = () => controlFromSlider(magMinSlider, magMaxSlider);
magMaxSlider.oninput = () => controlToSlider(magMinSlider, magMaxSlider);

// depth utils
const depthMinSlider = document.getElementById('depthMinSlider');
const depthMaxSlider = document.getElementById('depthMaxSlider');
const depthValues = document.getElementById('depthValues');

fillSlider(depthMinSlider, depthMaxSlider, '#C6C6C6', '#25daa5', depthMaxSlider);
setToggleAccessible(depthMaxSlider);
createDatalistOptions(depthValues, depthMinSlider, 25, 4);

depthMinSlider.oninput = () => controlFromSlider(depthMinSlider, depthMaxSlider);
depthMaxSlider.oninput = () => controlToSlider(depthMinSlider, depthMaxSlider);