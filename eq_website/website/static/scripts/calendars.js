function onChangeMaxCal(max_date){
    date_min.max = max_date;
}

function onChangeMinCal(min_date){
    if ('1950-01-01' < min_date) {
        date_max.min = min_date;
    }
    else {
        date_max.min = '1950-01-01';
    }
}