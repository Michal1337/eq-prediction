from django.shortcuts import render
from django.core.serializers import serialize

from .models import CountriesDF, Earthquake, Prediction

import requests
import json

import numpy as np
import tensorflow as tf


def map(request):
    params = {
        'mindepth': 0,
        'maxdepth': 100,
        'minmagnitude': 2,
        'maxmagnitude': 7,
        'limcount': 100
    }

    r = requests.get('http://127.0.0.1:8000/api/eqs',
                     params=params)
    context = {
        "points": r.json()
    }

    with open('../assets/data/plates.json', 'r') as f:
        plates = json.load(f)
    context['plates'] = plates

    with open('../assets/data/regions.json', 'r') as f:
        regions = json.load(f)
    context['regions'] = regions

    countries = CountriesDF.objects.all()
    continents = CountriesDF.objects.order_by().values_list('continent', flat=True).distinct()
    continents = list(continents)
    context["countries"] = countries
    context["continents"] = continents

    return render(request, 'test/map.html', context=context)


def eq_data(request):
    eqs = Earthquake.objects.all().order_by("-mag")
    eqs = eqs[:10]
    print(eqs)
    geo_json = serialize("geojson", eqs, geometry_field="location",
                         fields=['time', 'depth', 'mag', 'magType', 'place', 'alert', 'type', 'cdi', 'mmi', 'felt',
                                 'sig'])

    context = {'pins': json.loads(geo_json)}
    return render(request, 'test/eq_data.html', context=context)


def prediction(request):
    model = Prediction.objects.filter(available=True)[0]

    model_input = np.random.rand(1, 32, 5)
    m = tf.keras.models.load_model(model.file.path)
    logits = m.predict(model_input)
    probabilities = tf.nn.softmax(logits)

    context = {
        'input': model_input,
        'model': model,
        'logits': logits,
        'prob': probabilities
    }
    return render(request, 'test/prediction.html', context=context)


def api(request):
    return render(request, 'test/api.html')
