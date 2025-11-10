import time
import random

from django.shortcuts import render
from django.core.serializers import serialize
from django.http import JsonResponse, HttpResponse

from .models import CountriesDF, Earthquake, Prediction

import requests
import json
import tensorflow as tf

from modeling.interference import make_prediction


# Main app views
def home(request):
    return render(request, 'map/home.html')


def app(request):
    params = {
        "starttime": '2023-12-01',
        "endtime": '2023-12-31',
        'mindepth': 0,
        'maxdepth': 100,
        'minmagnitude': 2,
        'maxmagnitude': 7,
        'limcount': 100
    }

    r = requests.get('http://127.0.0.1:8000/api/eqs', params=params)
    context = {"points": r.json()}

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

    models = Prediction.objects.all()
    context['models'] = models

    return render(request, 'map/app.html', context=context)


# API views
def api_eqs(request):
    if request.method == 'GET':
        eqs = Earthquake.objects.all()

        args = {'minlatitude',
                'maxlatitude',
                'minlongitude',
                'maxlongitude',
                'starttime',
                'endtime',
                'mindepth',
                'maxdepth',
                'minmagnitude',
                'maxmagnitude',
                'limcount',
                'count'}

        # position filters
        if request.GET.get('minlatitude'):
            eqs = eqs.filter(latitude__gte=request.GET.get('minlatitude'))
        if request.GET.get('maxlatitude'):
            eqs = eqs.filter(latitude__lte=request.GET.get('maxlatitude'))
        if request.GET.get('minlongitude'):
            eqs = eqs.filter(longitude__gte=request.GET.get('minlongitude'))
        if request.GET.get('maxlongitude'):
            eqs = eqs.filter(longitude__lte=request.GET.get('maxlongitude'))

        # time filters
        if request.GET.get('starttime'):
            eqs = eqs.filter(time__gte=request.GET.get('starttime'))
        if request.GET.get('endtime'):
            eqs = eqs.filter(time__lte=request.GET.get('endtime'))

        
        # depth and magnitude filters
        if request.GET.get('mindepth'):
            eqs = eqs.filter(depth__gte=request.GET.get('mindepth'))
        if request.GET.get('maxdepth'):
            eqs = eqs.filter(depth__lte=request.GET.get('maxdepth'))
        if request.GET.get('minmagnitude'):
            eqs = eqs.filter(mag__gte=request.GET.get('minmagnitude'))
        if request.GET.get('maxmagnitude'):
            eqs = eqs.filter(mag__lte=request.GET.get('maxmagnitude'))

        # limit count
        limit = 20000
        if request.GET.get('limcount'):
            limit = int(request.GET.get('limcount'))

        if request.GET.get('count'):
            return JsonResponse({'count': len(eqs)}, safe=False)

        eqs = eqs.order_by("-mag")[:limit]

        geo_json = serialize("geojson", eqs, geometry_field="location",
                             fields=[
                                 'time',
                                 'depth',
                                 'mag',
                                 'magType',
                                 'place',
                                 'alert',
                                 'type',
                                 'cdi',
                                 'mmi',
                                 'felt',
                                 'sig'
                             ])
        return JsonResponse(json.loads(geo_json), safe=False)
    else:
        return HttpResponse(status=501)


def api_predict(request):
    if request.method == 'GET':
        if not request.GET.get('x'):
            return HttpResponse(status=400)
        if not request.GET.get('y'):
            return HttpResponse(status=400)

        data_source = 'usgs'
        if request.GET.get('data'):
            data_source = request.GET.get('data')

        if request.GET.get('model'):
            path = Prediction.objects.get(pk=request.GET.get('model')).file
        else:
            path = Prediction.objects.all()[0].file

        prediction = make_prediction(
            float(request.GET.get('x')),
            float(request.GET.get('y')),
            data_source,
            path
        )
        if type(prediction) is str:
            if prediction == "Wrong coords": return HttpResponse(prediction, status=400)
            return HttpResponse(prediction, status=206)
        return JsonResponse(float(prediction), safe=False)
    else:
        return HttpResponse(status=501)
