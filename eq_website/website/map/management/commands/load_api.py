import datetime
import requests
import pandas as pd
from tqdm.contrib import tzip

from django.core.management import BaseCommand

from map.models import Earthquake

ALREADY_LOADED_ERROR_MESSAGE = """
If you need to reload the child data,
first delete the data in the database.
Then, run `python manage.py migrate` for a new empty
database with tables
"""


class Command(BaseCommand):
    # Show this when the user types help
    help = "Loads data from API to database"

    def add_arguments(self, parser):
        parser.add_argument('--start_date', nargs='*', type=str, help='Starting date (of data)', default=["2023-12-01"])
        parser.add_argument('--end_date', nargs='*', type=str, help='Ending date (of data)', default=["2023-12-31"])

    def handle(self, *args, **kwargs):

        # Show this if the data already exist in the database
        if Earthquake.objects.exists():
            print('Earthquakes data already loaded... Aborting.')
            print(ALREADY_LOADED_ERROR_MESSAGE)
            return

        # Show this before loading the data into the database
        print("Loading earthquake data")

        starttime = kwargs['start_date'][0]
        endtime = kwargs['end_date'][0]

        datarange = pd.date_range(starttime, endtime, freq='W').tolist()
        datarange = [str(x)[:10] for x in datarange]
        if datarange[0] != starttime:
            datarange = [starttime] + datarange
        if datarange[-1] != endtime:
            datarange = datarange + [endtime]

        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

        for start, end in tzip(datarange[:-1], datarange[1:]):
            params = {
                'format': 'geojson',
                'starttime': start,
                'endtime': end
            }
            resp = requests.get(url, params=params).json()['features']
            for row in resp:
                eq = row['properties']
                if eq['mag'] is not None and eq['place'] is not None:
                    new_eq = Earthquake(
                        time=datetime.datetime.fromtimestamp(eq['time']/1000),
                        longitude=row['geometry']['coordinates'][0],
                        latitude=row['geometry']['coordinates'][1],
                        depth=row['geometry']['coordinates'][2],
                        mag=eq['mag'],
                        magType=eq['magType'],
                        place=eq['place'],
                        alert=eq['alert'],
                        type=eq["type"],
                        cdi=eq['cdi'],
                        mmi=eq['mmi'],
                        felt=eq['felt'],
                        sig=eq['sig']
                    )
                    new_eq.save()
