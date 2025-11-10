from django.contrib.gis.db import models
from django.contrib.gis.geos import Point

import datetime


class Earthquake(models.Model):
    time = models.DateTimeField()

    longitude = models.FloatField()
    latitude = models.FloatField()

    location = models.PointField(null=True, srid=4326, spatial_index=True, blank=True)

    depth = models.FloatField()
    mag = models.FloatField()
    magType = models.CharField(max_length=200, default='mwc', null=True)

    place = models.CharField(max_length=200, default='')

    alert = models.CharField(max_length=200, default='', null=True)
    type = models.CharField(max_length=200, default='', null=True)
    cdi = models.FloatField(null=True)
    mmi = models.FloatField(null=True)
    felt = models.IntegerField(null=True)
    sig = models.IntegerField(null=True)

    def save(self, *args, **kwargs):
        if self.longitude is not None and self.latitude is not None:
            self.location = Point(float(self.longitude), float(self.latitude))

        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.type} at {self.place}, {self.time.date()}"


class CountriesDF(models.Model):
    name = models.CharField(max_length=200)
    continent = models.CharField(max_length=200)
    lon = models.FloatField()
    lat = models.FloatField()

    def __str__(self):
        return f"Country: {self.name}, in {self.continent} at {self.lon}lon, {self.lat}lat"


class Prediction(models.Model):
    name = models.CharField(max_length=100, unique=True, default='', blank=True)
    last_update_timestamp = models.DateTimeField(auto_now_add=True, editable=False)
    file = models.FilePathField(null=False, path='../assets/models', allow_files=False, allow_folders=True)

    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1 = models.FloatField(null=True, blank=True)
    accuracy = models.FloatField(null=True, blank=True)

    def save(self, *args, **kwargs):
        self.last_update_timestamp = datetime.datetime.now()

        if self.name == '':
            self.name = f'Model_{self.id}'
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name
