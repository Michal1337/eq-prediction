from django.contrib import admin
from .models import CountriesDF, Earthquake, Prediction
from django.contrib.gis.admin import OSMGeoAdmin
from django.contrib.gis.geos import Point


class EarthquakeAdmin(OSMGeoAdmin):
    list_display = ("place", "type", "time", "longitude", "latitude", "mag")
    search_fields = ['place']

    exclude = ('location',)

    def longitude(self, instance):
        return instance.location.x if instance.location else None

    longitude.short_description = 'Longitude'

    def latitude(self, instance):
        return instance.location.y if instance.location else None

    latitude.short_description = 'Latitude'

    def save_model(self, request, obj, form, change):
        # If lon and lat are given, set the location field
        if obj.longitude is not None and obj.latitude is not None:
            obj.location = Point(float(obj.longitude), float(obj.latitude))
        super().save_model(request, obj, form, change)


admin.site.register(Earthquake, EarthquakeAdmin)


class CountriesDFAdmin(admin.ModelAdmin):
    list_display = ("name", "continent", "lon", "lat")


admin.site.register(CountriesDF, CountriesDFAdmin)


class PredictionAdmin(admin.ModelAdmin):
    list_display = ("name", "accuracy", "f1")


admin.site.register(Prediction, PredictionAdmin)
