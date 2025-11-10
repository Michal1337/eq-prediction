from csv import DictReader

from django.core.management import BaseCommand
from map.models import CountriesDF

ALREADY_LOADED_ERROR_MESSAGE = """
This command is only used to populate empty table.
If you need to reload the data, first drop all data from CountriesDF table.
"""


class Command(BaseCommand):
    # Show this when the user types help
    help = "Loads data from countries_good.csv"

    def handle(self, *args, **options):

        # Show this if the data already exist in the database
        if CountriesDF.objects.exists():
            print('Countries data already loaded... exiting.')
            print(ALREADY_LOADED_ERROR_MESSAGE)
            return

        # Show this before loading the data into the database
        print("Loading countries data...")

        # Code to load the data into database
        for row in DictReader(open('../assets/data/countries_good.csv')):
            country = CountriesDF(name=row['Country'], continent=row['Continent'], lat=row['latitude'],
                                  lon=row['longitude'])
            country.save()
