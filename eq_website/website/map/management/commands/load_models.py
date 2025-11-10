import os

from django.core.management import BaseCommand
from django.core.files.base import File
from map.models import Prediction

ALREADY_LOADED_ERROR_MESSAGE = """
This command is only used to populate empty table.
If you need to reload the data, first drop all data from Prediction table.
"""


class Command(BaseCommand):
    help = "Loads all models from \'assets/models\' into the database."

    def handle(self, *args, **options):
        if Prediction.objects.exists():
            print('Models already loaded... exiting.')
            print(ALREADY_LOADED_ERROR_MESSAGE)
            return

        print("Loading models...")
        for curr in os.listdir('../assets/models'):
            if os.path.isdir(os.path.join('../assets/models', curr)):
                print(f"Loading {curr}")
                model = Prediction(
                            name=curr,
                            file=os.path.join('../assets/models', curr)
                        )
                model.save()
