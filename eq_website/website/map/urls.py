from django.urls import path, include
from . import views
from . import test_views

api_patterns = [
    path('eqs', views.api_eqs, name='eqs_api'),
    path('predict', views.api_predict, name='predict_api')
]

test_patterns = [
    path('map', test_views.map, name='map_test'),
    path('eq_data', test_views.eq_data, name='eq_data_test'),
    path('prediction', test_views.prediction, name='prediction_test'),
    path('api', test_views.api, name='api_test')
]

urlpatterns = [
    path('', views.home, name='home'),
    path('app/', views.app, name='app'),
    path('test/', include(test_patterns)),
    path('api/', include(api_patterns))
]
