from django.urls import path
from .views import *

app_name = 'appWAR'
urlpatterns = [
    path('', WAR),
    path('practice/',WAR_practice),

]