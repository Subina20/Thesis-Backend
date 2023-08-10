
from django.urls import path
from . import views

urlpatterns = [
    path('skin_care_recommendations/', views.skin_care_recommendations, name='skin_care_recommendations'),
]
