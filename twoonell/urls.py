from django.conf.urls import url, include
from django.contrib import admin

from .views import index, upload
urlpatterns = [
    url(r'^$', index),
    url(r'^upload$', upload),
]