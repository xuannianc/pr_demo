from django.conf.urls import url

from .views import index,upload
urlpatterns = [
    url(r'^$', index),
    url(r'^upload$', upload),
]