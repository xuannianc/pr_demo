from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
# Create your views here.
from aip import AipOcr
from common.invoice import split
from django.http.response import JsonResponse
import os
import re
import numpy as np
import pandas as pd
import cv2

APP_ID = '11172427'
API_KEY = '6KUoA2v1eVXRAIcVXnOtrwlA'
SECRET_KEY = 'bMGaeRNO2fsPS0Tpl7HozcGDclvQByYm'
client = AipOcr(APP_ID, API_KEY, SECRET_KEY)


def index(request):
    return render(request, 'invoice/index.html')


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


@csrf_exempt
def upload(request):
    file = request.FILES['file']
    fs = FileSystemStorage()
    filename = fs.save(file.name, file)
    filepath = fs.path(filename)
    split(filepath)
    media_dir = os.path.dirname(filepath)
    scan_result = {}
    # 处理 exporter,consignee,notify party,buyer
    for field in ['exporter', 'consignee', 'notify_party', 'buyer']:
        field_path = os.path.join(media_dir, field + '.jpg')
        image = get_file_content(field_path)
        item_scan_result = client.basicGeneral(image)
        all_words = []
        for item in item_scan_result['words_result']:
            all_words.append(item['words'])
        scan_result[field.upper()] = all_words[1:]
    # 处理 invoice number
    details_path = os.path.join(media_dir, 'details.jpg')
    image = get_file_content(details_path)
    details_scan_result = client.basicGeneral(image)
    all_words = []
    details = {}
    for item in details_scan_result['words_result']:
        all_words.append(item['words'])
    del all_words[8]
    for idx in range(0, len(all_words), 2):
        details[all_words[idx]] = all_words[idx + 1]
    scan_result['details'] = details
    return JsonResponse(data=scan_result)