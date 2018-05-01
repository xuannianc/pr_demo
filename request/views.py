from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
# Create your views here.
from aip import AipOcr
from common.request import blur_and_split
from django.http.response import JsonResponse
import os
import re
import numpy as np
import pandas as pd

APP_ID = '11172427'
API_KEY = '6KUoA2v1eVXRAIcVXnOtrwlA'
SECRET_KEY = 'bMGaeRNO2fsPS0Tpl7HozcGDclvQByYm'
client = AipOcr(APP_ID, API_KEY, SECRET_KEY)


def index(request):
    return render(request, 'request/index.html')


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


@csrf_exempt
def upload(request):
    file = request.FILES['file']
    fs = FileSystemStorage()
    filename = fs.save(file.name, file)
    filepath = fs.path(filename)
    blur_and_split(filepath)
    # print(fs.url(filename))
    media_dir = os.path.dirname(filepath)
    # 处理 part1
    part1_path = os.path.join(media_dir, 'part1.jpg')
    image = get_file_content(part1_path)
    """ 调用通用文字识别, 图片参数为本地图片 """
    part1_scan_result = client.basicGeneral(image)
    all_words = []
    for item in part1_scan_result['words_result']:
        all_words.append(item['words'])

    all_fields = ['Payment No', 'Prepay', 'Description', 'Supplier', 'Supplier Name', 'Currency', 'VAT', 'Payment Term',
                  'Requested Date', 'BU', 'Alternate Payee']
    part1_result = {}
    for idx, words in enumerate(all_words):
        if words in all_fields:
            if not all_words[idx + 1] in all_fields:
                part1_result[words] = all_words[idx + 1]
            else:
                part1_result[words] = ''
    # 处理 part2
    df = pd.DataFrame(columns=['PO No', 'Sub-Brand Description', 'PO Description', 'Category', 'PO Payment Amount',
                               'Deductable Tax Amount', 'Invoice Amount'])
    row_num = None
    for i in range(2):
        image = get_file_content(os.path.join(media_dir, 'col' + str(i) + ".jpg"))
        result = client.basicGeneral(image)
        all_words = []
        for item in result['words_result']:
            all_words.append(item['words'])
        row_num = len(all_words)
        df[df.columns[i]] = all_words

    image = get_file_content(os.path.join(media_dir, "col2.jpg"))
    result = client.basicGeneral(image)
    all_words = []
    all_description = []
    for item in result['words_result']:
        all_words.append(item['words'])
    splits = np.array_split(np.array(all_words), row_num)
    for split in splits:
        all_description.append(' '.join(split.tolist()))
    df[df.columns[2]] = all_description

    for i in range(3, 7):
        image = get_file_content(os.path.join(media_dir, 'col' + str(i) + ".jpg"))
        result = client.basicGeneral(image)
        all_words = []
        for item in result['words_result']:
            all_words.append(item['words'])
        df[df.columns[i]] = all_words
    # 处理 part3
    part3_path = os.path.join(media_dir, 'part3.jpg')
    image = get_file_content(part3_path)
    """ 调用通用文字识别, 图片参数为本地图片 """
    part3_scan_result = client.basicGeneral(image)
    all_words = []
    for item in part3_scan_result['words_result']:
        all_words.append(item['words'])
    part3_result = {}
    for idx, words in enumerate(all_words):
        if re.match(r'^[Ii]?nput', words):
            part3_result['Input By'] = all_words[idx + 1]
        elif re.match(r'^[Bb]?udget', words):
            part3_result['Budget Owner'] = all_words[idx + 1]
        elif re.match(r'^[Ss]?tarted', words):
            part3_result['Started By'] = all_words[idx + 1]
        elif re.match(r'^[Aa]?pproved', words):
            part3_result['Approved By'] = all_words[idx + 1]
        elif re.match(r'^[Pp]?ending', words):
            part3_result['Pending By'] = all_words[idx + 1]
    return JsonResponse(data={'part1_result': part1_result,
                              'part2_result': df.to_dict(),
                              'part3_result': part3_result})


def handle_uploaded_file(f):
    with open('some/file/name.txt', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
