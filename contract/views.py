from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from aip import AipOcr
from common.contract import split
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
    return render(request, 'contract/index.html')


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

    part1_result = {}
    for idx, words in enumerate(all_words):
        if re.match(r'买方', words):
            part1_result['买方'] = ' '.join([words.split(':')[-1], all_words[idx + 2], all_words[idx + 4]])
        elif re.match(r'合同号', words):
            part1_result['合同号'] = words.split(':')[-1]
        elif re.match(r'签署地', words):
            part1_result['签署地'] = words.split(':')[-1]
        elif re.match(r'日期', words):
            part1_result['日期'] = words.split(':')[-1]
        elif re.match(r'卖方', words):
            part1_result['卖方'] = ' '.join(
                [words.split(':')[-1], all_words[idx + 1], all_words[idx + 2], all_words[idx + 3]])
    # 处理 part2
    ## 处理 item1234
    image = get_file_content(os.path.join(media_dir, 'item1234.jpg'))
    item1234_scan_result = client.basicGeneral(image)
    all_words = []
    for item in item1234_scan_result['words_result']:
        all_words.append(item['words'])
    part2_result = {}
    part2_result['1货物描述'] = all_words[0]
    part2_result['2数量'] = all_words[1]
    part2_result['3单价'] = all_words[2]
    part2_result['4总价'] = all_words[3]
    ## 处理 item6789 和 item9101112
    for image_file in ['item5678.jpg', 'item9101112.jpg']:
        image = get_file_content(os.path.join(media_dir, image_file))
        item_scan_result = client.basicGeneral(image)
        all_words = []
        for item in item_scan_result['words_result']:
            all_words.append(item['words'])
        for idx, words in enumerate(all_words):
            if re.match(r'^\d', words):
                if ':' in words:
                    if words.endswith(':'):
                        part2_result[words] = all_words[idx + 1]
                    else:
                        key, value = words.split(':')
                        part2_result[key] = value
                elif ';' in words:
                    if words.endswith(':'):
                        part2_result[words] = all_words[idx + 1]
                    else:
                        key, value = words.split(';')
                        part2_result[key] = value
                else:
                    part2_result[words] = ''
    return JsonResponse(data={
        'part1_result': part1_result,
        'part2_result': part2_result
    })
