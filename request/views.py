from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
# Create your views here.
from aip import AipOcr
from common.request import preprocessing, preprocessing2, correction
from django.http.response import JsonResponse
import os
import re
import numpy as np
import pandas as pd

APP_ID = '11172427'
API_KEY = '6KUoA2v1eVXRAIcVXnOtrwlA'
SECRET_KEY = 'bMGaeRNO2fsPS0Tpl7HozcGDclvQByYm'
client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

PART1_FIELDS = ['payment no', 'prepay', 'description', 'supplier', 'supplier name', 'currency', 'vat',
                'payment term', 'bu', 'requested date', 'alternate payee']


def index(request):
    return render(request, 'request/index.html')


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


def is_part1_field(words):
    # 没有数字
    if not re.search(r'\d', words):
        corrected_word_list = []
        for word in words.split(' '):
            corrected_word = correction(word)
            corrected_word_list.append(corrected_word)
        corrected_words = ' '.join(corrected_word_list)
        if corrected_words in PART1_FIELDS:
            return corrected_words, True
        else:
            return None, False
    return None, False


@csrf_exempt
def upload(request):
    file = request.FILES['file']
    fs = FileSystemStorage()
    filename = fs.save(file.name, file)
    filepath = fs.path(filename)
    row_num = preprocessing2(filepath)
    # print(fs.url(filename))
    media_dir = os.path.dirname(filepath)
    # 处理 part1
    part1_path = os.path.join(media_dir, 'request_part1.jpg')
    image = get_file_content(part1_path)
    part1_scan_result = client.basicGeneral(image)
    all_words = []
    for item in part1_scan_result['words_result']:
        all_words.append(item['words'])
    print(all_words)
    # all_fields = ['Payment No', 'Prepay', 'Description', 'Supplier', 'Supplier Name', 'Currency', 'VAT', 'Payment Term',
    #               'Requested Date', 'BU', 'Alternate Payee']
    part1_result = {}
    for idx, words in enumerate(all_words):
        corrected_words, is_true = is_part1_field(words)
        if is_true:
            if idx < len(all_words) - 1:
                corrected_words2, is_true2 = is_part1_field(all_words[idx + 1])
                if is_true2:
                    part1_result[corrected_words] = ''
                else:
                    part1_result[corrected_words] = all_words[idx + 1]
            else:
                part1_result[corrected_words] = ''
    # # 处理 part2
    # df = pd.DataFrame(columns=['PO No', 'Sub-Brand Description', 'PO Description', 'Category', 'PO Payment Amount',
    #                            'Deductable Tax Amount', 'Invoice Amount'])
    # ## column 0,1,3,4,5,6
    # for i in [j for j in range(7) if j != 2]:
    #     col_image = get_file_content(os.path.join(media_dir, "request_col" + str(i) + ".jpg"))
    # col_scan_result = client.basicGeneral(col_image)
    # all_words = []
    # for item in col_scan_result['words_result']:
    #     all_words.append(item['words'])
    # ### 补全未扫描的字段
    # if col_scan_result['words_result_num'] == row_num:
    #     all_words.extend([None] * (row_num - col_scan_result['words_result_num']))
    # df[df.columns[i]] = all_words
    # ## column 2
    # col2_image = get_file_content(os.path.join(media_dir, "request_col2.jpg"))
    # col2_scan_result = client.basicGeneral(col2_image)
    # all_words = []
    # all_description = []
    # for item in col2_scan_result['words_result']:
    #     all_words.append(item['words'])
    # splits = np.array_split(np.array(all_words), row_num)
    # for split in splits:
    #     all_description.append(' '.join(split.tolist()))
    # df[df.columns[2]] = all_description
    # # 处理 part3
    # part3_path = os.path.join(media_dir, 'request_part3.jpg')
    # part3_image = get_file_content(part3_path)
    # part3_scan_result = client.basicGeneral(part3_image)
    # all_words = []
    # for item in part3_scan_result['words_result']:
    #     all_words.append(item['words'])
    # part3_result = {}
    # for idx, words in enumerate(all_words):
    #     if re.match(r'^[Ii]?nput', words):
    #         part3_result['Input By'] = all_words[idx + 1]
    #     elif re.match(r'^[Bb]?udget', words):
    #         part3_result['Budget Owner'] = all_words[idx + 1]
    #     elif re.match(r'^[Ss]?tarted', words):
    #         part3_result['Started By'] = all_words[idx + 1]
    #     elif re.match(r'^[Aa]?pproved', words):
    #         part3_result['Approved By'] = all_words[idx + 1]
    #     elif re.match(r'^[Pp]?ending', words):
    #         part3_result['Pending By'] = all_words[idx + 1]
    return JsonResponse(data={'part1_result': part1_result})
    # 'part2_result': df.to_dict(),
    # 'part3_result': part3_result})


def handle_uploaded_file(f):
    with open('some/file/name.txt', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
