from django.shortcuts import render
import os
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from wand.image import Image
from .ocr import scan
import datetime
import cv2


def index(request):
    return render(request, 'irf/index.html')


def pdf2img(pdf_filepath, img_filepath):
    with Image(filename=pdf_filepath + '[0]', resolution=300) as img:
        print('width =', img.width)
        print('height =', img.height)
        print('pages = ', len(img.sequence))
        print('resolution = ', img.resolution)
        with img.convert('jpg') as converted:
            converted.save(filename=img_filepath)


# Create your views here.
@csrf_exempt
def upload(request):
    """
    upload_v3: align_v3 returns aligned image and ocr_v3 accepts an image
    :param request:
    :return:
    """
    print('Requests starts at {}'.format(datetime.datetime.now()))
    file = request.FILES['file']
    fs = FileSystemStorage()
    src_file_name = file.name
    src_file_name_noext, ext = os.path.splitext(src_file_name)
    dst_file_name = src_file_name_noext + str(datetime.datetime.now()) + ext
    print('dst_file_name={}'.format(dst_file_name))
    image_name = fs.save(dst_file_name, file)
    print('image_name={}'.format(image_name))
    image_path = fs.path(image_name)
    image = cv2.imread(image_path)
    print('\tScan starts at {}'.format(datetime.datetime.now()))
    scan_result = scan(image)
    print('\tScan ends at {}'.format(datetime.datetime.now()))
    print('Requests ends at {}'.format(datetime.datetime.now()))
    return JsonResponse(data=scan_result)
