from django.shortcuts import render
import os
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from wand.image import Image
from .ocr_v3 import scan
from .align_v3 import align
import datetime


def index(request):
    return render(request, 'ldf/index.html')


def pdf2img(pdf_filepath, img_filepath):
    with Image(filename=pdf_filepath, resolution=300) as img:
        print('width =', img.width)
        print('height =', img.height)
        print('pages = ', len(img.sequence))
        print('resolution = ', img.resolution)
        with img.convert('jpg') as converted:
            converted.save(filename=img_filepath)


# @csrf_exempt
# def upload(request):
#     """
#     upload_v2: align_v2 writes aligned image to a file and ocr_v2 accepts an image
#     :param request:
#     :return:
#     """
#     print('Starts at {}'.format(datetime.datetime.now()))
#     file = request.FILES['file']
#     fs = FileSystemStorage()
#     print(file.name)
#     imgname = fs.save(file.name, file)
#     imgpath = fs.path(imgname)
#     # imgname = os.path.splitext(pdfname)[0] + '.jpg'
#     media_dir = os.path.dirname(imgpath)
#     # imgpath = os.path.join(media_dir, imgname)
#     # pdf2img(pdfpath, imgpath)
#     aligned_imgname = 'aligned_' + imgname
#     aligned_imgpath = os.path.join(media_dir, aligned_imgname)
#     align(imgpath, os.path.join(media_dir, 'sh-0.jpg'), aligned_imgpath)
#     scan_result = scan(aligned_imgpath)
#     print('Ends at {}'.format(datetime.datetime.now()))
#     return JsonResponse(data=scan_result)

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
    media_dir = os.path.dirname(image_path)
    print('\tAlignment starts at {}'.format(datetime.datetime.now()))
    aligned_image = align(image_path, os.path.join(media_dir, 'sh-0.jpg'))
    print('\tAlignment ends at {}'.format(datetime.datetime.now()))
    print('\tScan starts at {}'.format(datetime.datetime.now()))
    scan_result = scan(aligned_image)
    print('\tScan ends at {}'.format(datetime.datetime.now()))
    print('Requests ends at {}'.format(datetime.datetime.now()))
    return JsonResponse(data=scan_result)
