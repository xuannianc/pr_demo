import os
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from wand.image import Image
from .ocr import scan


def index(request):
    return render(request, 'twoonell/index.html')


def pdf2img(pdf_filepath, img_filepath):
    with Image(filename=pdf_filepath, resolution=300) as img:
        print('width =', img.width)
        print('height =', img.height)
        print('pages = ', len(img.sequence))
        print('resolution = ', img.resolution)
        with img.convert('jpg') as converted:
            converted.save(filename=img_filepath)


@csrf_exempt
def upload(request):
    file = request.FILES['file']
    fs = FileSystemStorage()
    pdf_filename = fs.save(file.name, file)
    pdf_filepath = fs.path(pdf_filename)
    img_filename = os.path.splitext(pdf_filename)[0] + '.jpg'
    media_dir = os.path.dirname(pdf_filepath)
    img_filepath = os.path.join(media_dir, img_filename)
    pdf2img(pdf_filepath, img_filepath)
    scan_result = scan(img_filepath)
    return JsonResponse(data=scan_result)
