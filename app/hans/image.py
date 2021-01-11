import os
import numpy as np
from app.hans.config import WIDTH, HEIGHT, RATIO
from PIL import Image


def preprocess_data(source_dir):
    for category in os.listdir(source_dir):
        print("category >> ", category)
        try:
            preprocess_image(source_dir, category)
        except KeyboardInterrupt:
            print("Abort !")
            raise


def preprocess_image(base_path, category):
    source_path = os.path.join(base_path, category)

    for data_dir in os.listdir(source_path):
        data_path = os.path.join(source_path, data_dir)
        for data in os.listdir(data_path):
            image_path = os.path.join(data_path, data)
            image = Image.open(image_path)
            pixel = np.asarray(image, dtype=np.uint8)
            new_image = crop_resize(pixel)
            new_image = new_image.resize((WIDTH//RATIO, HEIGHT//RATIO))
            new_image.save(image_path)


def crop_resize(pixel):
    ymin, ymax, xmin, xmax = bbox(pixel[50:HEIGHT - 100, 50:WIDTH-50] < 215)

    pixel = pixel[ymin:ymax, xmin:xmax]

    return Image.fromarray(pixel)


def re_ratio_image(image, ratio=1):
    w, h = image.width, image.height

    fixedW, fixedH = w, h
    if w/h >= ratio:
        fixedH = w / ratio
    else:
        fixedW = h * ratio
    fixedW, fixedH = int(fixedW), int(fixedH)

    new_image = Image.new("RGB", (fixedW, fixedH))
    new_image.paste(image, ((fixedW - w)//2, (fixedH - h)//2))

    return new_image


def bbox(image):
    rows = np.any(image, axis=1)
    cols = np.any(image, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax
