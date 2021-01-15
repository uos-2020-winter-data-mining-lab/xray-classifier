import os
import json
import numpy as np
from collections import defaultdict
from app.hans.config import CATEGORIES


def read_file(file):
    filepath = os.path.join('data', f'{file}.txt')
    print(f">> Read {file} File '{filepath}'")

    read_txt_file = {
        'annotation': read_annotation_file,
        'classes': read_categories_file,
        'anchors': read_anchors_file
    }[file]

    return read_txt_file(filepath)


def read_annotation_file(txtpath):
    if not os.path.exists(txtpath):
        print("  >>  Annotation file is not found, annotating... ")
        annotating()

    with open(txtpath) as f:
        lines = f.readlines()

    return lines


def read_categories_file(txtpath):
    if not os.path.exists(txtpath):
        print("  >>  Classes file is not found, categoring... ")
        with open(txtpath, mode='w') as f:
            for cat in CATEGORIES:
                f.write(f'{cat}\n')

    class_names = get_classes(txtpath)
    num_classes = len(class_names)
    return class_names, num_classes


def read_anchors_file(txtpath):
    anchors = get_anchors(txtpath)

    return anchors


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def annotating():
    name_box_id = defaultdict(list)
    # id_name = dict()
    f = open(
        "data/raw/label/Train/CoCo/coco_astrophysics.json",
        encoding='utf-8')
    data = json.load(f)

    images = data['images']
    image_dict = dict()
    for img in images:
        image_path = img['path'].replace("D:\\view\\", "data\\raw\\dataset\\")
        image_dict[img['id']] = image_path

    categories = data['categories']
    category_dict = dict()
    for cat in categories:
        category_dict[cat['id']] = cat['name']

    annotations = data['annotations']
    for ant in annotations:
        id = ant['image_id']
        path = image_dict[id]

        cat_id = ant['category_id']
        cat_name = category_dict[cat_id]

        if cat_name not in CATEGORIES:
            continue

        cat_id = {
            'Aerosol': 1,   # Aerosol
            'GunParts': 2,  # Gunparts
            'Liquid': 3,    # Liquid
            'Scissors': 4,  # Scissors
            'USB': 5,       # USB
        }[cat_name]

        path = image_dict[id]
        if '\\Single_Default\\' not in path:
            continue

        name_box_id[path].append([ant['bbox'], cat_id])

    write_annotation_file(name_box_id)


def write_annotation_file(name_box_id):
    f = open('data/annotation.txt', 'w')
    for key in name_box_id.keys():
        f.write(key)
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])

            box_info = f"{x_min},{y_min},{x_max},{y_max},{int(info[1])}"
            f.write(box_info)
        f.write('\n')
    f.close()
