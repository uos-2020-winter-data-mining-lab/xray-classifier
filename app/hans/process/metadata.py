import os
import json
import numpy as np
from collections import defaultdict
from app.hans.config import CATEGORIES


def read_meta_files(meta_dir):
    if not os.path.exists(meta_dir):
        os.mkdir(meta_dir)

    annotation_path = os.path.join(meta_dir, 'annotation.txt')
    categories_path = os.path.join(meta_dir, 'classes.txt')
    anchors_path = os.path.join(meta_dir, 'anchors.txt')

    data = get_data(annotation_path)
    classes = get_classes(categories_path)
    num_classes = len(classes)
    anchors = get_anchors(anchors_path)

    return data, num_classes, anchors


def get_data(data_path):
    '''loads the annotations from a file'''
    if not os.path.exists(data_path):
        annotating(data_path)

    with open(data_path) as f:
        data = f.readlines()
    return data


def get_classes(classes_path):
    '''loads the classes from a file'''
    if not os.path.exists(classes_path):
        categorying(classes_path)

    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    if not os.path.exists(anchors_path):
        anchoring(anchors_path)

    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def annotating(filepath):
    with open("data/raw/label/Train/CoCo/coco_astrophysics.json") as f:
        data = json.load(f)

    # Image dict
    images = data['images']
    image_path_dict = dict()
    for img in images:
        image_path = img['path'].replace("D:\\view\\", "data\\raw\\dataset\\")
        image_path_dict[img['id']] = image_path

    # Categories dict
    categories = data['categories']
    category_name_dict = dict()
    for cat in categories:
        category_name_dict[cat['id']] = cat['name']

    # Annotations dict
    name_box_id = defaultdict(list)
    annotations = data['annotations']
    for ant in annotations:
        img_id = ant['image_id']
        img_path = image_path_dict[img_id]
        if '\\Single_Default\\' not in img_path:
            continue

        cat_id = ant['category_id']
        cat_name = category_name_dict[cat_id]
        if cat_name not in CATEGORIES:
            continue
        cat_id = CATEGORIES.index(cat_name)

        name_box_id[img_path].append([ant['bbox'], cat_id])

    write_annotation_file(name_box_id, filepath)


def write_annotation_file(name_box_id, filepath):
    with open(filepath, mode='w') as f:
        for key in name_box_id.keys():
            f.write(key)
            box_infos = name_box_id[key]
            for info in box_infos:
                x_min = int(info[0][0])
                y_min = int(info[0][1])
                x_max = x_min + int(info[0][2])
                y_max = y_min + int(info[0][3])

                box_info = f" {x_min},{y_min},{x_max},{y_max},{int(info[1])}"
                f.write(box_info)
            f.write('\n')


def categorying(filepath):
    with open(filepath, mode='w') as f:
        for cat in CATEGORIES:
            f.write(f'{cat}\n')


def anchoring(filepath):
    with open(filepath, mode='w') as f:
        f.write(
            "10,13,  16,30,  33,23,  "
            "30,61,  62,45,  59,119,  "
            "116,90,  156,198,  373,326"
        )
