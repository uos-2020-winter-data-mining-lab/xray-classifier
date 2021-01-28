"""
Load the Dataset for train YoLo model
"""
import os
import cv2
import json
import numpy as np
import pickle
from app.hans.anchors import get_anchors


def load_data(
    coco_dir,
    image_dir,
    resize_dir=None,
    pkl_file=None,
    split_rate=0.8,
    save_resize=None,
    given_labels=None,
):
    """
    Parameters
        coco_dir      : Path where CoCo foramat annotation files are stored
        image_dir     : Path where actual image files are stored
        resized_dir * : Path where processed image files are stored
        pkl_file      : Path where preprocessed data file(*.pkl file)
        split_rate    : Training data and data data split ratio
        save_resize * : Flag for saving option about resized images
        (*) Can be modified
    Returns
        train_data    : dataset for the train the model
        valid_data    : dataset for the validate the model
        labels        : labels(classes) for the prediction
        max_box_per_image
                      : Maximum number of objects presents per image
        anchors       : the Yolo anchors
    """

    print("  - Parse CoCo Annotation")
    data, labels = parse_coco_annotation(
        coco_dir=coco_dir,
        pkl_file=pkl_file,
        image_dir=image_dir,
        resize_dir=resize_dir,
        given_labels=given_labels
    )

    print("  - Resize the images")
    data = resizing(data, image_dir, resize_dir, save_resize)

    print("  - Get Anchors")
    anchors = get_anchors(data)
    max_box_per_image = max([len(img['object']) for img in data])

    print("  - shuffle and split the dataset")
    train_data, valid_data = split_data(data, split_rate)

    print(f'>> Train on all seen labels. \n {labels}')
    labels = sorted(labels.keys())

    # anchors = [
    #    45, 46, 53, 131, 100, 159,
    #    105, 75, 154, 264, 157, 38,
    #    171, 90, 227, 142, 400, 197,
    #    # 10, 13, 16, 30, 33, 23,
    #    # 30, 61, 62, 45, 59, 119,
    #    # 116, 90, 156, 198, 373, 326
    # ]

    return train_data, valid_data, labels, max_box_per_image, anchors


def parse_coco_annotation(
    coco_dir, pkl_file, image_dir, resize_dir, given_labels
):
    if pkl_file and os.path.exists(pkl_file):
        return load_pkl_file(pkl_file)

    print(f">> Load dataset from coco file in ({coco_dir})")
    data = []
    seen_labels = {}
    labels = {}

    for coco_file in sorted(os.listdir(coco_dir)):
        print(f"- {coco_file}")
        with open(os.path.join(coco_dir, coco_file), 'rt') as f:
            json_data = json.load(f)

        images = json_data['images']
        categories = json_data['categories']
        annotations = json_data['annotations']

        for img in images:
            img_path = os.path.normpath(img['path']).replace("Images", "")
            img['path'] = image_dir + img_path.split('view')[1]
            img['object'] = []

        for cat in categories:
            labels[cat['id']] = cat['name']

        for row in annotations:
            img_id = int(row['image_id']) - 1
            if not os.path.exists(images[img_id]['path']):
                continue

            obj = {}
            obj['name'] = labels[row['category_id']]

            if given_labels != []:
                if obj['name'] not in given_labels:
                    continue

            xmin, ymin, w, h = row['bbox']
            obj['xmin'], obj['xmax'] = xmin, xmin + w
            obj['ymin'], obj['ymax'] = ymin, ymin + h

            images[img_id]['object'] += [obj]

            if obj['name'] not in seen_labels:
                seen_labels[obj['name']] = 0

            seen_labels[obj['name']] += 1

        for img in images:
            if len(img['object']) > 0:
                data += [img]

    cache = {'data': data, 'seen_labels': seen_labels}
    write_pkl_file(pkl_file, cache)

    return data, seen_labels


def resizing(data, image_dir, resize_dir, save_resize=False):
    RESIZE_RATIO = 2
    X_OFFSET, Y_OFFSET = 8, 60
    write_file_count = 0
    for image in data:
        img_path = image['path']

        for obj in image['object']:
            obj['xmax'] = (obj['xmax'] - X_OFFSET) / RESIZE_RATIO
            obj['xmin'] = (obj['xmin'] - X_OFFSET) / RESIZE_RATIO
            obj['ymax'] = (obj['ymax'] - Y_OFFSET) / RESIZE_RATIO
            obj['ymin'] = (obj['ymin'] - Y_OFFSET) / RESIZE_RATIO

        if save_resize:
            image['path'] = img_path.replace(image_dir, resize_dir)
            if not os.path.exists(image['path']):
                source_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                cropped = source_img[
                    Y_OFFSET:Y_OFFSET+832,
                    X_OFFSET:X_OFFSET+1664
                ]
                h, w, _ = cropped.shape
                resized = cv2.resize(
                    cropped,
                    dsize=(w//RESIZE_RATIO, h//RESIZE_RATIO),
                    interpolation=cv2.INTER_AREA)

                make_path(resize_dir, image['path'])
                cv2.imwrite(image['path'], resized)
                write_file_count += 1
                if write_file_count % 100 == 99:
                    print(f"write({write_file_count+1}th): {image['path']}")
    return data


def split_data(data, split_rate):
    split_pivot = int(split_rate*len(data))
    np.random.seed(131)
    np.random.shuffle(data)
    np.random.seed(None)

    train_data = data[:split_pivot]
    valid_data = data[split_pivot:]
    return train_data, valid_data


def load_pkl_file(pkl_file):
    print(f">> Load dataset from ({pkl_file})")
    with open(pkl_file, 'rb') as handle:
        cache = pickle.load(handle)
    return cache['data'], cache['seen_labels']


def write_pkl_file(pkl_file, cache):
    print(f">> Write dataset to ({pkl_file})")
    with open(pkl_file, 'wb') as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)


def make_path(resize_dir, file_path):
    paths = os.path.normpath(file_path).split("\\")[3:-1]
    if not os.path.exists(os.path.dirname(file_path)):
        cur_dir = resize_dir
        for path in paths:
            cur_dir = os.path.join(cur_dir, path)
            try:
                os.mkdir(cur_dir)
            except FileExistsError:
                pass
