import os
import cv2
import json
import numpy as np
import pickle


def load_data(
    coco_dir=None,
    split_rate=0.9,
    pkl_file=None,
    save_resize=None,
):
    coco_dir = os.path.join('data', 'raw', 'label', 'Train', 'CoCo')
    data, labels = parse_coco_annotation(coco_dir, pkl_file)

    data = resizing(data, coco_dir, save_resize)

    train_valid_split = int(split_rate*len(data))
    np.random.seed(131)
    np.random.shuffle(data)
    np.random.seed(None)

    valid_ints = data[train_valid_split:]
    train_data = data[:train_valid_split]

    print(f'>> Train on all seen labels. \n {labels}')
    labels = labels.keys()

    max_box_per_image = max(
        [len(data['object']) for data in (data + valid_ints)])

    anchors = [
        10, 13, 16, 30, 33, 23,
        30, 61, 62, 45, 59, 119,
        116, 90, 156, 198, 373, 326
    ]

    return train_data, valid_ints, sorted(labels), max_box_per_image, anchors


def resizing(data, coco_dir, save_resize=False):
    print(">> Resizing ")
    for i, image in enumerate(data):
        source_path = image['path']
        image['path'] = image['path'].replace(
            "xray-dataset\\dataset\\", "xray-dataset\\resize\\")

        resize_ratio = 2
        for obj in image['object']:
            obj['xmax'] = (obj['xmax'] - 8) / resize_ratio
            obj['xmin'] = (obj['xmin'] - 8) / resize_ratio
            obj['ymax'] = (obj['ymax'] - 60) / resize_ratio
            obj['ymin'] = (obj['ymin'] - 60) / resize_ratio

        if save_resize:
            # cur_dir = "D:\\xray-dataset\\"
            # for path in image['path'].split("\\")[2:-1]:
            #     cur_dir = os.path.join(cur_dir, path)
            #     try:
            #         os.mkdir(cur_dir)
            #     except FileExistsError:
            #         pass

            source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)
            cropped = source_img[60:892, 8:1672]
            h, w, _ = cropped.shape
            resized = cv2.resize(
                cropped,
                dsize=(w//resize_ratio, h//resize_ratio),
                interpolation=cv2.INTER_AREA)
            cv2.imwrite(image['path'], resized)
    return data


def parse_coco_annotation(coco_dir, pkl_file='data/dataset.pkl'):
    if pkl_file is not None and os.path.exists(pkl_file):
        print(f">> Load dataset from {pkl_file} file")
        with open(pkl_file, 'rb') as handle:
            cache = pickle.load(handle)
        return cache['all_insts'], cache['seen_labels']
    all_insts = []
    seen_labels = {}
    categories = {}

    print(">> Load dataset from ")
    for coco_file in sorted(os.listdir(coco_dir)):
        print(f"- {coco_file}")

        coco_path = os.path.join(coco_dir, coco_file)
        with open(coco_path, 'rt') as f:
            data = json.load(f)

        categories_json = data['categories']
        for cat in categories_json:
            categories[cat['id']] = cat['name']

        images_json = data['images']
        obj = {'object': []}
        for i, image in enumerate(images_json):
            image['path'] = \
                "D:\\xray-dataset\\dataset\\" + \
                image['path'].split('view\\')[1]
            image['path'] = image['path'].replace("Images\\", "")
            image['object'] = []
            del(image['dataset_id'])

        annotations_json = data['annotations']
        for i, row in enumerate(annotations_json):
            obj = {}
            img_id = int(row['image_id']) - 1
            if not os.path.exists(images_json[img_id]['path']):
                continue

            obj['name'] = categories[row['category_id']]
            # if obj['name'] not in ['Aerosol', 'Alcohol']:
            #     continue

            xmin, ymin, w, h = row['bbox']
            xmax = xmin + w
            ymax = ymin + h

            obj['xmin'], obj['xmax'] = xmin, xmax
            obj['ymin'], obj['ymax'] = ymin, ymax
            images_json[img_id]['object'] += [obj]

            if obj['name'] in seen_labels:
                seen_labels[obj['name']] += 1
            else:
                seen_labels[obj['name']] = 1

        for i, image in enumerate(images_json):
            if len(image['object']) > 0:
                all_insts += [image]

    cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
    print(f">> Write pkl files {pkl_file}")
    with open(pkl_file, 'wb') as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return all_insts, seen_labels
