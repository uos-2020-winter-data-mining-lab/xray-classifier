import os
import json
import numpy as np
import pickle


def load_data(
    coco_dir=None,
    split_rate=0.9
):
    coco_dir = os.path.join('data', 'raw', 'label', 'Train', 'CoCo')
    train_ints, train_labels = parse_coco_annotation(coco_dir)

    train_valid_split = int(split_rate*len(train_ints))
    np.random.seed(131)
    np.random.shuffle(train_ints)
    np.random.seed(None)

    valid_ints = train_ints[train_valid_split:]
    train_ints = train_ints[:train_valid_split]

    print(f'>> Train on all seen labels. \n {train_labels}')
    labels = train_labels.keys()

    max_box_per_image = max(
        [len(inst['object']) for inst in (train_ints + valid_ints)])

    anchors = [
        10, 13, 16, 30, 33, 23,
        30, 61, 62, 45, 59, 119,
        116, 90, 156, 198, 373, 326
    ]

    return train_ints, valid_ints, sorted(labels), max_box_per_image, anchors


def parse_coco_annotation(coco_dir):
    if os.path.exists("data/dataset.pkl"):
        print("load from .pkl file")    
        with open("dataset.pkl", 'rb') as handle:
            cache = pickle.load(handle)
        return cache['all_insts'], cache['seen_labels']
    all_insts = []
    seen_labels = {}
    categories = {}

    for coco_file in sorted(os.listdir(coco_dir)):
        print(f">> file : {coco_file}")
        if 'astro' not in coco_file:
            continue

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
            # if 'Single_Default' not in images_json[img_id]['path']:
            #     continue

            obj['name'] = categories[row['category_id']]
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
    with open("data/dataset.pkl", 'wb') as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return all_insts, seen_labels
