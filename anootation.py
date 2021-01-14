import json
from collections import defaultdict

name_box_id = defaultdict(list)
id_name = dict()
f = open(
    "data/raw/label/Train/CoCo/coco_astrophysics.json",
    encoding='utf-8')
data = json.load(f)

categories = data['categories']
category_dict = dict()
for cat in categories:
    category_dict[cat['id']] = cat['name']

images = data['images']
image_dict = dict()
for img in images:
    image_path = img['path'].replace("D:\\view\\", "data\\raw\\dataset\\")
    image_dict[img['id']] = image_path

annotations = data['annotations']
for ant in annotations:
    id = ant['image_id']
    path = image_dict[id]

    cat_id = ant['category_id']
    cat_name = category_dict[cat_id]

    if cat_name not in ['Aerosol', 'GunParts', 'Liquid', 'Scissors', 'USB']:
        continue

    if cat_id == 24:    # Scissors
        cat_id = 0
    elif cat_id == 10:  # GunParts
        cat_id = 1
    elif cat_id == 1:  # Aerosol
        cat_id = 2
    elif cat_id == 18:  # Liquid
        cat_id = 3
    elif cat_id == 33:  # USB
        cat_id = 4

    path = image_dict[id]
    if '\\Single_Default\\' not in path:
        continue

    name_box_id[path].append([ant['bbox'], cat_id])

f = open('data/train.txt', 'w')
for key in name_box_id.keys():
    f.write(key)
    box_infos = name_box_id[key]
    for info in box_infos:
        x_min = int(info[0][0])
        y_min = int(info[0][1])
        x_max = x_min + int(info[0][2])
        y_max = y_min + int(info[0][3])

        box_info = " %d,%d,%d,%d,%d" % (
            x_min, y_min, x_max, y_max, int(info[1]))
        f.write(box_info)
    f.write('\n')
f.close()
