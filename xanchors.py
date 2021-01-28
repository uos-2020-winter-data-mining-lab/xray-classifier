import os
import random
import numpy as np
from app.hans.data import parse_coco_annotation


def IOU(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w*h/(c_w*c_h)
        elif c_w >= w and c_h <= h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else:
            # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity)  # will become (k,) shape

    return np.array(similarities)


def avg_IOU(anns, centroids):
    n, d = anns.shape
    sum = 0.

    for i in range(anns.shape[0]):
        sum += max(IOU(anns[i], centroids))

    return sum/n


def print_anchors(centroids):
    anchors = centroids.copy()

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    out_string = "anchors: ["
    for i in sorted_indices:
        out_string += str(int(anchors[i, 0]*416)) + \
            ',' + str(int(anchors[i, 1]*416)) + ', '

    print(out_string[:-2])


def _main_():
    num_anchors = 9

    TAG = 'yolov3-0127'
    coco_dir = os.path.join('data', 'CoCo')
    image_dir = os.path.join('D:\\', 'xray-dataset', 'dataset')
    resize_dir = os.path.join('D:\\', 'xray-dataset', 'resize')
    pkl_file = os.path.join('data', f'{TAG}.pkl')

    train_data, train_labels = parse_coco_annotation(
        coco_dir=coco_dir,
        image_dir=image_dir,
        resize_dir=resize_dir,
        pkl_file=pkl_file,
    )

    # run k_mean to find the anchors
    RESIZE_RATIO = 2
    X_OFFSET, Y_OFFSET = 8, 60
    annotation_dims = []
    for img in train_data:
        width, height = img['width'], img['height']
        # 1664, 832
        for obj in img['object']:
            # obj['xmax'] = (obj['xmax'] - X_OFFSET) / RESIZE_RATIO
            # obj['xmin'] = (obj['xmin'] - X_OFFSET) / RESIZE_RATIO
            # obj['ymax'] = (obj['ymax'] - Y_OFFSET) / RESIZE_RATIO
            # obj['ymin'] = (obj['ymin'] - Y_OFFSET) / RESIZE_RATIO

            relative_w = (float(obj['xmax']) - float(obj['xmin'])) / width
            relatice_h = (float(obj["ymax"]) - float(obj['ymin'])) / height
            annotation_dims.append(tuple(map(float, (relative_w, relatice_h))))

    annotation_dims = np.array(annotation_dims)
    centroids = run_kmeans(annotation_dims, num_anchors)

    # write anchors to file
    print(f"\naverage IOU for {num_anchors} "
          f"Anchors : {avg_IOU(annotation_dims, centroids)}")
    print_anchors(centroids)


def run_kmeans(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    prev_assignments = np.ones(ann_num)*(-1)
    iteration = 0
    old_distances = np.zeros((ann_num, anchor_num))

    indices = [random.randrange(ann_dims.shape[0]) for i in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            d = 1 - IOU(ann_dims[i], centroids)
            distances.append(d)
        # distances.shape = (ann_num, anchor_num)
        distances = np.array(distances)

        print(f"iteration {iteration}: "
              f"dists = {np.sum(np.abs(old_distances-distances))}")

        # assign samples to centroids
        assignments = np.argmin(distances, axis=1)

        if (assignments == prev_assignments).all():
            return centroids

        # calculate new centroids
        centroid_sums = np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]] += ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j]/(np.sum(assignments == j) + 1e-6)

        prev_assignments = assignments.copy()
        old_distances = distances.copy()


if __name__ == '__main__':
    _main_()
