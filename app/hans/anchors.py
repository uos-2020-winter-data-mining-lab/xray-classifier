import os
import numpy as np


def txt2boxes(data):
    boxes = []
    for img in data:
        for obj in img['object']:
            width = int(obj['xmax']) - int(obj['xmin'])
            height = int(obj['ymax']) - int(obj['ymin'])
            boxes.append([width, height])
    result = np.array(boxes)
    return result


def kmeans(boxes, k, dist=np.median):
    box_number = boxes.shape[0]
    distances = np.empty((box_number, k))
    last_nearest = np.zeros((box_number,))
    np.random.seed()
    clusters = boxes[
        np.random.choice(box_number, k, replace=False)]  # init k clusters
    while True:
        distances = 1 - iou(boxes, clusters, k)
        current_nearest = np.argmin(distances, axis=1)
        if (last_nearest == current_nearest).all():
            break  # clusters won't change
        for cluster in range(k):
            # update clusters
            clusters[cluster] = \
                dist(boxes[current_nearest == cluster], axis=0)
        last_nearest = current_nearest
    return clusters


def iou(boxes, clusters, k):  # 1 box -> k clusters
    n = boxes.shape[0]

    box_area = boxes[:, 0] * boxes[:, 1]
    box_area = box_area.repeat(k)
    box_area = np.reshape(box_area, (n, k))

    cluster_area = clusters[:, 0] * clusters[:, 1]
    cluster_area = np.tile(cluster_area, [1, n])
    cluster_area = np.reshape(cluster_area, (n, k))

    box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
    cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
    min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

    box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
    cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
    min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
    inter_area = np.multiply(min_w_matrix, min_h_matrix)

    result = inter_area / (box_area + cluster_area - inter_area)
    return result


def get_avg_iou(boxes, clusters, k):
    accuracy = np.mean([np.max(iou(boxes, clusters, k), axis=1)])
    return accuracy


def save_anchors(data):
    anchors = []
    with open("yolo_anchors.txt", 'w') as f:
        row = np.shape(data)[0]
        for i in range(row):
            anchors.append(int(data[i][0]))
            anchors.append(int(data[i][1]))
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)

    return anchors


def get_anchors(data):
    cluster_number = 9
    all_boxes = txt2boxes(data)

    if os.path.exists('yolo_anchors.txt'):
        with open('yolo_anchors.txt') as f:
            data = f.readline()

        anchors = list()
        for anchor in data.split(","):
            anchors.append(int(anchor.lstrip(' ')))
        return anchors
    else:
        result = kmeans(all_boxes, k=cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        anchors = save_anchors(result)

    avg_iou = get_avg_iou(all_boxes, result, cluster_number) * 100

    print(f" Anchor Accuracy: {avg_iou:.2f}%")

    return anchors
