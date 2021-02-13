import numpy as np
from .evaluate_util import (
    preprocess_input, decode_netout, do_nms, draw_boxes
)
from timeit import default_timer as timer


def evaluate(
    model, generator, labels, anchors, iou_threshold=0.50, obj_thresh=0.50,
    nms_thresh=0.45, net_shape=(416, 416), save_path=None, show_boxes=False
):
    net_h, net_w = net_shape
    num_classes = generator.num_classes

    # gather all detections and annotations
    gen_size = generator.size()
    all_detections = [[None for i in range(num_classes)] for j in range(gen_size)]
    all_annotations = [[None for i in range(num_classes)] for j in range(gen_size)]

    start = end = timer()
    for i in range(gen_size):
        image = [generator.load_image(i)]
        # make the boxes and the labels
        pred_boxes = get_yolo_boxes(
            model, image, labels, net_h, net_w, anchors, obj_thresh,
            nms_thresh, False
        )[0]

        score = np.array([box.get_score() for box in pred_boxes])
        pred_labels = np.array([box.label for box in pred_boxes])

        if len(pred_boxes) > 0:
            pred_boxes = np.array([
                [box.xmin, box.ymin, box.xmax, box.ymax, box.get_score()]
                for box in pred_boxes])
        else:
            pred_boxes = np.array([[]])

        # sort the boxes and the labels according to scores
        score_sort = np.argsort(-score)
        pred_labels = pred_labels[score_sort]
        pred_boxes = pred_boxes[score_sort]
        # copy detections to all_detections
        for label in range(num_classes):
            all_detections[i][label] = pred_boxes[pred_labels == label, :]

        annotations = generator.load_annotation(i)

        # copy detections to all_annotations
        for label in range(num_classes):
            all_annotations[i][label] = \
                annotations[annotations[:, 4] == label, :4].copy()

        if i % 100 == 0:
            current = end
            end = timer()
            print(f"{i:4d}/{gen_size:6d}th time {(end - current):.3f}, {(end - start):.3f}")

    # compute mAP by comparing all detections and all annotations
    average_precisions = {}

    for label in range(num_classes):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(gen_size):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(
                    np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and \
                   assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / \
            np.maximum(true_positives + false_positives,
                       np.finfo(np.float64).eps)

        # compute average precision
        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision

    return average_precisions


def get_yolo_boxes(
    model, images, labels, net_h, net_w, anchors, obj_thresh, nms_thresh,
    show_boxes
):
    image_h, image_w, _ = images[0].shape
    num_images = len(images)
    batch_input = np.zeros((num_images, net_h, net_w, 3))

    # preprocess the input
    for i in range(num_images):
        batch_input[i] = preprocess_input(images[i], net_h, net_w)

    # run the prediction
    batch_output = model.predict_on_batch(batch_input)
    batch_boxes = [None]*num_images

    for i in range(num_images):
        yolos = [batch_output[0][i], batch_output[1][i], batch_output[2][i]]
        boxes = []

        # decode the output of the network
        for j in range(len(yolos)):
            yolo_anchors = anchors[(2-j)*6:(3-j)*6]
            boxes += decode_netout(
                yolos[j], yolo_anchors, obj_thresh, net_h, net_w)

        # correct the sizes of the bounding boxes
        correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

        # suppress non-maximal boxes
        do_nms(boxes, nms_thresh)

        if False:  # show_boxes and i == (num_images) - 1:
            draw_boxes(images[i], boxes, labels, obj_thresh)

        batch_boxes[i] = boxes
    return batch_boxes


def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iou_xmax = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2])
    iou_xmin = np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    iou_ymax = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3])
    iou_ymin = np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iou_xmax - iou_xmin, 0)
    ih = np.maximum(iou_ymax - iou_ymin, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) \
        + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_w = net_w
        new_h = (image_h*net_w)/image_w
    else:
        new_h = net_w
        new_w = (image_w*net_h)/image_h

    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h

        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
