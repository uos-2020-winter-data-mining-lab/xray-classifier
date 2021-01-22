import numpy as np


def evaluate(
    model, generator, iou_threshold=0.5, obj_thresh=0.5,
    nms_thresh=0.45, net_h=416, net_w=416, save_path=None
):
    num_classes = generator.num_classes()
    anchors = generator.get_anchors()
    # gather all detections and annotations
    all_detections = [[
        None for i in range(num_classes)] for j in range(generator.size())]
    all_annotations = [[
        None for i in range(num_classes)] for j in range(generator.size())]

    for i in range(generator.size()):
        image = [generator.load_image(i)]
        # make the boxes and the labels
        pred_boxes = get_yolo_boxes(
            model, image, net_h, net_w, anchors, obj_thresh, nms_thresh)[0]

        score = np.array([box.get_score() for box in pred_boxes])
        pred_labels = np.array([box.label for box in pred_boxes])
        # print("score and labels: ", score, pred_labels)
        if len(pred_boxes) > 0:
            pred_boxes = np.array(
                [[box.xmin, box.ymin, box.xmax, box.ymax, box.get_score()] for box in pred_boxes])
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

    # compute mAP by comparing all detections and all annotations
    average_precisions = {}

    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
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

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
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
