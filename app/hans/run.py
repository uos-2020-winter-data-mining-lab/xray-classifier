import os
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from app.hans.config import INPUT_SHAPE
from app.hans.plot import plotting
from .models.train import create_model
from .models.model import preprocess_true_boxes
from .models.utils import get_random_data
DATA_DIR = 'data/'
RAW_DATA = f'{DATA_DIR}/raw/dataset'

config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def run():
    # Step 0 . Path
    annotation_path = os.path.join(DATA_DIR, 'train.txt')
    with open(annotation_path) as f:
        lines = f.readlines()

    classes_path = os.path.join(DATA_DIR, 'labels.txt')
    class_names = get_classes(classes_path)
    num_classes = len(class_names)

    anchors_path = os.path.join(DATA_DIR, 'anchors.txt')
    anchors = get_anchors(anchors_path)

    # Step 1 . Model Config
    input_shape = INPUT_SHAPE

    model = create_model(input_shape, anchors, num_classes, freeze_body=2)
    # model.summary(positions=[.35, .65, .73, 1.])

    val_split = 0.1
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    num_valid = int(len(lines) * val_split)
    num_train = len(lines) - num_valid

    model.compile(
        optimizer=Adam(lr=1e-4),
        loss={'yolo_loss': lambda y_true, y_pred: y_pred}
    )

    batch_size = 8
    print(f'Train on {num_train} samples, '
          f'val on {num_valid} samples, '
          f'with batch size {batch_size}.')

    train_generator = data_generator_wrapper(
        lines[:num_train], batch_size, input_shape, anchors, num_classes)

    valid_generator = data_generator_wrapper(
        lines[num_train:], batch_size, input_shape, anchors, num_classes)

    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(min_delta=0, patience=10, verbose=1)

    for i in range(len(model.layers)):
        model.layers[i].trainable = True

    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, num_train//batch_size),
        validation_data=valid_generator,
        validation_steps=max(1, num_valid//batch_size),
        epochs=100,
        callbacks=[reduce_lr, early_stopping])

    # Step0. config
    # RAW_DIR = os.path.join(DATA_DIR, 'raw\\dataset\\')
    # TRAIN_DIR = os.path.join(DATA_DIR, 'train\\')
    # VALID_DIR = os.path.join(DATA_DIR, 'valid\\')
    # TEST_DIR = os.path.join(DATA_DIR, 'test\\')

    # # Step1. Data Preparing
    # init_flag = False
    # print("DATA COPYING...")
    # if init_flag is True:
    #     copy_data(RAW_DIR)

    #     preprocess_data(TRAIN_DIR)
    #     preprocess_data(VALID_DIR)

    # # Step2. Data Generating
    # print("DATA GENERATING...")
    # train_generator = generator(TRAIN_DIR)
    # valid_generator = generator(VALID_DIR)

    # # Step5. Predict / Evaluate
    # # results = model.evaluate(test_data, test_labels)

    # # Step6. Plotting
    plotting(history)

    # # Step7. Output
    # print("RUN XRAY DATA \n epochs : {EPOCHS} \n")


if __name__ == '__main__':
    run()


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


def data_generator_wrapper(
    annotation_lines, batch_size, input_shape, anchors, num_classes
):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        return None
    return data_generator(
        annotation_lines, batch_size, input_shape, anchors, num_classes)


def data_generator(
    annotation_lines, batch_size, input_shape, anchors, num_classes
):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(
                annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(
            box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a
