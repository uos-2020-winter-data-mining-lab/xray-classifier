import os
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from app.hans.config import INPUT_SHAPE, BATCH_SIZE
from app.hans.plot import plotting
from .models.train import create_model
from .utils.data import read_file
from .utils.split import data_split, data_generator


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
    # Step 0 . Path Config
    data = read_file('annotation')
    class_names, num_classes = read_file('classes')
    anchors = read_file('anchors')

    # Step 1 . Data Preprocessing
    (train_data, valid_data), (num_train, num_valid) = data_split(data)

    # Step 2. Data Generating
    train_generator = data_generator(train_data, anchors)
    valid_generator = data_generator(valid_data, anchors)

    # Step 3. Model Setting
    model = create_model(anchors, freeze_body=2, weights='data/yolov3-320.h5')
    # model.summary(positions=[.35, .65, .73, 1.])

    # Step 4. Model Compile
    # model.compile(
    #     optimizer=Adam(lr=1e-4),
    #     loss={'yolo_loss': lambda y_true, y_pred: y_pred},
    # )

    # Step 5. Model Fitting
    # reduce_lr = ReduceLROnPlateau(factor=0.1, patience=3, verbose=1)
    # early_stopping = EarlyStopping(min_delta=0, patience=10, verbose=1)

    # history = model.fit(
    #     train_generator,
    #     steps_per_epoch=max(1, num_train//batch_size),
    #     validation_data=valid_generator,
    #     validation_steps=max(1, num_valid//batch_size),
    #     epochs=100,
    #     callbacks=[reduce_lr, early_stopping])

    # Step 6. Plotting
    # plotting(history)

    # Step 7. Predict / Evaluate
    # results = model.evaluate(test_data, test_labels)

    # Step 8. Output
    # print("RUN XRAY DATA \n epochs : {EPOCHS} \n")


if __name__ == '__main__':
    run()
