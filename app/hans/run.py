import os
import numpy as np
from PIL import Image
from keras import optimizers
from app.hans.model import set_model
from app.hans.data import copy_data
from app.hans.plot import plotting
from app.hans.image import preprocess_data
from app.hans.preprocess import generator

DATA_DIR = 'data/'
RAW_DATA = f'{DATA_DIR}/raw/dataset'
EPOCHS = 20
BATCH_SIZE = 64


def bbox(image):
    rows = np.any(image, axis=1)
    cols = np.any(image, axis=0)
    rmin, rmax = np.where(rows)[0, -1]
    cmin, cmax = np.where(cols)[0, -1]
    return rmin, rmax, cmin, cmax


def run():
    # Step0. config
    RAW_DIR = os.path.join(DATA_DIR, 'raw\\dataset\\')
    TRAIN_DIR = os.path.join(DATA_DIR, 'train\\')
    VALID_DIR = os.path.join(DATA_DIR, 'valid\\')
    # TEST_DIR = os.path.join(DATA_DIR, 'test\\')

    # Step1. Data Preparing
    init_flag = False
    print("DATA COPYING...")
    if init_flag is True:
        copy_data(RAW_DIR)

        preprocess_data(TRAIN_DIR)
        preprocess_data(VALID_DIR)

    # Step2. Data Generating
    print("DATA GENERATING...")
    train_generator = generator(TRAIN_DIR)
    valid_generator = generator(VALID_DIR)

    # Step3. Build a Model
    print("MODEL SETTING...")
    model = set_model(summary=True)
    print("MODEL COMPILE...")
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    # Step4. Fit the Model
    print("MODEL FITTING...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=valid_generator
        # validation_steps=50
    )

    # Step5. Predict / Evaluate
    # results = model.evaluate(test_data, test_labels)

    # Step6. Plotting
    plotting(history, EPOCHS)

    # Step7. Output
    print("RUN XRAY DATA \n epochs : {EPOCHS} \n")


if __name__ == '__main__':
    run()
