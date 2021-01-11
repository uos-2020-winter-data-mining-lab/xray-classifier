import os
import shutil
import numpy as np
from PIL import Image
from keras import optimizers
from data import make_path, copy_files, load_data
from model import set_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


DATA_DIR = '../../data'


def set_data_generator(source, target_dir):
    target_dir = os.path.join(source, target_dir)
    target_datagen = ImageDataGenerator(
        rotation_range=20,
        fill_mode='nearest',
    )
    target_generator = target_datagen.flow_from_directory(
        target_dir,
        target_size=(1080, 1920),
        batch_size=1,
        class_mode='categorical'
    )
    return target_generator


def draw_plot(history, metric):
    history_metric = history.history[metric]
    history_val_metric = history.history[f'val_{metric}']
    epochs = range(1, len(history_metric) + 1)
    plt.plot(epochs, history_metric, 'bo', label=f'Train {metric}')
    plt.plot(epochs, history_val_metric, 'b', label=f'Valid {metric}')
    plt.title(f'Training and validation {metric}')
    plt.legend()


def bbox(image):
    rows = np.any(image, axis=1)
    cols = np.any(image, axis=0)
    rmin, rmax = np.where(rows)[0, -1]
    cmin, cmax = np.where(cols)[0, -1]
    return rmin, rmax, cmin, cmax


def from_image(dir_path):
    file_list = os.listdir(os.path.join(dir_path))
    for image_file in file_list:
        image_path = os.path.join(dir_path, image_file)
        image = Image.open(image_path)
        pixel = np.array(image)
        print(pixel)


def image_preprocessor(source, target):
    target_dir = os.path.join(source, target)
    for dir_name in os.listdir(target_dir):
        dir_path = os.path.join(target_dir, dir_name)
        for category in os.listdir(dir_path):
            image_dir = os.path.join(dir_path, category)
            from_image(image_dir)


def prepare_data():
    source = f'{DATA_DIR}/raw/dataset'
    for data in os.listdir(source):
        path = os.path.join(source, data)
        if os.path.isdir(path) is True:
            load_data(source, data)


def run():
    # Prepare dataset
    prepare_data()

    train_generator = set_data_generator(DATA_DIR, 'train')
    valid_generator = set_data_generator(DATA_DIR, 'valid')

    model = set_model(summary=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    history = model.fit(
        train_generator,
        epochs=10,
        batch_size=16,
        validation_data=valid_generator,
        # validation_steps=50
    )

    draw_plot(history, 'acc')
    plt.figure()
    draw_plot(history, 'loss')
    plt.show()


if __name__ == '__main__':
    run()
