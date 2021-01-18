import cv2
import matplotlib.pyplot as plt
from app.hans.process.process import crop_and_resize_images


def plotting(history):
    history_dict = history.history
    epochs = range(1, len(history_dict['loss']) + 1)
    plotting_figure(history_dict, 'loss', epochs)

    plt.show()


def display_image(image_path):
    image = crop_and_resize_images(image_path)
    plt.imshow(image)
    plt.show()


def plotting_figure(history_dict, target, epochs):
    plt.figure()
    target_values = history_dict[target]
    val_target_values = history_dict[f'val_{target}']
    plt.plot(epochs, target_values, 'bo', label=f'Training {target}')
    plt.plot(epochs, val_target_values, 'b', label=f'Validation {target}')
    plt.title(f'Training and validation {target}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
