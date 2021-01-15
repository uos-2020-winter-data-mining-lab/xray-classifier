import numpy as np
from .utils import get_random_data, preprocess_true_boxes


def split(data, split_rate=0.9, shuffle_seed=10101):
    '''data split to train and valid'''
    np.random.seed(shuffle_seed)
    np.random.shuffle(data)
    np.random.seed(None)
    split_index = int(len(data)*split_rate)

    train_data = data[:split_index]
    valid_data = data[split_index:]

    print(">> "
          f"Train on {len(train_data)} samples, "
          f"Valid on {len(valid_data)} samples ")

    return train_data, valid_data


def generator(data, input_shape, batch_size, anchors, num_classes):
    '''data generator for fit_generator'''
    N = len(data)
    if N == 0 or batch_size <= 0:
        return None

    i = 0
    while True:
        image_data = list()
        box_data = list()

        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(data)
            image, box = get_random_data(data[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % N

        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(
            box_data, input_shape, anchors, num_classes)

        yield [image_data, *y_true], np.zeros(batch_size)
