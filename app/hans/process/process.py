import numpy as np
from app.hans.config import BATCH_SIZE, INPUT_SHAPE
from app.hans.models.utils import get_random_data
from app.hans.models.model import preprocess_true_boxes


def data_split(data, split_rate=0.9, shuffle_seed=10101):
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


def data_generator(data, anchors):
    '''data generator for fit_generator'''
    N = len(data)
    if N == 0 or BATCH_SIZE <= 0:
        return None

    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(BATCH_SIZE):
            if i == 0:
                np.random.shuffle(data)
            image, box = get_random_data(data[i], random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % N
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, INPUT_SHAPE, anchors)
        yield [image_data, *y_true], np.zeros(BATCH_SIZE)


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a
