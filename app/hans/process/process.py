import numpy as np


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
