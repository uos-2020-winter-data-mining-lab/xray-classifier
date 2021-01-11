from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from app.hans.config import WIDTH, HEIGHT, RATIO


def set_model(summary=False):
    INPUT_SHAPE = (WIDTH//RATIO, HEIGHT//RATIO, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(5, activation='softmax'))

    if summary is True:
        model.summary()

    return model
