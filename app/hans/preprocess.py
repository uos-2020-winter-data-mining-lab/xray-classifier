from keras.preprocessing.image import ImageDataGenerator
from app.hans.config import WIDTH, HEIGHT, RATIO


def generator(target_dir):
    target_datagen = ImageDataGenerator(
        rotation_range=20,
        fill_mode='nearest',
    )
    target_generator = target_datagen.flow_from_directory(
        target_dir,
        target_size=(WIDTH//RATIO, HEIGHT//RATIO),
        batch_size=1,
        class_mode='categorical'
    )
    return target_generator
