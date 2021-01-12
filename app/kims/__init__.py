import os, shutil

original_dataset_dir = '/Users/solbat/xray-classifier/data/raw' # 원본 데이터셋 디렉터리

base_dir = '/Users/solbat/xray-classifier/data/small' # 소규모 데이터셋을 저장할 디렉터리
os.mkdir(base_dir)

# 훈련용 검증용 테스트용 디렉터리 분류
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_aerosol_dir = os.path.join(train_dir, 'aerosol')
os.mkdir(train_aerosol_dir)
train_scissors_dir = os.path.join(train_dir, 'scissors')
os.mkdir(train_scissors_dir)
train_usb_dir = os.path.join(train_dir, 'usb')
os.mkdir(train_usb_dir)
train_liquid_dir = os.path.join(train_dir, 'liquid')
os.mkdir(train_liquid_dir)
train_gunparts_dir = os.path.join(train_dir, 'gunparts')
os.mkdir(train_gunparts_dir)

validation_aerosol_dir = os.path.join(validation_dir, 'aerosol')
os.mkdir(validation_aerosol_dir)
validation_scissors_dir = os.path.join(validation_dir, 'scissors')
os.mkdir(validation_scissors_dir)
validation_usb_dir = os.path.join(validation_dir, 'usb')
os.mkdir(validation_usb_dir)
validation_liquid_dir = os.path.join(validation_dir, 'liquid')
os.mkdir(validation_liquid_dir)
validation_gunparts_dir = os.path.join(validation_dir, 'gunparts')
os.mkdir(validation_gunparts_dir)

test_aerosol_dir = os.path.join(test_dir, 'aerosol')
os.mkdir(test_aerosol_dir)
test_scissors_dir = os.path.join(test_dir, 'scissors')
os.mkdir(test_scissors_dir)
test_usb_dir = os.path.join(test_dir, 'usb')
os.mkdir(test_usb_dir)
test_liquid_dir = os.path.join(test_dir, 'liquid')
os.mkdir(test_liquid_dir)
test_gunparts_dir = os.path.join(test_dir, 'gunparts')
os.mkdir(test_gunparts_dir)

# 에어로졸 350, 가위 1225, usb 500, 액체 625, 총기부품 1000
original_data_dir = os.path.join(original_dataset_dir, 'Aerosol/Single_Default')
original_data = os.listdir(original_data_dir)
for fname in original_data[0:174]:
    src = os.path.join(original_data_dir, fname)
    dst = os.path.join(train_aerosol_dir, fname)
    shutil.copyfile(src, dst)

for fname in original_data[174:261]:
    src = os.path.join(original_data_dir, fname)
    dst = os.path.join(validation_aerosol_dir, fname)
    shutil.copyfile(src, dst)

for fname in original_data[261:350]:
    src = os.path.join(original_data_dir, fname)
    dst = os.path.join(test_aerosol_dir, fname)
    shutil.copyfile(src, dst)

original_data_dir = os.path.join(original_dataset_dir, 'Scissors/Single_Default')
original_data = os.listdir(original_data_dir)
for fname in original_data[0:174]:
    src = os.path.join(original_data_dir, fname)
    dst = os.path.join(train_scissors_dir, fname)
    shutil.copyfile(src, dst)

for fname in original_data[174:261]:
    src = os.path.join(original_data_dir, fname)
    dst = os.path.join(validation_scissors_dir, fname)
    shutil.copyfile(src, dst)

for fname in original_data[261:350]:
    src = os.path.join(original_data_dir, fname)
    dst = os.path.join(test_scissors_dir, fname)
    shutil.copyfile(src, dst)

original_data_dir = os.path.join(original_dataset_dir, 'USB/Single_Default')
original_data = os.listdir(original_data_dir)
for fname in original_data[0:174]:
    src = os.path.join(original_data_dir, fname)
    dst = os.path.join(train_usb_dir, fname)
    shutil.copyfile(src, dst)

for fname in original_data[174:261]:
    src = os.path.join(original_data_dir, fname)
    dst = os.path.join(validation_usb_dir, fname)
    shutil.copyfile(src, dst)

for fname in original_data[261:350]:
    src = os.path.join(original_data_dir, fname)
    dst = os.path.join(test_usb_dir, fname)
    shutil.copyfile(src, dst)

original_data_dir = os.path.join(original_dataset_dir, 'Liquid/Single_Default')
original_data = os.listdir(original_data_dir)
for fname in original_data[0:174]:
    src = os.path.join(original_data_dir, fname)
    dst = os.path.join(train_liquid_dir, fname)
    shutil.copyfile(src, dst)

for fname in original_data[174:261]:
    src = os.path.join(original_data_dir, fname)
    dst = os.path.join(validation_liquid_dir, fname)
    shutil.copyfile(src, dst)

for fname in original_data[261:350]:
    src = os.path.join(original_data_dir, fname)
    dst = os.path.join(test_liquid_dir, fname)
    shutil.copyfile(src, dst)

original_data_dir = os.path.join(original_dataset_dir, 'GunParts/Single_Default')
original_data = os.listdir(original_data_dir)
for fname in original_data[0:174]:
    src = os.path.join(original_data_dir, fname)
    dst = os.path.join(train_gunparts_dir, fname)
    shutil.copyfile(src, dst)

for fname in original_data[174:261]:
    src = os.path.join(original_data_dir, fname)
    dst = os.path.join(validation_gunparts_dir, fname)
    shutil.copyfile(src, dst)

for fname in original_data[261:350]:
    src = os.path.join(original_data_dir, fname)
    dst = os.path.join(test_gunparts_dir, fname)
    shutil.copyfile(src, dst)

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation= 'relu', input_shape=(1080, 1920, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation= 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation= 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation= 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

# from keras import optimizers

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50
)

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_loss, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, acc, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
