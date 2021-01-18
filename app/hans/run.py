import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from app.hans.config import INPUT_SHAPE, BATCH_SIZE
from app.hans.process.metadata import read_meta_files
from app.hans.process.process import split, generator, preprocess_data
from app.hans.models.model import create_model
from app.hans.plot import display_image
from .plot import plotting


config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def run():
    # Step 0 . Config
    log_dir = 'logs/'
    data, num_classes, anchors = read_meta_files(meta_dir='data/metadata')
    input_shape, batch_size = INPUT_SHAPE, BATCH_SIZE

    # Step 1 . Load Data
    preprocess = False
    if preprocess:
        data = preprocess_data(data)
    train_data, valid_data = split(data)

    # Step 2. Data Generating
    train_generator = generator(
        train_data, input_shape, batch_size, anchors, num_classes)
    valid_generator = generator(
        valid_data, input_shape, batch_size, anchors, num_classes)

    # Step 3. Model Setting
    model = create_model(
        input_shape=input_shape,
        anchors=anchors,
        num_classes=num_classes,
        freeze_body=2,
        weights='data/yolov3-320.h5',
        summary=False
    )

    # Step 4. Model Compile
    model.compile(
        optimizer=Adam(lr=1e-4),
        loss={'yolo_loss': lambda y_true, y_pred: y_pred},
    )

    # Step 5. Model Fitting
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(min_delta=0, patience=10, verbose=1)
    ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=True,
        save_freq=3
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, len(train_data)//batch_size),
        validation_data=valid_generator,
        validation_steps=max(1, len(valid_data)//batch_size),
        epochs=30,
        callbacks=[reduce_lr, early_stopping])

    model.save_weights('data/trained_weights_final.h5')

    # Step 6. Plotting
    plotting(history)
    display_image(
        'data//raw//dataset//Astrophysics//Aerosol//Single_Default//'
        'H_8481.80-1090_01_153.png')

    # Step 7. Predict / Evaluate
    # results = model.evaluate(test_data, test_labels)

    # Step 8. Output
    # print("RUN XRAY DATA \n epochs : {EPOCHS} \n")


if __name__ == '__main__':
    run()
