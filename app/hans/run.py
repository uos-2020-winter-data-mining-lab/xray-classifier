import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from app.hans.config import INPUT_SHAPE, BATCH_SIZE
from app.hans.process.metadata import read_meta_files
from app.hans.process.process import split, BatchGenerator
from app.hans.models.yolo import create_model
# from utils.utils import evaluate
from .plot import plotting


config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def run():
    # Step 0 . Config
    log_dir = 'logs/'
    data, classes, anchors = read_meta_files(meta_dir='data/metadata')
    input_shape, batch_size = INPUT_SHAPE, BATCH_SIZE
    # Step 1 . Load Data
    train_data, valid_data = split(data)

    # Step 2. Data Generating
    train_generator = BatchGenerator(
        data=train_data,
        anchors=anchors,
        labels=classes,
    )
    valid_generator = BatchGenerator(
        data=valid_data,
        anchors=anchors,
        labels=classes,
    )

    # Step 3. Model Setting
    train_model, infer_model = create_model(
        image_input=input_shape,
        anchors=anchors,
        num_classes=len(classes),
        max_grid=[448, 448],
    )

    if False:
        train_model.summary(positions=[.35, .65, .73, 1.])

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

    history = train_model.fit(
        train_generator,
        steps_per_epoch=max(1, len(train_data)//batch_size),
        validation_data=valid_generator,
        validation_steps=max(1, len(valid_data)//batch_size),
        epochs=100,
        callbacks=[reduce_lr, early_stopping],
    )

    train_model.save_weights('data/trained_yolo.h5')
    plotting(history)
    infer_model.load_weights('data/trained_yolo.h5', by_name=True)

    # Step 6. Plotting

    # Step 7. Predict / Evaluate

    print("Evaluate")
    # average_precisions = evaluate(
    #     model=infer_model,
    #     data=valid_data,
    #     generator=valid_generator,
    #     anchors=anchors,
    #     num_classes=len(classes)
    # )

    # for label, average_precision in average_precisions.items():
    #     print(f'{classes[label]} : {average_precision:.4f}')

    # mAP = sum(average_precisions.values()) / len(average_precisions)
    # print(f'mAP: {mAP:.4f}')

    # display_image(
    #     'data//raw//dataset//Astrophysics//Aerosol//Single_Default//'
    #     'H_8481.80-1090_01_153.png')

    # Step 8. Output
    # print("RUN XRAY DATA \n epochs : {EPOCHS} \n")


if __name__ == '__main__':
    run()
