import os
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from app.hans.load_data import load_data
from app.hans.generator import BatchGenerator
from app.hans.yolo_model import create_model
from app.hans.evaluate import evaluate

config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def normalize(image):
    return image/255.


def run():
    # Step 0 . Config

    # Step 1 . Load Data
    train_data, valid_data, labels, max_box_per_image, anchors = load_data(
        coco_dir=os.path.join('data', 'raw', 'label', 'Train', 'CoCo'),
        split_rate=0.8
    )
    print("max box per image: ", max_box_per_image)
    print(f'\nTraining on {len(train_data)+len(valid_data)} items'
          f'\nTrain data {len(train_data)}, Valid data {len(valid_data)}'
          f'\n{len(labels)} labels : {str(labels)}')

    batch_size = 2

    # Step 2. Data Generating
    train_generator = BatchGenerator(
        data=train_data,
        anchors=anchors,
        labels=labels,
        downsample=32,
        max_box_per_image=max_box_per_image,
        batch_size=batch_size,
        min_net_size=288,
        max_net_size=448,
        shuffle=True,
        jitter=0.0,
        norm=normalize
    )
    valid_generator = BatchGenerator(
        data=valid_data,
        anchors=anchors,
        labels=labels,
        downsample=32,
        max_box_per_image=max_box_per_image,
        batch_size=batch_size,
        min_net_size=288,
        max_net_size=448,
        shuffle=True,
        jitter=0.0,
        norm=normalize
    )

    # Step 3. Model Setting
    try:
        print(" >> Model Setting")
        train_model, infer_model = create_model(
            num_classes=len(labels),
            anchors=anchors,
            max_box_per_image=max_box_per_image,
            max_grid=[448, 448],
            batch_size=batch_size,
            warmup_batches=0,
            ignore_thresh=0.5,
            multi_gpu=1,
            weights='data/yolov3-416-16.h5',
            lr=1e-4,
            grid_scales=[1, 1, 1],
            obj_scale=5,
            noobj_scale=1,
            xywh_scale=1,
            class_scale=1,
        )

        if False:
            train_model.summary(positions=[.35, .65, .73, 1.])
        # Step 5. Model Fitting
        reduce_lr = ReduceLROnPlateau(factor=0.1, patience=3, verbose=1)
        early_stopping = EarlyStopping(min_delta=0, patience=10, verbose=1)
        ModelCheckpoint(
            'logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
            monitor='val_loss',
            save_weights_only=True,
            save_best_only=True,
            save_freq=3
        )

        print(" >> Model Fitting")
        history = train_model.fit(
            train_generator,
            # steps_per_epoch=max(1, len(train_data)//batch_size),
            validation_data=valid_generator,
            # validation_steps=max(1, len(valid_data)//batch_size),
            epochs=1,
            callbacks=[reduce_lr, early_stopping],
        )
    except KeyboardInterrupt:
        print("abort!!!")
        return

    train_model.save_weights('data/trained-416-16.h5')
    print("done")
    return
    plotting(history)
    infer_model.load_weights('data/trained-416.h5', by_name=True)
    # Step 6. Plotting

    # Step 7. Predict / Evaluate

    print("Evaluate")
    average_precisions = evaluate(
        model=infer_model,
        generator=valid_generator,
    )

    for label, average_precision in average_precisions.items():
        print(f'{labels[label]} : {average_precision:.4f}')

    mAP = sum(average_precisions.values()) / len(average_precisions)
    print(f'mAP: {mAP:.4f}')

    # display_image(
    #     'data//raw//dataset//Astrophysics//Aerosol//Single_Default//'
    #     'H_8481.80-1090_01_153.png')

    # Step 8. Output
    # print("RUN XRAY DATA \n epochs : {EPOCHS} \n")


if __name__ == '__main__':
    run()
