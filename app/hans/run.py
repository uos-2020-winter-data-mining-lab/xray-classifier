import os
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from app.hans.data import load_data
from app.hans.generator import BatchGenerator
from app.hans.yolo_model import create_model
from app.hans.evaluate import evaluate

config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
tf.config.run_functions_eagerly(True)


def run():
    # Step 0. Config Options
    # Load Data Config
    given_labels = []  # , 'Bullet']
    TAG = f'{len(given_labels)}labels-0210'
    root_dir = os.path.join('/home', 'handal', 'xray-classifier')
    coco_dir = os.path.join(root_dir, 'data', 'CoCo')
    image_dir = os.path.join(root_dir, 'data', 'Image')
    resize_dir = os.path.join(root_dir, 'data', 'Resize')
    log_dir = os.path.join(root_dir, 'logs')
    pkl_file = os.path.join(root_dir, 'data', f'{TAG}.pkl')
    TAG = '0201'

    epochs = 100
    batch_size = 8
    split_rate = 0.8
    learning_rate = 1e-4
    multi_gpu = 1 
    run_model = True
    save_resize = False
    show_boxes = False
    net_shape = (320, 320)
    pretrained_weights = os.path.join(root_dir, 'data', f'{TAG}.h5')
    trained_weights = os.path.join(root_dir, 'data', f'{TAG}.h5')

    # Step 1 . Load Data
    print(">> Step 1. Load Data")
    train_data, valid_data, labels, max_box_per_image, anchors = load_data(
        coco_dir=coco_dir,
        image_dir=image_dir,
        resize_dir=resize_dir,
        pkl_file=pkl_file,
        split_rate=split_rate,
        save_resize=save_resize,
        given_labels=given_labels
    )

    # Step 2. Data Generating
    train_generator = BatchGenerator(
        data=train_data,
        anchors=anchors,
        labels=labels,
        max_box_per_image=max_box_per_image,
        batch_size=batch_size,
        net_shape=net_shape
    )
    valid_generator = BatchGenerator(
        data=valid_data,
        anchors=anchors,
        labels=labels,
        max_box_per_image=max_box_per_image,
        batch_size=batch_size,
        net_shape=net_shape
    )
    # Step 3. Model Setting
    print(">> Model Setting")
    print(f'\nBatch size : {batch_size}'
          f'\nEpochs : {epochs}')

    train_model, infer_model = create_model(
        num_classes=len(labels),
        anchors=anchors,
        max_box_per_image=max_box_per_image,
        max_grid=[448, 448],
        batch_size=batch_size,
        warmup_batches=0,
        ignore_thresh=0.5,
        multi_gpu=multi_gpu,
        weights=pretrained_weights,
        learning_rate=learning_rate,
        grid_scales=[1, 1, 1],
        obj_scale=5,
        noobj_scale=2,
        xywh_scale=1,
        class_scale=1,
    )
    if False:
        train_model.summary(positions=[.35, .65, .73, 1.])

    # Step 5. Model Fitting
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(min_delta=0, patience=10, verbose=1)
    checkpoint = ModelCheckpoint(
        os.path.join(log_dir, 'ep{epoch:03d}-loss{val_loss:.4f}.h5'),
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=True,
    )

    if run_model:
        print(">> Model Fitting")
        try:
            train_model.fit(
                train_generator,
                steps_per_epoch=max(1, len(train_data)//batch_size),
                validation_data=valid_generator,
                validation_steps=max(1, len(valid_data)//batch_size),
                epochs=epochs,
                callbacks=[checkpoint, reduce_lr, early_stopping],
                workers=4,
                max_queue_size=8
            )
            train_model.save_weights(trained_weights)
        except KeyboardInterrupt:
            print("abort!!!")
            return
    
    print(">> Done")
    return
    infer_model.load_weights(trained_weights, by_name=True)
    

    # Step 7. Predict / Evaluate
    print(">> Evaluate")
    average_precisions = evaluate(
        model=infer_model,
        generator=valid_generator,
        labels=labels,
        anchors=anchors,
        show_boxes=show_boxes,
        nms_thresh=0.45,
        net_shape=net_shape,
    )

    for label, average_precision in average_precisions.items():
        try:
            print(f'label {labels[label]:10} : {(average_precision*100):.2f}%')
        except Exception:
            pass 

    # Step 8. Output
    mAP = sum(average_precisions.values()) / (len(average_precisions) - 5)
    print(f'mAP: {(mAP*100):.2f}%')


if __name__ == '__main__':
    run()
