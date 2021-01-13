def Darknet53(inputs, training, data_format):
    inputs = Conv2D_fixed_padding(inputs, 32, 3, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = Conv2D_fixed_padding(inputs, 64, 3, strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = darknet53_residual_block(inputs, 32, training, data_format=data_format)
    inputs = Conv2D_fixed_padding(inputs, 128, 3, strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format=data_format)

    for _ in range(2):
        inputs = darknet53_residual_block(inputs, 64, training, data_format=data_format)
        inputs = Conv2D_fixed_padding(inputs, 256, 3, strides=2, data_format=data_format)
        inputs = batch_norm(inputs, training, data_format=data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(8):
        inputs = darknet53_residual_block(inputs, 256, training, data_format=data_format)

    route2 = inputs

    inputs = Conv2D_fixed_padding(inputs, 1024, 3, strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(4):
        inputs = darknet53_residual_block(inputs, 512, training, data_format)

    route3 = inputs

    return route1, route2, route3