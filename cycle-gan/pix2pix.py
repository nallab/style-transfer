"""Pix2pix
"""

import os
import time
from absl import flags
import tensorflow as tf

assert tf.__version__.startswith('2')

FLAGS = flags.FLAGS

flags.DEFINE_integer('buffer_size', 400, 'Shuffle buffer size')
flags.DEFINE_integer('batch_size', 1, 'Batch Size')
flags.DEFINE_integer('epochs', 1, 'Number of epochs')
flags.DEFINE_string('path', None, 'Path to the data folder')
flags.DEFINE_boolean('enable_function', True, 'Enable Function?')

IMG_WIDTH = 256
IMG_HEIGHT = 256
AUTOTUNE = tf.data.experimental.AUTOTUNE


def load(image_file):
    """
    Loads the image and generates input and target image.

    :param image_file: .jpeg file
    :return: Input image, target image
    """
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


def downsample(filters, size, norm_type='batchnrom', apply_norm=True):
    """

    :param filters:
    :param size:
    :param norm_type:
    :param apply_norm:
    :return:
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False)
    )

    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            pass
            # result.add(InstanceNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
    """

    :param filters:
    :param size:
    :param norm_type:
    :param apply_dropout:
    :return:
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False)
    )

    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        # result.add(InstanceNormalization())
        pass
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def unet_generator(output_channels, norm_type='batchnorm'):
    """
    
    :param output_channels: 
    :param norm_type: 
    :return: 
    """

    down_stack = [
        downsample(64, 4, norm_type, apply_norm=False),
        downsample(128, 4, norm_type),
        downsample(256, 4, norm_type),
        downsample(512, 4, norm_type),
        downsample(512, 4, norm_type),
        downsample(512, 4, norm_type),
        downsample(512, 4, norm_type),
        downsample(512, 4, norm_type),
    ]

    up_stack = [
        upsample(512, 4, norm_type, apply_dropout=True),
        upsample(512, 4, norm_type, apply_dropout=True),
        upsample(512, 4, norm_type, apply_dropout=True),
        upsample(512, 4, norm_type),
        upsample(256, 4, norm_type),
        upsample(128, 4, norm_type),
        upsample(64, 4, norm_type),
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 4, strides=2,
        padding='same', kernel_initializer=initializer,
        activation='tanh'
    )

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    x = inputs

    # Downsampling though the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def discriminator(norm_type='batchnorm', target=True):
    """
    PatchGan discriminator model (https://arxiv.org/abs/1611.07004).

    :param norm_type:
    :param target:
    :return:
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
    x = inp

    if target:
        tar = tf.keras.layers.Input(shape=[None,None, 3], name='target_image')
        x = tf.keras.layers.concatenate([inp, tar])

    down1 = downsample(64, 4, norm_type, False)(x)
    down2 = downsample(128, 4, norm_type)(down1)
    down3 = downsample(256, 4, norm_type)(down2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer,
        use_bias=False)(zero_pad1)

    if norm_type.lower() == 'batchnorm':
        norm1 = tf.keras.layers.BatchNormalization()(conv)
    elif norm_type.lower() == 'instancenorm':
        # norm1 = InstanceNormalization()(conv)
        pass

    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

    last = tf.keras.layers.Conv2D(
        1, 4, strides=1,
        kernel_initializer=initializer)(zero_pad2)

    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    else:
        return tf.keras.Model(inputs=inp, outputs=last)



# def run_main(argv):
#    del argv
#    kwargs = {'epochs': FLAGS.epochs, 'enable_function': FLAGS.enable_function,
#              'path': FLAGS.path, 'buffer_size': FLAGS.buffer_size,
#              'batch_size': FLAGS.batch_size}
#    main(**kwargs)
#
#
# def main(epochs, enable_function, path, buffer_size, batch_size):
#    path_to_folder = path
#
#    pix2pix_object = Pix2Pix2(epochs, enable_function)
#
#    train_dataset, _ = create_dataset(
#        os.path.join(path_to_folder, 'train/*.jpg'),
#        os.path.join(path_to_folder, 'test/*.jpg'),
#        buffer_size, batch_size)
#    checkpoint_pr = get_checkpoint_prefix()
#    print('Training...')
#
#    return pix2pix_object.train(train_dataset, checkpoint_pr)
#
#
# if __name__ == '__main__':
#    app.run(run_main)

