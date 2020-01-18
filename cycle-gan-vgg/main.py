import os
import time
import numpy as np
from glob import glob
import tensorflow as tf

from absl import app
from absl import flags

from models import cycleGAN
from util import utils

AUTOTUNE = tf.data.experimental.AUTOTUNE

FLAGS = flags.FLAGS

flags.DEFINE_integer('buffer_size', 1000, 'Shuffle  size')  # default 1000
flags.DEFINE_integer('batch_size', 1, 'Batch size')
flags.DEFINE_integer('epochs', 200, 'Number of epochs')  # default 40
flags.DEFINE_string('checkpoint_dir', "./checkpoints/train", 'Path to the data folder')
flags.DEFINE_bool('vgg', True, "Use VGG")

# Parameter
flags.DEFINE_integer('cycle_loss_weight', 10, 'Value of cycle_loss weight')
flags.DEFINE_integer('content_loss_weight', 1, 'Value of content_loss weight')
# flags.DEFINE_string('path', None, 'Path to the data folder')

IMG_WIDTH = 256
IMG_HEIGHT = 256


def random_crop(image):
    cropped_image = tf.image.random_crop(
        image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image


# normalizing the images to [-1, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def random_jitter(image):
    # resizing to 286 x 286 x 3
    image = tf.image.resize(image, [286, 286],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # randomly cropping to 256 x 256 x 3
    image = random_crop(image)

    # random mirroring
    image = tf.image.random_flip_left_right(image)

    return image


# 第2引数に label があったが、使っていなかったので削除
def preprocess_image_train(image):
    image = random_jitter(image)
    image = normalize(image)
    return image


def preprocess_image_test(image):
    image = normalize(image)
    return image


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])

    return image


# Run the trained model on the test dataset
# for inp in new_images.take(5):
#    print(inp)
#    generate_images(generator_f, inp)

# def get_test_data():
#    dataset, metadata = tfds.load('cycle_gan/horse2zebra',
#                                  with_info=True, as_supervised=True)
#
#    train_horses, train_zebras = dataset['trainA'], dataset['trainB']
#    test_horses, test_zebras = dataset['testA'], dataset['testB']
#
#    train_horses = train_horses.map(
#        preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
#        1000).batch(1)
#
#    train_zebras = train_zebras.map(
#        preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
#        1000).batch(1)
#
#    test_horses = test_horses.map(
#        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
#        1000).batch(1)
#
#    test_zebras = test_zebras.map(
#        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
#        1000).batch(1)
#
#    return train_horses, train_zebras, test_horses, test_zebras

def run_main(argv):
    del argv
    kwargs = {
        'epochs': FLAGS.epochs,
        'buffer_size': FLAGS.buffer_size,
        'batch_size': FLAGS.batch_size,
        'checkpoint_dir': FLAGS.checkpoint_dir,
        'vgg': FLAGS.vgg,
        'cycle_lambda': FLAGS.cycle_loss_weight,
        'content_lambda': FLAGS.content_loss_weight,
    }
    print("----------")
    print(kwargs)
    print("----------")

    main(**kwargs)


def main(epochs, buffer_size, batch_size, checkpoint_dir, vgg, cycle_lambda, content_lambda):
    print("[*]START!!")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print("[*]{} not found. create it".format(checkpoint_dir))

    cycle_gan_object = cycleGAN.CycleGAN(epochs, checkpoint_dir, vgg, cycle_lambda, content_lambda)

    # old_images, new_images, test_horses, _ = get_test_data()

    # Input train data.
    new_data = tf.data.TFRecordDataset('new.tfrec').map(preprocess_image)
    new_images = new_data.map(preprocess_image_train, num_parallel_calls=AUTOTUNE) \
        .take(buffer_size).cache().shuffle(buffer_size).batch(batch_size)
    old_data = tf.data.TFRecordDataset('old.tfrec').map(preprocess_image)
    old_images = old_data.map(preprocess_image_train, num_parallel_calls=AUTOTUNE) \
        .take(buffer_size).cache().shuffle(buffer_size).batch(batch_size)

    # Input test data.
    test_new_data = tf.data.TFRecordDataset('test_new.tfrec').map(preprocess_image)
    test_new_images = test_new_data.map(preprocess_image_test, num_parallel_calls=AUTOTUNE) \
        .cache().batch(batch_size)
    test_old_data = tf.data.TFRecordDataset('test_old.tfrec').map(preprocess_image)
    test_old_images = test_old_data.map(preprocess_image_test, num_parallel_calls=AUTOTUNE) \
        .cache().batch(batch_size)

    # Test
    # cycle_gan_object.test(test_images)

    # Train
    cycle_gan_object.train(old_images, new_images, test_old_images, test_new_images)

    # cycle_gan_object.test(test_horses)


if __name__ == '__main__':
    app.run(run_main)
