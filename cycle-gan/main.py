import os
import time
import numpy as np
from glob import glob
import tensorflow as tf

from absl import app
from absl import flags

import pix2pix
from util import utils

AUTOTUNE = tf.data.experimental.AUTOTUNE

FLAGS = flags.FLAGS

flags.DEFINE_integer('buffer_size', 1000, 'Shuffle buffer size')
flags.DEFINE_integer('batch_size', 1, 'Batch size')
flags.DEFINE_integer('epochs', 40, 'Number of epochs')
flags.DEFINE_string('checkpoint_path', "./checkpoints/train", 'Path to the data folder')
# flags.DEFINE_string('path', None, 'Path to the data folder')


IMG_WIDTH = 256
IMG_HEIGHT = 256


class CycleGAN(object):
    """
    CycleGAN class.
    """

    def __init__(self, epochs, checkpoint_path):
        self.epochs = epochs
        self.lambda_value = 10
        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        # Batch Normalization
        # self.generator_g = pix2pix.unet_generator(output_channels=3)
        # self.generator_f = pix2pix.unet_generator(output_channels=3)
        # self.discriminator_x = pix2pix.discriminator(target=False)
        # self.discriminator_y = pix2pix.discriminator(target=False)

        # Instance Normalization
        self.generator_g = pix2pix.unet_generator(output_channels=3, norm_type='instancenorm')
        self.generator_f = pix2pix.unet_generator(output_channels=3, norm_type='instancenorm')
        self.discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
        self.discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

        #  CheckPoint
        self.checkpoint_path = checkpoint_path
        self.checkpoint = tf.train.Checkpoint(
            generator_g=self.generator_g,
            generator_f=self.generator_f,
            discriminator_x=self.discriminator_x,
            discriminator_y=self.discriminator_y,
            generator_g_optimizer=self.generator_g_optimizer,
            generator_f_optimizer=self.generator_f_optimizer,
            discriminator_x_optimizer=self.discriminator_x_optimizer,
            discriminator_y_optimizer=self.discriminator_y_optimizer
        )
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_path, max_to_keep=5)

    def set_checkpoint(self):
        # if a checkpoint exists, restore the latest checkpoint.
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    def discriminator_loss(self, real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)
        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5

    def generator_loss(self, generated):
        return self.loss_obj(tf.ones_like(generated), generated)

    def calc_cycle_loss(self, real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

        return self.lambda_value * loss1

    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.lambda_value * 0.5 * loss

    @tf.function
    def train_step(self, real_x, real_y):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)

            total_cycle_loss = self.calc_cycle_loss(real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss,
                                              self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss,
                                              self.generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss,
                                                  self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss,
                                                  self.discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                                       self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                                       self.generator_f.trainable_variables))

        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                           self.discriminator_x.trainable_variables))

        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                           self.discriminator_y.trainable_variables))

    def train(self, domain_a, domain_b):
        """
        Train the CycleGAN
        """

        for epoch in range(self.epochs):
            start = time.time()
            n = 0
            for image_x, image_y in tf.data.Dataset.zip((domain_a, domain_b)):
                self.train_step(image_x, image_y)
                if n % 10 == 0:
                    print('.', end='')
                n += 1

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.checkpoint_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                    ckpt_save_path))

            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                               time.time() - start))


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


def preprocess_image_test(image, label):
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


def run_main(argv):
    del argv
    kwargs = {
        'epochs': FLAGS.epochs,
        'buffer_size': FLAGS.buffer_size,
        'batch_size': FLAGS.batch_size,
        'checkpoint_path': FLAGS.checkpoint_path,
    }

    main(**kwargs)


def main(epochs, buffer_size, batch_size, checkpoint_path):
    print("START!!")

    cycle_gan_object = CycleGAN(epochs, checkpoint_path)
    cycle_gan_object.set_checkpoint()

    new_data = tf.data.TFRecordDataset('new.tfrec').map(preprocess_image)
    new_images = new_data.map(preprocess_image_train, num_parallel_calls=AUTOTUNE) \
        .cache().shuffle(buffer_size).batch(batch_size)
    old_data = tf.data.TFRecordDataset('old.tfrec').map(preprocess_image)
    old_images = old_data.map(preprocess_image_train, num_parallel_calls=AUTOTUNE) \
        .cache().shuffle(buffer_size).batch(batch_size)

    return cycle_gan_object.train(old_images, new_images)


if __name__ == '__main__':
    app.run(run_main)
