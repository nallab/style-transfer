"""
CycleGAN + VGG
"""
import os
import time
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D

import matplotlib.pyplot as plt
import datetime
from models import pix2pix
from tensorflow.keras.models import load_model


class CycleGAN(object):
    """
    CycleGAN class.
    """

    def __init__(self, epochs, checkpoint_dir, cycle_lambda, content_lambda):
        self.epochs = epochs
        self.cycle_lambda = cycle_lambda
        self.content_lambda = content_lambda
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

        # Define our metrics
        self.generator_g_loss = tf.keras.metrics.Mean('generator_g_loss', dtype=tf.float32)
        self.generator_f_loss = tf.keras.metrics.Mean('generator_f_loss', dtype=tf.float32)
        self.discriminator_x_loss = tf.keras.metrics.Mean('discriminator_x_loss', dtype=tf.float32)
        self.discriminator_y_loss = tf.keras.metrics.Mean('discriminator_y_loss', dtype=tf.float32)
        self.content_loss = tf.keras.metrics.Mean('content_loss', dtype=tf.float32)
        self.cycle_loss = tf.keras.metrics.Mean('cycle_loss', dtype=tf.float32)

        #  CheckPoint
        self.checkpoint_dir = checkpoint_dir
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
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=5)

        # input_shape = (256, 256, 3)
        base_model = load_model('vgg16.h5')
        # base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
        tmp_vgg_output = base_model.get_layer("block4_conv3").output
        tmp_vgg_output = Conv2D(512, (3, 3), activation='linear', padding='same',
                                name='block4_conv4')(tmp_vgg_output)
        self.vgg = tf.keras.Model(inputs=base_model.input, outputs=tmp_vgg_output)

    def load_checkpoint(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')
        else:
            print('Do not load checkpoint...')

    def discriminator_loss(self, real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)
        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5

    def generator_loss(self, generated):
        return self.loss_obj(tf.ones_like(generated), generated)

    def calc_cycle_loss(self, real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

        return self.cycle_lambda * loss1

    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.lambda_value * 0.5 * loss

    def calc_content_loss(self, real_image, cycled_image):
        vgg_real_image = self.vgg(real_image)
        vgg_cycled_image = self.vgg(cycled_image)
        loss = tf.reduce_mean(tf.abs(vgg_real_image - vgg_cycled_image))

        return self.content_lambda * loss

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

            # L_con = E[ || VGG(G(p)) - VGG(p) || ]
            content_loss = self.calc_content_loss(real_x, fake_y)

            # Total generator loss = adversarial loss + cycle loss
            # total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            # total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            total_gen_g_loss = gen_g_loss + total_cycle_loss + content_loss + self.identity_loss(real_y, same_y)

            # ??間違い g の loss だと contetnは入力の前と出力のl1ノルムの必要があるのでは？？？
            # total_gen_g_loss = gen_g_loss + total_cycle_loss + self.content_loss(real_y, same_y) + self.identity_loss(
            #    real_y, same_y)

            # ドメインY側のcontent_loss は省いた
            # total_gen_f_loss = gen_f_loss + total_cycle_loss + self.content_loss(real_x, same_x) + self.identity_loss(real_x, same_x)

            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)
            # total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss, self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss, self.generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss, self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, self.discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                                       self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                                       self.generator_f.trainable_variables))

        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                           self.discriminator_x.trainable_variables))

        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                           self.discriminator_y.trainable_variables))

        # Loss for TensorBoard
        self.generator_g_loss(total_gen_g_loss)
        self.generator_f_loss(total_gen_f_loss)
        self.discriminator_x_loss(disc_x_loss)
        self.discriminator_y_loss(disc_y_loss)
        self.content_loss(content_loss)
        self.cycle_loss(total_cycle_loss)

    def train(self, domain_a, domain_b, test_a, test_b):
        """
        Train the CycleGAN
        """
        self.load_checkpoint(self.checkpoint_dir)

        #  Set up summary writers
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        if not os.path.exists("output"):
            os.mkdir("output")
            print("Create output directory")

        for epoch in range(self.epochs):
            print("[*] Epoch {}".format(epoch))
            start = time.time()
            n = 0
            for image_x, image_y in tf.data.Dataset.zip((domain_a, domain_b)):
                self.train_step(image_x, image_y)
                if n % 10 == 0:
                    print('.', end='')
                n += 1

            # Set scalar for TensorBoard
            with train_summary_writer.as_default():
                tf.summary.scalar('Generator_g_Loss', self.generator_g_loss.result(), step=epoch)
                tf.summary.scalar('Generator_f_Loss', self.generator_f_loss.result(), step=epoch)
                tf.summary.scalar('Discriminator_x_Loss', self.discriminator_x_loss.result(), step=epoch)
                tf.summary.scalar('Discriminator_y_Loss', self.discriminator_y_loss.result(), step=epoch)
                tf.summary.scalar('Content_Loss', self.content_loss.result(), step=epoch)
                tf.summary.scalar('Cycle_Loss', self.cycle_loss.result(), step=epoch)

            # Save checkpoint.
            if (epoch + 1) % 5 == 0:
               ckpt_save_path = self.checkpoint_manager.save()
               print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

            # Output transferred images.
            if (epoch + 1) % 10 == 0:
                output_dir_g = "output/" + "G" + str(epoch + 1)
                self.test_transfer(test_a, self.generator_g, output_dir_g)
                print('Transferring test_A for epoch {} at {}'.format(epoch + 1, output_dir_g))

                output_dir_f = "output/" + "F" + str(epoch + 1)
                self.test_transfer(test_b, self.generator_f, output_dir_f)
                print('Transferring test_B for epoch {} at {}'.format(epoch + 1, output_dir_f))

            # Monitoring.
            if (epoch + 1) % 50 == 0:
                if os.path.exists("slack"):
                    args = "-n 実験 -m Epoch:" + str(epoch + 1)
                    os.system("./slack " + args)

            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))

            template = 'gen_g_loss: {}, gen_f_loss: {}, disc_x_loss: {}, disc_y_loss: {}, con_loss: {}, cyc_loss: {}'
            print(template.format(
                self.generator_g_loss.result(),
                self.generator_f_loss.result(),
                self.discriminator_x_loss.result(),
                self.discriminator_y_loss.result(),
                self.content_loss.result(),
                self.cycle_loss.result(),
            ))

            # Reset metrics every epoch
            self.generator_g_loss.reset_states()
            self.generator_f_loss.reset_states()
            self.discriminator_x_loss.reset_states()
            self.discriminator_y_loss.reset_states()
            self.content_loss.reset_states()
            self.cycle_loss.reset_states()

        # Store model
        # self.generator_g.summary()
        # self.generator_g.save('g_g.h5')
        # self.generator_f.save('g_f.h5')
        # self.discriminator_x.save('d_x.h5')
        # self.discriminator_y.save('d_y.h5')

    def test_transfer(self, test_input, generator, output_dir):
        """
        経過観察用のテスト関数
        """
        os.mkdir(output_dir)

        n = int(0)
        for inp in test_input:
            prediction = generator(inp)
            plt.figure(figsize=(12, 12))
            display_list = [inp[0], prediction[0]]
            title = ['Input Image', 'Predicted Image']
            for i in range(2):
                plt.subplot(1, 2, i + 1)
                plt.title(title[i])
                # getting the pixel values between [0, 1] to plot it.
                plt.imshow(display_list[i] * 0.5 + 0.5)
                plt.axis('off')
            plt.savefig(output_dir + "/fig" + str(n))
            n = n + 1
            plt.close()

    def test(self, test_input):
        """Test the CycleGAN"""
        # 学習ずみ保存モデルからロードする、なければ終了
        # if not os.path.exists('g_g.h5'):
        #     print("学習ずみモデルがないよー")
        #     return

        self.load_checkpoint(self.checkpoint_dir)

        # generator = tf.keras.models.load_model('g_g.h5')
        # generator = self.generator_g
        generator = self.generator_f
        # print(generator.summary())

        n = int(0)
        for inp in test_input:
            prediction = generator(inp)
            plt.figure(figsize=(12, 12))
            display_list = [inp[0], prediction[0]]
            title = ['Input Image', 'Predicted Image']
            for i in range(2):
                plt.subplot(1, 2, i + 1)
                plt.title(title[i])
                # getting the pixel values between [0, 1] to plot it.
                plt.imshow(display_list[i] * 0.5 + 0.5)
                plt.axis('off')
            plt.savefig("fig" + str(n))
            n = n + 1
            plt.close()
#
#             sobel_inp = tf.image.sobel_edges(inp)
#             sobel_prediction = tf.image.sobel_edges(prediction)
#             # grad_inp_square = tf.math.reduce_sum(sobel_inp**2,axis=-1)
#             # grad_prediction_square = tf.math.reduce_sum(sobel_prediction**2,axis=-1)
#             # grad_inp = tf.sqrt(grad_inp_square)
#             # grad_prediction = tf.sqrt(grad_prediction_square)
#
#             # VGG
#             vgg_real_image = self.vgg(inp)
#             # print(type(vgg_real_image))
#             vgg_cycled_image = self.vgg(prediction)
#             # print(type(vgg_cycled_image))
#
#             # display_list = [sobel_inp, sobel_prediction]
#             # display_list = [grad_inp, grad_prediction_square]
#             display_list = [vgg_real_image, vgg_cycled_image]
#
#             plt.figure(figsize=(12, 12))
#             # display_list = [inp[0], prediction[0]]
#             title = ['Input Image', 'Predicted Image']
#             for i in range(2):
#                 plt.subplot(1, 2, i + 1)
#                 plt.title(title[i])
#                 # getting the pixel values between [0, 1] to plot it.
#                 # plt.imshow(display_list[i] * 0.5 + 0.5)
#                 # self.imshow(tf.abs(display_list[i][...,1]/4+0.5,display_list[i][...,0]/4+0.5))
#                 # self.imshow(display_list[i][...,0]/4)
#                 # self.imshow(display_list[i])
#                 plt.axis('off')
#             plt.savefig("fig" + str(n))
#             n = n + 1
