"""
準備用：データセットをTFRecord形式にする
"""
import tensorflow as tf
from glob import glob
from keras.preprocessing.image import load_img, img_to_array

OLD_PATH = '/home/datasets/old_data'
NEW_PATH = '/home/datasets/new_data'

old_paths = glob(OLD_PATH + "/image_*")
new_paths = glob(NEW_PATH + "/image_*")


# old_label = [str(0) for i in range(len(old_paths))]
# new_label = [str(1) for i in range(len(new_paths))]


def save_tfrec(paths, name):
    images_ds = tf.data.Dataset.from_tensor_slices(paths).map(tf.io.read_file)
    tfrec = tf.data.experimental.TFRecordWriter(name + '.tfrec')
    tfrec.write(images_ds)


save_tfrec(old_paths, 'old')
save_tfrec(new_paths, 'new')

