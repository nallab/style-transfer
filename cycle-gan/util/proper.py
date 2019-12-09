"""
準備用：データセットをTFRecord形式にする
"""
import tensorflow as tf
from absl import flags
from absl import app
from glob import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array

FLAGS = flags.FLAGS

flags.DEFINE_string('old_image_path', '/home/datasets/old_data', 'Path to the data folder')
flags.DEFINE_string('new_image_path', '/home/datasets/new_data', 'Path to the data folder')


# old_label = [str(0) for i in range(len(old_paths))]
# new_label = [str(1) for i in range(len(new_paths))]


def save_tfrec(paths, name):
    images_ds = tf.data.Dataset.from_tensor_slices(paths).map(tf.io.read_file)
    tfrec = tf.data.experimental.TFRecordWriter(name + '.tfrec')
    tfrec.write(images_ds)


def run_main(argv):
    del argv
    kwargs = {
        'old_path': FLAGS.old_image_path,
        'new_path': FLAGS.new_image_path,
    }
    main(**kwargs)


def main(old_path, new_path):
    old_paths = glob(old_path + "/image_*")
    new_paths = glob(new_path + "/image_*")

    save_tfrec(old_paths, 'old')
    save_tfrec(new_paths, 'new')


if __name__ == '__main__':
    app.run(run_main)
