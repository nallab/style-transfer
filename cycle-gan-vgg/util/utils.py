"""Utils
"""
import pprint
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime

pp = pprint.PrettyPrinter()


def imread(path, is_grayscale=False):
    if is_grayscale:
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def imsave(images, path):
    """

    :param images:
    :param path:
    :return:
    """
    return scipy.misc.imsave(path, images)


def inverse_transform(images):
    return (images + 1) / 2.


def generate_images(model, test_input):
    prediction = model(test_input)

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig('fig.png')

