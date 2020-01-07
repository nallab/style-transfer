import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

batch_size = 32
file_name = 'vgg16'
test_dir = 'dataset/test'
display_dir = 'dataset/display'
label = ['is_doraemon', 'is_not_doraemon']

# load model and weights
# json_string = open(file_name + '.json').read()
# model = model_from_json(json_string)
model = load_model(file_name + '.h5')

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# data generate
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# evaluate model
score = model.evaluate_generator(test_generator)
print('\n test loss:', score[0])
print('\n test_acc:', score[1])

# predict model and display images
files = os.listdir(display_dir)
print(files)
img = random.sample(files, 36)

plt.figure(figsize=(10, 10))
for i in range(36):
    temp_img = load_img(os.path.join(display_dir, img[i]), target_size=(256, 256))
    plt.subplot(6, 6, i + 1)
    plt.imshow(temp_img)
    # Images normalization
    temp_img_array = img_to_array(temp_img)
    temp_img_array = temp_img_array.astype('float32') / 255.0
    temp_img_array = temp_img_array.reshape((1, 256, 256, 3))
    # predict image
    img_pred = model.predict(temp_img_array)
    plt.title(label[np.argmax(img_pred)])
    # eliminate xticks,yticks
    plt.xticks([]), plt.yticks([])

plt.show()
