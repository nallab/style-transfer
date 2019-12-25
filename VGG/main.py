import tensorflow as tf
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint

n_categories = 2
batch_size = 32
train_dir = './dataset/train'
validation_dir = './dataset/validation'
file_name = 'vgg16'

base_model = VGG16(weights='imagenet', include_top=False,
                   # input_tensor=Input(shape=(224, 224, 3)))
                   input_tensor=Input(shape=(256, 256, 3)))
# add new layers instead of FC networks
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
prediction = Dense(n_categories, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=prediction)

# fix weights before VGG16 14layers
for layer in base_model.layers[:15]:
    layer.trainable = False

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

os.makedirs('models', exist_ok=True)
model_checkpoint = ModelCheckpoint(
    filepath=os.path.join('models', 'model_{epoch:02d}_{val_loss:.2f}.h5'),
    monitor='val_loss',
    verbose=1)

hist = model.fit_generator(train_generator,
                           epochs=100,
                           verbose=1,
                           validation_data=validation_generator,
                           callbacks=[model_checkpoint])

# save weights
model.save(file_name + '.h5')
