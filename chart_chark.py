import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pandas as pd
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
# import tensorflow_datasets as tfds
import tensorflow as tf

batch_size = 64
img_width = None
img_height = None

name = 'chart_high_fidelity_200h_20_proper'
# df = pd.read_pickle('chart_high_fidelity_100h_20_single_label_with_start.pkl')
df = pd.read_csv(f'/tf/stuff/data/{name}.csv')

# df = df[df['label'] != 1]
# df = pd.concat([df, df, df, df, df, df, df])

num_classes = len(list(df['label'].unique()))

train_slice = int(df.shape[0] * .7)

train_df = df[:train_slice]

def create_logit_slice(label, labels):
    new_logits = [0] * len(labels)
    mid = list(labels).index(labels[int(len(labels) / 2)])
    to = list(labels).index(label)
    if mid > to:
        for i in range(to,mid):
            new_logits[i] = 1
    if mid < to:
        for i in range(mid, to):
            new_logits[i] = 1
    else:
        new_logits[mid] = 1
    return new_logits

def create_logit(label, labels):
    new_logits = [0] * len(labels)
    new_logits[list(labels).index(label)] = 1
    return new_logits

labels = sorted(df['label'].unique())
label_logits = [create_logit_slice(label, labels) for label in df['label']]


image_data = []

for file in list(train_df['file_path']):
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    
    if not img_width:
        img_height, img_width, channels = img.shape
    image_data.append(img)

train_dataset = tf.data.Dataset.from_tensor_slices((image_data, label_logits[:train_slice])).batch(batch_size)

val_df = df[train_slice:]

image_data = []
labels = list(val_df['label'])

for file in list(val_df['file_path']):
    image_data.append(cv2.imread(file, cv2.IMREAD_UNCHANGED))

val_dataset = tf.data.Dataset.from_tensor_slices((image_data, label_logits[train_slice:])).batch(batch_size)

preprocessor = Sequential([
    layers.experimental.preprocessing.Normalization(),
    layers.experimental.preprocessing.Resizing(img_width, img_height),
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(factor=0.02),
    layers.experimental.preprocessing.RandomZoom(
        height_factor=0.2, width_factor=0.2
        ),
    layers.experimental.preprocessing.Rescaling(1./255)
])

preprocessed_train_ds = train_dataset.map(lambda x, y: (preprocessor(x), y))
image_batch, labels_batch = next(iter(preprocessed_train_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

preprocessed_val_ds = val_dataset.map(lambda x, y: (preprocessor(x), y))
image_batch, labels_batch = next(iter(preprocessed_val_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))



vc = df['label'].value_counts()
initial_bias = [(1 / x) * sum(vc) / len(vc) for x in vc]
output_bias = tf.keras.initializers.Constant(initial_bias)

model = Sequential([
    
    #     layers.experimental.preprocessing.Rescaling(1./255),
    #     layers.GaussianNoise(.2),
    layers.experimental.preprocessing.Resizing(img_width, img_height),
    # layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(factor=0.02),
    layers.experimental.preprocessing.RandomZoom(
        height_factor=0.2, width_factor=0.2
        ),
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(8, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.1),
    layers.Flatten(),

    #   layers.Dense(10, activation='relu'),
    layers.Dense(num_classes, activation=tf.keras.activations.hard_sigmoid, bias_initializer=output_bias)
])

# model = tf.keras.models.load_model(name)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)

epochs=10000
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    # callbacks=[callback],
    class_weight={x:y for x, y in enumerate(initial_bias)}
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig(fname=name)

model.save(name)