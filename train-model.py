import datetime
import os
import pathlib
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def process_path(file_path, IMG_DIM, CLASS_NAMES, training=False):
    parts = tf.strings.split(file_path, os.path.sep)
    label = (CLASS_NAMES == parts[-2])

    INPUT_DIM = 640

    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    if training:
        noise = 0.2
        crop_dim = tf.random.uniform([], minval=IMG_DIM, maxval=INPUT_DIM, dtype=tf.dtypes.int32)
        img = tf.image.random_crop(img, size=(crop_dim, crop_dim, 3))
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=noise)
        img = tf.image.random_contrast(img, lower=(1 / (1 + noise)), upper=(1 + noise))
        img = tf.image.random_saturation(img, lower=(1 / (1 + noise)), upper=(1 + noise))
        img = tf.image.random_hue(img, max_delta=noise)

    if not training:
        img = tf.image.resize_with_crop_or_pad(img, target_height=INPUT_DIM, target_width=INPUT_DIM)

    img = tf.image.resize(img, size=(IMG_DIM, IMG_DIM))

    return img, label


tuning_hyperparameters = False
initial_training = False

tf.random.set_seed(1)

t = datetime.datetime.now()
time_stamp = f'{t.year}{t.month}{t.day}{t.hour}{t.minute}{t.second}'

data_dir = pathlib.Path('.')
image_count = len(list(data_dir.glob(os.path.join('photos', '*', '*.jpg'))))
CLASS_NAMES = np.array([item.name for item in data_dir.glob(os.path.join('photos', '*'))])
pickle.dump(CLASS_NAMES, open(os.path.join('objects', 'CLASS_NAMES.pkl'), 'wb'))
list_ds = tf.data.Dataset.list_files(str(os.path.join('photos', '*', '*.jpg')))

if tuning_hyperparameters:
    num_test = int(0.3 * image_count)
    num_valid = int(0.1 * image_count)
else:
    num_test = int(0.05 * image_count)
    num_valid = int(0.05 * image_count)

num_train = image_count - num_valid - num_test

list_ds.shuffle(buffer_size=image_count, reshuffle_each_iteration=False)
list_ds_test = list_ds.take(num_test)
list_ds_train_valid = list_ds.skip(num_test)
list_ds_valid = list_ds_train_valid.take(num_valid)
list_ds_train = list_ds_train_valid.skip(num_valid)

IMG_DIM = 256
BATCH_SIZE = 64
buffer_size = 512
epochs = 100
steps_per_epoch = int(np.ceil(num_train / BATCH_SIZE))

ds_train = list_ds_train.map(lambda x: process_path(x, IMG_DIM, CLASS_NAMES, training=True))
ds_train = ds_train.shuffle(buffer_size=buffer_size).repeat().batch(BATCH_SIZE)
ds_valid = list_ds_valid.map(lambda x: process_path(x, IMG_DIM, CLASS_NAMES)).batch(BATCH_SIZE)

if initial_training:

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_DIM, IMG_DIM, 3)),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation=None)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

else:
    model = tf.keras.models.load_model(os.path.join('objects', 'model.h5'))

with open(os.path.join('logs', f'model_summary-{time_stamp}.txt'), 'w') as f_model_summary:
    model.summary(print_fn=(lambda x: f_model_summary.write('{}\n'.format(x))))

train_log = model.fit(ds_train, validation_data=ds_valid, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=2)

ds_test = list_ds_test.map(lambda x: process_path(x, IMG_DIM, CLASS_NAMES)).batch(BATCH_SIZE)
test_log = model.evaluate(ds_test, verbose=0)

with open(os.path.join('logs', f'test_log-{time_stamp}.txt'), 'w') as f_test_log:
    f_test_log.write('Test Loss:     {:.3f}\nTest Accuracy: {:.3f}'.format(test_log[0], test_log[1]))

model.save(os.path.join('objects', 'model.h5'))

hist = train_log.history

n_epochs = np.arange(len(hist['loss'])) + 1

df_hist = pd.DataFrame.from_dict(hist)
df_hist['epoch'] = n_epochs
df_hist.to_csv(os.path.join('logs', f'train_log_history-{time_stamp}.csv'), index=False)

fig = plt.figure(figsize=(6.5, 6.5), dpi=600)

ax = fig.add_subplot(2, 1, 1)
ax.plot(n_epochs, hist['loss'], '-', label='Training')
ax.plot(n_epochs, hist['val_loss'], '--', label='Validation')
ax.legend()
ax.set_xlabel('')
ax.set_ylabel('Loss')

ax = fig.add_subplot(2, 1, 2)
ax.plot(n_epochs, hist['accuracy'], '-', label='Training')
ax.plot(n_epochs, hist['val_accuracy'], '--', label='Validation')
ax.legend()
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')

plt.tight_layout()
plt.savefig(os.path.join('logs', f'train_log_history-{time_stamp}.png'))
plt.show()
