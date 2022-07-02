from collections import namedtuple
import time
import os
import pathlib
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def process_path(file_path, training=False):
    parts = tf.strings.split(file_path, os.path.sep)
    label = tf.where(CLASS_NAMES == parts[-2], 1, 0)

    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img, channels=CHANNELS, dtype=tf.float32, expand_animations=False)

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
    img = tf.math.multiply(img, 2)
    img = tf.math.subtract(img, 1)

    return img, label


def conv2d_block(layer_input, filters_local):
    output_layer = tf.keras.layers.Conv2D(filters=filters_local, kernel_size=KERNEL_SIZE, padding='same')(layer_input)
    output_layer = tf.keras.layers.ReLU()(output_layer)
    output_layer = tf.keras.layers.MaxPool2D(pool_size=2)(output_layer)
    return output_layer


def create_network():
    filters = FILTERS[0]
    model_input = tf.keras.Input(shape=(IMG_DIM, IMG_DIM, CHANNELS))
    model_output = conv2d_block(model_input, filters)

    while filters < FILTERS[1]:
        filters *= 2
        model_output = conv2d_block(model_output, filters)

    model_output = tf.keras.layers.GlobalAveragePooling2D()(model_output)
    model_output = tf.keras.layers.Dense(units=256)(model_output)
    model_output = tf.keras.layers.ReLU()(model_output)
    model_output = tf.keras.layers.Dense(units=len(CLASS_NAMES))(model_output)

    return tf.keras.Model(inputs=model_input, outputs=model_output)


def get_model_path(version=None):
    if version is None:
        return os.path.join('objects', 'model.h5')
    else:
        return os.path.join('objects', 'model_checkpoints', f'model_{version:03d}.h5')


def get_print_time(t):
    days = int(t / 86400)
    t = t - 86400 * days
    hours = int(t / 3600)
    t = t - 3600 * hours
    minutes = int(t / 60)
    t = t - 60 * minutes
    seconds = int(t)

    return ReadableTime(days, hours, minutes, seconds)


def plot_learning_curve():
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6.5, 6.5), dpi=600, constrained_layout=True)
    axs = axs.ravel()

    axs[0].plot(dict_loss["Images trained"], dict_loss["Loss (Training)"], label="Training", color="tab:blue")
    axs[0].plot(dict_loss["Images trained"], dict_loss["Loss (Validation)"], label="Validation", color="tab:orange")
    axs[0].legend()
    axs[0].set_xlim(left=0)
    axs[0].set_xlabel("Images trained")
    axs[0].set_ylabel("Loss")

    axs[1].plot(dict_loss["Images trained"], dict_loss["Accuracy (Training)"], label="Training", color="tab:blue")
    axs[1].plot(dict_loss["Images trained"], dict_loss["Accuracy (Validation)"], label="Validation", color="tab:orange")
    axs[1].legend()
    axs[1].set_xlim(left=0)
    axs[1].set_xlabel("Images trained")
    axs[1].set_ylabel("Accuracy")

    fig.savefig(os.path.join("logs", "learning_curve.png"))
    plt.close(fig)


def get_accuracy(y_real, y_model):
    model_index = np.argmax(y_model.numpy(), axis=1)
    real_index = np.argmax(y_real.numpy(), axis=1)
    return np.mean(model_index == real_index)


def get_loss_tensor(y_real, y_model):
    return loss_cce(y_real, y_model)


def get_validation_loss_accuracy():
    loss_valids = []
    accuracy_valids = []
    for images_valid, labels_valid in ds_valid:
        results_valid = model(images_valid)
        loss_valids.append(get_loss_tensor(labels_valid, results_valid).numpy())
        accuracy_valids.append(get_accuracy(results_valid, labels_valid))

    return np.mean(loss_valids), np.mean(accuracy_valids)


tuning_hyperparameters = False
model_version = 0

tf.random.set_seed(1)
ReadableTime = namedtuple('ReadableTime', ['days', 'hours', 'minutes', 'seconds'])

data_dir = pathlib.Path('.')
image_count = len(list(data_dir.glob(os.path.join('photos', '*', '*.jpg'))))
CLASS_NAMES = np.array([item.name for item in data_dir.glob(os.path.join('photos', '*'))])
pickle.dump(CLASS_NAMES, open(os.path.join('objects', 'CLASS_NAMES.pkl'), 'wb'))

if tuning_hyperparameters:
    num_test = int(0.3 * image_count)
    num_valid = int(0.1 * image_count)
else:
    num_test = int(0.05 * image_count)
    num_valid = int(0.05 * image_count)

list_ds = tf.data.Dataset.list_files(str(os.path.join('photos', '*', '*.jpg')))
list_ds.shuffle(buffer_size=image_count, reshuffle_each_iteration=False)
list_ds_test = list_ds.take(num_test)
list_ds_train_valid = list_ds.skip(num_test)
list_ds_valid = list_ds_train_valid.take(num_valid)
list_ds_train = list_ds_train_valid.skip(num_valid)

INPUT_DIM = 640
IMG_DIM = 256
BATCH_SIZE = 64
CHANNELS = 3
BUFFER_SIZE = 1024
FILTERS = (16, 256)
KERNEL_SIZE = 3
LOG_FREQUENCY = 12 * 60  # seconds

ds_train = list_ds_train.map(lambda x: process_path(x, training=True))
ds_train = ds_train.shuffle(buffer_size=BUFFER_SIZE).repeat().batch(BATCH_SIZE).repeat()
ds_valid = list_ds_valid.map(lambda x: process_path(x)).batch(BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)

if model_version == 0:
    model = create_network()
    dict_loss = {"Model version": [], "Images trained": [], "Time [s]": [],
                 "Loss (Training)": [], "Loss (Validation)": [],
                 "Accuracy (Training)": [], "Accuracy (Validation)": []}

    initial_batch_count = 0
    start_time = time.time()

else:
    model = tf.keras.models.load_model(get_model_path())

    dict_loss = pd.read_csv(os.path.join("logs", "loss.csv")).to_dict("list")
    initial_batch_count = int(dict_loss["Images trained"][-1] / BATCH_SIZE)
    start_time = time.time() - dict_loss["Time [s]"][-1]
    model_version = dict_loss["Model version"][-1]

with open(os.path.join('logs', 'model_summary.txt'), 'w') as f_model_summary:
    model.summary(print_fn=(lambda x: f_model_summary.write('{}\n'.format(x))))

last_status = time.time()
last_batch_count = initial_batch_count
last_status_loss = {"Loss (Training)": [], "Accuracy (Training)": []}

for batch_count, (images, labels) in enumerate(ds_train, start=initial_batch_count):

    with tf.GradientTape() as tape:
        model_result = model(images, training=True)
        loss = get_loss_tensor(labels, model_result)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    last_status_loss["Loss (Training)"].append(loss.numpy())
    last_status_loss["Accuracy (Training)"].append(get_accuracy(model_result, labels))

    if time.time() - last_status > LOG_FREQUENCY:
        model_version += 1
        model.save(get_model_path(model_version))
        model.save(get_model_path())

        total_time = time.time() - start_time
        print_time = get_print_time(total_time)
        images_per_hour = BATCH_SIZE * (batch_count - last_batch_count) / (time.time() - last_status) * 3600
        validation_loss, validation_accuracy = get_validation_loss_accuracy()

        dict_loss["Model version"].append(model_version)
        dict_loss["Images trained"].append(BATCH_SIZE * batch_count)
        dict_loss["Time [s]"].append(total_time)
        dict_loss["Loss (Training)"].append(np.mean(last_status_loss["Loss (Training)"]))
        dict_loss["Loss (Validation)"].append(validation_loss)
        dict_loss["Accuracy (Training)"].append(np.mean(last_status_loss["Accuracy (Training)"]))
        dict_loss["Accuracy (Validation)"].append(validation_accuracy)

        pd.DataFrame.from_dict(dict_loss).to_csv(os.path.join("logs", "loss.csv"), index=False)
        plot_learning_curve()

        print(
            f"Version: {dict_loss['Model version'][-1]:4d} | "
            f"Images trained: {dict_loss['Images trained'][-1]:8d} | "
            f"Time: {print_time.days}:{print_time.hours}:{print_time.minutes:02d}:{print_time.seconds:02d} | "
            f"Loss (T): {dict_loss['Loss (Training)'][-1]:6.2f} | "
            f"Loss (V): {dict_loss['Loss (Validation)'][-1]:6.2f} | "
            f"Accuracy (T): {dict_loss['Accuracy (Training)'][-1]:6.2f} | "
            f"Accuracy (V): {dict_loss['Accuracy (Validation)'][-1]:6.2f} | "
            f"Images per hour: {images_per_hour:6.0f}"
        )

        last_status = time.time()
        last_batch_count = batch_count + 0
        last_status_loss = {"Loss (Training)": [], "Accuracy (Training)": []}

"""

ds_test = list_ds_test.map(lambda x: process_path(x, IMG_DIM, CLASS_NAMES)).batch(BATCH_SIZE)
test_log = model.evaluate(ds_test, verbose=0)

with open(os.path.join('logs', f'test_log-{time_stamp}.txt'), 'w') as f_test_log:
    f_test_log.write('Test Loss:     {:.3f}\nTest Accuracy: {:.3f}'.format(test_log[0], test_log[1]))

"""
