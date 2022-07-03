from collections import namedtuple
import time
import os
import pathlib
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


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
    img = tf.clip_by_value(img, clip_value_min=0, clip_value_max=1)
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
        return os.path.join(S_OBJECTS, sub_dir, 'model.h5')
    else:
        return os.path.join(S_OBJECTS, sub_dir, S_MODEL_CHECKPOINTS, f'model_{version:03d}.h5')


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

    axs[0].plot(dict_loss[S_IMAGES_TRAINED], dict_loss[S_LOSS_T], label="Training", color="tab:blue")
    axs[0].plot(dict_loss[S_IMAGES_TRAINED], dict_loss[S_LOSS_V], label="Validation", color="tab:orange")
    axs[0].legend()
    axs[0].set_xlim(left=0)
    axs[0].set_xlabel(S_IMAGES_TRAINED)
    axs[0].set_ylabel("Loss")

    axs[1].plot(dict_loss[S_IMAGES_TRAINED], dict_loss[S_ACCURACY_T], label="Training", color="tab:blue")
    axs[1].plot(dict_loss[S_IMAGES_TRAINED], dict_loss[S_ACCURACY_V], label="Validation", color="tab:orange")
    axs[1].legend()
    axs[1].set_xlim(left=0)
    axs[1].set_xlabel(S_IMAGES_TRAINED)
    axs[1].set_ylabel("Accuracy")

    fig.savefig(os.path.join(S_LOGS, sub_dir, "learning_curve.png"))
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


def get_train_validation_test_folds():
    validation_dict = {}
    test_dict = {}
    train_dict = {}

    list_train = []
    list_validation = []
    list_test = []

    for class_name in CLASS_NAMES:
        validation_dict[class_name] = []
        test_dict[class_name] = []
        train_dict[class_name] = []
        class_list = list(data_dir.glob(os.path.join('photos', f"{class_name}", '*.jpg')))
        for n_sorted_images, class_image_path in enumerate(class_list, start=0):
            class_image_path_string = str(class_image_path)

            if n_sorted_images == 0:
                train_dict[class_name].append(class_image_path_string)

            elif (len(train_dict[class_name]) / n_sorted_images) < TRAIN_SPLIT:
                train_dict[class_name].append(class_image_path_string)

            elif (len(validation_dict[class_name]) / n_sorted_images) < VALIDATION_SPLIT:
                validation_dict[class_name].append(class_image_path_string)

            elif (len(test_dict[class_name]) / n_sorted_images) < TEST_SPLIT:
                test_dict[class_name].append(class_image_path_string)

        print(f"{class_name} | "
              f"Train: {len(train_dict[class_name]) / len(class_list):.1%} | "
              f"Validation: {len(validation_dict[class_name]) / len(class_list):.1%} | "
              f"Test: {len(test_dict[class_name]) / len(class_list):.1%}")

        list_train.extend(train_dict[class_name])
        list_validation.extend(validation_dict[class_name])
        list_test.extend(test_dict[class_name])
        random.shuffle(list_train)
        random.shuffle(list_validation)
        random.shuffle(list_test)

    return list_train, list_validation, list_test


S_MODEL_VERSION = "Model version"
S_IMAGES_TRAINED = "Images trained"
S_TIME = "Time [s]"
S_LOSS_T = "Loss (Training)"
S_LOSS_V = "Loss (Validation)"
S_ACCURACY_T = "Accuracy (Training)"
S_ACCURACY_V = "Accuracy (Validation)"
S_COMPLETE = "complete"
S_TUNING = "tuning"
S_LOGS = "logs"
S_MODEL_CHECKPOINTS = "model_checkpoints"
S_OBJECTS = "objects"

INPUT_DIM = 640
IMG_DIM = 256
BATCH_SIZE = 64
CHANNELS = 3
BUFFER_SIZE = 1024
FILTERS = (16, 256)
KERNEL_SIZE = 3
LOG_FREQUENCY = 12 * 60  # seconds
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
TRAIN_SPLIT = 1.0 - VALIDATION_SPLIT - TEST_SPLIT

tuning_hyperparameters = True
model_version = 0

tf.random.set_seed(1)
ReadableTime = namedtuple('ReadableTime', ['days', 'hours', 'minutes', 'seconds'])

data_dir = pathlib.Path('.')
CLASS_NAMES = np.array([item.name for item in data_dir.glob(os.path.join('photos', '*'))])

train_paths, valid_paths, test_paths = get_train_validation_test_folds()
total_paths = train_paths + valid_paths + test_paths

if tuning_hyperparameters:
    sub_dir = S_TUNING
    ds_train = tf.data.Dataset.from_tensor_slices(train_paths).map(lambda x: process_path(x, training=True))
    ds_valid = tf.data.Dataset.from_tensor_slices(valid_paths).map(lambda x: process_path(x)).batch(BATCH_SIZE)
    ds_test = tf.data.Dataset.from_tensor_slices(test_paths).map(lambda x: process_path(x)).batch(BATCH_SIZE)

else:
    sub_dir = S_COMPLETE
    ds_train = tf.data.Dataset.from_tensor_slices(total_paths).map(lambda x: process_path(x, training=True))
    ds_valid = None
    ds_test = None

pickle.dump(CLASS_NAMES, open(os.path.join(S_OBJECTS, sub_dir, 'CLASS_NAMES.pkl'), 'wb'))
ds_train = ds_train.shuffle(buffer_size=BUFFER_SIZE).repeat().batch(BATCH_SIZE)  # THIS IS THE PROPER ORDER
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)

if model_version == 0:
    model = create_network()
    dict_loss = {S_MODEL_VERSION: [], S_IMAGES_TRAINED: [], S_TIME: [],
                 S_LOSS_T: [], S_LOSS_V: [],
                 S_ACCURACY_T: [], S_ACCURACY_V: []}

    initial_batch_count = 0
    start_time = time.time()

else:
    model = tf.keras.models.load_model(get_model_path(model_version))

    df_loss = pd.read_csv(os.path.join(S_LOGS, sub_dir, "loss.csv"))

    mask = df_loss.loc[:, S_MODEL_VERSION] <= model_version
    dict_loss = df_loss.loc[mask, :].to_dict("list")
    initial_batch_count = int(dict_loss[S_IMAGES_TRAINED][-1] / BATCH_SIZE)
    start_time = time.time() - dict_loss[S_TIME][-1]
    model_version = dict_loss[S_MODEL_VERSION][-1]

with open(os.path.join(S_LOGS, sub_dir, 'model_summary.txt'), 'w') as f_model_summary:
    model.summary(print_fn=(lambda x: f_model_summary.write('{}\n'.format(x))))

last_status = time.time()
last_batch_count = initial_batch_count + 0
last_status_loss = {S_LOSS_T: [], S_ACCURACY_T: []}

for batch_count, (images, labels) in enumerate(ds_train, start=initial_batch_count):

    with tf.GradientTape() as tape:
        model_result = model(images, training=True)
        loss = get_loss_tensor(labels, model_result)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    last_status_loss[S_LOSS_T].append(loss.numpy())
    last_status_loss[S_ACCURACY_T].append(get_accuracy(model_result, labels))

    if time.time() - last_status > LOG_FREQUENCY:
        model_version += 1
        model.save(get_model_path(model_version))
        model.save(get_model_path())

        total_time = time.time() - start_time
        print_time = get_print_time(total_time)
        images_per_hour = BATCH_SIZE * (batch_count - last_batch_count) / (time.time() - last_status) * 3600

        dict_loss[S_MODEL_VERSION].append(model_version)
        dict_loss[S_IMAGES_TRAINED].append(BATCH_SIZE * batch_count)
        dict_loss[S_TIME].append(total_time)
        dict_loss[S_LOSS_T].append(np.mean(last_status_loss[S_LOSS_T]))
        dict_loss[S_ACCURACY_T].append(np.mean(last_status_loss[S_ACCURACY_T]))

        if tuning_hyperparameters:
            validation_loss, validation_accuracy = get_validation_loss_accuracy()
            dict_loss[S_LOSS_V].append(validation_loss)
            dict_loss[S_ACCURACY_V].append(validation_accuracy)
        else:
            dict_loss[S_LOSS_V].append(None)
            dict_loss[S_ACCURACY_V].append(None)

        pd.DataFrame.from_dict(dict_loss).to_csv(os.path.join(S_LOGS, sub_dir, "loss.csv"), index=False)
        plot_learning_curve()

        print(
            f"{S_MODEL_VERSION}: {dict_loss[S_MODEL_VERSION][-1]:4d} | "
            f"{S_IMAGES_TRAINED}: {dict_loss[S_IMAGES_TRAINED][-1]:8d} | "
            f"Time: {print_time.days}:{print_time.hours}:{print_time.minutes:02d}:{print_time.seconds:02d} | "
            f"{S_LOSS_T}: {dict_loss[S_LOSS_T][-1]:6.4f} | "
            f"{S_LOSS_V}: {dict_loss[S_LOSS_V][-1]:6.4f} | "
            f"{S_ACCURACY_T}: {dict_loss[S_ACCURACY_T][-1]:6.2%} | "
            f"{S_ACCURACY_V}: {dict_loss[S_ACCURACY_V][-1]:6.2%} | "
            f"Images per hour: {images_per_hour:5.0f}"
        )

        last_status = time.time()
        last_batch_count = batch_count + 0
        last_status_loss = {S_LOSS_T: [], S_ACCURACY_T: []}

"""


test_log = model.evaluate(ds_test, verbose=0)

with open(os.path.join('logs', f'test_log-{time_stamp}.txt'), 'w') as f_test_log:
    f_test_log.write('Test Loss:     {:.3f}\nTest Accuracy: {:.3f}'.format(test_log[0], test_log[1]))

"""
