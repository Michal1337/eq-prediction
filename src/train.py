import numpy as np
import tensorflow as tf
import pandas as pd

SEED = 1337
tf.random.set_seed(SEED)

from .params import BATCH_SIZE, EPOCHS, D_MODEL, NUM_HEADS, NUM_LAYERS
from .model import Transformer


def transform_element(features, context, labels):
    return (features, context), labels


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        self.filepath = filepath
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.filepath + f"model_v3_{epoch+1}.keras")


@tf.keras.saving.register_keras_serializable()
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}


if __name__ == "__main__":
    ds_train = tf.data.Dataset.load("../data/ds_train/")
    ds_val = tf.data.Dataset.load("../data/ds_val/")
    ds_test = tf.data.Dataset.load("../data/ds_test/")

    ds_train = ds_train.map(transform_element)
    ds_val = ds_val.map(transform_element)
    ds_test = ds_test.map(transform_element)

    neg, pos = 3094591, 199845
    total = neg + pos

    ds_train = (
        ds_train.batch(BATCH_SIZE)
        .shuffle(2000, seed=SEED, reshuffle_each_iteration=False)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    ds_val = ds_val.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    model = Transformer(num_layers=NUM_LAYERS, d_model=D_MODEL, num_heads=NUM_HEADS)

    learning_rate = CustomSchedule(D_MODEL)
    model.compile(
        tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy", tf.keras.metrics.Recall(), tf.keras.metrics.Precision()],
    )

    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    history = model.fit(
        ds_train,
        epochs=EPOCHS,
        validation_data=ds_val,
        callbacks=[CustomCallback("../models/v3/")],
    )

    print(model.evaluate(ds_test))

    model.save("../models/model_v3.keras")

    # save history
    history = pd.DataFrame.from_dict(history.history)
    history.to_csv("../models/history_model_v3.csv", index=False)
