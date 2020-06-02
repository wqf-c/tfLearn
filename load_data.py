from tensorflow.keras import layers, optimizers, datasets
import tensorflow as tf

def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, tf.float32)/255.0
    y = tf.cast(y, tf.int64)
    return x, y

(x, y), (x_val, y_val) = datasets.fashion_mnist.load_data()
y = tf.one_hot(y, depth=10)
y_val = tf.one_hot(y_val, depth=10)

ds = tf.data.Dataset.from_tensor_slices((x, y))
ds = ds.map(prepare_mnist_features_and_labels)
ds = ds.shuffle(60000).batch(100)
ds = ds.repeat(10)