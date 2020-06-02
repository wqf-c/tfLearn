import tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

import  os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def process(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    return x, y

(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
batches = 128
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(process).shuffle(10000).batch(batches)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(process).batch(batches)

model = keras.Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10)
])
model.build(input_shape=[None, 28*28])
model.summary()
optimizer = optimizers.Adam(lr=1e-3)
loss_meter = metrics.Mean()
accu_meter = metrics.Accuracy()

for epoch in range(30):
    for step, (x, y) in enumerate(train_db):
        x = tf.reshape(x, [-1, 28*28])
        with tf.GradientTape() as tape:
            logits = model(x)
            y_onehot = tf.one_hot(y, depth=10)
            loss_mse = tf.reduce_mean(tf.losses.mse(y_onehot, logits))
            loss_ce = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
            loss_ce = tf.reduce_mean(loss_ce)
            loss_meter.update_state(loss_ce)

        grads = tape.gradient(loss_ce, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(loss_meter.result().numpy())
            loss_meter.reset_states()
            #print(epoch, step, 'loss:', float(loss_ce), float(loss_mse))

    totalNum, totalCorrect = 0, 0
    for x_t, y_t in test_db:
        x_t = tf.reshape(x_t, [-1, 28*28])
        logits = model(x_t)
        pred = tf.nn.softmax(logits, 1)
        #用途：返回最大的那个数值所在的下标
        pred = tf.argmax(pred, 1)
        pred = tf.cast(pred, dtype=tf.int32)
        # pred:[b]
        # y: [b]
        # correct: [b], True: equal, False: not equal
        accu_meter.update_state(y_t, pred)
        correct = tf.equal(pred, y_t)
        correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))
        totalCorrect += correct
        totalNum += x_t.shape[0]

    acc = totalCorrect / totalNum
    print(accu_meter.result().numpy())
    accu_meter.reset_states()
    #print(epoch, 'test acc:', acc)

