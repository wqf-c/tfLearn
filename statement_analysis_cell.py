import  os
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
from    tensorflow.keras import layers


tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

bachsz = 128

total_words = 128
max_review_len = 80
embedding_len = 100

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(bachsz, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(bachsz, drop_remainder=True)

print('x_train shape:', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train[5])

class MyRnn(keras.Model):
    def __init__(self, units):
        super(MyRnn, self).__init__()
        self.state0 = [tf.zeros([bachsz, units])]
        self.state1 = [tf.zeros([bachsz, units])]
        # transform text to embedding representation
        # [b, 80] => [b, 80, 100]
        #句子转向量。。。embedding也是训练得到的？
        self.embedding = layers.Embedding(total_words, embedding_len,
                                          input_length=max_review_len)
        self.rnn_cell0 = layers.SimpleRNNCell(units, dropout=0.5)
        self.rnn_cell1 = layers.SimpleRNNCell(units, dropout=0.5)

        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None):
        x = inputs
        x = self.embedding(x)
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1):
            out0, state0 = self.rnn_cell0(word, state0, training)
            out1, state1 = self.rnn_cell1(out0, state1, training)

        x = self.outlayer(out1)
        prob = tf.sigmoid(x)

        return prob

def main():
    units = 64
    epochs = 4

    model = MyRnn(units)
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    model.fit(db_train, epochs=epochs, validation_freq=db_test)
    model.evaluate(db_test)

if __name__ == '__main__':
    main()
