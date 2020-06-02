import tensorflow as tf
import numpy as np
from    tensorflow.keras import layers, optimizers, datasets

tf.constant(1)
tf.constant(2.2, dtype=tf.double)
tf.constant([1.1, 2.2], dtype=tf.double)
with tf.device('cpu'):
    a = tf.constant([1])

a.device
#cpu数据转换到gpu上
aa = a.gpu()

a.numpy()

a.shape()
#多少维
a.ndim

tf.is_tensor(a)

a.dtype == tf.float32

d = np.range(4)
dd = tf.convert_to_tensor(d, dtype=tf.int32)
tf.cast(dd, dtype=tf.float32)

c = tf.range(5)
cc = tf.Variable(c)
ccc = tf.Variable(c, name='input_data')
cc.numpy()

#2行2列
tf.zeros([2, 2])
#1行两列
tf.ones([2])

tf.fill([2, 2], 8)
#正态分布
tf.random.normal([2, 2], mean=1, stddev=1)
#截取的正态分布
tf.random.truncated_normal([2, 2], mean=0, stddev=1)
#均匀
tf.random.uniform([2, 2], minval=0, maxval=10)

#打散
idx= tf.random(10)
idx = tf.random.shuffle(idx)
a = tf.random.normal([10, 784])
b = tf.random.uniform([10], maxval=10, dtype=tf.int32)
a = tf.gather(a, idx)
b = tf.gather(b, idx)

out = tf.random.uniform([4, 10])
y = tf.random(4)
y = tf.one_hot(y, depth=10)
loss = tf.keras.losses.mse(y, out)

x = tf.random.normal([4, 784])
net = layers.Dense(10)
net.build(4, 784)
net(x).shape

net.kernel.shape

net.bias.shape

a = tf.ones([4, 5, 5, 3])
a[0][0]
a[1, 2]
a = tf.random(10)
a[1:3:1]
#向量
a[-1:]
#常量
a[-1]

a[1, ..., 2]

a = tf.random.normal([4, 35, 8])
#打乱顺序或者提取指定的维度
a = tf.gather(a, axis=0, indices=[2,  3, 0])

#多维度
tf.gather_nd(a, [0])  #35, 8
tf.gather_nd(a, [0, 1]) #[8]
tf.gather_nd(a, [0, 1, 2]) #标量
tf.gather_nd(a, [[0, 1, 2]])#[1]
tf.gather_nd(a, [[0, 0], [1, 1]]) #[2, 8]
a = tf.constant([4, 28, 28, 3])
tf.boolean_mask(a, mask=[True, True, False], axis=3)
a = tf.ones([2, 3, 4])
tf.boolean_mask(a, mask=[[True, True, False], [True, False, True]])#[3, 4]

a = tf.random.normal([4, 28, 28, 3])
tf.reshape(a, [4, 784, 3])
tf.reshape(a, [4, -1, 3])

a = tf.random.normal([4, 28, 28, 3])
tf.transpose(a)  #3, 28, 28, 4
tf.transpose(a, perm=[0, 1, 3, 2])

tf.expand_dims(a, axis=0)

#去掉为1的维度
tf.squeeze(tf.zeros([1, 2, 1, 1, 3]))
a = tf.zeros([1, 2, 1, 1, 1])
tf.squeeze(a, axis=-1)