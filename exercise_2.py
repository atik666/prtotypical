import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, Dense, Flatten, MaxPooling2D, Activation, BatchNormalization, TimeDistributed
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD,Adam

def conv_net():
    convnet = Sequential()
    for i in range(4):
        convnet.add(Conv2D(64,(3,3),padding='same',input_shape=input_shape))
        convnet.add(BatchNormalization())
        convnet.add(Activation('relu'))
        convnet.add(MaxPooling2D())
    convnet.add(Flatten())
    return convnet

input_shape = (None, 84, 84, 3)

conv = conv_net()
conv_5d = TimeDistributed(conv)

# Input samples
sample = Input(input_shape)
sample_feature = conv_5d(sample)

# Input Queries
query = Input(input_shape)
query_feature = conv_5d(query)

def reduce_tensor(x):
    return tf.reduce_mean(x, axis=1)

def reshape_query(x):
    return tf.reshape(x, [-1, tf.shape(x)[-1]])

class_center = Lambda(reduce_tensor)(sample_feature)
query_feature = Lambda(reshape_query)(query_feature)

def prior_dist(x):
    sample_center, query_feature = x
    q2 = tf.reduce_sum(query_feature ** 2, axis=1, keepdims=True)
    s2 = tf.reduce_sum(sample_feature ** 2, axis=1, keepdims=True)
    qdots = tf.matmul(query_feature, tf.transpose(sample_center))
    return tf.nn.softmax(-(tf.sqrt(q2 + tf.transpose(s2) - 2 * qdots)))

pred = Lambda(prior_dist)([class_center, query_feature])
combine = Model([sample, query], pred)
optimizer = Adam()
combine.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])