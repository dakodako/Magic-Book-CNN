import tensorflow as tf 
import numpy as np
class Model(object):
    def __init__(self, batch_size = 50, learning_rate = 0.001, n_input = 128*128*3, n_labels = 4, n_classes = 2):
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._n_labels = n_labels
        self._n_classes = n_classes
        self._n_input = n_input
    def _conv2d(self, x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)


    def _maxpool2d(self, x, k=1):
    # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    # Create model
    def conv_net(self, x, weights, biases, dropout, n_labels, imageWidth):
    # Reshape input picture
        
        x = tf.reshape(x, shape=[-1, 128, 128, 3])

        # Convolution Layer
        conv1 = self._conv2d(x, weights['wc1'], biases['bc1'],strides = 1)
        # Max Pooling (down-sampling)
        conv1 = self._maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = self._conv2d(conv1, weights['wc2'], biases['bc2'],strides = 1)
        # Max Pooling (down-sampling)
        conv2 = self._maxpool2d(conv2, k=2)
        # Convolution Layer
        conv3 = self._conv2d(conv2, weights['wc3'], biases['bc3'], strides = 1)
        # Max Pooling (down-sampling)
        conv3 = self._maxpool2d(conv3, k = 2)
        # Convolutional Layer
        conv4 = self._conv2d(conv3, weights['wc4'], biases['bc4'], strides = 1)
        # Max Pooling (down_sampling)
        conv4 = self._maxpool2d(conv4, k = 2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        conv4_flat = tf.reshape(conv4, [-1, weights['wd1'].get_shape().as_list()[0]])
        # First fully connected layer (for classifier)
        fc1 = tf.add(tf.matmul(conv4_flat, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)
        # First fully connected layer (for bounding box regression)
        fc2 = tf.add(tf.matmul(conv4_flat, weights['wd2']), biases['bd2'])
        fc2 = tf.nn.relu(fc2)
        # Second fully connected layer (for bounding box regression)
        fc3 = tf.add(tf.matmul(fc2, weights['wd3']), biases['bd3'])
        fc3 = tf.nn.relu(fc3)
        # Third fully connected layer (for bounding box regression)
        fc4 = tf.add(tf.matmul(fc3, weights['wd4']), biases['bd4'])
        fc4 = tf.nn.relu(fc4)
        # Output (for bounding box regression)
        out2 = tf.add(tf.matmul(fc4, weights['out2']), biases['out2'])
        out2 = tf.reshape(out2, [-1, n_labels, imageWidth])


        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

        return out, out2
    def create_weights(self, n_classes, n_labels):
        weights = {
            # 3x3 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([3, 3, 3, 32]), name = 'wc1'),
            # 3x3 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64]), name = 'wc2'),
            # 3x3 conv, 64 inputs, 128 outputs
            'wc3': tf.Variable(tf.random_normal([3, 3, 64, 128]), name = 'wc3'),
            'wc4': tf.Variable(tf.random_normal([3, 3, 128, 128]), name = 'wc4'),
            # fully connected, 16*16*128 inputs, 1024 outputs, wd1 for classifier
            'wd1': tf.Variable(tf.random_normal([8*8*128, 1024]), name = 'wd1'),
            # fully connected, 16*16*128 inputs, 1024 outputs, wd2 for regressor
            'wd2': tf.Variable(tf.random_normal([8*8*128, 1024]), name = 'wd2'),
            # fully connected, 1024 inputs, 1024 outputs
            'wd3': tf.Variable(tf.random_normal([1024,1024]), name = 'wd3'),
            # fully connected, 1024 inputs, 1024 outputs
            'wd4': tf.Variable(tf.random_normal([1024,1024]), name = 'wd4'),
            # 1024 inputs, 2 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, n_classes]), name = 'w_out1'),
            # 1024 inputs, n_labels * 128 outputs (bounding box regression)
            'out2':tf.Variable(tf.random_normal([1024, n_labels * 128]), name = 'w_out2')
        }
        return weights
    def create_biases(self, n_classes, n_labels):

        biases = {
            'bc1': tf.Variable(tf.random_normal([32]), name = 'bc1'),
            'bc2': tf.Variable(tf.random_normal([64]), name = 'bc2'),
            'bc3': tf.Variable(tf.random_normal([128]), name = 'bc3'),
            'bc4': tf.Variable(tf.random_normal([128]), name = 'bd4'),
            'bd1': tf.Variable(tf.random_normal([1024]), name = 'bd1'),
            'bd2': tf.Variable(tf.random_normal([1024]), name = 'bd2'),
            'bd3': tf.Variable(tf.random_normal([1024]), name = 'bd3'),
            'bd4': tf.Variable(tf.random_normal([1024]), name = 'bd4'),
            'out': tf.Variable(tf.random_normal([n_classes]), name = 'b_out1'),
            'out2': tf.Variable(tf.random_normal([n_labels * 128]), name = 'b_out2')
        }
        return biases
    def loss2(self, logits, labels):
        target = tf.argmax(labels, axis = 2)
        evaluation = tf.argmax(logits, axis=2)
        target = tf.cast(target, tf.float32)
        evaluation = tf.cast(evaluation, tf.float32)
        error = tf.reduce_mean(tf.divide(tf.abs(tf.subtract(target, evaluation)),tf.constant([128.0], dtype = tf.float32)))
        return error*100000
    def loss(self,  logits, labels):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
        return loss
    def train(self, loss, var_list):
        train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(loss, var_list=var_list)
        return train_op
    def decode(self, logits, axis):
        return tf.argmax(logits, axis = axis)
    def _compare(self, arg1, arg2):
        
        return tf.equal(arg1, arg2)
    def accuracy(self, arg1, arg2):
        correct_pred = self._compare(arg1, arg2)
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    def generate_probabilities(self, pred):
        return tf.nn.softmax(pred)

