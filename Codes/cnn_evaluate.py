import tensorflow as tf
import numpy as np
from scipy import misc
import os
import random
from preprocess import generate_new_box, read_data_2, read_data_3, labelFolder, imageFolder
from postprocess import bb_intersection_over_union, draw_boxes
from Model import Model
modelPath = 'output/model2.ckpt'
# Network Parameters
n_input = 128*128*3 # data input (img shape: 128*128*3)
n_labels = 4 # bounding box
n_classes = 2 # YES or NO


# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input], name = 'image_data')
y = tf.placeholder(tf.float32, [None, n_classes], name = 'image_label')
y_label = tf.placeholder(tf.float32, [None, n_labels, 128], name = 'image_box')
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# import model
model = Model()
weights = model.create_weights(n_classes = n_classes, n_labels = n_labels)
biases = model.create_biases(n_classes = n_classes, n_labels = n_labels)

pred, pred2 = model.conv_net(x, weights, biases, keep_prob, n_labels, 128)
cost1 = model.loss(pred, y)
cost2 = model.loss(pred2, y_label)

probability1 = tf.nn.softmax(pred)
probability2 = tf.nn.softmax(pred2)
decode1 = model.decode(pred, axis = 1)
decode2 = model.decode(pred2, axis = 2)
target2 = model.decode(y_label, axis = 2)


accuracy = model.accuracy(model.decode(pred,1), model.decode(y,1))
accuracy2 = model.accuracy(decode2, target2)

# Read evaluation data in
_,valid,_  = read_data_2(labelFolder, imageFolder)
valid_data, valid_label, valid_box, valid_names = read_data_3(valid)
new_valid_box = generate_new_box(valid_box)
valid_label = (np.arange(2) == valid_label[:,None]).astype(np.int32)



saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, modelPath)
    print("Model restored")
    i = 0
    total_accu = 0
    total_accu1 = 0
    total_error = 0
    batch_size = 50
    while i + batch_size < valid_data.shape[0]:
        offset = i
        names = valid_names[offset:offset+batch_size]
        batch_x, batch_y, batch_z = valid_data[offset:(offset+batch_size),:], valid_label[offset:(offset+batch_size),:], new_valid_box[offset:(offset+batch_size),:]
        test_accu1 = sess.run(accuracy, feed_dict = {x:batch_x, y:batch_y, y_label: batch_z, keep_prob:1.0})
        test_loss, test_accu = sess.run([cost2, accuracy2], feed_dict = {x: batch_x, y: batch_y, y_label: batch_z, keep_prob:1.0})
        total_accu = total_accu + test_accu
        total_accu1 = total_accu1 + test_accu1
        evaluation0, evaluation, target = sess.run([decode1, decode2, target2], feed_dict = {x:batch_x, y:batch_y, y_label: batch_z, keep_prob:1.0})
        error = np.mean(np.divide(np.absolute(np.subtract(target, evaluation)),[128.0]))
        draw_boxes(batch_y, evaluation0, evaluation, names, imageFolder)
        total_error = total_error + error
        i = i + batch_size
    accu1 = total_accu1/((1.0*i)/50)
    accu = total_accu/((1.0*i)/50)
    error = total_error/((1.0*i)/50)
    # Calculate accuracy for validation set
    print("Testing Accuracy for Classfication:" + "{:.6f}".format(accu1))
    print("Testing Accuracy for Regression:" + "{:.6f}".format(accu))
    print("Testing Accuracy 2 for Regression:" + "{:.6f}".format(1-error))
   
    