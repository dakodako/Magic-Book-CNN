'''
A Convolutional Network implementation using TensorFlow library.
'''

import tensorflow as tf
import numpy as np
from scipy import misc
import os
import random
from preprocess import generate_new_box, read_data_2, read_data_3, labelFolder, imageFolder
from postprocess import bb_intersection_over_union, draw_boxes
from Model import Model
modelPath = 'output/model2.ckpt'
# Parameters
learning_rate = 0.001
training_iters = 60000
batch_size = 50
display_step = 10

# Data Parameters
imageWidth = 128
imageHeight = 128
channels = 3
# Network Parameters
n_input = imageHeight*imageWidth*channels 
n_labels = 4
n_classes = 2 
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input], name = 'image_data')
y = tf.placeholder(tf.float32, [None, n_classes], name = 'image_label')
y_label = tf.placeholder(tf.float32, [None, n_labels, imageWidth], name = 'image_box')
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

model = Model()
weights = model.create_weights(n_classes = n_classes, n_labels = n_labels)
biases = model.create_biases(n_classes = n_classes, n_labels = n_labels)

saver = tf.train.Saver()
# Construct model
pred, pred2 = model.conv_net(x, weights, biases, keep_prob, n_labels, imageWidth)

# Probability Matrix of the predictions
probability1 = model.generate_probabilities(pred)
probability2 = model.generate_probabilities(pred2)

# Turn the one-hot-encoded predictions and inputs to bounding box labels
decode1 = model.decode(y, axis = 1)
decode2 = model.decode(pred2, axis = 2)
target2 = model.decode(y_label, axis = 2)

# Calculate the loss for the predictions
cost1 = model.loss(pred, y)
cost2 = model.loss(pred2, y_label)

# Set the layers to be trained in each stage
var_list1 = [weights['wc1'],weights['wc2'],weights['wd1'],weights['out'],biases['bc1'],biases['bc2'],biases['bd1'],biases['out']]
var_list2 = [weights['wd2'], weights['out2'], biases['bd2'],biases['out2']]

# optimizer for classification
optimizer = model.train(cost1, var_list1)

# optimizer for regression
optimizer2 = model.train(cost2, var_list2)

# Evaluate model
accuracy = model.accuracy(model.decode(pred,1), model.decode(y,1))
accuracy2 = model.accuracy(decode2, target2)

# Loading Data
train, valid, _ = read_data_2(labelFolder, imageFolder)
train_data, train_label, train_box,_ = read_data_3(train)
valid_data, valid_label, valid_box,valid_names = read_data_3(valid)

# Turning labels to one-hot code
train_label = (np.arange(2) == train_label[:,None]).astype(np.int32)
valid_label = (np.arange(2) == valid_label[:,None]).astype(np.int32)

# Turning bounding box labels to one-hot code
new_train_box = generate_new_box(train_box)
new_valid_box = generate_new_box(valid_box)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    
    while step * batch_size < training_iters:
        offset = (step * batch_size) % (len(train_data) - batch_size)
        batch_y,batch_x,batch_z  = train_label[offset:(offset + batch_size),:],train_data[offset:(offset + batch_size), :], new_train_box[offset:(offset + batch_size), :]
       
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, y_label:batch_z,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost1, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              y_label: batch_z,
                                                              keep_prob: 1.})
            regression_loss = sess.run(cost2, feed_dict={x:batch_x, y:batch_y, y_label:batch_z, keep_prob:1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for validation images
        
    i = 0
    total_accu = 0

   
    while i + batch_size < valid_data.shape[0]:
        names = valid_names[i:i+batch_size]
        batch_x, batch_y, batch_z = valid_data[i:(i+batch_size),:], valid_label[i:(i+batch_size),:], new_valid_box[i:(i+batch_size),:]
        test_loss, test_accu = sess.run([cost1, accuracy], feed_dict = {x: batch_x, y: batch_y, y_label: batch_z, keep_prob:1.0})
    
        total_accu = total_accu + test_accu
        i = i + batch_size
    accu = total_accu/((1.0*i)/50)
    # Calculate accuracy for validation set
    print("Testing Accuracy:" + "{:.6f}".format(accu))
    
    step = 1
    # train the regression head for finding the bounding box
    training_iters = 250000
    while step * batch_size < training_iters:
        offset = (step * batch_size) % (len(train_data) - batch_size)
        batch_y,batch_x,batch_z  = train_label[offset:(offset + batch_size),:],train_data[offset:(offset + batch_size), :], new_train_box[offset:(offset + batch_size), :]
        sess.run(optimizer2, feed_dict={x: batch_x, y:batch_y, y_label:batch_z,keep_prob:dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            target = sess.run(target2, feed_dict = {x:batch_x, y:batch_y, y_label:batch_z, keep_prob:dropout})
            pred = sess.run(decode2, feed_dict = {x: batch_x, y: batch_y, y_label:batch_z, keep_prob:dropout})
            loss, acc = sess.run([cost2, accuracy2], feed_dict={x: batch_x, y: batch_y, y_label: batch_z, keep_prob: 1.})
            error = np.mean(np.divide(np.absolute(np.subtract(target, pred)),[128.0]))
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc) + ", Training Error = " + \
                  "{:.6f}".format(error) + ",Training Accuracy 2= "+\
                  "{:.6f}".format(1-error))
            
        step += 1
    saver.save(sess, modelPath)
    print("Optimization Finished!")
    # Caluate accuracy for validation set
    i = 0
    total_accu = 0
    total_error = 0
    while i + batch_size < valid_data.shape[0]:
        offset = i
        names = valid_names[offset:offset+batch_size]
        batch_x, batch_y, batch_z = valid_data[offset:(offset+batch_size),:], valid_label[offset:(offset+batch_size),:], new_valid_box[offset:(offset+batch_size),:]
        test_loss, test_accu = sess.run([cost2, accuracy2], feed_dict = {x: batch_x, y: batch_y, y_label: batch_z, keep_prob:1.0})
        total_accu = total_accu + test_accu
        evaluation0, evaluation, target = sess.run([decode1, decode2, target2], feed_dict = {x:batch_x, y:batch_y, y_label: batch_z, keep_prob:1.0})
        error = np.mean(np.divide(np.absolute(np.subtract(target, evaluation)),[128.0]))
        draw_boxes(batch_y, evaluation0, evaluation, names, imageFolder)
        total_error = total_error + error
        i = i + batch_size
    accu = total_accu/((1.0*i)/50)
    error = total_error/((1.0*i)/50)
    # Calculate accuracy for validation set
    print("Testing Accuracy:" + "{:.6f}".format(accu))
    print("Testing Accuracy 2:" + "{:.6f}".format(1-error))
    print("Testing Error:"+"{:.6f}".format(error))
   
   
            
