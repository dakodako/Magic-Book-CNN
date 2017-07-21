#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
import os
from scipy import misc
import sys
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
import sys
import argparse
from preprocess import read_data_2, read_data_3, labelFolder, imageFolder
#import cv2 
tf.logging.set_verbosity(tf.logging.INFO)

from scipy import misc
import numpy as np
test_file = "/home/sensiflow/Documents/magicbook/test.txt"
valid_file = "/home/sensiflow/Documents/magicbook/valid.txt"
train_file = "/home/sensiflow/Documents/magicbook/train.txt"
def read_data(filename):
  test_file = open(filename,"r")
  lines = test_file.readlines()
  #print lines
  content = [x.strip() for x in lines]
  images = []
  labels = []
  for item in content:
    #print eval(item)[0]
    image_name = "/home/sensiflow/Documents/magicbook/data/" + eval(item)[0]
    #print image_name
    image = misc.imread(image_name)
    image = np.array(image,dtype = np.float32)
    image = image.flatten()
    images.append(image)
    
    labels.append(int(eval(item)[1]))
    
  images = np.array(images,dtype = np.float32)
  labels = np.array(labels, dtype  = np.int32)
  return images, labels
def preprocess_image2(image):
  image = cv2.cvtColor(image, cv2.COLOR_BRG2GRAY)
  sum_img = 0
  for i in range(image.shape[1]):
    for j in range(image.shape[0]):
      sum_img = sum_img + image[j][i];
  avg = (sum_img*1.0)/(image.shape[1]*image.shape[2])
  for i in range(image.shape[1]):
    for j in range(image.shape[0]):
      image[j][i] = image[j][i] - avg;
  return image
      #average = (int(image[j][i][0]) + int(image[j][i][1]) + int(image[j][i][2])*1.0)/3;
      #image[j][i][0] = image[j][i][0] - average;
      #image[j][i][1] = image[j][i][1] - average;
      #image[j][i][2] = image[j][i][2] - average;
  
def preprocess_image(image):
  for i in range(image.shape[1]):
    for j in range(image.shape[0]):
      average = (int(image[j][i][0]) + int(image[j][i][1]) + int(image[j][i][2])*1.0)/3;
      image[j][i][0] = image[j][i][0] - average;
      image[j][i][1] = image[j][i][1] - average;
      image[j][i][2] = image[j][i][2] - average;
  return image
def preprocess_images(image_flattened):
  sum_img = 0
  #for i in range(image_flattened):
    #sum_img = sum_img + image_flattened[i]
  image_flattened /= 255.0

  # for j in range(len(image_flattened)):
  #   image_flattened[j] = (image_flattened[j]*1.0)/255
  return image_flattened
def read_images(fileNames):
  img1 = cv2.imread(fileNames[0])
  img1 = preprocess_image(img1)
  img1 = np.array(img1,dtype = np.float32)
  img1 = img1.flatten()
  #img1 = preprocess_images(img1)
  img2 = cv2.imread(fileNames[1])
  img2 = preprocess_image(img2)
  img2 = np.array(img2,dtype = np.float32)
  img2 = img2.flatten()
  #img2 = preprocess_images(img2)
  img3 = np.vstack((img1,img2))
  for i in range(2,len(fileNames)):
    img = cv2.imread(fileNames[2])
    img = preprocess_image(img)
    img = np.array(img,dtype = np.float32)
    img = img.flatten()
    img3 = np.vstack((img3,img))

  return img3
train_dir = "/Users/didichi/Desktop/image2/train_1"
test_dir = "/Users/didichi/Desktop/image2/test"

def readfiles(dir):

  image_stack = []
  for filename in os.listdir(dir):
    if filename == ".DS_Store":
      continue
    #print filename
    img = misc.imread(os.path.join(dir,filename))
    img = np.array(img,dtype = np.float32)
    img = img.flatten()
    if img is not None:
      image_stack.append(img)
  image_stack = np.array(image_stack, dtype = np.float32)
  #print image_stack.shape
  return image_stack

def cnn_model_fn(features, labels, mode):
  #print(features.dtype)
  #print(labels.dtype)
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  
  input_layer = tf.reshape(features, [-1, 128, 128, 3])
  # the input layer is [batch_size, 28, 28, 1]
  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  
  conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=32,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu)
  
    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  
    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 32 * 32 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=2)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    #loss = tf.losses.softmax_cross_entropy(
        #onehot_labels=onehot_labels, logits=logits)

    #indices = tf.cast(labels,tf.int32)
    #print(onehot_labels.shape)
    #print(logits.shape)
    
    loss = tf.losses.hinge_loss(labels = onehot_labels, logits = logits)
    tf.summary.scalar('hinge_loss',loss)
  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="RMSProp")

  # Generate Predictions
  predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def main(unused_argv):
  # Load training and eval data
  # Each entry in the tensor is a pixel intensity between 0 and 1, 
  # for a particular pixel in a particular image.
  
  #depth = tf.constant(2)
  #one_hot_encoded = tf.one_hot(indices = original_indices, depth = depth)
  

  #mnist = learn.datasets.load_dataset("mnist")
  #train_data = read_images(fileNames) # mnist.train.images  # Returns np.array
  train_data,train_labels = read_data(train_file)
  #train_labels = original_indices # np.asarray(mnist.train.labels, dtype=np.int32)
  
  #eval_data = read_images(fileNames2) # mnist.test.images  # Returns np.array
  eval_data,eval_labels = read_data(valid_file)
  #eval_labels = original_indices2

  train, valid, _ = read_data_2(labelFolder, imageFolder)
  train_data, train_labels,_,_ = read_data_3(train)
  eval_data, eval_labels,_,_ = read_data_3(valid)
  '''
  print(train_data.dtype)
  print(train_labels.dtype)
  print(eval_data.dtype)
  print(eval_labels.dtype)
  
  print train_data.shape
  print train_labels.shape
  print eval_data.shape
  print eval_labels.shape
  #print train_data[0]
  '''
  
  sess = tf.InteractiveSession()
  validation_metrics = {
    "accuracy":
      tf.contrib.learn.MetricSpec(
        metric_fn = tf.contrib.metrics.streaming_accuracy,
        prediction_key = "classes"
      ),
    "precision":
      tf.contrib.learn.MetricSpec(
        metric_fn = tf.contrib.metrics.streaming_precision,
        prediction_key = "classes"
      ),
    "recall":
      tf.contrib.learn.MetricSpec(
        metric_fn = tf.contrib.metrics.streaming_recall,
        prediction_key = "classes"
      )
  }
  validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    eval_data,eval_labels,
    every_n_steps = 50,
    metrics = validation_metrics,
    early_stopping_metric = "loss",
    early_stopping_metric_minimize = True, 
    early_stopping_rounds = 200)
  # Create the Estimator
 # validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(eval_data,eval_labels,every_n_steps = 50)
  mnist_classifier = learn.Estimator(
      model_fn=cnn_model_fn, 
      model_dir="/tmp/book_covnet_model",
      config = tf.contrib.learn.RunConfig(save_checkpoints_steps = 1000))
  

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  #tensors_to_log = {"probabilities": "softmax_tensor"}
  #logging_hook = tf.train.LoggingTensorHook(
      #tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  mnist_classifier.fit(
      x=train_data,
      y=train_labels,
      batch_size=50,
      steps=40000,
      monitors=[validation_monitor])

  # Configure the accuracy metric for evaluation
  metrics = {
      "accuracy":
          learn.MetricSpec(
              metric_fn=tf.metrics.accuracy, prediction_key="classes"),
  }
  #tf.summary.scalar('accuracy',tf.metric.accuracy)

  # Evaluate the model and print results
  eval_results = mnist_classifier.evaluate(
      x=eval_data, y=eval_labels, metrics=metrics)
  #accuracy = eval_results["accuracy"]
  #tf.summary.scalar('accuracy',accuracy)
  print(eval_results)

  #merged = tf.summary.merge_all()
 # train_writer = tf.summary.FileWriter('/tmp/book' + '/train',sess.graph)
  #test_writer = tf.summary.FileWriter('/tmp/book' + '/test')
  #tf.global_variables_initializer().run()



if __name__ == "__main__":
  tf.app.run()
