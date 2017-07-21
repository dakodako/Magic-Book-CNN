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
tf.logging.set_verbosity(tf.logging.INFO)


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
  
  # Read in data

  train, valid, _ = read_data_2(labelFolder, imageFolder)
  train_data, train_labels,_,_ = read_data_3(train)
  eval_data, eval_labels,_,_ = read_data_3(valid)
  
  
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

  



if __name__ == "__main__":
  tf.app.run()
