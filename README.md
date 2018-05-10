# Magic Book - Using Convolutional Neural Network for Book Detection

In order to project the content onto a blank page in a book, a program which can locate the book is required. 
This project uses convolutional neural network to localize the labels on four corners of the book so that a precise location of the book can be found.




## Getting Started

### Prerequisites

* Python 3
* Tensorflow r1.2
* Scipy
* Numpy
* PIL
* (CUDA 8.0 if GPU acceleration is required)


### Running the programs

* preprocess.py has functions that read data from folders
* postprocess.py has functions that are ran after evaluation, such as drawing bounding boxes
* Model.py stores the convolutional neural network structure
* Run cnn_train.py to train the model and stores the model in '/output' directory under the directory that cnn_train.py is stored
* Run cnn_evaluate.py to retore the model from '/output' and evaluate the model with the validation data set and it will prints out the accuracy for the classifier and the regression

* The image data are stored in the folder "Image/003" under the project folder
* The labels of the image data are store in the folder "Labels/003" under the project folder
* The label of each image is stored in a single text file: the first line indicates whether the image has an icon or not, the second line indicates the coordinate of the icon ([0,0,0,0] for none)
* The labels are created by the application BBox-Label-Tool from github (https://github.com/puzzledqs/BBox-Label-Tool)

## Current Results

* The current convolutional neural network consists of four convolutional layers, two fully connected layers for classifier and four fully connected layers for regression
* The accuracy of the classification on the validation set is 0.9458 (after 60000 training iterations on 5446 128x128x3 images)
* The accuracy of the regression on the validation set is 0.588 (after 250000 training iterations on 5446 128x128x3 images)




