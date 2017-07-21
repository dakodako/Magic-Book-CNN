import os
import numpy as np
from scipy import misc
import random
from PIL import Image
import numpy as np
currentPath = os.path.dirname(os.path.abspath(__file__)) 
parentPath = os.path.abspath(os.path.join(currentPath, os.pardir))
labelFolder = parentPath + "/Labels/003"
imageFolder = parentPath + "/Images/003"

'''
    read_data_2 reads data in the folder and generates three lists of tuples
    input:
        folder: the folder stores the labels of images the labels of text files with 
        the first line is either 0 or 1 
        0: the image does not have the label
        1: the image has the label
        second line is a list of four numbers shows the top left and botton right coordinates of the bounding box
        imageFolder: the folder stores the images
    returns: three lists of tuple
        [(0 or 1),(string: image_name),(a list of four float number:bounding_box)]
        train: the training dataset
        valid: the validation dataset
        test: the testing dataset
'''
def read_data_2(folder,imageFolder):
    positive = []
    negative = []
    #image_file_names = []
    for filename in os.listdir(folder):
        infilename = os.path.join(folder,filename)
        if not os.path.isfile(infilename): 
            continue
        base, extension = os.path.splitext(filename)

        imageFileName = base + '.jpg'
        with open(infilename) as f:
            content = f.readlines()
           
            if len(content) == 1 and eval(content[0]) == 0:
                tup = [0,imageFileName,[0,0,0,0]]
                negative.append(tup)
            
            if len(content) == 2:
                
                line = content[1].split()
                
                if line:
                    line = [int(i) for i in line]
                   
                    tup = [1, imageFileName, line]
                    positive.append(tup)
        
        start = 0
        train_positive = positive[start:int(len(positive)*0.7)]
        test_positive = positive[int(len(positive)*0.7):int(len(positive)*0.8)]
        valid_positive = positive[int(len(positive)*0.8):len(positive)]
        train_negative = negative[start:int(len(negative)*0.7)]
        test_negative = negative[int(len(negative)*0.7):int(len(negative)*0.8)]
        valid_negative = negative[int(len(negative)*0.8):len(negative)]

        train = train_positive + train_negative
        valid = valid_positive + valid_negative
        test = test_positive + test_negative
        random.seed(5)
        train = random.sample(train, len(train))
        valid = random.sample(valid, len(valid))
        test = random.sample(test, len(test))

    return train, valid, test
'''
    read_pos takes the same inputs as read_data_2 but returns only the positive data
'''
def read_pos(folder, imageFolder):
    label_data1 = []
    label_data = []
    image_data = []
    positive = []
   
    for filename in os.listdir(folder):
        infilename = os.path.join(folder,filename)
        if not os.path.isfile(infilename): 
            continue
        base, extension = os.path.splitext(filename)

        imageFileName = base + '.jpg'
        
        with open(infilename) as f:
            content = f.readlines()
            if len(content) == 2:
                
                line = content[1].split()
                label_data1.append(1)
                if line:
                    line = [int(i) for i in line]
                    for j in range(len(line)):
                        line[j] = (line[j]*1.0)/128
                    tup = [1, imageFileName, line]
                    positive.append(tup)
        
        start = 0
        train_positive = positive[start:int(len(positive)*0.7)]
        test_positive = positive[int(len(positive)*0.7):int(len(positive)*0.8)]
        valid_positive = positive[int(len(positive)*0.8):len(positive)]
    return train_positive, valid_positive, test_positive

# read_data_3 
# input: one of the outcomes from read_data_2, the format of the list of tuples is 
# [label (integer), image_name (string), bounding_box_labels (list)]
# returns four numpy arrays
# images: a numpy array with shape of [number of images, 128*128*3] (image raw data)
# labels: a numpy array with shape [number of images] consists of 0s and 1s 
# indicting whether the image has or does not have the label
# box: a numpy array with shape [number of images, 4] indicating 
# the bounding box in the images

def read_data_3(set):
    image_names = []
    images = []
    labels = []
    box = []
    for item in set:
        image_names.append(item[1])
        image_name = imageFolder + '/' + item[1]
        #print image_name
        image = misc.imread(image_name)
        image = np.array(image,dtype = np.float32)
        image = image.flatten()
        images.append(image)
        labels.append(item[0])
        box.append(item[2])
    images = np.array(images,dtype = np.float32)
    labels = np.array(labels, dtype  = np.int32)
    box = np.array(box, dtype = np.float32)
    return images, labels, box, image_names

# generate_new_box turns each bounding box in an image into a one-hot code
# input: a numpy array with shape [number of images, 4] which indicates the bounding box in the images
# output: a numpy array with shape [number of images, 4, 128] 
def generate_new_box(box):

    new_box = np.zeros((box.shape[0],box.shape[1],128),dtype = np.float32)
   
    for i in range(box.shape[0]):
        for j in range(box.shape[1]):
            a = np.zeros(128)
            if (box[i][j] >= 128):
                box[i][j] = 127
            np.put(a,box[i][j],1)
            new_box[i][j] = np.array(a, dtype = np.float32)
    return new_box

