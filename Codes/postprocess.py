import numpy as np
from PIL import Image, ImageDraw
savePath = '/home/sensiflow/Documents/magicbook/Final_codes2/Results/'
'''
	bb_intersection_over_union measures the accuracy of a bounding box prediction
	only works fine when the prediction can form a rectangle
	boxA: prediction
	boxB: ground truth
'''
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou
'''
	draw_box draws a bounding box on an image
	name: the file name of the image
	imageFolder: the path for the image
	box: a list of four numbers shows the top left and botton right coordinates
'''
def draw_box(name, imageFolder, box):
	image_name = imageFolder +'/'+ name
	base = Image.open(image_name).convert("RGB")
	draw = ImageDraw.Draw(base)
	draw.rectangle(((box[0],box[1]),(box[2],box[3])), outline = "black")
	output_name = savePath + "output_"+name
	base.save(output_name, "JPEG")
'''
	draw_boxes draws bounding boxes on a validation set
	valid_label: labels of the validation set that indicates whether an image has the label or not
	classify: the classification result
	boxes: bounding box result
	names: names of the images
	folder: the path to get these images
'''
def draw_boxes( valid_label, classify, boxes, names, folder):

	for i in range(valid_label.shape[0]):
		prediction0 = classify[i]
		#print(prediction0)
		#print(valid_label[i:i+1,:])
		if(prediction0 == 1):
			prediction = boxes[i]
			#print(names[i]) #prints out the name of the images that detected to have a label
			draw_box(names[i], folder, prediction)