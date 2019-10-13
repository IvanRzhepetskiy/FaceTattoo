# USAGE
# python detect_face_parts.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
import math


from PIL import ImageFont, ImageDraw, Image
from PIL.Image import QUAD
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2


print(face_utils.FACIAL_LANDMARKS_IDXS.items())
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-t", "--text", required=False,
	help="path to input image")
args = vars(ap.parse_args())
# TODO Поменять если будет лоб

def text_to_columns(text):
	text = text

	if len(text) >= 0:
		a = 0
	return text



text = ''

def split_text(text):
    text = text
    text_list = text.split(" ")
    left_text = ''
    right_text = ''
    symbol_count = 0
    rightFLAG = False
    for i in text_list:
        symbol_count += len(i)
        if symbol_count >= len(text) // 2:
            rightFLAG = True
        if not rightFLAG:
            left_text += i + ' '
        if rightFLAG:
            right_text += i + ' '
    return left_text, right_text


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


image = cv2.imread(args["image"])
#image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#newImg = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

#cv2.imshow('Resized Image', newImg)

#cv2.waitKey(0)
#TODO
left_text, right_text = split_text("I'm so happy! Summer is finally back to town!")

image_final = image

import textwrap
lines_left = textwrap.wrap(left_text, width=8)
lines_right = textwrap.wrap(right_text, width=8)
print(lines_right)

def create_tattoo(main_img, text_tattoo, font):
	#first create white blank
	print(main_img.shape)
	x,y,c = main_img.shape
	blank_image2 = 255 * np.ones(shape=[x, y, c], dtype=np.uint8)
	cv2.imshow("White Blank", blank_image2)
	cv2.waitKey(0)
	text_to_show = text_tattoo

	# Convert the image to RGB (OpenCV uses BGR)
	cv2_im_rgb = cv2.cvtColor(blank_image2, cv2.COLOR_BGR2RGB)
	pil_im = Image.fromarray(cv2_im_rgb)
	draw = ImageDraw.Draw(pil_im)
	font = ImageFont.truetype("Tattoo v2.ttf", 30)
	# Draw the text
	draw.text((10,700), text_to_show, font=font, fill='rgb(0, 0, 0)')

	# Get back the image to OpenCV
	cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
	cv2.imshow('Fonts', cv2_im_processed)
	cv2.imwrite('font_size.jpg', cv2_im_processed)
	cv2.waitKey(0)

	cv2.destroyAllWindows()
	pass

create_tattoo(image, left_text, "fd")

#create_tattoo(image, 'addad', "sdfdf")
# detect faces in the grayscale image
print("Rects are here")
rects = detector(gray, 1)

print(rects)
x, y = 0, 0
for i, d in enumerate(rects):
	print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
		i, d.left(), d.top(), d.right(), d.bottom()))
	x, y = d.left(), d.top()
cv2.circle(image, (x, y), 2, (0, 255, 0), -1)





# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the landmark (x, y)-coordinates to a NumPy array
	print("Face was found")
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	for (i, rect) in enumerate(rects):
		# Make the prediction and transfom it to numpy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		#TODO Delete
		# Draw on our image, all the finded cordinate points (x,y)
		#for (x, y) in shape:
			#cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

	# Show the image
	# For nose On the left we take 80% between point 2 and 32
	print()
	point_left_1 = list()
	num = shape[31][0] - math.sqrt(pow(shape[1][0]-shape[31][0], 2) + pow(shape[1][1]-shape[31][1],2) ) * 0.1
	point_left_1.append(num)
	point_left_1.append(shape[31][1])

	pts = np.array([shape[4], shape[48], point_left_1, shape[39], shape[36], shape[1]], np.int32)
	# TODO Poly lines
	#cv2.polylines(image, [pts], True, (0, 255, 255))

	point_right_1 = list()
	num = shape[35][0] + math.sqrt(pow(shape[15][0] - shape[35][0], 2) + pow(shape[15][1] - shape[35][1], 2)) * 0.1
	point_right_1.append(num)
	point_right_1.append(shape[35][1])
	# TODO Poly lines
	pts = np.array([shape[12], shape[54], point_right_1, shape[42] , shape[45], shape[15]], np.int32)
	#cv2.polylines(image, [pts], True, (0, 255, 255))



	cv2.imshow("Output", image)
	cv2.imwrite('output.jpg', image)

	left_dist = math.sqrt(pow(shape[1][0] - shape[36][0], 2) + pow(shape[1][1] - shape[36][1], 2))
	cv2_im_rgb = cv2.cvtColor(image_final, cv2.COLOR_BGR2RGB)
	pil_im = Image.fromarray(cv2_im_rgb)
	draw = ImageDraw.Draw(pil_im)
	font = ImageFont.truetype("Tattoo v3.ttf", int(left_dist//4.9))
    #TODO
	font2 = ImageFont.truetype("Tattoo v2.ttf", int(left_dist//4.9))
	y_text = int(left_dist//4.9) * 2
	i = 1
	for line in lines_left:
		i = i + 1
		width, height = font.getsize(line)
		draw.text((shape[i][0]+3 / 2 - i*3 + 11, shape[1][1]-int(left_dist//3) + y_text - int(left_dist//4.9)) , line, font=font, fill=(58,49,49,0))
		y_text += height


	y_text = int(left_dist // 4.9) * 2
	i = 1
	for line in lines_right:
		i = i + 1
		width, height = font.getsize(line)
		draw.text(
			(shape[35][0]+ 10, shape[1][1] - int(left_dist // 3) + y_text - int(left_dist // 4.9)/2),
			line, font=font2, fill=(10, 20, 26, 0))
		y_text += height





    #left_text

	#draw.text((shape[2][0]+2, shape[1][1]-int(left_dist//3)), left_text, font=font, fill=(58,49,49,0))

	cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

	cv2.imshow('Final', cv2_im_processed)
	cv2.waitKey(0)
	cv2.imwrite('final_1.jpg',cv2_im_processed )


	left_dist = math.sqrt(pow(shape[1][0]-shape[36][0], 2) + pow(shape[1][1]-shape[36][1],2) )
	right_dist = math.sqrt(pow(shape[45][0]-shape[15][0], 2) + pow(shape[45][1]-shape[15][1],2) )
	print(left_dist)
	print(right_dist)
	polynom_y_40_49 = math.sqrt(pow(shape[39][0]-shape[48][0], 2) + pow(shape[39][1]-shape[48][1],2) )
	polynom_x_4_49 = math.sqrt(pow(shape[5][0]-shape[48][0], 2) + pow(shape[5][1]-shape[48][1],2) )

	left_polynom_area = (polynom_y_40_49 * polynom_y_40_49) * 1.3

	polynom_y_42_54 = math.sqrt(pow(shape[42][0] - shape[54][0], 2) + pow(shape[42][1] - shape[54][1], 2))
	polynom_x_55_14 = math.sqrt(pow(shape[54][0] - shape[13][0], 2) + pow(shape[54][1] - shape[13][1], 2))

	right_polynom_area = (polynom_y_42_54 * polynom_x_55_14) * 1.3

	cv2.waitKey(0)

