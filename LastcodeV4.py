######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
from videostream import VideoStream

# Fetch the service account key JSON file contents
cred = credentials.Certificate('drone-detection-js-firebase-adminsdk-4xh9r-3ba93b9ccd.json')
# Initialize the app with a service account, granting admin privileges
cred = credentials.Certificate('firestore-1add2-firebase-adminsdk-7vjg4-6c20413010.json')
firebase_admin.initialize_app(cred)
db = firestore.client()


# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='800x600')
parser.add_argument('--savedir', help='saveimage folder',
                    default='DroneImg')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
imgfolder = args.savedir
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

dnum=0
capture_drone=0
time_appear_drone = 0
zero_count=0
boxthickness = 3
linethickness = 2
# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)
rectangule_color = (10, 255, 0)
#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()
    d=0
    distance = 0
    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    capframe = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
    
    num = []
    now = time.gmtime(time.time())
    y = str(now.tm_year)
    mo = str(now.tm_mon)
    d = str(now.tm_mday)
    h = str(now.tm_hour-3)
    mi = str(now.tm_min)
    
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), rectangule_color, boxthickness)  #xmax = x+w ymax = y+h 
            time_appear_drone = time_appear_drone + 1
            
            print('scores',scores)
            if scores[i] > 0.4:
                num.append(scores[i])
                #print('num',num)
                
            if len(num) != 0:
                most = num.index(max(num))
                lx = int(max(1,(boxes[most][1] * imW)))
                ly = int(max(1,(boxes[most][0] * imH)))
                lw = int(min(imW,(boxes[most][3] * imW)))
                lh = int(min(imH,(boxes[most][2] * imH)))
                x_medium = int((lx+lw)/2)
                y_medium = int((ly+lh)/2)
                cv2.line(frame, (x_medium, 0), (x_medium, 720), (10, 255, 0), linethickness)
                cv2.line(frame, (0, y_medium), (1280, y_medium), (10, 255, 0), linethickness)
                d = lw-lx
                distance = float(-(3/35)*d+90)

            # Draw label
            #object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            #print('Number of Drone', len(num))
            label = '%s: %d%%' % ('Drone', int(scores[i]*100)) # Example: 'person: 72%' #object_name
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            
    
    if time_appear_drone > 10:
        rectangule_color = (0,0,255)
        boxthickness = 7
        capture_drone += 1
        zero_count=0
        if dnum != len(num):
            dnum=len(num)
            doc_ref_drone = db.collection(u'robot1').document(u'sky')
            doc_ref_drone.set({
            u'Num_of_drone': len(num),
            u'date' : date
            })
            #print('sending data',dnum)
        
    else :
        rectangule_color = (10, 255, 0)
        thickness = 3
    
    
    if len(num) == 0:
        time_appear_drone = 0
        zero_count += 1
        if zero_count == 10:
            doc_ref_drone = db.collection(u'robot1').document(u'sky')
            doc_ref_drone.set({
            u'Num_of_drone': 0,
            u'date' : date
            })
            #print('sendfirebasevalue 0')  #once
           
        if zero_count == 1000:
            zero_count=0
            
    if time_appear_drone <= 0:
        time_appear_drone=0
    
    date =str(y)+str(mo)+str(d)+str(h)+str(mi)
    reimage = cv2.resize(capframe,(800,600))
    
    if capture_drone == 20:
        cv2.imwrite(imgfolder +'/' + 'D'+ date + '.jpg', reimage)
        f = open(imgfolder+'/'+'D'+ date +".xml", 'w')
        time.sleep(0.1)
        f.write('<annotation>\n\t<folder>'+imgfolder+'</folder>\n')
        f.write('\t<filename>'+date+'</filename>\n<path>imgfolder.jpg</path>\n')
        f.write('\t<source>\n\t/t<database>Unknown</database>\n\t</source>\n')
        f.write('\t<size>\n\t\t<width>800</width>\n\t\t<height>600</height>\n')
        f.write('\t\t<depth>3</depth>\n\t</size>\n\t<segmented>0</segmented>\n')
        f.write('\t<object>\n\t\t<name>Drone</name>\n\t\t<pose>Unspecified</pose>\n')
        f.write('\t/t<truncated>0</truncated>\n\t\t<difficult>0</difficult>\n')
        f.write('\t\t<bndbox>\n\t\t\t<xmin>'+str(lx)+'</xmin>\n')
        f.write('\t\t\t<ymin>'+str(ly)+'</ymin>\n')
        f.write('\t\t\t<xmax>'+str(lw)+'</xmax>\n')
        f.write('\t\t\t<ymax>'+str(lh)+'</ymax>\n')
        f.write('\t\t</bndbox>\n\t</object>\n</annotation>')
        time.sleep(0.1)
        f.close
        #print("succese")
        #print('saved_image++++++++++++++++++++++++++++++++++++')
        capture_drone=0
    
    if capture_drone>18 and capture_drone<=20:
        cv2.putText(frame,'Drone IMG Saved',(600,650),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2,cv2.LINE_AA)
    
    
    
    if distance < 0:
        distance=0
    
    text = "Number of drone is : {} ".format(len(num))
    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    cv2.putText(frame, text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    cv2.putText(frame,'Distance: {0:.1f}cm'.format(distance),(30,150),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    cv2.putText(frame,'DAY: ' + str(y) +'Y'+str(mo)+'M'+str(d)+'D'+str(h)+'H'+str(mi)+'m',(30,650),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
     
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()





