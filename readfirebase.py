import os
import argparse
import cv2
import numpy as np
import sys
#from mail import sendEmail
from stepmotor import StepMotor
import time
import threading
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
import importlib.util

cred = credentials.Certificate('drone-detection-js-firebase-adminsdk-4xh9r-3ba93b9ccd.json')
# Initialize the app with a service account, granting admin privileges
cred = credentials.Certificate('firestore-1add2-firebase-adminsdk-7vjg4-6c20413010.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

while True:
    t= StepMotor()
    doc_ref_drone = db.collection(u'robot1').document(u'control')
    dic = doc_ref_drone.get().to_dict()   
    if dic['state'] == 1:
        t.left()
    elif dic['state'] == 2:
        t.right()

        
        

