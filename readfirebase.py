import os
import argparse
import cv2
import numpy as np
import sys
#from mail import sendEmail
from motor import StepMotor
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
panmotor=0
tiltmotor=0
currentstate=0
controlcount=0
while True:
    t= StepMotor()
    doc_ref_drone = db.collection(u'robot1').document(u'control')
    dic = doc_ref_drone.get().to_dict()
    try :
        pandatabase = int(dic['left_right'])
    except KeyError:
        print('no doc')
        pass
    tiltdatabase = int(dic['up_down'])
    try :
        controldatabase = int(dic['left_rightcontrol'])
        print(controldatabase)
    except KeyError:
        print('no doc')
        pass

    try :
        if pandatabase == 1:
            t.right()
        elif pandatabase == -1:
            t.left()
        elif pandatabase == 0:
            t.stop()
            controlcount = 0
            print('stop')
    except NameError :
        print('no left_right')
        pass
        
    try :
        if controldatabase == 1:
            controlcount =+1
        elif controldatabase == -1:
            controlcount =-1
            print(controlcount)

        else :
            controlcount = 0
    except NameError :
        print('no left_rightcontrol')
        pass
        
    
    if panmotor != controlcount :
        currentstate = controlcount - panmotor
        controlcount = panmotor
        print('currentstate : ',currentstate)

    
    '''
    if pandatabase != panmotor:
        currentstate = pandatabase - panmotor
        panmotor = pandatabase
        print('pan : ',panmotor)
        print('currentstate : ',currentstate)
     '''
    
    if currentstate == 1 :
        t.right()
        currentstate = 0
    elif currentstate == -1 :
        t.left()
        currentstate = 0
    else :
        if currentstate > 1:
            for i in range(currentstate):
                t.right()
                currentstate = 0
        elif currentstate < -1 :
            for i in range(abs(currentstate)):
                t.left()
                currentstate = 0
    
    
    '''   
    if tiltdatabase != tiltmotor:
        tiltmotor = tiltdatabase
        print('tilt : ',tiltmotor)
    '''
        
        

