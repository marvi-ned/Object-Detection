import cv2
import numpy as np

# img = cv2.imread('lena.png')
thres = 0.6
cap = cv2.VideoCapture('Road_traffic_video2.mp4')
cap.set(3,640)
cap.set(4,480)

classNames = []
classFile = 'coco.names'
with open(classFile, 'r') as f:
    classNames = f.read().splitlines()
# print(classNames)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(200,200)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

while(cap.isOpened()):
    success,img = cap.read()
    classIDs, confs, bbox = net.detect(img,confThreshold=thres)
    print(classIDs,bbox)

    if len(classIDs) !=0:
        for classID,confidence,box in zip(classIDs.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classID-1].upper(),(box[0]+10,box[1]+30),
            cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)

    cv2.imshow('Output',img)
    cv2.waitKey(1) & 0xFF == ord('q')