import torch
from ultralytics import YOLO
import cv2
import cvzone
from sort import *
import telepot
import math
import time

token = '5874824687:AAFcoH_7pxY_cQ2d87WuyMkidwrwJ1BhDx0'
receiver_id = '1646471129'
bot = telepot.Bot(token)
parser = argparse.ArgumentParser(description='Car crash detection using yoloV8')

# add optional argument
parser.add_argument('--source', type=str,default=0, help='input file path')
#parser.add_argument('--output', type=str, help='output file path')

args = parser.parse_args()

# access argument values
input_file = args.source
#output_file = args.output

path=input_file
print(path)


# vw, vh = int(cap.get(3)), int(cap.get(4))
# cap.set(3,vw)
# cap.set(4,vh)





#cap = cv2.VideoCapture(0) #webcam
#cap.set(3,1280)
#cap.set(4,720)
cap=cv2.VideoCapture(path)
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
#cap = cv2.VideoCapture("./videos/videoplayback.mp4")

model = YOLO('./Yolo-Weights/yolov8s.pt')

#list of class names
classNames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush']

codec = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter('./results_person/person_processed.avi' , codec, 24, (frame_width, frame_height))
while True:
    success, img = cap.read()
    results = model(img,stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #for open cv
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 =int(x1),int(y1),int(x2),int(y2)
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
                #for cvzone(custom lib for cool bounding boxs)                           B   G   R
                #def cornerRect(img, bbox, l=30, t=5, rt=1,colorR=(255, 0, 255), colorC=(0, 255, 0)):
            w, h = x2-x1, y2-y1
                #cvzone.cornerRect(img,(x1,y1,w,h), l=10, t=2, colorR=(255,255,0), colorC=(0,0,255))
                #confidence
            conf = math.ceil((box.conf[0]*100))/100
                #class name
            cls = int(box.cls[0])
            currentClass=classNames[cls]
            if currentClass == "person" and conf>0.3:
                cvzone.cornerRect(img, (x1, y1, w, h), l=10, t=2, colorR=(255, 255, 0), colorC=(0, 0, 255))
                cvzone.putTextRect(img,f'{classNames[cls]} : {conf}', (max(0,x1+2), max(20, y1+10)), scale=0.7, thickness=1, colorT=(0,0,255), colorR=(255,255,0), offset=2) #putTextRect(img, text, pos, scale=3, thickness=3, colorT=(255, 255, 255),colorR=(255, 0, 255), font=cv2.FONT_HERSHEY_PLAIN,offset=10, border=None, colorB=(0, 255, 0)):
    out.write(img)
    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

