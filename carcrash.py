
import argparse
from ultralytics import YOLO
import cv2
import cvzone
from sort import *
from itertools import combinations
import telepot
import math


token = '5874824687:AAFcoH_7pxY_cQ2d87WuyMkidwrwJ1BhDx0'
receiver_id = '1646471129'
bot = telepot.Bot(token)

parser = argparse.ArgumentParser(description='Car crash detection using yoloV8')

# add optional argument
parser.add_argument('--source', type=str, help='input file path')
#parser.add_argument('--output', type=str, help='output file path')

args = parser.parse_args()

# access argument values
input_file = args.source
#output_file = args.output

path=input_file
if path == 0:
    cap= cv2.VideoCapture(0)
elif path == 1:
    cap=cv2.VideoCapture()
else:
    cap = cv2.VideoCapture(path)

vw, vh = int(cap.get(3)), int(cap.get(4))
cap.set(3,vw)
cap.set(4,vh)

#cap = cv2.VideoCapture("./videos/videoplayback.mp4")
#frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
model = YOLO('./Yolo-Weights/yolov8m.pt')

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

tracker = Sort(max_age=1, min_hits=2,iou_threshold=0.3)
codec = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter('./results_crash/crash_processed.avi' , codec, 24, (vw, vh))
while True:
    cen=dict()
    success, img = cap.read()
    results = model(img,stream=True)
    # if (cap.isOpened() == False):
    #     print('Error while trying to read video. Please check path again')
    # frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    # vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0]
    # resize_height, resize_width = vid_write_image.shape[:2]
    # #out_video_name = "output" if path.isnumeric else f"{input_path.split('/')[-1].split('.')[0]}"
    # out = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 24,(resize_width, resize_height))
    detections = np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #for open cv
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 =int(x1),int(y1),int(x2),int(y2)

            #(x1,y1) = (top left corner of bbox)
            #(x2,y2) = (bottom rignt corner point)

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
            if (currentClass == "car" or currentClass == "bus" or currentClass == "truck" or currentClass == "motorcycle" or currentClass == "bicycle") and conf>0.55 :
                    cvzone.cornerRect(img, (x1, y1, w, h), l=10, t=2, colorR=(255, 255, 0), colorC=(0, 0, 0))
                    #cvzone.putTextRect(img,f'{classNames[cls]} : {conf}', (max(0,x1+2), max(20, y1+10)), scale=0.7, thickness=1, colorT=(0,0,255), colorR=(255,255,0), offset=2) #putTextRect(img, text, pos, scale=3, thickness=3, colorT=(255, 255, 255),colorR=(255, 0, 255), font=cv2.FONT_HERSHEY_PLAIN,offset=10, border=None, colorB=(0, 255, 0)): #uncomment this to show class name and confidence
                    currentArray = np.array([x1,y1,x2,y2,conf])
                    detections = np.vstack((detections,currentArray))


    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1,y1,x2,y2,id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        #print(result)
        w, h = x2 - x1, y2 - y1
        #cvzone.cornerRect(img, (x1, y1, w, h), l=10, t=2, colorR=(255, 255, 255), colorC=(0, 0, 0))
       # cvzone.putTextRect(img, f'{int(id)}', (max(0, x1 + 2), max(20, y1 -20)), scale=0.7,
                           #thickness=1, colorT=(0, 0, 255), colorR=(255, 255, 0), offset=2)
        # center points of bounding boxs
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cen[id] = (int(cx), int(cy), int(x1), int(y1), int(x2), int(y2))
        vehicle_red_zone_list=[]
        vehicle_red_line_list=[]
        for (id1, p1), (id2, p2) in combinations(cen.items(),2):
            if not ((p1[2] >= p2[4]) or (p1[4] <= p2[2]) or (p1[5] <= p2[3]) or (p1[3] >= p2[5])):
                if id1 not in vehicle_red_zone_list:
                    vehicle_red_zone_list.append(id1)  # Add Id to a list
                    vehicle_red_line_list.append(p1[0:2])  # Add points to the list
                if id2 not in vehicle_red_zone_list:
                    vehicle_red_zone_list.append(id2)  # Same for the second id
                    vehicle_red_line_list.append(p2[0:2])
        for idx, box in cen.items():  # dict (1(key):red(value), 2 blue)  idx - key  box - value
            if idx in vehicle_red_zone_list:  # if id is in red zone list
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 0, 255),2)  # Create Red bounding boxes  #starting point, ending point size of 2

                #cvzone.cornerRect(img, (x1, y1, w, h), l=10, t=4, colorR=(0, 0, 255), colorC=(0, 0, 255))
            #else:
                #cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 1)  # Create Green bounding boxes
                #cvzone.cornerRect(img, (x1, y1, w, h), l=10, t=2, colorR=(0, 255, 0), colorC=(0, 0, 0))
                #time_start_falling = 0
                #time_now = 0

    out.write(img)
    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

#model = YOLO('../Yolo-Weights/yolov8m.pt')
#results = model("images/bus.jpg", show=True)
#cv2.waitKey(0)