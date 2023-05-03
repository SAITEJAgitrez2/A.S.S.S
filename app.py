#================================================================ 
#Import necessary libraries
from flask import Flask, render_template, Response, request
import cv2
import os
import subprocess
import time
#Deploying the app using flask
#Initialize the Flask app

app = Flask(__name__)

@app.route('/')
def Shady():
    return render_template('Shady.html')
    
@app.route('/FallDetection', methods=['GET', 'POST'])
def FallDetection():
    global case
    case = 'fall'
   
    return render_template('FallDetection.html')

    
@app.route('/ObjectDetection', methods=['GET', 'POST'])
def ObjectDetection():
    global case
    case = 'object'
    return render_template('ObjectDetection.html')
    
@app.route('/VehicleCrashDetection', methods=['GET', 'POST'])
def VehicleCrashDetection():
    global case
    case = 'vehicle'
    return render_template('VehicleCrashDetection.html')

@app.route('/ContactUs')
def ContactUs():
    return render_template('ContactUs.html')
    
@app.route('/Video', methods=['GET', 'POST'])
def VideoF():
    #global video_link
    video_linkF = request.form.get('videolinkF')
    #if mprg=='fall':
    path_fall = video_linkF.strip()
    # Define the file path and arguments for the second Python code you want to run
    file_path = "D:/randomtrys/Fall-Detection-using-YOLOv7-Pose-Estimation/run_pose.py"
    args = ['--source', path_fall, '--device', '0']
    # Use the os module to change the working directory to the directory containing the second Python code
    os.chdir(file_path.rsplit('/', 1)[0])
    # Use subprocess to run the second Python code with the specified arguments
    subprocess.run(['python', file_path] + args, check=True)
    # Change the working directory back to the original directory
    script_path = "D:/randomtrys/S.H.A.D.Y-main/S.H.A.D.Y-main/DeployedApptheonewhichisworking/DeployedApp/app.py"

# get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(script_path))

# change the current working directory to the script directory
    os.chdir(script_dir)
    time.sleep(3)
    return render_template('Video.html')

#def video():
    # Serve the video file
 #    return send_file('C:/Users/raj20/Desktop/A.S.S.S-main/A.S.S.S-main/Fall-Detection-using-YOLOv7-Pose-Estimation/output_result.mp4', mimetype='video/mp4')

# for object detection
@app.route('/VideoO', methods=['GET', 'POST'])
def VideoO():
    
    video_linkO = request.form.get('videolinkO')
    #if mprg=='fall':
    path_fall = video_linkO.strip()
    # Define the file path and arguments for the second Python code you want to run
    file_path = "D:/v8/pythonprj/yolov8/runyolo/yolov8tresspassing.py"
    args = ['--source',path_fall]
    # Use the os module to change the working directory to the directory containing the second Python code
    os.chdir(file_path.rsplit('/', 1)[0])
    # Use subprocess to run the second Python code with the specified arguments
    subprocess.run(['python',  file_path], check=True)
    # Change the working directory back to the original directory
    script_path = "D:/randomtrys/S.H.A.D.Y-main/S.H.A.D.Y-main/DeployedApptheonewhichisworking/DeployedApp/app.py"

# get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(script_path))

# change the current working directory to the script directory
    os.chdir(script_dir)
    time.sleep(3)
    return render_template('Video.html')

#for carcrash detection
@app.route('/VideoC', methods=['GET', 'POST'])
def VideoC():
    
    video_linkC = request.form.get('videolinkC')
    #if mprg=='fall':
    path_fall = video_linkC.strip()
    # Define the file path and arguments for the second Python code you want to run
    file_path = "D:/v8/pythonprj/yolov8/runyolo/carcrash.py"
    args = ['--source', path_fall]
    # Use the os module to change the working directory to the directory containing the second Python code
    os.chdir(file_path.rsplit('/', 1)[0])
    # Use subprocess to run the second Python code with the specified arguments
    subprocess.run(['python', file_path] + args, check=True)
    # Change the working directory back to the original directory
    script_path = "D:/randomtrys/S.H.A.D.Y-main/S.H.A.D.Y-main/DeployedApptheonewhichisworking/DeployedApp/app.py"

# get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(script_path))

# change the current working directory to the script directory
    os.chdir(script_dir)
    time.sleep(3)
    return render_template('Video.html')

@app.route('/video_feed')
def video_feed():
    return render_template('Video.html')

if __name__ == "__main__":
    app.run(debug=True)