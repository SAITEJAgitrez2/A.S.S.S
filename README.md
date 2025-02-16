# A.S.S.S
Automated Safety Surveillance System 

Installation 
Before following the following steps make sure the host system meets the hardware requirement, Nvidia GPU is needed.

1) Setting up CUDA and CUDNN (version cuda 11.0)
2) Setting up python (version 3.10.9)
3) Install an IDE (Visual Studio code/ PyCharm)
4) Install specified python modules in requirement.txt files



## Overview
The **Automated Safety Surveillance System** is designed to detect and alert authorities about **trespassing, vehicle crashes, and human falls** in real-time. Using advanced **YOLO-based deep learning models**, this system improves security and reduces response times in **public spaces, workplaces, and transportation**.

## Abstract
This study presents an **automated surveillance system** that detects anomalies using **state-of-the-art YOLO versions (YOLOv7 & YOLOv8)** for **pose estimation and object detection**. The system is integrated with **Telegram messaging services** to provide instant alerts upon detecting potential hazards.  
The results demonstrate **high accuracy** in identifying trespassing, falls, and crashes, making it a **reliable alternative** to manual surveillance.

## Key Features
- **Real-time Trespassing Detection** 🏠  
- **Vehicle Crash Detection** 🚗💥  
- **Human Fall Detection** 🏃⬇️  
- **Automated Alerts via Telegram** 📩  
- **Uses YOLOv7 for Pose Estimation & YOLOv8 for Object Detection** 🎯  

## Methodology
1. **Data Collection**  
   - Utilized large-scale datasets like **COCO** for training models.  
2. **Model Selection & Training**  
   - Used **pre-trained YOLOv7 & YOLOv8** models to eliminate the need for custom training.  
3. **Real-Time Processing**  
   - Implemented on hardware with **GPU acceleration (CUDA, CUDNN, PyTorch)** for faster detections.  
4. **Alert System Implementation**  
   - Telegram bot sends alerts upon detecting any anomaly using **OpenCV** for image capture.  

## Technologies Used
| **Component**         | **Technology**      |
|-----------------------|--------------------|
| **Machine Learning**  | YOLOv7, YOLOv8 |
| **Programming**       | Python, OpenCV |
| **Libraries**         | NumPy, PyTorch, TensorFlow |
| **Frameworks**        | Flask (Web UI), Telegram API |
| **Hardware**         | GPU: Nvidia GTX 1660 Ti, RAM: 32GB DDR4 |

## Results
- The system effectively detects and classifies **trespassing, falls, and vehicle crashes**.
- **YOLO-based detection is significantly faster** than previous RCNN and Faster-RCNN approaches.
- Telegram bot alerts are sent **instantly**, reducing response time in critical situations.
- **High accuracy and reduced false positives** compared to traditional fall detection methods.

## Conclusion
This project successfully demonstrates **a real-time, automated safety surveillance system** that **outperforms traditional methods** in speed, accuracy, and automation.  
The YOLO-based models provide **efficient detection and classification**, making it a **valuable solution for security and public safety**.  

## Future Enhancements
- **Integration with IoT devices** for more comprehensive monitoring.  
- **Enhancing YOLO models** to improve detection accuracy in crowded environments.  
- **Deploying on cloud infrastructure** for scalable, real-time processing.  
- **Adding audio-based alerts** alongside Telegram notifications.  

## References
1. **Fang Li, M. K. H. Leung, M. Mangalvedhekar, M. Balakrishnan**, _"Automated Video Surveillance and Alarm System,"_ 2008 International Conference on Machine Learning and Cybernetics.  
2. **Sumathi R, P. Raveena, P. Rakshana, P. Nigila, P. Mahalakshmi**, _"Real-Time Protection of Farmlands from Animal Intrusion,"_ 2022 IEEE World Conference on Applied Intelligence and Computing.  
3. **Juan R. Terven, Diana M. Cordova-Esparaza**, _"A Comprehensive Review of YOLO: From YOLOv1 to YOLOv8 and Beyond."_
