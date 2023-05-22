### Run with webcam, cpu
python run_pose.py --source 0 

### Run with other IO device
python run_pose.py --source path_of_device

### Run with webcam, gpu (0: all threads, 1, 2, 3,...)
python run_pose.py --source 0 --device 0
