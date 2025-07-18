# Tracking a Bowler in Cricket

Welcome to my Git repository. The objective of this project is to gain insight into a bowler's action in cricket by tracking their skeleton from 2D video and using landmarks to make inferences on the attempted delivery. Originally concieved as part of my final year university project at the University of Essex.

## Usage

This repository uses a Python Virtual Environment. The environment is already installed and needs to be activated before code can be run. To activate, open the mediapipe directory and type the following into your command line or terminal:

    cd mediapipe/
    source bin/activate

Without activation, the external libraries won't be found and the code will not run.

Once in inside the active virtual environment, install required dependecies with the following command:

    pip install -r requirements.txt

### Batch Processing Videos

Due to **OpenCV**, the library used for opening and parsing videos, all videos you wish to process **must** be in *.mp4* format. Create an empty directory/folder and place all *.mp4* videos into it, ensuring only .mp4  extension videos are contained in the folder. You must also provide an output file path, which **must** be a *.txt* file path. To process the videos, type the following:

    python3 mp_process.py <video directory path> <output file path>

Some messages will occur here, but as long as the process keeps running, you can ignore these.

You will also be provided a *.txt* file named *filename_export.txt* containing the filenames of the files you processed in the order they were processed. To rename this to save it getting overwritten on the next batch process, type the following:

    mv filename_export.txt <new file name>

### Throwing Detection

An ability of this software is to measure the elbow flexion angle of the right arm bowler's bowling arm. Once you have processed your videos, the *.txt* file provided for the export is used to perform this analysis. To analyse your data, type the following:

    python3 throwing_detection.py <output file path> <filename export path>

Once the libraries are imported, the data is analysed and a bar chart will be produced showing the angles of the elbow flexion with the ICC regulation 15 degree threshold line.

### Building a 3D Animated Skeleton from 2D Video

A file conatining a bowlers delivery (from run up, through the delivery) can be passed to mediapipe_video_3d.py and run like so:

    python3 mediapipe_video_3d.py <video>.mp4

The output file will be created as a .mp4 file with the name:

    bowling_wireframe.mp4

