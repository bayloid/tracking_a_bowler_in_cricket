#--------------------------------------------------------------------------------------------------     
# Mediapipe Helper Program              
#--------------------------------------------------------------------------------------------------     
#--------------------------------------------------------------------------------------------------     
# Imports                                                                           
#--------------------------------------------------------------------------------------------------   
print("Importing external libraries...")
import ffmpeg
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import urllib.request
import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import PyQt5
from PIL import Image
from IPython.display import Video
from IPython.display import display
import importlib
import math
import matplotlib.ticker as ticker

nb_helpers = importlib.import_module("mediapipe-python.nb_helpers")
print("Imports successful!")
#------------------------------------------------------------------------------
def directory_size(dir_path):
    """
    Returns the number of files in a directory.

    :param dir_path: path of the directory (from working directory)
    :return: int value of number of files in the directory
    """
    count = 0
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
    return count
    

def load_directories():
    """
    Takes the directory paths off of the command line call
    
    :returns: two strings, the paths of the two given directories.
    """
    directory_path = sys.argv[1]
    directory_path2 = sys.argv[2]
    return directory_path, directory_path2

def process_video(path):
    """
    Processess a video using the mediapipe library, capturing the landmarks
    for each frame. Video is opened and read by OpenCV (cv2).

    :param path: path of the file from the working directory.
    :return: 

    """
    file = path
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic
    landmarks_by_frame = [[]]
    try:
         with mp_pose.Pose(min_detection_confidence=0.2,
                              min_tracking_confidence=0.2) as pose:
             cap = cv2.VideoCapture(file)

             if cap.isOpened() == False:
                 print("Error opening video file!")
                 raise TypeError

             print(f"\nFile {file} opened for processing...")

             length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
             data = []
             frame_num = 0

             while cap.isOpened:
                    ret, image = cap.read()
                    if not ret:
                        break

                    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    results = pose.process(image)
                    try:
                        landmarks = results.pose_world_landmarks.landmark
                        data.append(landmarks)
                    except AttributeError:
                        print("Landmark not found, skipping frame!")
                    frame_num += 1
    except TypeError:
         print(f"Error with file, {file}!")
                        
    return data

def process_dir(dir_path,export=False,export_name=None):
    """
    process_dir iterates through a given directory and attempts to process each video
    in the directory and returns the landmarks. An export can also be made to use with
    other functions.

    :param dir_path: path of the directory of videos to process (given from the current
    working directory).
    :param export: boolean value, if true, an export is made to a text file of all directories.
    :param export_name: the user can provide a name for the export file, if none is given, the
    :export file name is based off of the directory path name,
    :return: list of landmarks by video and frame.

    """
    dir_len = directory_size(dir_path)
    dir_landmarks = [[]]*dir_len
    index = 0
    for filename in sorted (os.listdir(dir_path)):
        file = os.path.join(dir_path,filename)
        video_landmarks = []
        video_landmarks = process_video(file)
        dir_landmarks[index] = video_landmarks
        index+=1
        
    if(export):
        print("Exporting landmarks...")
        if(export_name == None):
            export_name = dir_path + "_export.txt"
        try:
            print(f"Opening export file: {export_name}...")
            export_file = open(export_name,"w")
            for video in dir_landmarks:
                for landmarks in video:
                    for landmark in landmarks:
                        x = str(landmark.x)
                        y = str(landmark.y)
                        z = str(landmark.z)
                        string = x+","+y+","+z+";"
                        export_file.write(string)
                    export_file.write("\nnext_frame\n")
                export_file.write("\nnext_video\n")
            export_file.close()
        except IOError:
            print("IOError has occured")
        try:
            filenames_file = open("filenames_export.txt","w")
            for filename in os.listdir(dir_path):
                filenames_file.write(filename+"\n")
            filenames_file.close()
        except IOError:
            print("IOError has occured")
    
    return dir_landmarks

def get_landmarks_by_landmark_pos(video_landmarks,index):
    """
    get_landmarks_by_landmark_pos takes a list of landmarks in a given video iterates through
    frame by frame and picks out the landmark position given by index. Each landmark position
    on the pose has a given index.

    :param video_landmarks: a list of landmarks in a given video split by frame and landmark
    position
    :param index: an integer value of the position of the given landmark. should be an int
    between 0-32. As per mediapipe documentation, the landmarks are given as:
    0  - nose
    1  - left eye (inner)
    2  - left eye
    3  - left eye (outer)
    4  - right eye (inner)
    5  - right eye
    6  - right eye (outer)
    7  - left ear
    8  - right ear
    9  - mouth (left)
    10 - mouth (right)
    11 - left shoulder
    12 - right shoulder
    13 - left elbow
    14 - right elbow
    15 - left wrist
    16 - right wrist
    17 - left pinky
    18 - right pinky
    19 - left index
    20 - right index
    21 - left thumb
    22 - right thumb
    23 - left hip
    24 - right hip
    25 - left knee
    26 - right knee
    27 - left ankle
    28 - right ankle
    29 - left heel
    30 - right heel
    31 - left foot index
    32 - right foot index
    :returns: list of coordinates at the given landmark by frame.
    
    """
    landmark_coordinates = []
    num_frames = len(video_landmarks)
    for j in range(0,num_frames):
        frame_landmarks = get_landmarks_by_frame_index(j,video_landmarks)
        if(len(frame_landmarks)>=index):
            landmark_coordinates.append(frame_landmarks[index])

    return landmark_coordinates

        
def get_release_point(wrist_coordinates,elbow_coordinates,shoulder_coordinates):
    """

    get_release_point takes a list of wrist_coordinates by frame (as outputted by
    get_landmarks_by_landmark_pos with index 16 (for right arm bowler) and elbow
    coordinates are outputted by the same function with index 14. For the sake of
    simplicity, the release point is given by the highest point by bowling arm reaches
    during the bowling action.

    :param wrist_coordinates: a list of x,y,z coordinates for the wrist by frame.
    :param elbow_coordinates; a list of x,y,z coordinates for the elbow by frame.
    :returns: two pairs of coordinates, one for the wrist x,y and one for the elbow
    x,y.
    
    """
    coord_length = len(wrist_coordinates)
    wrist_release_y = 9999.9
    wrist_release_x = 0.0
    elbow_release_y = 0.0
    elbow_release_x = 0.0
    shoulder_release_y = 0.0
    shoulder_release_x = 0.0
    release_frame = 0
    for i in range(0,coord_length):
        if(float(wrist_coordinates[i][1]) < wrist_release_y):
            wrist_release_y = float(wrist_coordinates[i][1])
            wrist_release_x  = float(wrist_coordinates[i][0])
            elbow_release_x = float(elbow_coordinates[i][0])
            elbow_release_y = float(elbow_coordinates[i][1])
            shoulder_release_y = float(shoulder_coordinates[i][1])
            shoulder_release_x = float(shoulder_coordinates[i][0])
            release_frame = i

    return [wrist_release_x,wrist_release_y],[elbow_release_x,elbow_release_y],[shoulder_release_x,shoulder_release_y],release_frame

def plot_landmarks(landmarks):
    """
    plot_landmarks plots a list of landmarks on a scatter graph.

    :param landmarks: coordinates of landmarks with x at index [,0] and y at [,1]

    """
    landmarks = np.array(landmarks)
    plt.scatter(landmarks[:,0],landmarks[:,1])
    plt.show()
    
def compare_landmarks(l1,l2,title="Comparison Between Two Sets of Landmarks",
                      x_label="X",y_label="Y"):
    """
    compare_landmarks takes two lists of landmark coordinates and plots them on the
    same scatter graph to compare the positions.

    """
    l1=np.array(l1)
    l2=np.array(l2)
    plt.scatter(l1[:,0],l1[:,1])
    plt.scatter(l2[:,0],l2[:,1])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    plt.show()

def get_landmarks_by_video_index(index,landmarks_array):
    landmarks = landmarks_array[index]
    return landmarks

def get_landmarks_by_frame_index(frame,landmarks):
    return landmarks[frame]

def get_release_points_of_dir(dir_landmarks):
    release_points_by_video = []
    elbow_at_release_by_video = []
    shoulder_at_release_by_video = []
    release_frames = []
    for i in range(0,len(dir_landmarks)):
        video_landmarks = get_landmarks_by_video_index(i,dir_landmarks)
        wrist_coordinates = get_landmarks_by_landmark_pos(video_landmarks,16)
        elbow_coordinates = get_landmarks_by_landmark_pos(video_landmarks,14)
        shoulder_coordinates = get_landmarks_by_landmark_pos(video_landmarks,12)
        release_point , elbow_at_release ,shoulder_at_release,release_frame = get_release_point(wrist_coordinates,
                                                                   elbow_coordinates,
                                                             shoulder_coordinates)
        release_points_by_video.append(release_point)
        elbow_at_release_by_video.append(elbow_at_release)
        shoulder_at_release_by_video.append(shoulder_at_release)
        release_frames.append(release_frame)
        
    return release_points_by_video, elbow_at_release_by_video, shoulder_at_release_by_video, release_frames

def calc_straightness(release_points,elbow_points,shoulder_points):
    length = len(release_points)
    straightness_lst = []
    for i in range(0,length):
        rp_x, rp_y = release_points[i]
        ep_x, ep_y = elbow_points[i]
        sh_x, sh_y = shoulder_points[i]
        straightness = rp_x-ep_x
        height = rp_y - sh_y
        straightness_lst.append([straightness,height])

    return straightness_lst

def import_landmarks(file_name):
    landmarks_by_video = []
    try:
        file = open(file_name,"r")
        content = file.read()
        videos = content.split("\nnext_video\n")
        for video in videos:
            frames = video.split("\nnext_frame\n")
            landmarks_by_frame = []
            for frame in frames:
                landmark_list = []
                landmarks = frame.split(';')
                for landmark in landmarks:
                    xyz = landmark.split(',')
                    if (xyz != [''] and xyz !=[]):
                        landmark_list.append(xyz)
                if(landmark_list!=[''] and landmark_list!=[]):    
                    landmarks_by_frame.append(landmark_list)
            if(landmarks_by_frame != []):        
                landmarks_by_video.append(landmarks_by_frame)

    except IOError:
        print("An error occured during import!")
        return []
    return landmarks_by_video

def import_filenames(filename):
    filenames = []
    try:
        file = open(filename,"r")
        lines = file.readlines()
        filenames = [line.strip() for line in lines if line.strip() != ""]
        file.close()
    except IOError:
        print("Error occured during filename import!")

    return filenames
def calc_elbow_angle(wrist_at_release,elbow_at_release,shoulder_at_release):
    angles_by_video = []

    for i in range(0,len(wrist_at_release)):
        c = wrist_at_release[i]
        b = elbow_at_release[i]
        a = shoulder_at_release[i]

        angle = angle_of_two_vectors(a,b,c)

        angles_by_video.append(angle)

    return angles_by_video

def angle_of_two_vectors(a,b,c):
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c

    AB = (x2-x1,y2-y1)
    BC = (x3-x2,y3-y2)

    dot_product = AB[0] * BC[0] + AB[1] * BC[1]

    magnitude_AB = math.sqrt(AB[0]**2 + AB[1]**2)
    magnitude_BC = math.sqrt(BC[0]**2 + BC[1]**2)

    angle_radians = math.acos(dot_product / (magnitude_AB * magnitude_BC))

    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees

def plot_skeleton(frame_landmarks):
    landmarks = np.array(frame_landmarks)
    x = []
    y = []
    for coordinates in landmarks:
        x.append(float(coordinates[0])*-1)
        y.append(float(coordinates[1])*-1)

    x=np.array(x)
    y=np.array(y)

    print(x)
    print(y)
    
    plt.figure(figsize=(6,6))
    plt.scatter(x,y)
    
    plt.gca().set_aspect('equal')
    tick_step = 0.2  # You can change this to whatever fits your data
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(tick_step))
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(tick_step))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def extract_frame(frame_num, video_path,output_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    ret, frame = cap.read()
    
    if ret:
        
        success = cv2.imwrite(output_path,frame)
        if not success:
            print("Error writing image, please provide a valid .jpg path!")

    cap.release()
