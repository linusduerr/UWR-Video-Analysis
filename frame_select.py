from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np

def nothing(x):
    pass

# Input argument parsing
parser = argparse.ArgumentParser(description='This program takes input video streams of a UWR game and switches between the angles automatically')
parser.add_argument('--angles', type=int, help='Number of different angles to be analysed', default=2)
parser.add_argument('--filename', type=str, help='Base file name without angle index and extension', default='')
parser.add_argument('--ext', type=str, help='File extension for video files', default='.mp4')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2, GSOC).', default='MOG2')
parser.add_argument('--delay', type=str, help='Frame number to start videos on.', default='1')
parser.add_argument('--input', type=str, help='Name of the input folder containing the clips can be given here.', default='input')
parser.add_argument('-pauses', action='store_true', help='If this argument is given, the program will attempt to cut game pauses out')
args = parser.parse_args()

# Constants
INPUT_PATH = args.input + '/'
ANGLES = args.angles
MIN_WAIT = 60                       # Minimum number of frames between conecutive angle changes
LOWER_WEIGHT = 3                    # Factor with which the lower part of the image is weighted (Upper is weighted with 1)
LOWER_THRESH = 3                    # Threshold for the mean of the lower image below which a pause is detected
UPPER_THRESH = 35                   # Threshold for the mean of the upper image below which a pause is detected
PAUSE_END_FRAMES = 150              # Number of frames from the end of a pause added back into the final video
JUMP_FRAMES = 60                    # Number of frames to skip when hitting spacebar

# Creates window to display the output frame
cv.namedWindow('Frame', cv.WINDOW_NORMAL)
cv.resizeWindow('Frame', 320, 180)

# Creates all other windows for each angle (Foreground mask, input stream, background detected by algorithm)
for i in range(ANGLES):
    cv.namedWindow('Mask ' + str(i), cv.WINDOW_NORMAL)
    cv.namedWindow('In ' + str(i), cv.WINDOW_NORMAL)
    cv.namedWindow('BG ' + str(i), cv.WINDOW_NORMAL)
    cv.resizeWindow('Mask ' + str(i), 320, 180)
    cv.resizeWindow('In ' + str(i), 320, 225)
    cv.resizeWindow('BG ' + str(i), 320, 180)
    cv.createTrackbar('Middle', 'In ' + str(i), 50, 100, nothing)

# Creates background subtractors for each angle
if args.algo == 'MOG2':
    backSub = [cv.createBackgroundSubtractorMOG2() for i in range(ANGLES)]
elif args.algo == 'KNN':
    backSub = [cv.createBackgroundSubtractorKNN() for i in range(ANGLES)]
elif args.algo == 'GSOC':
    backSub = [cv.bgsegm.createBackgroundSubtractorGSOC() for i in range(ANGLES)]

# Filenames of all input clips
capNames = [args.filename + str(i) + args.ext for i in range(ANGLES)]

learningRates = [0.0002 for i in range(ANGLES)]

# Video captures for each angle
caps = [cv.VideoCapture(cv.samples.findFileOrKeep(INPUT_PATH + capNames[i])) for i in range(ANGLES)]
for i in range(ANGLES):
    if not caps[i].isOpened():
        print('Unable to open angle ' + str(i))
        exit(0)

# Output video writer
writer = cv.VideoWriter('out.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))

# If a delay was given, this sets all video captures to the corresponding frame
for i in range(ANGLES):
    caps[i].set(1, int(args.delay))

# Factors for each input stream determining the percentage of the image counted as upper part
middleFactors = [0.5 for i in range(ANGLES)]

# Tries to determine whether the game is running based on where players are
def is_game_paused(masks, middleFactors):
    lowerMeans = [masks[i][round(middleFactors[i]*masks[i].shape[0]) :, :].mean() for i in range(ANGLES)]
    upperMeans = [masks[i][: round(middleFactors[i]*masks[i].shape[0]), :].mean() for i in range(ANGLES)]
    return all([lowerMeans[i] < LOWER_THRESH for i in range(ANGLES)]) and all([upperMeans[i] < UPPER_THRESH for i in range(ANGLES)])

# Scores the masks such that the highest score is given to the frame where most action is taking place
def score(masks, middleFactors):
    lowerMeans = [masks[i][round(middleFactors[i]*masks[i].shape[0]) :, :].mean() for i in range(ANGLES)]
    upperMeans = [masks[i][: round(middleFactors[i]*masks[i].shape[0]), :].mean() for i in range(ANGLES)]
    return [((LOWER_WEIGHT + 1 - 2 * middleFactors[i]) / (2 * (1 - middleFactors[i])))*lowerMeans[i] + upperMeans[i] for i in range(ANGLES)]

# This list stores frames during a detected pause to fill back in some frames at the end of each pause
# Since the detection tends to detect late when the game has restarted
pauseframes = []

# Stores the currently selected viewing angle
angle = -1

# These variables are used to prevent angle switches to fast after each other
switch_allowed = True
count = 0

while True:
    # Setting image split based on trackbars
    for i in range(ANGLES):
        middleFactors[i] = (cv.getTrackbarPos('Middle', 'In ' + str(i)) + 1) / 102

    # Reading frames
    frames = [caps[i].read() for i in range(ANGLES)]
    frames = [frames[i][1] for i in range(ANGLES)]

    # Checking that frames were read successfully
    for i in range(ANGLES):
        if frames[i] is None:
            print("No more frames in angle " + str(i))
            break    
    
    # Reducing size of frames to perform analysis
    smallFrames = [cv.resize(frames[i], (320, 180), interpolation=cv.INTER_LINEAR) for i in range(ANGLES)]

    # Doing the background subtraction
    masks = [backSub[i].apply(smallFrames[i], learningRate=learningRates[i]) for i in range(ANGLES)]

    # Getting the currently detected background
    backgrounds = [backSub[i].getBackgroundImage() for i in range(ANGLES)]

    # Increment frame counter since last switch and check if the minimum waiting time has been reached
    count += 1
    if not switch_allowed and count > MIN_WAIT:
        switch_allowed = True

    # Calculate the highest scoring angle
    scores = np.array(score(masks, middleFactors))
    for i in range(ANGLES):
        print(str(i) + ': ' + str(round(scores[i])), end=', ')
    print('', end='\r')
    maxScoredAngle = np.argmax(scores)

    # If allowed, switch to that angle
    if angle == maxScoredAngle:
        frame = frames[maxScoredAngle]
    elif switch_allowed:
        frame = frames[maxScoredAngle]
        angle = maxScoredAngle
        switch_allowed = False
        count = 0
    else:
        frame = frames[angle]

    # If a pause is detected, store frames and add red border around output frame
    if args.pauses and is_game_paused(masks, middleFactors):
        pauseframes.append(frame)
        if len(pauseframes) > PAUSE_END_FRAMES: 
            pauseframes.pop(0)
        frame = cv.copyMakeBorder(frame, 20, 20, 20, 20, cv.BORDER_CONSTANT, None, (0,0,255))
    else:
        # Else fill in the end of the last pause and write frame to output
        if len(pauseframes) > 0:
            for i in range(len(pauseframes)):
                writer.write(pauseframes[i])
            pauseframes = []
        writer.write(frame)
    
    # Display all important images in windows
    cv.imshow('Frame', frame)
    for i in range(ANGLES):
        cv.imshow('Mask ' + str(i), masks[i])
        cv.imshow('In ' + str(i), cv.line(smallFrames[i],(0, round(middleFactors[i]*masks[i].shape[0])),(masks[i].shape[1] - 1, round(middleFactors[i]*masks[i].shape[0])),(0,0,255),1))
        cv.imshow('BG ' + str(i), backgrounds[i])
    
    # Handles key presses
    keyboard = cv.waitKey(1)
    if keyboard == 'q' or keyboard == 27:
        break
    elif keyboard == 32:
        frameNum = caps[0].get(cv.CAP_PROP_POS_FRAMES)
        for i in range(ANGLES):
            caps[i].set(1, frameNum + JUMP_FRAMES)
