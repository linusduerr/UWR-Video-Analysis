from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--L', type=str, help='Filename of left clips without extension', default='L')
parser.add_argument('--R', type=str, help='Filename of right clips without extension', default='R')
parser.add_argument('--ext', type=str, help='File extension for video files', default='.mp4')
parser.add_argument('--clipsL', type=str, help='Number of clips left video is divided into', default='1')
parser.add_argument('--clipsR', type=str, help='Number of clips right video is divided into', default='1')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2, GSOC).', default='MOG2')
args = parser.parse_args()

cv.namedWindow('Frame', cv.WINDOW_NORMAL)
cv.namedWindow('FG Mask Left', cv.WINDOW_NORMAL)
cv.namedWindow('FG Mask Right', cv.WINDOW_NORMAL)
cv.namedWindow('Left', cv.WINDOW_NORMAL)
cv.namedWindow('Right', cv.WINDOW_NORMAL)
cv.namedWindow('Left BG', cv.WINDOW_NORMAL)
cv.namedWindow('Right BG', cv.WINDOW_NORMAL)

cv.resizeWindow('Frame', 320, 180)
cv.resizeWindow('FG Mask Left', 320, 180)
cv.resizeWindow('FG Mask Right', 320, 180)
cv.resizeWindow('Left', 320, 180)
cv.resizeWindow('Right', 320, 180)
cv.resizeWindow('Left BG', 320, 180)
cv.resizeWindow('Right BG', 320, 180)

if args.algo == 'MOG2':
    lBackSub = cv.createBackgroundSubtractorMOG2()
    rBackSub = cv.createBackgroundSubtractorMOG2()
elif args.algo == 'KNN':
    lBackSub = cv.createBackgroundSubtractorKNN()
    rBackSub = cv.createBackgroundSubtractorKNN()
elif args.algo == 'GSOC':
    lBackSub = cv.bgsegm.createBackgroundSubtractorGSOC()
    rBackSub = cv.bgsegm.createBackgroundSubtractorGSOC()

clipsL = int(args.clipsL)
clipsR = int(args.clipsR)
if (clipsL == 1):
    lCapName = args.L + args.ext
    rCapName = args.R + args.ext
else:
    lCapName = args.L + '1' + args.ext
    rCapName = args.R + '1' + args.ext

curClipL = 1
curClipR = 1
learningRateL = 0.0002
learningRateR = 0.0002

lCap = cv.VideoCapture(cv.samples.findFileOrKeep(lCapName))
rCap = cv.VideoCapture(cv.samples.findFileOrKeep(rCapName))

writer = cv.VideoWriter('out.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))

lCap.set(1, 27000)
rCap.set(1, 27000)

if not lCap.isOpened():
    print('Unable to open: ' + args.L)
    exit(0)
if not rCap.isOpened():
    print('Unable to open: ' + args.R)
    exit(0)
while True:
    ret, lFrameOrig = lCap.read()
    ret, rFrameOrig = rCap.read()
    if lFrameOrig is None:
        print('Trying to switch left clip')
        if curClipL < clipsL:
            curClipL += 1
            lCapName = args.L + str(curClipL) + args.ext
            print("Reading: ", lCapName)
            lCap = cv.VideoCapture(cv.samples.findFileOrKeep(lCapName))
            if not lCap.isOpened():
                print('Unable to open: ' + args.L)
                exit(0)
            ret, lFrameOrig = lCap.read()
            if lFrameOrig is None:
                break
        else:
            print("No more clips on left")
            break
    if rFrameOrig is None:
        print('Trying to switch right clip')
        if curClipR < clipsR:
            curClipR += 1
            rCapName = args.R + str(curClipR) + args.ext
            print("Reading: ", rCapName)
            rCap = cv.VideoCapture(cv.samples.findFileOrKeep(rCapName))
            if not rCap.isOpened():
                print('Unable to open: ' + args.R)
                exit(0)
            ret, rFrameOrig = rCap.read()
            if rFrameOrig is None:
                break
        else:
            print("No more clips on right")
            break
    
    
    lFrame = cv.resize(lFrameOrig, (320, 180), interpolation=cv.INTER_LINEAR)
    rFrame = cv.resize(rFrameOrig, (320, 180), interpolation=cv.INTER_LINEAR)
    lMask = lBackSub.apply(lFrame, learningRate=learningRateL)
    rMask = rBackSub.apply(rFrame, learningRate=learningRateR)

    lBG = lBackSub.getBackgroundImage()
    rBG = rBackSub.getBackgroundImage()

    if lMask.mean() < 7 and rMask.mean() < 7:
        frame = np.zeros(lFrameOrig.shape)
    elif lMask.mean() > rMask.mean():
        frame = lFrameOrig
        writer.write(frame)
    else:
        frame = rFrameOrig
        writer.write(frame)
    
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask Left', lMask)
    cv.imshow('FG Mask Right', rMask)
    cv.imshow('Left', lFrame)
    cv.imshow('Right', rFrame)
    cv.imshow('Left BG', lBG)
    cv.imshow('Right BG', rBG)
    
    keyboard = cv.waitKey(1)
    if keyboard == 'q' or keyboard == 27:
        break
    elif keyboard == 32:
        lframeNum = lCap.get(cv.CAP_PROP_POS_FRAMES)
        rframeNum = rCap.get(cv.CAP_PROP_POS_FRAMES)
        lCap.set(1, lframeNum + 30)
        rCap.set(1, rframeNum + 30)
    elif keyboard == 108:
        learningRateL = 1
    elif keyboard == 114:
        learningRateR = 1
