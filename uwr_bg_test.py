from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np

INPUT_PATH = 'input/'

parser = argparse.ArgumentParser(description='This program takes two input video streams of a UWR game and switches between the angles automatically')
parser.add_argument('--L', type=str, help='Filename of left clips without extension', default='L')
parser.add_argument('--R', type=str, help='Filename of right clips without extension', default='R')
parser.add_argument('--ext', type=str, help='File extension for video files', default='.mp4')
parser.add_argument('--clipsL', type=str, help='Number of clips left video is divided into', default='1')
parser.add_argument('--clipsR', type=str, help='Number of clips right video is divided into', default='1')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2, GSOC).', default='MOG2')
parser.add_argument('--delay', type=str, help='Frame number to start videos on.', default='1')
parser.add_argument('-pauses', action='store_true', help='If this argument is given, the program will attempt to cut game pauses out')
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

lCap = cv.VideoCapture(cv.samples.findFileOrKeep(INPUT_PATH + lCapName))
rCap = cv.VideoCapture(cv.samples.findFileOrKeep(INPUT_PATH + rCapName))

writer = cv.VideoWriter('out.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))

lCap.set(1, int(args.delay))
rCap.set(1, int(args.delay))

if not lCap.isOpened():
    print('Unable to open: ' + args.L)
    exit(0)
if not rCap.isOpened():
    print('Unable to open: ' + args.R)
    exit(0)
def is_game_paused(lMask, rMask):
    lowerHalfMeanR = rMask[rMask.shape[0]//2 :, :].mean()
    lowerHalfMeanL = lMask[lMask.shape[0]//2 :, :].mean()
    upperHalfMeanL = rMask[: rMask.shape[0]//2, :].mean()
    upperHalfMeanR = lMask[: lMask.shape[0]//2, :].mean()
    return lowerHalfMeanL < 1 and lowerHalfMeanR < 1 and upperHalfMeanL < 40 and upperHalfMeanR < 40

def score(mask):
    lowerHalfMean = mask[mask.shape[0]//2 :, :].mean()
    upperHalfMean = mask[: mask.shape[0]//2, :].mean()
    return 3*lowerHalfMean + upperHalfMean

while True:
    ret, lFrameOrig = lCap.read()
    ret, rFrameOrig = rCap.read()

    lFrameOrig = cv.UMat(lFrameOrig)
    rFrameOrig = cv.UMat(rFrameOrig)
    if lFrameOrig is None:
        print('Trying to switch left clip')
        if curClipL < clipsL:
            curClipL += 1
            lCapName = args.L + str(curClipL) + args.ext
            print("Reading: ", lCapName)
            lCap = cv.VideoCapture(cv.samples.findFileOrKeep(INPUT_PATH + lCapName))
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
            rCap = cv.VideoCapture(cv.samples.findFileOrKeep(INPUT_PATH + rCapName))
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

    if args.pauses and is_game_paused(lMask, rMask):
        frame = np.zeros(lFrameOrig.shape)
    elif score(cv.UMat.get(lMask)) > score(cv.UMat.get(rMask)):
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
        learningRateL = 0
    elif keyboard == 114:
        learningRateR = 0
