###########################################################################
#
# File:   proj1.py
# Author: Dylan Jeffers & Callen Rain
# Date:   January 29, 2015
#
# Written for ENGR 27 - Computer Vision - Homework 1
#
###########################################################################
#
# This program performs a temporal average of pixels in a movie, allows the user 
# to select parameters for thresholding, and tracks moving objects over time

import numpy as np
import cv2
import cv
import struct
import cvk2
import random
import os.path
import sys

# Given a frame and an average image, compute the distance between corresponding
# pixels
def computeDifferences(frame, average):
    differences =  average.astype(float) - frame.astype(float)
    return np.sqrt(differences*differences).astype('uint8')

# Creates a VideoCapture object from a given filename
def createCapture(filename):
    capture = cv2.VideoCapture(filename)
    if capture:
        print 'Opened file', filename

    # Bail if error.
    if not capture:
        print 'Error opening video capture!'
        sys.exit(1)

    return capture

# Creates a VideoWriter object from a gievn frame and a filename
def createWriter(frame, filename):
    fourcc, ext = (struct.unpack('i', 'MP42')[0], 'avi')
    fname = filename + '.' + ext
    fps = 30

    h, w, d = frame.shape

    # Now set up a VideoWriter to output video.
    writer = cv2.VideoWriter(fname, fourcc, fps, (w, h))

    if not writer:
        print 'Error opening writer'
        sys.exit(1)
    else:
        print 'Opened', fname, 'for output.'
        return writer

# Sets up trackbars to allow the user to input parameters
def recordParams(differences):
    win = cv2.namedWindow("Parameter Selection")

    # create trackbars
    cv.CreateTrackbar("Threshold", "Parameter Selection", 0, 255, nothing) 
    cv.CreateTrackbar("Erode Size", "Parameter Selection", 0, 10, nothing) 
    cv.CreateTrackbar("Dilate Size", "Parameter Selection", 0, 10, nothing)
    cv.CreateTrackbar("Open Size", "Parameter Selection", 0, 10, nothing)

    print "Hit ESC to save your choices for the parameters"

    # While user has not his 'ESC'
    while True:
        # Get the current values of the trackbars
        thresholdValue = cv2.getTrackbarPos("Threshold", "Parameter Selection")
        erodeSize = cv2.getTrackbarPos("Erode Size", "Parameter Selection")
        dilateSize = cv2.getTrackbarPos("Dilate Size", "Parameter Selection")
        openSize = cv2.getTrackbarPos("Open Size", "Parameter Selection")

        # Apply the current operators based on the threshold values
        mask = applyOperators(differences, thresholdValue, erodeSize, \
            dilateSize, openSize)

        # Show the modified image to the user
        cv2.imshow("Parameter Selection", mask) 
        ch = cv2.waitKey(5)
        if ch == 27:
            break

    cv2.destroyWindow("Parameter Selection")
    return thresholdValue, erodeSize, dilateSize, openSize

# This is nothing function to use for the trackbars
def nothing(*arg):
    pass

# Apply morphological operators to the image with the current parameters
def applyOperators(image, thresholdValue, erodeSize, dilateSize, openSize):
    mask = cv2.threshold(image, thresholdValue, 255, cv2.THRESH_BINARY)[1]

    if erodeSize != 0: 
        erodeElt = cv2.getStructuringElement(cv2.MORPH_RECT,(erodeSize,erodeSize))
        mask = cv2.erode(mask, erodeElt)

    if dilateSize != 0: 
        dilateElt = cv2.getStructuringElement(cv2.MORPH_RECT,(dilateSize,dilateSize))
        mask = cv2.dilate(mask, dilateElt)

    if openSize != 0: 
        openElt = cv2.getStructuringElement(cv2.MORPH_RECT,(openSize,openSize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, openElt)

    return mask


def labelAndWaitForKey(image):
    cv2.imshow('Image', image)
    while cv2.waitKey(15) < 0: pass

def findClosest(pt, prev_pts):
    min = 1000000
    closest = 0
    for i, val in enumerate(prev_pts):
        dist = np.sqrt((val[0]-pt[0])**2 + (val[1]-pt[1])**2)
        if dist < min:
            min = dist
            closest = (val, i, min)
    return closest[0], closest[1], closest[2]

# Creates an image consisting of the temporal average of each pixel
def createAverageImage(filename):
    print "Creating average image..."

    capture = createCapture(filename)

    # Fetch the first frame and bail if none.
    ok, frame = capture.read()
    if not ok or frame is None:
        print 'No frames in video'
        sys.exit(1)

    h, w, d = frame.shape
    
    composite = np.zeros((h, w, 3)) # image composite of all frames
    numFrames = 0 #counter for number of frames in video
    fps = 30

    # accumulator image
    gray = np.empty((h, w), 'uint8')

    # Loop until movie is ended or user hits ESC:
    while 1:
        # Get the frame.
        ok, frame = capture.read(frame)

        # Bail if none.
        if not ok or frame is None:
            break
        else:
            # accumulate frame values and the number seen so far
            composite += frame
            numFrames += 1

    # Obtain the final image using 
    composite = (composite / numFrames).astype('uint8')
    return composite

def processVideo(filename, avg):
    capture = createCapture(filename)

    # Fetch the first frame and bail if none.
    ok, frame = capture.read()
    if not ok or frame is None:
        print 'No frames in video'
        sys.exit(1)

    # Create a writer for the output video
    writer = createWriter(frame, filename.split('.')[0] + "-captured")

    # Compute the differences between the frame and the average
    differences = computeDifferences(frame, avg)

    # Let the user pick parameters for thresholding and morphological operators
    thresholdValue, erodeSize, dilateSize, openSize = recordParams(differences)

    # Path image is instantiated here to accumulate the contour paths over 
    # each frame.
    display_paths = np.zeros((avg.shape[0], avg.shape[1], 3), dtype = 'uint8')

    # These lists keep track of colors and points between different frames
    prev_contours = []
    prev_colors = []

    # Loop until movie is ended or user hits ESC:
    while 1:

        # Get the frame.
        ok, frame = capture.read(frame)

        # Bail if none.
        if not ok or frame is None:
            break
        else:
            # Grab the differences between the frame and the average 
            differences = computeDifferences(frame, avg)

            # Get the current mask based on the 
            mask = applyOperators(differences, thresholdValue, \
                erodeSize, dilateSize, openSize)

            # Copy the mask because the contours function is destructive
            mask_copy = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

            # Initialize an image to draw the contours on each frame
            display = np.zeros((mask_copy.shape[0], mask_copy.shape[1], 3),
                      dtype='uint8')

            # Find the contours in the image
            contours = cv2.findContours(mask_copy, cv2.RETR_CCOMP,
                            cv2.CHAIN_APPROX_SIMPLE)

            # Initialize lists for colors and points that we match
            mean_list = []
            color_list = []

            ccolors = cvk2.getccolors()
            white = (255,255,255)

            # For each contour in the image
            for j in range(len(contours[0])):

                # Compute some statistics about this contour.
                info = cvk2.getcontourinfo(contours[0][j])

                # Get the mean location for the contour
                mu = info['mean']

                # Assuming we have some previous contour points to refer back to
                if len(prev_contours) > 0:
                    # Find the closest point in the previous frame
                    closest_point, index, dist = findClosest(cvk2.a2ti(mu), \
                        prev_contours)

                    # Don't draw giant lines everywhere (this is a hack)
                    if dist < 250:
                        color = prev_colors[index]
                        cv2.line(display_paths, cvk2.a2ti(mu), \
                            closest_point, color)
                    # Pick a random color instead
                    else:
                        color = random.choice(ccolors)
                # If this is the first iteration, pick a random color
                else: 
                    color = random.choice(ccolors)

                # Append matched colors to new lists
                mean_list.append(cvk2.a2ti(mu))
                color_list.append(color)
                
                # Draw the contour as a colored region on the display image.
                cv2.drawContours( display, contours[0], j, color, -1 )
                cv2.circle(display_paths, cvk2.a2ti(mu), 1, color, 1, cv2.CV_AA)
     
            # Assign to new lists so we have access next iteration
            prev_contours = mean_list
            prev_colors = color_list

        combined = np.hstack((frame, display + display_paths))

        # Write to writer.
        writer.write(combined)

        # Display on the screen side-by-side with original
        cv2.imshow('Video', combined)

        # Delay for 5ms and get a key
        k = cv2.waitKey(10)

        # Check for ESC hit:
        if k % 0x100 == 27:
            break

def main():
    # Make sure number of arguments is right
    if len(sys.argv) != 2:
        print "Usage: python proj1.py <video_filename>"
        sys.exit(1)

    filename = sys.argv[1]
    average_filename = filename.split('.')[0] + "-average.png"

    # Check to see if temporal average aleady exists
    if os.path.isfile(filename):
        print "Found temporal average file already created"
        average = cv2.imread(average_filename)
    # Else make the new file
    else:
        average = createAverageImage(filename)
        cv2.imwrite(average_filename, composite)

    processVideo(filename, average)

if __name__ == "__main__":
    main()