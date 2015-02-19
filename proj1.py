import numpy as np
import cv2
import cv
import struct
import cvk2

def computeDifferences(frame, average):
    differences =  average.astype(float) - frame.astype(float)
    return np.sqrt(differences*differences).astype('uint8')

def createCapture(filename):
    capture = cv2.VideoCapture(filename)
    if capture:
        print 'Opened file', filename

    # Bail if error.
    if not capture:
        print 'Error opening video capture!'
        sys.exit(1)

    return capture

def createWriter(frame):
    fourcc, ext = (struct.unpack('i', 'MP42')[0], 'avi')
    fname = 'captured.'+ext
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

def recordParams(differences):
    win = cv2.namedWindow("Window")
    cv.CreateTrackbar("Threshold", "Window", 0, 255, nothing) 
    cv.CreateTrackbar("Erode Size", "Window", 0, 10, nothing) 
    cv.CreateTrackbar("Dilate Size", "Window", 0, 10, nothing)
    cv.CreateTrackbar("Open Size", "Window", 0, 10, nothing)
    # cv.CreateTrackbar("Close Size", "Window", 0, 10, nothing) 

    while True:
        thresholdValue = cv2.getTrackbarPos("Threshold", "Window")
        erodeSize = cv2.getTrackbarPos("Erode Size", "Window")
        dilateSize = cv2.getTrackbarPos("Dilate Size", "Window")
        openSize = cv2.getTrackbarPos("Open Size", "Window")
        # closeSize = cv2.getTrackbarPos("Close Size", "Window")

        mask = applyOperators(differences, thresholdValue, erodeSize, \
            dilateSize, openSize)

        cv2.imshow("Window", mask) 
        ch = cv2.waitKey(5)
        if ch == 27:
            break

    return thresholdValue, erodeSize, dilateSize, openSize

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

    # if closeSize != 0: 
    #     closeElt = cv2.getStructuringElement(cv2.MORPH_RECT,(closeSize,closeSize))
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closeElt)

    return mask

def nothing(*arg):
    pass

def labelAndWaitForKey(image):
    cv2.imshow('Image', image)
    while cv2.waitKey(15) < 0: pass


def createAverageImage(filename):
    capture = createCapture(filename)

    # Fetch the first frame and bail if none.
    ok, frame = capture.read()
    if not ok or frame is None:
        print 'No frames in video'
        sys.exit(1)

    h, w, d = frame.shape
    
    composite = np.zeros((h, w)) #image composite of all frames
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
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            composite += gray
            numFrames += 1

    composite = (composite / numFrames).astype('uint8')
    return composite

def thresholdImage(filename, average):
    capture = createCapture(filename)

    # Fetch the first frame and bail if none.
    ok, frame = capture.read()
    if not ok or frame is None:
        print 'No frames in video'
        sys.exit(1)

    writer = createWriter(frame)
    differences = computeDifferences(frame, average)
    thresholdValue, erodeSize, dilateSize, openSize = recordParams(differences)

    # Loop until movie is ended or user hits ESC:
    while 1:

        # Get the frame.
        ok, frame = capture.read(frame)

        # Bail if none.
        if not ok or frame is None:
            break
        else: 
            differences = computeDifferences(frame, average)
            mask = applyOperators(differences, thresholdValue, \
                erodeSize, dilateSize, openSize)

            mask_copy = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

            display = np.zeros((mask_copy.shape[0], mask_copy.shape[1], 3),
                      dtype='uint8')
            contours = cv2.findContours(mask_copy, cv2.RETR_CCOMP,
                            cv2.CHAIN_APPROX_SIMPLE)

            ccolors = cvk2.getccolors()
            white = (255,255,255)

            # For each contour in the image
            for j in range(len(contours[0])):

                # Draw the contour as a colored region on the display image.
                cv2.drawContours( display, contours[0], j, ccolors[j % len(ccolors)], -1 )

                # Compute some statistics about this contour.
                info = cvk2.getcontourinfo(contours[0][j])

                # Mean location and basis vectors can be useful.
                mu = info['mean']
                b1 = info['b1']
                b2 = info['b2']

                # Annotate the display image with mean and basis vectors.
                cv2.circle( display, cvk2.a2ti(mu), 3, white, 1, cv2.CV_AA )
                cv2.line( display, cvk2.a2ti(mu), cvk2.a2ti(mu+2*b1), white, 1, cv2.CV_AA )
                cv2.line( display, cvk2.a2ti(mu), cvk2.a2ti(mu+2*b2), white, 1, cv2.CV_AA )
     
        # Write to writer.
        writer.write(display)

        # Throw it up on the screen.
        cv2.imshow('Video', display)

        # Delay for 5ms and get a key
        k = cv2.waitKey(5)

        # Check for ESC hit:
        if k % 0x100 == 27:
            break

def main():
    #composite = createAverageImage("flies1.avi")
    composite = cv2.imread("average.png")
    #cv2.imwrite("average.png", composite)
    #labelAndWaitForKey(composite)
    thresholdImage("flies1.avi", composite)

if __name__ == "__main__":
    main()