import numpy as np
import cv2
import struct
import cvk2


def labelAndWaitForKey(image):
    cv2.imshow('Image', image)

    while cv2.waitKey(15) < 0: pass

def createAverageImage(video):

    capture = cv2.VideoCapture(video)
    if capture:
        print 'Opened file', video

    # Bail if error.
    if not capture:
        print 'Error opening video capture!'
        sys.exit(1)

    # Fetch the first frame and bail if none.
    ok, frame = capture.read()
    if not ok or frame is None:
        print 'No frames in video'
        sys.exit(1)

    # Now set up a VideoWriter to output video.
    w = frame.shape[1]
    h = frame.shape[0]

    #image composite of all frames
    composite = np.zeros((h, w))
    #counter for number of frames in video
    numFrames = 0

    fps = 30
    """
    # One of these combinations should hopefully work on your platform:
    fourcc, ext = (struct.unpack('i', 'MP42')[0], 'avi')
    #fourcc, ext = (struct.unpack('i', 'DIVX')[0], 'avi')
    #fourcc, ext = (struct.unpack('i', 'U263')[0], 'mov')
    #fourcc, ext = (struct.unpack('i', 'PIM1')[0], 'mpg')

    #filename = 'captured.'+ext

    writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
    if not writer:
        print 'Error opening writer'
    else:
        print 'Opened', filename, 'for output.'
        writer.write(frame)
      """
    #for first frame
    gray = np.empty((h, w), 'uint8')
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    composite += gray
    numFrames += 1
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
        # # Write if we have a writer.
        # if writer:
        #     writer.write(frame)

        # # Throw it up on the screen.
        # cv2.imshow('Video', frame)

        # Delay for 5ms and get a key
        k = cv2.waitKey(5)

        # Check for ESC hit:
        if k % 0x100 == 27:
            break

    composite = composite / numFrames
    composite = composite.astype('uint8')
    return composite

def thresholdImage(filename, average):

    capture = cv2.VideoCapture(filename)
    if capture:
        print 'Opened file', filename

    # Bail if error.
    if not capture:
        print 'Error opening video capture!'
        sys.exit(1)

    # Fetch the first frame and bail if none.
    ok, frame = capture.read()
    if not ok or frame is None:
        print 'No frames in video'
        sys.exit(1)

    fourcc, ext = (struct.unpack('i', 'MP42')[0], 'avi')

    fname = 'captured.'+ext
    fps = 30
    # Now set up a VideoWriter to output video.
    w = frame.shape[1]
    h = frame.shape[0]
    writer = cv2.VideoWriter(fname, fourcc, fps, (w, h))
    if not writer:
        print 'Error opening writer'
    else:
        print 'Opened', fname, 'for output.'
        # writer.write(frame)

    # Loop until movie is ended or user hits ESC:
    while 1:

        # Get the frame.
        ok, frame = capture.read(frame)

        # Bail if none.
        if not ok or frame is None:
            break
        else: 
            average_float = average.astype(float)
            frame_float = frame.astype(float)
            differences_float =  np.absolute(average_float - frame_float)
            differences = differences_float.astype('uint8')
            print differences
            mask = cv2.threshold(differences, 40, 255, cv2.THRESH_BINARY)
     
        # # Write if we have a writer.
        if writer:
            writer.write(mask[1])

        # Throw it up on the screen.
        cv2.imshow('Video', mask[1])

        # Delay for 5ms and get a key
        k = cv2.waitKey(5)

        # Check for ESC hit:
        if k % 0x100 == 27:
            break

def connectedComponents(filename):

    capture = cv2.VideoCapture(filename)
    if capture:
        print 'Opened file', filename

    # Bail if error.
    if not capture:
        print 'Error opening video capture!'
        sys.exit(1)

    # Fetch the first frame and bail if none.
    ok, frame = capture.read()
    if not ok or frame is None:
        print 'No frames in video'
        sys.exit(1)

    fourcc, ext = (struct.unpack('i', 'MP42')[0], 'avi')

    fname = 'contours.'+ext
    fps = 30
    # Now set up a VideoWriter to output video.
    w = frame.shape[1]
    h = frame.shape[0]
    writer = cv2.VideoWriter(fname, fourcc, fps, (w, h))
    if not writer:
        print 'Error opening writer'
    else:
        print 'Opened', fname, 'for output.'

    # Loop until movie is ended or user hits ESC:
    while 1:

        # Get the frame.
        ok, frame = capture.read(frame)

        # Bail if none.
        if not ok or frame is None:
            break
        else: 
            frame_copy = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            display = np.zeros((frame.shape[0], frame.shape[1], 3),
                      dtype='uint8')
            contours = cv2.findContours(frame_copy, cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)

            ccolors = cvk2.getccolors()
            white = (255,255,255)

            # For each contour in the image
            print len(contours[0])
            for j in range(len(contours[0])):

                # Draw the contour as a colored region on the display image.
                cv2.drawContours( display, contours[0], j, ccolors[j % len(ccolors)], -1 )

                # # Compute some statistics about this contour.
                # info = cvk2.getcontourinfo(contours[0][j])

                # # Mean location and basis vectors can be useful.
                # mu = info['mean']
                # b1 = info['b1']
                # b2 = info['b2']

                # # Annotate the display image with mean and basis vectors.
                # cv2.circle( display, cvk2.a2ti(mu), 3, white, 1, cv2.CV_AA )
                # cv2.line( display, cvk2.a2ti(mu), cvk2.a2ti(mu+2*b1), white, 1, cv2.CV_AA )
                # cv2.line( display, cvk2.a2ti(mu), cvk2.a2ti(mu+2*b2), white, 1, cv2.CV_AA )
                 
        # # Write if we have a writer.
        if writer:
            writer.write(display)

        # Throw it up on the screen.
        cv2.imshow('Video', display)

        # Delay for 5ms and get a key
        k = cv2.waitKey(1000)

        # Check for ESC hit:
        if k % 0x100 == 27:
            break
#composite = createAverageImage("flies1.avi")
composite = cv2.imread("average.png")
#cv2.imwrite("average.png", composite)
#labelAndWaitForKey(composite)
#thresholdImage("flies1.avi", composite)
connectedComponents("captured.avi")

#print composite


