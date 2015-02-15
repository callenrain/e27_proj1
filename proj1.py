import numpy as np
import cv2

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
    composite = np.zeros((h, w, 3))
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
    composite += frame
    numFrames += 1
    # Loop until movie is ended or user hits ESC:
    while 1:

        # Get the frame.
        ok, frame = capture.read(frame)

        # Bail if none.
        if not ok or frame is None:
            break
        else:
        	composite += frame
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
    print(composite)
    print numFrames
    composite = composite / numFrames
    composite = composite.astype('uint8')
    return composite


composite = createAverageImage("flies1.avi")
labelAndWaitForKey(composite)
#print composite


