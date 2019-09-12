import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import time
import math

n=0

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

while True:
    frame = vs.read()
    curr = frame.copy()
    frame = imutils.resize(frame, width=400)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if n == 0:
        prev = frame.copy()
        x, y = frame.shape
        n=n+1
        
    prev64 = np.float32(prev)  
    frame64 = np.float32(frame)    
    
    shift = cv2.phaseCorrelate(prev64, frame64)
    radius = int(math.sqrt(shift[0][0]*shift[0][0]+shift[0][1]*shift[0][1]))

    #define point for drawing to screen
    center = 200-50, 200             
   
    #change threshold in this if statement to increase or decrease detection sensitivity
    if shift[0][0] >= 2 or shift[0][0] <= -2 or shift[0][1] >= 2 or shift[0][1] <= -2 and radius > 1.5:
    #if radius > 1.5:
        ts = time.time()
        readable = time.ctime(ts)
        print("camera movement detected @ " + str(readable))
        cv2.putText(
            curr, #numpy array on which text is written
                "Camera Motion Detected", #text
                center, #position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX, #font family
                1, #font size
                (0, 0, 255, 0), #font color
                2) #font stroke

    else:
        cv2.putText(
    curr, #numpy array on which text is written
        "Camera Stable", #text
        center, #position at which writing has to start
        cv2.FONT_HERSHEY_SIMPLEX, #font family
        1, #font size
        (0, 255, 0, 0), #font color
        2) #font stroke

    # show the output frame
    cv2.imshow("Frame", curr)
    key = cv2.waitKey(75) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
 
    # update the FPS counter
    fps.update()
    #reset shift
    
    radius = 0
    shift = None
    prev = frame.copy()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()