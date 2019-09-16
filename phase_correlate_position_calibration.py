import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import time
import math

#iterator
n=0

#threshold for detection sensitivity
threshold = 10


# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

#mainloop
while True:
    #read in frame from vs
    frame = vs.read()
    curr = frame.copy()
    frame = imutils.resize(frame, width=400)

    #convert to required format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #if initial frame, save copy
    if n == 0:
        initial = frame.copy()
        prev = frame.copy()
        x, y = frame.shape
        n=n+1
    
    #convert frames to requried format
    initial64 = np.float32(initial)
    prev64 = np.float32(prev)  
    frame64 = np.float32(frame)    
    
    #calculate phase-correlation between current and previous frame
    (shift_x, shift_y), sf = cv2.phaseCorrelate(initial64, frame64)
    radius = int(math.sqrt(shift_x*shift_x+shift_y*shift_y))

    #define point for logging camera-state to screen
    center = 200-50, 200             
   
    #change threshold value in this statement to increase or decrease detection sensitivity
    if shift_x >= threshold or shift_x <= -threshold or shift_y >= threshold or shift_y <= -threshold and radius > 1.5:
        #get current time for logging detected motion
        ts = time.time()
        readable = time.ctime(ts)
        
        #print timestamp when movement detected
        print("camera movement detected @ " + str(readable) + ' x: ' + str(shift_x) + ' y: ' + str(shift_y))
        
        #log camera state to screen
        cv2.putText(
            curr, #numpy array on which text is written
                "Camera out of position", #text
                center, #position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX, #font family
                1, #font size
                (0, 0, 255, 0), #font color
                2) #font stroke

    else:
        #log camera-state to screen
        cv2.putText(
    curr, #numpy array on which text is written
        "Position Stable", #text
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
    shift_x = None
    shift_y = None
    sf = None
    prev = frame.copy()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()