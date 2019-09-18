import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import time
import math

class CameraTranslationDetect(object):
    '''
    Class for calculating translational shift betwen two frames
    '''
    version = '0.1'
    
    def __init__(self):
        'initializer method'
        print('Hello, world.')
        
    def detect_phase_shift(self, prev_frame, curr_frame):
        'returns detected sub-pixel phase-shift between two frames'
        prev_frame = np.float32(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))    # convert to required type
        curr_frame = np.float32(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY))    # convert to required type
         #calculate phase-correlation between current and previous frame
        shift = cv2.phaseCorrelate(prev_frame, curr_frame)
        return shift

      
vs = VideoStream(src=0).start()    # initialize the video stream
time.sleep(2.0)    # allow camera to warm up
fps = FPS().start()    # initialize the FPS counter

n=0    # incrementer

threshold = 2    # detection sensitivity value

center = 200-50, 200   #define point for logging camera-state to screen

obj = CameraTranslationDetect()    # instantiate CameraTranslationDetect object and pass in reference frame

#mainloop
while True:
    frame = vs.read()    # read frame from video stream
    frame = imutils.resize(frame, width=400)    # resize output window

    if n == 0:    # check if firt iteration
        initial = frame.copy()    # store first frame from stream
        prev = frame.copy()
        n=n+1
     
    (shift_x, shift_y), sf = obj.detect_phase_shift(prev, frame)    # pass subsequent frame into class method, returns translational shift
    
    if shift_x >= threshold or shift_x <= -threshold or shift_y >= threshold or shift_y <= -threshold:    # check detected shift against threshold
        ts = time.time()    #get current time for logging detected motion
        readable = time.ctime(ts)    # convert timestamp to readable format
        
        print("camera movement detected @ " +    #print timestamp and x, y translation when motion detected
              str(readable) + ' x: ' + 
              str(shift_x) + ' y: ' + 
              str(shift_y))     

        cv2.putText(    #update screen
            frame,    #numpy array on which text is written
                "Motion Detected", #text
                center, #position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX, #font family
                0.8, #font size
                (0, 0, 255, 0), #font color
                2) #font stroke
    else:
        cv2.putText(frame, "Position Stable", center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0, 0), 2) 

    cv2.imshow("Frame", frame)    #display output
    
    key = cv2.waitKey(75) & 0xFF
    if key == ord("q"):    # if the `q` key was pressed, break from the loop
        break
 
    fps.update()    # update the FPS counter
    
    #reset shift
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
