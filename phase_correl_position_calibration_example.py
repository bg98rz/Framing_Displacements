import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import time
import math

from numpy.fft import fft2, ifft2, fftshift

class CameraTranslationDetect(object):
    '''
    Class for calculating translational shift betwen two frames
    '''
    version = '2019.0.1'
    
    def __init__(self, initial_frame)
    'initializer method'
        self.initial_frame = np.float32(cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY))    # convert to required type
        
    def detect_phase_shift(self, curr_frame):
        'returns detected sub-pixel phase-shift between two frames'
        curr_frame = np.float32(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY))    # convert to required type
         #calculate phase-correlation between current and previous frame
        shift = cv2.phaseCorrelate(self.initial_frame, curr_frame)
        return shift

    def fft_phase_shift(self, im0, im1):
        'stand-alone implementation - returns x, y translation between two frames'
        shape = im0.shape
        f0 = fft2(im0)
        f1 = fft2(im1)
        ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
        t0, t1, c = np.unravel_index(np.argmax(ir), shape)
        if t0 > shape[0] // 2:
            t0 -= shape[0]
        if t1 > shape[1] // 2:
            t1 -= shape[1]
        return [t0, t1]
    
    
    # implementation
    
vs = VideoStream(src=0).start()    # initialize the video stream
time.sleep(2.0)    # allow camera to warm up
fps = FPS().start()    # initialize the FPS counter

n=0    # incrementer

threshold = 30    # detection sensitivity value

center = 200-50, 200   #define point for logging camera-state to screen

#mainloop
while True:
    frame = vs.read()    # read frame from video stream
    frame = imutils.resize(frame, width=400)    # resize output window

    if n == 0:    # check if firt iteration
        initial = frame.copy()    # store first frame from stream
        obj = CameraTranslationDetect(initial)    # instantiate CameraTranslationDetect object and pass in reference frame
        prev = frame.copy()
        n=n+1
     
    (shift_x, shift_y), sf = obj.detect_phase_shift(frame)    # pass subsequent frame into class method, returns translational shift
    
    if shift_x >= threshold or shift_x <= -threshold or shift_y >= threshold or shift_y <= -threshold:    # check detected shift against threshold
        ts = time.time()    #get current time for logging detected motion
        readable = time.ctime(ts)    # convert timestamp to readable format
        
        print("camera movement detected @ " +    #print timestamp and x, y translation when motion detected
              str(readable) + ' x: ' + 
              str(shift_x) + ' y: ' + 
              str(shift_y))     

        cv2.putText(    #update screen
            frame,    #numpy array on which text is written
                "Camera out of position", #text
                center, #position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX, #font family
                1, #font size
                (0, 0, 255, 0), #font color
                2) #font stroke
    else:
        cv2.putText(frame, "Position Stable", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0, 0), 2) 

    cv2.imshow("Frame", frame)    #display output
    key = cv2.waitKey(75) & 0xFF
 

    if key == ord("q"):    # if the `q` key was pressed, break from the loop
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
