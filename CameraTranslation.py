import cv2
import numpy as np
import math

from numpy.fft import fft2, ifft2, fftshift

class CameraTranslation(object):
    '''
    Class for calculating translational shift betwen two frames
    '''
    version = '2019.0.1'
    
    def __init__(self, initial_frame)
    'initializer method'
        self.initial_frame = np.float32(cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY))    # convert to required type
        
    def detect_phase_shift(self, curr_frame):
        'opencv implementation - returns detected sub-pixel phase-shift between two frames'
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
