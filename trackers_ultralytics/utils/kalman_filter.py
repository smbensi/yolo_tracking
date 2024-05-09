# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/utils/kalman_filter.py

import numpy as np
import scipy.linalg

class KalmanFilterXYAH:
    """
    For bytetrack. A simple Kalamn filter for tracking bounding boxes in image space.
    
    The 8-dimensional state space (x, y, a, h, vx, vy, va, vh) contains the bounding box center position (x, y), aspect ratio a, height h and their respective velocities
    
    Object motion follows a constant velocity model. The bounding box location (x, y, a, h) is taken as direct observation of the state space (linear observation model)
    """
    
    def __init__(self) -> None:
        """Initialize Kalman filter model matrices with motion and observation uncertainty weights."""
        ndim, dt = 4, 1.0
        
        # Create Kalman filter model matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        # Motion and observation uncertainty are chosen relative to the current state estimate. These weights control the amount of uncertainty in the model
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
    
    def initiate(self, measurement:np.ndarray) -> tuple: