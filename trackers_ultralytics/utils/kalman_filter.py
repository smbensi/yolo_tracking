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
        """
        create track from unassociated measurement

        Args:
            measurement (np.ndarray): bbox coordinates (x, y, a, h), with center position (x,y), aspect ratio a and height h

        Returns:
            tuple([ndarray, ndarray]): returns the mean vector (8 dimensional) and covariance matrix (8x8 dimensional) of the new track. Unobserved velocities are initialized to 0 mean
        """
        
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance
    
    def predict(self, mean:np.ndarray, covariance:np.ndarray) -> tuple:
        """
        Run Kalman filter prediction step

        Args:
            mean (np.ndarray): The 8 dim mean vector of the object state at the previous time step
            covariance (np.ndarray): The 8x8 dim covariance matrix of the object state at the previous time step

        Returns:
            tuple[ndarray, ndarray]: Returns the mean vector and covariance matrix of the predicted state. Unobserved velocities are initialized at 0 mean
        """
        
        std_pos = [
            2 * self._std_weight_position * mean[3],
            2 * self._std_weight_position * mean[3],
            1e-2,
            2 * self._std_weight_position * mean[3],
        ]
        std_vel = [
            10 * self._std_weight_velocity * mean[3],
            10 * self._std_weight_velocity * mean[3],
            1e-5,
            10 * self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        
        return mean, covariance