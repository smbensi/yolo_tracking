#https://github.com/smbensi/yolo_tracking/blob/master/boxmot/trackers/strongsort/sort/tracker.py#L13


from __future__ import absolute_import

import numpy as np


from boxmot.motion.cmc import get_cmc_method
from boxmot.utils.matching import chi2inv95

class Tracker:
    """
    This is the multi-target tracker
    
    Parameters:
        metric: nn_matching.NearestNeighborDistanceMetric
            A distance metric for measurement-to-track association
        max_age: int
            Maximum number of missed misses before a track is deleted
        n_init: int
            Number of consecutive detections before the track is confirmed. The track state is set to 'Deleted' if a miss occurs within the first 'n_init' frames
            
    Attributes:
        metric: nn_matching.NearestNeighborDistanceMetric
            A distance metric for measurement-to-track association
        max_age: int
            Maximum number of missed misses before a track is deleted
        n_init: int
            Number of frames that a track remains in initialization phase
        tracks : list[Track]
            The list of active tracks at the current time step
    """
    
    GATING_THRESHOLD = np.sqrt(chi2inv95[4])
    
    def __init__(self, metric, max_iou_dist=0.9, max_age=30, n_init=3, _lambda=0,
                 ema_alpha=0.9, mc_lambda=0.995) -> None:
        self.metric = metric
        self.max_iou_dist = max_iou_dist
        self.max_age = max_age
        self.n_init = n_init
        self._lambda = _lambda
        self.ema_alpha = ema_alpha
        self.mc_lambda = mc_lambda
        
        self.tracks = []
        self.next_id = 1
        self.cmc = get_cmc_method('ecc')()
        
    def predict(self):
        """
        propagate track state distributions one time step forward
        
        This function should be called once every time step, before 'update'
        """
        for track in self.tracks:
            track.predict()
    
    def increment_ages(self):
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()
            
    
    def update(self):
        """
        Perform measurement update and track management
        
        Paramaters:
            detections : List[deep_sort.detection.Detection]
                A list of detections at the current time step
        """
        
        