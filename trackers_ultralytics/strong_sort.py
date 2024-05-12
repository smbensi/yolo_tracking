import numpy as np

from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.trackers.strongsort.sort.tracker import Tracker
from boxmot.utils.matching import NearestNeighborDistanceMetric
from boxmot.motion.cmc import get_cmc_method

class StrongSORT:
    def __init__(self, model_weights, device, fp16,
                 per_class=False, max_dist=0.2, max_iou_dist=0.7, max_age=30,
                 n_init=1, nn_budget=100, mc_lambda=0.995, ema_alpha=0.9) -> None:
        
        self.per_class = per_class
        rab = ReidAutoBackend(
            weights=model_weights, device=device, half=fp16
        )
        
        self.model = rab.get_backend()
        self.tracker = Tracker(
            metric=NearestNeighborDistanceMetric("cosine", max_dist, nn_budget),
            max_iou_dist=max_iou_dist,
            max_age=max_age,
            n_init=n_init,
            mc_lambda=mc_lambda,
            ema_alpha=ema_alpha
        )
        self.cmc = get_cmc_method('ecc')()