import numpy as np
import scipy
from scipy.spatial.distance import cdist

from ultralytics.utils.metrics import batch_probiou, bbox_ioa

try:
    import lap # for linear assignment
    
    assert lap.__version__ # verify package is not directory
except (ImportError, AssertionError, AttributeError):
    from ultralytics.utils.checks import check_requirements
    
    check_requirements("lapx>=0.5.2")
    import lap
    
def linear_assignment(cost_matrix: np.ndarray, thresh:float, use_lap:bool = True) -> tuple:
    """
    Perform linear assignment using scipy or lap.lapjv

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments
        thresh (float): Threshold for considering an assignment valid
        use_lap (bool, optional): Whether to use lap.lapjv. Defaults to True.

    Returns:
        tuple with:
            - matched indices
            - unmatched indices from 'a'
            - unmatched indices from 'b'
    """
    pass

def iou_distance(atracks: list, btracks: list) -> np.ndarray:
    """
    Compute cost based on Intersection over Union (IoU) between tracks

    Args:
        atracks (list[Strack] | list[np.ndarray]): List of tracks 'a' or bounding boxes
        btracks (list[Strack] | list[np.ndarray]): List of tracks 'b' or bounding boxes

    Returns:
        np.ndarray: Cost matrix computed on IoU
    """
    
    if atracks and isinstance(atracks[0], np.ndarray) or btracks and isinstance(btracks[0], np.ndarray):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.xywha if track.angle is not None else track.xyxy for track in atracks]
        btlbrs = [track.xywha if track.angle is not None else track.xyxy for track in btracks]
        
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if len(atlbrs) and len(btlbrs):
        if len(atlbrs[0]) == 5 and len(btlbrs[0]) == 5:
            ious = batch_probiou(
                np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32)
            ).numpy()
        else:
            ious = bbox_ioa(
                np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32),
                iou=True
            )
    return 1 - ious # cost matrix


def embedding_distance(tracks:list, detections:list, metric:str = 'cosine') -> np.ndarray:
    """
    Compute distance between tracks and detections based on embeddings.

    Args:
        tracks (list): _description_
        detections (list): _description_
        metric (str, optional): _description_. Defaults to 'cosine'.

    Returns:
        np.ndarray: Cost matrix computed based on embeddings
    """
    
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    track_features