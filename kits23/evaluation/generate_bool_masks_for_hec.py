from typing import Union, Tuple

import numpy as np


def construct_HEC_from_segmentation(segmentation: np.ndarray, label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    """
    Takes a segmentation as input (integer map with values indicating what class a voxel belongs to) and returns a
    boolean array based on where the selected label/HEC is. If label is a tuple, all pixels belonging to any of the
    listed classes will be set to True in the results. The rest remains False.
    """
    if not isinstance(label, (tuple, list)):
        return segmentation == label
    else:
        if len(label) == 1:
            return segmentation == label[0]
        else:
            mask = np.zeros(segmentation.shape, dtype=bool)
            for l in label:
                mask[segmentation == l] = True
            return mask

