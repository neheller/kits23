import numpy as np


def dice(prediction: np.ndarray, reference: np.ndarray):
    """
    Both predicion and reference have to be bool (!) numpy arrays. True is interpreted as foreground, False is background
    """
    intersection = np.count_nonzero(prediction & reference)
    numel_pred = np.count_nonzero(prediction)
    numel_ref = np.count_nonzero(reference)
    if numel_ref == 0 and numel_pred == 0:
        return np.nan
    else:
        return 2 * intersection / (numel_ref + numel_pred)
