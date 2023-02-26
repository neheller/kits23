"""The official package of the 2023 Kidney Tumor Segmentation Challenge"""

from kits23._version import __VERSION__


TRAINING_CASE_NUMBERS = list(range(300)) + list(range(400, 595))
TESTING_CASE_NUMBERS = list(range(595, 713))

EXCLUDE_LIST = [
    443, 465, 501, 514, 528, 589, 600, 607, 616, 622, 624, 630, 645, 665
]
