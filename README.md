# KiTS23

The official repository of the 2023 Kidney Tumor Segmentation Challenge

[Challenge Homepage](https://kits23.kits-challenge.org/)

## Timeline

- **April 1**: Training dataset release
- **July 7**: Registration and preliminary manuscript deadline
- **July 21 - July 28**: Prediction submissions accepted
- **July 31**: Results announced

## News

Check here for the latest news about the KiTS23 dataset and starter code!

## Usage

This repository is meant to serve two purposes:

1. To help you **download the dataset**
2. To allow you to benchmark your model using the **official implementation of the metrics**

We recommend starting by installing this `kits23` package using pip. Once you've done this, you'll be able to use the command-line download entrypoint and call the metrics functions from your own codebase.

### Installation

This should be as simple as cloning and installing with pip.

```bash
git clone https://github.com/neheller/kits23
cd kits23
pip3 install -e .
```

We're running Python 3.10.6 on Ubuntu and we suggest that you do too. If you're running a different version of Python 3 and you discover a problem, please [submit an issue](https://github.com/neheller/kits23/issues/new) and we'll try to help. Python 2 or earlier is not supported. If you're running Windows or MacOS, we will do our best to help but we have limited ability to support these environments.

### Data Download

Once the `kits23` package is installed, you should be able to run the following command from the terminal.

```bash
kits23_download_data
```

This will place the data in the `dataset/` folder.

### Using the Metrics

The simplest way for you to compute the official metrics on your own predictions is to import and call the `compute_metrics` function from within your own python code.

```python
from kits23.metrics import compute_metrics

my_prediction = ...  # Your model goes here (convert to numpy)
metrics_dict = compute_metrics(my_prediction, case_id="case_00XXX")
```

Where `my_prediction` is a `numpy` array of shape `(n_slices, height, width)`.

Alternatively, you may supply your own reference label. This might be useful if your approach involves cropping our regions of interest from the scan.

```python
compute_metrics(my_prediction, label=reference_label, case_id="case_00XXX")
```

The reason you must supply the `case_id` is because the surface dice metric relies on a tolerance value which requires knowledge of the voxel spacing. On the off chance that you'd like to use this function to compute metrics on a generic prediction and label pair, you can supply the spacing information yourself as below.

```python
compute_metrics(my_prediction, label=reference_label, spacing=(z_spacing_mm, y_spacing_mm, x_spacing_mm))
```

The return value from `compute_metrics()` is a dictionary object with the following form:

```json
{
    "by_HEC": {
        "1-kidney_or_2-tumor_or_3-cyst": {
            "surface_dice": ...,
            "volumetric_dice": ...
        },
        "2-tumor_or_3-cyst": {
            "surface_dice": ...,
            "volumetric_dice": ...
        },
        "2-tumor": {
            "surface_dice": ...,
            "volumetric_dice": ...
        }
    },
    "mean_surface_dice": ...,
    "mean_volumetric_dice": ...
}
```

Another way that you can compute metrics is by using the `kits23_compute_metrics` command line entrypoint. Start by creating a directory with files as follows

```text
my_predictions/
├── prediction_00123.nii.gz
├── prediction_00234.nii.gz
└── ...
```

Then, run the following from the terminal

```bash
kits23_compute_metrics <path/to/my_predictions> <path/to/output.json>
```

which will store the above dictionary data in a JSON file.

## License and Attribution

The code in this repository is under an MIT License. The data that this code downloads is under a CC BY-NC-SA (Attribution-NonCommercial-ShareAlike) license. If you would like to use this data for commercial purposes, please contact Nicholas Heller at helle246@umn.edu. Please note, we do not consider training a model for the *sole purpose of participation in this competition* to be commercial use. Therefore, industrial teams are strongly encouraged to participate. If you are an academic researcher interested in using this dataset in your work, you needn't ask for permission.

If this project is useful to your research, please cite our most recent KiTS challenge paper in Medical Image Analysis (\[[html](https://www.sciencedirect.com/science/article/abs/pii/S1361841520301857)\] \[[pdf](https://arxiv.org/pdf/1912.01054.pdf)\])

```bibtex
@article{heller2020state,
  title={The state of the art in kidney and kidney tumor segmentation in contrast-enhanced CT imaging: Results of the KiTS19 Challenge},
  author={Heller, Nicholas and Isensee, Fabian and Maier-Hein, Klaus H and Hou, Xiaoshuai and Xie, Chunmei and Li, Fengyi and Nan, Yang and Mu, Guangrui and Lin, Zhiyong and Han, Miofei and others},
  journal={Medical Image Analysis},
  pages={101821},
  year={2020},
  publisher={Elsevier}
}
```
