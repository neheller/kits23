"""Unit tests to ensure the data meets certain guidelines before uploading

This module is intended for the KiTS23 admins only, though others may find it
useful to ensure their download succeeded
"""
from pathlib import Path

import numpy as np
import nibabel as nib

from kits23 import TRAINING_CASE_NUMBERS


def test_segmentation_exists():
    """Ensure that the segmentation files exist for each case"""
    # Iterate through each training case
    for case_num in TRAINING_CASE_NUMBERS:
        # Get path to segmentation file
        case_id = f"case_{case_num:05d}"
        case_pth = Path(__file__).parent.parent / "dataset" / case_id
        seg_pth = case_pth / "segmentation.nii.gz"

        # Ensure that the segmentation file exists
        assert seg_pth.exists()


def test_image_exists():
    """Ensure that the image files exist for each case (after download)"""
    # Iterate through each training case
    for case_num in TRAINING_CASE_NUMBERS:
        # Get path to segmentation file
        case_id = f"case_{case_num:05d}"
        case_pth = Path(__file__).parent.parent / "dataset" / case_id
        img_pth = case_pth / "imaging.nii.gz"

        # Ensure that the segmentation file exists
        assert img_pth.exists()


def test_segmentation_dtype():
    """Ensure that the segmentation files are uint8"""
    # Iterate through each training case
    for case_num in TRAINING_CASE_NUMBERS:
        # Get path to segmentation file
        case_id = f"case_{case_num:05d}"
        case_pth = Path(__file__).parent.parent / "dataset" / case_id
        seg_pth = case_pth / "segmentation.nii.gz"

        # Load segmentation file
        seg = nib.load(str(seg_pth))

        # Ensure that the segmentation file is uint8
        assert np.asanyarray(seg.dataobj).dtype == np.uint8


def test_image_dtype():
    """Ensure that the image files are float32"""
    # Iterate through each training case
    for case_num in TRAINING_CASE_NUMBERS:
        # Get path to segmentation file
        case_id = f"case_{case_num:05d}"
        case_pth = Path(__file__).parent.parent / "dataset" / case_id
        img_pth = case_pth / "imaging.nii.gz"
        assert img_pth.exists()

        # Load segmentation file
        img = nib.load(str(img_pth))

        # Ensure that the segmentation file is uint8
        assert np.asanyarray(img.dataobj).dtype == np.float32


def test_segmentation_orientation():
    """Ensure all segmentations have the same orientation"""
    # Get path to segmentation file
    case_id = f"case_{TRAINING_CASE_NUMBERS[0]:05d}"
    case_pth = Path(__file__).parent.parent / "dataset" / case_id
    seg_pth = case_pth / "segmentation.nii.gz"

    # Load segmentation file
    seg = nib.load(str(seg_pth))

    # Get orientation
    orientation = nib.aff2axcodes(seg.affine)

    # Iterate through each training case
    for case_num in TRAINING_CASE_NUMBERS[1:]:
        # Get path to segmentation file
        case_id = f"case_{case_num:05d}"
        case_pth = Path(__file__).parent.parent / "dataset" / case_id
        seg_pth = case_pth / "segmentation.nii.gz"

        # Load segmentation file
        seg = nib.load(str(seg_pth))

        # Ensure that the segmentation file is uint8
        assert nib.aff2axcodes(seg.affine) == orientation


def test_segmentation_shape():
    """In cases where imaging exists, ensure that the segmentation shape
    matches"""
    # Iterate through each training case
    for case_num in TRAINING_CASE_NUMBERS:
        # Get path to segmentation file
        case_id = f"case_{case_num:05d}"
        case_pth = Path(__file__).parent.parent / "dataset" / case_id
        seg_pth = case_pth / "segmentation.nii.gz"

        # Load segmentation file
        seg = nib.load(str(seg_pth))

        # Get segmentation shape
        seg_shape = seg.shape

        # Get path to imaging file
        img_pth = case_pth / "imaging.nii.gz"

        # If imaging file exists, load it
        if img_pth.exists():
            img = nib.load(str(img_pth))

            # Ensure that the segmentation file is uint8
            assert img.shape == seg_shape


def test_orientation_match():
    """For each case, ensure that the affine matrices for the imaging and
    corresponding segmentation match"""
    # Iterate through each training case
    for case_num in TRAINING_CASE_NUMBERS:
        # Get path to segmentation file
        case_id = f"case_{case_num:05d}"
        case_pth = Path(__file__).parent.parent / "dataset" / case_id
        seg_pth = case_pth / "segmentation.nii.gz"

        # Load segmentation file
        seg = nib.load(str(seg_pth))

        # Get path to imaging file
        img_pth = case_pth / "imaging.nii.gz"

        # If imaging file exists, load it
        if img_pth.exists():
            img = nib.load(str(img_pth))

            # Ensure that the segmentation file is uint8
            assert np.allclose(img.affine, seg.affine)
