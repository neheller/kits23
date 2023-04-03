"""Checks the submission folder against the test image folder to make sure all
cases are present and the shapes and affine matrices match."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import nibabel as nib
from tqdm import tqdm


def check_submission_folder(submission_pth: Path, image_pth: Path):
    """Check the submission folder against the image folder to make sure all
    cases are present and the shapes and affine matrices match.

    Parameters:
        submission_pth (Path): Path to the submission folder
        image_pth (Path): Path to the image folder

    Returns:
        None
    """
    # Keep track of issues
    num_issues = 0
    value_warning = False

    # Pad console output with newline
    print()

    # Ensure that the image folder is not empty
    image_pths = sorted(list(image_pth.glob("*.nii.gz")))
    if len(image_pths) == 0:
        print("No images found in the image folder. Please check path.\n")
        return

    # Iterate over nifti files in the image folder
    for img_file_pth in tqdm(image_pths):
        # Get expected name of submission file
        sub_file_pth = submission_pth / img_file_pth.name

        # Check that the submission file exists
        if not sub_file_pth.exists():
            num_issues += 1
            print("Missing submission file:", str(img_file_pth.name))
            continue

        # Load the image and submission files
        img_nib: nib.Nifti1Image = nib.load(str(img_file_pth))
        sub_nib: nib.Nifti1Image = nib.load(str(sub_file_pth))

        # Check that the shapes match
        if img_nib.shape != sub_nib.shape:
            num_issues += 1
            print(
                f"Shape mismatch for {img_file_pth.name}:",
                str(img_file_pth.name), "\n",
                "Expected", img_nib.shape, "but got", sub_nib.shape
            )
            continue

        # Check that the affine matrices match
        affine_total_diff = np.abs((img_nib.affine - sub_nib.affine)).sum()
        if affine_total_diff > 1e-3:
            num_issues += 1
            print(f"Affine mismatch for {img_file_pth.name}")
            continue

        # Check that the data range is meaningful
        sub_np = np.asanyarray(
            sub_nib.dataobj
        ).clip(-1, 4).round().astype(np.int8)
        if sub_np.min() < 0 or sub_np.max() > 3:
            value_warning = True
            print(str(img_file_pth.name), "has values outside of [0, 3]")

    # Pad for messages
    print()

    # Warn if necessary
    if value_warning:
        print(
            "WARNING: At least one case has values outside of [0, 3]\n"
            "Values outside of [0, 3] will be treated as background\n"
        )

    # Print summary
    if num_issues != 0:
        print("Found", num_issues, "errors\n")
        return

    # Print success message
    print("Ready for submission :)\n")


def parse_arguments():
    """Parse command line arguments and ensure directories exist"""
    # Define and retrieve arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--submission-folder", type=str, required=True,
        help="Path to the folder with submission files"
    )
    parser.add_argument(
        "--images-folder", type=str, required=True,
        help="Path to the folder with test data (extracted from download link)"
    )
    args = parser.parse_args()

    # Check that the provided paths exist
    sub_pth, img_pth = Path(args.submission_folder), Path(args.images_folder)
    try:
        sub_pth = sub_pth.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Submission folder {args.submission_folder} does not exist"
        )
    try:
        img_pth = img_pth.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Test image folder {args.images_folder} does not exist"
        )

    return sub_pth, img_pth


def main():
    # Parse and resolve argument paths
    submission_pth, images_pth = parse_arguments()

    # Perform the check
    check_submission_folder(submission_pth, images_pth)


if __name__ == "__main__":
    main()
    