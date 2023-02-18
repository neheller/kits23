"""Functions called by command-line entrypoints"""
from pathlib import Path
from argparse import ArgumentParser

from kits23.download import download_dataset


def download_data_entrypoint():
    download_dataset()


def compute_metrics_entrypoint():
    # Define and parse command line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--predictions-folder", '-p',
        help="The folder containing the predicted segmentations",
        dest="predictions_folder",
        required=True
    )
    parser.add_argument(
        "--destination", '-d',
        help="The path to the file to write",
        required=True
    )
    args = parser.parse_args()

    # Ensure input files exist and output file can be written
    try:
        src_pth = Path(args.predictions_folder).resolve(strict=True)
    except Exception as e:
        print("\nAn error occurred while resolving the predictions folder\n")
        raise(e)
    try:
        dst_pth = Path(args.destination).resolve()
        dst_pth = dst_pth.parent.resolve(strict=True) / dst_pth.name
        with dst_pth.open('w') as f:
            f.write("{}")
    except Exception as e:
        print("\nAn error occurred while writing to the destination file\n")
        raise(e)

    # TODO
    print("\nThis functionality has not yet been implemented\n")
    raise NotImplementedError()
