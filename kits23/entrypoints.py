"""Functions called by command-line entrypoints"""
from pathlib import Path
from argparse import ArgumentParser

from kits23.download import download_dataset


def download_data_entrypoint():
    download_dataset()

# Evaluation entrypoint was moved to evaluation/entry_point.py
