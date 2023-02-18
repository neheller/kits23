"""A script to download the KiTS23 dataset into this repository"""
from tqdm import tqdm
from pathlib import Path
import urllib.request
import shutil


DST_PTH = Path(__file__).resolve().parent.parent / "dataset"


def get_destination(case_id: str, create: bool = False):
    destination = DST_PTH / case_id / "imaging.nii.gz"
    if create:
        destination.parent.mkdir(exist_ok=True)
    return destination


def cleanup(tmp_pth: Path, e: Exception):
    if tmp_pth.exists():
        tmp_pth.unlink()
    raise(e)


def download_case(case_num: int):
    remote_name = f"master_{case_num:05d}.nii.gz"
    url = f"https://kits19.sfo2.digitaloceanspaces.com/{remote_name}"
    destination = get_destination(f"case_{case_num:05d}", True)
    tmp_pth = destination.parent / f".partial.{destination.name}"
    try:
        urllib.request.urlretrieve(url, str(tmp_pth))
        shutil.move(str(tmp_pth), str(destination))
    except Exception as e:
        cleanup(tmp_pth, e)


def download_dataset():
    # Make output directory if it doesn't exist already
    DST_PTH.mkdir(exist_ok=True)

    # Determine which cases still need to be downloaded
    left_to_download = []
    for case_num in range(300):
        case_id = f"case_{case_num:05d}"
        dst = get_destination(case_id)
        if not dst.exists():
            left_to_download = left_to_download + [case_num]

    # Show progressbar as cases are downloaded
    print(f"\nFound {len(left_to_download)} cases to download\n")
    for case_num in (pbar := tqdm(left_to_download)):
        pbar.set_description(f"Dowloading case_{case_num:05d}...")
        download_case(case_num)


if __name__ == "__main__":
    download_dataset()
