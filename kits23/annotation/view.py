import os
import sys
from pathlib import Path


def main():
    case_num = int(sys.argv[1])
    case_id = f"case_{case_num:05d}"
    src_pth = Path(__file__).resolve().parent.parent.parent / "dataset"
    img_pth = src_pth / case_id / "imaging.nii.gz"
    seg_pth = src_pth / case_id / "segmentation.nii.gz"
    os.system(f"itksnap -g {str(img_pth)} -s {str(seg_pth)}")


if __name__ == "__main__":
    main()
