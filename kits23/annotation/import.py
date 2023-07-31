import argparse
import shutil
from pathlib import Path
import json

import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt

from kits23 import TRAINING_CASE_NUMBERS, TESTING_CASE_NUMBERS
from kits23.annotation.postprocessing import delineation_to_seg, load_json, \
    write_json
from kits23.configuration.labels import KITS_LABEL_NAMES, \
    LABEL_AGGREGATION_ORDER
from kits23.configuration.paths import LEGACY_SRC_DIR, SRC_DIR, TRAINING_DIR, \
    TESTING_DIR, CACHE_FILE, KITS21_PATH


def get_case_dir(case):
    src_dir_msg = (
        "SRC_DIR was none, this is most likely due to KITS21_SERVER_DATA not "
        "being in your environment variables. This functionality is intended "
        "to be used only by the KiTS organizers."
    )
    assert LEGACY_SRC_DIR is not None, src_dir_msg
    assert SRC_DIR is not None, src_dir_msg
    # TODO remove hardcoding -- test both to find it
    page = int(case // 50)
    src_dir = LEGACY_SRC_DIR
    if case >= 300:
        src_dir = SRC_DIR

    tst = "training_data"
    if case not in TRAINING_CASE_NUMBERS:
        tst = "testing_data"
    return (
        src_dir / tst / f"cases_{page:05d}" / f"case_{case:05d}"
    ).resolve(strict=True)


def get_all_case_dirs():
    # TODO set this number dynamically
    return [
        get_case_dir(i) for i in TRAINING_CASE_NUMBERS + TESTING_CASE_NUMBERS
    ]


def get_region_dir(case_dir, region):
    return (case_dir / region).resolve(strict=True)


def get_all_region_dirs(case_dir):
    return [r for r in case_dir.glob("*")]


def get_instance_dir(region_dir, instance):
    return (region_dir / "{:02d}".format(instance)).resolve(strict=True)


def get_all_instance_dirs(region_dir):
    return [i for i in region_dir.glob("*")]


def get_existing_instances(region_dir):
    case_id = region_dir.parent.name
    base_dir = Path(__file__).resolve().parent.parent / "data"
    if int(case_id.split("_")[-1]) >= 300:
        base_dir = TESTING_DIR
    seg_dir = base_dir / case_id / "instances"
    return [x for x in seg_dir.glob("*{}*".format(region_dir.name))]


def get_delineation(instance_dir, delineation):
    return (instance_dir / f"delineation{delineation}").resolve(strict=True)


def get_all_delineations(instance_dir):
    return [d for d in instance_dir.glob("delineation*")]


def get_most_recent_save(parent_dir):
    # Get latest file and list of remainder
    try:
        srt_files = sorted([s for s in parent_dir.glob("*")])
        latest = srt_files[-1]
    except Exception as e:
        print()
        print("Error finding most recent save in", str(parent_dir))
        raise(e)

    return latest


def update_raw(delineation_path, case_id, in_test_set):
    # Get parent directory (create if necessary)
    destination_parent = TRAINING_DIR / case_id
    if in_test_set:
        destination_parent = TESTING_DIR / case_id
    if not destination_parent.exists():
        destination_parent.mkdir()
    destination_parent = destination_parent / "raw"
    if not destination_parent.exists():
        destination_parent.mkdir()
    else:
        shutil.rmtree(str(destination_parent))
        destination_parent.mkdir()

    custom_hilums = None
    if (destination_parent / "meta.json").exists():
        with (destination_parent / "meta.json").open() as f:
            old_meta = json.loads(f.read())
            if "custom_hilums" in old_meta:
                custom_hilums = old_meta["custom_hilums"]

    # Get source directory
    src = delineation_path.parent.parent.parent.parent

    # Copy all annotation files to destination
    shutil.copytree(str(src), str(destination_parent), dirs_exist_ok=True)

    if custom_hilums is not None:
        with (destination_parent / "meta.json").open() as f:
            new_meta = json.loads(f.read())
        with (destination_parent / "meta.json").open('w') as f:
            new_meta["custom_hilums"] = custom_hilums
            f.write(json.dumps(new_meta, indent=2))


def get_localization(delineation_path):
    return get_most_recent_save(
        delineation_path.parent.parent / "localization"
    )


def get_artery_localization(delineation_path):
    pth = (
        delineation_path.parent.parent.parent.parent / "artery" / "00" /
        "localization"
    )
    if not pth.exists():
        return None
    return get_most_recent_save(pth)


def get_image_path(case_id, in_test_set):
    if in_test_set:
        return (TESTING_DIR / case_id / "imaging.nii.gz").resolve(strict=True)
    else:
        return (TRAINING_DIR / case_id / "imaging.nii.gz").resolve(strict=True)


def save_segmentation(
    case_id, region_type, delineation_path, n1img, in_test_set
):
    # Create name of destination file
    annotation_num = int(delineation_path.parent.name[-1])
    instance_num = int(delineation_path.parent.parent.name)
    filename = (
        f"{region_type}_instance-{instance_num+1}_"
        f"annotation-{annotation_num}.nii.gz"
    )

    # Get parent directory (create if necessary)
    destination_parent = TRAINING_DIR / case_id
    if in_test_set:
        destination_parent = TESTING_DIR / case_id
    if not destination_parent.exists():
        destination_parent.mkdir()
    destination_parent = destination_parent / "instances"
    if not destination_parent.exists():
        destination_parent.mkdir()
    destination = destination_parent / filename

    # Save file
    n1img = nib.Nifti1Image(n1img.get_fdata().astype(np.uint8), n1img.affine)
    nib.save(n1img, str(destination))


def run_import(delineation_path):
    # Useful values
    region_type = delineation_path.parent.parent.parent.name
    case_id = delineation_path.parent.parent.parent.parent.name
    in_test_set = int(case_id[-5:]) in TESTING_CASE_NUMBERS

    # Copy updated raw data
    update_raw(delineation_path, case_id, in_test_set)

    # Kidneys require hilum information from the localization
    localization = None
    if region_type == "kidney":
        localization = get_localization(delineation_path)

    # Path to underlying CT scan stored as .nii.gz
    image_path = get_image_path(case_id, in_test_set)

    meta_path = image_path.parent / "raw" / "meta.json"
    meta = load_json(meta_path)

    # Compute and save segmentation based on delineation
    seg_nib = delineation_to_seg(
        region_type, image_path, delineation_path, meta,
        case_id, localization
    )
    save_segmentation(
        case_id, region_type, delineation_path, seg_nib, in_test_set
    )


def maybe_supplement(seg_np, seg_pth):
    # Check whether there is a supplemental file
    suppl_pth = seg_pth.parent / (".suppl." + seg_pth.name)
    if not suppl_pth.exists():
        return seg_np

    # Indicate that we are supplementing
    print("supplementing", str(seg_pth))

    # Load supplemental file
    suppl_nib = nib.load(str(suppl_pth))
    suppl_np = np.asanyarray(suppl_nib.dataobj)

    # Apply necessary additions
    seg_np[suppl_np == 2] = 1

    # Apply interior fill
    for slice_id in range(seg_np.shape[0]):
        if np.sum(seg_np[slice_id] == 1) == 0:
            continue
        slice_copy = seg_np[slice_id].copy()
        mask = np.zeros(
            (slice_copy.shape[0]+2, slice_copy.shape[1]+2), np.uint8
        )
        cv2.floodFill(slice_copy, mask, (0, 0), 1)
        seg_np[slice_id][slice_copy == 0] = 1

    return seg_np

def aggregate(parent, region, idnum, agg, affine, agtype="maj"):

    seg_files = [x for x in parent.glob("{}*.nii.gz".format(region))]
    instances = [int(x.stem.split("_")[1].split("-")[1]) for x in seg_files]
    unq_insts = sorted(list(set(instances)))

    reg_agg = None
    for inst in unq_insts:
        inst_agg = None
        n_anns = 0
        for tins, tfnm in zip(instances, seg_files):
            if tins != inst:
                continue
            seg_nib = nib.load(str(tfnm))
            n_anns += 1
            if inst_agg is None:
                inst_agg = np.asanyarray(seg_nib.dataobj)
                # If there is a supplemental file for this case, use it
                inst_agg = maybe_supplement(inst_agg, tfnm)
                affine = seg_nib.affine
            else:
                inst_agg = inst_agg + np.asanyarray(seg_nib.dataobj)

        if agtype == "maj":
            inst = np.greater(inst_agg, n_anns/2).astype(inst_agg.dtype)
        elif agtype == "or":
            inst = np.greater(inst_agg, 0).astype(inst_agg.dtype)
        elif agtype == "and":
            inst = np.equal(inst_agg, n_anns).astype(inst_agg.dtype)

        if reg_agg is None:
            reg_agg = np.copy(inst)
        else:
            reg_agg = np.logical_or(reg_agg, inst).astype(reg_agg.dtype)

    # If no info here, just return what we started with
    if reg_agg is None:
        return agg, affine

    if agg is None:
        agg = idnum*reg_agg
    else:
        agg = np.where(
            np.logical_not(np.equal(reg_agg, 0)), idnum*reg_agg, agg
        )

    return agg, affine


def purge_file(file_pth: Path):
    if input(f"Delete {str(file_pth)}? [y/n]: ") == "y":
        file_pth.unlink()


def aggregate_case(case_id, cache):
    base_dir = TRAINING_DIR
    if int(case_id.split("_")[-1]) in TESTING_CASE_NUMBERS:
        base_dir = TESTING_DIR

    segs =  base_dir / case_id / "instances"

    # Delete instances that no-longer exist (if any)
    for seg in segs.glob("*.nii.gz"):
        # Ignore supplemental files
        if seg.stem[0] == ".":
            continue
        purge = True
        rtype = seg.stem.split("_")[0]
        instnum = int(seg.stem.split("_")[1].split("-")[1]) - 1
        cache_key_prefix = f"{case_id}/{rtype}/{instnum:02d}"
        for key in cache:
            if key[:len(cache_key_prefix)] == cache_key_prefix:
                purge = False
        if purge:
            purge_file(seg)

    affine = None
    agg = None
    for label_id in LABEL_AGGREGATION_ORDER:
        agg, affine = aggregate(
            segs, KITS_LABEL_NAMES[label_id], label_id, agg, affine,
            agtype="maj"
        )
    if agg is not None:
        nib.save(
            nib.Nifti1Image(agg.astype(np.uint8), affine),
            str(base_dir / case_id / "segmentation.nii.gz")
        )


def cleanup(case_dir):
    base_dir = Path(__file__).resolve().parent.parent / "data"
    if int(case_dir.name.split("_")[-1]) >= 300:
        base_dir = TESTING_DIR
    case_dir = base_dir / case_dir.name / "raw"
    region_dirs = get_all_region_dirs(case_dir)
    for region_dir in region_dirs:
        instance_dirs = get_all_instance_dirs(region_dir)
        for instance_dir in instance_dirs:
            sessions = [x for x in instance_dir.glob("*")]
            for sess in sessions:
                srt_files = sorted([s for s in sess.glob("*")])
                for f in srt_files[:-1]:
                    f.unlink()


def main(args):
    cache = {}
    if CACHE_FILE.exists():
        cache = load_json(CACHE_FILE)
    cli = True
    if args.case is not None:
        case_dirs = [get_case_dir(args.case)]
    else:
        cli = False
        case_dirs = get_all_case_dirs()

    for case_dir in case_dirs:
        print(case_dir.name)
        if int(case_dir.name[-5:]) < 300:
            if args.import_from_kits21:
                dst_dir = TRAINING_DIR / case_dir.name
                if int(case_dir.name[-5:]) in TESTING_CASE_NUMBERS:
                    dst_dir = TESTING_DIR
                src_dir = KITS21_PATH / "kits21" / "data" / case_dir.name
                shutil.copytree(
                    str(src_dir),
                    dst_dir, dirs_exist_ok=True
                )
                shutil.move(
                    str(dst_dir / "aggregated_MAJ_seg.nii.gz"),
                    str(dst_dir / "segmentation.nii.gz")
                )
                shutil.move(
                    str(dst_dir / "segmentations"),
                    str(dst_dir / "instances")
                )
                for prev_agg in dst_dir.glob("aggregated_*"):
                    prev_agg.unlink()
            continue
        elif args.import_from_kits21:
            continue
        reaggregate = args.reaggregate
        if cli and args.region is not None:
            region_dirs = [get_region_dir(case_dir, args.region)]
        else:
            cli = False
            region_dirs = get_all_region_dirs(case_dir)

        for region_dir in region_dirs:
            # Skip regions no longer being used
            if region_dir.name in ["artery", "vein", "ureter"]:
                continue
            if cli and args.instance is not None:
                instance_dirs = [
                    get_instance_dir(region_dir, args.instance - 1)
                ]
            else:
                cli = False
                instance_dirs = get_all_instance_dirs(region_dir)

            for instance_dir in instance_dirs:
                if cli and args.delineation is not None:
                    delineations = [
                        get_delineation(instance_dir, args.delineation)
                    ]
                else:
                    delineations = get_all_delineations(instance_dir)

                for delineation in delineations:
                    dln_file = get_most_recent_save(delineation)
                    cache_key = str(
                        delineation.relative_to(
                            delineation.parent.parent.parent.parent
                        )
                    )
                    if (
                        args.regenerate or cache_key not in cache or
                        cache[cache_key] != dln_file.name
                    ):
                        try:
                            run_import(dln_file)
                            cache[cache_key] = dln_file.name
                            write_json(CACHE_FILE, cache)
                            reaggregate = True
                        except Exception as e:
                            print(f"Error importing {str(dln_file)}")
                            raise(e)
                    else:
                        print("Skipping", cache_key)

            # Delete any instances that were generated before but don't exist anymore
            generated_instances = get_existing_instances(region_dir)
            for gi in generated_instances:
                if (
                    int(gi.stem.split("instance-")[1][0]) not in [
                        int(x.name)+1 for x in instance_dirs
                    ]
                ):
                    print("Deleting legacy file:", str(gi.name))
                    gi.unlink()
                    reaggregate = True

        if reaggregate:
            aggregate_case(case_dir.name, cache)
        
        # Clean up all unused raw files
        cleanup(case_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--case",
        help="The index of the case to import",
        type=int
    )
    parser.add_argument(
        "-r", "--region",
        help="The type of region to import",
        type=str
    )
    parser.add_argument(
        "-i", "--instance",
        help="The index of the instance of that region to import",
        type=int
    )
    parser.add_argument(
        "-d", "--delineation",
        help="Index of the delineation of instance to import (1, 2, or 3)",
        type=int
    )
    parser.add_argument(
        "--regenerate",
        help="Regenerate segmentations regardless of cached values",
        action="store_true"
    )
    parser.add_argument(
        "--reaggregate",
        help="Reaggregate segmentations regardless of whether it was changed",
        action="store_true"
    )
    parser.add_argument(
        "--import-from-kits21",
        help="Import files from KiTS21 dataset",
        action="store_true"
    )
    if __name__ == "__main__":
        cl_args = parser.parse_args()
        main(cl_args)
