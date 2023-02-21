"""A script to open a given case in ITKSNAP in order to decide whether it needs
further refinement"""
import os
import json
from pathlib import Path

from kits23 import TRAINING_CASE_NUMBERS, TESTING_CASE_NUMBERS
from kits23.configuration.paths import TRAINING_DIR, TESTING_DIR


REVIEW_RESULTS_PTH = Path(__file__).resolve().parent / "review_results.json"
CACHE_PTH = Path(__file__).resolve().parent / "cache.json"


def needs_review(case_num, case_id, results, cache):
    # KiTS21 cases don't need additional reviewing
    if case_num < 300:
        return False

    # Cases without any segmentation file yet generated don't need reviewing
    src_pth = TRAINING_DIR
    if case_num not in TRAINING_CASE_NUMBERS:
        src_pth = TESTING_DIR
    if not (src_pth / case_id / "segmentation.nii.gz").exists():
        return False

    # If case has no results, it needs reviewing
    if case_id not in results:
        return True

    # If there is a cached file used to generate the newest mask that was not
    # considered in the last review, it needs reviewing
    for key, val in cache.items():
        if key[:len(case_id)] == case_id:
            if val not in results[case_id]["delineation_files"]:
                return True

    # Otherwise, no new reviewing is needed
    return False


def review_case(case, results, cache):
    case_id = case["case_id"]
    src_pth = TRAINING_DIR
    if not case["training"]:
        src_pth = TESTING_DIR

    # Initialize empty review result
    review_result = {
        "decision": None,
        "notes": None,
        "delineation_files": []
    }

    # Keep track of which delineation files are being reviewed here
    for key, val in cache.items():
        if key[:len(case_id)] == case_id:
            review_result["delineation_files"].append(val)

    # Open the scan
    img_pth = src_pth / case_id / "imaging.nii.gz"
    seg_pth = src_pth / case_id / "segmentation.nii.gz"
    os.system(f"itksnap -g {str(img_pth)} -s {str(seg_pth)}")

    # Solicit a decision
    decision = None
    while decision is None:
        raw_decision = input("[y/n]: ")
        if raw_decision.strip() == "y":
            decision = "y"
        elif raw_decision.strip() == "n":
            decision = "n"
    review_result["decision"] = decision

    # If decision is "no", store notes
    notes = None
    if decision == "n":
        notes = input("Notes: ")

    # Store result
    review_result["decision"] = decision
    review_result["notes"] = notes
    results[case_id] = review_result
    with REVIEW_RESULTS_PTH.open('w') as f:
        f.write(json.dumps(results, indent=2))


def main():
    # Load prior review results and cache
    results = {}
    if REVIEW_RESULTS_PTH.exists():
        with REVIEW_RESULTS_PTH.open() as f:
            results = json.loads(f.read())
    cache = {}
    if CACHE_PTH.exists():
        with CACHE_PTH.open() as f:
            cache = json.loads(f.read())

    # Iterate through case_ids and decide whether to review based on cache
    review_queue = []
    for case_num in TRAINING_CASE_NUMBERS + TESTING_CASE_NUMBERS:
        case_id = f"case_{case_num:05d}"
        if needs_review(case_num, case_id, results, cache):
            review_queue.append({
                "case_id": case_id,
                "training": case_num in TRAINING_CASE_NUMBERS
            })

    # Review each case that needs reviewing
    for review_ind, case in enumerate(review_queue):
        case_id = case["case_id"]
        print(
            f"Reviewing case {case_id} "
            f"({review_ind + 1} of {len(review_queue)})..."
        )
        review_case(case, results, cache)


if __name__ == "__main__":
    main()
