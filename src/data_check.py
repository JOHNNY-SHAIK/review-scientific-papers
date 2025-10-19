"""
Simple dataset validation script.

Usage:
    python src/data_check.py data/mldataset_sample.json
or
    python src/data_check.py data/mldataset.json
"""

import json
import sys
from collections import Counter

EXPECTED_FIELDS = {"paper_id", "citation_context", "label"}
VALID_LABELS = {"Positive", "Neutral", "Negative"}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def validate_record(rec, idx):
    # Each record should be a dict with the expected fields and types
    if not isinstance(rec, dict):
        return False, f"Record {idx} is not a JSON object"
    missing = EXPECTED_FIELDS - rec.keys()
    if missing:
        return False, f"Record {idx} missing fields: {missing}"
    # basic type checks
    if not isinstance(rec["paper_id"], str) or not rec["paper_id"]:
        return False, f"Record {idx} paper_id must be non-empty string"
    if not isinstance(rec["citation_context"], str) or not rec["citation_context"].strip():
        return False, f"Record {idx} citation_context must be non-empty string"
    if rec["label"] not in VALID_LABELS:
        return False, f"Record {idx} label '{rec['label']}' not in {VALID_LABELS}"
    return True, ""


def main(path):
    print(f"Loading dataset from: {path}")
    data = load_json(path)
    if not isinstance(data, list):
        print("ERROR: dataset root should be a JSON array of records.")
        sys.exit(1)
    print(f"Total records: {len(data)}")

    errors = []
    label_counts = Counter()
    for i, rec in enumerate(data, 1):
        ok, msg = validate_record(rec, i)
        if not ok:
            errors.append(msg)
        else:
            label_counts[rec["label"]] += 1

    if errors:
        print("\nValidation errors found:")
        for e in errors[:20]:
            print(" -", e)
        print(f"\nTotal errors: {len(errors)}")
        sys.exit(2)

    print("All records passed basic validation.")
    print("Label distribution:")
    for lbl in sorted(VALID_LABELS):
        print(f"  {lbl}: {label_counts.get(lbl, 0)}")

    # Print 2 sample records for manual inspection
    print("\nSample records (first 2):")
    for rec in data[:2]:
        print("----")
        print("paper_id:", rec["paper_id"])
        print("label:", rec["label"])
        print("citation_context:", rec["citation_context"][:200])
    print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/data_check.py path/to/mldataset.json")
        sys.exit(1)
    main(sys.argv[1])
