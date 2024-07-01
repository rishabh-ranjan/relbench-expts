import json
import hashlib
from pathlib import Path


def db_hashes(
    in_dir="~/.cache/relbench_upload/",
    out_file="/lfs/local/0/ranjanr/relbench/relbench/datasets/hashes.json",
):
    hashes = {}
    in_dir = Path(in_dir).expanduser()
    for dataset_dir in in_dir.iterdir():
        db_zip = f"{dataset_dir}/db.zip"
        with open(db_zip, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()
        hashes[f"{dataset_dir.name}/db.zip"] = sha256
    with open(out_file, "w") as f:
        json.dump(hashes, f, indent=2)


def task_hashes(
    in_dir="~/.cache/relbench_upload/",
    out_file="/lfs/local/0/ranjanr/relbench/relbench/tasks/hashes.json",
):
    hashes = {}
    in_dir = Path(in_dir).expanduser()
    dataset_dirs = list(in_dir.iterdir())
    for dataset_dir in dataset_dirs:
        for task_zip in Path(f"{dataset_dir}/tasks").iterdir():
            with open(task_zip, "rb") as f:
                sha256 = hashlib.sha256(f.read()).hexdigest()
            hashes[f"{dataset_dir.name}/tasks/{task_zip.name}"] = sha256
    with open(out_file, "w") as f:
        json.dump(hashes, f, indent=2)
