import json
import pkgutil
from functools import lru_cache

import pooch

from ..data import Dataset
from . import amazon, avito, event, f1, fake, hm, stack, trial

dataset_registry = {}

hashes_file = pkgutil.get_data(__name__, "datasets/hashes.json")
with open(hashes_file, "r") as f:
    hashes = json.load(f)

DOWNLOAD_REGISTRY = pooch.create(
    path=pooch.os_cache("relbench"),
    base_url="https://relbench.stanford.edu/staging_data/",  # TODO: change
    registry=hashes,
)


def register_dataset(
    name: str,
    cls: Dataset,
    *args,
    **kwargs,
):
    cache_dir = f"{pooch.os_cache('relbench')}/{name}"
    kwargs = {"cache_dir": cache_dir, **kwargs}
    dataset_registry[name] = (cls, args, kwargs)


def get_dataset_names():
    return list(dataset_registry.keys())


def download_dataset(name: str) -> None:
    DOWNLOAD_REGISTRY.fetch(
        f"{name}/db.zip",
        processor=pooch.Unzip(extract_dir="db"),
        progressbar=True,
    )


@lru_cache(maxsize=None)
def get_dataset(name: str, download=False) -> Dataset:
    if download:
        download_dataset(name)
    cls, args, kwargs = dataset_registry[name]
    dataset = cls(*args, **kwargs)
    return dataset


register_dataset("rel-amazon", amazon.AmazonDataset)
register_dataset("rel-avito", avito.AvitoDataset)
register_dataset("rel-event", event.EventDataset)
register_dataset("rel-f1", f1.F1Dataset)
register_dataset("rel-hm", hm.HMDataset)
register_dataset("rel-stack", stack.StackDataset)
register_dataset("rel-trial", trial.TrialDataset)
