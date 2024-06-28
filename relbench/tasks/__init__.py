from collections import defaultdict
from functools import lru_cache

import pooch

from ..data import BaseTask
from ..datasets import get_dataset
from . import amazon, avito, event, f1, hm, stack, trial

task_registry = defaultdict(dict)

DOWNLOAD_REGISTRY = pooch.create(
    path=pooch.os_cache("relbench"),
    base_url="https://relbench.stanford.edu/staging_data/",  # TODO: change
    registry={},
)


def register_task(
    dataset_name: str,
    task_name: str,
    cls,
    *args,
    **kwargs,
):
    task_registry[dataset_name][task_name] = (cls, args, kwargs)


def get_task_names(dataset_name: str):
    return list(task_registry[dataset_name].keys())


def download_task(dataset_name: str, task_name: str) -> None:
    try:
        DOWNLOAD_REGISTRY.fetch(
            f"{dataset_name}/tasks/{task_name}.zip",
            processor=pooch.Unzip(extract_dir=task_name),
            progressbar=True,
        )
    except ValueError:
        print("failed to download, will attempt to make task from scratch")


@lru_cache(maxsize=None)
def get_task(dataset_name: str, task_name: str, download=True) -> BaseTask:
    if download:
        download_task(dataset_name, task_name)
    dataset = get_dataset(dataset_name)
    cls, args, kwargs = task_registry[dataset_name][task_name]
    task = cls(dataset, *args, **kwargs)
    return task


register_task("rel-amazon", "user-churn", amazon.UserChurnTask)
register_task("rel-amazon", "user-ltv", amazon.UserLTVTask)
register_task("rel-amazon", "item-churn", amazon.ItemChurnTask)
register_task("rel-amazon", "item-ltv", amazon.ItemLTVTask)
register_task("rel-amazon", "user-item-purchase", amazon.UserItemPurchaseTask)
register_task("rel-amazon", "user-item-rate", amazon.UserItemRateTask)
register_task("rel-amazon", "user-item-review", amazon.UserItemReviewTask)

register_task("rel-avito", "ads-clicks", avito.AdsClicksTask)
register_task("rel-avito", "user-visits", avito.UserVisitsTask)
register_task("rel-avito", "user-clicks", avito.UserClicksTask)
register_task("rel-avito", "user-ad-visit", avito.UserAdVisitTask)

register_task("rel-event", "user-attendance", event.UserAttendanceTask)
register_task("rel-event", "user-repeat", event.UserRepeatTask)
register_task("rel-event", "user-ignore", event.UserIgnoreTask)

register_task("rel-f1", "driver-position", f1.DriverPositionTask)
register_task("rel-f1", "driver-dnf", f1.DriverDNFTask)
register_task("rel-f1", "driver-top3", f1.DriverTop3Task)
register_task("rel-f1", "driver-constructor-result", f1.DriverConstructorResultTask)

register_task("rel-hm", "user-item-purchase", hm.UserItemPurchaseTask)
register_task("rel-hm", "user-churn", hm.UserChurnTask)
register_task("rel-hm", "item-sales", hm.ItemSalesTask)

register_task("rel-stack", "user-engagement", stack.UserEngagementTask)
register_task("rel-stack", "post-votes", stack.PostVotesTask)
register_task("rel-stack", "user-badge", stack.UserBadgeTask)
register_task("rel-stack", "user-post-comment", stack.UserPostCommentTask)
register_task("rel-stack", "post-post-related", stack.PostPostRelatedTask)

register_task("rel-trial", "study-outcome", trial.StudyOutcomeTask)
register_task("rel-trial", "study-adverse", trial.StudyAdverseTask)
register_task("rel-trial", "study-withdrawal", trial.StudyWithdrawalTask)
register_task("rel-trial", "site-success", trial.SiteSuccessTask)
register_task("rel-trial", "condition-sponsor-run", trial.ConditionSponsorRunTask)
register_task("rel-trial", "site-sponsor-run", trial.SiteSponsorRunTask)
