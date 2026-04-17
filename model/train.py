"""
Incremental GANomaly training loop for ADFuzz.
"""

import json
import math
import os
import random
import shutil
import sys
import threading
from ctypes import CDLL, POINTER, c_int, c_size_t, c_void_p

from options import Options
from lib.checkTest import loadTestData
from lib.model import Ganomaly


mutex = threading.Lock()

SHM_SIZE = 2
SHM_ID = 123456789
SHM_DATA_ID = 12345671
SHM_DATA_SIZE = 4096
SHM_DATA_NUM = 1


def ensure_dir(path):
    """Create a directory tree if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def reset_dir(path):
    """Remove every file under a materialized dataset directory.

    The training pool itself is preserved elsewhere; these directories are only
    regenerated views that GANomaly consumes for the next incremental round.
    """
    ensure_dir(path)
    for entry in os.listdir(path):
        entry_path = os.path.join(path, entry)
        if os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
        else:
            os.remove(entry_path)


def list_regular_files(path):
    """Return sorted plain files so dataset materialization stays deterministic."""
    if not os.path.isdir(path):
        return []
    return [
        os.path.join(path, name)
        for name in sorted(os.listdir(path))
        if os.path.isfile(os.path.join(path, name))
    ]


def parse_sample_label(file_name):
    """Extract the path / state label encoded in the AFLNet sample filename."""
    return file_name.split(",")[-1]


def model_state_path(opt):
    return os.path.join(opt.outf, opt.name, "retrain_state.json")


def retrain_metrics_path(opt):
    return os.path.join(opt.dataroot, "retrain_metrics.json")


def load_state(opt):
    """Load the persistent data-pool state used across retraining rounds.

    The state file lets us keep a sliding recent window plus historical
    representatives without exploding the on-disk dataset every time we retrain.
    """
    path = model_state_path(opt)
    if not os.path.exists(path):
        return {
            "recent_samples": [],
            "history_samples": {},
            "history_seen": {},
            "training_distribution": {},
            "last_validation_precision": 1.0,
        }

    with open(path, "r", encoding="utf-8") as handle:
        state = json.load(handle)

    state.setdefault("recent_samples", [])
    state.setdefault("history_samples", {})
    state.setdefault("history_seen", {})
    state.setdefault("training_distribution", {})
    state.setdefault("last_validation_precision", 1.0)
    return state


def save_state(opt, state):
    """Persist the current pool metadata after each successful update round."""
    path = model_state_path(opt)
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


def prune_archive(opt, state):
    """Delete archived raw samples that are no longer referenced by the pool.

    This is the budget-enforcement step for disk growth: only recent-window
    samples and retained historical representatives are allowed to stay.
    """
    archive_dir = os.path.join(opt.dataroot, "pool", "archive")
    keep = {sample["path"] for sample in state["recent_samples"] if os.path.exists(sample["path"])}

    for sample_paths in state["history_samples"].values():
        for path in sample_paths:
            if os.path.exists(path):
                keep.add(path)

    for path in list_regular_files(archive_dir):
        if path not in keep:
            os.remove(path)


def read_retrain_metrics(opt):
    """Read optional AFL-side trigger metrics emitted for the next round."""
    path = retrain_metrics_path(opt)
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}


def ingest_collection_samples(opt, state):
    """Move freshly collected AFL samples into the bounded long-lived pool.

    We maintain two views at once:
    1) a recent sliding window that tracks the latest workload,
    2) a per-class representative set maintained with reservoir replacement.
    """
    collection_dir = os.path.join(opt.dataroot, "collection", "1")
    archive_dir = os.path.join(opt.dataroot, "pool", "archive")
    ensure_dir(archive_dir)

    for source_path in list_regular_files(collection_dir):
        base_name = os.path.basename(source_path)
        label = parse_sample_label(base_name)
        archived_name = f"{len(state['recent_samples']):08d}_{base_name}"
        archived_path = os.path.join(archive_dir, archived_name)

        shutil.copy2(source_path, archived_path)
        state["recent_samples"].append({"path": archived_path, "label": label})

        state["history_seen"][label] = state["history_seen"].get(label, 0) + 1
        history_bucket = state["history_samples"].setdefault(label, [])

        if len(history_bucket) < opt.history_per_class:
            history_bucket.append(archived_path)
        else:
            # Reservoir replacement keeps old high-frequency classes bounded
            # while still giving newer examples a fair chance to enter.
            seen = state["history_seen"][label]
            replace_idx = random.randint(0, seen - 1)
            if replace_idx < opt.history_per_class:
                history_bucket[replace_idx] = archived_path

        os.remove(source_path)

    max_recent = max(1, opt.recent_window_size)
    if len(state["recent_samples"]) > max_recent:
        state["recent_samples"] = state["recent_samples"][-max_recent:]


def sample_budget_trim(paths, max_items):
    """Cap the training set size for a single incremental retraining round."""
    if len(paths) <= max_items:
        return paths
    return paths[:max_items]


def normalize_distribution(counts):
    """Convert class / path counts to a normalized probability distribution."""
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in counts.items()}


def build_materialized_dataset(opt, state):
    """Materialize a bounded train/test view from the pooled archive.

    Recent and representative samples are merged here, then split into:
    - `train/1`: normal training data,
    - `test/1`: held-out frequent-class validation data,
    - `test/2`: low-frequency / abnormal validation data.
    """
    train_root = os.path.join(opt.dataroot, "train", "1")
    test_normal_root = os.path.join(opt.dataroot, "test", "1")
    test_abnormal_root = os.path.join(opt.dataroot, "test", "2")

    reset_dir(train_root)
    reset_dir(test_normal_root)
    reset_dir(test_abnormal_root)

    recent_by_label = {}
    for sample in state["recent_samples"]:
        recent_by_label.setdefault(sample["label"], []).append(sample["path"])

    label_counts = {label: len(paths) for label, paths in recent_by_label.items()}
    sorted_labels = sorted(label_counts, key=label_counts.get, reverse=True)

    if sorted_labels:
        # Frequent labels approximate the dominant traffic modes seen by AFLNet.
        # They are the classes where preserving precision matters the most for
        # both filtering safety and incremental warm-start quality.
        cutoff = max(1, len(sorted_labels) // 2)
        frequent_labels = set(sorted_labels[:cutoff])
    else:
        frequent_labels = set()

    train_samples = []
    test_normal_samples = []
    test_abnormal_samples = []

    for label, paths in recent_by_label.items():
        unique_paths = []
        seen = set()
        for path in paths + state["history_samples"].get(label, []):
            if os.path.exists(path) and path not in seen:
                seen.add(path)
                unique_paths.append(path)

        holdout = unique_paths[:opt.validation_per_class]
        remainder = unique_paths[opt.validation_per_class:]

        if label in frequent_labels:
            test_normal_samples.extend(holdout)
            train_samples.extend(remainder)
        else:
            test_abnormal_samples.extend(unique_paths)

    if not test_abnormal_samples and train_samples:
        test_abnormal_samples.append(train_samples.pop())
    if not test_normal_samples and train_samples:
        test_normal_samples.append(train_samples.pop(0))
    if not train_samples and test_normal_samples:
        train_samples.append(test_normal_samples[0])

    train_samples = sample_budget_trim(train_samples, max(1, opt.incremental_max_samples))

    for index, source_path in enumerate(train_samples):
        shutil.copy2(source_path, os.path.join(train_root, f"train_{index:06d}_{os.path.basename(source_path)}"))
    for index, source_path in enumerate(test_normal_samples):
        shutil.copy2(source_path, os.path.join(test_normal_root, f"normal_{index:06d}_{os.path.basename(source_path)}"))
    for index, source_path in enumerate(test_abnormal_samples):
        shutil.copy2(source_path, os.path.join(test_abnormal_root, f"abnormal_{index:06d}_{os.path.basename(source_path)}"))

    training_counts = {}
    for source_path in train_samples:
        label = parse_sample_label(os.path.basename(source_path).split("_", 1)[-1])
        training_counts[label] = training_counts.get(label, 0) + 1

    state["training_distribution"] = normalize_distribution(training_counts)


def train_model_incrementally(opt, state, addr, shmdata):
    """Run one bounded warm-start training round and then return to inference."""
    dataloader = loadTestData(opt)
    model = Ganomaly(opt, dataloader)
    result = model.train()
    state["last_validation_precision"] = result["precision"]
    model.validate_batch_no_shm(addr, mutex, shmdata)


def wait_for_retrain_signal(addr):
    """Block until AFLNet signals that a new collection phase has completed."""
    while True:
        if addr[SHM_DATA_NUM] == 3:
            return


def training_loop(addr, shmdata):
    """Main incremental-training loop driven by AFLNet shared-memory signals.

    Each iteration performs:
    1) absorb newly collected samples into the bounded pool,
    2) materialize a budget-limited dataset,
    3) warm-start incremental retraining,
    4) switch back to online validation / inference until the next signal.
    """
    opt = Options().parse()
    state = load_state(opt)

    while True:
        ingest_collection_samples(opt, state)
        build_materialized_dataset(opt, state)

        weight_dir = os.path.join(opt.outf, opt.name, "train", "weights")
        if os.path.isdir(weight_dir):
            opt.resume = weight_dir
        else:
            opt.resume = ''

        metrics = read_retrain_metrics(opt)
        if metrics:
            state["last_retrain_metrics"] = metrics

        train_model_incrementally(opt, state, addr, shmdata)
        prune_archive(opt, state)
        save_state(opt, state)
        wait_for_retrain_signal(addr)


if __name__ == '__main__':
    try:
        rt = CDLL('librt.so')
    except OSError:
        rt = CDLL('librt.so.1')

    shmget = rt.shmget
    shmget.argtypes = [c_int, c_size_t, c_int]
    shmget.restype = c_int

    shmat = rt.shmat
    shmat.argtypes = [c_int, POINTER(c_void_p), c_int]
    shmat.restype = POINTER(c_int)

    shmid = shmget(SHM_ID, SHM_SIZE, 0O644)
    if shmid < 0:
        sys.exit()

    addr = shmat(shmid, None, 0)

    shmdataid = []
    shmdata = []
    for i in range(SHM_DATA_NUM):
        shmdataid.append(shmget(SHM_DATA_ID + i, SHM_DATA_SIZE, 0O644))
        if shmdataid[i] < 0:
            sys.exit()
        shmdata.append(shmat(shmdataid[i], None, 0))

    training_loop(addr, shmdata)
