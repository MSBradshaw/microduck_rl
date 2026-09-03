"""Unit tests for MicroduckOnPolicyRunner's durable reward-history CSV.

Pure-logic tests only (no env/training loop): construct a bare instance via
__new__ and exercise _extras_means / _append_reward_history directly, the
same pattern as testing any other small stateful helper in this repo.
"""

import csv
import statistics

import pytest
import torch

from mjlab_microduck.tasks import MicroduckOnPolicyRunner


def _bare_runner(tmp_path, rewbuffer=None, lenbuffer=None):
    runner = MicroduckOnPolicyRunner.__new__(MicroduckOnPolicyRunner)
    runner.device = "cpu"
    runner.current_learning_iteration = 0
    runner._last_ep_extras = []
    runner._reward_history_path = str(tmp_path / "reward_history.csv")

    class _FakeLogger:
        pass

    runner.logger = _FakeLogger()
    runner.logger.rewbuffer = rewbuffer or []
    runner.logger.lenbuffer = lenbuffer or []
    return runner


def test_extras_means_averages_per_key_across_steps():
    runner = MicroduckOnPolicyRunner.__new__(MicroduckOnPolicyRunner)
    runner.device = "cpu"
    runner._last_ep_extras = [
        {"Episode_Reward/pose_split": torch.tensor(1.0), "Curriculum/split_depth": torch.tensor(0.55)},
        {"Episode_Reward/pose_split": torch.tensor(3.0), "Curriculum/split_depth": torch.tensor(0.55)},
    ]
    means = runner._extras_means()
    assert means["Episode_Reward/pose_split"] == 2.0
    assert means["Curriculum/split_depth"] == pytest.approx(0.55)


def test_extras_means_handles_missing_keys_and_scalars():
    runner = MicroduckOnPolicyRunner.__new__(MicroduckOnPolicyRunner)
    runner.device = "cpu"
    runner._last_ep_extras = [
        {"Episode_Reward/foo": 2.0},
        {"Episode_Reward/foo": torch.tensor(4.0)},  # non-tensor + tensor mixed, like rsl_rl handles
    ]
    means = runner._extras_means()
    assert means["Episode_Reward/foo"] == 3.0


def test_extras_means_empty_when_no_snapshot_yet():
    runner = MicroduckOnPolicyRunner.__new__(MicroduckOnPolicyRunner)
    runner.device = "cpu"
    runner._last_ep_extras = []
    assert runner._extras_means() == {}


def test_append_reward_history_writes_header_once_and_rows(tmp_path):
    runner = _bare_runner(tmp_path, rewbuffer=[10.0, 12.0], lenbuffer=[300.0])
    runner._last_ep_extras = [{"Episode_Reward/pose_split": torch.tensor(-0.1)}]
    runner.current_learning_iteration = 250
    runner._append_reward_history()

    runner.current_learning_iteration = 500
    runner.logger.rewbuffer = [14.0]
    runner._append_reward_history()

    with open(runner._reward_history_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["iteration"] == "250"
    assert float(rows[0]["mean_reward"]) == statistics.mean([10.0, 12.0])
    assert float(rows[0]["Episode_Reward/pose_split"]) == pytest.approx(-0.1)
    assert rows[1]["iteration"] == "500"
    assert float(rows[1]["mean_reward"]) == 14.0


def test_append_reward_history_noop_without_log_dir(tmp_path):
    runner = _bare_runner(tmp_path)
    runner._reward_history_path = None
    runner._append_reward_history()  # must not raise
