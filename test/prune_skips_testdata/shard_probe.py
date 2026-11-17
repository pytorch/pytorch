"""Observe the real sharder input for TestPruneSkipsAtCollection."""

import json
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--shard-probe-log")
    parser.addoption("--shard-probe-resume-from")


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    resume_from = config.getoption("shard_probe_resume_from")
    if resume_from:
        config.pluginmanager.get_plugin("stepcurrentplugin").lastrun = resume_from
    sharder = config.pluginmanager.get_plugin("pytestshardplugin")
    original_filter = sharder.filter_items_by_shard

    def record_shard_input(items, shard_id, num_shards):
        selected = original_filter(items, shard_id, num_shards)
        Path(config.getoption("shard_probe_log")).write_text(
            json.dumps(
                {
                    "before": [item.nodeid for item in items],
                    "after": [item.nodeid for item in selected],
                }
            )
        )
        return selected

    sharder.filter_items_by_shard = record_shard_input


def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid.endswith("::TestPruneFixture::test_runs"):
            item.add_marker(pytest.mark.skip(reason="marked during collection"))


def pytest_collection_finish(session):
    path = Path(session.config.getoption("shard_probe_log"))
    if not path.exists():
        return
    snapshot = json.loads(path.read_text())
    snapshot["final"] = [item.nodeid for item in session.items]
    path.write_text(json.dumps(snapshot))
