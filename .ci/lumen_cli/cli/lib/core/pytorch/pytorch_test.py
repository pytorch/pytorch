from __future__ import annotations

from typing import Any

from cli.lib.common.cli_helper import BaseRunner
from cli.lib.core.pytorch.lib import resolve_plan_for_test_config, run_test_plan


class PytorchTestRunner(BaseRunner):
    def __init__(self, args: Any) -> None:
        self.group_id = getattr(args, "group_id", None)
        self.test_config = getattr(args, "test_config", None)
        self.build_env = getattr(args, "build_env", None)
        self.test_id = getattr(args, "test_id", None)
        self.cmd = getattr(args, "cmd", None)
        self.shard_id = getattr(args, "shard_id", 1)
        self.num_shards = getattr(args, "num_shards", 1)
        self.no_upload = getattr(args, "no_upload", False)
        # --filter key=value pairs → dict
        raw_filters = getattr(args, "filter", []) or []
        self.filters = dict(f.split("=", 1) for f in raw_filters) if raw_filters else None

    def run(self) -> None:
        # --group-id: direct invocation or single-step repro, no resolution needed.
        # --test-config: resolve group_id from TEST_CONFIG + BUILD_ENV, replacing
        #   the if/elif dispatch in test.sh. build_env must be passed explicitly.
        group_id = self.group_id or resolve_plan_for_test_config(
            test_config=self.test_config,
            build_env=self.build_env,
        )
        run_test_plan(
            group_id=group_id,
            build_env=self.build_env,
            test_id=self.test_id,
            cmd=self.cmd,
            filters=self.filters,
            shard_id=self.shard_id,
            num_shards=self.num_shards,
            no_upload=self.no_upload,
        )
