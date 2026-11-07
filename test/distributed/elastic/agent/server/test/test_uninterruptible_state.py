#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

import os
import shutil
import signal
import tempfile
import time
import uuid
from unittest import mock
from unittest.mock import Mock

from torch.distributed.elastic.agent.server.api import WorkerSpec, WorkerState
from torch.distributed.elastic.agent.server.local_elastic_agent import (
    LocalElasticAgent,
    TORCHELASTIC_UNINTERRUPTIBLE_STATE_TIMEOUT,
)
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs
from torch.distributed.elastic.multiprocessing.errors import (
    ChildFailedError,
    ProcessFailure,
)
from torch.distributed.launcher.api import launch_agent, LaunchConfig
from torch.testing._internal.common_utils import (
    find_free_port,
    IS_MACOS,
    IS_WINDOWS,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_WITH_TSAN,
    TestCase,
)


READ_PROC_STATE = (
    "torch.distributed.elastic.agent.server.local_elastic_agent.read_proc_state"
)
HEALTHY_PID = 1001
STUCK_PID = 1002
STUCK_GLOBAL_RANK = 5
WORKER_SLEEP_SECONDS = 300


class UninterruptibleStateTest(TestCase):
    def setUp(self):
        super().setUp()
        self.log_dir = tempfile.mkdtemp(prefix=type(self).__name__)
        self.addCleanup(shutil.rmtree, self.log_dir, ignore_errors=True)

    def _stuck_rank_result(self, error_file: str):
        rdzv_handler = Mock()
        rdzv_handler.get_backend.return_value = "c10d"
        spec = WorkerSpec(
            role="default",
            local_world_size=2,
            entrypoint=time.sleep,
            args=(WORKER_SLEEP_SECONDS,),
            rdzv_handler=rdzv_handler,
            max_restarts=3,
            monitor_interval=0.01,
        )
        agent = LocalElasticAgent(
            spec,
            start_method="spawn",
            logs_specs=DefaultLogsSpecs(log_dir=self.log_dir),
        )
        agent._pcontext = Mock()
        agent._pcontext.pids.return_value = {0: HEALTHY_PID, 1: STUCK_PID}
        agent._pcontext.error_files = {0: error_file, 1: error_file}
        agent._worker_group.workers[0].global_rank = STUCK_GLOBAL_RANK - 1
        agent._worker_group.workers[1].global_rank = STUCK_GLOBAL_RANK
        agent._uninterruptible_state_first_seen[STUCK_PID] = time.monotonic() - 600
        agent._remaining_restarts = 3
        with mock.patch(
            READ_PROC_STATE, side_effect=lambda pid: "D" if pid == STUCK_PID else "S"
        ):
            result = agent._check_uninterruptible_state_timeout(
                agent._worker_group, timeout=60
            )
        self.assertEqual(agent._remaining_restarts, 0)
        return result

    def test_live_worker_error_file_is_not_parsed(self):
        error_file = os.path.join(self.log_dir, "error.json")
        with open(error_file, "w") as f:
            f.write('{"message": {"message": "trunc')

        result = self._stuck_rank_result(error_file)

        self.assertEqual(result.state, WorkerState.UNHEALTHY)
        self.assertEqual(list(result.failures), [STUCK_GLOBAL_RANK])
        failure = result.failures[STUCK_GLOBAL_RANK]
        self.assertEqual(failure.local_rank, 1)
        self.assertEqual(failure.pid, STUCK_PID)
        self.assertEqual(failure.exitcode, 1)
        self.assertEqual(failure.error_file, "<N/A>")
        self.assertIn("D-state", failure.message)

    def test_failure_describes_d_state(self):
        missing = os.path.join(self.log_dir, "missing.json")

        failure = self._stuck_rank_result(missing).failures[STUCK_GLOBAL_RANK]

        self.assertRegex(failure.message, r"D-state\) for 6\d\d\.\ds \(>= 60\.0s\)")
        # _record_worker_events reports error_file_data, not message.
        self.assertEqual(failure.error_file_data["message"], failure.message)
        # ProcessFailure shares its empty error data between instances.
        other = ProcessFailure(local_rank=0, pid=1, exitcode=1, error_file=missing)
        self.assertEqual(other.error_file_data["message"], "<NONE>")

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN or IS_WINDOWS or IS_MACOS,
        "test launches worker processes and reads Linux process state",
    )
    def test_launch_agent_raises_child_failed_error(self):
        worker_pids = set()

        def report_d_state(pid: int) -> str:
            worker_pids.add(pid)
            return "D"

        config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=2,
            run_id=str(uuid.uuid4()),
            rdzv_backend="c10d",
            rdzv_endpoint=f"127.0.0.1:{find_free_port()}",
            rdzv_configs={"timeout": 30},
            max_restarts=2,
            monitor_interval=0.05,
            start_method="spawn",
            logs_specs=DefaultLogsSpecs(log_dir=self.log_dir),
        )
        env = {TORCHELASTIC_UNINTERRUPTIBLE_STATE_TIMEOUT: "0.1"}
        try:
            with (
                mock.patch.dict(os.environ, env),
                mock.patch(READ_PROC_STATE, side_effect=report_d_state),
                self.assertRaises(ChildFailedError) as cm,
            ):
                launch_agent(config, time.sleep, [WORKER_SLEEP_SECONDS])
        finally:
            for pid in worker_pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        failures = cm.exception.failures
        self.assertEqual(sorted(failures), [0, 1])
        self.assertEqual({f.pid for f in failures.values()}, worker_pids)
        for global_rank, failure in failures.items():
            self.assertEqual(failure.local_rank, global_rank)
            self.assertEqual(failure.exitcode, 1)
            self.assertIn("D-state", failure.message)
        self.assertIn("D-state", str(cm.exception))


if __name__ == "__main__":
    run_tests()
