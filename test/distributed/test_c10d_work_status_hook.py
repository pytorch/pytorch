# Owner(s): ["oncall: distributed"]
#
# Tests for the WorkStatusHook (torch._C._distributed_c10d.WorkStatusHook):
# stock ProcessGroupNCCL's periodic work status C10dLogger record, reported for
# any backend from the ProcessGroup pre/post collective hooks.

import os
import shutil
import sys
import tempfile
import unittest
from datetime import timedelta
from unittest import mock

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch._C._distributed_c10d import WorkStatusHook
from torch.distributed.distributed_c10d import _world
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import (
    HardwareClassification,
    IS_FBCODE,
    IS_SANDCASTLE,
    IS_WINDOWS,
    run_tests,
    TEST_CUDA,
)
from torch.utils.cpp_extension import load_inline


_LOGGER_MODULE = "work_status_test_logger"
# Where the parent builds the logger; spawned ranks import it from there.
_LOGGER_DIR_ENV = "TEST_WORK_STATUS_LOGGER_DIR"

# A C10dLogger can only be registered from C++.
_LOGGER_SOURCE = r"""
#include <torch/csrc/distributed/c10d/logger.hpp>

#include <pybind11/stl.h>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <sstream>

namespace {

std::mutex mutex;
std::condition_variable cv;
std::vector<c10d::C10dLoggingData> records;

struct CapturingLogger : c10d::C10dLogger {
  CapturingLogger() : c10d::C10dLogger("test") {}

  void log(const c10d::C10dLoggingData& data) override {
    if (!data.integers.count("last_enqueued_work")) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex);
      records.push_back(data);
    }
    cv.notify_all();
  }
};

using Record = std::tuple<
    int64_t,
    std::map<std::string, std::string>,
    std::map<std::string, int64_t>>;

} // namespace

void install() {
  c10d::C10dLogger::registerLogger(std::make_unique<CapturingLogger>());
}

int64_t count() {
  std::lock_guard<std::mutex> lock(mutex);
  return static_cast<int64_t>(records.size());
}

// Waits for a record of pg_name, at index `since` or later, holding every given
// key and value. Returns its index and contents.
Record wait_record(
    const std::string& pg_name,
    int64_t since,
    const std::map<std::string, std::string>& strings,
    const std::map<std::string, int64_t>& integers) {
  pybind11::gil_scoped_release no_gil;
  std::unique_lock<std::mutex> lock(mutex);
  Record result;
  const c10d::C10dLoggingData* last = nullptr;
  const bool found = cv.wait_for(lock, std::chrono::minutes(2), [&] {
    for (size_t i = since; i < records.size(); ++i) {
      const auto& r = records[i];
      if (r.strings.at("pg_name") != pg_name) {
        continue;
      }
      last = &r;
      bool match = true;
      for (const auto& [k, v] : strings) {
        match = match && r.strings.at(k) == v;
      }
      for (const auto& [k, v] : integers) {
        match = match && r.integers.at(k) == v;
      }
      if (match) {
        result = {static_cast<int64_t>(i), r.strings, r.integers};
        return true;
      }
    }
    return false;
  });
  if (!found) {
    std::ostringstream os;
    if (last != nullptr) {
      for (const auto& [k, v] : last->strings) {
        os << k << "=" << v << " ";
      }
      for (const auto& [k, v] : last->integers) {
        os << k << "=" << v << " ";
      }
    }
    TORCH_CHECK(false, "no matching record for ", pg_name, "; last: ", os.str());
  }
  return result;
}
"""

_KEYS = {
    "pg_id",
    "rank",
    "global_rank",
    "pg_name",
    "pg_desc",
    *(
        f"last_{kind}_{field}"
        for kind in ("enqueued", "started", "completed")
        for field in ("work", "work_name", "numel_in", "numel_out")
    ),
}

# Never observed by the hook; stock's defaults when collectives are not timed.
_STARTED = {
    "last_started_work": -1,
    "last_started_work_name": "",
    "last_started_numel_in": 0,
    "last_started_numel_out": 0,
}


def _op(kind, seq, name, numel_in, numel_out):
    return {
        f"last_{kind}_work": seq,
        f"last_{kind}_work_name": name,
        f"last_{kind}_numel_in": numel_in,
        f"last_{kind}_numel_out": numel_out,
    }


_skip_load_inline = unittest.skipIf(
    IS_FBCODE or IS_SANDCASTLE or IS_WINDOWS, "load_inline is unavailable"
)


def _needs_logger(test):
    """Marks a test that installs the capturing logger."""
    test._needs_logger = True
    return _skip_load_inline(test)


def _build_logger():
    """Builds the capturing logger once, in the parent, for spawned ranks."""
    if _LOGGER_DIR_ENV in os.environ:
        return
    build_dir = tempfile.mkdtemp(prefix="work_status_logger_")
    unittest.addModuleCleanup(shutil.rmtree, build_dir, ignore_errors=True)
    load_inline(
        name=_LOGGER_MODULE,
        cpp_sources=_LOGGER_SOURCE,
        functions=["install", "count", "wait_record"],
        build_directory=build_dir,
    )
    os.environ[_LOGGER_DIR_ENV] = build_dir
    unittest.addModuleCleanup(os.environ.pop, _LOGGER_DIR_ENV, None)


def _install_logger():
    sys.path.insert(0, os.environ[_LOGGER_DIR_ENV])
    import work_status_test_logger as logger  # type: ignore[import-not-found]

    logger.install()
    return logger


class _WorkStatusTestBase(MultiProcessTestCase):
    backend_name = "gloo"
    device_type = "cpu"
    period_ms = "50"

    @property
    def world_size(self):
        return 2

    @property
    def device(self):
        if self.device_type == "cuda":
            return torch.device("cuda", self.rank)
        return torch.device(self.device_type)

    def setUp(self):
        super().setUp()
        if getattr(getattr(self, self._testMethodName), "_needs_logger", False):
            _build_logger()
        # Read once per process, at the first attach.
        env = mock.patch.dict(
            os.environ, {"TORCH_DIST_WORK_STATUS_PERIOD_MS": self.period_ms}
        )
        env.start()
        self.addCleanup(env.stop)
        self._spawn_processes()

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _init_pg(self, backend=None):
        if self.device_type == "cuda":
            torch.cuda.set_device(self.rank)
        self.store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend or self.backend_name,
            world_size=self.world_size,
            rank=self.rank,
            store=self.store,
            timeout=timedelta(seconds=60),
        )
        return dist.group.WORLD

    def _ones(self, n):
        return torch.ones(n, device=self.device)


class AbstractWorkStatusHookTest:
    hw_classification = HardwareClassification.GENERIC

    def test_attached_until_destroyed(self):
        pg = self._init_pg()
        self.assertIsInstance(_world.pg_work_status_hooks[pg], WorkStatusHook)
        sub = dist.new_group([1])
        if self.rank == 1:
            self.assertIn(sub, _world.pg_work_status_hooks)
            status = _world.pg_work_status_hooks[sub].status()
            self.assertEqual(status["rank"], 0)
            self.assertEqual(status["global_rank"], 1)
            dist.destroy_process_group(sub)
            self.assertNotIn(sub, _world.pg_work_status_hooks)
        dist.destroy_process_group()
        self.assertEqual(_world.pg_work_status_hooks, {})

    def test_initial_record(self):
        pg = self._init_pg()
        status = _world.pg_work_status_hooks[pg].status()
        self.assertEqual(set(status), _KEYS)
        self.assertIsInstance(status["pg_id"], int)
        del status["pg_id"]
        self.assertEqual(
            status,
            {
                "rank": self.rank,
                "global_rank": self.rank,
                "pg_name": pg.group_name,
                "pg_desc": "default_pg",
                **_STARTED,
                **_op("enqueued", -1, "", 0, 0),
                **_op("completed", -1, "", 0, 0),
            },
        )

    @_needs_logger
    def test_ops_record(self):
        logger = _install_logger()
        pg = self._init_pg()
        peer = 1 - self.rank
        root = self.rank == 0
        ones = self._ones
        # Without a future or completion push (gloo's send/recv and
        # reduce_scatter), an op only ever shows as enqueued.
        futureless = (
            {"SEND", "RECV", "REDUCE_SCATTER"} if self.backend_name == "gloo" else ()
        )
        # (op, last enqueued (seq, name, numel in, numel out)), run in order.
        # Collectives and p2p ops are numbered separately, as in stock.
        ops = [
            (lambda: dist.all_reduce(ones(1)), (1, "ALLREDUCE", 1, 1)),
            (
                lambda: dist.send(ones(4), peer) if root else dist.recv(ones(4), peer),
                (1, "SEND" if root else "RECV", 4, 4),
            ),
            (
                lambda: dist.all_gather([ones(3), ones(3)], ones(3)),
                (2, "ALLGATHER", 3, 6),
            ),
            (
                lambda: dist.all_gather_into_tensor(ones(6), ones(3)),
                (3, "_ALLGATHER_BASE", 3, 6),
            ),
            (
                lambda: dist.reduce_scatter(ones(3), [ones(3), ones(3)]),
                (4, "REDUCE_SCATTER", 6, 3),
            ),
            (lambda: dist.broadcast(ones(2), src=0), (5, "BROADCAST", 2, 2)),
            (
                lambda: dist.all_to_all_single(ones(4), ones(4)),
                (6, "ALLTOALL_BASE", 4, 4),
            ),
            (
                lambda: dist.gather(ones(5), [ones(5), ones(5)] if root else None),
                (7, "GATHER", 5, 10 if root else 0),
            ),
            (
                lambda: dist.scatter(ones(5), [ones(5), ones(5)] if root else None),
                (8, "SCATTER", 10 if root else 0, 5),
            ),
            (dist.barrier, (9, "BARRIER", 0, 0)),
        ]
        since = 0
        completed = None
        for i, (run, enqueued) in enumerate(ops):
            run()
            if enqueued[1] not in futureless:
                completed = enqueued
            want = {**_op("enqueued", *enqueued), **_op("completed", *completed)}
            strings = {k: v for k, v in want.items() if isinstance(v, str)}
            integers = {k: v for k, v in want.items() if isinstance(v, int)}
            since, rec_strings, rec_integers = logger.wait_record(
                pg.group_name, since, strings, integers
            )
            record = {**rec_strings, **rec_integers}
            self.assertEqual(set(record), _KEYS, i)
            self.assertEqual({k: record[k] for k in _STARTED}, _STARTED, i)
            self.assertEqual(record["rank"], self.rank, i)
            self.assertEqual(record["global_rank"], self.rank, i)
            self.assertEqual(record["pg_desc"], "default_pg", i)

    @_needs_logger
    def test_pending_op_not_completed(self):
        logger = _install_logger()
        pg = self._init_pg()
        dist.all_reduce(self._ones(1))
        allreduce = ("ALLREDUCE", 1, 1)
        if self.rank == 0:
            # Rank 1 holds back, so this all_reduce cannot complete.
            work = dist.all_reduce(self._ones(1), async_op=True)
            want = {
                **_op("enqueued", 2, *allreduce),
                **_op("completed", 1, *allreduce),
            }
            logger.wait_record(
                pg.group_name,
                logger.count(),
                {k: v for k, v in want.items() if isinstance(v, str)},
                {k: v for k, v in want.items() if isinstance(v, int)},
            )
            self.store.set("release", "1")
            work.wait()
        else:
            self.store.wait(["release"])
            dist.all_reduce(self._ones(1))
        logger.wait_record(
            pg.group_name,
            0,
            {},
            {"last_enqueued_work": 2, "last_completed_work": 2},
        )

    @_needs_logger
    def test_batch_p2p_not_completed(self):
        logger = _install_logger()
        pg = self._init_pg()
        peer = 1 - self.rank
        dist.all_reduce(self._ones(1))
        ops = [
            dist.P2POp(dist.isend, self._ones(2), peer),
            dist.P2POp(dist.irecv, self._ones(2), peer),
        ]
        for work in dist.batch_isend_irecv(ops):
            work.wait()
        # Batched p2p ops are never seen completing, only passed over.
        want = {
            **_op("enqueued", 2, "RECV", 2, 2),
            **_op("completed", 1, "ALLREDUCE", 1, 1),
        }
        logger.wait_record(
            pg.group_name,
            logger.count(),
            {k: v for k, v in want.items() if isinstance(v, str)},
            {k: v for k, v in want.items() if isinstance(v, int)},
        )
        dist.all_reduce(self._ones(1))
        logger.wait_record(
            pg.group_name,
            0,
            {},
            {"last_enqueued_work": 2, "last_completed_work": 2},
        )

    @_needs_logger
    def test_remove_stops_tracking(self):
        _install_logger()
        pg = self._init_pg()
        hook = _world.pg_work_status_hooks[pg]
        hook.remove()
        hook.remove()
        dist.all_reduce(self._ones(1))
        status = hook.status()
        self.assertEqual(status["last_enqueued_work"], -1)
        self.assertEqual(status["last_completed_work"], -1)

    @_needs_logger
    def test_numbered_before_logger(self):
        pg = self._init_pg()
        dist.all_reduce(self._ones(1))
        dist.all_reduce(self._ones(1))
        # Ops are numbered before a logger is registered, but not tracked.
        status = _world.pg_work_status_hooks[pg].status()
        self.assertEqual(status["last_enqueued_work"], -1)
        logger = _install_logger()
        dist.all_reduce(self._ones(1))
        logger.wait_record(
            pg.group_name,
            0,
            {},
            {"last_enqueued_work": 3, "last_completed_work": 3},
        )


def _make_test_class(backend_name, device_type):
    name = backend_name.title().replace("-", "") + "WorkStatusHookTest"
    cls = type(
        name,
        (AbstractWorkStatusHookTest, _WorkStatusTestBase),
        {
            "backend_name": backend_name,
            "device_type": device_type,
            "hw_classification": HardwareClassification.CUDA
            if device_type == "cuda"
            else HardwareClassification.GENERIC,
        },
    )
    cls = unittest.skipIf(
        not dist.is_backend_available(backend_name),
        f"{backend_name} backend is not available",
    )(cls)
    if device_type == "cuda":
        cls = unittest.skipIf(
            not TEST_CUDA or torch.cuda.device_count() < 2,
            "requires at least 2 GPUs",
        )(cls)
    return cls


for _backend, _device in [("gloo", "cpu"), ("nccl2", "cuda"), ("nccl-lazy", "cuda")]:
    _cls = _make_test_class(_backend, _device)
    globals()[_cls.__name__] = _cls
del _backend, _device, _cls


@unittest.skipIf(
    not TEST_CUDA or torch.cuda.device_count() < 2, "requires at least 2 GPUs"
)
@unittest.skipIf(
    not dist.is_backend_available("nccl2"), "nccl2 backend is not available"
)
class Nccl2CudaWorkStatusHookTest(_WorkStatusTestBase):
    """CUDA-specific behaviour, on a backend that pushes completions."""

    hw_classification = HardwareClassification.CUDA
    backend_name = "nccl2"
    device_type = "cuda"

    @_needs_logger
    def test_graph_capture_not_tracked(self):
        logger = _install_logger()
        pg = self._init_pg()
        tensor = self._ones(1)
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dist.all_reduce(tensor)
        graph.replay()
        dist.all_reduce(tensor)
        # Like stock, the captured op is numbered but never enqueued.
        want = {
            **_op("enqueued", 3, "ALLREDUCE", 1, 1),
            **_op("completed", 3, "ALLREDUCE", 1, 1),
        }
        logger.wait_record(
            pg.group_name,
            0,
            {k: v for k, v in want.items() if isinstance(v, str)},
            {k: v for k, v in want.items() if isinstance(v, int)},
        )

    @_needs_logger
    def test_mixed_backends(self):
        logger = _install_logger()
        pg = self._init_pg(f"cpu:gloo,cuda:{self.backend_name}")
        # Each op completes through its own backend's mechanism.
        for seq, tensor in enumerate(
            [torch.ones(1), self._ones(1), torch.ones(1)], start=1
        ):
            dist.all_reduce(tensor)
            logger.wait_record(
                pg.group_name,
                0,
                {},
                {"last_enqueued_work": seq, "last_completed_work": seq},
            )
        # Which backend runs a barrier is not known, so it is only passed over.
        dist.barrier()
        logger.wait_record(
            pg.group_name,
            0,
            {"last_enqueued_work_name": "BARRIER"},
            {"last_enqueued_work": 4, "last_completed_work": 3},
        )
        dist.all_reduce(torch.ones(1))
        logger.wait_record(
            pg.group_name,
            0,
            {},
            {"last_enqueued_work": 5, "last_completed_work": 5},
        )

    @_needs_logger
    def test_debug_detail(self):
        logger = _install_logger()
        # ProcessGroupWrapper forwards the wrapped backend's completion hooks.
        dist.set_debug_level(dist.DebugLevel.DETAIL)
        pg = self._init_pg()
        self.assertEqual(
            type(pg._get_backend(self.device)).__name__, "_ProcessGroupWrapper"
        )
        dist.all_reduce(self._ones(1))
        logger.wait_record(
            pg.group_name,
            0,
            {},
            {"last_enqueued_work": 1, "last_completed_work": 1},
        )


class WorkStatusHookLogAllTest(_WorkStatusTestBase):
    hw_classification = HardwareClassification.GENERIC
    # Long enough that only _log_all emits during the test.
    period_ms = str(3600 * 1000)

    @_needs_logger
    def test_log_all_emits_status(self):
        logger = _install_logger()
        pg = self._init_pg()
        hook = _world.pg_work_status_hooks[pg]
        work = dist.all_reduce(self._ones(1), async_op=True)
        # Callbacks run in order, so the hook's has run once this one has.
        work.get_future().then(lambda fut: None).wait()
        status = hook.status()
        self.assertEqual(status["last_completed_work"], 1)

        before = logger.count()
        WorkStatusHook._log_all()
        self.assertEqual(logger.count(), before + 1)
        index, strings, integers = logger.wait_record(pg.group_name, before, {}, {})
        self.assertEqual(index, before)
        self.assertEqual({**strings, **integers}, status)


class WorkStatusHookDisabledTest(_WorkStatusTestBase):
    hw_classification = HardwareClassification.GENERIC
    period_ms = "0"

    def test_not_attached(self):
        pg = self._init_pg()
        self.assertNotIn(pg, _world.pg_work_status_hooks)
        self.assertIsNone(WorkStatusHook.attach(pg))


@unittest.skipIf(
    not TEST_CUDA or torch.cuda.device_count() < 2, "requires at least 2 GPUs"
)
@unittest.skipIf(not dist.is_nccl_available(), "requires nccl")
class WorkStatusHookStockNcclTest(_WorkStatusTestBase):
    """Stock ProcessGroupNCCL reports its own status and is left alone."""

    hw_classification = HardwareClassification.CUDA

    device_type = "cuda"

    def test_stock_nccl_not_attached(self):
        # Read where "nccl" is first resolved, so set before init. Set in the
        # rank's own process.
        os.environ["TORCH_DIST_USE_NCCL2"] = "0"
        pg = self._init_pg("nccl")
        self.assertEqual(pg._get_backend(self.device).name(), "nccl")
        self.assertNotIn(pg, _world.pg_work_status_hooks)
        self.assertIsNone(WorkStatusHook.attach(pg))


if __name__ == "__main__":
    run_tests()
