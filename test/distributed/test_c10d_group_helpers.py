# Owner(s): ["oncall: distributed"]

"""Single-process unit tests for c10d group creation and abort helpers."""

import datetime
from unittest import mock

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


_WORLD_ATTRS = (
    "pg_map",
    "pg_names",
    "pg_group_ranks",
    "pg_backend_config",
    "pg_to_tag",
    "tags_to_pg",
    "pg_coalesce_state",
)


class _WorldStateTestCase(TestCase):
    """Restores the global ``_world`` state mutated by a test."""

    def setUp(self):
        super().setUp()
        self._saved_world = {
            attr: getattr(c10d._world, attr).copy() for attr in _WORLD_ATTRS
        }
        self._saved_default_pg = c10d._world.default_pg
        self._saved_group_count = c10d._world.group_count

    def tearDown(self):
        for attr, value in self._saved_world.items():
            container = getattr(c10d._world, attr)
            container.clear()
            container.update(value)
        c10d._world.default_pg = self._saved_default_pg
        c10d._world.group_count = self._saved_group_count
        super().tearDown()


class TestGroupNameHashSalt(_WorldStateTestCase):
    """The hash salt must be ``_world.group_count``, which every rank advances in
    lockstep, not ``len(_world.pg_names)``, which diverges under asymmetric
    membership and makes ranks disagree on the store prefix."""

    hw_classification = HardwareClassification.GENERIC

    def _register_dummy_pgs(self, n):
        for i in range(n):
            c10d._world.pg_names[object()] = c10d.GroupName(f"dummy_{i}")

    def test_hash_salt_uses_group_count(self):
        ranks = [0, 1, 2, 3]
        c10d._world.group_count = 5
        name_a = c10d._hash_ranks_to_str(ranks)
        c10d._world.group_count = 6
        name_b = c10d._hash_ranks_to_str(ranks)
        self.assertNotEqual(name_a, name_b)

    def test_hash_independent_of_pg_names_length(self):
        ranks = [0, 1, 2, 3]
        c10d._world.group_count = 9
        c10d._world.pg_names.clear()
        name_few = c10d._hash_ranks_to_str(ranks)
        self._register_dummy_pgs(5)
        name_many = c10d._hash_ranks_to_str(ranks)
        self.assertEqual(name_few, name_many)

    def test_two_asymmetric_ranks_agree_on_name(self):
        ranks = [0, 1, 2, 3]
        c10d._world.group_count = 3
        c10d._world.pg_names.clear()
        self._register_dummy_pgs(2)
        name_member = c10d._process_group_name(ranks, use_hashed_name=True)
        c10d._world.group_count = 3
        c10d._world.pg_names.clear()
        name_non_member = c10d._process_group_name(ranks, use_hashed_name=True)
        self.assertEqual(name_member, name_non_member)

    def test_process_group_name_increments_group_count(self):
        for use_hashed_name in (True, False):
            c10d._world.group_count = 10
            c10d._process_group_name([0, 1], use_hashed_name=use_hashed_name)
            self.assertEqual(c10d._world.group_count, 11)

    def test_nonhashed_name_is_group_count_before_increment(self):
        c10d._world.group_count = 42
        name = c10d._process_group_name([0, 1], use_hashed_name=False)
        self.assertEqual(name, "42")
        self.assertEqual(c10d._world.group_count, 43)

    def test_repeated_hashed_same_ranks_get_distinct_names(self):
        c10d._world.group_count = 0
        first = c10d._process_group_name([2, 3], use_hashed_name=True)
        second = c10d._process_group_name([2, 3], use_hashed_name=True)
        self.assertNotEqual(first, second)


class TestSplitSource(_WorldStateTestCase):
    hw_classification = HardwareClassification.GENERIC
    TIMEOUT = datetime.timedelta(seconds=30)

    def _parent(self, split_src=None):
        parent = mock.MagicMock()
        parent.bound_device_id = torch.device("cuda:0")
        if split_src is None:
            split_src = mock.MagicMock()
            split_src.supports_splitting = True
        parent._get_backend.return_value = split_src
        return parent

    def _split_source(self, parent_backend, backend, split_src=None):
        parent = self._parent(split_src)
        pg_map = {} if parent_backend is None else {parent: (parent_backend, None)}
        with mock.patch.dict(c10d._world.pg_map, pg_map, clear=True):
            return c10d._get_split_source(parent, backend)

    def test_missing_parent_world_state_skips_split(self):
        self.assertIsNone(self._split_source(None, "cuda:nccl"))

    def test_missing_device_mappings_skip_split(self):
        self.assertIsNone(self._split_source("cpu:gloo", "xpu:xccl"))

    def test_malformed_backend_mapping_skips_split(self):
        self.assertIsNone(self._split_source("cuda:nccl", "cuda:nccl:extra"))

    def _drive_non_member(self, split_src, requested_backend="cuda:nccl"):
        """Run ``_new_process_group_helper`` on a rank outside the new group."""
        default = self._parent(split_src)
        default.rank.return_value = 0
        with (
            mock.patch.dict(
                c10d._world.pg_map,
                {default: ("cuda:nccl", dist.HashStore())},
                clear=True,
            ),
            mock.patch.multiple(
                c10d,
                is_initialized=lambda: True,
                _get_default_group=lambda: default,
            ),
            mock.patch.object(c10d, "BackendConfig") as backend_config,
        ):
            res = c10d._new_process_group_helper(
                group_size=2,
                group_rank=0,
                global_ranks_in_group=[1, 2],  # rank 0 is not a member
                backend=requested_backend,
                store=dist.HashStore(),
                group_name=c10d.GroupName(self.id()),
                timeout=self.TIMEOUT,
            )
        self.assertEqual(res, (dist.GroupMember.NON_GROUP_MEMBER, None))
        backend_config.assert_not_called()

    def test_non_member_performs_nocolor_split(self):
        split_src = mock.MagicMock()
        split_src.supports_splitting = True
        self._drive_non_member(split_src)
        split_src.perform_nocolor_split.assert_called_once_with(torch.device("cuda:0"))

    def test_non_member_skips_parent_split_for_different_backend(self):
        split_src = mock.MagicMock()
        split_src.supports_splitting = True
        self._drive_non_member(split_src, requested_backend="cuda:mccl")
        split_src.perform_nocolor_split.assert_not_called()

    def test_split_source_without_nocolor_split_skips_split(self):
        split_src = mock.MagicMock(spec=["supports_splitting"])
        split_src.supports_splitting = True
        self.assertIsNone(
            self._split_source("cuda:nccl", "cuda:nccl", split_src=split_src)
        )

    def test_non_member_skips_nocolor_split_without_method(self):
        split_src = mock.MagicMock(spec=["supports_splitting"])
        split_src.supports_splitting = True
        self._drive_non_member(split_src)


class TestAbortReleasesNvshmemAfterAbort(_WorldStateTestCase):
    """The NVSHMEM team release synchronizes the device, so it must follow the
    abort; otherwise it waits on hung kernels."""

    hw_classification = HardwareClassification.GENERIC

    def _register_pg(self, calls, name):
        pg = mock.MagicMock()
        pg._get_backend.return_value = mock.MagicMock(spec=[])
        calls.attach_mock(pg.abort, f"abort_{name}")
        c10d._world.pg_map[pg] = (None, None)
        c10d._world.pg_names[pg] = c10d.GroupName(name)
        c10d._world.pg_group_ranks[pg] = {}
        c10d._world.pg_backend_config[pg] = ""
        c10d._world.pg_to_tag[pg] = None
        return pg

    def _patches(self, calls):
        return mock.patch.multiple(
            c10d,
            _release_nvshmem_team_pool=calls.release,
            _unregister_process_group=lambda group_name: None,
            _unregister_all_process_groups=lambda: None,
            _update_default_pg=lambda pg: setattr(c10d._world, "default_pg", pg),
        )

    def test_release_follows_abort(self):
        calls = mock.Mock()
        c10d._world.default_pg = self._register_pg(calls, "world")
        group_a = self._register_pg(calls, "a")
        self._register_pg(calls, "b")
        call = mock.call

        with self._patches(calls):
            c10d._abort_process_group(group_a)
            self.assertEqual(calls.mock_calls, [call.abort_a(), call.release("a")])

            calls.reset_mock()
            c10d._abort_process_group()

        self.assertCountEqual(
            calls.mock_calls[:2], [call.abort_world(), call.abort_b()]
        )
        self.assertCountEqual(
            calls.mock_calls[2:], [call.release("world"), call.release("b")]
        )
        self.assertEqual(c10d._world.pg_map, {})
        self.assertIsNone(c10d._world.default_pg)

    def test_release_failure_does_not_stop_cleanup(self):
        calls = mock.Mock()
        calls.release.side_effect = RuntimeError(
            "CUDA error: unspecified launch failure"
        )
        c10d._world.default_pg = self._register_pg(calls, "world")
        group_a = self._register_pg(calls, "a")
        self._register_pg(calls, "b")

        with self._patches(calls):
            with self.assertLogs(c10d.logger, level="WARNING") as logs:
                c10d._abort_process_group(group_a)
            self.assertIn("NVSHMEM team pool of group a", "\n".join(logs.output))
            self.assertNotIn(group_a, c10d._world.pg_map)

            with self.assertLogs(c10d.logger, level="WARNING") as logs:
                c10d._abort_process_group()

        output = "\n".join(logs.output)
        self.assertIn("NVSHMEM team pool of group world", output)
        self.assertIn("NVSHMEM team pool of group b", output)
        self.assertEqual(c10d._world.pg_map, {})
        self.assertIsNone(c10d._world.default_pg)


class TestMonitoredBarrierGuard(TestCase):
    """monitored_barrier needs a CPU backend, not a group named "gloo"."""

    hw_classification = HardwareClassification.GENERIC
    TIMEOUT = datetime.timedelta(seconds=30)

    def _group(self, device_types):
        group = mock.MagicMock()
        group._device_types = device_types
        return group

    def test_multi_backend_group_dispatches(self):
        group = self._group([torch.device("cpu"), torch.device("cuda")])
        dist.monitored_barrier(group=group, timeout=self.TIMEOUT, wait_all_ranks=True)
        group.monitored_barrier.assert_called_once_with(
            self.TIMEOUT, wait_all_ranks=True
        )

    def test_group_without_cpu_backend_raises(self):
        group = self._group([torch.device("cuda")])
        with self.assertRaisesRegex(ValueError, "CPU-capable backend"):
            dist.monitored_barrier(group=group, timeout=self.TIMEOUT)
        group.monitored_barrier.assert_not_called()


if __name__ == "__main__":
    run_tests()
