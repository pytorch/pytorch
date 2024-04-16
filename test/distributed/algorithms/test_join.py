# Owner(s): ["oncall: distributed"]

import contextlib
import os
import sys
from typing import Any, Optional

import torch
import torch.distributed as dist

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.distributed.algorithms.join import Join, Joinable, JoinHook
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    require_n_gpus_for_nccl_backend,
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN

if TEST_WITH_DEV_DBG_ASAN:
    print("Skip dev-asan as torch + multiprocessing spawn have known issues", file=sys.stderr)
    sys.exit(0)

BACKEND = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO
WORLD_SIZE = min(4, max(2, torch.cuda.device_count()))

# Constants used for testing post-hooks
BEFORE_CONSTANT = 41
AFTER_CONSTANT = 42


class AllReducerJoinHook(JoinHook):
    r"""
    Join hook for :class:`AllReducer`.

    Arguments:
        allreducer (AllReducer): the :class:`AllReducer` object using this
            hook.
        num_allreduces (int): the number of all-reduces to shadow per
            iteration.
        run_post_hook (bool): a flag enabling the post-hook logic.
    """
    def __init__(
        self,
        allreducer,
        num_allreduces,
        run_post_hook
    ):
        self.allreducer = allreducer
        self.num_allreduces = num_allreduces
        self.run_post_hook = run_post_hook

    def main_hook(self):
        r"""
        Shadows each all-reduce; the number of all-reduces is passed into the
        constructor as ``num_allreduces``.
        """
        device = self.allreducer.device
        for _ in range(self.num_allreduces):
            t = torch.zeros(1, device=device)
            dist.all_reduce(t)

    def post_hook(self, is_last_joiner: bool):
        r"""
        Broadcasts a tensor containing a magic constant ``AFTER_CONSTANT`` from
        the last joiner to all other processes.
        """
        if not self.run_post_hook:
            return
        rank = dist.get_rank(self.allreducer.process_group)
        common_rank = self.allreducer.find_common_rank(rank, is_last_joiner)
        device = self.allreducer.device
        if rank == common_rank:
            self.allreducer.post_hook_tensor = torch.tensor([AFTER_CONSTANT], device=device)
        dist.broadcast(self.allreducer.post_hook_tensor, src=common_rank)


class AllReducer(Joinable):
    r"""
    Example :class:`Joinable` that performs some number of all-reduces as its
    per-iteration collective communication.
    """
    def __init__(self, device, process_group):
        super().__init__()
        self.device = device
        self.process_group = process_group
        self.post_hook_tensor = torch.tensor([BEFORE_CONSTANT], device=self.device)

    def __call__(self, num_allreduces=1):
        r"""
        All-reduces a dim-1 one tensor ``num_allreduces``-many times, and
        returns the total result.
        """
        Join.notify_join_context(self)
        device = self.device
        total = 0
        for _ in range(num_allreduces):
            t = torch.ones(1, device=device)
            dist.all_reduce(t)
            total += t.item()
        return total

    def join_hook(self, **kwargs) -> JoinHook:
        r"""
        Returns a join hook that shadows some number of all-reduces; by default,
        this number is 1.
        """
        num_allreduces = kwargs.get("num_allreduces", 1)
        run_post_hook = kwargs.get("run_post_hooks", False)
        return AllReducerJoinHook(
            self,
            num_allreduces,
            run_post_hook
        )

    @property
    def join_device(self) -> torch.device:
        return self.device

    @property
    def join_process_group(self) -> Any:
        return self.process_group

    def find_common_rank(self, rank, to_consider):
        r"""
        Returns the max rank of the ones to consider over the process group.
        """
        common_rank = torch.tensor(
            [rank if to_consider else -1],
            device=self.device
        )
        dist.all_reduce(common_rank, op=dist.ReduceOp.MAX, group=self.process_group)
        common_rank = common_rank.item()
        assert common_rank >= 0
        return common_rank

class TestJoin(MultiProcessTestCase):
    r"""Test cases for the generic join context."""
    def setUp(self):
        super().setUp()
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["BACKEND"] = BACKEND
        self._spawn_processes()

    @property
    def device(self):
        return torch.device(self.rank) if BACKEND == dist.Backend.NCCL \
            else torch.device("cpu")

    @property
    def world_size(self):
        return WORLD_SIZE

    @property
    def process_group(self):
        return dist.group.WORLD

    def tearDown(self):
        try:
            dist.destroy_process_group()
        except AssertionError:
            pass
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def dist_init(self, rank, world_size, backend=BACKEND):
        store = dist.FileStore(self.file_name, world_size)
        return dist.init_process_group(
            backend=backend,
            store=store,
            rank=rank,
            world_size=world_size
        )

    def construct_uneven_inputs(self, base, offset, device=None):
        r"""
        Returns uneven inputs: rank i gets ``base`` + i * ``offset`` inputs.
        """
        if device is None:
            device = self.device
        return [torch.zeros(1, device=device) for _ in range(base + self.rank * offset)]

    def construct_even_inputs(self, base, device=None):
        r"""Returns even inputs: each rank gets ``base`` inputs."""
        if device is None:
            device = self.device
        return [torch.zeros(1, device=device) for _ in range(base)]

    @property
    def base_num_inputs(self):
        r"""Base number of inputs to be used by all ranks."""
        return 3

    @property
    def offset(self):
        r"""Rank i gets i * ``offset`` additional inputs."""
        return 1

    def _test_join_base(
        self,
        uneven_inputs: bool,
        num_joinables: int,
        enable: bool,
        throw_on_early_termination: bool,
        num_allreduces: int,
        run_post_hooks: bool,
        expected_total: Optional[int] = None,
    ):
        r"""
        Skeleton for all :class:`Join` tests.

        Arguments:
            uneven_inputs (bool): ``True`` to use uneven inputs; ``False``
                otherwise.
            num_joinables (int): number of :class:`AllReducer` s to construct.
            enable (bool): ``True`` to enable the join context manager;
                ``False`` otherwise.
            throw_on_early_termination (bool): ``True`` to raise an exception
                upon detecting uneven inputs; ``False`` otherwise.
            num_allreduces (int): number of all-reduces to perform per input.
            run_post_hooks (bool): ``True`` to run post-hooks; ``False``
                otherwise.
            expected_total (Optional[int]): ``None`` to not check the expected
                all-reduce total; otherwise, the expected total; default is
                ``None``.
        """
        self.dist_init(self.rank, self.world_size)

        allreducers = [
            AllReducer(self.device, self.process_group)
            for _ in range(num_joinables)
        ]
        for allreducer in allreducers:
            self.assertEqual(allreducer.post_hook_tensor.item(), BEFORE_CONSTANT)

        inputs = self.construct_uneven_inputs(self.base_num_inputs, self.offset) \
            if uneven_inputs \
            else self.construct_even_inputs(self.base_num_inputs)
        allreduce_total = 0

        # Expect a `RuntimeError` if `throw_on_early_termination=True`
        # Rank 0 exhausts its inputs first
        expected_msg = "Rank 0 exhausted all inputs." if self.rank == 0 \
            else "Detected at least one rank that exhausted inputs. " \
            "Throwing across all ranks."
        with self.assertRaisesRegex(
            RuntimeError,
            expected_msg
        ) if throw_on_early_termination else contextlib.nullcontext():
            with Join(
                allreducers,
                enable=enable,
                throw_on_early_termination=throw_on_early_termination,
                num_allreduces=num_allreduces,
                run_post_hooks=run_post_hooks
            ):
                for _ in inputs:
                    for allreducer in allreducers:
                        allreduce_total += allreducer(num_allreduces)

        if throw_on_early_termination:
            return

        # Check `expected_total` if not `None`
        if expected_total:
            self.assertEqual(allreduce_total, expected_total)

        # All `AllReduce` instances should receive the updated
        # `post_hook_tensor` from the last-joined process
        if run_post_hooks:
            for allreducer in allreducers:
                self.assertEqual(allreducer.post_hook_tensor.item(), AFTER_CONSTANT)

    @require_n_gpus_for_nccl_backend(
        WORLD_SIZE, BACKEND
    )
    def test_single_joinable_main_hooks(self):
        r"""Tests the main hooks of a single :class:`Joinable`."""
        num_joinables = 1
        num_allreduces = 1
        run_post_hooks = False
        # Non-joined processes all-reduce a 1, so this rank's all-reduce total
        # should be precisely equal to the total number of inputs processed
        # before it joined
        expected_total = self.world_size * self.base_num_inputs
        # Rank i runs for i additional iterations
        for num_joined in range(1, self.rank + 1):
            expected_total += (self.world_size - num_joined) * self.offset

        self._test_join_base(
            uneven_inputs=True,
            num_joinables=num_joinables,
            enable=True,
            throw_on_early_termination=False,
            num_allreduces=num_allreduces,
            run_post_hooks=run_post_hooks,
            expected_total=expected_total
        )

    @require_n_gpus_for_nccl_backend(
        WORLD_SIZE, BACKEND
    )
    def test_single_joinable_post_hooks(self):
        r"""Tests the post-hooks of a single :class:`Joinable`."""
        num_joinables = 1
        num_allreduces = 0  # set to 0 to skip the main hooks
        run_post_hooks = False

        self._test_join_base(
            uneven_inputs=True,
            num_joinables=num_joinables,
            enable=True,
            throw_on_early_termination=False,
            num_allreduces=num_allreduces,
            run_post_hooks=run_post_hooks,
            expected_total=None
        )

    @require_n_gpus_for_nccl_backend(
        WORLD_SIZE, BACKEND
    )
    def test_single_joinable(self):
        r"""
        Tests the main hooks and post-hooks of a single :class:`Joinable`
        together.

        This combines ``test_single_joinable_main_hooks()`` and
        ``test_single_joinable_post_hooks()`` into a single test to ensure that
        main hooks and post-hooks operate correctly together.
        """
        num_joinables = 1
        num_allreduces = 1
        run_post_hooks = True

        expected_total = self.world_size * self.base_num_inputs
        for num_joined in range(1, self.rank + 1):
            expected_total += (self.world_size - num_joined) * self.offset

        self._test_join_base(
            uneven_inputs=True,
            num_joinables=num_joinables,
            enable=True,
            throw_on_early_termination=False,
            num_allreduces=num_allreduces,
            run_post_hooks=run_post_hooks,
            expected_total=expected_total
        )

    @require_n_gpus_for_nccl_backend(
        WORLD_SIZE, BACKEND
    )
    def test_multiple_joinables(self):
        r"""
        Tests the main hooks and post-hooks of multiple :class:`Joinable` s
        together.

        This generalizes ``test_single_joinable()`` to multiple
        :class:`Joinable` s.
        """
        num_joinables = 3
        num_allreduces = 1
        run_post_hooks = True

        expected_total = self.world_size * self.base_num_inputs
        for num_joined in range(1, self.rank + 1):
            expected_total += (self.world_size - num_joined) * self.offset
        # The expected total is now multiplied by a factor of `NUM_JOINABLES`
        expected_total *= num_joinables

        self._test_join_base(
            uneven_inputs=True,
            num_joinables=num_joinables,
            enable=True,
            throw_on_early_termination=False,
            num_allreduces=num_allreduces,
            run_post_hooks=run_post_hooks,
            expected_total=expected_total
        )

    @require_n_gpus_for_nccl_backend(
        WORLD_SIZE, BACKEND
    )
    def test_single_joinable_disable(self):
        r"""Tests ``enable=False`` for a single :class:`Joinable`."""
        num_joinables = 1
        num_allreduces = 1
        uneven_inputs = False
        enable = False
        run_post_hooks = False

        expected_total = self.world_size * self.base_num_inputs

        self._test_join_base(
            uneven_inputs=uneven_inputs,
            num_joinables=num_joinables,
            enable=enable,
            throw_on_early_termination=False,
            num_allreduces=num_allreduces,
            run_post_hooks=run_post_hooks,
            expected_total=expected_total
        )

    @require_n_gpus_for_nccl_backend(
        WORLD_SIZE, BACKEND
    )
    def test_multiple_joinable_disable(self):
        r"""
        Tests ``enable=False`` for multiple :class:`Joinable` s.

        This generalizes ``test_single_joinable_disable`` to multiple
        :class:`Joinable` s.
        """
        num_joinables = 3
        num_allreduces = 1
        uneven_inputs = False
        enable = False
        run_post_hooks = False

        expected_total = self.world_size * self.base_num_inputs * num_joinables

        self._test_join_base(
            uneven_inputs=uneven_inputs,
            num_joinables=num_joinables,
            enable=enable,
            throw_on_early_termination=False,
            num_allreduces=num_allreduces,
            run_post_hooks=run_post_hooks,
            expected_total=expected_total
        )

    @require_n_gpus_for_nccl_backend(
        WORLD_SIZE, BACKEND
    )
    def test_single_joinable_throw(self):
        r"""
        Tests ``throw_on_early_termination=True`` for a single
        :class:`Joinable`.
        """
        num_joinables = 1
        num_allreduces = 1
        throw_on_early_termination = True
        run_post_hooks = False

        self._test_join_base(
            uneven_inputs=True,
            num_joinables=num_joinables,
            enable=True,
            throw_on_early_termination=throw_on_early_termination,
            num_allreduces=num_allreduces,
            run_post_hooks=run_post_hooks,
            expected_total=None
        )

    @require_n_gpus_for_nccl_backend(
        WORLD_SIZE, BACKEND
    )
    def test_multiple_joinables_throw(self):
        r"""
        Tests ``throw_on_early_termination=True`` for multiple
        :class:`Joinable` s together.

        This generalizes ``test_single_joinable_throw`` to multiple
        :class:`Joinable` s.
        """
        num_joinables = 3
        num_allreduces = 1
        throw_on_early_termination = True
        run_post_hooks = False

        self._test_join_base(
            uneven_inputs=True,
            num_joinables=num_joinables,
            enable=True,
            throw_on_early_termination=throw_on_early_termination,
            num_allreduces=num_allreduces,
            run_post_hooks=run_post_hooks,
            expected_total=None
        )

    @require_n_gpus_for_nccl_backend(
        WORLD_SIZE, BACKEND
    )
    def test_join_kwargs(self):
        r"""
        Tests passing keyword arguments to the context manager.
        """
        num_joinables = 1
        num_allreduces = 2
        run_post_hooks = False

        expected_total = self.world_size * self.base_num_inputs
        for num_joined in range(1, self.rank + 1):
            expected_total += (self.world_size - num_joined) * self.offset
        # The expected total is now multiplied by a factor of `NUM_ALLREDUCES`
        expected_total *= num_allreduces

        self._test_join_base(
            uneven_inputs=True,
            num_joinables=num_joinables,
            enable=True,
            throw_on_early_termination=False,
            num_allreduces=num_allreduces,
            run_post_hooks=run_post_hooks,
            expected_total=expected_total
        )

if __name__ == "__main__":
    run_tests()
