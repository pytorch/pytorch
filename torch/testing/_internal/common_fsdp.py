import logging
import os
import tempfile
import functools
from statistics import mean
import itertools
import sys
import unittest
from unittest import mock
import gc
import contextlib
import subprocess

import torch
import torch.nn as nn
from torch.distributed import rpc
import torch.multiprocessing as mp
from torch import Tensor

from torch.distributed._fsdp import FullyShardedDataParallel
from torch.distributed._fsdp.fully_sharded_data_parallel import TrainingState

from typing import Callable, List, Any, Optional, Union, Generator, Dict, Tuple

keys = ["reshard_after_forward", "mixed_precision", "flatten_parameters"]
CONFIG_OPTIONS = [dict(zip(keys, config)) for config in itertools.product([True, False], repeat=len(keys))]

class DummyProcessGroup:
    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


class DeviceAndTypeCheckModule(nn.Module):
    """A simple module for checking Tensor devices and dtypes."""

    def __init__(
        self,
        expected_input_dtype: Optional[torch.dtype] = None,
        expected_input_device: Optional[torch.device] = None,
        expected_param_dtype: Optional[torch.dtype] = None,
        expected_param_device: Optional[torch.device] = None,
        expected_loss_dtype: Optional[torch.dtype] = None,
        expected_loss_device: Optional[torch.device] = None,
        expected_buffer_dtype: Optional[torch.device] = None,
    ):
        super().__init__()
        self.expected_input_dtype = expected_input_dtype
        self.expected_input_device = expected_input_device
        self.expected_param_dtype = expected_param_dtype
        self.expected_param_device = expected_param_device
        self.expected_loss_dtype = expected_loss_dtype
        self.expected_loss_device = expected_loss_device
        self.expected_buffer_dtype = expected_buffer_dtype

        self.linear = nn.Linear(5, 5)
        self.register_buffer("buffer", torch.rand((5,)))

    def _check(
        self,
        key: str,
        x: Union[torch.device, torch.dtype],
        expected: Union[Optional[torch.device], Optional[torch.dtype]],
    ) -> None:
        assert expected in {None, x}, f"{key} ({x}) != expected ({expected})"

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        x = input[0]
        self._check("input.dtype", x.dtype, self.expected_input_dtype)
        self._check("input.device", x.device, self.expected_input_device)

        param = self.linear.weight
        self._check("param.dtype", param.dtype, self.expected_param_dtype)
        self._check("param.device", param.device, self.expected_param_device)
        self._check("buffer.dtype", self.buffer.dtype, self.expected_buffer_dtype)  # type: ignore[arg-type]
        x = x + self.buffer
        loss = (self.linear(x) + self.buffer).sum()
        self._check("loss.dtype", loss.dtype, self.expected_loss_dtype)
        self._check("loss.device", loss.device, self.expected_loss_device)

        return loss


@functools.lru_cache()
def get_cycles_per_ms() -> float:
    """Measure and return approximate number of cycles per millisecond for torch.cuda._sleep
    Copied from: github.com/pytorch/pytorch/blob/master/test/test_cuda.py
    """

    def measure() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda._sleep(1000000)
        end.record()
        end.synchronize()
        cycles_per_ms = 1000000 / start.elapsed_time(end)
        return cycles_per_ms

    # Get 10 values and remove the 2 max and 2 min and return the avg.
    # This is to avoid system disturbance that skew the results, e.g.
    # the very first cuda call likely does a bunch of init, which takes
    # much longer than subsequent calls.
    #
    # Tested on both Tesla V100, Quadro GP100, Titan RTX, RTX 3090 GPUs
    # and seems to return stable values. Therefore, we enable caching
    # using lru_cache decorator above.
    num = 10
    vals = []
    for _ in range(num):
        vals.append(measure())
    vals = sorted(vals)
    return mean(vals[2 : num - 2])


def dist_init(rank: int, world_size: int, filename: str, filename_rpc: str = "") -> bool:
    """
    Initialize torch distributed, based on a temporary file shared across ranks, which makes it possible for unrelated
    tests to be run concurrently.
    Return false if not enough GPUs present in the system.
    .. warning: This limits the usecase to all ranks being on the same node
    """
    print(f"dist init r={rank}, world={world_size}")

    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    url = "file://" + filename

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl" and torch.cuda.device_count() < world_size:
        logging.warning("Requested world size cannot be reached on this machine, not enough GPUs")
        return False

    torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size, init_method=url)

    if torch.cuda.is_available() and torch.cuda.device_count():
        torch.cuda.set_device(rank % torch.cuda.device_count())

    return True


def teardown() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def get_world_sizes() -> List[int]:
    limit = torch.cuda.device_count()
    return [x for x in [1, 2, 4, 8] if x <= limit]


def rmf(filename: str) -> None:
    """Remove a file like rm -f."""
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


def spawn_for_all_world_sizes(test_func: Callable, world_sizes: List[int] = get_world_sizes(), args: Any = None) -> None:
    if args is None:
        args = []
    for world_size in world_sizes:
        _, filename = tempfile.mkstemp()
        _, filename_rpc = tempfile.mkstemp()

        try:
            # (lefaudeux) Let mp handle the process joining, join=False and handling context has
            # been unstable in the past.
            mp.spawn(test_func, args=(world_size, filename, filename_rpc, *args), nprocs=world_size, join=True)
        finally:
            rmf(filename)
            rmf(filename_rpc)


def objects_are_equal(a: Any, b: Any, raise_exception: bool = False, dict_key: Optional[str] = None) -> bool:
    """
    Test that two objects are equal. Tensors are compared to ensure matching
    size, dtype, device and values.
    """
    if type(a) is not type(b):
        if raise_exception:
            raise ValueError(f"type mismatch {type(a)} vs. {type(b)}")
        return False
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            if raise_exception:
                raise ValueError(f"keys mismatch {a.keys()} vs. {b.keys()}")
            return False
        for k in a.keys():
            if not objects_are_equal(a[k], b[k], raise_exception, k):
                return False
        return True
    elif isinstance(a, (list, tuple, set)):
        if len(a) != len(b):
            if raise_exception:
                raise ValueError(f"length mismatch {len(a)} vs. {len(b)}")
            return False
        return all(objects_are_equal(x, y, raise_exception) for x, y in zip(a, b))
    elif torch.is_tensor(a):
        try:
            # assert_allclose doesn't strictly test shape, dtype and device
            shape_dtype_device_match = a.size() == b.size() and a.dtype == b.dtype and a.device == b.device
            if not shape_dtype_device_match:
                if raise_exception:
                    msg = f"sizes: {a.size()} vs. {b.size()}, "
                    msg += f"types: {a.dtype} vs. {b.dtype}, "
                    msg += f"device: {a.device} vs. {b.device}"
                    raise AssertionError(msg)
                else:
                    return False
            # assert_allclose.
            torch.testing.assert_allclose(a, b)
            return True
        except (AssertionError, RuntimeError) as e:
            if raise_exception:
                if dict_key and isinstance(e, AssertionError):
                    # Add dict key to the assertion error.
                    msg = e.args[0]
                    new_msg = f"For dict key '{dict_key}': {msg}"
                    raise AssertionError(new_msg) from None
                else:
                    raise e
            else:
                return False
    else:
        return a == b


def dump_all_tensors(rank: int) -> None:
    """Useful tool for debugging memory issues from the python side."""
    if rank != 0:
        return
    for obj in gc.get_objects():
        try:
            ttype = str(type(obj))
            if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                print(ttype, obj.shape, obj.dtype, obj.device, obj.storage().size())
        except Exception:
            pass
    print(torch.cuda.memory_summary())


@contextlib.contextmanager
def temp_files_ctx(num: int) -> Generator:
    """ A context to get tempfiles and ensure they are cleaned up. """
    files = [tempfile.mkstemp()[1] for _ in range(num)]

    try:
        yield tuple(files)
    finally:
        # temp files could have been removed, so we use rmf.
        for name in files:
            rmf(name)


def init_and_run(fn, args, rank, world_size, filename, filename_rpc):
    dist_init(rank, world_size, filename, filename_rpc)
    group = torch.distributed.new_group()
    fn(rank, group, *args)


def spawn_and_init(fn, args=None, **spawn_kwargs):
    if args is None:
        args = ()

    run_fn = functools.partial(init_and_run, fn, args)
    spawn_for_all_world_sizes(run_fn, **spawn_kwargs)


@contextlib.contextmanager
def in_temporary_directory() -> Generator:
    """
    Context manager to create a temporary direction and remove
    it at the end of the context
    """
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        try:
            yield temp_dir
        finally:
            os.chdir(old_cwd)


class TransformerWithSharedParams(nn.Module):
    def __init__(self, group, *unused_args, d_vocab=23, d_model=16, add_bn=True, **unused_kwargs):
        super().__init__()
        self.rank = group.rank()
        self.world_size = group.size()
        torch.manual_seed(0)  # keep everything deterministic
        assert d_vocab >= 12  # we use torch.arange(12) as input
        self.embed_tokens = nn.Embedding(d_vocab, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=8, dropout=0.1,
        )
        self.output_proj = nn.Linear(d_model, d_vocab)

        # share the embedding and output projection weights
        self.output_proj.weight = self.embed_tokens.weight
        self.register_buffer("vocab_bias", self.embed_tokens.weight.new_ones((d_model,)))
        self.register_buffer("long_buffer", torch.zeros_like(self.vocab_bias, dtype=torch.long))  # type: ignore[arg-type]

        self.bs = 2
        self.bn = torch.nn.BatchNorm1d(self.bs) if add_bn else torch.nn.Identity()

    def get_input(self, device):
        torch.manual_seed(1 + self.rank)  # keep everything deterministic
        src = torch.arange(12, device=device).view(6, self.bs)  # T x B
        tgt = torch.arange(self.bs * 4, device=device).view(4, self.bs)  # T x B
        return (src, tgt)

    def forward(self, src_ids, tgt_ids):
        src = self.embed_tokens(src_ids)
        src = src + self.vocab_bias + self.long_buffer.type_as(src)  # type: ignore[operator]
        tgt = self.embed_tokens(tgt_ids)
        tgt = self.bn(tgt)
        x = self.transformer(src, tgt)
        return self.output_proj(x)

    def get_loss(self, input, output):
        _, tgt = input
        return nn.functional.cross_entropy(output.view(-1, output.size(-1)), tgt.view(-1), reduction="sum")

    def run_backward(self, loss):
        loss.backward()


class NestedWrappedModule(nn.Module):
    # def __init__(self, group, wrapper_config, wrap_everything=False, checkpoint=False):
    def __init__(self, group, wrapper_config, wrap_everything=False):
        super().__init__()
        self.rank = group.rank()
        self.world_size = group.size()
        self.wrapper_config = wrapper_config

        def _maybe_wrap(layer):
            if wrapper_config is not None:
                return FullyShardedDataParallel(layer, group, **wrapper_config)
            return layer

        torch.manual_seed(0)  # keep everything deterministic
        self.module = nn.Sequential(
            nn.Linear(8, 4),
            _maybe_wrap(nn.Sequential(_maybe_wrap(nn.Linear(4, 16)), nn.Linear(16, 16),)),
            _maybe_wrap(nn.Linear(16, 4)),
            nn.Linear(4, 8),
        )

        # Wrap all modules triggers a corner case where root FSDP doesn't have any params.
        # Test it with checkpoint_wrapper as well to validate final backward callback
        # is queued correctly when root FSDP does not have any params and every layer is
        # wrapped as FSDP(checkpoint(module)).
        if wrap_everything:
            # if checkpoint:
            #    self.module = nn.Sequential(
            #        _maybe_wrap(checkpoint_wrapper(nn.Linear(8, 4))),
            #        _maybe_wrap(checkpoint_wrapper(nn.Linear(4, 16))),
            #        _maybe_wrap(checkpoint_wrapper(nn.Linear(16, 4))),
            #        _maybe_wrap(checkpoint_wrapper(nn.Linear(4, 8))),
            #    )
            # else:
            self.module = nn.Sequential(
                _maybe_wrap(nn.Linear(8, 4)),
                _maybe_wrap(nn.Linear(4, 16)),
                _maybe_wrap(nn.Linear(16, 4)),
                _maybe_wrap(nn.Linear(4, 8)),
            )

    def get_input(self, device):
        torch.manual_seed(1 + self.rank)  # keep everything deterministic
        return (torch.rand(4, 8, device=device),)

    def forward(self, x):
        return self.module(x)

    def get_loss(self, input, output):
        loss = output.sum()
        return loss

    def run_backward(self, loss):
        loss.backward()


class MixtureOfExperts(NestedWrappedModule):
    # def __init__(self, group, wrapper_config, checkpoint_act=False, delay_before_free_ms=0):
    def __init__(self, group, wrapper_config, delay_before_free_ms=0):
        super().__init__(group, wrapper_config)
        self.group = group
        self.delay_before_free_ms = delay_before_free_ms

        # "expert" params are different on each rank
        torch.manual_seed(42 + group.rank())
        d_expert = 23
        d_shared = 12
        d_input = 8
        expert = nn.Linear(d_expert, d_shared)

        self.num_expert_params = sum([p.numel() for p in expert.parameters()])
        for p in expert.parameters():
            p.expert = True  # type: ignore[attr-defined]

        # everything else is shared
        torch.manual_seed(0)

        shared = nn.Linear(d_shared, d_expert)

        # if checkpoint_act:
        #    expert = checkpoint_wrapper(expert)
        #    shared = checkpoint_wrapper(shared)

        if wrapper_config is not None:
            # we create a process group of size 1 for the expert params
            expert_group = torch.distributed.new_group([group.rank()])  # world size 1 means no shard
            expert = FullyShardedDataParallel(expert, expert_group, **wrapper_config)  # type: ignore[assignment]

            shared = FullyShardedDataParallel(shared, group, **wrapper_config)  # type: ignore[assignment]

        self.module = nn.Sequential(nn.Linear(d_input, d_shared), shared, expert, nn.Linear(d_shared, d_input))

    def forward(self, x):
        if self.delay_before_free_ms > 0:
            expert = self.module[2]
            if isinstance(expert, FullyShardedDataParallel):
                orig_free_full_params = self.module[2]._free_full_params

                def _free_full_params_with_delay(*args):
                    torch.cuda._sleep(int(self.delay_before_free_ms * get_cycles_per_ms()))
                    return orig_free_full_params(*args)

                assert hasattr(expert, "_free_full_params")
                with mock.patch.object(expert, "_free_full_params", _free_full_params_with_delay):
                    return self.module(x)

        return self.module(x)

    def run_backward(self, loss):
        loss.backward()

        # manually reduce gradients if not wrapped in FullyShardedDataParallel
        if self.wrapper_config is None:
            with torch.no_grad():
                for p in self.parameters():
                    if hasattr(p, "expert"):
                        continue  # these params don't need grad reduction
                    p.grad.data.div_(self.world_size)
                    torch.distributed.all_reduce(p.grad.data, group=self.group)


# How to use remote-pdb: https://gist.github.com/sshleifer/9d43351957179c13606e015b072927d4
# All helper functions called by spawn must be either @classmethod, @staticmethod
class DistributedTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available, skipping test")
        if sys.platform == "win32":
            raise unittest.SkipTest("NCCL doesn't support Windows, skipping test")
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("distributed tests require 2+ GPUs, skipping")

    def tearDown(self):
        teardown()

    @staticmethod
    def _train_for_several_steps(model, num_steps, autocast, lr=0.01, norm_type=None):
        model_device = next(model.parameters()).device
        # use SGD with momentum instead of Adam, since Adam is scale invariant
        # and this makes it bad for tests
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        for _ in range(num_steps):
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=autocast):
                # Inputs always cuda regardless of move_grads_cpu, or model.device
                input = model.module.get_input(torch.device("cuda"))
                output = model(*input)
                loss = model.module.get_loss(input, output).to(model_device)
            assert loss.dtype == torch.float32
            model.module.run_backward(loss)
            if norm_type is not None:
                clip_norm = 0.3
                if isinstance(model, FullyShardedDataParallel):
                    model.clip_grad_norm_(clip_norm, norm_type)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, norm_type)
            optim.step()
        if isinstance(model, FullyShardedDataParallel):
            model.assert_state(TrainingState.IDLE)
        return loss.detach()

    @staticmethod
    def get_wrapped_model(group, cuda_first=False, config=None, **model_kwargs) -> FullyShardedDataParallel:
        if config is None:
            config = {}
        if cuda_first:
            model = FullyShardedDataParallel(TransformerWithSharedParams(group, **model_kwargs).cuda(), group, **config)
        else:
            model = FullyShardedDataParallel(TransformerWithSharedParams(group, **model_kwargs), group, **config).cuda()
        return model

    @classmethod
    def _test_identical_outputs(
        cls, model_init_fn, config, rank, group, num_steps=2, use_cuda=True, lr=0.01, ref_ddp_fn=None, norm_type=2,
    ):
        if config.get("mixed_precision", False):
            autocast = True
            # Force the compute dtype to be torch.float32 so that we get
            # identical results as PyTorch DDP when using autocast. Note that
            # this will cause the all-gather to happen in FP32, which is slower
            # than necessary in most cases.
            config["compute_dtype"] = torch.float32
        else:
            autocast = False

        # Establish reference behavior with PyTorch DDP (+ optionally autocast).
        model = model_init_fn(group=group, wrapper_config=None).cuda()
        if ref_ddp_fn is None:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[rank], output_device=rank, process_group=group
            )
        else:
            model = ref_ddp_fn(model, group)
        ref_loss = cls._train_for_several_steps(model, num_steps, autocast, lr=lr, norm_type=norm_type)
        ref_state_dict = model.module.state_dict()
        if config.get("cpu_offload", False):
            ref_loss = ref_loss.cpu()
            for k in ref_state_dict.keys():
                ref_state_dict[k] = ref_state_dict[k].cpu()

        # Confirm we get the same behavior using FullyShardedDataParallel.
        model = FullyShardedDataParallel(model_init_fn(group=group, wrapper_config=config), group, **config)
        if use_cuda:
            model = model.cuda()
        else:
            assert next(model.parameters()).device == torch.device("cpu")
        shard_loss = cls._train_for_several_steps(model, num_steps, autocast, lr=lr, norm_type=norm_type)
        shard_state_dict = model.state_dict()

        try:
            torch.testing.assert_allclose(ref_loss, shard_loss)
            assert objects_are_equal(ref_state_dict, shard_state_dict, raise_exception=True)
        except (AssertionError, RuntimeError) as e:
            raise Exception(f"FullyShardedDataParallel didn't match PyTorch DDP using config: {config}\n\n {e}")


def state_dict_norm(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute the norm from a state_dict for simple comparison."""
    norm = torch.zeros(1)
    for v in state.values():
        if not v.is_floating_point():
            v = v.float()
        norm += v.norm()
    return norm


def torch_cuda_version(compiled: bool = False) -> Tuple[int, ...]:
    if compiled:
        numbering = torch.version.cuda.split(".")[:2]  # type: ignore[attr-defined]
    else:
        def get_smi_ver() -> str:  # type: ignore[return]
            """Get CUDA version from nvidia-smi"""
            for line in subprocess.check_output("nvidia-smi".split()).decode("utf-8").split("\n"):
                if "CUDA Version" in line:
                    res = line.split()[8]
                    assert res.startswith("10.") or res.startswith("11."), res
                    return res
            AssertionError("can not get cuda version")

        _smi_ver = get_smi_ver()
        numbering = _smi_ver.split(".")[:2]
    return tuple(int(n) for n in numbering)
