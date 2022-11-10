# Owner(s): ["oncall: distributed"]

import itertools
import sys
from contextlib import suppress
from copy import deepcopy
from functools import partial
from typing import Any, Dict

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.fsdp import (
    CPUOffload,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    LocalStateDictConfig,
    MixedPrecision,
    ShardedStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp._shard_utils import _gather_state_dict
from torch.distributed.fsdp._unshard_param_utils import FLAT_PARAM
from torch.distributed.fsdp.wrap import enable_wrap, transformer_auto_wrap_policy, wrap
from torch.nn import Linear, Module, TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    _assert_module_states,
    _get_state_dict,
    _zero_model,
    CUDAInitMode,
    FSDPInitMode,
    FSDPTest,
    get_full_params,
    SkipModel,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

INNER_SHAPE = [4, 4]
OUTER_SHAPE = [4, 5]
BUFFER_SHAPE = [5, 5]

NON_ROOT_FSDP_PREFIX = "non_fsdp_lin"

_UNFLATTENED_STATE_DICT_IMPLS = ["state_dict", "sharded_state_dict"]
_FLATTENED_STATE_DICT_IMPLS = ["local_state_dict"]
_SUPPORTED_STATE_DICT_IMPLS = (
    _UNFLATTENED_STATE_DICT_IMPLS + _FLATTENED_STATE_DICT_IMPLS
)

STATE_DICT_MAPPING = {
    "state_dict": StateDictType.FULL_STATE_DICT,
    "local_state_dict": StateDictType.LOCAL_STATE_DICT,
    "sharded_state_dict": StateDictType.SHARDED_STATE_DICT,
}


class Model(Module):
    def __init__(self, wrap_fsdp, register_buffers=False, ignore_inner=False):
        super().__init__()
        self.inner = Linear(*INNER_SHAPE)
        if register_buffers:
            self.inner.register_buffer("buffer", torch.randn(BUFFER_SHAPE))
            self.inner.register_buffer(
                "non_persistent_buffer", torch.randn(BUFFER_SHAPE), persistent=False
            )
        if wrap_fsdp:
            self.inner = FSDP(
                self.inner, ignored_modules=([self.inner] if ignore_inner else [])
            )
        self.outer = Linear(*OUTER_SHAPE)
        if register_buffers:
            self.outer.register_buffer("buffer", torch.randn(BUFFER_SHAPE))
            self.outer.register_buffer(
                "non_persistent_buffer", torch.randn(BUFFER_SHAPE), persistent=False
            )

    def forward(self, x):
        # Forward twice.
        i = self.inner(x)
        j = self.inner(x)
        return self.outer(i + j)


class TestFSDPStateDict(FSDPTest):
    @property
    def world_size(self):
        return 2

    def _broadcast_state_dict(self, model, state_dict):
        if not isinstance(model, FSDP):
            # For non-FSDP root, some parts of the model state on rank 0 may
            # not be on CPU, so we move everything to CPU to avoid issues like:
            # https://github.com/pytorch/pytorch/issues/77113.
            for param_name, param in state_dict.items():
                if param.device != torch.device("cpu"):
                    state_dict[param_name] = param.cpu()

        olist = [state_dict if self.rank == 0 else None]
        dist.broadcast_object_list(olist)
        state_dict = olist[0]
        # Ensure that the state is on CUDA
        for param_name in state_dict.keys():
            state_dict[param_name] = state_dict[param_name].cuda()
        return state_dict

    def _compare_models(self, model, model_new, assert_fn, check_fp16=False):
        assert assert_fn in (self.assertEqual, self.assertNotEqual)
        with FSDP.summon_full_params(model):
            with FSDP.summon_full_params(model_new):
                params = list(model.parameters())
                params_new = list(model_new.parameters())
                # Regardless of `assert_fn`, the number of parameters should be
                # the same
                self.assertEqual(len(params), len(params_new))
                assert_fn(params, params_new)
                if check_fp16:
                    for tensor in model_new.parameters():
                        self.assertEqual(tensor.dtype, torch.float16)

    def _get_simple_nested_model(
        self, *fsdp_args, wrap=True, checkpoint_wrap=False, **fsdp_kwargs
    ):
        if wrap:
            lin1 = nn.Linear(10, 10, bias=False).cuda()
            lin2 = nn.Linear(10, 10, bias=False).cuda()
            if checkpoint_wrap:
                lin1 = checkpoint_wrapper(lin1)
                lin2 = checkpoint_wrapper(lin2)
            seq = nn.Sequential(FSDP(lin1, *fsdp_args, **fsdp_kwargs), lin2)
            if checkpoint_wrap:
                seq = checkpoint_wrapper(seq)
            model = FSDP(seq, *fsdp_args, **fsdp_kwargs)
        else:
            model = nn.Sequential(
                nn.Linear(10, 10, bias=False).cuda(),
                nn.Linear(10, 10, bias=False).cuda(),
            )
        return model

    def _get_simple_model(self, *fsdp_args, checkpoint_wrap=False, **fsdp_kwargs):
        lin = nn.Linear(10, 10, bias=False).cuda()
        if checkpoint_wrap:
            lin = checkpoint_wrapper(lin)
        model = FSDP(lin, *fsdp_args, **fsdp_kwargs)
        return model

    def _get_non_fsdp_root_module(self, *fsdp_args, wrap=True, **fsdp_kwargs):
        class FSDPContainer(nn.Module):
            def __init__(self, fsdp_1, fsdp_2):
                super().__init__()
                self.non_fsdp_lin = nn.Linear(10, 10, bias=False).cuda()
                self.fsdp_1 = fsdp_1
                self.fsdp_2 = fsdp_2

            def forward(self, x):
                x = self.non_fsdp_lin(x)
                x = self.fsdp_1(x)
                x = self.fsdp_2(x)
                return x

        return FSDPContainer(
            self._get_simple_nested_model(*fsdp_args, wrap=wrap, **fsdp_kwargs),
            self._get_simple_nested_model(*fsdp_args, wrap=wrap, **fsdp_kwargs),
        )

    def _get_state_dict_mgr(
        self,
        model: nn.Module,
        state_dict_type: str,
        state_dict_rank0_and_offload: bool,
    ):
        _state_dict_type = STATE_DICT_MAPPING[state_dict_type]
        if state_dict_type == "state_dict":
            config = FullStateDictConfig(
                rank0_only=state_dict_rank0_and_offload,
                offload_to_cpu=state_dict_rank0_and_offload,
            )
        elif state_dict_type == "local_state_dict":
            config = LocalStateDictConfig(
                offload_to_cpu=state_dict_rank0_and_offload,
            )
        elif state_dict_type == "sharded_state_dict":
            config = ShardedStateDictConfig(
                offload_to_cpu=state_dict_rank0_and_offload,
            )
        else:
            raise ValueError("Unsupported state_dict_type")
        return FSDP.state_dict_type(model, _state_dict_type, config)

    def _validate_state_dict_contents(
        self, model, fsdp_state_dict, state_dict_rank0_and_offload, ignore_keys=None
    ):
        if state_dict_rank0_and_offload:
            if self.rank == 0:
                self.assertNotEqual(fsdp_state_dict, {})
                for key, tensor in fsdp_state_dict.items():
                    if ignore_keys and key in ignore_keys:
                        continue
                    self.assertEqual(
                        tensor.device,
                        torch.device("cpu"),
                        f"{key} is unexpectedly on device {tensor.device}",
                    )
            else:
                # For non-FSDP roots, the non FSDP portion can still have parameters on rank 0,
                # so bypass the check for now.
                if isinstance(model, FSDP):
                    self.assertEqual(fsdp_state_dict, {})

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _UNFLATTENED_STATE_DICT_IMPLS)
    @parametrize(
        "checkpoint_wrap",
        ["source", "dest", "both", "source_after_wrap", "both_after_wrap"],
    )
    @parametrize("rank0_only_and_offload", [False, True])
    def test_fsdp_state_dict_with_activation_checkpoint(
        self, state_dict_type, checkpoint_wrap, rank0_only_and_offload
    ):
        """Tests saving the state dict, zeroing a target model's parameters, and
        loading the state dict, where the source and target models may have a
        checkpoint wrapper."""

        def apply_ac_to_linears(model) -> None:
            non_reentrant_wrapper = partial(
                checkpoint_wrapper,
                offload_to_cpu=False,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=lambda submodule: isinstance(submodule, nn.Linear),
            )

        for model_call in [
            partial(self._get_simple_model),
            partial(self._get_simple_nested_model),
        ]:
            model = model_call(checkpoint_wrap=(checkpoint_wrap in ("source", "both")))
            if checkpoint_wrap in ("source_after_wrap", "both_after_wrap"):
                apply_ac_to_linears(model)
            with self._get_state_dict_mgr(
                model, state_dict_type, rank0_only_and_offload
            ):
                state_dict = _gather_state_dict(_get_state_dict(model, False, False))
                # Possibly wrap new model in activation checkpoint wrapper to test save/
                # load with this wrapper
                model_new = model_call(
                    checkpoint_wrap=(checkpoint_wrap in ("dest", "both"))
                )
                if checkpoint_wrap == "both_after_wrap":
                    apply_ac_to_linears(model_new)
                _zero_model(model_new)
                self._compare_models(model, model_new, self.assertNotEqual)
                if rank0_only_and_offload:
                    state_dict = self._broadcast_state_dict(model, state_dict)
                # Would fail if checkpoint_wrapper did not correctly implement state_dict pre/post hooks
                model_new.load_state_dict(state_dict, strict=True)
                self._compare_models(model, model_new, self.assertEqual)

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _UNFLATTENED_STATE_DICT_IMPLS)
    @parametrize("rank0_only_and_offload", [False, True])
    def test_state_dict_with_manual_ac_wrapper(
        self,
        state_dict_type: str,
        rank0_only_and_offload: bool,
    ):
        """
        Tests saving and loading a state dict for a model manually wrapped with
        ``FSDP(CheckpointWrapper(module))``, where the ``CheckpointWrapper`` is
        wrapped before FSDP.

        TODO: Investigate why the test above does not cover everything in this
        test and de-duplicate afterwards.
        """
        if state_dict_type == "sharded_state_dict" and rank0_only_and_offload:
            return  # not supported
        model_ac = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
        )
        # Manually wrap FSDP without AC
        model_no_ac = deepcopy(model_ac)
        for i, layer in enumerate(model_no_ac.transformer.encoder.layers):
            model_no_ac.transformer.encoder.layers[i] = FSDP(layer)
        for i, layer in enumerate(model_no_ac.transformer.decoder.layers):
            model_no_ac.transformer.decoder.layers[i] = FSDP(layer)
        model_no_ac.transformer = FSDP(model_no_ac.transformer)

        # Manually wrap FSDP with AC as `FSDP(CheckpointWrapper(module))`
        for i, layer in enumerate(model_ac.transformer.encoder.layers):
            layer = checkpoint_wrapper(layer)
            model_ac.transformer.encoder.layers[i] = FSDP(layer)
        for i, layer in enumerate(model_ac.transformer.decoder.layers):
            layer = checkpoint_wrapper(layer)
            model_ac.transformer.decoder.layers[i] = FSDP(layer)
        model_ac.transformer = FSDP(model_ac.transformer)

        # Save, load, and compare the two models
        with self._get_state_dict_mgr(
            model_no_ac, state_dict_type, rank0_only_and_offload
        ):
            state_dict_no_ac = model_no_ac.state_dict()
        with self._get_state_dict_mgr(
            model_ac, state_dict_type, rank0_only_and_offload
        ):
            state_dict_ac = model_ac.state_dict()
        self.assertEqual(state_dict_ac.keys(), state_dict_no_ac.keys())
        if rank0_only_and_offload:
            state_dict_no_ac = self._broadcast_state_dict(model_no_ac, state_dict_no_ac)
            state_dict_ac = self._broadcast_state_dict(model_ac, state_dict_ac)
        with self._get_state_dict_mgr(
            model_no_ac, state_dict_type, rank0_only_and_offload
        ):
            model_no_ac.load_state_dict(state_dict_no_ac)
        with self._get_state_dict_mgr(
            model_ac, state_dict_type, rank0_only_and_offload
        ):
            model_ac.load_state_dict(state_dict_ac)
        self._compare_models(model_ac, model_no_ac, self.assertEqual)

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _SUPPORTED_STATE_DICT_IMPLS)
    def test_state_dict_with_shared_parameters(self, state_dict_type):
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerEncoderLayer, TransformerDecoderLayer},
        )
        model_creator = partial(
            TransformerWithSharedParams.init,
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_BEFORE,
            {"auto_wrap_policy": auto_wrap_policy},
        )

        fsdp_model = model_creator()
        with self._get_state_dict_mgr(fsdp_model, state_dict_type, False):
            state_dict = fsdp_model.state_dict()

        new_model = model_creator()
        _zero_model(new_model, zero_buffers=True)
        with self._get_state_dict_mgr(new_model, state_dict_type, False):
            new_model.load_state_dict(state_dict)

    @skip_if_lt_x_gpu(2)
    @parametrize("use_orig_params", [False, True])
    def test_state_dict_rank0_offload_save_load_flow(self, use_orig_params: bool):
        """Tests saving a model checkpoint only on rank 0 and loading it only
        on rank 0 with ``sync_module_states=True`` to emulate the workflow to
        avoid redundant CPU memory usage."""
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerEncoderLayer, TransformerDecoderLayer},
        )
        fsdp_kwargs = {
            "auto_wrap_policy": auto_wrap_policy,
            "use_orig_params": use_orig_params,
        }
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_BEFORE,
            fsdp_kwargs,
        )
        # Force model parameters and buffers to be nonzero
        with FSDP.summon_full_params(fsdp_model):
            for tensor in itertools.chain(
                fsdp_model.parameters(), fsdp_model.buffers()
            ):
                if torch.count_nonzero(tensor) == 0:
                    with torch.no_grad():
                        tensor.add_(
                            torch.tensor(1, dtype=tensor.dtype, device=tensor.device)
                        )
        with self._get_state_dict_mgr(fsdp_model, "state_dict", True):
            state_dict = deepcopy(_get_state_dict(fsdp_model))
        # Initialize a non-wrapped model on all ranks
        new_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
        )
        _zero_model(new_model, zero_buffers=True)
        # Only load the checkpoint on rank 0
        if self.rank == 0:
            new_model.load_state_dict(state_dict, strict=True)
        _assert_module_states(
            new_model,
            process_group=self.process_group,
            assert_fn=self.assertNotEqual,
        )
        # Broadcast the module states from rank 0 with `sync_module_states=True`
        new_fsdp_model = FSDP(
            new_model,
            device_id=torch.cuda.current_device(),
            auto_wrap_policy=auto_wrap_policy,
            sync_module_states=True,
        )
        # Check FSDP models are equal across ranks
        with FSDP.summon_full_params(new_fsdp_model):
            _assert_module_states(
                new_fsdp_model,
                process_group=self.process_group,
                assert_fn=self.assertEqual,
            )
        # Check FSDP models correctly loaded the checkpoint
        with FSDP.summon_full_params(fsdp_model):
            with FSDP.summon_full_params(new_fsdp_model):
                params = list(fsdp_model.parameters())
                params_new = list(new_fsdp_model.parameters())
                self.assertEqual(params, params_new)

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _SUPPORTED_STATE_DICT_IMPLS)
    @parametrize(
        "cpu_offload",
        [CPUOffload(offload_params=True), CPUOffload(offload_params=False)],
    )
    @parametrize("fp16", [True, False])
    @parametrize("state_dict_rank0_and_offload", [True, False])
    @parametrize("use_orig_params", [True, False])
    def test_basic_save_and_load_state_dict(
        self,
        state_dict_type: StateDictType,
        cpu_offload: bool,
        fp16: bool,
        state_dict_rank0_and_offload: bool,
        use_orig_params: bool,
    ):
        """
        Tests that we can save a state_dict and load it into a blank model
        with various configs such as fp16 and cpu offload and parameters
        match as expected.
        """
        if (state_dict_rank0_and_offload and state_dict_type != "state_dict") or (
            use_orig_params and state_dict_type not in _UNFLATTENED_STATE_DICT_IMPLS
        ):
            return  # not supported
        for model_call in [
            partial(
                self._get_non_fsdp_root_module,
                cpu_offload=cpu_offload,
                use_orig_params=use_orig_params,
            ),
            partial(
                self._get_simple_nested_model,
                cpu_offload=cpu_offload,
                use_orig_params=use_orig_params,
            ),
            partial(
                self._get_simple_model,
                cpu_offload=cpu_offload,
                use_orig_params=use_orig_params,
            ),
        ]:
            model = model_call()

            ctx = self._get_state_dict_mgr(
                model, state_dict_type, state_dict_rank0_and_offload
            )
            with ctx:
                fsdp_state_dict = _get_state_dict(
                    model, cpu_offload.offload_params, fp16
                )

            ignore_keys = [
                k for k in fsdp_state_dict.keys() if NON_ROOT_FSDP_PREFIX in k
            ]

            self._validate_state_dict_contents(
                model,
                fsdp_state_dict,
                state_dict_rank0_and_offload,
                ignore_keys=ignore_keys,
            )
            if fp16:
                # Verify fp16 is the type
                for tensor in fsdp_state_dict.values():
                    self.assertEqual(tensor.dtype, torch.float16)

            model_new = model_call()
            if not cpu_offload.offload_params:
                model_new = model_new.cuda()
            if fp16:
                model_new.half()

            # zero the model to ensure parameters are different.
            _zero_model(model_new)
            self._compare_models(model, model_new, self.assertNotEqual)

            # Verify parameters are the same in the new model.
            if state_dict_rank0_and_offload:
                fsdp_state_dict = self._broadcast_state_dict(model, fsdp_state_dict)
            with FSDP.state_dict_type(model_new, STATE_DICT_MAPPING[state_dict_type]):
                model_new.load_state_dict(fsdp_state_dict, strict=True)

            self._compare_models(model, model_new, self.assertEqual, check_fp16=fp16)

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _SUPPORTED_STATE_DICT_IMPLS)
    @parametrize("mixed_precision", [True, False])
    @parametrize("state_dict_rank0_and_offload", [True, False])
    def test_save_and_load_after_forward_state_dict(
        self, state_dict_type, mixed_precision, state_dict_rank0_and_offload
    ):
        """
        Test that saving after some training results in params being updated as
        expected.
        """
        if state_dict_rank0_and_offload and state_dict_type != "state_dict":
            return
        torch.cuda.set_device(self.rank)
        mixed_precision = (
            MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
            if mixed_precision
            else None
        )
        model = self._get_simple_nested_model(mixed_precision=mixed_precision)
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        initial_params = get_full_params(model)
        for _ in range(6):
            inp = torch.randn(1, 10, device=torch.cuda.current_device())
            output = model(*inp)
            loss = output.sum()
            expected_dtype = torch.float32 if mixed_precision is None else torch.float16
            self.assertEqual(expected_dtype, loss.dtype)
            loss.backward()
            optim.step()

        trained_params = get_full_params(model)
        # Ensure some training occured
        self.assertNotEqual(initial_params, trained_params)
        # Save a copy of the state_dict
        fsd_mgr = self._get_state_dict_mgr(
            model, state_dict_type, state_dict_rank0_and_offload
        )
        with fsd_mgr:
            state_dict = model.state_dict()
            if state_dict_type == "state_dict":
                state_dict = {k: v.clone() for k, v in state_dict.items()}
            else:
                for sharded_tensor in state_dict.values():
                    shard = sharded_tensor._local_shards[0]
                    shard.tensor = shard.tensor.clone().detach_()
        self._validate_state_dict_contents(
            model, state_dict, state_dict_rank0_and_offload
        )
        _zero_model(model)

        # Ensure checkpointed params have the full param dtype
        for tensor in state_dict.values():
            self.assertEqual(tensor.dtype, torch.float32)

        # Load state_dict into zeroed model
        if state_dict_rank0_and_offload:
            state_dict = self._broadcast_state_dict(model, state_dict)

        with FSDP.state_dict_type(model, STATE_DICT_MAPPING[state_dict_type]):
            model.load_state_dict(state_dict, strict=True)
        loaded_params = get_full_params(model)
        self.assertEqual(loaded_params, trained_params)

    def _initialize_model(
        self,
        wrap_fsdp: bool,
        wrap_ddp: bool = True,
        register_buffers: bool = False,
    ):
        # keep everything deterministic for input data
        torch.manual_seed(0)

        model = Model(wrap_fsdp, register_buffers=register_buffers).cuda()
        if wrap_fsdp:
            model = FSDP(model)
        elif wrap_ddp:
            model = DistributedDataParallel(model, device_ids=[self.rank])
        return model

    @staticmethod
    def _state_dict(model: Module, state_dict_type: str):
        try:
            enum_val = STATE_DICT_MAPPING[state_dict_type]
        except KeyError:
            raise ValueError(f"No state_dict type for {state_dict_type}")

        with FSDP.state_dict_type(model, enum_val):
            return model.state_dict()

    @staticmethod
    def _load_state_dict(
        model: Module, state_dict_type: str, state_dict: Dict[str, Any]
    ):
        try:
            enum_val = STATE_DICT_MAPPING[state_dict_type]
        except KeyError:
            raise ValueError(f"No state_dict for {state_dict_type}")

        with FSDP.state_dict_type(model, enum_val):
            return model.load_state_dict(state_dict, strict=True)

    def _dist_train(
        self, wrap_fsdp: bool, state_dict_type: str = "", move_to_cpu: bool = False
    ):
        # TODO: Move this test to common_fsdp.
        model = self._initialize_model(wrap_fsdp)
        optim = SGD(model.parameters(), lr=0.1)

        in_data = torch.rand(64, 4, requires_grad=True, device=torch.device("cuda"))
        for _ in range(3):
            out = model(in_data)
            out.sum().backward()
            optim.step()
            optim.zero_grad()

        if wrap_fsdp:
            blank_model = FSDP(Model(True).cuda())
            _zero_model(blank_model)
            state_dict = self._state_dict(model, state_dict_type)
            if move_to_cpu:
                for key in list(state_dict.keys()):
                    tensor = state_dict[key]
                    if isinstance(tensor, torch.Tensor):
                        state_dict[key] = tensor.cpu()
                    else:
                        shards = tensor.local_shards()
                        if shards:
                            shards[0].tensor = shards[0].tensor.cpu()

            self._load_state_dict(blank_model, state_dict_type, state_dict)
            return get_full_params(blank_model)
        else:
            return list(model.parameters())

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _SUPPORTED_STATE_DICT_IMPLS)
    def test_state_dict_save_load_flow(self, state_dict_type):
        for move_to_cpu in [True, False]:
            with self.subTest(move_to_cpu=move_to_cpu):
                fsdp_params = self._dist_train(
                    wrap_fsdp=True,
                    state_dict_type=state_dict_type,
                    move_to_cpu=move_to_cpu,
                )
                ddp_params = self._dist_train(wrap_fsdp=False)
                self.assertEqual(ddp_params, fsdp_params)

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _SUPPORTED_STATE_DICT_IMPLS)
    def test_fsdp_state_dict_keys(self, state_dict_type):
        state_dict = self._state_dict(self._initialize_model(True), state_dict_type)
        if state_dict_type == "local_state_dict":
            self.assertEqual(
                set([FLAT_PARAM, f"inner.{FLAT_PARAM}"]), state_dict.keys()
            )
        elif state_dict_type in ("state_dict", "sharded_state_dict"):
            # Keys should match local model.
            local_model = self._initialize_model(wrap_fsdp=False, wrap_ddp=False)
            local_keys = local_model.state_dict().keys()
            self.assertEqual(state_dict.keys(), local_keys)
        else:
            raise NotImplementedError(f"No test for {state_dict_type}!")

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _UNFLATTENED_STATE_DICT_IMPLS)
    @parametrize("state_dict_rank0_and_offload", [True, False])
    @parametrize("fsdp_root", [True, False])
    def test_state_dict_load_into_local_module(
        self,
        state_dict_type,
        state_dict_rank0_and_offload,
        fsdp_root,
    ):
        """
        Tests that FSDP's state_dict can be loaded into a local model.
        """
        if state_dict_rank0_and_offload and state_dict_type != "state_dict":
            return
        if not fsdp_root:
            model = self._get_non_fsdp_root_module()
        else:
            model = self._initialize_model(wrap_fsdp=True, register_buffers=True)
        optim = SGD(model.parameters(), lr=0.1)
        if not fsdp_root:
            in_data = torch.randn(
                1, 10, requires_grad=True, device=torch.device("cuda")
            )
        else:
            in_data = torch.rand(64, 4, requires_grad=True, device=torch.device("cuda"))
        for _ in range(3):
            out = model(in_data)
            out.sum().backward()
            optim.step()
            optim.zero_grad()

        with FSDP.summon_full_params(model):
            fsdp_params = deepcopy(list(model.parameters()))

        # get FSDP state_dict. Note that by default we return full_state_dict.
        sd_mgr = self._get_state_dict_mgr(
            model, state_dict_type, state_dict_rank0_and_offload
        )
        with sd_mgr:
            fsdp_state_dict = model.state_dict()

        ignore_keys = [k for k in fsdp_state_dict.keys() if NON_ROOT_FSDP_PREFIX in k]
        self._validate_state_dict_contents(
            model,
            fsdp_state_dict,
            state_dict_rank0_and_offload,
            ignore_keys=ignore_keys,
        )
        # Create zeroed local model
        if not fsdp_root:
            blank_local_model = self._get_non_fsdp_root_module(wrap=False)
        else:
            blank_local_model = self._initialize_model(
                wrap_fsdp=False, wrap_ddp=False, register_buffers=True
            )

        # Nothing should be FSDP
        for mod in blank_local_model.modules():
            self.assertFalse(isinstance(mod, FSDP))

        for param in blank_local_model.parameters():
            with torch.no_grad():
                param.zero_()

        fsdp_state_dict = _gather_state_dict(fsdp_state_dict)

        # Load fsdp's full state dict into the local and verify params are as
        # expected.
        if state_dict_rank0_and_offload:
            fsdp_state_dict = self._broadcast_state_dict(model, fsdp_state_dict)

        # if self.rank == 0:
        blank_local_model.load_state_dict(fsdp_state_dict, strict=True)
        local_params = list(blank_local_model.parameters())
        for fsdp_param, local_param in zip(fsdp_params, local_params):
            self.assertEqual(fsdp_param, local_param)

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _SUPPORTED_STATE_DICT_IMPLS)
    @parametrize("double_nest", [True])
    def test_state_dict_skip_module(self, state_dict_type, double_nest):
        torch.cuda.set_device(self.rank)

        def _create_module(wrap_fsdp=True):
            LINEAR_SKIP = "linear_skip"
            ctx = enable_wrap(wrapper_cls=FSDP) if wrap_fsdp else suppress()
            with ctx:
                module = SkipModel(double_nest=double_nest)
                # Full name of linear_skip param tensors in SkipModel, as would be
                # stored in checkpoint.
                linear_skip_tensor_names = [
                    k
                    for k in dict(module.named_parameters()).keys()
                    if LINEAR_SKIP in k
                ]
                # skip SkipModule
                linear_skip = getattr(module, LINEAR_SKIP)
                delattr(module, LINEAR_SKIP)
                # Wrap FSDP
                fsdp = wrap(module)
                # reattach
                setattr(module, LINEAR_SKIP, linear_skip)
                return fsdp, linear_skip_tensor_names

        fsdp, linear_skip_tensor_names = _create_module()
        # Run a forward pass
        inp = torch.randn((1, 10), device=torch.cuda.current_device())
        loss = fsdp(inp)
        loss.sum().backward()

        with FSDP.state_dict_type(fsdp, STATE_DICT_MAPPING[state_dict_type]):
            state_dict = fsdp.state_dict()
        if self.rank == 0 and state_dict_type != "local_state_dict":
            sd_keys = list(state_dict.keys())
            expected = list(SkipModel(double_nest=False).state_dict().keys())
            self.assertEqual(sorted(sd_keys), sorted(expected))
            # TODO: parameters in linear_skip_tensor_names should not be handled
            # by FSDP.state_dict(). Have a check once this is implemented in
            # FSDP.state_dict().

        # Check that it can be loaded into FSDP.
        new_fsdp, _ = _create_module()
        _zero_model(new_fsdp)
        for (p1, p2) in zip(fsdp.parameters(), new_fsdp.parameters()):
            self.assertNotEqual(p1, p2)
        with FSDP.state_dict_type(new_fsdp, STATE_DICT_MAPPING[state_dict_type]):
            if state_dict_type != "local_state_dict":
                # FlatParameter has not supported deepcopy yet.
                state_dict = deepcopy(state_dict)
            new_fsdp.load_state_dict(state_dict, strict=True)
        for (p1, p2) in zip(fsdp.parameters(), new_fsdp.parameters()):
            self.assertEqual(p1, p2)

        # Test that the checkpoint can be loaded into a local model.
        local, _ = _create_module(wrap_fsdp=False)
        for param in local.parameters():
            with torch.no_grad():
                param.zero_()

        with fsdp.summon_full_params(fsdp):
            for (p1, p2) in zip(fsdp.parameters(), local.parameters()):
                self.assertNotEqual(p1, p2)

        if state_dict_type == "local_state_dict":
            return
        state_dict = _gather_state_dict(state_dict)
        with fsdp.summon_full_params(fsdp):
            if self.rank == 0:
                local.load_state_dict(state_dict, strict=True)
                for (p1, p2) in zip(fsdp.parameters(), local.parameters()):
                    self.assertEqual(p1, p2)

    @skip_if_lt_x_gpu(2)
    def test_wrong_state_dict_config(self):
        model = FSDP(Model(wrap_fsdp=True).cuda())
        with self.assertRaisesRegex(RuntimeError, "Expected state_dict_config of type"):
            with model.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, LocalStateDictConfig()
            ):
                pass

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _UNFLATTENED_STATE_DICT_IMPLS)
    @parametrize("prefix", [True, False])
    @parametrize("ignore_inner", [True, False])
    def test_state_dict_with_ignored_modules(
        self, state_dict_type, prefix, ignore_inner
    ):
        # Initialize an FSDP-wrapped model with an ignored module that includes
        # both parameters and a buffer
        model = Model(
            wrap_fsdp=True, register_buffers=True, ignore_inner=ignore_inner
        ).cuda()
        ignored_modules = [model.outer]
        ignored_tensor_to_tensor_name = {
            model.outer.bias: "outer.bias",
            model.outer.weight: "outer.weight",
        }
        if ignore_inner:
            ignored_tensor_to_tensor_name = {
                **ignored_tensor_to_tensor_name,
                model.inner.bias: "inner.bias",
                model.inner.weight: "inner.weight",
            }
        # Note that when model.inner is not ignored this test also ensures
        # non-ignored buffers are not cloned.
        buffer_to_buffer_name = {
            model.inner.buffer: "inner.buffer",
            model.outer.buffer: "outer.buffer",
        }
        fsdp_model = FSDP(model, ignored_modules=ignored_modules)
        prefix_str = "foo." if prefix else ""
        with FSDP.state_dict_type(fsdp_model, STATE_DICT_MAPPING[state_dict_type]):
            sd1 = _gather_state_dict(fsdp_model.state_dict(prefix=prefix_str))
        with FSDP.summon_full_params(fsdp_model):
            fsdp_params = deepcopy(list(fsdp_model.parameters()))
        # Check that the ignored parameters and all buffers are not cloned
        for tensor, tensor_name in {
            **ignored_tensor_to_tensor_name,
            **buffer_to_buffer_name,
        }.items():
            prefixed_tensor_name = f"{prefix_str}{tensor_name}"
            self.assertTrue(prefixed_tensor_name in sd1)
            self.assertEqual(
                tensor.data_ptr(),
                sd1[prefixed_tensor_name].data_ptr(),
                f"{prefixed_tensor_name}",
            )
        # Check that the state dict can be loaded into a non-wrapped version of
        # the model
        nonwrapped_model = Model(wrap_fsdp=False, register_buffers=True).cuda()
        for param in nonwrapped_model.parameters():
            with torch.no_grad():
                param.zero_()

        to_load = {k[len(prefix_str) :]: v for k, v in sd1.items()}
        nonwrapped_model.load_state_dict(to_load, strict=True)
        local_params = list(nonwrapped_model.parameters())
        for fsdp_param, local_param in zip(fsdp_params, local_params):
            self.assertEqual(fsdp_param, local_param)
        # Check that if we save a state dict again, the ignored parameters and
        # buffer still have the same data pointer
        with FSDP.state_dict_type(fsdp_model, STATE_DICT_MAPPING[state_dict_type]):
            sd2 = fsdp_model.state_dict(prefix=prefix_str)
        for tensor, tensor_name in {
            **ignored_tensor_to_tensor_name,
            **buffer_to_buffer_name,
        }.items():
            prefixed_tensor_name = f"{prefix_str}{tensor_name}"
            self.assertTrue(prefixed_tensor_name in sd2)
            self.assertEqual(tensor.data_ptr(), sd2[prefixed_tensor_name].data_ptr())
            self.assertEqual(
                sd1[prefixed_tensor_name].data_ptr(),
                sd2[prefixed_tensor_name].data_ptr(),
            )

    @skip_if_lt_x_gpu(2)
    def test_state_dict_type(self):
        module = SkipModel(double_nest=True)
        with enable_wrap(wrapper_cls=FSDP):
            fsdp = wrap(module)
        with FSDP.state_dict_type(fsdp, StateDictType.LOCAL_STATE_DICT):
            pass
        for module in FSDP.fsdp_modules(fsdp):
            self.assertEqual(module._state_dict_type, StateDictType.FULL_STATE_DICT)


instantiate_parametrized_tests(TestFSDPStateDict)

if __name__ == "__main__":
    run_tests()
