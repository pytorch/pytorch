# Owner(s): ["module: dynamo"]
"""FakeQuantize and the observers under torch.compile and torch.export.

observer_enabled and fake_quant_enabled are buffers, so the compiled forward has to apply
them without Python control flow. It only does that for the observers it knows the state of
and for qparams that already exist; every other module keeps running the eager forward,
which graph-breaks.
"""

import copy
import functools
import unittest.mock
import warnings

import torch
import torch._dynamo.exc
import torch._dynamo.test_case
import torch.nn as nn
from torch._dynamo.testing import CompileCounterWithBackend
from torch._dynamo.utils import counters
from torch.ao.quantization import (
    DeQuantStub,
    disable_observer,
    FakeQuantize,
    FusedMovingAvgObsFakeQuantize,
    get_default_qat_qconfig,
    get_default_qat_qconfig_mapping,
    get_default_qconfig_mapping,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
    prepare_qat,
    QuantStub,
)
from torch.ao.quantization.quantize_fx import prepare_qat_fx
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    skipCPUIf,
    skipCUDAIf,
)
from torch.testing._internal.common_utils import HardwareClassification, parametrize
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA_AND_TRITON


_FAKE_QUANT_CONFIGS = (
    "per_tensor_min_max",
    "per_tensor_moving_average",
    "per_channel_min_max",
    "per_channel_moving_average",
    "fixed_qparams",
)
_COMPILE_BACKENDS = ("eager", "aot_eager")
_PER_CHANNEL = {
    "qscheme": torch.per_channel_symmetric,
    "dtype": torch.qint8,
    "quant_min": -128,
    "quant_max": 127,
    "ch_axis": 1,
}


def _both_nested_graph_break_modes(test):
    # torch._dynamo.test_case.TestCase forces nested_graph_breaks=True, which is not the
    # default, and the graph breaks of these tests behave differently in the two modes
    @functools.wraps(test)
    def wrapper(self, *args, **kwargs):
        for nested_graph_breaks in (False, True):
            with self.subTest(nested_graph_breaks=nested_graph_breaks):
                torch._dynamo.reset()
                with torch._dynamo.config.patch(
                    nested_graph_breaks=nested_graph_breaks
                ):
                    test(self, *args, **kwargs)

    return wrapper


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.l1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(16, 4)
        self.dequant = DeQuantStub()

    def forward(self, x):
        return self.dequant(self.l2(self.relu(self.l1(self.quant(x)))))


class _FakeQuantizeCompileHelpers:
    def _make_fake_quant(self, config, device):
        if config == "fixed_qparams":
            from torch.ao.quantization.fake_quantize import (
                default_fixed_qparams_range_0to1_fake_quant,
            )

            return default_fixed_qparams_range_0to1_fake_quant().to(device)
        observer, kwargs = {
            "per_tensor_min_max": (MinMaxObserver, {}),
            "per_tensor_moving_average": (MovingAverageMinMaxObserver, {}),
            "per_channel_min_max": (PerChannelMinMaxObserver, _PER_CHANNEL),
            "per_channel_moving_average": (
                MovingAveragePerChannelMinMaxObserver,
                _PER_CHANNEL,
            ),
        }[config]
        return FakeQuantize(observer=observer, **kwargs).to(device)

    def _fused(self, per_channel, device):
        if per_channel:
            fake_quant = FusedMovingAvgObsFakeQuantize(
                observer=MovingAveragePerChannelMinMaxObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,
                ch_axis=0,
            )
        else:
            fake_quant = FusedMovingAvgObsFakeQuantize(
                observer=MovingAverageMinMaxObserver,
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
            )
        return fake_quant.to(device)

    def _prepared_model(self, kind, device):
        torch.manual_seed(0)
        model = _Mlp().train()
        if kind == "fx_fbgemm":
            model = prepare_qat_fx(
                model, get_default_qat_qconfig_mapping("fbgemm"), (torch.randn(2, 8),)
            )
        else:
            # "version=0" has the non-fused per-channel weight fake-quant
            model.qconfig = get_default_qat_qconfig(
                "fbgemm", version=0 if kind == "eager_version0" else 1
            )
            model = prepare_qat(model)
        return model.to(device)

    def _assert_grads_equal(self, mod, ref):
        for (name, param), (_, ref_param) in zip(
            mod.named_parameters(), ref.named_parameters()
        ):
            self.assertEqual(param.grad, ref_param.grad, msg=name)

    def _check_fullgraph(self, mod, make_input, compile_backend, warm_up=False):
        if warm_up:
            mod(make_input())
        torch._dynamo.reset()
        ref = copy.deepcopy(mod)
        compiled = torch.compile(mod, backend=compile_backend, fullgraph=True)
        for _ in range(2):
            x_ref = make_input().requires_grad_()
            x_compiled = x_ref.detach().clone().requires_grad_()
            expected, actual = ref(x_ref), compiled(x_compiled)
            self.assertEqual(actual, expected)
            expected.sum().backward()
            actual.sum().backward()
            self.assertEqual(x_compiled.grad, x_ref.grad)
        # The observers update their running statistics in place.
        self.assertEqual(mod.state_dict(), ref.state_dict())
        self._assert_grads_equal(mod, ref)


class TestFakeQuantizeCompile(
    _FakeQuantizeCompileHelpers, torch._dynamo.test_case.TestCase
):
    hw_classification = HardwareClassification.ACCELERATOR

    # ------------------------------------------------------------------ fullgraph
    @parametrize("compile_backend", _COMPILE_BACKENDS)
    @parametrize("per_channel", (False, True), name_fn=lambda b: f"per_channel_{b}")
    def test_fused_fullgraph(self, device, per_channel, compile_backend):
        self._check_fullgraph(
            self._fused(per_channel, device),
            lambda: torch.randn(8, 4, device=device),
            compile_backend,
            # a per-channel module that has not run yet graph-breaks, see below
            warm_up=per_channel,
        )

    @parametrize("compile_backend", _COMPILE_BACKENDS)
    @_both_nested_graph_break_modes
    def test_fresh_fused_per_channel_graph_breaks(self, device, compile_backend):
        # The fake kernel of the fused op does not create the per-channel statistics, which
        # fails the AOT backends, so the module has to graph-break as it did while it was
        # skipped by Dynamo.
        fake_quant = self._fused(True, device)
        ref = copy.deepcopy(fake_quant)
        x = torch.randn(8, 4, device=device)
        torch._dynamo.reset()
        counters.clear()
        self.assertEqual(torch.compile(fake_quant, backend=compile_backend)(x), ref(x))
        self.assertEqual(fake_quant.state_dict(), ref.state_dict())
        # one break, at the call, and what follows it does not trace the op again
        self.assertEqual(sum(counters["graph_break"].values()), 1)
        torch._dynamo.reset()
        with self.assertRaises(torch._dynamo.exc.Unsupported) as cm:
            torch.compile(
                self._fused(True, device), backend=compile_backend, fullgraph=True
            )(x)
        message = str(cm.exception)
        self.assertIn("run one eager forward first, with the observer enabled", message)
        self.assertIn("_fused_fake_quantize_forward_before_statistics_exist", message)
        self.assertNotIn("Remove the `torch.compiler.disable` call", message)

    @parametrize("compile_backend", _COMPILE_BACKENDS)
    @parametrize("qengine", ("fbgemm", "qnnpack"))
    def test_qat_linear_fullgraph(self, device, qengine, compile_backend):
        qat_linear = torch.ao.nn.qat.Linear(
            4, 3, qconfig=get_default_qat_qconfig(qengine)
        ).to(device)
        self._check_fullgraph(
            qat_linear,
            lambda: torch.randn(8, 4, device=device),
            compile_backend,
            warm_up=qengine == "fbgemm",  # its weight fake-quant is a per-channel one
        )

    @parametrize("compile_backend", _COMPILE_BACKENDS)
    @parametrize(
        "name",
        (
            "default_fake_quant",
            "default_weight_fake_quant",
            "default_per_channel_weight_fake_quant",
            "default_embedding_fake_quant",
            "default_fixed_qparams_range_0to1_fake_quant",
            "default_fixed_qparams_range_neg1to1_fake_quant",
        ),
    )
    def test_default_fake_quant_fullgraph(self, device, name, compile_backend):
        import torch.ao.quantization.fake_quantize as fake_quantize

        fake_quant = getattr(fake_quantize, name)().to(device)
        self._check_fullgraph(
            fake_quant,
            lambda: torch.randn(8, 4, device=device),
            compile_backend,
            # per-channel qparams only exist after the first eager call
            warm_up=fake_quant.is_per_channel,
        )

    @parametrize("compile_backend", _COMPILE_BACKENDS)
    @parametrize("config", _FAKE_QUANT_CONFIGS)
    @parametrize("observer_enabled", (True, False), name_fn=lambda b: f"observer_{b}")
    @parametrize(
        "fake_quant_enabled", (True, False), name_fn=lambda b: f"fake_quant_{b}"
    )
    def test_fake_quantize_fullgraph(
        self, device, config, observer_enabled, fake_quant_enabled, compile_backend
    ):
        fake_quant = self._make_fake_quant(config, device)
        if config.startswith("per_channel"):
            # per-channel qparams are only traced once they exist
            fake_quant(torch.randn(8, 4, device=device))
        fake_quant.enable_observer(observer_enabled)
        fake_quant.enable_fake_quant(fake_quant_enabled)

        torch._dynamo.reset()
        ref = copy.deepcopy(fake_quant)
        compiled = torch.compile(fake_quant, backend=compile_backend, fullgraph=True)
        for _ in range(2):
            x_ref = torch.randn(8, 4, device=device, requires_grad=True)
            x_compiled = x_ref.detach().clone().requires_grad_()
            expected = ref(x_ref)
            actual = compiled(x_compiled)
            self.assertEqual(actual, expected)
            expected.sum().backward()
            actual.sum().backward()
            self.assertEqual(x_compiled.grad, x_ref.grad)
        self.assertEqual(fake_quant.state_dict(), ref.state_dict())

    @parametrize("compile_backend", _COMPILE_BACKENDS)
    @parametrize("config", _FAKE_QUANT_CONFIGS)
    def test_flags_change_without_recompile(self, device, config, compile_backend):
        fake_quant = self._make_fake_quant(config, device)
        fake_quant(torch.randn(8, 4, device=device))
        torch._dynamo.reset()
        ref = copy.deepcopy(fake_quant)
        counter = CompileCounterWithBackend(compile_backend)
        compiled = torch.compile(fake_quant, backend=counter, fullgraph=True)
        for observer_enabled, fake_quant_enabled in (
            (True, True),
            (False, True),
            (True, False),
            (False, False),
            (True, True),
        ):
            for mod in (fake_quant, ref):
                mod.enable_observer(observer_enabled)
                mod.enable_fake_quant(fake_quant_enabled)
            x = torch.randn(8, 4, device=device)
            self.assertEqual(compiled(x), ref(x))
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(fake_quant.state_dict(), ref.state_dict())

    @parametrize("compile_backend", _COMPILE_BACKENDS)
    @parametrize("config", ("per_tensor_min_max", "per_channel_min_max"))
    def test_out_of_range_zero_point_with_fake_quant_disabled(
        self, device, config, compile_backend
    ):
        # The per-channel fake-quantize kernel runs on every step under compile and
        # checks the range of an integer zero_point, which eager only does when
        # fake-quant is enabled. The per-tensor kernel does not check it.
        fake_quant = self._make_fake_quant(config, device)
        fake_quant(torch.randn(8, 4, device=device))
        fake_quant.disable_observer()
        fake_quant.zero_point.fill_(1000)  # outside of [-128, 127] and of [0, 255]
        ref = copy.deepcopy(fake_quant)
        torch._dynamo.reset()
        compiled = torch.compile(fake_quant, backend=compile_backend, fullgraph=True)
        x = torch.randn(8, 4, device=device)
        for mod in (fake_quant, ref):
            mod.disable_fake_quant()
        self.assertEqual(compiled(x), x)
        self.assertEqual(ref(x), x)
        for mod in (fake_quant, ref):
            mod.enable_fake_quant()
        if config == "per_channel_min_max":
            with self.assertRaisesRegex(RuntimeError, "must be between"):
                ref(x)
            with self.assertRaisesRegex(RuntimeError, "must be between"):
                compiled(x)
        else:
            self.assertEqual(compiled(x), ref(x))

    @parametrize("dtype", (torch.float64, torch.bfloat16), name_fn=lambda d: str(d)[6:])
    @parametrize("config", ("per_tensor_min_max", "per_channel_moving_average"))
    def test_input_dtype_matches_eager(self, device, config, dtype):
        # everything the compiled forward runs on the input, whether or not eager does
        for observer_enabled, fake_quant_enabled in (
            (True, True),
            (False, False),
            (False, True),
            (True, False),
        ):
            fake_quant = self._make_fake_quant(config, device)
            fake_quant(torch.randn(8, 4, device=device))
            fake_quant.enable_observer(observer_enabled)
            fake_quant.enable_fake_quant(fake_quant_enabled)
            ref = copy.deepcopy(fake_quant)
            torch._dynamo.reset()
            compiled = torch.compile(fake_quant, backend="eager", fullgraph=True)
            x = torch.randn(8, 4, device=device).to(dtype)
            self.assertEqual(compiled(x), ref(x))
            self.assertEqual(fake_quant.state_dict(), ref.state_dict())

    @parametrize("config", _FAKE_QUANT_CONFIGS)
    def test_flag_buffers_other_than_zero_and_one(self, device, config):
        # eager only treats 1 as enabled
        fake_quant = self._make_fake_quant(config, device)
        fake_quant(torch.randn(8, 4, device=device))
        fake_quant.observer_enabled.fill_(2)
        fake_quant.fake_quant_enabled.fill_(2)
        ref = copy.deepcopy(fake_quant)
        torch._dynamo.reset()
        compiled = torch.compile(fake_quant, backend="eager", fullgraph=True)
        x = torch.randn(8, 4, device=device)
        self.assertEqual(compiled(x), ref(x))
        self.assertEqual(fake_quant.state_dict(), ref.state_dict())

    @_both_nested_graph_break_modes
    def test_fresh_standalone_per_channel_observer_matches_eager(self, device):
        # observers are inlined as well, not only inside FakeQuantize
        observer = PerChannelMinMaxObserver(ch_axis=1).to(device)
        ref = copy.deepcopy(observer)
        compiled = torch.compile(observer, backend="eager")
        for _ in range(2):
            x = torch.randn(8, 4, device=device)
            self.assertEqual(compiled(x), ref(x))
            self.assertEqual(observer.state_dict(), ref.state_dict())

    @parametrize("qconfig", ("min_max", "moving_average", "histogram"))
    @_both_nested_graph_break_modes
    def test_ptq_calibration_matches_eager(self, device, qconfig):
        # prepare_fx calibration calls the observers directly, they are inlined now
        from torch.ao.quantization import QConfig, QConfigMapping
        from torch.ao.quantization.quantize_fx import prepare_fx

        if qconfig == "histogram":
            qconfig_mapping = get_default_qconfig_mapping("fbgemm")
        else:
            observer = {
                "min_max": MinMaxObserver,
                "moving_average": MovingAverageMinMaxObserver,
            }[qconfig]
            qconfig_mapping = QConfigMapping().set_global(
                QConfig(
                    activation=observer.with_args(dtype=torch.quint8),
                    weight=observer.with_args(
                        dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
                    ),
                )
            )
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).eval()
        prepared = prepare_fx(model, qconfig_mapping, (torch.randn(2, 8),)).to(device)
        ref = copy.deepcopy(prepared)
        counter = CompileCounterWithBackend("eager")
        counters.clear()
        compiled = torch.compile(prepared, backend=counter)
        for _ in range(2):
            x = torch.randn(4, 8, device=device)
            self.assertEqual(compiled(x), ref(x))
        self.assertEqual(prepared.state_dict(), ref.state_dict())
        if qconfig == "histogram":
            # the histogram search is not traced
            self.assertGreater(sum(counters["graph_break"].values()), 0)
        else:
            self.assertEqual(sum(counters["graph_break"].values()), 0)
            self.assertEqual(counter.frame_count, 1)

    # ----------------------------------------------- what is not traced, and why
    @parametrize(
        "config",
        (
            "histogram",
            "subclass_with_extra_state",
            "subclass_without_new_state",
            "fresh_per_channel",
        ),
    )
    @_both_nested_graph_break_modes
    def test_graph_break_matches_eager(self, device, config):
        # These modules are not traced (the observer is not one that FakeQuantize knows
        # the state of, or the per-channel statistics do not exist yet), so they have to
        # run exactly like eager, including with the observer disabled.
        from torch.ao.quantization.fake_quantize import default_histogram_fake_quant

        class ExtraState(MovingAverageMinMaxObserver):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.register_buffer("num_calls", torch.zeros(()))

            def forward(self, x):
                self.num_calls += 1
                return super().forward(x)

        class NoNewState(MovingAverageMinMaxObserver):
            pass

        fake_quant = {
            "histogram": default_histogram_fake_quant,
            "subclass_with_extra_state": lambda: FakeQuantize(observer=ExtraState),
            "subclass_without_new_state": lambda: FakeQuantize(observer=NoNewState),
            "fresh_per_channel": lambda: self._make_fake_quant(
                "per_channel_moving_average", device
            ),
        }[config]().to(device)

        torch._dynamo.reset()
        ref = copy.deepcopy(fake_quant)
        counters.clear()
        compiled = torch.compile(fake_quant, backend="eager")
        flags = [
            (False, False),
            (True, True),
            (False, True),
            (True, False),
            (False, False),
        ]
        if config != "fresh_per_channel":
            # the fake-quant needs qparams: a per-channel module that has none yet fails
            # in eager when only the fake-quant is enabled
            flags.insert(1, (False, True))
        for observer_enabled, fake_quant_enabled in flags:
            for mod in (fake_quant, ref):
                mod.enable_observer(observer_enabled)
                mod.enable_fake_quant(fake_quant_enabled)
            x = torch.randn(8, 4, device=device)
            self.assertEqual(compiled(x), ref(x))
            self.assertEqual(fake_quant.state_dict(), ref.state_dict())

        self.assertEqual(sum(counters["graph_break"].values()), 1)
        if config in ("subclass_with_extra_state", "subclass_without_new_state"):
            torch._dynamo.reset()
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "Use one of these classes"
            ):
                torch.compile(fake_quant, backend="eager", fullgraph=True)(x)

    @parametrize("config", ("per_channel_min_max", "per_channel_moving_average"))
    @_both_nested_graph_break_modes
    def test_fresh_per_channel_needs_an_eager_call(self, device, config):
        # compile before the first forward, as prepare_qat(model); torch.compile(model)
        fake_quant = self._make_fake_quant(config, device)
        torch._dynamo.reset()
        with self.assertRaises(torch._dynamo.exc.Unsupported) as cm:
            torch.compile(fake_quant, backend="eager", fullgraph=True)(
                torch.randn(8, 4, device=device)
            )
        message = str(cm.exception)
        self.assertIn("run one eager forward first", message)
        # the observer has to be enabled for that step to create the statistics
        self.assertIn("with the observer enabled", message)
        # the function that is called says what it is for, and the hint does not tell the
        # user to remove a call inside torch
        self.assertIn("_fake_quantize_forward_before_statistics_exist", message)
        self.assertNotIn("Remove the `torch.compiler.disable` call", message)
        # after that call the same module is traced
        fake_quant(torch.randn(8, 4, device=device))
        self._check_fullgraph(
            fake_quant, lambda: torch.randn(8, 4, device=device), "eager"
        )

    @_both_nested_graph_break_modes
    def test_channel_count_change_after_reset_graph_breaks(self, device):
        # eager only lets the number of channels change once the observer was reset
        fake_quant = self._make_fake_quant("per_channel_moving_average", device)
        fake_quant(torch.randn(8, 4, device=device))
        fake_quant.activation_post_process.reset_min_max_vals()
        ref = copy.deepcopy(fake_quant)
        strict = copy.deepcopy(fake_quant)
        x = torch.randn(8, 6, device=device)
        torch._dynamo.reset()
        counters.clear()
        self.assertEqual(torch.compile(fake_quant, backend="eager")(x), ref(x))
        self.assertEqual(fake_quant.state_dict(), ref.state_dict())
        self.assertEqual(sum(counters["graph_break"].values()), 1)
        torch._dynamo.reset()
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "run one eager forward first"
        ):
            torch.compile(strict, backend="eager", fullgraph=True)(x)

    @_both_nested_graph_break_modes
    def test_input_with_other_channel_count_is_not_traced(self, device):
        # without a reset eager fails for another number of channels, and the compiled
        # path must not run into that inside the traced region
        fake_quant = self._make_fake_quant("per_channel_moving_average", device)
        fake_quant(torch.randn(8, 4, device=device))
        ref = copy.deepcopy(fake_quant)
        x = torch.randn(8, 6, device=device)
        mismatch = "must match the size of tensor"
        with self.assertRaisesRegex(RuntimeError, mismatch):
            ref(x)
        torch._dynamo.reset()
        with self.assertRaisesRegex(RuntimeError, mismatch):
            torch.compile(copy.deepcopy(fake_quant), backend="eager")(x)
        torch._dynamo.reset()
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "do not have the shape that observing.*Run one eager forward with this input "
            "first",
        ):
            torch.compile(fake_quant, backend="eager", fullgraph=True)(x)

    @parametrize("hook", ("forward_hook", "forward_pre_hook"))
    @_both_nested_graph_break_modes
    def test_observer_with_hooks_is_not_traced(self, device, hook):
        # a hook may update state that is restored when the observer is disabled, and it
        # would run on every step under compile
        def count_calls(module, *args):
            module.hook_calls = getattr(module, "hook_calls", 0) + 1

        fake_quant = FakeQuantize(observer=MovingAverageMinMaxObserver).to(device)
        getattr(fake_quant.activation_post_process, f"register_{hook}")(count_calls)
        ref = copy.deepcopy(fake_quant)
        torch._dynamo.reset()
        counters.clear()
        compiled = torch.compile(fake_quant, backend="eager")
        for observer_enabled in (True, False, True, False):
            for mod in (fake_quant, ref):
                mod.enable_observer(observer_enabled)
            x = torch.randn(8, 4, device=device)
            self.assertEqual(compiled(x), ref(x))
        self.assertEqual(sum(counters["graph_break"].values()), 1)
        self.assertEqual(
            fake_quant.activation_post_process.hook_calls,
            ref.activation_post_process.hook_calls,
        )
        torch._dynamo.reset()
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "has forward hooks.*Remove the hooks"
        ):
            torch.compile(fake_quant, backend="eager", fullgraph=True)(x)

    @parametrize("hook", ("forward_hook", "forward_pre_hook"))
    @_both_nested_graph_break_modes
    def test_global_module_hooks_are_not_traced(self, device, hook):
        # register_module_forward_hook applies to the observer as well, and is traced
        # when the observer is
        calls = []

        def count_observer_calls(module, *args):
            if isinstance(module, MovingAverageMinMaxObserver):
                calls.append(1)

        register = getattr(torch.nn.modules.module, f"register_module_{hook}", None)
        fake_quant = FakeQuantize(observer=MovingAverageMinMaxObserver).to(device)
        ref = copy.deepcopy(fake_quant)
        handle = register(count_observer_calls)
        try:
            counters.clear()
            compiled = torch.compile(fake_quant, backend="eager")
            for observer_enabled in (True, False, True, False):
                for mod in (fake_quant, ref):
                    mod.enable_observer(observer_enabled)
                x = torch.randn(8, 4, device=device)
                calls.clear()
                expected = ref(x)
                eager_calls = len(calls)
                calls.clear()
                self.assertEqual(compiled(x), expected)
                self.assertEqual(len(calls), eager_calls)
            self.assertEqual(sum(counters["graph_break"].values()), 1)
            torch._dynamo.reset()
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "has forward hooks.*Remove the hooks"
            ):
                torch.compile(fake_quant, backend="eager", fullgraph=True)(x)
        finally:
            handle.remove()

    @_both_nested_graph_break_modes
    def test_fake_quantize_subclass_is_not_traced(self, device):
        # calculate_qparams of a subclass may do anything, and the traced forward runs it
        # on every step, also while the observer is disabled
        calls = []

        class Subclass(FakeQuantize):
            def calculate_qparams(self):
                calls.append(1)
                return super().calculate_qparams()

        fake_quant = Subclass(observer=MinMaxObserver).to(device)
        ref = copy.deepcopy(fake_quant)
        counters.clear()
        compiled = torch.compile(fake_quant, backend="eager")
        for observer_enabled in (True, False, True, False):
            for mod in (fake_quant, ref):
                mod.enable_observer(observer_enabled)
            x = torch.randn(8, 4, device=device)
            calls.clear()
            expected = ref(x)
            eager_calls = len(calls)
            calls.clear()
            self.assertEqual(compiled(x), expected)
            self.assertEqual(len(calls), eager_calls)
        self.assertEqual(fake_quant.state_dict(), ref.state_dict())
        self.assertEqual(sum(counters["graph_break"].values()), 1)
        torch._dynamo.reset()
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Use one of these classes"
        ):
            torch.compile(fake_quant, backend="eager", fullgraph=True)(x)

    @_both_nested_graph_break_modes
    def test_observer_that_is_not_an_observer_base_falls_back(self, device):
        class Observer(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.inner = MinMaxObserver(quant_min=0, quant_max=255)
                self.quant_min, self.quant_max = 0, 255
                self.dtype = torch.quint8
                self.qscheme = torch.per_tensor_affine
                self.is_dynamic = False

            def forward(self, x):
                return self.inner(x)

            def calculate_qparams(self):
                return self.inner.calculate_qparams()

        fake_quant = FakeQuantize(observer=Observer).to(device)
        ref = copy.deepcopy(fake_quant)
        torch._dynamo.reset()
        counters.clear()
        compiled = torch.compile(fake_quant, backend="eager")
        for observer_enabled in (True, False, True):
            for mod in (fake_quant, ref):
                mod.enable_observer(observer_enabled)
            x = torch.randn(8, 4, device=device)
            self.assertEqual(compiled(x), ref(x))
        self.assertEqual(fake_quant.state_dict(), ref.state_dict())
        self.assertEqual(sum(counters["graph_break"].values()), 1)
        torch._dynamo.reset()
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Use one of these classes"
        ):
            torch.compile(fake_quant, backend="eager", fullgraph=True)(x)

    # -------------------------------------------------------- whole QAT models
    @parametrize("compile_backend", _COMPILE_BACKENDS)
    @parametrize("kind", ("eager_fbgemm", "eager_version0", "fx_fbgemm"))
    @_both_nested_graph_break_modes
    def test_fresh_qat_model_matches_eager(self, device, kind, compile_backend):
        # prepare_qat(model); torch.compile(model), with no eager step in between, then
        # what a QAT run does: train, freeze the observers, a new batch size, eval. The
        # default (fbgemm) weight fake-quant is a fused per-channel one, which has to
        # graph-break here because its fake kernel does not create the statistics.
        model = self._prepared_model(kind, device)
        ref = copy.deepcopy(model)
        compiled = torch.compile(model, backend=compile_backend)

        def step(batch, train=True):
            x = torch.randn(batch, 8, device=device)
            expected, actual = ref(x), compiled(x)
            self.assertEqual(actual, expected)
            if train:
                expected.sum().backward()
                actual.sum().backward()
                self._assert_grads_equal(model, ref)

        for _ in range(2):
            step(4)
        model.apply(disable_observer)
        ref.apply(disable_observer)
        for _ in range(2):
            step(4)
        step(7)
        model.eval()
        ref.eval()
        step(4, train=False)
        self.assertEqual(model.state_dict(), ref.state_dict())

    def test_qat_model_fullgraph_after_one_step(self, device):
        # eager-mode prepare_qat models observe through a forward hook, which Dynamo
        # skips, so only FX models can be traced without a graph break
        model = self._prepared_model("fx_fbgemm", device)
        model(torch.randn(4, 8, device=device))  # the per-channel statistics exist now
        ref = copy.deepcopy(model)
        counter = CompileCounterWithBackend("aot_eager")
        counters.clear()
        compiled = torch.compile(model, backend=counter, fullgraph=True)
        for _ in range(2):
            x = torch.randn(4, 8, device=device)
            expected, actual = ref(x), compiled(x)
            self.assertEqual(actual, expected)
            expected.sum().backward()
            actual.sum().backward()
            self._assert_grads_equal(model, ref)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(model.state_dict(), ref.state_dict())

    @_both_nested_graph_break_modes
    def test_fresh_qat_model_fullgraph_says_what_to_do(self, device):
        model = self._prepared_model("fx_fbgemm", device)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "run one eager forward first"
        ):
            torch.compile(model, backend="aot_eager", fullgraph=True)(
                torch.randn(4, 8, device=device)
            )

    @parametrize("compiled", (False, True))
    def test_moving_average_observer_values(self, device, compiled):
        # absolute values, not a comparison with eager: the first observation replaces
        # the infinities the buffers start with, the next one is averaged in
        for cls, reduce_dim in (
            (MovingAverageMinMaxObserver, None),
            (MovingAveragePerChannelMinMaxObserver, 1),
        ):
            torch._dynamo.reset()
            observer = cls(averaging_constant=0.25).to(device)
            # a fresh per-channel observer resizes its buffers, which graph-breaks
            run = (
                torch.compile(observer, backend="eager", fullgraph=reduce_dim is None)
                if compiled
                else observer
            )

            def extremes(x):
                if reduce_dim is None:
                    return x.amin(), x.amax()
                return x.amin(reduce_dim), x.amax(reduce_dim)

            x1 = torch.randn(8, 4, device=device)
            x2 = x1 * 2 + 1
            run(x1)
            lo1, hi1 = extremes(x1)
            self.assertEqual(observer.min_val, lo1)
            self.assertEqual(observer.max_val, hi1)
            run(x2)
            lo2, hi2 = extremes(x2)
            self.assertEqual(observer.min_val, lo1 + 0.25 * (lo2 - lo1))
            self.assertEqual(observer.max_val, hi1 + 0.25 * (hi2 - hi1))

    # -------------------------------------------------------------- qparams, export
    @parametrize(
        "config",
        (
            "min_max_affine",
            "min_max_symmetric_qint8",
            "min_max_symmetric_quint8",
            "per_channel_affine",
            "per_channel_symmetric",
            "per_channel_float_qparams",
        ),
    )
    def test_calculate_qparams_matches_eager(self, device, config):
        observer_cls, kwargs = {
            "min_max_affine": (
                MinMaxObserver,
                {"qscheme": torch.per_tensor_affine, "dtype": torch.quint8},
            ),
            "min_max_symmetric_qint8": (
                MinMaxObserver,
                {"qscheme": torch.per_tensor_symmetric, "dtype": torch.qint8},
            ),
            "min_max_symmetric_quint8": (
                MinMaxObserver,
                {"qscheme": torch.per_tensor_symmetric, "dtype": torch.quint8},
            ),
            "per_channel_affine": (
                PerChannelMinMaxObserver,
                {"qscheme": torch.per_channel_affine, "dtype": torch.quint8},
            ),
            "per_channel_symmetric": (
                PerChannelMinMaxObserver,
                {"qscheme": torch.per_channel_symmetric, "dtype": torch.qint8},
            ),
            "per_channel_float_qparams": (
                PerChannelMinMaxObserver,
                {
                    "qscheme": torch.per_channel_affine_float_qparams,
                    "dtype": torch.quint8,
                },
            ),
        }[config]
        observer = observer_cls(**kwargs).to(device)

        def compiled_qparams():
            torch._dynamo.reset()
            return torch.compile(
                observer.calculate_qparams, backend="eager", fullgraph=True
            )()

        # not observed yet: the default scale and zero_point (per-channel buffers are
        # empty, per-tensor ones hold inf/-inf); eager warns about it and returns an
        # int64 zero_point whatever the qscheme, while the compiled variant keeps the
        # dtype of the computed one (a dtype cannot depend on the data): int32 for the
        # affine per-tensor qscheme, the same as eager for the others
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unobserved = observer.calculate_qparams()
        (scale, zero_point), (expected_scale, expected_zero_point) = (
            compiled_qparams(),
            unobserved,
        )
        self.assertEqual(scale, expected_scale)
        if config == "min_max_affine":
            self.assertEqual(expected_zero_point.dtype, torch.int64)
            self.assertEqual(zero_point.dtype, torch.int32)
            self.assertEqual(zero_point, expected_zero_point, exact_dtype=False)
        else:
            self.assertEqual(zero_point, expected_zero_point)

        observer(torch.randn(16, 4, device=device))
        # dtypes and values after observing
        for actual, expected in zip(compiled_qparams(), observer.calculate_qparams()):
            self.assertEqual(actual, expected)

    @parametrize("strict", (False, True), name_fn=lambda b: f"strict_{b}")
    @parametrize(
        "config",
        (
            "fused",
            "per_tensor_moving_average",
            "per_channel_moving_average",
            "qat_linear",
        ),
    )
    def test_export(self, device, config, strict):
        # torch.compiler.is_compiling() is also true while exporting, so the compiled
        # path is what torch.export traces
        if config == "fused":
            module = self._fused(False, device)
        elif config == "qat_linear":
            module = torch.ao.nn.qat.Linear(
                4, 3, qconfig=get_default_qat_qconfig("fbgemm")
            ).to(device)
        else:
            module = self._make_fake_quant(config, device)
        if config in ("per_channel_moving_average", "qat_linear"):
            # per-channel qparams exist after the first eager call
            module(torch.randn(8, 4, device=device))
        ref = copy.deepcopy(module)
        torch._dynamo.reset()
        exported = torch.export.export(
            module, (torch.randn(8, 4, device=device),), strict=strict
        ).module()
        for _ in range(2):
            x = torch.randn(8, 4, device=device)
            self.assertEqual(exported(x), ref(x))
        # The observers update their statistics in place. Export drops the buffers that
        # nothing reads (eps), and no other one may be missing.
        exported_state = exported.state_dict()
        required = {name for name in ref.state_dict() if not name.endswith("eps")}
        self.assertTrue(required <= set(exported_state), required - set(exported_state))
        for name in required:
            self.assertEqual(exported_state[name], ref.state_dict()[name])

    def test_export_of_a_module_that_is_not_traced_says_why(self, device):
        # non-strict export has no graph break to report the reason with
        fake_quant = self._make_fake_quant("per_channel_moving_average", device)
        with self.assertRaisesRegex(
            RuntimeError,
            "FakeQuantize with MovingAveragePerChannelMinMaxObserver: .*"
            "run one eager forward first",
        ):
            torch.export.export(
                fake_quant, (torch.randn(8, 4, device=device),), strict=False
            )
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "run one eager forward first"
        ):
            torch.export.export(
                fake_quant, (torch.randn(8, 4, device=device),), strict=True
            )

        # the reasons of the other blockers say what to do as well
        class Subclass(MovingAverageMinMaxObserver):
            pass

        hooked = FakeQuantize(observer=MovingAverageMinMaxObserver).to(device)
        hooked.activation_post_process.register_forward_hook(
            lambda module, args, output: None
        )
        for module, remedy in (
            (FakeQuantize(observer=Subclass).to(device), "Use one of these classes"),
            (hooked, "Remove the hooks"),
        ):
            with self.assertRaisesRegex(RuntimeError, remedy):
                torch.export.export(
                    module, (torch.randn(8, 4, device=device),), strict=False
                )

    @parametrize("kind", ("observer_subclass", "fresh_per_channel", "observer_hooks"))
    def test_draft_export_runs_the_eager_forward(self, device, kind):
        # draft_export propagates real tensors, which can take the branch on
        # observer_enabled, so it does not get the error that export gets
        class Subclass(MovingAverageMinMaxObserver):
            pass

        if kind == "observer_subclass":
            fake_quant = FakeQuantize(observer=Subclass)
        elif kind == "observer_hooks":
            fake_quant = FakeQuantize(observer=MovingAverageMinMaxObserver)
            fake_quant.activation_post_process.register_forward_hook(
                lambda module, args, output: None
            )
        else:
            fake_quant = self._make_fake_quant("per_channel_moving_average", device)
        fake_quant = fake_quant.to(device)
        ref = copy.deepcopy(fake_quant)
        x = torch.randn(8, 4, device=device)
        exported = torch.export.draft_export(fake_quant, (x,)).module()
        self.assertEqual(exported(x), ref(x))

    def test_export_of_a_fresh_per_channel_fused_module(self, device):
        # strict export says what to do; non-strict export traces the op with fake tensors,
        # and the exported program creates the statistics like eager
        fused = self._fused(True, device)
        ref = copy.deepcopy(fused)
        x = torch.randn(8, 4, device=device)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "run one eager forward first"
        ):
            torch.export.export(copy.deepcopy(fused), (x,), strict=True)
        exported = torch.export.export(fused, (x,), strict=False).module()
        for _ in range(2):
            x = torch.randn(8, 4, device=device)
            self.assertEqual(exported(x), ref(x))
        self.assertEqual(exported.state_dict(), ref.state_dict())

    @parametrize(
        "config",
        ("per_tensor_min_max", "per_channel_min_max", "per_channel_float_qparams"),
    )
    def test_cudagraph_unsafe_ops_in_the_graph(self, device, config):
        # The per-channel kernel synchronizes with the device to check the range of an
        # integer zero_point, which CUDA graphs cannot capture (the fused helper op is in
        # FORBIDDEN_CUDAGRAPH_OPS); a floating-point zero_point is not checked. Inductor
        # then runs that op's partition without CUDA graphs (graph_partition), or does not
        # capture the whole graph (graph_partition=False, cpp_wrapper, cudagraphs).
        from torch._dynamo.backends.common import aot_autograd
        from torch._inductor.utils import is_cudagraph_unsafe_fx_node

        unsafe = []

        def fw_compiler(gm, example_inputs):
            unsafe.extend(
                str(node.target)
                for node in gm.graph.nodes
                if node.op == "call_function" and is_cudagraph_unsafe_fx_node(node)
            )
            return gm

        if config == "per_channel_float_qparams":
            fake_quant = FakeQuantize(
                observer=PerChannelMinMaxObserver,
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8,
                qscheme=torch.per_channel_affine_float_qparams,
                ch_axis=0,
            ).to(device)
        else:
            fake_quant = self._make_fake_quant(config, device)
        if config.startswith("per_channel"):
            fake_quant(torch.randn(8, 4, device=device))
        torch._dynamo.reset()
        torch.compile(
            fake_quant, backend=aot_autograd(fw_compiler=fw_compiler), fullgraph=True
        )(torch.randn(8, 4, device=device))
        if config == "per_channel_min_max":
            self.assertEqual(
                unsafe, ["aten.fake_quantize_per_channel_affine_cachemask.default"]
            )
        else:
            self.assertEqual(unsafe, [])

    @_both_nested_graph_break_modes
    def test_negative_ch_axis_is_traced(self, device):
        # the observer accepts a negative ch_axis, and the kernel (which does not) is
        # also called while fake-quant is disabled, as in a calibration step
        fake_quant = FakeQuantize(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            ch_axis=-1,
        ).to(device)
        # eager raises for a negative axis once fake-quant is enabled
        fake_quant.disable_fake_quant()
        fake_quant(torch.randn(8, 4, device=device))
        ref = copy.deepcopy(fake_quant)
        ref_positive = copy.deepcopy(fake_quant)
        ref_raises = copy.deepcopy(fake_quant)
        ref_positive.ch_axis = 1
        ref_positive.activation_post_process.ch_axis = 1
        torch._dynamo.reset()
        compiled = torch.compile(fake_quant, backend="eager", fullgraph=True)
        for fake_quant_enabled in (False, True):
            for mod in (fake_quant, ref, ref_positive):
                mod.enable_fake_quant(fake_quant_enabled)
            x = torch.randn(8, 4, device=device)
            expected = ref_positive(x)
            if not fake_quant_enabled:
                self.assertEqual(ref(x), expected)
            else:
                # the documented difference: the eager forward raises, the traced one
                # returns what the positive axis gives
                ref_raises.enable_fake_quant()
                with self.assertRaisesRegex(RuntimeError, "axis"):
                    ref_raises(x)
            self.assertEqual(compiled(x), expected)
            # the observer's min_val and max_val, scale and zero_point and the flags
            self.assertEqual(fake_quant.state_dict(), ref_positive.state_dict())

    @parametrize("config", ("per_tensor_moving_average", "per_channel_moving_average"))
    @_both_nested_graph_break_modes
    def test_device_without_the_kernel_is_not_traced(self, device, config):
        # The traced forward calls the kernel on every step, eager only when fake-quant is
        # enabled, so on a device that has no kernel (mps) a calibration step, which
        # works in eager, must keep running eagerly.
        import torch.ao.quantization.fake_quantize as fake_quantize_module

        fake_quant = self._make_fake_quant(config, device)
        fake_quant(torch.randn(8, 4, device=device))
        fake_quant.disable_fake_quant()
        ref = copy.deepcopy(fake_quant)
        strict = copy.deepcopy(fake_quant)
        exported = copy.deepcopy(fake_quant)
        x = torch.randn(8, 4, device=device)
        # marked like the real one, which Dynamo calls while tracing
        no_kernel = torch._dynamo.assume_constant_result(lambda device_type: False)
        with unittest.mock.patch.object(
            fake_quantize_module, "_device_has_fake_quantize_kernels", no_kernel
        ):
            torch._dynamo.reset()
            counters.clear()
            self.assertEqual(torch.compile(fake_quant, backend="eager")(x), ref(x))
            self.assertEqual(fake_quant.state_dict(), ref.state_dict())
            self.assertEqual(sum(counters["graph_break"].values()), 1)
            torch._dynamo.reset()
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "no kernel for the fake-quantize ops"
            ):
                torch.compile(strict, backend="eager", fullgraph=True)(x)
            with self.assertRaisesRegex(
                RuntimeError,
                "FakeQuantize with .*: .*no kernel for the fake-quantize ops.*"
                "Run the module eagerly",
            ):
                torch.export.export(exported, (x,), strict=False)
        # with the kernel, the same module is traced in one graph
        torch._dynamo.reset()
        self._check_fullgraph(
            fake_quant, lambda: torch.randn(8, 4, device=device), "eager"
        )

    def test_eager_qparams_have_the_expected_values(self, device):
        # numbers worked out by hand, to check what eager computes (the other tests compare
        # the compiled forward with the eager one that is in the same tree)
        x = torch.tensor([[-1.0, 3.0], [0.5, 2.0]], device=device)

        def int_tensor(values):
            return torch.tensor(values, dtype=torch.int32, device=device)

        # affine: scale = (max - min) / 255, zero_point = -round(min / scale)
        fake_quant = FakeQuantize(
            observer=MinMaxObserver, quant_min=0, quant_max=255, dtype=torch.quint8
        ).to(device)
        fake_quant(x)
        self.assertEqual(fake_quant.scale, torch.tensor([4 / 255], device=device))
        self.assertEqual(fake_quant.zero_point, int_tensor([64]))
        # symmetric qint8: scale = max(|min|, max) / 127.5, zero_point = 0
        fake_quant = FakeQuantize(
            observer=MinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
        ).to(device)
        fake_quant(x)
        self.assertEqual(fake_quant.scale, torch.tensor([3 / 127.5], device=device))
        self.assertEqual(fake_quant.zero_point, int_tensor([0]))
        # per-channel, and another number of channels once the observer was reset
        fake_quant = FakeQuantize(
            observer=PerChannelMinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_channel_affine,
            ch_axis=0,
        ).to(device)
        fake_quant(x)  # the range always includes 0: [-1, 3] and [0, 2]
        self.assertEqual(
            fake_quant.scale, torch.tensor([4 / 255, 2 / 255], device=device)
        )
        self.assertEqual(fake_quant.zero_point, int_tensor([64, 0]))
        fake_quant.activation_post_process.reset_min_max_vals()
        fake_quant(torch.tensor([[0.0, 1.0], [-2.0, 3.0], [1.0, 5.0]], device=device))
        self.assertEqual(
            fake_quant.scale,
            torch.tensor([1 / 255, 5 / 255, 5 / 255], device=device),
        )
        self.assertEqual(fake_quant.zero_point, int_tensor([0, 102, 0]))

    def test_fake_quantize_scriptable(self, device):
        # the compile-only paths must not get in the way of TorchScript
        for name, module in (
            (
                "min_max",
                FakeQuantize(observer=MinMaxObserver, quant_min=0, quant_max=255),
            ),
            ("moving_average", FakeQuantize(observer=MovingAverageMinMaxObserver)),
            (
                "per_channel",
                FakeQuantize(
                    observer=MovingAveragePerChannelMinMaxObserver, **_PER_CHANNEL
                ),
            ),
            ("fused", self._fused(False, device)),
            ("fused_per_channel", self._fused(True, device)),
        ):
            module = module.to(device)
            with self.subTest(name):
                ref = copy.deepcopy(module)
                scripted = torch.jit.script(module)
                x = torch.randn(8, 4, device=device)
                for _ in range(2):
                    self.assertEqual(scripted(x), ref(x))
                self.assertEqual(scripted.calculate_qparams(), ref.calculate_qparams())
                self.assertEqual(scripted.state_dict(), ref.state_dict())


class TestFakeQuantizeInductor(
    _FakeQuantizeCompileHelpers, torch._dynamo.test_case.TestCase
):
    hw_classification = HardwareClassification.ACCELERATOR

    @skipCPUIf(not HAS_CPU, "Inductor needs a C++ compiler")
    @skipCUDAIf(not HAS_CUDA_AND_TRITON, "Inductor needs Triton")
    @parametrize(
        "config",
        ("fused", "per_tensor_moving_average", "per_channel_moving_average"),
    )
    def test_inductor(self, device, config):
        torch.manual_seed(0)
        if config == "fused":
            fake_quant = self._fused(False, device)
        else:
            fake_quant = self._make_fake_quant(config, device)
        self._check_fullgraph(
            fake_quant,
            lambda: torch.randn(8, 4, device=device),
            "inductor",
            warm_up=config.startswith("per_channel"),
        )


class TestFakeQuantizeKernelChecks(torch._dynamo.test_case.TestCase):
    # what the compiled forward decides from the dispatcher and the graph, which does not
    # depend on the device that runs the tests
    hw_classification = HardwareClassification.GENERIC

    def test_cudagraph_check_depends_on_the_zero_point_dtype(self):
        # the decision is made from the dtype of the zero_point node; without a known
        # dtype the op is taken to be unsafe
        from torch._inductor.utils import is_cudagraph_unsafe_fx_node

        graph = torch.fx.Graph()
        x, scale, zero_point = (
            graph.placeholder(n) for n in ("x", "scale", "zero_point")
        )
        node = graph.call_function(
            torch.ops.aten.fake_quantize_per_channel_affine_cachemask.default,
            (x, scale, zero_point, 0, -128, 127),
        )
        self.assertTrue(is_cudagraph_unsafe_fx_node(node))
        for dtype, unsafe in (
            (torch.int32, True),
            (torch.int64, True),
            (torch.float32, False),
        ):
            zero_point.meta["val"] = torch.empty(4, dtype=dtype)
            self.assertEqual(is_cudagraph_unsafe_fx_node(node), unsafe, dtype)

    def test_kernel_lookup_by_device(self):
        from torch.ao.quantization.fake_quantize import (
            _device_has_fake_quantize_kernels,
        )

        self.assertTrue(_device_has_fake_quantize_kernels("cpu"))
        self.assertTrue(_device_has_fake_quantize_kernels("meta"))
        # no kernel is registered for mps
        self.assertFalse(_device_has_fake_quantize_kernels("mps"))
        # a device type without a dispatch key falls back to eager instead of failing
        self.assertFalse(_device_has_fake_quantize_kernels("not_a_device"))
        # privateuseone, and the name that rename_privateuse1_backend gives it, are
        # PrivateUse1: traced once the backend registers the kernels
        self.assertFalse(_device_has_fake_quantize_kernels("privateuseone"))
        with torch.library._scoped_library("aten", "IMPL") as lib:
            for op in (
                "fake_quantize_per_channel_affine_cachemask",
                "_fake_quantize_per_tensor_affine_cachemask_tensor_qparams",
            ):
                lib.impl(op, lambda *args: None, "PrivateUse1")
            self.assertTrue(_device_has_fake_quantize_kernels("privateuseone"))
            self.assertTrue(_device_has_fake_quantize_kernels("cpu"))
        self.assertFalse(_device_has_fake_quantize_kernels("privateuseone"))


instantiate_device_type_tests(TestFakeQuantizeCompile, globals())
instantiate_device_type_tests(TestFakeQuantizeInductor, globals(), except_for="mps")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
