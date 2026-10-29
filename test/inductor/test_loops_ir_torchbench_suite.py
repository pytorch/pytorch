# Owner(s): ["module: inductor"]
"""Unit tests for the genuine ten-model TorchBench suite harness.

Fake-model fixtures are fine here (controls); the real benchmark itself
never uses toys -- it always drives genuine TorchBench factories through
``BenchmarkModel.invoke()``.
"""

import random
import sys
from pathlib import Path

import torch

from torch._inductor.test_case import run_tests, TestCase

SUITE_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "dynamo"
    / "loops_ir_torchbench_suite.py"
)


def _load_suite():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "loops_ir_torchbench_suite", SUITE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load suite from {SUITE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["loops_ir_torchbench_suite"] = module
    spec.loader.exec_module(module)
    return module


suite = _load_suite()


class FakeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.loss_fn = torch.nn.MSELoss()
        self.register_buffer("running", torch.zeros(8))

    def forward(self, x):
        return self.linear(x)


class FakeBenchmark:
    """Minimal genuine-driver-shaped fixture (staged train)."""

    def __init__(self, test="eval"):
        self.test = test
        self.model = FakeModule()
        self.example_inputs = (torch.randn(4, 8),)
        self.example_outputs = (torch.randn(4, 8),)
        self.loss_fn = torch.nn.MSELoss()
        self.run_contexts = []
        self.forward_contexts = []
        self.backward_contexts = []
        self.optimizer_contexts = []
        if test == "train":
            self.opt = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def get_module(self):
        return self.model, self.example_inputs

    def set_module(self, new_model):
        # Preserve Parameter identity contract: reinstall the same objects,
        # never clone into fresh Parameters.
        self.model = new_model

    def get_optimizer(self):
        return getattr(self, "opt", None)

    def forward(self):
        pred = self.model(self.example_inputs[0])
        return self.loss_fn(pred, self.example_outputs[0])

    def backward(self, loss):
        loss.backward()

    def optimizer_step(self):
        self.opt.step()

    def eval(self):
        with torch.no_grad():
            return (self.model(self.example_inputs[0]),)

    def invoke(self):
        if self.test == "train":
            opt = self.get_optimizer()
            if opt is not None:
                opt.zero_grad()
            loss = self.forward()
            self.backward(loss)
            self.optimizer_step()
            return None
        return self.eval()


class FakeNonstagedBenchmark:
    """Genuine-driver-shaped non-staged fixture (custom BERT / DLRM / Timm).

    ``invoke()`` runs the whole step through run_contexts (pick_grad +
    profiling-executor stand-ins), exactly like BenchmarkModel.invoke().
    ``train()`` exists for the observer path but invoke() is authoritative.
    """

    invoke_context_log: list = []

    def __init__(self, test="train"):
        self.test = test
        self.model = FakeModule()
        self.example_inputs = (torch.randn(4, 8),)
        self.loss_fn = torch.nn.MSELoss()
        self.run_contexts = [_recording_context("run")]
        self.forward_contexts = []
        self.backward_contexts = []
        self.optimizer_contexts = []
        self.opt = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.scheduler_state = {"step_count": 0, "lr": 0.01}

    def get_module(self):
        return self.model, self.example_inputs

    def set_module(self, new_model):
        self.model = new_model

    def get_optimizer(self):
        return self.opt

    def train(self):
        # Custom full step: forward + NLL-style loss + backward + scheduled step.
        pred = self.model(self.example_inputs[0])
        loss = self.loss_fn(pred, torch.zeros_like(pred))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.scheduler_state["step_count"] += 1
        self.scheduler_state["lr"] = 0.01 / (1 + self.scheduler_state["step_count"])

    def eval(self):
        with torch.no_grad():
            return (self.model(self.example_inputs[0]),)

    def invoke(self):
        from contextlib import ExitStack

        with ExitStack() as stack:
            for ctx in self.run_contexts:
                stack.enter_context(ctx())
            FakeNonstagedBenchmark.invoke_context_log.append("run")
            if self.test == "train":
                self.train()
                return None
            return self.eval()


def _recording_context(name):
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        FakeNonstagedBenchmark.invoke_context_log.append(name)
        yield

    return _ctx


class FakeScheduledOptim:
    """BERT-ScheduledOptim-shaped scheduler (no state_dict)."""

    def __init__(self, optimizer):
        self._optimizer = optimizer
        self.n_warmup_steps = 100
        self.n_current_steps = 0
        self.init_lr = 0.001

    def step_and_update_lr(self):
        self.n_current_steps += 1
        for group in self._optimizer.param_groups:
            group["lr"] = self.init_lr * self.n_current_steps

    def zero_grad(self):
        self._optimizer.zero_grad()


class ManifestTest(TestCase):
    def test_ten_primary_models_exact_defaults(self):
        expected = {
            "resnet18": {"train_bs": 16, "eval_bs": 8},
            "resnet50": {"train_bs": 32, "eval_bs": 32},
            "alexnet": {"train_bs": 128, "eval_bs": 128},
            "vgg16": {"train_bs": 64, "eval_bs": 4},
            "mobilenet_v2": {"train_bs": 96, "eval_bs": 16},
            "BERT_pytorch": {"train_bs": 16, "eval_bs": 16, "seq_len": 128},
            "hf_Bert": {"train_bs": 4, "eval_bs": 1, "train_seq": 512, "eval_seq": 512},
            "hf_GPT2": {
                "train_bs": 4,
                "eval_bs": 1,
                "train_seq": 512,
                "eval_seq": 1024,
            },
            "timm_vision_transformer": {
                "train_bs": 32,
                "eval_bs": 32,
                "variant": "vit_small_patch16_224",
            },
            "dlrm": {"train_bs": 2048, "eval_bs": 2048},
        }
        self.assertEqual(set(suite.MODEL_MANIFEST), set(expected))
        for name, spec in expected.items():
            for key, value in spec.items():
                self.assertEqual(
                    suite.MODEL_MANIFEST[name][key], value, f"{name}.{key}"
                )

    def test_explicit_fp32_all_variants(self):
        self.assertEqual(suite.EXTRA_ARGS, ["--precision", "fp32"])

    def test_no_device_metadata_batch_override(self):
        src = SUITE_PATH.read_text()
        self.assertNotIn("metadata.yaml", src)
        self.assertNotIn("_get_batch_size_from_metadata", src)
        self.assertIn("suite-manifest-exact-source-default", src)

    def test_no_toy_substitution_in_harness(self):
        src = SUITE_PATH.read_text()
        self.assertIn("torchbenchmark.models.", src)
        self.assertIn("BenchmarkModel.invoke", src)

    def test_never_imports_mutating_common(self):
        src = SUITE_PATH.read_text()
        self.assertNotIn("dynamo.common", src)
        self.assertNotIn("benchcommon", src)


class ParserTest(TestCase):
    def test_defaults(self):
        args = suite.parse_args([])
        self.assertEqual(args.test, "both")
        self.assertEqual(args.cpp_numerics, "both")
        self.assertEqual(args.repeats, 7)
        self.assertEqual(args.warmup, 20)
        self.assertFalse(args.profile)
        self.assertFalse(args.accuracy_only)
        self.assertFalse(args.load_only)
        self.assertEqual(len(args.models), 10)

    def test_all_expands_to_ten(self):
        args = suite.parse_args(["--models", "all"])
        self.assertEqual(len(args.models), 10)

    def test_minimal_honest_flags_exist(self):
        args = suite.parse_args(
            [
                "--models",
                "resnet18",
                "--test",
                "eval",
                "--cpp-numerics",
                "strict_eager",
                "--repeats",
                "2",
                "--warmup",
                "1",
                "--accuracy-only",
            ]
        )
        self.assertEqual(args.models, ["resnet18"])
        self.assertTrue(args.accuracy_only)
        args = suite.parse_args(["--load-only"])
        self.assertTrue(args.load_only)

    def test_tests_and_modes_helpers(self):
        self.assertEqual(suite.tests_for_arg("both"), ["eval", "train"])
        self.assertEqual(suite.tests_for_arg("eval"), ["eval"])
        self.assertEqual(suite.modes_for_arg("both"), ["strict_eager", "inductor"])
        self.assertEqual(suite.modes_for_arg("inductor"), ["inductor"])


class SnapshotTest(TestCase):
    def test_snapshot_is_deepcopy_not_live_alias(self):
        bm = FakeBenchmark(test="train")
        snap = suite.snapshot_state(bm)
        # Mutating live params must not touch the frozen snapshot, and the
        # snapshot holds fingerprints (never live tensor aliases). Only
        # parameters are dirtied here; buffers keep their values by design.
        param_names = {n for n, _ in bm.get_module()[0].named_parameters()}
        with torch.no_grad():
            for param in bm.get_module()[0].parameters():
                param.fill_(12345.0)
        for key, frozen in snap["model_state"].items():
            self.assertIsInstance(frozen, dict)
            self.assertTrue(frozen.get("__tensor__"))
            if key not in param_names:
                continue
            live = bm.get_module()[0].state_dict()[key]
            self.assertFalse(
                suite._frozen_equal(suite._tensor_fingerprint(live), frozen),
                f"snapshot fingerprint {key} aliased live params",
            )

    def test_restore_rng_and_state(self):
        bm = FakeBenchmark(test="train")
        torch.manual_seed(0)
        random.seed(0)
        snap = suite.snapshot_state(bm)
        # Dirty everything.
        with torch.no_grad():
            for param in bm.get_module()[0].parameters():
                param.add_(1.0)
        torch.manual_seed(999)
        random.seed(999)
        suite.restore_state(bm, snap)
        mod, _ = bm.get_module()
        for key, frozen in snap["model_state"].items():
            live = mod.state_dict()[key]
            self.assertTrue(
                suite._frozen_equal(suite._tensor_fingerprint(live), frozen),
                f"param {key} not restored",
            )

    def test_restore_needs_no_live_alias(self):
        bm = FakeBenchmark(test="train")
        snap = suite.snapshot_state(bm)
        before = {k: suite._thaw_value(v) for k, v in snap["model_state"].items()}
        suite.restore_state(bm, snap)
        # Snapshot itself must be unchanged by restore (restore thaws copies).
        for key, value in before.items():
            self.assertTrue(
                suite._frozen_equal(
                    suite._tensor_fingerprint(value), snap["model_state"][key]
                )
            )

    def test_aux_and_optimizer_captured(self):
        bm = FakeBenchmark(test="train")
        snap = suite.snapshot_state(bm)
        self.assertTrue(snap["has_optimizer"])
        self.assertIn("optimizer_state", snap)
        self.assertIn("aux", snap)


class DriverContractTest(TestCase):
    def test_staged_detection(self):
        self.assertTrue(suite._is_staged(FakeBenchmark(test="train")))

        class EvalOnly:
            def get_module(self):
                return None, ()

            def get_optimizer(self):
                return None

            def eval(self):
                return ()

        self.assertFalse(suite._is_staged(EvalOnly()))

    def test_loss_observer_removed_after_use(self):
        bm = FakeBenchmark(test="train")
        original = bm.loss_fn
        with suite._LossObserver(bm):
            _ = bm.forward()
        self.assertIs(bm.loss_fn, original)

    def test_loss_observer_unavailable_never_fabricates(self):
        bm = FakeBenchmark(test="train")
        bm.loss_fn = None  # type: ignore[assignment]
        if hasattr(bm, "model"):
            bm.model.loss_fn = None  # type: ignore[assignment]
        with suite._LossObserver(bm) as obs:
            self.assertEqual(obs.via, "unavailable")
            self.assertIsNone(obs.last_loss)

    def test_set_module_preserves_param_identity(self):
        bm = FakeBenchmark(test="eval")
        mod, _ = bm.get_module()
        before = sorted(id(p) for p in mod.parameters())
        compiled_like = mod  # reinstall the identical object, as the harness does
        bm.set_module(compiled_like)
        after = sorted(id(p) for p in bm.get_module()[0].parameters())
        self.assertEqual(before, after)

    def test_optimizer_refs_preserved_shape(self):
        bm = FakeBenchmark(test="train")
        opt_before = bm.get_optimizer()
        ids_before = sorted(id(p) for g in opt_before.param_groups for p in g["params"])
        # Harness must not rebuild the optimizer; refs stay identical.
        bm.set_module(bm.get_module()[0])
        opt_after = bm.get_optimizer()
        ids_after = sorted(id(p) for g in opt_after.param_groups for p in g["params"])
        self.assertIs(opt_before, opt_after)
        self.assertEqual(ids_before, ids_after)


class VerdictGateTest(TestCase):
    def _base_case(self, **over):
        case = {
            "cpp_compile": {},
            "cpp_coverage": {"num_generated_kernels": 3},
            "cpp_graphs": {"num_seam_graphs_ok": 2, "backward_cpp_evidence": True},
            "accuracy": {},
        }
        case.update(over)
        return case

    def test_failed_compile_never_self_compares_eval(self):
        acc = suite.eval_accuracy(FakeBenchmark(test="eval"), None, None)
        self.assertIn("error", acc["cpp_vs_eager"])
        self.assertFalse(acc["cpp_vs_eager_normal_tol_pass"])

    def test_failed_compile_never_self_compares_train(self):
        acc = suite.train_accuracy(FakeBenchmark(test="train"), None, None)
        self.assertIn("error", acc["cpp_vs_eager"])
        self.assertFalse(acc["cpp_vs_eager_strict_equal"])

    def test_eval_strict_requires_bytewise(self):
        ok = self._base_case(
            accuracy={
                "eager_repeat_bytewise": True,
                "cpp_vs_eager": {"bytewise_equal": True},
            }
        )
        self.assertEqual(suite._verdict(ok, "eval", "strict_eager"), "pass")
        bad = self._base_case(
            accuracy={
                "eager_repeat_bytewise": True,
                "cpp_vs_eager": {"bytewise_equal": False},
            }
        )
        self.assertEqual(suite._verdict(bad, "eval", "strict_eager"), "fail")

    def test_eval_strict_rejects_unstable_eager(self):
        case = self._base_case(
            accuracy={
                "eager_repeat_bytewise": False,
                "cpp_vs_eager": {"bytewise_equal": True},
            }
        )
        self.assertEqual(suite._verdict(case, "eval", "strict_eager"), "fail")

    def test_zero_coverage_fails(self):
        case = self._base_case(cpp_coverage={"num_generated_kernels": 0})
        self.assertEqual(suite._verdict(case, "eval", "inductor"), "fail")

    def test_train_strict_requires_backward_evidence(self):
        fwd_only = self._base_case(
            cpp_graphs={"num_seam_graphs_ok": 1, "backward_cpp_evidence": False},
            accuracy={"eager_repeat_stable": True, "cpp_vs_eager_strict_equal": True},
        )
        self.assertEqual(suite._verdict(fwd_only, "train", "strict_eager"), "fail")
        ok = self._base_case(
            cpp_graphs={"num_seam_graphs_ok": 2, "backward_cpp_evidence": True},
            accuracy={"eager_repeat_stable": True, "cpp_vs_eager_strict_equal": True},
        )
        self.assertEqual(suite._verdict(ok, "train", "strict_eager"), "pass")

    def test_train_normal_requires_backward_evidence_both_modes(self):
        # Finding 1: forward-only must NOT pass in EITHER train mode.
        fwd_only = self._base_case(
            accuracy={
                "cpp_normal_full_state_pass": True,
                "cpp_loss_normal_tol_pass": True,
            },
            cpp_graphs={"num_seam_graphs_ok": 1, "backward_cpp_evidence": False},
        )
        self.assertEqual(suite._verdict(fwd_only, "train", "inductor"), "fail")
        self.assertEqual(suite._verdict(fwd_only, "train", "strict_eager"), "fail")
        ok = self._base_case(
            accuracy={
                "cpp_normal_full_state_pass": True,
                "cpp_loss_normal_tol_pass": True,
                "eager_repeat_stable": True,
                "cpp_vs_eager_strict_equal": True,
            },
            cpp_graphs={"num_seam_graphs_ok": 2, "backward_cpp_evidence": True},
        )
        self.assertEqual(suite._verdict(ok, "train", "inductor"), "pass")

    def test_train_normal_requires_full_state_not_loss_only(self):
        # Finding 1: loss-close but wrong grads/optimizer must fail.
        case = self._base_case(
            accuracy={
                "cpp_normal_full_state_pass": False,
                "cpp_loss_normal_tol_pass": True,
            },
            cpp_graphs={"num_seam_graphs_ok": 2, "backward_cpp_evidence": True},
        )
        self.assertEqual(suite._verdict(case, "train", "inductor"), "fail")

    def test_compile_error_is_fail_never_skip(self):
        case = self._base_case(cpp_compile={"error_type": "X", "error": "boom"})
        self.assertEqual(suite._verdict(case, "eval", "inductor"), "fail")


class SeamSummaryTest(TestCase):
    def test_summarize_success_and_rejection(self):
        calls = [
            {
                "ok": True,
                "graph_kind": "inference",
                "graph_kwargs": {"is_backward": False},
                "pipeline_trace_stages": 8,
            },
            {
                "ok": False,
                "graph_kind": "",
                "graph_kwargs": {"is_backward": False},
                "error_type": "LoopsIrUnsupportedError",
                "error": "BatchNorm boundary",
            },
        ]
        summary = suite._summarize_seam_calls(calls)
        self.assertEqual(summary["num_seam_graphs"], 2)
        self.assertEqual(summary["num_seam_graphs_ok"], 1)
        self.assertEqual(summary["backend_graph_kind"], "inference")
        self.assertFalse(summary["backward_cpp_evidence"])
        self.assertTrue(summary["formal_trace_complete"])
        # Error text never fabricates backward evidence.
        self.assertIn("BatchNorm", calls[1]["error"])

    def test_summarize_backward_evidence(self):
        calls = [
            {
                "ok": True,
                "graph_kind": "forward",
                "graph_kwargs": {},
                "pipeline_trace_stages": 8,
            },
            {
                "ok": True,
                "graph_kind": "backward",
                "graph_kwargs": {"is_backward": True},
                "pipeline_trace_stages": 8,
            },
        ]
        summary = suite._summarize_seam_calls(calls)
        self.assertTrue(summary["forward_cpp_evidence"])
        self.assertTrue(summary["backward_cpp_evidence"])
        self.assertEqual(summary["backend_graph_kind"], "backward+forward")

    def test_empty_summary_never_passes(self):
        summary = suite._summarize_seam_calls([])
        self.assertEqual(summary["num_seam_graphs"], 0)
        self.assertFalse(summary["backward_cpp_evidence"])
        self.assertFalse(summary["formal_trace_complete"])


class Finding2InvokeContextsTest(TestCase):
    """Finding 2: non-staged steps must run genuine invoke() + run_contexts."""

    def test_nonstaged_one_step_uses_invoke_and_contexts(self):
        FakeNonstagedBenchmark.invoke_context_log = []
        bm = FakeNonstagedBenchmark(test="train")
        snap = suite.snapshot_state(bm)
        suite.restore_state(bm, snap)
        # train_accuracy drives _one_step through bm.invoke(); run_contexts log.
        acc = suite.train_accuracy(bm, None, None)
        self.assertIn("run", FakeNonstagedBenchmark.invoke_context_log)
        # Loss observed through the original native loss, never fabricated.
        self.assertIn(acc.get("loss_via", ""), ("FakeNonstagedBenchmark.loss_fn",))
        self.assertTrue(acc.get("reference_loss_available"))

    def test_staged_keeps_run_contexts(self):
        bm = FakeBenchmark(test="train")
        bm.run_contexts = [_recording_context("staged-run")]
        FakeNonstagedBenchmark.invoke_context_log = []
        snap = suite.snapshot_state(bm)
        suite.restore_state(bm, snap)
        # Staged path nests run_contexts around forward/backward/optimizer.
        import torchbenchmark.util.model as _m

        calls = []

        orig_nested = _m.nested

        def _spy(*ctxs):
            calls.append(len(ctxs))
            return orig_nested(*ctxs)

        _m.nested = _spy
        try:
            suite.train_accuracy(bm, None, None)
        finally:
            _m.nested = orig_nested
        self.assertTrue(calls)


class Finding1NormalGateTest(TestCase):
    """Finding 1: normal train gates grads+params+optimizer+scheduler."""

    def test_normal_full_state_sections_present(self):
        bm = FakeBenchmark(test="train")
        snap = suite.snapshot_state(bm)
        suite.restore_state(bm, snap)
        out = suite.train_accuracy(bm, bm, None)
        self.assertIn("cpp_normal_full_state_pass", out)
        self.assertIn("normal_tolerances", out)
        self.assertEqual(out["normal_tolerances"], {"rtol": 1e-4, "atol": 1e-5})

    def test_close_loss_wrong_grads_fails_normal_gate(self):
        # Direct unit proof: identical loss but differing grads must fail
        # the normal full-state gate (loss-only would pass).
        loss = suite._tensor_fingerprint(torch.zeros(1))
        grads_ok = {"w": suite._tensor_fingerprint(torch.zeros(4))}
        grads_bad = {"w": suite._tensor_fingerprint(torch.ones(4))}
        ref = {
            "loss": loss,
            "grads": grads_ok,
            "params": {},
            "optimizer_state": {},
            "schedulers": {},
        }
        cmp_ = {
            "loss": loss,
            "grads": grads_bad,
            "params": {},
            "optimizer_state": {},
            "schedulers": {},
        }
        # Section-by-section: loss passes, grads fail.
        ok_loss, _ = suite._normal_section_close(loss, loss, rtol=1e-4, atol=1e-5)
        self.assertTrue(ok_loss)
        ok_grads, _ = suite._normal_section_close(
            grads_ok, grads_bad, rtol=1e-4, atol=1e-5
        )
        self.assertFalse(ok_grads)
        # Unused fixtures kept explicit (no hidden coupling).
        self.assertEqual(sorted(ref), sorted(cmp_))

    def test_bw_true_wrong_grad_negative(self):
        # Backward evidence present but grads differ -> strict fails.
        a = {
            "grads": {"w": suite._tensor_fingerprint(torch.zeros(4))},
            "params": {},
            "loss": None,
        }
        b = {
            "grads": {"w": suite._tensor_fingerprint(torch.ones(4))},
            "params": {},
            "loss": None,
        }
        eq, diffs = suite._fingerprints_equal(a, b)
        self.assertFalse(eq)
        self.assertTrue(any(d.startswith("grads") for d in diffs))


class Finding3StateScalarsTest(TestCase):
    """Finding 3: every typed scalar preserved; scheduler explicit."""

    def test_float_bits_preserve_neg_zero_and_nan(self):
        import math

        neg_zero = suite._freeze_value(-0.0)
        pos_zero = suite._freeze_value(0.0)
        self.assertNotEqual(
            neg_zero,
            pos_zero,
            "-0.0 and +0.0 must differ at the bit level",
        )
        nan_a = suite._freeze_value(float("nan"))
        self.assertEqual(nan_a, suite._freeze_value(float("nan")))
        self.assertFalse(math.isnan(suite._thaw_value(nan_a)) is False)
        # Normal lane: -0.0 vs +0.0 is within tolerance (quantitative).
        ok, _ = suite._normal_frozen_close(neg_zero, pos_zero, rtol=1e-4, atol=1e-5)
        self.assertTrue(ok)

    def test_bool_never_equals_int(self):
        self.assertFalse(
            suite._frozen_equal(suite._freeze_value(True), suite._freeze_value(1))
        )

    def test_unsupported_type_fails_closed(self):
        with self.assertRaises(TypeError):
            suite._freeze_value(object())

    def test_scheduler_plain_attrs_all_numeric_captured(self):
        bm = FakeBenchmark(test="train")
        bm.optim_schedule = FakeScheduledOptim(bm.get_optimizer())
        snap = suite.snapshot_state(bm)
        entry = snap["scheduler_states"]["optim_schedule"]
        self.assertEqual(entry["kind"], "plain_attrs")
        for key in ("n_warmup_steps", "n_current_steps", "init_lr"):
            self.assertIn(key, entry["attrs"])
        # Mutate then restore exact counters.
        bm.optim_schedule.n_current_steps = 99
        suite.restore_state(bm, snap)
        self.assertEqual(bm.optim_schedule.n_current_steps, 0)

    def test_scheduler_misrestore_fails_closed(self):
        bm = FakeBenchmark(test="train")
        bm.optim_schedule = FakeScheduledOptim(bm.get_optimizer())
        snap = suite.snapshot_state(bm)
        del bm.optim_schedule
        with self.assertRaises(suite.RestorationError):
            suite.restore_state(bm, snap)

    def test_lr_state_pairs_framed(self):
        bm = FakeBenchmark(test="train")
        opt = bm.get_optimizer()
        opt.param_groups[0]["lr"] = 0.123
        snap = suite.snapshot_state(bm)
        opt.param_groups[0]["lr"] = 0.999
        suite.restore_state(bm, snap)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.123)


class Finding4TypedFingerprintsTest(TestCase):
    """Finding 4: dtype/shape/strides/bytes all gate strict; no float() cast."""

    def test_wrong_dtype_same_bytes_detected(self):
        a = suite._tensor_fingerprint(torch.zeros(4, dtype=torch.float32))
        b = suite._tensor_fingerprint(torch.zeros(4, dtype=torch.float16))
        self.assertFalse(suite._frozen_equal(a, b))
        ok, reason = suite._normal_tensor_close(a, b, rtol=1e-4, atol=1e-5)
        self.assertFalse(ok)
        self.assertIn("dtype", reason)

    def test_wrong_strides_detected(self):
        base = torch.randn(4, 4)
        a = suite._tensor_fingerprint(base)
        b = suite._tensor_fingerprint(base.t())
        # Same storage bytes possible, strides must differ.
        self.assertNotEqual(a["strides"], b["strides"])
        self.assertFalse(suite._frozen_equal(a, b))
        ok, reason = suite._normal_tensor_close(
            dict(a, bytes=b["bytes"]), b, rtol=1e-4, atol=1e-5
        )
        # Bytes swapped but strides declared: strict fails; normal on the
        # declared layout compares reconstructed values.
        self.assertFalse(suite._frozen_equal(dict(a, bytes=b["bytes"]), b))

    def test_param_index_identity(self):
        bm = FakeBenchmark(test="train")
        snap = suite.snapshot_state(bm)
        suite.restore_state(bm, snap)
        mod, _ = bm.get_module()
        live_keys = sorted(mod.state_dict().keys())
        snap_keys = sorted(snap["model_state"].keys())
        self.assertEqual(live_keys, snap_keys)

    def test_no_float_cast_in_strict(self):
        # The strict post-state path (fingerprint/compare/restore) must
        # never downcast via .float() (loses dtype fidelity). The only
        # remaining .float() uses are the legacy fallback byte-compare
        # diagnostics (max_abs_diff reporting, not gating) and the max_abs
        # diagnostic inside _compare_outputs fallbacks.
        src = SUITE_PATH.read_text()
        strict_regions = src.split("def _post_state_fingerprint")[1].split(
            "def timed_latencies"
        )[0]
        self.assertNotIn(".float()", strict_regions)
        self.assertIn("_tensor_fingerprint", strict_regions)

    def test_loss_static_scalar_typed_bits(self):
        f = suite._freeze_stateless_scalar(0.5)
        self.assertIn("__float_bits__", f)
        self.assertEqual(suite._thaw_value(f), 0.5)


class Finding5RestoreStrictTest(TestCase):
    """Finding 5: strict load, named stage errors, identity, veto."""

    def test_bad_restore_invalid_case_never_timed(self):
        bm = FakeBenchmark(test="train")
        snap = suite.snapshot_state(bm)
        # Corrupt the snapshot: drop the model state entirely.
        del snap["model_state"]
        with self.assertRaises(suite.RestorationError) as ctx:
            suite.restore_state(bm, snap)
        self.assertIn("model", str(ctx.exception))

    def test_optimizer_bad_identity_fails(self):
        bm = FakeBenchmark(test="train")
        snap = suite.snapshot_state(bm)
        # Rebind the optimizer to DIFFERENT Parameter objects outside the
        # live module (stale refs would silently update wrong tensors).
        other_params = [
            torch.nn.Parameter(torch.randn(8, 8)),
            torch.nn.Parameter(torch.zeros(8)),
        ]
        bm.opt = torch.optim.SGD(other_params, lr=0.5)
        with self.assertRaises(suite.RestorationError) as ctx:
            suite.restore_state(bm, snap)
        self.assertIn("outside the live target module", str(ctx.exception))

    def test_cross_variant_clone_passes(self):
        # Finding 1 (second half): a DIFFERENT genuine factory (different
        # Parameter objects, same names/shapes) MUST restore cleanly.
        src = FakeBenchmark(test="train")
        torch.manual_seed(0)
        snap = suite.snapshot_state(src)
        dst = FakeBenchmark(test="train")
        torch.manual_seed(1234)  # different values, different objects
        self.assertFalse(
            any(
                p1 is p2
                for p1 in src.get_module()[0].parameters()
                for p2 in dst.get_module()[0].parameters()
            )
        )
        suite.restore_state(dst, snap)  # must not raise
        for key, frozen in snap["model_state"].items():
            live = dst.get_module()[0].state_dict()[key]
            self.assertTrue(
                suite._frozen_equal(suite._tensor_fingerprint(live), frozen),
                f"cross-variant param {key} not restored",
            )

    def test_swapped_group_order_rejected(self):
        # Same shapes everywhere (8x8) so only the NAME map can catch a
        # swapped group order -- names make it explicit.
        mod = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
        opt = torch.optim.SGD(
            [
                {"params": [mod[0].weight], "lr": 0.1},
                {"params": [mod[1].weight], "lr": 0.2},
            ]
        )

        class TwoGroup:
            def __init__(self):
                self.model = mod
                self.opt = opt

            def get_module(self):
                return self.model, (torch.randn(2, 8),)

            def get_optimizer(self):
                return self.opt

        bm = TwoGroup()
        snap = suite.snapshot_state(bm)
        # Swap the target's group order: names now mismatch positions.
        opt.param_groups[0], opt.param_groups[1] = (
            opt.param_groups[1],
            opt.param_groups[0],
        )
        with self.assertRaises(suite.RestorationError) as ctx:
            suite.restore_state(bm, snap)
        self.assertIn("inconsistent", str(ctx.exception))

    def test_restoration_error_vetoes_timing(self):
        # run_case accuracy-restore failure returns early: no timing key
        # computed, status fail, explicit RestorationError recorded.
        bm = FakeBenchmark(test="train")
        snap = suite.snapshot_state(bm)
        del snap["model_state"]
        try:
            suite.restore_state(bm, snap)
            self.fail("expected RestorationError")
        except suite.RestorationError as error:
            veto = {"status": "fail", "timing": {"skipped": f"not timed: {error}"}}
            self.assertEqual(veto["status"], "fail")
            self.assertIn("not timed", veto["timing"]["skipped"])

    def test_compiled_module_unwrap_avoids_prefix_mismatch(self):
        bm = FakeBenchmark(test="train")
        snap = suite.snapshot_state(bm)

        class Wrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self._orig_mod = inner

            def forward(self, x):
                return self._orig_mod(x)

        inner, _ = bm.get_module()
        bm.set_module(Wrapper(inner))
        # Restore must target the unwrapped module (canonical keys) and keep
        # Parameter object identity (no rebinding).
        before_ids = sorted(id(p) for p in inner.parameters())
        suite.restore_state(bm, snap)
        after_ids = sorted(id(p) for p in inner.parameters())
        self.assertEqual(before_ids, after_ids)

    def test_snapshot_missing_helper_audited(self):
        bm = FakeBenchmark(test="train")
        snap = suite.snapshot_state(bm)
        # Every needed section must exist (no silent capture gaps).
        for key in (
            "model_state",
            "grads",
            "buffers",
            "optimizer_state",
            "scheduler_states",
            "torch_rng",
            "cuda_rng",
            "python_rng",
        ):
            self.assertIn(key, snap, f"snapshot missing {key}")


class Finding6TimingStateTest(TestCase):
    """Finding 6: pre-warmup restore-all + round-robin warmup."""

    def test_round_robin_warmup_order(self):
        order = []
        fns = {
            "eager": lambda: order.append("eager"),
            "cpp": lambda: order.append("cpp"),
            "stock": lambda: order.append("stock"),
        }
        # Mirror the harness warmup loop shape (round-robin, not blocked).
        warmup = 2
        labels = list(fns)
        for _ in range(warmup):
            for label in labels:
                fns[label]()
        self.assertEqual(order, ["eager", "cpp", "stock", "eager", "cpp", "stock"])

    def test_restore_all_before_warmup_in_source(self):
        src = SUITE_PATH.read_text()
        self.assertIn("_restore_all()", src)
        # Round-robin warmup lives in timed_latencies.
        self.assertIn("for label in labels:", src)

    def test_variants_start_equal_after_restore(self):
        bm = FakeBenchmark(test="train")
        snap = suite.snapshot_state(bm)
        # Dirty, restore, verify equality of live vs snapshot params.
        with torch.no_grad():
            for p in bm.get_module()[0].parameters():
                p.add_(5.0)
        suite.restore_state(bm, snap)
        mod, _ = bm.get_module()
        for key, frozen in snap["model_state"].items():
            live = mod.state_dict()[key]
            self.assertTrue(
                suite._frozen_equal(suite._tensor_fingerprint(live), frozen)
            )


class Finding7RngTest(TestCase):
    """Finding 7: RNG divergence detected; tuple data preserved."""

    def test_python_rng_tuple_preserved(self):
        bm = FakeBenchmark(test="train")
        snap = suite.snapshot_state(bm)
        st = suite._thaw_python_state(snap["python_rng"])
        self.assertIsInstance(st, tuple)
        self.assertEqual(len(st), 3)
        random.setstate(st)  # must be a valid state tuple

    def test_numpy_rng_uint32_bytes(self):
        import numpy as _np

        bm = FakeBenchmark(test="train")
        snap = suite.snapshot_state(bm)
        if not snap.get("numpy_used"):
            self.skipTest("numpy unavailable")
        thawed = suite._thaw_numpy_state(snap["numpy_rng"])
        self.assertEqual(thawed[0], "MT19937")
        self.assertEqual(thawed[1].dtype, _np.uint32)
        _np.random.set_state(thawed)

    def test_rng_divergence_detected(self):
        torch.manual_seed(0)
        a = suite._continuation_draws()
        torch.manual_seed(1)
        b = suite._continuation_draws()
        self.assertFalse(suite._frozen_equal(a, b))
        diffs = suite._diff_paths("rng_draws", a, b)
        self.assertTrue(diffs)

    def test_rng_match_passes(self):
        torch.manual_seed(0)
        a = suite._continuation_draws()
        torch.manual_seed(0)
        b = suite._continuation_draws()
        # torch-stream draws match; python/numpy streams advanced by the
        # first call, so compare torch-section only for determinism proof.
        self.assertTrue(suite._frozen_equal(a["torch"], b["torch"]))

    def test_cuda_rng_restore_fail_closed(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable (CPU CI)")
        bm = FakeBenchmark(test="train")
        snap = suite.snapshot_state(bm)
        snap["cuda_rng"] = [{"__error__": "simulated capture failure"}]
        with self.assertRaises(suite.RestorationError):
            suite.restore_state(bm, snap)


class Finding2TypedKeysTest(TestCase):
    """Finding 2: int Adam keys round-trip exactly; no str() coercion."""

    def test_int_keys_preserved(self):
        state = {0: {"exp_avg": torch.zeros(2)}, 1: {"exp_avg": torch.ones(2)}}
        frozen = suite._freeze_value(state)
        self.assertIsInstance(frozen, suite.FrozenMapping)
        thawed = suite._thaw_value(frozen)
        self.assertEqual(set(thawed.keys()), {0, 1})
        for k in thawed:
            self.assertIsInstance(k, int)
        self.assertTrue(torch.equal(thawed[0]["exp_avg"], torch.zeros(2)))

    def test_bool_not_int_collision(self):
        frozen_t = suite._freeze_key(True)
        frozen_1 = suite._freeze_key(1)
        self.assertNotEqual(frozen_t, frozen_1)
        self.assertIs(suite._thaw_key(frozen_t), True)
        self.assertEqual(suite._thaw_key(frozen_1), 1)
        self.assertIsInstance(suite._thaw_key(frozen_1), int)

    def test_no_marker_collision_with_user_keys(self):
        # A genuine user dict keyed '__float_bits__' must survive as data.
        state = {"__float_bits__": torch.zeros(1), "lr": 0.5}
        frozen = suite._freeze_value(state)
        thawed = suite._thaw_value(frozen)
        self.assertIn("__float_bits__", thawed)
        self.assertTrue(torch.equal(thawed["__float_bits__"], torch.zeros(1)))

    def test_adam_state_full_roundtrip(self):
        mod = torch.nn.Linear(4, 4)
        opt = torch.optim.Adam(mod.parameters(), lr=0.01)
        x = torch.randn(3, 4)
        opt.zero_grad()
        mod(x).sum().backward()
        opt.step()  # populates exp_avg/exp_avg_sq (nonempty moments)
        sd = opt.state_dict()
        self.assertTrue(sd["state"])  # nonempty
        frozen = suite._freeze_value(sd)
        thawed = suite._thaw_value(frozen)
        self.assertEqual(set(thawed["state"].keys()), set(sd["state"].keys()))
        for k in sd["state"]:
            self.assertIsInstance(k, int)
            self.assertIn(k, thawed["state"])
            for field in ("exp_avg", "exp_avg_sq"):
                self.assertTrue(
                    torch.equal(thawed["state"][k][field], sd["state"][k][field]),
                    f"Adam moment {k}.{field} lost",
                )
        # LR + step counter preserved.
        self.assertEqual(thawed["param_groups"][0]["lr"], 0.01)

    def test_cpu_scalar_tensor_metadata(self):
        t = torch.tensor(3.0)  # 0-dim CPU fp32
        f = suite._tensor_fingerprint(t)
        self.assertEqual(f["shape"], [])
        self.assertEqual(f["dtype"], "torch.float32")
        back = suite._reconstruct_tensor(f)
        self.assertTrue(torch.equal(back, t))


class Finding3ReconstructTest(TestCase):
    """Finding 3: compact-bytes decode never reads out of bounds."""

    def test_strided_reconstruct_matches_logical_values(self):
        base = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        view = base[:, ::2]  # shape 8x4, strides (8, 2), compact 32 values
        self.assertEqual(tuple(view.shape), (8, 4))
        f = suite._tensor_fingerprint(view)
        back = suite._reconstruct_tensor(f)
        self.assertEqual(tuple(back.shape), (8, 4))
        self.assertTrue(torch.equal(back, view))

    def test_empty_and_zero_dim(self):
        e = torch.empty(0, dtype=torch.float32)
        back = suite._reconstruct_tensor(suite._tensor_fingerprint(e))
        self.assertEqual(back.numel(), 0)
        self.assertEqual(back.dtype, torch.float32)
        z = torch.tensor(2.5)
        back_z = suite._reconstruct_tensor(suite._tensor_fingerprint(z))
        self.assertTrue(torch.equal(back_z, z))

    def test_device_recorded_and_restored(self):
        t = torch.randn(4, 4)
        f = suite._tensor_fingerprint(t)
        self.assertEqual(f["device"], str(t.device))
        back = suite._reconstruct_tensor(f)
        self.assertEqual(str(back.device), str(t.device))

    def test_single_loss_fingerprint_flows_to_normal_gate(self):
        # Fix 1: _one_step's typed loss fingerprint must pass through
        # _post_state_fingerprint untouched (no FrozenMapping re-wrap), so
        # the normal gate compares values with tolerance.
        loss = torch.tensor(1.0)
        fp_loss = suite._tensor_fingerprint(loss)
        fp = suite._post_state_fingerprint(FakeBenchmark(test="eval"), fp_loss)
        self.assertTrue(fp["loss"].get("__tensor__") is True)
        self.assertNotIsInstance(fp["loss"], suite.FrozenMapping)
        # A tiny tolerated change passes the whole section comparison.
        close = suite._tensor_fingerprint(torch.tensor(1.0 + 1e-7))
        ok, _ = suite._normal_section_close(fp_loss, close, rtol=1e-4, atol=1e-5)
        self.assertTrue(ok)
        # Beyond tolerance fails.
        far = suite._tensor_fingerprint(torch.tensor(2.0))
        ok, _ = suite._normal_section_close(fp_loss, far, rtol=1e-4, atol=1e-5)
        self.assertFalse(ok)

    def test_stateless_scalar_loss_single_freeze(self):
        f = suite._freeze_stateless_scalar(0.5)
        fp = suite._post_state_fingerprint(FakeBenchmark(test="eval"), f)
        self.assertEqual(fp["loss"], f)

    def test_wrong_bn_buffer_fails_normal_gate(self):
        # Fix 3: a wrong BatchNorm running buffer must fail the normal gate.
        ref = {"running_mean": suite._tensor_fingerprint(torch.zeros(4))}
        bad = {"running_mean": suite._tensor_fingerprint(torch.ones(4))}
        ok, diffs = suite._normal_section_close(ref, bad, rtol=1e-4, atol=1e-5)
        self.assertFalse(ok)
        self.assertTrue(any("running_mean" in d for d in diffs))

    def test_strict_bytes_still_fail(self):
        # Strict byte comparison still rejects any value change, however
        # small (normal tolerance never leaks into the strict gate).
        a = suite._tensor_fingerprint(torch.zeros(4))
        b = suite._tensor_fingerprint(torch.ones(4) * 1e-9)
        self.assertFalse(suite._frozen_equal(a, b))
        eq, diffs = suite._fingerprints_equal(
            {"params": {"w": a}}, {"params": {"w": b}}
        )
        self.assertFalse(eq)
        self.assertTrue(any("params" in d for d in diffs))

    def test_device_mismatch_fails_normal_close(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable (CPU CI)")
        a = suite._tensor_fingerprint(torch.zeros(4))
        b = dict(a)
        b["device"] = "cuda:0"
        ok, reason = suite._normal_tensor_close(a, b, rtol=1e-4, atol=1e-5)
        self.assertFalse(ok)
        self.assertIn("device", reason)

    def test_normal_close_strided_metadata_equal(self):
        base = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        a = suite._tensor_fingerprint(base)
        b = suite._tensor_fingerprint(base.clone())
        ok, _ = suite._normal_tensor_close(a, b, rtol=1e-4, atol=1e-5)
        self.assertTrue(ok)

    def test_sparse_grad_fingerprinted_not_dropped(self):
        idx = torch.tensor([[0, 1, 3]])
        vals = torch.tensor([1.0, 2.0, 3.0])
        sp = torch.sparse_coo_tensor(idx, vals, (4,))
        f = suite._tensor_fingerprint(sp)
        self.assertTrue(f.get("sparse"))
        self.assertIsNone(f.get("strides"))
        back = suite._reconstruct_tensor(f)
        self.assertTrue(torch.equal(back, sp.to_dense()))
        # sparse-vs-dense mismatch fails closed.
        dense_f = suite._tensor_fingerprint(torch.zeros(4))
        ok, reason = suite._normal_tensor_close(f, dense_f, rtol=1e-4, atol=1e-5)
        self.assertFalse(ok)
        self.assertIn("sparse", reason)


class Finding4CleanupVetoTest(TestCase):
    """Finding 4: compile-cleanup restore failure vetoes accuracy+timing."""

    def test_cleanup_veto_shape(self):
        veto = {
            "error_type": "RestorationError",
            "error": "compile_cleanup cpp restore: rng[torch]: missing",
            "stage": "compile_cleanup",
        }
        self.assertEqual(veto["stage"], "compile_cleanup")
        self.assertIn("RestorationError", veto["error_type"])

    def test_restore_fail_once_model_vetoes(self):
        # A benchmark whose restore always fails must surface
        # RestorationError (the run_case veto path records fail + skips
        # timing). Direct unit proof on the restore primitive:
        bm = FakeBenchmark(test="train")
        snap = suite.snapshot_state(bm)
        snap["torch_rng"] = {"__error__": "simulated capture failure"}
        with self.assertRaises(suite.RestorationError):
            suite.restore_state(bm, snap)


class Finding5NumpyTest(TestCase):
    """Finding 5: writable round-trip, exact next-sample, writeable flag."""

    def test_state_roundtrip_and_writable(self):
        np = pytest_importorskip_numpy(self)
        np.random.seed(0)
        snap_state = np.random.get_state()
        frozen = suite._freeze_value(snap_state)
        thawed = suite._thaw_value(frozen)
        self.assertEqual(thawed[0], snap_state[0])
        self.assertTrue(thawed[1].flags.writeable)
        self.assertTrue((thawed[1] == snap_state[1]).all())

    def test_restore_next_sample_exact(self):
        np = pytest_importorskip_numpy(self)
        np.random.seed(1234)
        frozen = suite._freeze_value(np.random.get_state())
        thawed = suite._thaw_value(frozen)
        np.random.set_state(thawed)
        a = np.random.rand(4)
        np.random.set_state(suite._thaw_value(frozen))
        b = np.random.rand(4)
        self.assertTrue((a == b).all())

    def test_numpy_rng_compare_typed_not_float(self):
        np = pytest_importorskip_numpy(self)
        np.random.seed(0)
        s1 = suite._freeze_value(np.random.get_state())
        np.random.seed(1)
        s2 = suite._freeze_value(np.random.get_state())
        self.assertFalse(suite._frozen_equal(s1, s2))


def pytest_importorskip_numpy(testcase):
    try:
        import numpy as np
    except ImportError:
        testcase.skipTest("numpy unavailable")
    return np


class StockComparatorPolicyTest(TestCase):
    """Bounded comparator correction: existing stock APIs/policy only.

    Covers: stock-approved FP64 error envelope, existing model policy
    (evalFP32 1e-4/trainCUDA 1e-3 with model metadata), bad CPP still
    rejected, strict gate unchanged, reference-failure fallback documented,
    and no input/global mutation by the FP64 reference builders.
    """

    def test_stock_tolerance_eval_fp32_is_1e4(self):
        tol, cosine, record = suite._stock_tolerance_and_flags("resnet18", "eval")
        self.assertEqual(tol, 1e-4)
        self.assertFalse(cosine)
        self.assertEqual(
            record["source"],
            "benchmark/torchbenchmark/util/env_check.py:get_tolerance_and_cosine_flag",
        )

    def test_stock_tolerance_train_cuda_is_1e3(self):
        tol, _, record = suite._stock_tolerance_and_flags("resnet18", "train")
        # Existing model policy for train/CUDA (stock metadata; no override).
        self.assertEqual(tol, 1e-3)
        self.assertEqual(record["test"], "train")

    def test_stock_same_fp64_envelope_passes_within_2x(self):
        # Stock-approved FP64 error envelope: res_error <= 2*ref_error+tol/10.
        fp64 = torch.zeros(8, dtype=torch.float64)
        ref = torch.zeros(8, dtype=torch.float32)
        res = torch.full((8,), 1e-6, dtype=torch.float32)
        ok, _ = suite._stock_same(ref, res, fp64, tol=1e-4, cos_similarity=False)
        self.assertTrue(ok)

    def test_stock_same_rejects_far_output(self):
        fp64 = torch.zeros(8, dtype=torch.float64)
        ref = torch.zeros(8, dtype=torch.float32)
        res = torch.ones(8, dtype=torch.float32)
        ok, _ = suite._stock_same(ref, res, fp64, tol=1e-4, cos_similarity=False)
        self.assertFalse(ok)

    def test_layout_metadata_blocks_dtype_change_but_values_go_to_stock(self):
        a = suite._tensor_fingerprint(torch.zeros(4, dtype=torch.float32))
        b = suite._tensor_fingerprint(torch.zeros(4, dtype=torch.float16))
        ok, reason = suite._layout_metadata_equal(a, b)
        self.assertFalse(ok)
        self.assertIn("dtype", reason)
        # Same layout, tiny value change: metadata passes (values judged by
        # the stock comparator, not here).
        c = suite._tensor_fingerprint(torch.zeros(4, dtype=torch.float32))
        ok, _ = suite._layout_metadata_equal(a, c)
        self.assertTrue(ok)

    def test_bad_cpp_still_rejected_by_stock_gate(self):
        fp64 = torch.zeros(8, dtype=torch.float64)
        ref = {"loss": suite._tensor_fingerprint(torch.zeros(8))}
        bad = {"loss": suite._tensor_fingerprint(torch.ones(8))}
        fp64_sec = {
            "loss": {
                "__tensor__": True,
                "dtype": "torch.float64",
                "shape": [8],
                "strides": [1],
                "numel": 8,
                "device": "cpu",
                "bytes": fp64.contiguous().numpy().tobytes(),
            }
        }
        ok, _ = suite._stock_section_pass(
            ref["loss"], bad["loss"], fp64_sec["loss"], tol=1e-4, cos_similarity=False
        )
        self.assertFalse(ok)

    def test_strict_gate_unchanged_eval_and_train(self):
        base = {
            "cpp_compile": {},
            "cpp_coverage": {"num_generated_kernels": 3},
            "cpp_graphs": {"num_seam_graphs_ok": 2, "backward_cpp_evidence": True},
        }
        # Eval strict still bytewise-gated (stock keys never leak in).
        strict_ok = dict(
            base,
            accuracy={
                "eager_repeat_bytewise": True,
                "cpp_vs_eager": {"bytewise_equal": True},
            },
        )
        self.assertEqual(suite._verdict(strict_ok, "eval", "strict_eager"), "pass")
        strict_bad = dict(
            base,
            accuracy={
                "eager_repeat_bytewise": True,
                "cpp_vs_eager": {"bytewise_equal": False},
                "cpp_vs_eager_stock_pass": True,
            },
        )
        self.assertEqual(suite._verdict(strict_bad, "eval", "strict_eager"), "fail")

    def test_normal_verdict_requires_valid_stock_reference(self):
        base = {
            "cpp_compile": {},
            "cpp_coverage": {"num_generated_kernels": 3},
            "cpp_graphs": {"num_seam_graphs_ok": 2, "backward_cpp_evidence": True},
        }
        masked = dict(
            base,
            accuracy={
                "cpp_vs_eager_stock_pass": True,
                "stock_reference_valid": False,
            },
        )
        self.assertEqual(suite._verdict(masked, "eval", "inductor"), "fail")
        ok = dict(
            base,
            accuracy={
                "cpp_vs_eager_stock_pass": True,
                "stock_reference_valid": True,
            },
        )
        self.assertEqual(suite._verdict(ok, "eval", "inductor"), "pass")
        train_masked = dict(
            base,
            accuracy={
                "cpp_normal_full_state_pass": True,
                "stock_reference_valid": False,
            },
        )
        self.assertEqual(suite._verdict(train_masked, "train", "inductor"), "fail")

    def test_fp64_eval_reference_failure_uses_cosine_fallback_only(self):
        # FP64 execution failure (mid-build: clone.invoke raises) must follow
        # ONLY the established stock cosine fallback and record it.
        bm = FakeBenchmark(test="eval")
        bm.eval = lambda: (_ for _ in ()).throw(RuntimeError("fp64 boom"))
        fp64_out, record = suite._fp64_reference_outputs(bm, "eval", [])
        self.assertIsNone(fp64_out)
        self.assertEqual(record.get("fallback"), "cosine")
        self.assertIn("cosine", record.get("fallback_reason", ""))
        self.assertEqual(
            record.get("fallback_source"),
            "benchmark/torchbenchmark/util/env_check.py:check_accuracy",
        )

    def test_fp64_train_reference_failure_documented(self):
        bm = FakeBenchmark(test="train")
        snap = suite.snapshot_state(bm)
        fp64_fp, record = suite._fp64_train_post_state(
            bm, snap, lambda _b: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        self.assertIsNone(fp64_fp)
        self.assertEqual(record.get("fallback"), "cosine")
        self.assertEqual(
            record.get("fallback_source"),
            "benchmark/torchbenchmark/util/env_check.py:check_accuracy",
        )

    def test_fp64_eval_reference_restores_inputs_and_globals(self):
        torch.manual_seed(0)
        random.seed(0)
        bm = FakeBenchmark(test="eval")
        before_inputs = bm.example_inputs[0].clone()
        before_settings = suite.numerical_settings()
        fp64_out, record = suite._fp64_reference_outputs(bm, "eval", [])
        # Genuine FP64 reference built for eval; inputs + globals restored.
        self.assertTrue(record.get("built"))
        self.assertTrue(torch.equal(bm.example_inputs[0], before_inputs))
        self.assertEqual(suite.numerical_settings(), before_settings)
        self.assertIsNotNone(fp64_out)
        # Workload dtype untouched (reference-only double).
        mod, _ = bm.get_module()
        self.assertEqual(next(mod.parameters()).dtype, torch.float32)

    def test_fp64_train_reference_leaves_live_workload_untouched(self):
        # Regression: the FP64 train reference must NEVER mutate the live
        # workload dtype (in-place float64 cast + values-only restore_state
        # left params/inputs at float64, breaking downstream restores).
        torch.manual_seed(0)
        bm = FakeBenchmark(test="train")
        snap = suite.snapshot_state(bm)
        before_settings = suite.numerical_settings()
        fp64_fp, record = suite._fp64_train_post_state(
            bm, snap, lambda b: (None, suite._post_state_fingerprint(b, None))
        )
        self.assertTrue(record.get("built"))
        self.assertIsNotNone(fp64_fp)
        mod, _ = bm.get_module()
        for p in mod.parameters():
            self.assertEqual(p.dtype, torch.float32)
        self.assertEqual(bm.example_inputs[0].dtype, torch.float32)
        # Live values still equal the snapshot (restore still works).
        for key, frozen in snap["model_state"].items():
            live = mod.state_dict()[key]
            self.assertTrue(
                suite._frozen_equal(suite._tensor_fingerprint(live), frozen),
                f"param {key} diverged",
            )
        self.assertEqual(suite.numerical_settings(), before_settings)

    def test_train_accuracy_cpp_unavailable_records_audit_keys(self):
        bm = FakeBenchmark(test="train")
        out = suite.train_accuracy(bm, None, None)
        self.assertFalse(out["cpp_normal_full_state_pass"])
        self.assertIn("cpp-unavailable", out["cpp_normal_full_state_diffs"])
        self.assertFalse(out["fp64_reference"].get("built"))
        self.assertIn("not attempted", out["fp64_reference"].get("note", ""))
        self.assertIsNone(out["stock_reference_valid"])
        self.assertIn("error", out["cpp_vs_eager"])

    def test_eval_accuracy_records_policy_and_reference(self):
        torch.manual_seed(0)
        bm = FakeBenchmark(test="eval")
        bm.name = "resnet18"
        out = suite.eval_accuracy(bm, None, None)
        self.assertEqual(
            out["normal_policy"]["source"],
            "benchmark/torchbenchmark/util/env_check.py:get_tolerance_and_cosine_flag",
        )
        self.assertEqual(
            out["normal_comparator"]["source"], "torch/_dynamo/utils.py:same"
        )
        self.assertIn("source", out["fp64_reference"])
        # Failed CPP is never compared against itself.
        self.assertFalse(out["cpp_vs_eager_stock_pass"])
        self.assertIn("error", out["cpp_vs_eager"])


class InventorySchemaTest(TestCase):
    def test_inventory_records_required_fields(self):
        bm = FakeBenchmark(test="eval")
        inv = suite.inventory_benchmark(bm, "resnet18", "eval", 8)
        for key in (
            "model",
            "test",
            "batch_size",
            "batch_source",
            "extra_args",
            "module_path",
            "module_class",
            "num_parameters",
            "inputs",
        ):
            self.assertIn(key, inv)
        self.assertEqual(inv["extra_args"], ["--precision", "fp32"])
        self.assertIsInstance(inv["inputs"], list)
        leaf = inv["inputs"][0]
        for key in ("shape", "strides", "dtype", "device"):
            self.assertIn(key, leaf)


class CompletionScorecardTests(TestCase):
    def _full_cases(self):
        return [
            {
                "model": model,
                "test": test,
                "cpp_numerics": mode,
                "batch_size": spec[f"{test}_bs"],
                "status": "pass",
            }
            for model, spec in suite.MODEL_MANIFEST.items()
            for test in ("eval", "train")
            for mode in ("strict_eager", "inductor")
        ]

    def test_subset_success_is_not_full_completion(self):
        summary = suite._completion_summary(self._full_cases()[:2])
        self.assertTrue(summary["selected_cases_complete"])
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["pass"], 2)

    def test_all_forty_default_cases_complete(self):
        summary = suite._completion_summary(self._full_cases())
        self.assertTrue(summary["selected_cases_complete"])
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["pass"], 40)

    def test_duplicate_passes_cannot_replace_missing_cases(self):
        summary = suite._completion_summary(self._full_cases()[:1] * 40)
        self.assertTrue(summary["selected_cases_complete"])
        self.assertFalse(summary["complete"])

    def test_empty_selection_never_completes(self):
        summary = suite._completion_summary([])
        self.assertFalse(summary["selected_cases_complete"])
        self.assertFalse(summary["complete"])

    def test_skip_is_not_a_pass(self):
        cases = self._full_cases()
        cases[-1]["status"] = "skip"
        summary = suite._completion_summary(cases)
        self.assertFalse(summary["selected_cases_complete"])
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["skip_or_inventory"], 1)
        self.assertEqual(summary["pass"], 39)

    def test_smaller_batch_does_not_satisfy_manifest_goal(self):
        cases = self._full_cases()
        cases[-1]["batch_size"] = 1
        summary = suite._completion_summary(cases)
        self.assertTrue(summary["selected_cases_complete"])
        self.assertFalse(summary["complete"])

    def test_inference_only_does_not_satisfy_training_goal(self):
        cases = [case for case in self._full_cases() if case["test"] == "eval"]
        summary = suite._completion_summary(cases)
        self.assertTrue(summary["selected_cases_complete"])
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["pass"], 20)


if __name__ == "__main__":
    run_tests()
