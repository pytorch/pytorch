from __future__ import annotations

import os
import tempfile
import textwrap
import unittest

import yaml

from torchgen import native_aot
from torchgen.dest.register_dispatch_key import RegisterDispatchKey
from torchgen.gen import (
    get_grouped_native_functions,
    LineLoader,
    parse_native_yaml_struct,
)
from torchgen.model import DispatchKey, NativeFunctionsGroup
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import Target


# A minimal structured group (out declares the kernel), an unstructured op for
# validation failures, and a two-overload structured base (embmulti.alpha/.beta).
NATIVE_YAML = """\
- func: embfoo.out(Tensor self, int k, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  dispatch:
    CUDA: embfoo_out_cuda

- func: embfoo(Tensor self, int k) -> Tensor
  structured_delegate: embfoo.out
  dispatch: {}

- func: embbar(Tensor self) -> Tensor
  dispatch:
    CUDA: embbar_cuda

- func: embmulti.alpha_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  dispatch:
    CUDA: embmulti_alpha_out_cuda

- func: embmulti.alpha(Tensor self) -> Tensor
  structured_delegate: embmulti.alpha_out
  dispatch: {}

- func: embmulti.beta_out(Tensor self, int k, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  dispatch:
    CUDA: embmulti_beta_out_cuda

- func: embmulti.beta(Tensor self, int k) -> Tensor
  structured_delegate: embmulti.beta_out
  dispatch: {}
"""


def _parse_fixture():
    es = yaml.load(NATIVE_YAML, Loader=LineLoader)
    parsed = parse_native_yaml_struct(
        es, set(), path="fixture", skip_native_fns_gen=True
    )
    return parsed


_MIN_DECL = """\
ATEN_OP = "{op}"
DISPATCH_KEY = "CUDA"
KERNEL_MODULE = "kernel.py"


def kernel_precompile_grid():
    return [{{"N": 1}}]


def covered_axes(self):
    return {{}}


def cpp_dispatch(spec):
    return "true"


def cpp_launch(spec, launch_fn):
    return f"{{launch_fn}}();"
"""


def _write_declaration(ops_dir: str, op: str, body: str) -> None:
    op_dir = os.path.join(ops_dir, op)
    os.makedirs(op_dir, exist_ok=True)
    with open(os.path.join(op_dir, "aot.py"), "w") as f:
        f.write(body)


class TestDeclarationParsing(unittest.TestCase):
    def test_parse_minimal_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write_declaration(d, "embfoo", _MIN_DECL.format(op="embfoo"))
            manifests = native_aot.parse_native_aot_manifests(d)
        self.assertEqual(list(manifests), [(DispatchKey.CUDA, "embfoo")])
        m = manifests[(DispatchKey.CUDA, "embfoo")]
        self.assertEqual(m.stub_name(), "embfoo_aot_stub")
        self.assertEqual(m.fn_type_name(), "embfoo_aot_fn")

    def test_missing_dir_is_empty(self) -> None:
        self.assertEqual(native_aot.parse_native_aot_manifests("/nonexistent"), {})

    def test_overload_qualified_op_accepted(self) -> None:
        # decl_id sanitizes the dot, since it names a directory.
        with tempfile.TemporaryDirectory() as d:
            _write_declaration(d, "embfoo", _MIN_DECL.format(op="embfoo.out"))
            manifests = native_aot.parse_native_aot_manifests(d)
            (m,) = manifests.values()
            self.assertEqual(m.op, "embfoo.out")
            self.assertEqual(m.stub_name(), "embfoo_out_aot_stub")
            self.assertEqual(m.fn_type_name(), "embfoo_out_aot_fn")

    def test_missing_exports_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write_declaration(d, "embfoo", 'ATEN_OP = "embfoo"\n')
            with self.assertRaisesRegex(RuntimeError, "DISPATCH_KEY"):
                native_aot.parse_native_aot_manifests(d)

    def test_unconditional_defaults_false_and_is_read(self) -> None:
        # The default decides whether set_aot_enabled(False) reaches an op that did
        # not declare the flag, so reading a missing attribute as True unmasks all.
        with tempfile.TemporaryDirectory() as d:
            _write_declaration(d, "embfoo", _MIN_DECL.format(op="embfoo"))
            (m,) = native_aot.parse_native_aot_manifests(d).values()
            self.assertFalse(m.unconditional)
        with tempfile.TemporaryDirectory() as d:
            _write_declaration(
                d, "embfoo", _MIN_DECL.format(op="embfoo") + "UNCONDITIONAL = True\n"
            )
            (m,) = native_aot.parse_native_aot_manifests(d).values()
            self.assertTrue(m.unconditional)

    def test_non_bool_unconditional_rejected(self) -> None:
        # Truthiness would accept "false" or 0/1 as True.
        with tempfile.TemporaryDirectory() as d:
            _write_declaration(
                d, "embfoo", _MIN_DECL.format(op="embfoo") + 'UNCONDITIONAL = "yes"\n'
            )
            with self.assertRaisesRegex(RuntimeError, "UNCONDITIONAL must be a bool"):
                native_aot.parse_native_aot_manifests(d)

    def test_spec_arity_convention_enforced(self) -> None:
        # cpp_dispatch is called once per spec point, so it must take one.
        bad = _MIN_DECL.format(op="embfoo").replace(
            "def cpp_dispatch(spec):", "def cpp_dispatch():"
        )
        with tempfile.TemporaryDirectory() as d:
            _write_declaration(d, "embfoo", bad)
            with self.assertRaisesRegex(RuntimeError, "cpp_dispatch.*per-point"):
                native_aot.parse_native_aot_manifests(d)


class TestManifestValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.grouped = get_grouped_native_functions(_parse_fixture().native_functions)

    def test_structured_op_accepted(self) -> None:
        m = native_aot.NativeAotManifest(op="embfoo", dispatch_key=DispatchKey.CUDA)
        native_aot.validate_native_aot_manifests(
            {(DispatchKey.CUDA, "embfoo"): m}, self.grouped
        )

    def test_unstructured_op_rejected(self) -> None:
        m = native_aot.NativeAotManifest(op="embbar", dispatch_key=DispatchKey.CUDA)
        with self.assertRaisesRegex(RuntimeError, "not a structured op"):
            native_aot.validate_native_aot_manifests(
                {(DispatchKey.CUDA, "embbar"): m}, self.grouped
            )

    def test_unknown_op_rejected(self) -> None:
        m = native_aot.NativeAotManifest(op="embmissing", dispatch_key=DispatchKey.CUDA)
        with self.assertRaisesRegex(RuntimeError, "not a structured op"):
            native_aot.validate_native_aot_manifests(
                {(DispatchKey.CUDA, "embmissing"): m}, self.grouped
            )

    def test_ambiguous_base_name_rejected(self) -> None:
        # embmulti has two structured overloads, so the error must say to qualify.
        m = native_aot.NativeAotManifest(op="embmulti", dispatch_key=DispatchKey.CUDA)
        with self.assertRaisesRegex(RuntimeError, "ambiguous.*qualify"):
            native_aot.validate_native_aot_manifests(
                {(DispatchKey.CUDA, "embmulti"): m}, self.grouped
            )

    def test_qualified_overload_accepted(self) -> None:
        m = native_aot.NativeAotManifest(
            op="embmulti.beta", dispatch_key=DispatchKey.CUDA
        )
        native_aot.validate_native_aot_manifests(
            {(DispatchKey.CUDA, "embmulti.beta"): m}, self.grouped
        )

    def test_qualified_unknown_overload_rejected(self) -> None:
        m = native_aot.NativeAotManifest(
            op="embmulti.gamma", dispatch_key=DispatchKey.CUDA
        )
        with self.assertRaisesRegex(RuntimeError, "no structured group named"):
            native_aot.validate_native_aot_manifests(
                {(DispatchKey.CUDA, "embmulti.gamma"): m}, self.grouped
            )


class TestHookCodegen(unittest.TestCase):
    def setUp(self) -> None:
        parsed = _parse_fixture()
        self.group = next(
            g
            for g in get_grouped_native_functions(parsed.native_functions)
            if isinstance(g, NativeFunctionsGroup)
        )
        self.manifest = native_aot.NativeAotManifest(
            op="embfoo", dispatch_key=DispatchKey.CUDA
        )
        self.backend_index = parsed.backend_indices[DispatchKey.CUDA]

    def test_stub_declaration(self) -> None:
        decl = native_aot.gen_stub_declaration(self.manifest, self.group)
        self.assertExpectedInline(
            decl,
            """\
using embfoo_aot_fn = bool (*)(const at::Tensor & self, int64_t k, const at::Tensor & out);
DECLARE_DISPATCH(embfoo_aot_fn, embfoo_aot_stub)
""",
        )

    def test_stub_definition(self) -> None:
        defn = native_aot.gen_stub_definition(self.manifest)
        self.assertExpectedInline(
            defn,
            """\
DEFINE_DISPATCH(embfoo_aot_stub);
REGISTER_NO_CPU_DISPATCH(embfoo_aot_stub)
""",
        )

    def _wrapper_body(self, with_manifest: bool) -> str:
        manifests = {"embfoo": self.manifest} if with_manifest else {}
        gen = RegisterDispatchKey(
            self.backend_index,
            Target.ANONYMOUS_DEFINITION,
            SelectiveBuilder.get_nop_selector(),
            rocm=False,
            symint=True,
            class_method_name=None,
            skip_dispatcher_op_registration=False,
            native_aot_manifests=manifests,
        )
        return "\n".join(gen(self.group))

    def test_wrapper_emits_stub_consultation(self) -> None:
        body = self._wrapper_body(with_manifest=True)
        # Nothing in `stub(device, args...)` says it launches a kernel, and the
        # short-circuit is what makes op.impl the fallback.
        self.assertIn("// native-AOT: the last conjunct is the LAUNCH", body)
        self.assertIn("&& short-circuits", body)
        self.assertIn(
            "if (!(at::globalContext().allowNativeAot() && at::native::embfoo_aot_stub.is_device_supported(c10::DeviceType::CUDA) && at::native::embfoo_aot_stub(c10::DeviceType::CUDA, self, k, op.outputs_[0]))) { op.impl(self, k, op.outputs_[0]); }",
            body,
        )
        # out= variant funnels through the same stub.
        self.assertIn(
            "if (!(at::globalContext().allowNativeAot() && at::native::embfoo_aot_stub.is_device_supported(c10::DeviceType::CUDA) && at::native::embfoo_aot_stub(c10::DeviceType::CUDA, self, k, op.maybe_get_output(0)))) { op.impl(self, k, op.maybe_get_output(0)); }",
            body,
        )

    def test_wrapper_unchanged_without_manifest(self) -> None:
        body = self._wrapper_body(with_manifest=False)
        self.assertNotIn("aot_stub", body)
        self.assertIn("op.impl(self, k, op.outputs_[0]);", body)

    def test_unconditional_op_gates_on_the_private_mask(self) -> None:
        # These kernels are the implementation, so set_aot_enabled(False) must not
        # reach them; the private mask is still needed to reach stock aten.
        self.manifest = native_aot.NativeAotManifest(
            op="embfoo", dispatch_key=DispatchKey.CUDA, unconditional=True
        )
        body = self._wrapper_body(with_manifest=True)
        self.assertNotIn("allowNativeAot()", body)
        self.assertIn("// declared UNCONDITIONAL", body)
        self.assertIn(
            "if (!(!at::globalContext().maskUnconditionalNativeAot() && at::native::embfoo_aot_stub.is_device_supported(c10::DeviceType::CUDA) && at::native::embfoo_aot_stub(c10::DeviceType::CUDA, self, k, op.outputs_[0]))) { op.impl(self, k, op.outputs_[0]); }",
            body,
        )
        # Unconditional constrains who may switch the path off, not which shapes
        # the grid covers, so declining still falls back.
        self.assertIn("op.impl(self, k, op.outputs_[0]);", body)
        # The ordinary comment enumerates "switched off, unsupported device, or
        # declined"; this op has no switched-off case, so it must not claim one.
        self.assertNotIn("switched off", body)

    # assertExpectedInline without pulling in torch's expecttest plumbing.
    def assertExpectedInline(self, actual: str, expected: str) -> None:
        self.assertEqual(actual, textwrap.dedent(expected))


if __name__ == "__main__":
    unittest.main()
