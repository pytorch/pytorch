import os
import tempfile
import unittest


os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from nonstrict import compile_artifact, CompiledArtifact, jit, make_fx, Target

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


try:
    import jax
    import jax.numpy as jnp
except Exception as e:
    jax = None
    jnp = None
    JAX_IMPORT_ERROR = e
else:
    JAX_IMPORT_ERROR = None


def _identity_backend(gm, example_inputs):
    return gm


@unittest.skipIf(jax is None, f"JAX unavailable: {JAX_IMPORT_ERROR}")
class TestNonstrictJitAgainstJax(TestCase):
    def test_make_fx_static_argnames_pytree_inputs(self):
        def f(xs, *, scale):
            return {"y": xs["x"] * scale}

        gm = make_fx(f, static_argnames="scale")({"x": torch.ones(2)}, scale=2)
        self.assertEqual(gm(torch.ones(2))[0], torch.tensor([2.0, 2.0]))
        self.assertTrue(hasattr(gm, "_nonstrict_jit_in_spec"))
        self.assertTrue(hasattr(gm, "_nonstrict_jit_out_spec"))

    def test_make_fx_always_uses_fake_tracing_mode(self):
        def f(x):
            return x + 1

        gm = make_fx(f)(torch.ones(2))
        placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
        self.assertEqual(len(placeholders), 1)
        from torch._subclasses.fake_tensor import FakeTensor

        self.assertIsInstance(placeholders[0].meta["val"], FakeTensor)

    def test_make_fx_rejects_non_tensor_dynamic_args(self):
        with self.assertRaisesRegex(TypeError, "dynamic arguments must be Tensors"):
            make_fx(lambda x, label: x)(torch.ones(2), "abc")

    def test_jit_static_argnames_pytree_io_and_cache(self):
        torch_traces = 0
        jax_traces = 0

        def torch_f(xs, *, scale):
            nonlocal torch_traces
            torch_traces += 1
            return {"y": xs["x"] * scale}

        def jax_f(xs, *, scale):
            nonlocal jax_traces
            jax_traces += 1
            return {"y": xs["x"] * scale}

        torch_jit = jit(torch_f, static_argnames="scale", backend=_identity_backend)
        jax_jit = jax.jit(jax_f, static_argnames="scale")

        torch_x = torch.ones(2)
        jax_x = jnp.ones(2, dtype=jnp.float32)
        self.assertEqual(
            torch_jit({"x": torch_x}, scale=2)["y"], torch.tensor([2.0, 2.0])
        )
        self.assertTrue(
            (jax_jit({"x": jax_x}, scale=2)["y"] == jnp.array([2.0, 2.0])).all()
        )
        self.assertEqual(torch_traces, 1)
        self.assertEqual(jax_traces, 1)

        torch_jit({"x": torch_x + 1}, scale=2)
        jax_jit({"x": jax_x + 1}, scale=2)
        self.assertEqual(torch_traces, 1)
        self.assertEqual(jax_traces, 1)

        torch_jit({"x": torch.ones(3)}, scale=2)
        jax_jit({"x": jnp.ones(3, dtype=jnp.float32)}, scale=2)
        self.assertEqual(torch_traces, 2)
        self.assertEqual(jax_traces, 2)

        torch_jit({"x": torch.ones(3)}, scale=3)
        jax_jit({"x": jnp.ones(3, dtype=jnp.float32)}, scale=3)
        self.assertEqual(torch_traces, 3)
        self.assertEqual(jax_traces, 3)

    def test_jit_rejects_non_tensor_dynamic_args_like_jax_rejects_non_array_args(self):
        with self.assertRaisesRegex(TypeError, "dynamic arguments must be Tensors"):
            jit(lambda x, label: x, backend=_identity_backend)(torch.ones(2), "abc")

        with self.assertRaises(TypeError):
            jax.jit(lambda x, label: x)(jnp.ones(2, dtype=jnp.float32), "abc")

    def test_jax_accepts_dynamic_python_scalars_but_this_prototype_does_not(self):
        self.assertTrue(
            (
                jax.jit(lambda x, n: x + n)(jnp.ones(2, dtype=jnp.float32), 2)
                == jnp.array([3.0, 3.0])
            ).all()
        )
        self.assertTrue(
            (
                jax.jit(lambda x, n: x + n)(jnp.ones(2, dtype=jnp.float32), 2.0)
                == jnp.array([3.0, 3.0])
            ).all()
        )

        with self.assertRaisesRegex(TypeError, "dynamic arguments must be Tensors"):
            jit(lambda x, n: x + n, backend=_identity_backend)(torch.ones(2), 2)
        with self.assertRaisesRegex(TypeError, "dynamic arguments must be Tensors"):
            jit(lambda x, n: x + n, backend=_identity_backend)(torch.ones(2), 2.0)

    def test_jit_requires_hashable_static_args_like_jax(self):
        with self.assertRaisesRegex(TypeError, "must be hashable"):
            jit(lambda x, scale: x, static_argnames="scale", backend=_identity_backend)(
                torch.ones(2), []
            )

        with self.assertRaisesRegex(ValueError, "Non-hashable static arguments"):
            jax.jit(lambda x, scale: x, static_argnames="scale")(
                jnp.ones(2, dtype=jnp.float32), []
            )

    def test_compile_artifact_static_args_are_bound_at_materialization_time(self):
        def torch_f(xs, *, target):
            return {"y": xs["x"] * target}

        torch_artifact = compile_artifact(torch_f, static_argnames="target")(
            {"x": torch.ones(2)}, target=2
        )

        self.assertEqual(
            torch_artifact.run({"x": torch.ones(2)})["y"],
            torch.tensor([2.0, 2.0]),
        )

        with self.assertRaises(TypeError):
            torch_artifact.run({"x": torch.ones(2)}, target=2)

    def test_compile_artifact_rejects_non_inductor_backend(self):
        materialize = compile_artifact(lambda x: x + 1, backend=_identity_backend)
        with self.assertRaisesRegex(NotImplementedError, "backend='inductor'"):
            materialize(torch.ones(2))

    def test_compile_artifact_is_curried_function(self):
        x = torch.ones(2)
        artifact = compile_artifact(lambda x: x + 1)(x)
        self.assertEqual(artifact.run(x), torch.tensor([2.0, 2.0]))

    def test_compile_artifact_save_load(self):
        def torch_f(x):
            return x.sin() + 1

        x = torch.randn(4)
        torch_artifact = compile_artifact(torch_f)(x)
        self.assertEqual(torch_artifact.target, Target.current())

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "artifact.nonstrict")
            torch_artifact.save(path)
            loaded = CompiledArtifact.load(path)
            self.assertEqual(loaded.run(x), torch_f(x))


if __name__ == "__main__":
    run_tests()
