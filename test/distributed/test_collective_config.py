# Owner(s): ["oncall: distributed"]

import io
from contextlib import contextmanager

from test_c10d_pybackend import create_process_group, RecordingBackend

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives
from torch._C._distributed_c10d import (
    _register_process_group,
    _unregister_process_group,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class FunctionalCollectiveConfigTest(TestCase):
    def test_optional_config_schemas(self):
        for name in (
            "all_reduce",
            "all_gather_into_tensor",
            "reduce_scatter_tensor",
            "all_to_all_single",
        ):
            op = getattr(torch.ops._c10d_functional, name).default
            argument = op._schema.arguments[-1]
            self.assertEqual(argument.name, "config")
            self.assertTrue(argument.has_default_value())
            self.assertIsNone(argument.default_value)
            self.assertFalse(hasattr(torch.ops._c10d_functional, name + "_config"))
            from torch._inductor.utils import get_collective_config

            graph = torch.fx.Graph()
            args = (None,) * (len(op._schema.arguments) - 1)
            for config in (None, {}, {"min_ctas": 2}):
                positional = graph.call_function(op, (*args, config))
                keyword = graph.call_function(op, args, {"config": config})
                self.assertEqual(get_collective_config(positional), config)
                self.assertEqual(get_collective_config(keyword), config)

    @contextmanager
    def _backend(self, device="cpu", supports_config=True):
        class RejectingBackend(RecordingBackend):
            def allreduce(self, tensors, opts):
                if opts.config is not None:
                    raise RuntimeError(
                        "backend does not support per-collective configuration"
                    )
                return super().allreduce(tensors, opts)

        backend_type = RecordingBackend if supports_config else RejectingBackend
        backend = backend_type(0, 1)
        group = create_process_group(backend)
        if device == "cuda":
            if not torch.cuda.is_available():
                self.skipTest("CUDA required")
            group._register_backend(
                torch.device("cuda"), dist.ProcessGroup.BackendType.CUSTOM, backend
            )
        _register_process_group(group.group_name, group)
        try:
            yield backend, group
        finally:
            _unregister_process_group(group.group_name)
            torch._dynamo.reset()

    @parametrize(
        "name",
        [
            "all_reduce",
            "all_gather_into_tensor",
            "reduce_scatter_tensor",
            "all_to_all_single",
        ],
    )
    @parametrize(
        "frontend",
        ["eager", "aot_eager", "inductor", "export_strict", "export_nonstrict"],
    )
    def test_config_survives(self, name, frontend):
        config = {
            "min_ctas": 1,
            "max_ctas": None,
            "alg_selection": "ring",
            "force_alg_selection": False,
            "vendor_count": 1,
            "vendor_0_vendor_id": 7,
            "vendor_0_option_id": 8,
            "vendor_0_str_value": "value",
        }
        op = getattr(torch.ops._c10d_functional, name)
        args = {
            "all_reduce": ("sum",),
            "all_gather_into_tensor": (1,),
            "reduce_scatter_tensor": ("sum", 1),
            "all_to_all_single": ([4], [4]),
        }[name]
        with self._backend() as (backend, group):
            group_name = group.group_name

            class Module(torch.nn.Module):
                def forward(self, tensor):
                    return torch.ops._c10d_functional.wait_tensor(
                        op(tensor, *args, group_name, config)
                    )

            tensor = torch.ones(4)
            if frontend.startswith("export"):
                exported = torch.export.export(
                    Module(), (tensor,), strict=frontend == "export_strict"
                )
                buffer = io.BytesIO()
                torch.export.save(exported, buffer)
                buffer.seek(0)
                module = torch.export.load(buffer).module()
            else:
                module = torch.compile(Module(), backend=frontend, fullgraph=True)
            result = module(tensor)
            self.assertEqual(result, tensor + 2 if name == "all_reduce" else tensor)
            self.assertEqual(tensor, torch.ones(4))
            self.assertEqual(len(backend.calls), 1)
            self.assertEqual(backend.calls[0][-1].config, config)
            self.assertEqual(backend.wait_count, 1)

    def test_unsupported_backend(self):
        with self._backend(supports_config=False) as (backend, group):
            with self.assertRaisesRegex(
                RuntimeError, "does not support per-collective configuration"
            ):
                torch.ops._c10d_functional.all_reduce(
                    torch.ones(4), "sum", group.group_name, {"min_ctas": 1}
                )
            self.assertEqual(backend.calls, [])

    @parametrize("frontend", ["make_fx", "rewrite", "inductor"])
    @parametrize("tracing_mode", ["real", "fake"])
    def test_raw_config_tracing(self, frontend, tracing_mode):
        from torch._guards import tracing, TracingContext
        from torch._inductor._functionalize_collectives import (
            _functionalize_inplace_collectives,
            _unbox_process_group_torchbinds,
        )
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.passes.regional_inductor import regional_inductor

        with self._backend() as (backend, group):
            config = {"user_profiler_tag": 7}
            pg = group.boxed()
            reduce_op = dist.ReduceOp(dist.ReduceOp.SUM).boxed()

            def fn(x):
                output = x.clone()
                torch.ops.c10d.allreduce_(
                    [output], pg, reduce_op, None, False, config=config
                )
                return output + 1

            expected = fn(torch.ones(4))
            gm = make_fx(fn, tracing_mode=tracing_mode)(torch.ones(4))
            backend.calls.clear()
            if frontend == "rewrite":
                _functionalize_inplace_collectives(gm)
                _unbox_process_group_torchbinds(gm)
            elif frontend == "inductor":
                for node in gm.graph.nodes:
                    if node.op not in ("placeholder", "output"):
                        node.meta.setdefault("custom", {})["compile_with_inductor"] = {
                            "inductor_configs": {}
                        }
                fake_mode = next(
                    n.meta["val"].fake_mode
                    for n in gm.graph.nodes
                    if n.op == "placeholder"
                )
                with tracing(TracingContext(fake_mode)):
                    gm = regional_inductor(gm)
            result = (
                gm([torch.ones(4)]) if frontend == "inductor" else gm(torch.ones(4))
            )
            self.assertEqual(result, expected)
            self.assertEqual([call[-1].config for call in backend.calls], [config])

    def test_serialization(self):
        try:
            from nccl.core import NCCLCollConfig, VendorOption
        except ImportError:
            self.skipTest("nccl4py required")
        from torch.distributed._collective_config import (
            _deserialize_nccl_config,
            _serialize_nccl_config,
        )

        config = NCCLCollConfig(
            min_ctas=1,
            max_ctas=2,
            force_alg_selection=False,
            vendor_options=(
                VendorOption(1, 2, int_value=3),
                VendorOption(4, 5, str_value="value"),
            ),
        )
        values = _serialize_nccl_config(config)
        self.assertEqual(_deserialize_nccl_config(values), config)
        config.max_ctas = 4
        self.assertEqual(values["max_ctas"], 2)
        with self.assertRaisesRegex(TypeError, "config must be"):
            _serialize_nccl_config(object())
        config.vendor_options = (VendorOption(1, 2, raw_value=1),)
        with self.assertRaisesRegex(NotImplementedError, "raw-pointer vendor options"):
            _serialize_nccl_config(config)

    @parametrize("frontend", ["eager", "aot_eager", "inductor"])
    @parametrize(
        "name",
        [
            "all_reduce",
            "all_gather_into_tensor",
            "reduce_scatter_tensor",
            "all_to_all_single",
        ],
    )
    def test_backward_preserves_config(self, frontend, name):
        with self._backend() as (backend, group):
            tensor = torch.ones(4, requires_grad=True)
            config = {"min_ctas": 2}
            group_name = group.group_name

            op = getattr(torch.ops._c10d_functional, name)
            args = {
                "all_reduce": ("sum",),
                "all_gather_into_tensor": (1,),
                "reduce_scatter_tensor": ("sum", 1),
                "all_to_all_single": ([4], [4]),
            }[name]

            def fn(tensor):
                return op(tensor, *args, group_name, config)

            result = torch.compile(fn, backend=frontend, fullgraph=True)(tensor)
            result.sum().backward()
            self.assertEqual(
                tensor.grad, torch.full((4,), 3.0 if name == "all_reduce" else 1.0)
            )
            self.assertEqual(
                [call[-1].config for call in backend.calls], [config, config]
            )


if __name__ == "__main__":
    run_tests()
