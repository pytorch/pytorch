# Owner(s): ["oncall: distributed"]

import io
from contextlib import contextmanager

from test_c10d_pybackend import create_process_group, RecordingBackend

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
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
            "all_reduce_coalesced",
            "all_reduce_coalesced_",
            "all_gather_into_tensor_coalesced",
            "reduce_scatter_tensor_coalesced",
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

    def test_config_dict(self):
        try:
            from nccl.core import NCCLCollConfig
        except ImportError:
            self.skipTest("nccl4py required")
        from torch.distributed._collective_config import _collective_config_dict

        config = NCCLCollConfig(min_ctas=1, max_ctas=2, force_alg_selection=False)
        values = _collective_config_dict(config)
        expected = dict(config.__dict__)
        del expected["vendor_options"]
        self.assertEqual(values, expected)
        self.assertEqual(NCCLCollConfig(**values), config)
        config.max_ctas = 4
        self.assertEqual(values["max_ctas"], 2)
        self.assertIsNone(_collective_config_dict(None))
        self.assertEqual(_collective_config_dict({"min_ctas": 2}), {"min_ctas": 2})
        with self.assertRaises(TypeError):
            _collective_config_dict(object())

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

    @parametrize("frontend", ["eager", "aot_eager", "inductor"])
    @parametrize(
        "name",
        [
            "all_reduce_coalesced",
            "all_gather_into_tensor_coalesced",
            "reduce_scatter_tensor_coalesced",
        ],
    )
    def test_coalesced_config(self, frontend, name):
        with self._backend() as (backend, group):
            tensors = [torch.ones(4, requires_grad=True) for _ in range(2)]
            config = {"min_ctas": 2}
            group_name = group.group_name

            op = getattr(torch.ops._c10d_functional, name)
            args = {
                "all_reduce_coalesced": ("sum",),
                "all_gather_into_tensor_coalesced": (1,),
                "reduce_scatter_tensor_coalesced": ("sum", 1),
            }[name]

            def fn(tensors):
                return op(tensors, *args, group_name, config)

            fn = fn if frontend == "eager" else torch.compile(fn, backend=frontend)
            results = fn(tensors)
            sum(result.sum() for result in results).backward()
            expected = 4.0 if name == "all_reduce_coalesced" else 1.0
            for tensor in tensors:
                self.assertEqual(tensor.grad, torch.full((4,), expected))
            self.assertEqual(
                [call[-1].config for call in backend.calls], [config, config]
            )

    def test_config_value_key(self):
        from torch._inductor.utils import collective_config_value_key

        keys = [
            collective_config_value_key(config)
            for config in (
                None,
                {"max_ctas": 1},
                {"max_ctas": True},
                {"max_ctas": 1.0},
                {"vendor_options": [1]},
                {"vendor_options": (1,)},
            )
        ]
        self.assertEqual(len(set(keys)), len(keys))
        self.assertEqual(
            collective_config_value_key({"a": 1, "b": {"c": [2]}}),
            collective_config_value_key({"b": {"c": [2]}, "a": 1}),
        )

    def test_nested_config_dict(self):
        from torch.distributed._collective_config import _collective_config_dict

        config = {"a": {"b": [1, (2.0, True)]}, "c": (), "d": []}
        self.assertEqual(
            _collective_config_dict(config), {"a": {"b": [1, (2.0, True)]}}
        )

    def test_recompile_on_config_change(self):
        from torch._dynamo.testing import CompileCounter

        with self._backend() as (backend, group):
            counter = CompileCounter()

            @torch.compile(backend=counter, fullgraph=True)
            def fn(x, config):
                return funcol.all_reduce(x, "sum", group, config=config)

            configs = [{"max_ctas": 1}, {"max_ctas": 2}, {"max_ctas": 2}]
            for config in configs:
                fn(torch.ones(4), config)
            self.assertEqual(counter.frame_count, 2)
            self.assertEqual([call[-1].config for call in backend.calls], configs)

    def test_bucket_keys(self):
        from torch._inductor.fx_passes import bucketing

        c10d = torch.ops._c10d_functional
        cases = (
            (bucketing._ar_group_key, c10d.all_reduce.default, ("sum",)),
            (bucketing._ag_group_key, c10d.all_gather_into_tensor.default, (1,)),
            (
                bucketing._ag_group_key_multidtype,
                c10d.all_gather_into_tensor.default,
                (1,),
            ),
            (
                bucketing._rs_group_key,
                c10d.reduce_scatter_tensor.default,
                ("sum", 1),
            ),
        )
        with self._backend() as (backend, group):
            graph = torch.fx.Graph()
            x = graph.placeholder("x")
            for key, op, args in cases:
                keys = set()
                for config in (None, {"max_ctas": 1}, {"max_ctas": True}):
                    node = graph.call_function(op, (x, *args, group.group_name, config))
                    node.meta["val"] = torch.empty(4)
                    keys.add(key(node))
                self.assertEqual(len(keys), 3)

    def test_decomp_skips_config(self):
        from torch._inductor.fx_passes.decomp_comms import find_all_gather_ancestor

        c10d = torch.ops._c10d_functional
        for config in (None, {"max_ctas": 1}):
            graph = torch.fx.Graph()
            x = graph.placeholder("x")
            ag = graph.call_function(
                c10d.all_gather_into_tensor.default, (x, 2, "group", config)
            )
            wait = graph.call_function(c10d.wait_tensor.default, (ag,))
            info = find_all_gather_ancestor(wait)
            if config is None:
                self.assertIs(info.ag_node, ag)
            else:
                self.assertIsNone(info)


class LocalTensorCollectiveConfigTest(TestCase):
    world_size = 2

    def setUp(self):
        super().setUp()
        dist.init_process_group("fake", rank=0, world_size=self.world_size)

    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()

    def test_config(self):
        from torch.distributed._local_tensor import LocalTensor, LocalTensorMode

        config = {"max_ctas": 1}
        group_name = dist.group.WORLD.group_name
        c10d = torch.ops._c10d_functional
        with LocalTensorMode(self.world_size):
            x = LocalTensor({rank: torch.full((2,), float(rank)) for rank in range(2)})
            results = [
                c10d.all_reduce(x, "sum", group_name, config),
                c10d.all_gather_into_tensor(x, 2, group_name, config),
                c10d.reduce_scatter_tensor(x, "sum", 2, group_name, config),
                c10d.all_to_all_single(x, [1, 1], [1, 1], group_name, config),
            ]
            results = [c10d.wait_tensor(result) for result in results]
            y = x.clone()
            dist.all_reduce(y, config=config)
        expected = [[1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0], [0.0, 1.0]]
        for result, values in zip(results, expected):
            self.assertEqual(result._local_tensors[0], torch.tensor(values))
        self.assertEqual(y._local_tensors[1], torch.ones(2))


if __name__ == "__main__":
    run_tests()
