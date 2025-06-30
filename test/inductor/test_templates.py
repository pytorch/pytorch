# Owner(s): ["module: inductor"]
# ruff: noqa: F841
import contextlib
import textwrap
import unittest.mock
from typing import Callable

import torch
from torch._inductor import config
from torch._inductor.codegen.common import RemovedArg
from torch._inductor.ir import FixedLayout
from torch._inductor.select_algorithm import (
    autotune_select_algorithm,
    PartialRender,
    TritonTemplate,
    TritonTemplateKernel,
)
from torch._inductor.utils import run_and_get_kernels
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    requires_gpu,
    requires_triton,
    RUN_GPU,
)
from torch.utils._ordered_set import OrderedSet

from .test_torchinductor import TestCase


@contextlib.contextmanager
def patch_lowering(lowering_overrides) -> Callable[[], None]:
    import torch._inductor.lowering as inductor_lowering

    with unittest.mock.patch.dict(inductor_lowering.lowerings):
        # Apply overrides
        for fn, (
            decomp_fn,
            broadcast,
            type_promotion_kind,
            convert_input_to_bool,
        ) in lowering_overrides.items():
            inductor_lowering._register_lowering(
                fn,
                decomp_fn,
                broadcast=broadcast,
                type_promotion_kind=type_promotion_kind,
                convert_input_to_bool=convert_input_to_bool,
                lowering_dict=inductor_lowering.lowerings,
            )

        yield


class TestTemplateRender(TestCase):
    @requires_gpu()
    @requires_triton()
    @config.patch(cuda_backend="triton")
    def test_finalized_subclass_hooks(self):
        """
        Tests that all registered triton template hooks have been finalized,
        especially in the case that the hooks are finalized manually by the
        caller i.e. by calling template.finalize_hook(hook_name)
        """
        class ExtensionTritonTemplateKernel(TritonTemplateKernel):
            # Custom hook
            def output_ptr_var(self) -> str:
                """
                Return the variable name of the output pointer argument used in the
                kernel template
                """

                def hook() -> str:
                    assert len(self.args.output_buffers) > 0
                    # Since this template has a single output_node, pick the output buffer
                    # that has not been removed
                    for value in self.args.output_buffers.values():
                        if not isinstance(value, RemovedArg):
                            return value

                assert "<OUTPUT_PTR_VAR>" not in self.render_hooks
                self.render_hooks["<OUTPUT_PTR_VAR>"] = hook
                return "<OUTPUT_PTR_VAR>"

            def store_output(self, output_ptr_name, value, indent_width=4):
                with self.create_subgraph_body("<STORE_OUTPUT>"):
                    self.body.writeline(f"tl.store({output_ptr_name}, {value})")
                    self.codegen_body()

                self.args.output(self.output_node.get_name())

                def hook() -> str:
                    self.codegen_body()
                    self.cse.invalidate(OrderedSet())
                    return textwrap.indent(
                        self.body.getvalue(), " " * indent_width
                    ).strip()

                assert "<STORE_OUTPUT>" not in self.render_hooks
                self.render_hooks["<STORE_OUTPUT>"] = hook
                return "<STORE_OUTPUT>"

            def render(
                self, template, kwargs, record_input_dependent_tracked_event=False
            ):
                if record_input_dependent_tracked_event:
                    self.cached_replay_events = []

                template_env = {
                    fn.__name__: self.record_input_dependent_tracked_event()(fn)
                    if record_input_dependent_tracked_event
                    else fn
                    for fn in [
                        self.def_kernel,
                        self.size,
                        self.stride,
                        self.store_output,
                        self.load_input,
                        self.make_load,
                        self.modification,
                        self.gen_argdefs,
                        self.gen_defines,
                        # This function registers a hook that the scheduler does
                        # not directly finalize
                        self.output_ptr_var,
                    ]
                }
                return PartialRender(
                    template.render(**template_env, **kwargs),
                    self.render_hooks,
                )

        class ExtensionTritonTemplate(TritonTemplate):
            kernel_type = ExtensionTritonTemplateKernel

        add_template = ExtensionTritonTemplate(
            name="add",
            grid=lambda *args, **kwargs: (1, 1, 1),
            source=(
                r"""
{{def_kernel("A", "B")}}
    xoffset = tl.program_id(0)
    xindex = xoffset + tl.arange(0, XBLOCK)
    output_block_ptr = tl.make_block_ptr(
        base={{output_ptr_var()}},
        shape=[32], strides=[1], offsets=[xoffset],
        block_shape=[XBLOCK], order=[0])
    tmp0 = tl.load(A + xindex)
    tmp1 = tl.load(B + xindex)
    tmp2 = tmp0 + tmp1
    {{store_output("output_block_ptr", "tmp2")}}
    """
            ),
        )

        XBLOCK = 32

        def add_override(a, b, alpha=None):
            layout = FixedLayout(a.get_device(), a.get_dtype(), a.get_size())
            choices = []
            add_template.maybe_append_choice(
                choices,
                input_nodes=(a, b),
                layout=layout,
                num_stages=1,
                num_warps=2,
                XBLOCK=XBLOCK,
            )
            return autotune_select_algorithm("add", choices, [a, b], layout)

        with patch_lowering(
            {
                torch.ops.aten.add.Tensor: (
                    add_override,
                    True,
                    ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
                    False,
                )
            }
        ):

            @torch.compile
            def add(a, b):
                return a + b

            a = torch.zeros((XBLOCK,), device=GPU_TYPE)
            b = torch.zeros((XBLOCK,), device=GPU_TYPE)

            _result, kernels = run_and_get_kernels(add, a, b)
            assert len(kernels) == 1
            assert "output_block_ptr" in kernels[0]


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if RUN_GPU:
        run_tests(needs="filelock")
