from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING, Union

import jax.numpy as jnp
import sympy

from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.codegen.pallas import PallasKernel
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.ir import (
    Buffer,
    ChoiceCaller,
    IRNode,
    Layout,
    OutputSpec,
    ShapeAsConstantBuffer,
    TemplateBuffer,
    TensorBox,
)
from torch._inductor.select_algorithm import KernelTemplate


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# --- Templates ---

MATMUL_KERNEL_TEMPLATE = """
def <KERNEL_NAME>_kernel(x_ref, y_ref, z_ref) -> None:
  @pl.when(pl.program_id(2) == 0)
  def _() -> None:
    z_ref[...] = jnp.zeros_like(z_ref)

  acc = z_ref[...].astype({{acc_dtype}})
  acc += jnp.dot(
      x_ref[...],
      y_ref[...],
      preferred_element_type={{acc_dtype}}
  )
  z_ref[...] = acc.astype(z_ref.dtype)
"""

PALLAS_CALL_TEMPLATE = """
@jax.jit
def <KERNEL_NAME>_pallas_call(x, y, *, bm={{bm}}, bk={{bk}}, bn={{bn}}):
  m, k = x.shape
  _, n = y.shape
  return pl.pallas_call(
      <KERNEL_NAME>_kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      in_specs=[pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))],
      out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
      grid=(m // bm, n // bn, k // bk),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary")),
  )(x, y)
"""

MAIN_FUNCTION_TEMPLATE = """
def <KERNEL_NAME>_main({{main_fn_args}}, stream=None):
    # Enable JAX x64 mode for float64/int64 support
    jax.config.update('jax_enable_x64', True)

    # Convert torch tensors to JAX arrays via dlpack
    x_tpu_jax = jax.dlpack.from_dlpack({{x}}.__dlpack__())
    y_tpu_jax = jax.dlpack.from_dlpack({{y}}.__dlpack__())

    # Call the jitted Pallas kernel
    z_tpu_jax = <KERNEL_NAME>_pallas_call(x_tpu_jax, y_tpu_jax)

    # Convert the result from JAX to torch via dlpack and copy to the output
    {{z}}.copy_(torch.from_dlpack(z_tpu_jax))
"""

# --- IR Node for Pallas Kernels ---


class PallasTemplateBuffer(TemplateBuffer):
    def __init__(
        self,
        layout: OutputSpec,
        inputs: Sequence[IRNode],
        make_kernel_render: Optional[Callable[..., Any]],
        template: PallasTemplate,
    ) -> None:
        super().__init__(layout, inputs, make_kernel_render)
        self.template = template


# --- Choice Caller ---


class PallasChoiceCaller(ChoiceCaller):
    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        description: str,
        make_kernel_render: Callable[
            ..., tuple[PallasTemplateKernel, Callable[[], str]]
        ],
        template: PallasTemplate,
    ) -> None:
        super().__init__(name, input_nodes, layout, description)
        self.make_kernel_render = make_kernel_render
        self.template = template

    def call_name(self) -> str:
        return f"pallas_template_kernels.{self.name}"

    def to_callable(self) -> Callable[..., Any]:
        def benchmark_stub(*args, **kwargs):
            return 0.0

        return benchmark_stub

    def hash_key(self) -> str:
        return self.name

    def output_node(self) -> Union[TensorBox, ShapeAsConstantBuffer]:
        return TensorBox.create(
            PallasTemplateBuffer(
                layout=self.layout,
                inputs=self.input_nodes,
                make_kernel_render=self.make_kernel_render,
                template=self.template,
            )
        )


# --- Pallas Kernel Class ---


class PallasTemplateKernel(PallasKernel):
    def __init__(
        self,
        template: PallasTemplate,
        choice_name: str,
        layout: Layout,
        input_nodes: list[IRNode],
        output_node: IRNode,
        **kwargs,
    ):
        self.template = template
        self.choice_name = choice_name
        self.kwargs = kwargs
        self.named_input_nodes = {}  # type: ignore[var-annotated]
        self.input_nodes = input_nodes

        numel = sympy.prod(layout.size)
        tiling = {"x": numel, "r0_": sympy.S.One}
        features = SIMDKernelFeatures([], numel)

        super().__init__(tiling=tiling, features=features)

        for node in input_nodes:
            self.args.input(node.get_name())
        self.args.output(output_node.get_name())

    def render(self) -> str:
        full_code = IndentedBuffer()
        full_code.writeline("import torch")
        full_code.writeline("import jax")
        full_code.writeline("import jax.numpy as jnp")
        full_code.writeline("from jax.experimental import pallas as pl")
        full_code.writeline("from jax.experimental.pallas import tpu as pltpu")
        full_code.writeline(self.template._render_kernel(self.input_nodes))
        full_code.writeline(self.template._render_pallas_call(self.kwargs))
        full_code.writeline(self.template._render_main_fn(self.args))

        return full_code.getvalue()


# --- Template Classes ---


class PallasTemplate(KernelTemplate, ABC):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @property
    def uid(self) -> str:
        return self.name

    @staticmethod
    def _template_from_string(source: str):
        import jinja2

        return jinja2.Template(source)

    def generate(self, **kwargs: Any) -> PallasChoiceCaller:
        input_nodes = kwargs.pop("input_nodes")
        layout = kwargs.pop("layout")
        choice_name = self.get_choice_name(**kwargs)
        description = self.get_description(**kwargs)
        make_kernel_render = self.get_make_kernel_render(choice_name, layout, **kwargs)

        return PallasChoiceCaller(
            name=choice_name,
            input_nodes=input_nodes,
            layout=layout,
            description=description,
            make_kernel_render=make_kernel_render,
            template=self,
        )

    @abstractmethod
    def get_choice_name(self, **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_description(self, **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_make_kernel_render(
        self, choice_name: str, layout: Layout, **kwargs
    ) -> Callable[..., tuple[PallasTemplateKernel, Callable[[], str]]]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _render_kernel(input_nodes) -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _render_pallas_call(kwargs: Any) -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _render_main_fn(args: Any) -> str:
        raise NotImplementedError


class PallasTpuBlockMatmulTemplate(PallasTemplate):
    def __init__(self, name: str):
        super().__init__(name)
        self._src_hash = hashlib.sha256(
            (
                MATMUL_KERNEL_TEMPLATE + PALLAS_CALL_TEMPLATE + MAIN_FUNCTION_TEMPLATE
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _render_kernel(input_nodes) -> str:
        assert len(input_nodes) == 2, "Matmul kernel expects 2 inputs"
        x_dtype = input_nodes[0].get_layout().dtype
        y_dtype = input_nodes[1].get_layout().dtype

        if x_dtype == jnp.int8 and y_dtype == jnp.int8:
            acc_dtype = "jnp.int32"
        else:
            acc_dtype = "jnp.float32"

        return PallasTemplate._template_from_string(MATMUL_KERNEL_TEMPLATE).render(
            acc_dtype=acc_dtype
        )

    @staticmethod
    def _render_pallas_call(kwargs: Any) -> str:
        bm = kwargs.get("bm", 128)
        bk = kwargs.get("bk", 128)
        bn = kwargs.get("bn", 128)
        return PallasTemplate._template_from_string(PALLAS_CALL_TEMPLATE).render(
            bm=bm, bk=bk, bn=bn
        )

    @staticmethod
    def _render_main_fn(args: Any) -> str:
        arg_defs, _, _, _ = args.python_argdefs()
        main_fn_args = [a.name for a in arg_defs]
        return PallasTemplate._template_from_string(MAIN_FUNCTION_TEMPLATE).render(
            main_fn_args=", ".join(main_fn_args),
            x=main_fn_args[0],
            y=main_fn_args[1],
            z=main_fn_args[2],
        )

    @property
    def src_hash(self) -> str:
        return self._src_hash

    def get_choice_name(self, **kwargs) -> str:
        bm = kwargs.get("bm", 128)
        bk = kwargs.get("bk", 128)
        bn = kwargs.get("bn", 128)

        config_params = f"{bm}_{bk}_{bn}"
        hasher = hashlib.sha256()
        hasher.update(self.src_hash.encode("utf-8"))
        hasher.update(config_params.encode("utf-8"))
        choice_hash = hasher.hexdigest()[:8]
        return f"{self.name}_{choice_hash}"

    def get_description(self, **kwargs) -> str:
        bm = kwargs.get("bm", 128)
        bk = kwargs.get("bk", 128)
        bn = kwargs.get("bn", 128)

        return f"bm={bm}, bk={bk}, bn={bn}"

    def get_make_kernel_render(
        self, choice_name: str, layout: Layout, **kwargs
    ) -> Callable[..., tuple[PallasTemplateKernel, Callable[[], str]]]:
        def make_kernel_render_fn(template_node, *args, **kwargs_render):
            output_node = template_node
            input_nodes = template_node.inputs
            kernel = PallasTemplateKernel(
                self, choice_name, layout, input_nodes, output_node, **kwargs
            )
            return kernel, kernel.render

        return make_kernel_render_fn


pallas_tpu_block_mm_template = PallasTpuBlockMatmulTemplate("pallas_tpu_block_mm")
