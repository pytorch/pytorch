# mypy: allow-untyped-defs
import functools
import hashlib
import logging
import os
from collections.abc import Sequence
from typing import cast

from torch._inductor.utils import Placeholder
from torch.utils._ordered_set import OrderedSet

from ... import config
from ...codecache import code_hash, get_path
from ...ir import FlyDSLTemplateBuffer
from ...scheduler import BaseSchedulerNode, BaseScheduling, SchedulerNode
from ...select_algorithm import PartialRender
from ...utils import get_fused_kernel_name, get_kernel_metadata
from ...virtualized import V
from ..common import BackendFeature, IndentedBuffer


log = logging.getLogger(__name__)


@functools.lru_cache(None)
def _get_flydsl_device_arch(device_index: int) -> str | None:
    """Return the cached ROCm architecture reported for a device."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device_index)
            arch = getattr(props, "gcnArchName", None)
            if arch:
                return str(arch).split(":", 1)[0]
    except Exception:
        log.debug("Could not determine FlyDSL GPU arch", exc_info=True)
    return None


class FlyDSLScheduling(BaseScheduling):
    """Scheduling implementation for FlyDSL template kernels."""

    @classmethod
    def get_backend_features(cls, device) -> OrderedSet[BackendFeature]:
        return OrderedSet()

    @staticmethod
    def is_flydsl_template(node: BaseSchedulerNode) -> bool:
        return isinstance(node, SchedulerNode) and isinstance(
            node.node, FlyDSLTemplateBuffer
        )

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        return False

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        return False

    def define_kernel(
        self, src_code_str: str, node_schedule, precompile_metadata=None
    ) -> str:
        wrapper = V.graph.wrapper_code

        if src_code_str in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code_str]
        else:
            fused_name = (
                get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
                if config.triton.descriptive_names
                else ""
            )

            kernel_hash = hashlib.sha256(src_code_str.encode("utf-8")).hexdigest()[:8]
            if fused_name == "fused":
                kernel_name = f"flydsl_{kernel_hash}"
            else:
                kernel_name = f"flydsl_{fused_name}_{kernel_hash}"
            wrapper.src_to_kernel[src_code_str] = kernel_name
            src_code_str = src_code_str.replace(
                str(Placeholder.KERNEL_NAME), kernel_name
            )

            _, _, kernel_path = get_path(code_hash(src_code_str), "py")

            compile_wrapper = IndentedBuffer()
            compile_wrapper.writeline(f"async_compile.flydsl({kernel_name!r}, r'''")
            compile_wrapper.splice(src_code_str, strip=True)
            if precompile_metadata is not None:
                compile_wrapper.writeline(
                    f"''', precompile_metadata={precompile_metadata!r})"
                )
            else:
                compile_wrapper.writeline("''')")

            metadata_comment = f"# kernel path: {kernel_path}"
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment += "\n" + origins + "\n" + detailed_origins
            wrapper.define_kernel(
                kernel_name, compile_wrapper.getvalue(), metadata_comment
            )
        return kernel_name

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ):
        if not self.is_flydsl_template(template_node):
            raise AssertionError(
                "Template node passed to FlyDSLScheduling.codegen_template must be a "
                "SchedulerNode that wraps a FlyDSLTemplateBuffer"
            )
        if epilogue_nodes:
            raise AssertionError("FlyDSL doesn't support epilogue fusion yet")
        if prologue_nodes:
            raise AssertionError("FlyDSL doesn't support prologue fusion yet")

        template_node = cast(SchedulerNode, template_node)
        ftb: FlyDSLTemplateBuffer = cast(FlyDSLTemplateBuffer, template_node.node)

        kernel, render = ftb.make_kernel_render(ftb)  # type: ignore[misc]
        template_node.mark_run()
        src_code = render()
        if isinstance(src_code, PartialRender):
            src_code_str = src_code.finalize_all()
        else:
            src_code_str = src_code

        precompile_metadata = self._build_precompile_metadata(kernel, ftb)

        with V.set_kernel_handler(kernel):
            node_schedule = [template_node]
            kernel_name = self.define_kernel(
                src_code_str, node_schedule, precompile_metadata
            )
        self.codegen_comment(node_schedule, kernel_name)
        kernel.call_kernel(kernel_name, ftb)
        V.graph.removed_buffers |= kernel.removed_buffers
        self.free_buffers_in_scheduler()

    def _build_precompile_metadata(self, kernel, ftb):
        """Extract concrete tensor metadata for FlyDSL subprocess precompilation."""
        if not kernel._template_signature_defined:
            return None

        precompile_shapes = {}
        precompile_strides = {}
        precompile_dtypes = {}

        try:
            for arg_name, input_node in kernel._template_input_args:
                template_name = arg_name.removeprefix("arg_")
                size = input_node.get_size()
                precompile_shapes[template_name] = [int(s) for s in size]
                stride = input_node.get_stride()
                precompile_strides[template_name] = [int(s) for s in stride]
                precompile_dtypes[template_name] = str(
                    input_node.get_dtype()
                ).removeprefix("torch.")

            output_size = ftb.layout.size
            precompile_shapes["output"] = [int(s) for s in output_size]
            output_stride = ftb.layout.stride
            precompile_strides["output"] = [int(s) for s in output_stride]
            precompile_dtypes["output"] = str(ftb.layout.dtype).removeprefix("torch.")
        except (TypeError, RuntimeError, ValueError):
            log.debug(
                "Skipping FlyDSL precompile metadata: symbolic sizes cannot be "
                "resolved to concrete values"
            )
            return None

        device = ftb.layout.device
        device_index = device.index if device.index is not None else 0

        import torch

        device_capability = None
        if torch.cuda.is_available():
            device_capability = torch.cuda.get_device_capability(device_index)

        metadata: dict[str, object] = {
            "precompile_shapes": precompile_shapes,
            "precompile_strides": precompile_strides,
            "precompile_dtypes": precompile_dtypes,
            "device_index": device_index,
            "device_capability": device_capability,
        }

        flydsl_gpu_arch = self._build_flydsl_gpu_arch(device_index)
        if flydsl_gpu_arch is not None:
            metadata["flydsl_gpu_arch"] = flydsl_gpu_arch

        return metadata

    @staticmethod
    def _build_flydsl_gpu_arch(device_index) -> str | None:
        """Best-effort ROCm arch string for FlyDSL worker precompilation."""
        arch = os.environ.get("FLYDSL_GPU_ARCH")
        if arch:
            return arch.split(":", 1)[0]

        hsa_arch = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
        if hsa_arch:
            if hsa_arch.startswith("gfx"):
                return hsa_arch
            if hsa_arch.count(".") == 2:
                major, minor, stepping = hsa_arch.split(".")
                try:
                    return f"gfx{major}{minor}{int(stepping):x}"
                except ValueError:
                    log.debug("Ignoring invalid HSA_OVERRIDE_GFX_VERSION=%s", hsa_arch)

        return _get_flydsl_device_arch(device_index)
