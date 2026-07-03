# mypy: allow-untyped-defs
"""Import adapter for `cutlass.operators` and legacy `cutlass_api`."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any


_CANONICAL_API = "cutlass.operators"
_LEGACY_API = "cutlass_api"

_proxy_cache: dict[str, ModuleType] = {}


def get_operator_api() -> ModuleType:
    """Return `cutlass.operators`, falling back to legacy `cutlass_api`."""
    try:
        return importlib.import_module(_CANONICAL_API)
    except ModuleNotFoundError as e:
        if e.name not in ("cutlass", _CANONICAL_API):
            raise
    return importlib.import_module(_LEGACY_API)


def _is_canonical_api(api: ModuleType | None = None) -> bool:
    api = api or get_operator_api()
    return api.__name__ == _CANONICAL_API


def is_available() -> bool:
    try:
        get_operator_api()
    except ImportError:
        return False
    return True


def _get_submodule(name: str) -> ModuleType:
    api = get_operator_api()
    module_name = f"{api.__name__}.{name}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name == module_name and hasattr(api, name):
            return getattr(api, name)
        raise


def _copy_module_proxy(cache_key: str, module: ModuleType) -> ModuleType:
    proxy = _proxy_cache.get(cache_key)
    if proxy is None:
        proxy = ModuleType(module.__name__)
        proxy.__dict__.update(module.__dict__)
        _proxy_cache[cache_key] = proxy
    return proxy


def _set_module_attr(module: ModuleType, name: str, value: Any) -> None:
    setattr(module, name, value)


def _get_module_attr(module: ModuleType, name: str) -> Any:
    return getattr(module, name)


def _target_sm_cc(target_sm: Any) -> int | None:
    if target_sm is None:
        return None
    cc = getattr(target_sm, "cc", None)
    if cc is not None:
        return int(cc)
    digits = "".join(ch for ch in str(target_sm) if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def _metadata_min_cc(metadata: Any) -> int:
    targets = getattr(metadata, "supported_targets", None)
    target_ccs = [_target_sm_cc(target) for target in targets or ()]
    target_ccs = [cc for cc in target_ccs if cc is not None]
    if target_ccs:
        return min(target_ccs)
    operator_class = getattr(metadata, "operator_class", None)
    return int(getattr(operator_class, "designed_for_min_cc", 0))


def get_arguments_module() -> ModuleType:
    arguments = _get_submodule("arguments")
    if not _is_canonical_api() or not hasattr(arguments, "ScaledOperand"):
        return arguments

    proxy = _copy_module_proxy("canonical_arguments", arguments)
    DenseTensor = arguments.DenseTensor
    ScaledOperand = arguments.ScaledOperand

    def dense_element_type(self):
        dtype = self.tensor.dtype
        try:
            return get_dtype_utils_module().cutlass_type_from_torch_type(dtype)
        except KeyError:
            return dtype

    def dense_shape(self):
        return self.tensor.shape

    def dense_stride(self):
        stride = self.tensor.stride
        return stride() if callable(stride) else stride

    def scaled_base(self):
        return self.quantized

    def scaled_attr(name):
        def getter(self):
            return getattr(self.quantized, name)

        return getter

    for cls, attr, getter in (
        (DenseTensor, "element_type", dense_element_type),
        (DenseTensor, "shape", dense_shape),
        (DenseTensor, "stride", dense_stride),
        (ScaledOperand, "base", scaled_base),
        (ScaledOperand, "tensor", scaled_attr("tensor")),
        (ScaledOperand, "element_type", scaled_attr("element_type")),
        (ScaledOperand, "shape", scaled_attr("shape")),
        (ScaledOperand, "stride", scaled_attr("stride")),
    ):
        if not hasattr(cls, attr):
            setattr(cls, attr, property(getter))

    if not hasattr(proxy, "ScaledTensor"):

        class ScaledTensor(ScaledOperand):
            pass

        ScaledTensor.__module__ = arguments.__name__
        _set_module_attr(proxy, "ScaledTensor", ScaledTensor)
    return proxy


def get_artifact_module() -> ModuleType:
    artifact = _get_submodule("artifact")
    if not _is_canonical_api():
        return artifact

    proxy = _copy_module_proxy("canonical_artifact", artifact)
    if getattr(proxy.CompiledArtifact, "_torchinductor_legacy_ctor", False):
        return proxy

    from cutlass.operators.arch import TargetSm

    class CompiledArtifact(artifact.CompiledArtifact):
        def __init__(self, compiled_obj, operator_obj, compiled_for=None):
            if isinstance(compiled_obj, artifact.CompiledArtifact):
                compiled_for = compiled_for or compiled_obj.compiled_for
                compiled_obj = compiled_obj.compiled_obj
            if compiled_for is None:
                metadata = getattr(operator_obj, "metadata", None)
                compiled_for = TargetSm.ensure(str(_metadata_min_cc(metadata)))
            super().__init__(compiled_obj, operator_obj, compiled_for)

    CompiledArtifact.__module__ = artifact.__name__
    CompiledArtifact._torchinductor_legacy_ctor = True
    _set_module_attr(proxy, "CompiledArtifact", CompiledArtifact)
    return proxy


def get_library_module() -> ModuleType:
    if not _is_canonical_api():
        return _get_submodule("library")

    arguments = get_arguments_module()
    proxy = _proxy_cache.get("canonical_library")
    if proxy is None:
        proxy = ModuleType(f"{_CANONICAL_API}.library")
        _set_module_attr(proxy, "ScaleMode", _get_module_attr(arguments, "ScaleMode"))
        _set_module_attr(
            proxy, "ScaleSwizzleMode", _get_module_attr(arguments, "ScaleSwizzleMode")
        )
        _proxy_cache["canonical_library"] = proxy
    return proxy


def get_metadata_module() -> ModuleType:
    """Return operator metadata with legacy NVGEMM aliases when needed."""
    metadata = _get_submodule("metadata")
    if not _is_canonical_api() or hasattr(metadata, "KernelMetadata"):
        return metadata

    proxy = _copy_module_proxy("canonical_metadata", metadata)
    if hasattr(proxy, "KernelMetadata"):
        return proxy

    from cutlass.operators.arch import TargetSm
    from cutlass.operators.mma import BlackwellTcgen05Mma

    class DenseTensorAttributes(metadata.DenseTensorConstraints):
        pass

    class ScaledTensorAttributes(metadata.ScaledOperandConstraints):
        def __init__(self, base, scale, mode, swizzle):
            super().__init__(
                quantized=base,
                scale=scale,
                mode=mode,
                swizzle=swizzle,
            )

        @property
        def base(self):
            return self.quantized

    class Sm100DesignMetadata(metadata.Sm100DesignMetadata):
        def __init__(
            self,
            tile_shape,
            cluster_shape,
            use_2cta_mma,
            use_tma_store,
            tile_scheduler=None,
            fallback_cluster_shape=None,
            *,
            mma_instruction_type=None,
            num_smem_stages=None,
        ):
            super().__init__(
                use_2cta_mma=use_2cta_mma,
                use_tma_store=use_tma_store,
                tile_scheduler=tile_scheduler,
                fallback_cluster_shape=fallback_cluster_shape,
                mma_instruction_type=mma_instruction_type or BlackwellTcgen05Mma,
                tile_shape=tile_shape,
                cluster_shape=cluster_shape,
                num_smem_stages=num_smem_stages,
            )

    class KernelMetadata(metadata.OperatorMetadata):
        def __init__(
            self,
            *,
            operands,
            design,
            kernel_name,
            kernel_class,
            min_cc,
            epilogue=None,
            supported_targets=None,
        ):
            if supported_targets is None:
                supported_targets = [TargetSm.ensure(str(min_cc))]
            super().__init__(
                operator_name=kernel_name,
                operator_class=kernel_class,
                supported_targets=supported_targets,
                operands=operands,
                design=design,
                epilogue=epilogue,
            )
            self.kernel_name = kernel_name
            self.kernel_class = kernel_class
            self.min_cc = min_cc

    for cls in (
        DenseTensorAttributes,
        ScaledTensorAttributes,
        Sm100DesignMetadata,
        KernelMetadata,
    ):
        cls.__module__ = metadata.__name__

    _set_module_attr(proxy, "DenseTensorAttributes", DenseTensorAttributes)
    _set_module_attr(proxy, "ScaledTensorAttributes", ScaledTensorAttributes)
    _set_module_attr(proxy, "Sm100DesignMetadata", Sm100DesignMetadata)
    _set_module_attr(proxy, "KernelMetadata", KernelMetadata)
    return proxy


def get_status_module() -> ModuleType:
    return _get_submodule("status")


def get_utils_module() -> ModuleType:
    utils = _get_submodule("utils")
    if not _is_canonical_api():
        return utils

    proxy = _copy_module_proxy("canonical_utils", utils)
    if not hasattr(proxy, "strides_to_layout_string"):
        _set_module_attr(
            proxy,
            "strides_to_layout_string",
            importlib.import_module(
                f"{utils.__name__}.tensor"
            ).strides_to_layout_string,
        )
    if not hasattr(proxy, "to_cuda_stream"):
        _set_module_attr(
            proxy,
            "to_cuda_stream",
            importlib.import_module(f"{utils.__name__}.device").to_cuda_stream,
        )
    if not hasattr(proxy, "tuple_to_string"):
        _set_module_attr(
            proxy,
            "tuple_to_string",
            importlib.import_module(f"{utils.__name__}.common").tuple_to_string,
        )
    if not hasattr(proxy, "cutlass_type_from_torch_type"):
        dtype = importlib.import_module(f"{utils.__name__}.dtype")
        _set_module_attr(
            proxy, "cutlass_type_from_torch_type", dtype.cutlass_type_from_torch_type
        )
        _set_module_attr(proxy, "dtype", dtype)
    return proxy


def get_dtype_utils_module() -> ModuleType:
    utils = get_utils_module()
    if hasattr(utils, "cutlass_type_from_torch_type"):
        return utils
    if hasattr(utils, "dtype"):
        return utils.dtype
    return _get_submodule("utils.dtype")


def _canonical_cutedsl_kernel_module() -> ModuleType:
    proxy = _proxy_cache.get("canonical_cutedsl_kernel")
    if proxy is not None:
        return proxy

    from cutlass.operators.providers.cutedsl.operator import CuteDslOperator

    arguments = get_arguments_module()

    class CuteDslKernel(CuteDslOperator):
        supported_args_type = _get_module_attr(arguments, "GemmArguments")
        designed_for_min_cc = 100

        def __init_subclass__(cls, **kwargs):
            pass

        def _compile(self, args, target_sm=None):
            cc = _target_sm_cc(target_sm)
            return self.compile(args, cc=cc)

        def _run(self, args, compiled_artifact, stream, workspace=None):
            raise NotImplementedError

        @classmethod
        def generate_operators(
            cls,
            metadata_filter=None,
            epilogue_args=None,
            target_sm=None,
            args=None,
        ):
            cc = _target_sm_cc(target_sm)
            return cls.generate_kernels(
                metadata_filter,
                epilogue_args=epilogue_args,
                cc=cc,
            )

        @classmethod
        def _generate_operators(
            cls,
            metadata_filter,
            epilogue_args=None,
            target_sm=None,
            args=None,
        ):
            return cls.generate_operators(
                metadata_filter,
                epilogue_args=epilogue_args,
                target_sm=target_sm,
                args=args,
            )

    CuteDslKernel.__module__ = f"{_CANONICAL_API}.providers.cutedsl.kernel"
    proxy = ModuleType(CuteDslKernel.__module__)
    _set_module_attr(proxy, "CuteDslKernel", CuteDslKernel)
    _proxy_cache["canonical_cutedsl_kernel"] = proxy
    return proxy


def get_provider_submodule(name: str) -> ModuleType:
    if not _is_canonical_api():
        return _get_submodule(f"providers.{name}")
    if name == "cutedsl.kernel":
        return _canonical_cutedsl_kernel_module()
    if name == "cutedsl.utils":
        return importlib.import_module(
            f"{_CANONICAL_API}.providers.cutedsl.integration_utils.mma"
        )
    return _get_submodule(f"providers.{name}")


def _canonical_provider_module_name(name: str) -> str:
    mapping = {
        "cutedsl.utils": f"{_CANONICAL_API}.providers.cutedsl.integration_utils.mma",
        "cutedsl.gemm.sm100_static_persistent": f"{_CANONICAL_API}.providers.cutedsl.gemm.sm100_persistent",
    }
    return mapping.get(name, f"{_CANONICAL_API}.providers.{name}")


def provider_module_names(names: list[str]) -> list[str]:
    api = get_operator_api()
    if api.__name__ == _CANONICAL_API:
        return [_canonical_provider_module_name(name) for name in names]
    return [f"{api.__name__}.providers.{name}" for name in names]


def _ensure_legacy_workspace_size(kernel: Any) -> None:
    if getattr(kernel, "_torchinductor_workspace_size_compat", False):
        return

    get_workspace_size = kernel.get_workspace_size

    def legacy_get_workspace_size(args):
        workspace_size = get_workspace_size(args)
        return getattr(workspace_size, "size_bytes", workspace_size)

    kernel.get_workspace_size = legacy_get_workspace_size
    kernel._torchinductor_workspace_size_compat = True


def _ensure_legacy_kernel_metadata(kernel: Any) -> Any:
    metadata = kernel.metadata
    if not all(
        hasattr(metadata, attr) for attr in ("kernel_name", "kernel_class", "min_cc")
    ):
        metadata_module = get_metadata_module()
        metadata = metadata_module.KernelMetadata(
            operands=metadata.operands,
            design=metadata.design,
            kernel_name=metadata.operator_name,
            kernel_class=metadata.operator_class,
            min_cc=_metadata_min_cc(metadata),
            epilogue=metadata.epilogue,
            supported_targets=metadata.supported_targets,
        )
        kernel._metadata = metadata

    _ensure_legacy_workspace_size(kernel)
    return kernel


def get_kernels() -> Any:
    api = get_operator_api()
    if hasattr(api, "get_kernels"):
        return api.get_kernels()
    return [_ensure_legacy_kernel_metadata(kernel) for kernel in api.get_operators()]


def ensure_fp4_dtype_registered() -> None:
    """Patch `cutlass.operators` or legacy `cutlass_api` for FP4."""
    import torch

    utils = get_dtype_utils_module()
    try:
        utils.cutlass_type_from_torch_type(torch.float4_e2m1fn_x2)
    except KeyError:
        import cutlass

        orig = utils.cutlass_type_from_torch_type

        def patched(dtype):
            if dtype == torch.float4_e2m1fn_x2:
                return cutlass.Float4E2M1FN
            return orig(dtype)

        _set_module_attr(utils, "cutlass_type_from_torch_type", patched)
