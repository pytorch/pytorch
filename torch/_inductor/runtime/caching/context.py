"""Context management for PyTorch Inductor runtime caching.

This module provides context classes for collecting configuration and environment
information used in caching decisions for PyTorch's Inductor runtime.
"""

import json
from abc import ABC, abstractmethod
from base64 import b64encode
from functools import cache
from hashlib import sha256
from typing import Any, Optional, Sequence
from typing_extensions import override, TypedDict

import torch


class _Context(ABC):
    """Abstract base class for context providers.

    Context providers collect specific configuration and environment information
    that affects compilation and runtime behavior.
    """

    @staticmethod
    @abstractmethod
    def forms_of_context() -> Sequence[str]:
        """Return a sequence of context form names provided by this context class.

        Returns:
            A sequence of strings representing the available context forms.
        """


class _RuntimeContext(_Context):
    """Context provider for runtime configuration and environment settings.

    Collects configuration settings that affect runtime behavior but not
    compilation, such as Inductor configs, determinism settings, and CUDA
    matmul precision configurations.
    """

    @override
    @staticmethod
    def forms_of_context() -> Sequence[str]:
        """Return the runtime context forms provided by this class.

        Returns:
            A sequence containing the available runtime context forms:
            - "inductor_configs": PyTorch Inductor configuration settings
            - "torch_determinism_configs": Deterministic algorithm settings
            - "cuda_matmul_precision_configs": CUDA matrix multiplication precision settings
        """
        return (
            "inductor_configs",
            "torch_determinism_configs",
            "cuda_matmul_precision_configs",
        )

    @staticmethod
    def inductor_configs() -> dict[str, Any]:
        """Get portable Inductor configuration settings.

        Returns:
            A dictionary containing Inductor configuration settings,
            including private configs.
        """
        from torch._inductor import config

        return config.save_config_portable(ignore_private_configs=False)

    @staticmethod
    def torch_determinism_configs() -> dict[str, Any]:
        """Get PyTorch deterministic algorithm configuration settings.

        Returns:
            A dictionary containing deterministic algorithm settings:
            - Whether deterministic algorithms are enabled
            - Whether deterministic algorithm warnings are enabled
            - Fill uninitialized memory setting
        """
        return {
            "torch.are_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "torch.is_deterministic_algorithms_warn_only_enabled": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "torch.utils.deterministic.fill_uninitialized_memory": (
                torch.utils.deterministic.fill_uninitialized_memory  # type: ignore[attr-defined]
            ),
        }

    @staticmethod
    def cuda_matmul_precision_configs() -> dict[str, Any]:
        """Get CUDA matrix multiplication precision configuration settings.

        Returns:
            A dictionary containing CUDA matmul precision settings:
            - FP32 precision setting
            - FP16 reduced precision reduction allowance
            - BF16 reduced precision reduction allowance
        """
        return {
            "torch.backends.cuda.matmul.fp32_precision": torch.backends.cuda.matmul.fp32_precision,
            "torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            ),
            "torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
        }


class _CompileContext(_Context):
    """Context provider for compilation-related configuration and environment settings.

    Collects information that affects compilation behavior, such as PyTorch and Triton
    versions, runtime environment, and accelerator properties.
    """

    @override
    @staticmethod
    def forms_of_context() -> Sequence[str]:
        """Return the compile context forms provided by this class.

        Returns:
            A sequence containing the available compile context forms:
            - "torch_version_hash": PyTorch version hash
            - "triton_version_hash": Triton version hash (if available)
            - "runtime": Runtime type (CUDA/HIP/None)
            - "runtime_version": Runtime version string
            - "accelerator_properties": GPU/accelerator properties
        """
        return (
            "torch_version_hash",
            "triton_version_hash",
            "runtime",
            "runtime_version",
            "accelerator_properties",
        )

    @cache
    @staticmethod
    def torch_version_hash() -> str:
        """Get base64-encoded PyTorch version hash.

        Returns:
            A base64-encoded string representing the PyTorch version hash.
        """
        from torch._inductor.codecache import torch_key

        return b64encode(torch_key()).decode()

    @cache
    @staticmethod
    def triton_version_hash() -> Optional[str]:
        """Get Triton version key if Triton is available.

        Returns:
            Triton version key if Triton is available, None otherwise.
        """
        from torch._inductor.runtime.triton_compat import HAS_TRITON, triton_key

        return triton_key() if HAS_TRITON else None

    @cache
    @staticmethod
    def runtime() -> Optional[str]:
        """Determine the runtime type based on available backends.

        Returns:
            "CUDA" if CUDA is available, "HIP" if HIP is available, None otherwise.
        """
        return "CUDA" if torch.version.cuda else "HIP" if torch.version.hip else None

    @cache
    @staticmethod
    def runtime_version() -> Optional[str]:
        """Get the version string for the detected runtime.

        Returns:
            Version string for the current runtime (CUDA or HIP), or None if
            no supported runtime is detected.
        """
        return {
            "CUDA": torch.version.cuda,
            "HIP": torch.version.hip,
        }.get(_CompileContext.runtime())  # type: ignore[arg-type]

    @cache
    @staticmethod
    def accelerator_properties() -> Optional[str]:
        """Get string representation of CUDA device properties.

        Returns:
            String representation of CUDA device properties if a runtime is
            available, None otherwise.
        """
        return (
            repr(torch.cuda.get_device_properties())
            if _CompileContext.runtime()
            else None
        )


class SelectedRuntimeContext(TypedDict):
    inductor_configs: bool
    torch_determinism_configs: bool
    cuda_matmul_precision_configs: bool


class SelectedCompileContext(TypedDict):
    torch_version_hash: bool
    triton_version_hash: bool
    runtime: bool
    runtime_version: bool
    accelerator_properties: bool


class IsolationSchema(TypedDict):
    """Schema for specifying which context forms to include in cache isolation.

    Attributes:
        runtime_context: Either True (include all runtime context), False (exclude all),
                        or a SelectedRuntimeContext dict specifying which forms to include.
        compile_context: Either True (include all compile context), False (exclude all),
                        or a SelectedCompileContext dict specifying which forms to include.
    """

    runtime_context: SelectedRuntimeContext | bool
    compile_context: SelectedCompileContext | bool


_DEFAULT_ISOLATION_SCHEMA: IsolationSchema = IsolationSchema(
    runtime_context=True, compile_context=True
)


def _isolation_context(
    ischema: IsolationSchema = _DEFAULT_ISOLATION_SCHEMA,
) -> dict[str, Any]:
    """Generate context data based on the isolation schema.

    Args:
        ischema: Schema specifying which context forms to include.
                Defaults to including all runtime and compile context.

    Returns:
        A dictionary containing the selected context data with keys
        "runtime_context" and "compile_context", where each value is
        either None (if excluded) or a dict of context form data.
    """
    isolation_context: dict[str, Any] = {}
    for context_name, context_cls in (
        ("runtime_context", _RuntimeContext),
        ("compile_context", _CompileContext),
    ):
        selected_context: Optional[dict[str, Any]] = None
        if ischema[context_name] is True:  # type: ignore[literal-required]
            selected_context = {
                form_of_context: getattr(context_cls, form_of_context)()
                for form_of_context in context_cls.forms_of_context()
            }
        elif ischema[context_name] is False:  # type: ignore[literal-required]
            selected_context = None
        else:
            selected_context = {}
            for form_of_context in ischema[context_name]:  # type: ignore[literal-required]
                selected = ischema[context_name][form_of_context]  # type: ignore[literal-required]
                if selected:
                    selected_context[form_of_context] = getattr(
                        context_cls, form_of_context
                    )()
            selected_context = selected_context or None
        isolation_context[context_name] = selected_context
    return isolation_context


def _isolation_key(ischema: IsolationSchema = _DEFAULT_ISOLATION_SCHEMA) -> str:
    """Generate a unique key for the given isolation schema.

    Args:
        ischema: Schema specifying which context forms to include.
                Defaults to including all runtime and compile context.

    Returns:
        A 32-character hexadecimal string that uniquely identifies
        the context specified by the isolation schema.
    """
    return sha256(
        json.dumps(_isolation_context(ischema), sort_keys=True).encode()
    ).hexdigest()[:32]
