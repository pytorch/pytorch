"""
Utilities for handling C++ templates in CUDA kernel compilation.
"""

from typing import Any, Optional


def generate_template_instantiation(
    template_code: str,
    template_name: str,
    template_params: dict[str, Any],
    wrapper_name: Optional[str] = None,
) -> tuple[str, str]:
    """
    Generate C++ template instantiation and C-linkage wrapper for NVRTC compilation.

    Args:
        template_code: The C++ template code
        template_name: Name of the template function/class
        template_params: Dictionary of template parameters and their values
        wrapper_name: Name for the extern "C" wrapper function

    Returns:
        tuple: (instantiation_code, wrapper_function_name)
    """
    if wrapper_name is None:
        wrapper_name = f"{template_name}_wrapper"

    template_args = ", ".join(str(v) for v in template_params.values())
    instantiation_code = f"""
{template_code}

// Explicit instantiation
template __global__ void {template_name}<{template_args}>();

extern "C" __global__ void {wrapper_name}() {{
    {template_name}<{template_args}>();
}}
"""

    return instantiation_code, wrapper_name


def prepare_cutlass_kernel(
    cutlass_template: str,
    element_type: str = "float",
    layout: str = "cutlass::layout::RowMajor",
    opcode_class: str = "cutlass::arch::OpClassTensorOp",
    arch: str = "cutlass::arch::Sm80",
    wrapper_name: Optional[str] = None,
) -> tuple[str, str]:
    """
    Prepare a CUTLASS template kernel for NVRTC compilation.

    Args:
        cutlass_template: The CUTLASS template code
        element_type: Data type for the computation
        layout: Matrix layout
        opcode_class: CUTLASS operation class
        arch: Target architecture
        wrapper_name: Name for the wrapper function

    Returns:
        tuple: (prepared_code, wrapper_name)
    """
    if wrapper_name is None:
        wrapper_name = "cutlass_kernel_wrapper"

    prepared_code = f"""
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/arch/arch.h>

{cutlass_template}

extern "C" __global__ void {wrapper_name}(
    {element_type} const* A,
    {element_type} const* B,
    {element_type}* C,
    int M, int N, int K,
    {element_type} alpha,
    {element_type} beta
) {{
    using ElementA = {element_type};
    using ElementB = {element_type};
    using ElementC = {element_type};
    using ElementAccumulator = {element_type};

    using LayoutA = {layout};
    using LayoutB = {layout};
    using LayoutC = {layout};
}}
"""

    return prepared_code, wrapper_name


def wrap_template_kernel(
    template_code: str,
    template_name: str,
    template_types: list[str],
    function_signature: str,
    function_body: str,
    wrapper_name: Optional[str] = None,
) -> tuple[str, str]:
    """
    Wrap a templated kernel function with explicit instantiation and extern "C" wrapper.

    Args:
        template_code: The template definition code
        template_name: Name of the template function
        template_types: List of types to instantiate the template with
        function_signature: Signature of the wrapper function (parameters)
        function_body: Body of the wrapper function that calls the template
        wrapper_name: Name for the extern "C" wrapper

    Returns:
        tuple: (complete_code, wrapper_name)
    """
    if wrapper_name is None:
        wrapper_name = f"{template_name}_wrapper"

    template_spec = ", ".join(template_types)

    complete_code = f"""
{template_code}

template __global__ void {template_name}<{template_spec}>({function_signature});

extern "C" __global__ void {wrapper_name}({function_signature}) {{
{function_body}
}}
"""

    return complete_code, wrapper_name
