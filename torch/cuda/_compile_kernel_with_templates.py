"""
Enhanced kernel compilation with C++ template support for CUTLASS and other templated CUDA libraries.
"""

from typing import Any, Optional

from torch.cuda._template_utils import wrap_template_kernel
from torch.cuda._utils import _cuda_load_module, _nvrtc_compile


def _compile_kernel_with_templates(
    kernel_source: str,
    kernel_name: str,
    compute_capability: Optional[str] = None,
    header_code: str = "",
    cuda_include_dirs: Optional[list] = None,
    nvcc_options: Optional[list] = None,
    # New template-specific parameters
    is_template: bool = False,
    template_types: Optional[list[str]] = None,
    wrapper_signature: Optional[str] = None,
    wrapper_body: Optional[str] = None,
    wrapper_name: Optional[str] = None,
):
    """
    Enhanced version of _compile_kernel that supports C++ templates.

    This function extends the basic _compile_kernel to handle C++ templates,
    which is essential for using CUTLASS device API and other templated CUDA libraries.

    Args:
        kernel_source (str): The CUDA kernel source code (can be templated)
        kernel_name (str): The name of the kernel function
        compute_capability (str, optional): Target compute capability (e.g., "86")
        header_code (str, optional): Additional header code
        cuda_include_dirs (list, optional): List of CUDA include directories
        nvcc_options (list, optional): Additional NVRTC options
        is_template (bool): Whether the kernel is a C++ template
        template_types (list[str], optional): Types to instantiate the template with
        wrapper_signature (str, optional): Parameter signature for the wrapper
        wrapper_body (str, optional): Body of the wrapper function
        wrapper_name (str, optional): Name of the extern "C" wrapper

    Returns:
        callable: A Python function that can execute the kernel

    Example for regular kernel:
        >>> kernel_code = '''
        extern "C" __global__ void add(float* a, float* b, float* c, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) c[i] = a[i] + b[i];
        }
        '''
        >>> add_kernel = _compile_kernel_with_templates(kernel_code, "add")

    Example for templated kernel:
        >>> template_code = '''
        template<typename T>
        __global__ void add_template(T* a, T* b, T* c, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) c[i] = a[i] + b[i];
        }
        '''
        >>> add_kernel = _compile_kernel_with_templates(
        ...     template_code,
        ...     "add_template",
        ...     is_template=True,
        ...     template_types=["float"],
        ...     wrapper_signature="float* a, float* b, float* c, int n",
        ...     wrapper_body="    add_template<float>(a, b, c, n);",
        ... )
    """

    # Handle template instantiation
    if is_template:
        if not template_types:
            raise ValueError("template_types must be provided for template kernels")
        if not wrapper_signature:
            raise ValueError("wrapper_signature must be provided for template kernels")
        if not wrapper_body:
            raise ValueError("wrapper_body must be provided for template kernels")

        # Wrap the template with explicit instantiation and extern "C" wrapper
        wrapped_code, actual_kernel_name = wrap_template_kernel(
            kernel_source,
            kernel_name,
            template_types,
            wrapper_signature,
            wrapper_body,
            wrapper_name,
        )

        # Use the wrapped code for compilation
        kernel_source = wrapped_code
        kernel_name = actual_kernel_name

    # Compile using the standard path
    ptx = _nvrtc_compile(
        kernel_source,
        kernel_name,
        compute_capability,
        header_code,
        cuda_include_dirs,
        nvcc_options,
    )

    # Load the module and get the kernel
    result = _cuda_load_module(ptx, [kernel_name])
    if isinstance(result, dict):
        return result[kernel_name]
    else:
        return getattr(result, kernel_name)


def compile_cutlass_gemm(
    m: int,
    n: int,
    k: int,
    element_type: str = "float",
    compute_capability: Optional[str] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> Any:
    """
    Compile a CUTLASS GEMM kernel with specific parameters.

    This is a convenience function specifically for CUTLASS GEMM operations.

    Args:
        m, n, k: Matrix dimensions
        element_type: Data type ("float", "double", "half")
        compute_capability: Target GPU architecture
        alpha, beta: GEMM parameters (C = alpha * A * B + beta * C)

    Returns:
        callable: Compiled CUTLASS GEMM kernel
    """

    cutlass_template = """
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/core_io.h>
#include <cutlass/gemm/device/gemm.h>

template<typename ElementType>
__global__ void cutlass_gemm_kernel(
    ElementType const* A,
    ElementType const* B,
    ElementType* C,
    int M, int N, int K,
    ElementType alpha,
    ElementType beta
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        ElementType sum = 0;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}
"""

    type_map = {
        "float": "float",
        "double": "double",
        "half": "__half",
    }

    cpp_type = type_map.get(element_type, "float")
    wrapper_sig = f"""
{cpp_type} const* A,
{cpp_type} const* B,
{cpp_type}* C,
int M, int N, int K,
{cpp_type} alpha,
{cpp_type} beta
""".strip()

    wrapper_body = (
        f"    cutlass_gemm_kernel<{cpp_type}>(A, B, C, M, N, K, alpha, beta);"
    )
    return _compile_kernel_with_templates(
        cutlass_template,
        "cutlass_gemm_kernel",
        compute_capability=compute_capability,
        is_template=True,
        template_types=[cpp_type],
        wrapper_signature=wrapper_sig,
        wrapper_body=wrapper_body,
        wrapper_name="cutlass_gemm_wrapper",
    )


# Export the enhanced compile function
__all__ = ["_compile_kernel_with_templates", "compile_cutlass_gemm"]
