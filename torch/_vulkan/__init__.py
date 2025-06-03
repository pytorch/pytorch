# mypy: allow-untyped-defs
"""
This package enables an interface for accessing the Vulkan backend in Python.
"""
from typing import Union

import torch
from torch import Tensor


def _device_count() -> int:
    r"""Returns the number of available Vulkan devices."""
    # TODO: actually get the number!
    return int(torch.is_vulkan_available())


def _compile_shader(name: str, source: str):
    r"""Compiles compute shader from source and allows one to invoke kernels
    defined there from the comfort of Python runtime.
    We rely on the shader to follow a number of important conventions:
    - we determine the number of arguments by looking for lines in the source starting with "layout(set"
    - specialization constants must be used to set workgroup size, exactly like so:
      layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
    - Binding 0 is a uniform restrict writeonly image3D where the output is to be written.
    - Bindings 1-N (where N is the arity of the op) are uniform sampler3D objects.
    - Binding N+1 is a uniform Block containing a single ivec3 indicating the operation
      size. (TODO: This should probably be a push constant (or specialization constant
      as done in the ExecuTorch backend???), but currently we are just matching existing
      shader convention.)

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_VULKAN)
        >>> myabs = torch._vulkan.compile_shader("myabs", '''
        ... layout(std430) buffer;
        ...
        ... /* Qualifiers: layout - storage - precision - memory */
        ...
        ... layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D   uOutput;
        ... layout(set = 0, binding = 1)         uniform PRECISION                    sampler3D uInput;
        ... layout(set = 0, binding = 2)         uniform PRECISION restrict           Block {
        ...   ivec3 size;
        ... } uBlock;
        ...
        ... layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
        ...
        ... void main() {
        ...   const ivec3 pos = ivec3(gl_GlobalInvocationID);
        ...
        ...   if (all(lessThan(pos, uBlock.size.xyz))) {
        ...     const vec4 intex = texelFetch(uInput, pos, 0);
        ...     imageStore(
        ...         uOutput,
        ...         pos,
        ...         abs(intex));
        ...   }
        ... }
        >>> x = torch.tensor([1., 2., -3.], device="vulkan")
        >>> x_abs = torch.empty_like(x)
        >>> myabs(x_abs, x)
        >>> print(x_abs.cpu())
        tensor([1., 2., 3.])
    """
    from pathlib import Path

    from torch.utils._cpp_embed_headers import _embed_headers

    if not hasattr(torch._C, "_vulkan_compileShader"):
        raise RuntimeError("Vulkan is not available")
    source = _embed_headers(
        [l + "\n" for l in source.split("\n")],
        [Path(__file__).parent.parent / "include"],
        set(),
    )
    return torch._C._vulkan_compileShader(name, source)


def _is_available() -> bool:
    return _device_count() > 0
