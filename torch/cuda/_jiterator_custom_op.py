# mypy: allow-untyped-defs
"""Opaque native Jiterator launches for torch.compile."""

import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch._tensor_iterator import TensorIterator


_library = torch.library.Library("jiterator", "FRAGMENT")
_library.define(
    "_launch(str code, str name, bool return_by_ref, int num_outputs, "
    "Tensor[] tensors, bool[] conjugate_bits, bool[] negative_bits, "
    "str[] extra_names, Scalar[] extra_values, "
    "Tensor[] extra_tensors, int[] extra_tensor_positions) -> Tensor[]",
    tags=(torch.Tag.needs_exact_strides, torch.Tag.cudagraph_unsafe),
)


def _launch(
    code,
    name,
    return_by_ref,
    num_outputs,
    tensors,
    conjugate_bits,
    negative_bits,
    extra_names,
    extra_values,
    extra_tensors,
    extra_tensor_positions,
):
    # Compilation can materialize math views; native Jiterator reads raw storage.
    launch_tensors = []
    for tensor, conjugate, negative in zip(tensors, conjugate_bits, negative_bits):
        if tensor.is_conj() != conjugate:
            tensor = torch.conj_physical(tensor) if conjugate else tensor.resolve_conj()
        if tensor.dtype != torch.bool and tensor.is_neg() != negative:
            tensor = torch.neg(tensor) if negative else tensor.resolve_neg()
        launch_tensors.append(tensor)
    extra_values = list(extra_values)
    for position, tensor in zip(extra_tensor_positions, extra_tensors):
        extra_values.insert(position, tensor)
    result = torch._C._cuda_jiterator_compile_and_launch_kernel(
        code,
        name,
        return_by_ref,
        num_outputs,
        tuple(launch_tensors),
        dict(zip(extra_names, extra_values)),
    )
    outputs = [result] if num_outputs == 1 else list(result)
    if any(a.stride() != b.stride() for a, b in zip(tensors, launch_tensors)):
        for i, layout in enumerate(_output_metadata(tensors, num_outputs)):
            if outputs[i].stride() != layout.stride():
                outputs[i] = torch.empty_strided(
                    layout.size(),
                    layout.stride(),
                    dtype=layout.dtype,
                    device=tensors[0].device,
                ).copy_(outputs[i])
    return outputs


_library.impl("_launch", _launch, "CompositeExplicitAutograd")
_library.impl("_launch", torch.library.fallthrough_kernel, "Autograd")
# Native Jiterator reads storage directly, including inputs with lazy math bits.
_library.impl("_launch", torch.library.fallthrough_kernel, "Conjugate")
_library.impl("_launch", torch.library.fallthrough_kernel, "Negative")


@torch.library.register_fake("jiterator::_launch")
def _launch_fake(
    code,
    name,
    return_by_ref,
    num_outputs,
    tensors,
    conjugate_bits,
    negative_bits,
    extra_names,
    extra_values,
    extra_tensors,
    extra_tensor_positions,
):
    if not tensors or any(
        t.device.type != "cuda" or t.device != tensors[0].device for t in tensors
    ):
        raise RuntimeError("jiterator inputs must be on the same CUDA device")
    return [
        torch.empty_strided(
            o.size(), o.stride(), dtype=o.dtype, device=tensors[0].device
        )
        for o in _output_metadata(tensors, num_outputs)
    ]


def _output_metadata(tensors, num_outputs):
    # Native CUDA promotion preserves the original strides. Meta avoids CPU
    # promotion temporaries, which would change the iterator's output layout.
    with unset_fake_temporarily():
        operands = [
            torch.empty_strided(
                tuple(int(s) for s in t.size()),
                tuple(int(s) for s in t.stride()),
                dtype=t.dtype,
                device="meta",
            )
            for t in tensors
        ]
        iterator = TensorIterator(
            outputs=[None] * num_outputs,
            const_inputs=operands,
            promote_inputs_to_common_dtype=True,
            cast_common_dtype_to_outputs=True,
            enforce_safe_casting_to_output=True,
            check_all_same_dtype=False,
        )
        return [iterator.output(i) for i in range(num_outputs)]
