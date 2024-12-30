#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using weight_to_int4pack_fn = void (*)(const Tensor&, const Tensor&);
using int4pack_mm_fn =
    void (*)(const Tensor&, const Tensor&, const Tensor&, int, const Tensor&);
using int8pack_mm_fn =
    void (*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&);
using dyn_quant_pack_4bit_weight_fn = void (*)(
    Tensor&,
    const Tensor&,
    const Tensor&,
    const std::optional<Tensor>& bias,
    const int64_t,
    const int64_t,
    const int64_t);
using dyn_quant_matmul_4bit_fn = void (*)(
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const int64_t,
    const int64_t,
    const int64_t,
    const int64_t);

DECLARE_DISPATCH(weight_to_int4pack_fn, weight_to_int4pack_stub)
DECLARE_DISPATCH(int4pack_mm_fn, int4pack_mm_stub)
DECLARE_DISPATCH(int8pack_mm_fn, int8pack_mm_stub)
DECLARE_DISPATCH(
    dyn_quant_pack_4bit_weight_fn,
    dyn_quant_pack_4bit_weight_stub);
DECLARE_DISPATCH(dyn_quant_matmul_4bit_fn, dyn_quant_matmul_4bit_stub);

} // namespace at::native
