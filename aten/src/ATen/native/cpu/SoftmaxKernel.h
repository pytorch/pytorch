#pragma once

#include <ATen/native/DispatchStub.h>
#include <cstdint>
#include <c10/util/Exception.h>

namespace at {
class Tensor;

namespace native {

using forward_fn = void (*)(const Tensor&, const Tensor&);
using backward_fn = void(*)(const Tensor &, const Tensor &, const Tensor&);

DECLARE_DISPATCH(forward_fn, softmax_lastdim_kernel);
DECLARE_DISPATCH(forward_fn, log_softmax_lastdim_kernel);
DECLARE_DISPATCH(backward_fn, softmax_backward_lastdim_kernel);
DECLARE_DISPATCH(backward_fn, log_softmax_backward_lastdim_kernel);

using forward_fn_with_dim = void(*)(const Tensor &, const Tensor &, const int64_t);
using backward_fn_with_dim =
    void (*)(const Tensor&, const Tensor&, const Tensor&, const int64_t);

DECLARE_DISPATCH(forward_fn_with_dim, softmax_kernel);
DECLARE_DISPATCH(forward_fn_with_dim, log_softmax_kernel);
DECLARE_DISPATCH(backward_fn_with_dim, softmax_backward_kernel);
DECLARE_DISPATCH(backward_fn_with_dim, log_softmax_backward_kernel);

/*
* [NOTE] SDPA + BOOLEAN_MASK
* This is a specialized path for Scaled Dot Product Attention (SDPA) with boolean masks.
* The key semantic requirement in SDPA is that fully masked-out rows in the attention
* matrix must result in all zeros, regardless of the input values.
*
* The process:
* 1. Compute softmax normally along the specified dimension.
* 2. Apply the mask by setting all masked positions to zero.
*
* This approach ensures:
* - Fully masked-out rows correctly result in all zeros, which is crucial for
*   the proper functioning of attention mechanisms in transformer architectures.
*
* Note: This method is specifically designed to meet the semantic requirements of
* SDPA with boolean masks. And is not intended for other uses cases. The semantics in fact
* are opposite for the boolean mask_ compared to how its typically called
*/

enum class MaskType {
    SRC_MASK = 0,            // Mask of shape (L, L) Corresponding to a source mask
    PADDING_MASK = 1,        // Mask of shape (B, L)
    DEFAULT_MASK = 2,        // Generic mask, shape should match input shape
    SDPA_BOOL_MASK = 3,       // Boolean mask for element-wise masking
    ERROR = 4                // Error state
};

inline MaskType toMaskType(int64_t value) {
    if (value >= 0 && value <= 3) {
        return static_cast<MaskType>(value);
    }
    TORCH_CHECK(false, "Invalid MaskType value");
    return MaskType::ERROR;
}

inline int64_t toIntMaskType(MaskType value) {
    return static_cast<int64_t>(value);
}


}
}
