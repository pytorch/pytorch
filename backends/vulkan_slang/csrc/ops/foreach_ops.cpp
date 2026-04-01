// Foreach (fused) optimizer ops: loop over tensor lists calling single-tensor ops.
// These enable PyTorch's fused Adam/SGD optimizers to work with the Vulkan backend.
// A true fused implementation would dispatch a single kernel; these are unfused
// but correct. Performance optimization is future work.

#include "ops.h"
#include "dispatch.h"

#include <torch/library.h>

namespace torch_vulkan { namespace ops {

// Helper: get mutable reference from TensorList (safe for in-place foreach ops)
static at::Tensor& mutable_ref(at::TensorList list, size_t i) {
    return const_cast<at::Tensor&>(list[i]);
}

// ── _foreach_add_ (scalar) ──────────────────────────────────────
void vulkan_foreach_add_scalar_(at::TensorList self, const at::Scalar& scalar) {
    for (size_t i = 0; i < self.size(); i++) {
        vulkan_add_scalar_(mutable_ref(self, i), scalar, at::Scalar(1));
    }
}

// ── _foreach_add_ (list) ────────────────────────────────────────
void vulkan_foreach_add_list_(at::TensorList self, at::TensorList other,
                                const at::Scalar& alpha) {
    TORCH_CHECK(self.size() == other.size(),
                "_foreach_add_: self and other must have same length");
    for (size_t i = 0; i < self.size(); i++) {
        vulkan_add_(mutable_ref(self, i), other[i], alpha);
    }
}

// ── _foreach_mul_ (scalar) ──────────────────────────────────────
void vulkan_foreach_mul_scalar_(at::TensorList self, const at::Scalar& scalar) {
    for (size_t i = 0; i < self.size(); i++) {
        vulkan_mul_scalar_(mutable_ref(self, i), scalar);
    }
}

// ── _foreach_addcmul_ ──────────────────────────────────────────
void vulkan_foreach_addcmul_(at::TensorList self, at::TensorList tensor1,
                               at::TensorList tensor2, const at::Scalar& value) {
    TORCH_CHECK(self.size() == tensor1.size() && self.size() == tensor2.size(),
                "_foreach_addcmul_: all lists must have same length");
    for (size_t i = 0; i < self.size(); i++) {
        vulkan_addcmul_(mutable_ref(self, i), tensor1[i], tensor2[i], value);
    }
}

// ── _foreach_addcdiv_ ──────────────────────────────────────────
void vulkan_foreach_addcdiv_(at::TensorList self, at::TensorList tensor1,
                               at::TensorList tensor2, const at::Scalar& value) {
    TORCH_CHECK(self.size() == tensor1.size() && self.size() == tensor2.size(),
                "_foreach_addcdiv_: all lists must have same length");
    for (size_t i = 0; i < self.size(); i++) {
        vulkan_addcdiv_(mutable_ref(self, i), tensor1[i], tensor2[i], value);
    }
}

// ── _foreach_sqrt ───────────────────────────────────────────────
std::vector<at::Tensor> vulkan_foreach_sqrt(at::TensorList self) {
    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (const auto& t : self) {
        result.push_back(vulkan_sqrt(t));
    }
    return result;
}

// ── _foreach_neg ────────────────────────────────────────────────
std::vector<at::Tensor> vulkan_foreach_neg(at::TensorList self) {
    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (const auto& t : self) {
        result.push_back(vulkan_neg(t));
    }
    return result;
}

// ── _foreach_div_ (scalar) ──────────────────────────────────────
void vulkan_foreach_div_scalar_(at::TensorList self, const at::Scalar& scalar) {
    for (size_t i = 0; i < self.size(); i++) {
        vulkan_div_scalar_(mutable_ref(self, i), scalar);
    }
}

// ── _foreach_lerp_ ─────────────────────────────────────────────
void vulkan_foreach_lerp_(at::TensorList self, at::TensorList end,
                            const at::Scalar& weight) {
    TORCH_CHECK(self.size() == end.size(),
                "_foreach_lerp_: self and end must have same length");
    for (size_t i = 0; i < self.size(); i++) {
        vulkan_lerp_(mutable_ref(self, i), end[i], weight);
    }
}

// ── _foreach_maximum ────────────────────────────────────────────
std::vector<at::Tensor> vulkan_foreach_maximum(at::TensorList self, at::TensorList other) {
    TORCH_CHECK(self.size() == other.size(),
                "_foreach_maximum: lists must have same length");
    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); i++) {
        // max(a, b) = where(a >= b, a, b)
        auto mask = vulkan_ge(self[i], other[i]);
        result.push_back(vulkan_where(mask, self[i], other[i]));
    }
    return result;
}

}} // namespace torch_vulkan::ops
