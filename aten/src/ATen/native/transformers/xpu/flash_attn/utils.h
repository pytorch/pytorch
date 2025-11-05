#pragma once

#include <ATen/ATen.h>

namespace sycltla {

enum class ATTN_TENSOR_LAYOUT {
  BHSD, // batchsize, headnum, seqlen, headdim
  BSHD, // batchsize, seqlen, headnum, headdim
  BXD, // in case headnum==1 or seqlen==1, which is compatible with BHSD/BSHD
  UNSUPPORTED
};

inline std::string to_string(ATTN_TENSOR_LAYOUT layout) {
  switch (layout) {
    case ATTN_TENSOR_LAYOUT::BHSD:
      return "BHSD";
    case ATTN_TENSOR_LAYOUT::BSHD:
      return "BSHD";
    case ATTN_TENSOR_LAYOUT::BXD:
      return "BXD";
    case ATTN_TENSOR_LAYOUT::UNSUPPORTED:
      return "UNSUPPORTED";
    default:
      return "UNKNOWN";
  }
}

inline ATTN_TENSOR_LAYOUT get_attn_tensor_layout(const at::Tensor& t) {
  // sdpa's tensor shape are in BHSD format
  if (t.is_contiguous(at::MemoryFormat::Contiguous)) {
    if (t.size(1) == 1 || t.size(2) == 1) {
      return ATTN_TENSOR_LAYOUT::BXD;
    }
    return ATTN_TENSOR_LAYOUT::BHSD;
  } else if (t.transpose(1, 2).is_contiguous(at::MemoryFormat::Contiguous)) {
    if (t.size(1) == 1 || t.size(2) == 1) {
      return ATTN_TENSOR_LAYOUT::BXD;
    }
    return ATTN_TENSOR_LAYOUT::BSHD;
  } else {
    return ATTN_TENSOR_LAYOUT::UNSUPPORTED;
  }
}

inline ATTN_TENSOR_LAYOUT fuse_attn_tensor_layout(
    ATTN_TENSOR_LAYOUT layout1,
    ATTN_TENSOR_LAYOUT layout2) {
  if (layout1 == ATTN_TENSOR_LAYOUT::UNSUPPORTED ||
      layout2 == ATTN_TENSOR_LAYOUT::UNSUPPORTED) {
    return ATTN_TENSOR_LAYOUT::UNSUPPORTED;
  }
  if (layout1 == layout2) {
    return layout1;
  }
  // if one is BXD, return the other one
  if (layout1 == ATTN_TENSOR_LAYOUT::BXD) {
    return layout2;
  }
  if (layout2 == ATTN_TENSOR_LAYOUT::BXD) {
    return layout1;
  }
  // otherwise, incompatible
  return ATTN_TENSOR_LAYOUT::UNSUPPORTED;
}

inline at::Tensor attn_tensor_to_layout(
    const at::Tensor& t,
    ATTN_TENSOR_LAYOUT target_layout) {
  if (target_layout == ATTN_TENSOR_LAYOUT::UNSUPPORTED ||
      target_layout == ATTN_TENSOR_LAYOUT::BXD) {
    TORCH_CHECK(
        false, "FlashAttentionXPU: only support BHSD or BSHD as target layout");
  }

  ATTN_TENSOR_LAYOUT layout = get_attn_tensor_layout(t);
  at::Tensor output = t;
  if (layout == ATTN_TENSOR_LAYOUT::UNSUPPORTED ||
      layout == ATTN_TENSOR_LAYOUT::BXD || 
      layout != target_layout) {
    if (target_layout == ATTN_TENSOR_LAYOUT::BHSD) {
      // convert to BHSD
      output = t.contiguous(at::MemoryFormat::Contiguous);
      layout = ATTN_TENSOR_LAYOUT::BHSD;
    } else {
      // convert to BSHD
      output = t.permute({0, 2, 1, 3})
                   .contiguous(at::MemoryFormat::Contiguous)
                   .permute({0, 2, 1, 3});
      layout = ATTN_TENSOR_LAYOUT::BSHD;
    }
  }

  return output;
}

} // namespace sycltla
