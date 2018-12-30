#pragma once

#include <ATen/cudnn/cudnn-wrapper.h>
#include <ATen/ATen.h>

namespace at { namespace native {

enum class Layout {
  // N = batch dimension, C = channels, F = feature dimension(s)
  NCF,
  NFC
};

struct LayoutPermutation {
  static constexpr size_t max_dims = 5;

  LayoutPermutation(Layout from, Layout to, int64_t ndim)
    : perm_(make_perm(from, to, ndim)), ndim_(ndim) {}

  static std::array<int64_t, max_dims> make_perm(Layout from, Layout to, int64_t ndim) {
    AT_ASSERT(ndim <= max_dims);
    // XXX: Note that only the first ndim entries will be used, so the rest can be
    // out of bound or even garbage.
    if (from == Layout::NCF && to == Layout::NFC) {
      std::array<int64_t, max_dims> r = {0, 2, 3, 4};
      r[ndim - 1] = 1;
      return r;
    } else if (from == Layout::NFC && to == Layout::NCF) {
      return {0, ndim - 1, 1, 2, 3};
    } else {
      AT_ASSERT(from == to);
      return {0, 1, 2, 3, 4};
    }
  }

  operator ArrayRef<int64_t>() const {
    return ArrayRef<int64_t>(perm_).slice(0, ndim_);
  }

  const std::array<int64_t, max_dims> perm_;
  const int64_t ndim_;
};

inline c10::optional<Layout> getLayout(const Tensor& t) {
  auto strides = t.strides();
  AT_ASSERT(strides.size() >= 3);
  if (strides[strides.size() - 1] == 1 && t.is_contiguous()) {
    return {Layout::NCF};
  }
  if (strides[1] == 1 &&
      t.permute(LayoutPermutation{Layout::NCF, Layout::NFC, t.dim()}).is_contiguous()) {
    return {Layout::NFC};
  }
  return {};
}

inline cudnnTensorFormat_t getCudnnTensorFormat(Layout l) {
  switch (l) {
    case Layout::NCF:
      return CUDNN_TENSOR_NCHW;
    case Layout::NFC:
      return CUDNN_TENSOR_NHWC;
  }
  AT_ASSERTM(false, "Unhandled layout in getCudnnTensorFormat");
}

inline Tensor contiguousIn(Tensor t, Layout layout) {
  switch (layout) {
    case Layout::NCF:
      return t.contiguous();
    case Layout::NFC:
      return t.permute(LayoutPermutation{Layout::NCF, Layout::NFC, t.dim()})
              .contiguous()
              .permute(LayoutPermutation{Layout::NFC, Layout::NCF, t.dim()});
  }
  AT_ASSERTM(false, "Unhandled layout in contiguous_in");
}

}} // namespace at::native
