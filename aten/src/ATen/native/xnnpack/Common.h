#pragma once

#include <ATen/ATen.h>

#ifdef USE_XNNPACK

#include <xnnpack.h>

namespace at {
namespace native {
namespace xnnpack {

struct Deleter final {
  void operator()(const xnn_operator_t op) const {
    xnn_delete_operator(op);
  }
};

using Operator = std::unique_ptr<xnn_operator, Deleter>;

struct ContextBase {
  Operator op;

  static constexpr float kMin = -std::numeric_limits<float>::infinity();
  static constexpr float kMax = std::numeric_limits<float>::infinity();
};

struct ContextLinear final : public ContextBase {
  int64_t output_channels;

  ContextLinear() = default;

  ContextLinear(Operator&& o, int64_t o_channels) {
    op = std::move(o);
    output_channels = o_channels;
  }
};

struct ContextConv2D final : public ContextBase {
  std::vector<int64_t> weight_size;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;

  ContextConv2D() = default;

  ContextConv2D(Operator&& o, std::vector<int64_t> w_size,
      std::vector<int64_t> padding_,
      std::vector<int64_t> stride_,
      std::vector<int64_t> dilation_) {
    op = std::move(o);
    weight_size = std::move(w_size);
    padding = std::move(padding_);
    stride = std::move(stride_);
    dilation = std::move(dilation_);
  }
};

namespace internal {

struct Layout final {
  // 4D Activation Maps
  struct Activation4D final {
    static constexpr size_t batch = 0u;
    static constexpr size_t channels = 1u;
    static constexpr size_t height = 2u;
    static constexpr size_t width = 3u;
  };

  // ND Activation Maps
  struct ActivationND final {
    // Some operators may not be limited to 4 dimensional tensors. In that scenario,
    // XNNPACK denotes that operator with an _nc suffix and expects all dimensions,
    // except channels, to be flattened into one argument: batch_size.
    static int64_t batch(const IntArrayRef tensor) {
      if (C10_UNLIKELY(tensor.empty())) {
        return -1;
      }

      // Handle the case where batch size is zero.
      int64_t batch = tensor[0];

      for (size_t index = 1u; index < (tensor.size() - 1u); ++index) {
        batch *= tensor[index];
      }

      return batch;
    };

    static int64_t channel(const IntArrayRef tensor) {
      if (C10_UNLIKELY(tensor.empty())) {
        return -1;
      }

      return tensor.back();
    };
  };

  // Convolution Filters
  struct Filter final {
    static constexpr size_t output = 0u;
    static constexpr size_t input = 1u;
    static constexpr size_t height = 2u;
    static constexpr size_t width = 3u;
  };

  // Parameters (Pooling Kernels, Dilation, Padding, Stride, etc.)
  struct Parameter final {
    static constexpr size_t height = 0u;
    static constexpr size_t width = 1u;
  };
};

bool available();

} // namespace internal
} // namespace xnnpack
} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
