#pragma once

#include <ATen/ATen.h>

#ifdef USE_XNNPACK

#include <xnnpack.h>
#include "caffe2/utils/threadpool/ThreadPoolXNNPACK.h"

namespace at {
namespace native {
namespace xnnpack {

struct Deleter final {
  void operator()(const xnn_operator_t op) const {
    xnn_delete_operator(op);
  }
};

using Operator = std::unique_ptr<xnn_operator, Deleter>;

struct ContextLinear final {
  Operator op;
  int64_t output_channels;

  ContextLinear() = delete;

  ContextLinear(Operator&& o, int64_t o_channels) {
    op = std::move(o);
    output_channels = o_channels;
  }
  static constexpr float kMin = -std::numeric_limits<float>::infinity();
  static constexpr float kMax = std::numeric_limits<float>::infinity();
};

struct ContextConv2D final {
  Operator op;
  std::array<int64_t, 4> weight_size_;
  std::array<int64_t, 2> padding_;
  std::array<int64_t, 2> stride_;
  std::array<int64_t, 2> dilation_;

  ContextConv2D() = delete;

  ContextConv2D(
      Operator&& o,
      std::array<int64_t, 4> weight_size,
      std::array<int64_t, 2> padding,
      std::array<int64_t, 2> stride,
      std::array<int64_t, 2> dilation)
      :  op(std::move(o)),
         weight_size_(weight_size),
         padding_(padding),
         stride_(stride),
         dilation_(dilation) {}
  static constexpr float kMin = -std::numeric_limits<float>::infinity();
  static constexpr float kMax = std::numeric_limits<float>::infinity();
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
      // Empty dimensions are invalid.
      if (C10_UNLIKELY(tensor.empty())) {
        return -1;
      }

      // 1D tensors have a batch size of 0.
      // This single dimension is to be considered as channels.
      if (tensor.size() == 1) {
        return 0;
      }

      // For 2D tensors, or above:
      int64_t batch = tensor[0];

      for (size_t index = 1u; index < (tensor.size() - 1u); ++index) {
        batch *= tensor[index];
      }

      return batch;
    };

    static int64_t channel(const IntArrayRef tensor) {
      // Empty tensor dimensions are invalid.
      if (C10_UNLIKELY(tensor.empty())) {
        return -1;
      }

      // For tensors with dimensionality of 1D or above, consider the last
      // dimension as number of channels.
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
