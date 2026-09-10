#pragma once

#ifdef USE_XNNPACK

#include <xnnpack.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <c10/util/ArrayRef.h>
#include <limits>
#include <memory>

namespace at::native::xnnpack {

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

  ContextLinear(Operator&& o, int64_t o_channels) : op(std::move(o)), output_channels(o_channels) {}
  static constexpr float kMin = -std::numeric_limits<float>::infinity();
  static constexpr float kMax = std::numeric_limits<float>::infinity();
};

// This contains information for both the transpose and non-transpose cases.
struct ContextConv2D final {
  Operator op;
  std::array<int64_t, 4> weight_size_;
  std::array<int64_t, 2> padding_;
  std::array<int64_t, 2> output_padding_;
  std::array<int64_t, 2> stride_;
  std::array<int64_t, 2> dilation_;
  bool transposed_;
  int64_t groups_;

  ContextConv2D() = delete;

  ContextConv2D(
      Operator&& o,
      std::array<int64_t, 4> weight_size,
      std::array<int64_t, 2> padding,
      std::array<int64_t, 2> output_padding,
      std::array<int64_t, 2> stride,
      std::array<int64_t, 2> dilation,
      bool transposed,
      int64_t groups)
      :  op(std::move(o)),
         weight_size_(weight_size),
         padding_(padding),
         output_padding_(output_padding),
         stride_(stride),
         dilation_(dilation),
         transposed_(transposed),
         groups_(groups) {}
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
      if (C10_UNLIKELY(tensor.empty())) {
        return -1;
      }

      // Handle the case where batch size is zero.
      int64_t batch = tensor[0];

      for (size_t index = 1u; index < (tensor.size() - 1u); ++index) {
        batch *= tensor[index];
      }

      return batch;
    }

    static int64_t channel(const IntArrayRef tensor) {
      if (C10_UNLIKELY(tensor.empty())) {
        return -1;
      }

      return tensor.back();
    }
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
} // namespace internal
} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */

namespace at::native::xnnpack {
bool available();
} // namespace at::native::xnnpack
