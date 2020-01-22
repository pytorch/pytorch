#pragma once

#include <ATen/ATen.h>

#ifdef USE_XNNPACK

#include <xnnpack.h>

// #pragma clang diagnostic fatal "-Wall -Wextra -Wpedantic"

namespace at {
namespace native {
namespace mobile {
namespace cpu {
namespace internal {

struct Layout final {
  // 4D Activation Maps
  struct Activation4D final {
    static constexpr size_t Batch = 0u;
    static constexpr size_t Channels = 1u;
    static constexpr size_t Height = 2u;
    static constexpr size_t Width = 3u;
  };

  // ND Activation Maps
  struct ActivationND final {
    static int64_t Batch(const IntArrayRef tensor) {
      if (C10_UNLIKELY(tensor.empty())) {
        return -1;
      }

      int64_t batch = 1;

      for (size_t index = 0u, dimension = tensor.size() - 1u; index < dimension; ++index) {
        batch *= tensor[index];
      }

      return batch;
    };

    static int64_t Channel(const IntArrayRef tensor) {
      if (C10_UNLIKELY(tensor.empty())) {
        return -1;
      }

      return tensor.back();
    };
  };

  // Convolution Filters
  struct Filter final {
    static constexpr size_t Output = 0u;
    static constexpr size_t Input = 1u;
    static constexpr size_t Height = 2u;
    static constexpr size_t Width = 3u;
  };

  // Parameters (Pooling Kernels, Dilation, Padding, Stride, etc.)
  struct Parameter final {
    static constexpr size_t Height = 0u;
    static constexpr size_t Width = 1u;
  };
};

struct Deleter final {
  void operator()(const xnn_operator_t op) const {
    xnn_delete_operator(op);
  }
};

using Operator = std::unique_ptr<xnn_operator, Deleter>;

} // namespace internal
} // namespace cpu
} // namespace mobile
} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
