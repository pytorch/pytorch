#include "caffe2/operators/locally_connected_op_util.h"

namespace caffe2 {
namespace lc_op_util {

void SetColumnBufferShapeImpl(
    const int N,
    const int kernel_size,
    const int output_image_size,
    const StorageOrder order,
    std::vector<int>* column_dims,
    std::vector<int>* column_transposed_dims,
    std::vector<int>* column_axes,
    std::vector<int>* column_transposed_axes) {
  *column_dims = order == StorageOrder::NCHW
      ? std::vector<int>{N, kernel_size, output_image_size}
      : std::vector<int>{N, output_image_size, kernel_size};
  *column_transposed_dims = order == StorageOrder::NCHW
      ? std::vector<int>{output_image_size, kernel_size, N}
      : std::vector<int>{output_image_size, N, kernel_size};
  *column_axes = order == StorageOrder::NCHW ? std::vector<int>{2, 1, 0}
                                             : std::vector<int>{1, 0, 2};
  if (column_transposed_axes != nullptr) {
    *column_transposed_axes = order == StorageOrder::NCHW
        ? std::vector<int>{2, 1, 0}
        : std::vector<int>{1, 0, 2};
  }
}

void SetYBufferShapeImpl(
    const int N,
    const int M,
    const int output_image_size,
    const StorageOrder order,
    std::vector<int>* Y_dims,
    std::vector<int>* Y_transposed_dims,
    std::vector<int>* Y_axes,
    std::vector<int>* Y_transposed_axes) {
  *Y_dims = order == StorageOrder::NCHW
      ? std::vector<int>{N, M, output_image_size}
      : std::vector<int>{N, output_image_size, M};
  *Y_transposed_dims = order == StorageOrder::NCHW
      ? std::vector<int>{output_image_size, M, N}
      : std::vector<int>{output_image_size, N, M};
  if (Y_axes != nullptr) {
    *Y_axes = order == StorageOrder::NCHW ? std::vector<int>{2, 1, 0}
                                          : std::vector<int>{1, 0, 2};
  }
  if (Y_transposed_axes != nullptr) {
    *Y_transposed_axes = order == StorageOrder::NCHW
        ? std::vector<int>{2, 1, 0}
        : std::vector<int>{1, 0, 2};
  }
}

} // namespace lc_op_util
} // namespace caffe2
