#include "caffe2/operators/locally_connected_op_util.h"

namespace caffe2 {
namespace lc_op_util {

void SetColumnBufferShape(
    const int N,
    const int kernel_size,
    const int output_image_size,
    const StorageOrder order,
    std::vector<int>* column_dims,
    std::vector<int>* column_transposed_dims,
    std::vector<int>* column_axes) {
  *column_dims = order == StorageOrder::NCHW
      ? std::vector<int>{N, kernel_size, output_image_size}
      : std::vector<int>{N, output_image_size, kernel_size};
  *column_transposed_dims = order == StorageOrder::NCHW
      ? std::vector<int>{output_image_size, kernel_size, N}
      : std::vector<int>{output_image_size, N, kernel_size};
  *column_axes = order == StorageOrder::NCHW ? std::vector<int>{2, 1, 0}
                                             : std::vector<int>{1, 0, 2};
}

void SetYBufferShape(
    const int N,
    const int M,
    const int output_image_size,
    const StorageOrder order,
    std::vector<int>* Y_dims,
    std::vector<int>* Y_transposed_dims,
    std::vector<int>* Y_axes) {
  *Y_dims = order == StorageOrder::NCHW
      ? std::vector<int>{N, M, output_image_size}
      : std::vector<int>{N, output_image_size, M};
  *Y_transposed_dims = order == StorageOrder::NCHW
      ? std::vector<int>{output_image_size, M, N}
      : std::vector<int>{output_image_size, N, M};
  *Y_axes = order == StorageOrder::NCHW ? std::vector<int>{2, 1, 0}
                                        : std::vector<int>{1, 0, 2};
}

} // namespace lc_op_util
} // namespace caffe2
