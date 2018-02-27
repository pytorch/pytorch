#include "caffe2/operators/locally_connected_op_util.h"

namespace caffe2 {
namespace lc_op_util {

void SetColumnBufferShapeImpl(
    const int N,
    const int kernel_dim,
    const StorageOrder order,
    const std::vector<int>& output_image_dims,
    std::vector<int>* column_dims,
    std::vector<int>* column_transposed_dims,
    std::vector<int>* column_axes,
    std::vector<int>* column_transposed_axes) {
  const int n_column_dims = output_image_dims.size() + 2;
  column_dims->resize(n_column_dims);
  column_transposed_dims->resize(n_column_dims);
  column_axes->resize(n_column_dims);
  if (order == StorageOrder::NCHW) {
    for (int i = 0; i < n_column_dims - 2; ++i) {
      column_dims->at(i + 2) = output_image_dims[i];
      column_transposed_dims->at(i) = output_image_dims[i];
      column_axes->at(i) = i + 2;
    }
    column_dims->at(0) = N;
    column_dims->at(1) = kernel_dim;
    column_transposed_dims->at(n_column_dims - 2) = kernel_dim;
    column_transposed_dims->at(n_column_dims - 1) = N;
    column_axes->at(n_column_dims - 1) = 0;
    column_axes->at(n_column_dims - 2) = 1;
  } else {
    for (int i = 0; i < n_column_dims - 2; ++i) {
      column_dims->at(i + 1) = output_image_dims[i];
      column_transposed_dims->at(i) = output_image_dims[i];
      column_axes->at(i) = i + 1;
    }
    column_dims->at(0) = N;
    column_dims->at(n_column_dims - 1) = kernel_dim;
    column_transposed_dims->at(n_column_dims - 2) = N;
    column_transposed_dims->at(n_column_dims - 1) = kernel_dim;
    column_axes->at(n_column_dims - 2) = 0;
    column_axes->at(n_column_dims - 1) = n_column_dims - 1;
  }
  if (column_transposed_axes != nullptr) {
    column_transposed_axes->resize(n_column_dims);
    for (int i = 0; i < n_column_dims; ++i) {
      column_transposed_axes->at(column_axes->at(i)) = i;
    }
  }
}

} // namespace lc_op_util
} // namespace caffe2
