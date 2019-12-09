#ifndef CAFFE2_OPERATORS_LOCALLY_CONNECTED_OP_UTIL_H_
#define CAFFE2_OPERATORS_LOCALLY_CONNECTED_OP_UTIL_H_

#include <vector>

#include "caffe2/core/types.h"

namespace caffe2 {
namespace lc_op_util {

struct ShapeParams {
  int N;
  int C;
  int M;
  int input_image_size;
  int output_image_size;
  int kernel_size;
  std::vector<int> X_dims;
  std::vector<int> column_slice_dims;
  std::vector<int> column_dims;
  std::vector<int> column_transposed_dims;
  std::vector<int> column_axes;
  std::vector<int> Y_dims;
  std::vector<int> Y_transposed_dims;
  std::vector<int> Y_axes;
};

struct CUDAConvNetShapeParams {
  int N;
  int C;
  int M;
  int X_H;
  int X_W;
  int Y_H;
  int Y_W;
};

CAFFE2_API void SetColumnBufferShape(
    int N,
    int kernel_dim,
    int output_image_size,
    const std::vector<int>& output_image_dims,
    StorageOrder order,
    std::vector<int>* column_slice_dims,
    std::vector<int>* column_dims,
    std::vector<int>* column_transposed_dims,
    std::vector<int>* column_axes);

CAFFE2_API void SetYBufferShape(
    int N,
    int M,
    int output_image_size,
    StorageOrder order,
    std::vector<int>* Y_dims,
    std::vector<int>* Y_transposed_dims,
    std::vector<int>* Y_axes);

} // namespace lc_op_util
} // namespace caffe2

#endif // CAFFE2_OPERATORS_LOCALLY_CONNECTED_OP_UTIL_H_
