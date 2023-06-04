#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/integral_image_op.h"

namespace caffe2 {

namespace {
__global__ void RowPassKernel(
    int count,
    int rows_out,
    int cols_out,
    int chans,
    const float* in,
    float* out) {
  CUDA_1D_KERNEL_LOOP(i, count) {
    // Figure out which row, channel, and batch element we're processing
    int row = i % rows_out;
    int chan = (i / rows_out) % chans;
    int ind = i / rows_out / chans;
    // Input is (H, W) and output is (H + 1, W + 1)
    int rows_in = rows_out - 1;
    int cols_in = cols_out - 1;
    // Row pointer to input data
    // Input data is shift (-1, -1) relative to output data, hence row - 1
    const float* row_in_data =
        in + cols_in * ((row - 1) + rows_in * (chan + ind * chans));
    // Row pointer to output data
    float* row_out_data =
        out + cols_out * (row + rows_out * (chan + ind * chans));
    // The first row and first column of the output is all zeros
    row_out_data[0] = 0.;
    if (row == 0) {
      for (int i = 1; i < cols_out; ++i) {
        row_out_data[i] = 0.;
      }
    } else {
      for (int i = 1; i < cols_out; ++i) {
        // Recall that input data is shift (-1, -1) relative to the output,
        // hence i - 1
        row_out_data[i] = row_out_data[i - 1] + row_in_data[i - 1];
      }
    }
  }
}

__global__ void RowPassGradientKernel(
    int count,
    int rows_out,
    int cols_out,
    int chans,
    const float* in,
    float* out) {
  CUDA_1D_KERNEL_LOOP(i, count) {
    // Figure out which row, channel, and batch element we're processing
    int row = i % rows_out;
    int chan = (i / rows_out) % chans;
    int ind = i / rows_out / chans;
    // Input in (H + 1, W + 1) and output is (H + 1, W)
    int rows_in = rows_out;
    int cols_in = cols_out + 1;
    // Col pointer to input data
    const float* row_in_data =
        in + cols_in * (row + rows_in * (chan + ind * chans));
    // Col pointer to output data
    float* row_out_data =
        out + cols_out * (row + rows_out * (chan + ind * chans));
    row_out_data[0] = row_in_data[0];
    for (int i = 1; i < cols_out; ++i) {
      row_out_data[i] = row_out_data[i - 1] + row_in_data[i];
    }
  }
}

__global__ void
ColPassKernel(int count, int rows_out, int cols_out, int chans, float* out) {
  CUDA_1D_KERNEL_LOOP(i, count) {
    // Figure out which col, channel, and batch element we're processing
    int col = i % cols_out;
    int chan = (i / cols_out) % chans;
    int ind = i / cols_out / chans;
    float* col_out_data =
        out + col + cols_out * rows_out * (chan + ind * chans);
    for (int i = 1; i < rows_out; ++i) {
      col_out_data[i * cols_out] += col_out_data[(i - 1) * cols_out];
    }
  }
}

__global__ void ColPassGradientKernel(
    int count,
    int rows_out,
    int cols_out,
    int chans,
    const float* in,
    float* out) {
  CUDA_1D_KERNEL_LOOP(i, count) {
    // Figure out which col, channel, and batch element we're processing
    int col = i % cols_out;
    int chan = (i / cols_out) % chans;
    int ind = i / cols_out / chans;
    // Input is (H + 1, W) and output is (H, W)
    int rows_in = rows_out + 1;
    int cols_in = cols_out;
    // Col pointer to input data
    const float* col_in_data =
        in + col + cols_in * rows_in * (chan + ind * chans);
    // Col pointer to output data
    float* col_out_data =
        out + col + cols_out * rows_out * (chan + ind * chans);
    col_out_data[0] = col_in_data[0];
    for (int i = 1; i < rows_out; ++i) {
      col_out_data[i * cols_out] =
          col_out_data[(i - 1) * cols_out] + col_in_data[i * cols_in];
    }
  }
}

} // namespace

template <>
bool IntegralImageOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);

  CAFFE_ENFORCE(X.dim() == 4, "Only supports 4D tensors for the moment");

  // Input is (N, C, H, W)
  // Output is (N, C, H + 1, W + 1)
  vector<int64_t> out_shape(X.sizes().vec());
  out_shape[2] += 1; // H + 1 output size
  out_shape[3] += 1; // W + 1 output size
  auto* Y = Output(0, out_shape, at::dtype<float>());

  const int chans = X.dim32(1);
  const int rows_out = Y->dim32(2);
  const int cols_out = Y->dim32(3);
  // Integral image over rows of input X
  const int row_pass_size = X.dim32(0) * chans * rows_out;
  RowPassKernel<<<
      CAFFE_GET_BLOCKS(row_pass_size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      row_pass_size,
      rows_out,
      cols_out,
      chans,
      X.data<float>(),
      Y->template mutable_data<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Integral image over columns of the integral image over rows
  const int col_pass_size = X.dim32(0) * chans * cols_out;
  ColPassKernel<<<
      CAFFE_GET_BLOCKS(col_pass_size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      col_pass_size,
      rows_out,
      cols_out,
      chans,
      Y->template mutable_data<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
bool IntegralImageGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0); // Original input to "forward" op
  auto& dY = Input(1); // Gradient of net w.r.t. output of "forward" op
                       // (aka "gradOutput")

  auto* dX = Output(
      0, X.sizes(), at::dtype<float>()); // Gradient of net w.r.t. input to
                                         // "forward" op (aka "gradInput")

  // Row pass reduces shape of dY from (N, C, H + 1, W + 1)
  // to (N, C, H + 1, W)
  // Col pass reduces shape to (N, C, H, W)
  vector<int64_t> row_pass_shape(dY.sizes().vec());
  row_pass_shape[3] -= 1;
  ReinitializeTensor(&row_pass_buffer_, row_pass_shape, at::dtype<float>().device(CUDA));
  const int chans = row_pass_buffer_.dim32(1);
  const int rows_out = row_pass_buffer_.dim32(2);
  const int cols_out = row_pass_buffer_.dim32(3);
  // Integral image over rows of input X
  const int row_pass_size = X.dim32(0) * chans * rows_out;
  RowPassGradientKernel<<<
      CAFFE_GET_BLOCKS(row_pass_size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      row_pass_size,
      rows_out,
      cols_out,
      chans,
      dY.data<float>(),
      row_pass_buffer_.mutable_data<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Integral image over columns of the integral image over rows
  const int col_pass_size = X.dim32(0) * chans * cols_out;
  ColPassGradientKernel<<<
      CAFFE_GET_BLOCKS(col_pass_size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      col_pass_size,
      rows_out - 1,
      cols_out,
      chans,
      row_pass_buffer_.data<float>(),
      dX->template mutable_data<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(IntegralImage, IntegralImageOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    IntegralImageGradient,
    IntegralImageGradientOp<float, CUDAContext>);

} // namespace caffe2
