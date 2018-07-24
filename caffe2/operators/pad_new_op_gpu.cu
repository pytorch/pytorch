#include <algorithm>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/pad_new_op.h"

namespace caffe2 {

namespace {
template <typename T>
__global__ void PadConst(
    const int nthreads,
    const T* const Xdata,
    const int* x_dims,
    const int* x_size_from,
    const int* y_size_from,
    const int n_dim,
    const int* pads_,
    T* const Ydata) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int* pos = new int[n_dim];
    int temp = index;
    for (int j = 0; j < n_dim; ++j) {
      pos[j] = temp / y_size_from[j];
      temp %= y_size_from[j];
    }
    int break_flag = false;
    for (int j = 0; j < n_dim; ++j) {
      pos[j] -= pads_[j];
      if (pos[j] < 0 || pos[j] >= x_dims[j]) {
        break_flag = true;
        break;
      }
    }
    if (break_flag == true) {
      continue;
    }
    int k = 0;
    for (int j = 0; j < n_dim; ++j) {
      k += x_size_from[j] * pos[j];
    }
    Ydata[index] = Xdata[k];
  }
}

template <typename T>
__global__ void PadGradientConst(
    const int nthreads,
    const T* const dYdata,
    const int* dY_size_from,
    const int* dX_size_from,
    const int n_dim,
    const int* pads_,
    T* const dXdata) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int* pos = new int[n_dim];
    int temp = index;
    for (int j = 0; j < n_dim; ++j) {
      pos[j] = temp / dX_size_from[j];
      temp %= dX_size_from[j];
    }
    for (int j = 0; j < n_dim; ++j) {
      pos[j] += pads_[j];
    }
    int k = 0;
    for (int j = 0; j < n_dim; ++j) {
      k += dY_size_from[j] * pos[j];
    }
    dXdata[index] = dYdata[k];
  }
}

} // namespace

template <>
bool PadOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  const int n_dim = X.ndim();
  const vector<TIndex> x_dims = X.dims();
  std::vector<int> x_dims_int(x_dims.begin(), x_dims.end());
  vector<TIndex> y_dims(n_dim, 0);
  for (int i = 0; i < n_dim; ++i) {
    y_dims[i] = X.dim32(i) + pads_[i] + pads_[n_dim + i];
  }
  Y->Resize(y_dims);
  const int output_size = Y->size();
  int* x_size_from = new int[n_dim];
  int* y_size_from = new int[n_dim];
  for (int i = 0; i < n_dim; i++) {
    x_size_from[i] = X.size_from_dim(i + 1);
    y_size_from[i] = Y->size_from_dim(i + 1);
  }
  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();

  auto pads_tmp = CUDAContext::New(2 * n_dim * sizeof(int));
  auto pads_device_ = (int*)pads_tmp.first;
  context_.CopyBytes<CPUContext, CUDAContext>(
      sizeof(int) * 2 * n_dim,
      static_cast<void*>(pads_.data()),
      static_cast<void*>(pads_device_));

  auto x_dims_tmp = CUDAContext::New(n_dim * sizeof(int));
  auto x_dims_device_ = (int*)x_dims_tmp.first;
  context_.CopyBytes<CPUContext, CUDAContext>(
      sizeof(int) * n_dim,
      static_cast<void*>(x_dims_int.data()),
      static_cast<void*>(x_dims_device_));

  auto x_size_from_tmp = CUDAContext::New(n_dim * sizeof(int));
  auto x_size_from_device_ = (int*)x_size_from_tmp.first;
  context_.CopyBytes<CPUContext, CUDAContext>(
      n_dim * sizeof(int),
      static_cast<void*>(x_size_from),
      static_cast<void*>(x_size_from_device_));

  auto y_size_from_tmp = CUDAContext::New(n_dim * sizeof(int));
  auto y_size_from_device_ = (int*)y_size_from_tmp.first;
  context_.CopyBytes<CPUContext, CUDAContext>(
      n_dim * sizeof(int),
      static_cast<void*>(y_size_from),
      static_cast<void*>(y_size_from_device_));

  switch (mode_) {
    case PadMode::CONSTANT:
      math::Set<float, CUDAContext>(output_size, value_, Ydata, &context_);
      PadConst<float>
          <<<CAFFE_GET_BLOCKS(output_size),
             CAFFE_CUDA_NUM_THREADS,
             0,
             context_.cuda_stream()>>>(
              output_size,
              Xdata,
              x_dims_device_,
              x_size_from_device_,
              y_size_from_device_,
              n_dim,
              pads_device_,
              Ydata);
      break;
    case PadMode::REFLECT:
      break;
    case PadMode::EDGE:
      break;
  }
  return true;
}

template <>
bool PadGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& dY = Input(0);
  auto* dX = Output(0);
  const int n_dim = dY.ndim();
  const vector<TIndex> dY_dims = dY.dims();
  vector<TIndex> dX_dims(n_dim, 0);
  for (int i = 0; i < n_dim; i++) {
    dX_dims[i] = dY.dim32(i) - pads_[i] - pads_[n_dim + i];
  }
  dX->Resize(dX_dims);
  const int output_size = dX->size();
  int* dY_size_from = new int[n_dim];
  int* dX_size_from = new int[n_dim];
  for (int i = 0; i < n_dim; ++i) {
    dY_size_from[i] = dY.size_from_dim(i + 1);
    dX_size_from[i] = dX->size_from_dim(i + 1);
  }
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();

  auto pads_tmp = CUDAContext::New(2 * n_dim * sizeof(int));
  auto pads_device_ = (int*)pads_tmp.first;
  context_.CopyBytes<CPUContext, CUDAContext>(
      sizeof(int) * 2 * n_dim,
      static_cast<void*>(pads_.data()),
      static_cast<void*>(pads_device_));

  auto dY_size_from_tmp = CUDAContext::New(n_dim * sizeof(int));
  auto dY_size_from_device_ = (int*)dY_size_from_tmp.first;
  context_.CopyBytes<CPUContext, CUDAContext>(
      n_dim * sizeof(int),
      static_cast<void*>(dY_size_from),
      static_cast<void*>(dY_size_from_device_));

  auto dX_size_from_tmp = CUDAContext::New(n_dim * sizeof(int));
  auto dX_size_from_device_ = (int*)dX_size_from_tmp.first;
  context_.CopyBytes<CPUContext, CUDAContext>(
      n_dim * sizeof(int),
      static_cast<void*>(dX_size_from),
      static_cast<void*>(dX_size_from_device_));

  math::Set<float, CUDAContext>(output_size, 0, dXdata, &context_);

  switch (mode_) {
    case PadMode::CONSTANT:
      PadGradientConst<float>
          <<<CAFFE_GET_BLOCKS(output_size),
             CAFFE_CUDA_NUM_THREADS,
             0,
             context_.cuda_stream()>>>(
              output_size,
              dYdata,
              dY_size_from_device_,
              dX_size_from_device_,
              n_dim,
              pads_device_,
              dXdata);
      break;
    case PadMode::REFLECT:
      break;
    case PadMode::EDGE:
      break;
  }
  return true;
}

REGISTER_CUDA_OPERATOR(Pad, PadOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(PadGradient, PadGradientOp<float, CUDAContext>);
} // namespace caffe2
