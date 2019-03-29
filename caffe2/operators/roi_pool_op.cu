#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/roi_pool_op.h"

namespace caffe2 {

namespace {

template <typename T>
inline __device__ T gpu_atomic_add(const T val, T* address);

template <>
inline __device__ float gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

template <typename T>
__global__ void ROIPoolForward(
    const int nthreads,
    const T* bottom_data,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const T* bottom_rois,
    T* top_data,
    int* argmax_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    int roi_start_w = roundf(offset_bottom_rois[1] * spatial_scale);
    int roi_start_h = roundf(offset_bottom_rois[2] * spatial_scale);
    int roi_end_w = roundf(offset_bottom_rois[3] * spatial_scale);
    int roi_end_h = roundf(offset_bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    T maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    const T* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        if (offset_bottom_data[bottom_index] > maxval) {
          maxval = offset_bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    if (argmax_data) {
      argmax_data[index] = maxidx;
    }
  }
}

template <typename T>
__global__ void ROIPoolBackward(
    const int nthreads,
    const T* top_diff,
    const int* argmax_data,
    const int num_rois,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    T* bottom_diff,
    const T* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    int bottom_offset = (roi_batch_ind * channels + c) * height * width;
    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_top_diff = top_diff + top_offset;
    T* offset_bottom_diff = bottom_diff + bottom_offset;
    const int* offset_argmax_data = argmax_data + top_offset;

    int argmax = offset_argmax_data[ph * pooled_width + pw];
    if (argmax != -1) {
      gpu_atomic_add(
          static_cast<T>(offset_top_diff[ph * pooled_width + pw]),
          offset_bottom_diff + argmax);
    }
  }
}

} // namespace

template <>
bool RoIPoolOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0); // Input data to pool
  auto& R = Input(1); // RoIs
  auto* Y = Output(0); // RoI pooled data
  auto* A = is_test_ ? nullptr : Output(1); // argmaxes

  // Handle empty rois
  if (R.numel() == 0) {
    Y->Resize(0, X.dim32(1), pooled_height_, pooled_width_);
    // mutable_data calls are needed to allocate the tensors
    Y->template mutable_data<float>();
    if (!is_test_) {
      A->Resize(Y->sizes());
      A->template mutable_data<int>();
    }
    return true;
  }

  Y->Resize(R.dim32(0), X.dim32(1), pooled_height_, pooled_width_);
  if (!is_test_) {
    A->Resize(Y->sizes());
  }
  int output_size = Y->numel();
  int* argmax_data = is_test_ ? nullptr : A->template mutable_data<int>();
  ROIPoolForward<float>
      <<<CAFFE_GET_BLOCKS(output_size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          output_size,
          X.data<float>(),
          spatial_scale_,
          X.dim32(1),
          X.dim32(2),
          X.dim32(3),
          pooled_height_,
          pooled_width_,
          R.data<float>(),
          Y->template mutable_data<float>(),
          argmax_data);
  return true;
}

template <>
bool RoIPoolGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0); // Input data to pool
  auto& R = Input(1); // RoIs
  auto& A = Input(2); // argmaxes
  auto& dY = Input(3); // Gradient of net w.r.t. output of "forward" op
  // (aka "gradOutput")

  auto* dX = Output(
      0, X.sizes(), at::dtype<float>()); // Gradient of net w.r.t. input to
                                         // "forward" op (aka "gradInput")
  // Must zero-out dX before accumulating gradients
  math::Set<float, CUDAContext>(
      dX->numel(), 0.f, dX->template mutable_data<float>(), &context_);
  if (dY.numel() > 0) { // Handle possibly empty gradient if there were no rois
    ROIPoolBackward<float>
        <<<CAFFE_GET_BLOCKS(dY.numel()),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            dY.numel(),
            dY.data<float>(),
            A.data<int>(),
            R.dim32(0),
            spatial_scale_,
            X.dim32(1),
            X.dim32(2),
            X.dim32(3),
            pooled_height_,
            pooled_width_,
            dX->template mutable_data<float>(),
            R.data<float>());
  }
  return true;
}

REGISTER_CUDA_OPERATOR(RoIPool, RoIPoolOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(RoIPoolGradient, RoIPoolGradientOp<float, CUDAContext>);

} // namespace caffe2
