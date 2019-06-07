#include <cub/block/block_reduce.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/rmac_regions_op.h"

namespace cub {

template <typename KeyT, typename ValueT>
inline __host__ __device__ bool operator<(
    const cub::KeyValuePair<KeyT, ValueT>& kv1,
    const cub::KeyValuePair<KeyT, ValueT>& kv2) {
  return (kv1.value < kv2.value) ||
      (kv1.value == kv2.value && kv2.key < kv1.key);
}

} // namespace cub

namespace caffe2 {

namespace {

__global__ void NumRMACRegionsKernel(
    const int W,
    const int H,
    const int min_step,
    const int max_step,
    const float overlap,
    const int scales,
    int* num_rois_data) {
  // steps(idx) regions for long dimension
  typedef cub::KeyValuePair<int, float> KeyValuePair; // <step, value>
  KeyValuePair kv, min_kv;
  min_kv.value = FLT_MAX;

  // Local reduction
  int minW = min(H, W);
  int diff = max(H, W) - minW;
  CUDA_1D_KERNEL_LOOP(index, max_step - min_step + 1) {
    kv.key = min_step + index;
    float b = diff / (1.0 * kv.key);
    kv.value = fabsf((minW * minW - minW * b) / (minW * minW) - overlap);

    if (kv < min_kv) {
      min_kv = kv;
    }
  }

  // Block-wise arg-min reduction to find step
  int step;
  {
    typedef cub::BlockReduce<KeyValuePair, CAFFE_CUDA_NUM_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    min_kv = BlockReduce(temp_storage).Reduce(min_kv, cub::Min());

    __shared__ int step_shared;
    if (threadIdx.x == 0) {
      step_shared = min_kv.key;
    }
    __syncthreads();
    step = step_shared;
  }

  // Region overplus per dimension
  int Wd = (W > H) ? step : 0;
  int Hd = (H > W) ? step : 0;

  // Local reduction to compute the total number of rois at all scales
  int num_rois = 0;
  CUDA_1D_KERNEL_LOOP(index, scales) {
    int l = index + 1;
    int region_size = 2 * minW / (l + 1);
    num_rois += (region_size > 0) ? ((l + Wd) * (l + Hd)) : 0;
  }

  // Block-wise sum reduction to compute num_rois at all scales
  {
    typedef cub::BlockReduce<int, CAFFE_CUDA_NUM_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    num_rois = BlockReduce(temp_storage).Sum(num_rois);
  }

  if (threadIdx.x == 0) {
    num_rois_data[0] = num_rois;
    num_rois_data[1] = Wd;
    num_rois_data[2] = Hd;
  }
}

__global__ void RMACRegionsKernel(
    const int W,
    const int H,
    const int N,
    const int* num_rois_data,
    float* output) {
  int num_rois = num_rois_data[0];
  int Wd = num_rois_data[1];
  int Hd = num_rois_data[2];

  // Block-wide temp shared storage for intermediate ROI results to avoid
  // uncoalesced writes to global mem
  __shared__ float output_shared[CAFFE_CUDA_NUM_THREADS * 5];

  CUDA_1D_KERNEL_LOOP(index, N) {
    int batch_id = index / num_rois;
    int roi_id = index % num_rois;

    int roi[5];
    roi[0] = batch_id;

    // Find the scale corresponding to this index and the roi_id relative
    // to the scale.
    int l = 0;
    int num_rois_at_scale = 0;
    do {
      roi_id -= num_rois_at_scale;
      l++;
      num_rois_at_scale = (l + Wd) * (l + Hd);
    } while (roi_id - num_rois_at_scale >= 0);

    int region_size = 2 * min(H, W) / (l + 1);
    float bw =
        (l + Wd - 1 > 0) ? ((W - region_size) / (1.0 * (l + Wd - 1))) : 0;
    float bh =
        (l + Hd - 1 > 0) ? ((H - region_size) / (1.0 * (l + Hd - 1))) : 0;

    int i = roi_id / (l + Hd);
    int j = roi_id % (l + Hd);

    roi[1] = bw * i;
    roi[2] = bh * j;
    // Careful with the borders
    if (roi[1] + region_size > W) {
      roi[1] -= (roi[1] + region_size - W);
    }
    if (roi[2] + region_size > H) {
      roi[2] -= (roi[2] + region_size - H);
    }
    roi[3] = roi[1] + region_size - 1;
    roi[4] = roi[2] + region_size - 1;

    // Writing directly to output (global memory) will result in uncoalesced
    // writes. Write output to shared mem first and then write ROI results to
    // global output in a coalesced manner.
    __syncthreads(); // Since output_shared is reused across loop iterations
    for (int i = 0; i < 5; ++i) {
      output_shared[threadIdx.x * 5 + i] = roi[i];
    }
    __syncthreads();
    int offset = index - threadIdx.x;
    float* output_offset = output + offset * 5;
    int num_threads = min(blockDim.x, N - offset); // Active threads in block
    for (int i = 0; i < 5; ++i) {
      output_offset[num_threads * i + threadIdx.x] =
          output_shared[num_threads * i + threadIdx.x];
    }
  }
}

} // namespace

template <>
bool RMACRegionsOp<CUDAContext>::RunOnDevice() {
  const auto& X = Input(0); // Input tensor
   // RoIs

  if (X.numel() == 0) {
    return true;
  }

  int batch_size = X.dim32(0);
  int H = X.dim32(2);
  int W = X.dim32(3);

  // Compute number of regions
  int min_step = 1;
  int max_step = 6;
  ReinitializeTensor(&num_rois_, {3}, at::dtype<int>().device(CUDA)); // num_rois, Wd, Hd
  NumRMACRegionsKernel<<<
      1,
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      W,
      H,
      min_step,
      max_step,
      overlap_,
      scales_,
      num_rois_.mutable_data<int>());

  // Bit awkward, but the size of the output tensor depends on the output of
  // NumRMACRegionsKernel (number of RoIs), so need to copy that to CPU
  // to Resize() output appropriately.
  int num_rois = 0;
  context_.CopyBytesToCPU(sizeof(int), num_rois_.data<int>(), &num_rois);
  int N = batch_size * num_rois;
  auto* output = Output(0, {N, 5}, at::dtype<float>()); // [batch_id x1 y1 x2 y2]

  // Compute region coordinates
  RMACRegionsKernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      W, H, N, num_rois_.data<int>(), output->template mutable_data<float>());

  return true;
}

REGISTER_CUDA_OPERATOR(RMACRegions, RMACRegionsOp<CUDAContext>);

} // namespace caffe2
