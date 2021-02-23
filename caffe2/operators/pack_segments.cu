#include <cub/cub.cuh>
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/pack_segments.h"

namespace caffe2 {

namespace {

template <typename T, typename Data_T>
__global__ void PackSegmentsKernel(
    const Data_T* data_ptr,
    const T* lengths_ptr,
    const T* lengths_cum_sum,
    const T max_length,
    const int64_t num_seq,
    const int64_t cell_size,
    Data_T padding,
    bool* presence_ptr,
    Data_T* out_ptr) {
  CUDA_1D_KERNEL_LOOP(i, num_seq * max_length * cell_size) {
    int seq = (i / cell_size) / max_length;
    int cell = (i / cell_size) % max_length;
    int offset = i % cell_size;
    if (presence_ptr && offset == 0) {
      presence_ptr[i / cell_size] = cell < lengths_ptr[seq];
    }
    if (cell >= lengths_ptr[seq]) {
      out_ptr[i] = padding;
    } else {
      int32_t idx = (lengths_cum_sum[seq] + cell) * cell_size + offset;
      out_ptr[i] = data_ptr[idx];
    }
  }
}

template <typename T, typename Data_T>
__global__ void UnpackSegmentsKernel(
    const Data_T* data_ptr,
    const T* lengths_ptr,
    const T* lengths_cum_sum,
    const T max_length,
    const int64_t num_seq,
    const int64_t cell_size,
    Data_T* out_ptr) {
  CUDA_1D_KERNEL_LOOP(i, num_seq * max_length * cell_size) {
    int seq = (i / cell_size) / max_length;
    int cell = (i / cell_size) % max_length;
    int offset = i % cell_size;
    if (cell < lengths_ptr[seq]) {
      int idx = (lengths_cum_sum[seq] + cell) * cell_size + offset;
      out_ptr[idx] = data_ptr[i];
    }
  }
}

template <typename T>
int64_t int_array_sum(
    const T* dev_array,
    int64_t num_items,
    Tensor& dev_buffer,
    Tensor& dev_sum,
    Tensor& host_sum,
    CUDAContext& context) {
  // Retrieve buffer size
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(
      nullptr,
      temp_storage_bytes,
      dev_array,
      dev_sum.mutable_data<int64_t>(),
      num_items,
      context.cuda_stream());

  // Allocate temporary storage
  auto buffer_size = (temp_storage_bytes + sizeof(T)) / sizeof(T);
  dev_buffer.Resize(buffer_size);
  void* dev_temp_storage = static_cast<void*>(dev_buffer.mutable_data<T>());

  // Find sumimum
  cub::DeviceReduce::Sum(
      dev_temp_storage,
      temp_storage_bytes,
      dev_array,
      dev_sum.mutable_data<int64_t>(),
      num_items,
      context.cuda_stream());

  // Copy to host
  host_sum.CopyFrom(dev_sum);
  context.FinishDeviceComputation();
  return *host_sum.data<int64_t>();
}

template <typename T>
T array_max(
    const T* dev_array,
    int64_t num_items,
    Tensor& dev_max_buffer,
    Tensor& dev_max,
    Tensor& host_max,
    CUDAContext& context) {
  // Retrieve buffer size
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Max(
      nullptr,
      temp_storage_bytes,
      dev_array,
      dev_max.mutable_data<T>(),
      num_items,
      context.cuda_stream());

  // Allocate temporary storage
  auto buffer_size = (temp_storage_bytes + sizeof(T)) / sizeof(T);
  dev_max_buffer.Resize(buffer_size);
  void* dev_temp_storage = static_cast<void*>(dev_max_buffer.mutable_data<T>());

  // Find maximum
  cub::DeviceReduce::Max(
      dev_temp_storage,
      temp_storage_bytes,
      dev_array,
      dev_max.mutable_data<T>(),
      num_items,
      context.cuda_stream());

  // Copy to host
  host_max.CopyFrom(dev_max);
  context.FinishDeviceComputation();
  return *host_max.data<T>();
}

template <typename T>
void array_prefix_sum_exclusive(
    const T* dev_array,
    const int32_t num_items,
    Tensor& prefix_buffer,
    Tensor& prefix_sum,
    CUDAContext& context) {
  // Retrieve buffer size
  size_t temp_storage_bytes = 0;
  prefix_sum.Resize(num_items);
  cub::DeviceScan::ExclusiveSum(
      nullptr,
      temp_storage_bytes,
      dev_array,
      prefix_sum.mutable_data<T>(),
      num_items,
      context.cuda_stream());

  // Allocate temporary storage
  auto buffer_size = (temp_storage_bytes + sizeof(T)) / sizeof(T);
  prefix_buffer.Resize(buffer_size);
  void* dev_temp_storage = static_cast<void*>(prefix_buffer.mutable_data<T>());

  // Exclusive sum
  cub::DeviceScan::ExclusiveSum(
      dev_temp_storage,
      temp_storage_bytes,
      dev_array,
      prefix_sum.mutable_data<T>(),
      num_items,
      context.cuda_stream());
}

} // namespace

template <>
template <typename T>
bool PackSegmentsOp<CUDAContext>::DoRunWithType() {
  return DispatchHelper<TensorTypes2<char, int32_t, int64_t, float>, T>::call(
      this, Input(DATA));
}

template <>
template <typename T, typename Data_T>
bool PackSegmentsOp<CUDAContext>::DoRunWithType2() {
  const auto& data = Input(DATA);
  const auto& lengths = Input(LENGTHS);
  int64_t num_seq = lengths.dim(0);
  const Data_T* data_ptr = data.data<Data_T>();
  const T* lengths_ptr = lengths.data<T>();
  auto* out = Output(0);
  Tensor* presence_mask = nullptr;
  if (return_presence_mask_) {
    presence_mask = Output(1);
  }

  CAFFE_ENFORCE_GE(data.dim(), 1, "DATA should be at least 1-D");
  CAFFE_ENFORCE_EQ(lengths.dim(), 1, "LENGTH should be 1-D");

  // Find the length of the longest sequence.
  dev_max_length_.Resize(1);
  host_max_length_.Resize(1);

  T temp = num_seq > 0 ? array_max<T>(
                             lengths_ptr,
                             num_seq,
                             dev_buffer_,
                             dev_max_length_,
                             host_max_length_,
                             context_)
                       : 0;
  if (max_length_ != -1) {
    CAFFE_ENFORCE_GE(
        max_length_,
        temp,
        "Pre-defined max_length should be greater than the real max_length");
    temp = max_length_;
  }
  const T& max_length = temp;
  // Compute prefix sum over the lengths
  array_prefix_sum_exclusive<T>(
      lengths_ptr, num_seq, dev_buffer_, dev_lengths_prefix_sum_, context_);
  bool* presence_mask_data = nullptr;
  if (return_presence_mask_) {
    std::vector<int64_t> presence_shape{lengths.numel(), max_length};
    presence_mask->Resize(presence_shape);
    presence_mask_data = presence_mask->template mutable_data<bool>();
  }

  // create output tensor
  auto shape = data.sizes().vec(); // Shape of out is batch_size x max_len x ...
  shape[0] = max_length;
  shape.insert(shape.begin(), lengths.numel());
  out->Resize(shape);
  Data_T* out_ptr = static_cast<Data_T*>(out->raw_mutable_data(data.meta()));

  // Return empty out (with the proper shape) if first dim is 0.
  if (!data.dim(0)) {
    return true;
  }

  // Do padding
  Data_T padding = out->IsType<float>() ? padding_ : 0;
  int64_t cell_size = data.numel() / data.dim(0);
  PackSegmentsKernel<<<
      CAFFE_GET_BLOCKS(num_seq * max_length * cell_size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      data_ptr,
      lengths_ptr,
      dev_lengths_prefix_sum_.data<T>(),
      max_length,
      num_seq,
      cell_size,
      padding,
      presence_mask_data,
      out_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
template <typename T>
bool UnpackSegmentsOp<CUDAContext>::DoRunWithType() {
  return DispatchHelper<TensorTypes2<char, int32_t, int64_t, float>, T>::call(
      this, Input(DATA));
}
template <>
template <typename T, typename Data_T>
bool UnpackSegmentsOp<CUDAContext>::DoRunWithType2() {
  const auto& data = Input(DATA);
  const auto& lengths = Input(LENGTHS);
  int64_t num_seq = lengths.dim(0);
  const Data_T* data_ptr = data.data<Data_T>();
  const T* lengths_ptr = lengths.data<T>();
  auto* out = Output(0);

  CAFFE_ENFORCE_GE(data.dim(), 1, "DATA should be at least 1-D");
  CAFFE_ENFORCE_EQ(lengths.dim(), 1, "LENGTH should be 1-D");
  // Compute prefix sum over the lengths
  array_prefix_sum_exclusive<T>(
      lengths_ptr, num_seq, dev_buffer_, dev_lengths_prefix_sum_, context_);

  // compute max of the lengths
  dev_max_length_.Resize(1);
  host_max_length_.Resize(1);
  T temp = num_seq > 0 ? array_max<T>(
                             lengths_ptr,
                             num_seq,
                             dev_buffer_,
                             dev_max_length_,
                             host_max_length_,
                             context_)
                       : 0;
  if (max_length_ != -1) {
    CAFFE_ENFORCE_EQ(
        max_length_,
        data.dim(1),
        "max_length should be equal to the packed segments");

    CAFFE_ENFORCE_GE(
        max_length_,
        temp,
        "Pre-defined max_length should be greater than the real max_length");

    temp = max_length_;
  }
  const T& max_length = temp;
  // compute num of cells: sum of the lengths
  dev_num_cell_.Resize(1);
  host_num_cell_.Resize(1);
  const int64_t num_cell = int_array_sum<T>(
      lengths_ptr,
      num_seq,
      dev_buffer_,
      dev_num_cell_,
      host_num_cell_,
      context_);

  // create output tensor
  auto shape = data.sizes().vec();
  CAFFE_ENFORCE_EQ(
      shape[0], lengths.dim(0), "LENGTH should match DATA in dimension 0");
  shape.erase(shape.begin());
  shape[0] = num_cell;
  out->Resize(shape);
  Data_T* out_ptr = static_cast<Data_T*>(out->raw_mutable_data(data.meta()));

  // Return empty out (with the proper shape) if any of the dimensions is 0.
  if (data.dim(0) == 0 || data.dim(1) == 0) {
    return true;
  }

  // Unpack
  int64_t cell_size = data.numel() / (data.dim(0) * data.dim(1));
  UnpackSegmentsKernel<<<
      CAFFE_GET_BLOCKS(num_seq * max_length * cell_size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      data_ptr,
      lengths_ptr,
      dev_lengths_prefix_sum_.data<T>(),
      max_length,
      num_seq,
      cell_size,
      out_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(UnpackSegments, UnpackSegmentsOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(PackSegments, PackSegmentsOp<CUDAContext>);
} // namespace caffe2
