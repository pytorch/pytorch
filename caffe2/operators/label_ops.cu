#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/label_ops.h"

namespace caffe2 {

__global__ void SplitLabelKernel(
    const int N,
    const int64_t* label_ind_data,
    const float* label_val_data,
    const int* offset_map,
    const int* eid_map,
    float* const* label_vec,
    int* const* eid_vec) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    auto ind = label_ind_data[i];
    auto val = label_val_data[i];
    auto offset = offset_map[i];

    label_vec[ind][offset] = val;
    eid_vec[ind][offset] = eid_map[i];
  }
}

template <>
bool SparseLabelSplitOp<float, CUDAContext>::RunOnDevice() {
  CAFFE_ENFORCE_EQ(OutputSize(), 2 * num_labels_ + 1);

  auto& len = Input(0);
  auto& label_ind = Input(1);
  auto& label_val = Input(2);
  const auto* len_data = len.data<int32_t>();
  const auto* label_ind_data = label_ind.data<int64_t>();
  const float* label_val_data = label_val.data<float>();

  auto N_len = len.dim(0);
  auto N = label_ind.dim(0);

  vector<int32_t> len_cpu(N_len);
  context_.CopyToCPU(N_len, len_data, len_cpu.data());

  vector<int64_t> label_ind_cpu(N);
  context_.CopyToCPU(N, label_ind_data, label_ind_cpu.data());

  CAFFE_ENFORCE_EQ(
      label_val.dim(0),
      N,
      "label_index should have the same length as label_value");

  CAFFE_ENFORCE_EQ(
      std::accumulate(len_cpu.data(), len_cpu.data() + N_len, 0),
      N,
      "The sum of length should be equal to the length of other inputs");

  vector<int> n_example_per_task(num_labels_, 0);
  vector<int> offset_map(N, 0);
  vector<int> eid_map(N, 0);

  int pos = 0;
  for (int i = 0; i < N_len; i++) {
    auto cur_len = len_cpu[i];
    for (int l = 0; l < cur_len; l++) {
      auto label_id = label_ind_cpu[pos];
      // label_id should start from 0
      CAFFE_ENFORCE_LT(label_id, num_labels_, "label_index out of range");
      CAFFE_ENFORCE_GE(label_id, 0, "label_index out of range");
      offset_map[pos] = n_example_per_task[label_id];
      n_example_per_task[label_id]++;
      eid_map[pos] = i;

      pos++;
    }
  }

  vector<float*> label_output(num_labels_);
  vector<int*> eid_output(num_labels_);

  for (int i = 0; i < num_labels_; i++) {
    auto* labels = Output(i, {n_example_per_task[i]}, at::dtype<float>());
    auto* eids =
        Output(i + num_labels_, {n_example_per_task[i]}, at::dtype<int>());

    label_output[i] = labels->mutable_data<float>();
    eid_output[i] = eids->mutable_data<int>();
  }


  auto* offset_map_output = Output(2 * num_labels_, {N}, at::dtype<int>());
  auto* offset_map_gpu = offset_map_output->mutable_data<int>();

  if (N == 0) {
    return true;
  }

  context_.CopyFromCPU<int>(N, offset_map.data(), offset_map_gpu);

  eid_map_buffer_.Resize(N);
  auto* eid_map_gpu = eid_map_buffer_.mutable_data<int>();

  context_.CopyFromCPU<int>(N, eid_map.data(), eid_map_gpu);

  label_output_ptr_buffer_.Resize(num_labels_);
  auto* label_output_gpu = label_output_ptr_buffer_.mutable_data<float*>();

  context_.Copy<float*, CPUContext, CUDAContext>(
      num_labels_, label_output.data(), label_output_gpu);

  eid_output_ptr_buffer_.Resize(num_labels_);
  auto* eid_output_gpu = eid_output_ptr_buffer_.mutable_data<int*>();

  context_.Copy<int*, CPUContext, CUDAContext>(
      num_labels_, eid_output.data(), eid_output_gpu);

  SplitLabelKernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,
      label_ind_data,
      label_val_data,
      offset_map_gpu,
      eid_map_gpu,
      label_output_gpu,
      eid_output_gpu);

  return true;
}

__global__ void FillValGradKernel(
    const int N,
    const int64_t* label_ind_data,
    const int32_t* offset_map,
    const float* const* val_grad_vec,
    float* output_data) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    auto offset = offset_map[i];
    auto ind = label_ind_data[i];

    output_data[i] = val_grad_vec[ind][offset];
  }
}

template <>
bool SparseLabelSplitGradientOp<float, CUDAContext>::RunOnDevice() {
  CAFFE_ENFORCE_EQ(InputSize(), num_labels_ + 3);

  auto& len = Input(0);
  auto& label_ind = Input(1);
  auto& offset_map = Input(num_labels_ + 2);
  const auto* len_data = len.data<int32_t>();
  const auto* label_ind_data = label_ind.data<int64_t>();
  const auto* offset_map_data = offset_map.data<int32_t>();

  auto N_len = len.dim(0);
  auto N = label_ind.dim(0);

  auto* output = Output(0, label_ind.sizes(), at::dtype<float>());
  auto* output_data = output->mutable_data<float>();

  if (N == 0) {
    return true;
  }

  vector<const float*> val_grad_vec(num_labels_);
  for (int i = 0; i < num_labels_; i++) {
    auto& val_grad = Input(i + 2);
    val_grad_vec[i] = val_grad.data<float>();
  }

  val_grad_ptr_buffer_.Resize(num_labels_);
  auto* val_grad_vec_gpu = val_grad_ptr_buffer_.mutable_data<const float*>();

  context_.Copy<const float*, CPUContext, CUDAContext>(
      num_labels_, val_grad_vec.data(), val_grad_vec_gpu);

  FillValGradKernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N, label_ind_data, offset_map_data, val_grad_vec_gpu, output_data);

  return true;
}

REGISTER_CUDA_OPERATOR(
    SparseLabelSplit,
    SparseLabelSplitOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    SparseLabelSplitGradient,
    SparseLabelSplitGradientOp<float, CUDAContext>);

} // namespace caffe2
