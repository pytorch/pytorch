#include "concat_dnnlowp_op.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "dnnlowp_partition.h"

namespace caffe2 {

using namespace std;

template <typename T>
ConcatDNNLowPOp<T>::ConcatDNNLowPOp(
    const OperatorDef& operator_def,
    Workspace* ws)
    : BaseType(operator_def, ws) {
  if (HasArgument("axis")) {
    axis_ = this->template GetSingleArgument<int>("axis", -1);
    add_axis_ = this->template GetSingleArgument<int>("add_axis", 0);
  } else {
    axis_ = GetDimFromOrderString(
        this->template GetSingleArgument<string>("order", "NCHW"));
    add_axis_ = 0;
  }
  CAFFE_ENFORCE_GE(axis_, 0);
  requantization_params_.resize(InputSize());
}

template <typename T>
bool ConcatDNNLowPOp<T>::RunOnDevice() {
  GetQuantizationParameters_();

  auto* output = OutputTensorCPU_(0);
  Tensor* split = nullptr;
  int* axis_data = nullptr;
  if (OutputSize() >= 2) {
    split = this->template Output<Tensor>(1, CPU);
    split->Resize(vector<int64_t>(1, InputSize()));
    axis_data = split->template mutable_data<int>();
  }
  auto& input_zero = InputTensorCPU_(0);
  CAFFE_ENFORCE_LT(
      axis_,
      input_zero.ndim() + (add_axis_ ? 1 : 0),
      "Axis not in input ndim range.");
  for (int i = 1; i < InputSize(); ++i) {
    CAFFE_ENFORCE(
        InputTensorCPU_(i).dtype() == input_zero.dtype(),
        "All inputs must have the same type, expected: ",
        input_zero.dtype().name(),
        " but got: ",
        InputTensorCPU_(i).dtype().name(),
        " for input: ",
        i);
  }

  int before = 1, after = 1;
  vector<int64_t> output_dims(input_zero.sizes().vec());
  for (int i = 0; i < input_zero.ndim(); ++i) {
    if (i == axis_ && !add_axis_) {
      continue;
    }
    int dim = input_zero.dim32(i);
    if (dim == 0) {
      // In C2, batch size 0 is allowed, so we should just early return.
      return true;
    }
    if (i < axis_) {
      before *= dim;
    } else { // i > axis_ || i == axis_ && add_axis_
      after *= dim;
    }
    // check the input dims are compatible.
    for (int j = 1; j < InputSize(); ++j) {
      int dim_j = InputTensorCPU_(j).dim32(i);
      CAFFE_ENFORCE(
          dim == dim_j,
          "Expect dimension = ",
          dim,
          " got ",
          dim_j,
          " at axis = ",
          i,
          " for input: ",
          j,
          ". The input tensors can only have different dimensions "
          "when arg 'add_axis' = 0 and along the axis = ",
          axis_,
          " <",
          InputTensorCPU_(0).sizes(),
          "> vs <",
          InputTensorCPU_(j).sizes(),
          ">.");
    }
  }

  int output_channels = 0;
  for (int i = 0; i < InputSize(); ++i) {
    auto dim = add_axis_ ? 1 : InputTensorCPU_(i).dim32(axis_);
    if (axis_data) {
      axis_data[i] = dim;
    }
    output_channels += dim;
  }
  if (add_axis_) {
    output_dims.insert(output_dims.begin() + axis_, output_channels);
  } else {
    output_dims[axis_] = output_channels;
  }
  output->Resize(output_dims);

  if (input_zero.ndim() == 0 || before == 0 || after == 0) {
    LOG(WARNING) << "The input tensor size is 0!";
    return true;
  }

  size_t output_offset = 0;

  char* output_data = reinterpret_cast<char*>(GetQuantizedOutputData_());

  for (int i = 0; i < InputSize(); ++i) {
    auto& input = InputTensorCPU_(i);
    auto axis_dim = add_axis_ ? 1 : input.dim32(axis_);

    vector<T> input_temp(input.numel());
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int nthreads = dnnlowp_get_num_threads();
      int tid = dnnlowp_get_thread_num();
      int before_begin, before_end;
      int after_begin, after_end;

      Get1DPartitionOf2D(
          before,
          axis_dim * after,
          nthreads,
          tid,
          &before_begin,
          &before_end,
          &after_begin,
          &after_end);

      int j_begin = before_begin * axis_dim * after + after_begin;
      int j_end = (before_end - 1) * axis_dim * after + after_end;

      if (InputTensorCPU_(i).template IsType<T>()) {
        const T* input_data = input.template data<T>();
        for (int j = j_begin; j < j_end; ++j) {
          input_temp[j] = fbgemm::Requantize<T>(
              input_data[j] - in_qparams_[i].zero_point,
              requantization_params_[i]);
        }
      } else {
        fbgemm::Quantize<T>(
            input.template data<float>() + j_begin,
            input_temp.data() + j_begin,
            j_end - j_begin,
            out_qparams_);
      }

      math::CopyMatrix<CPUContext>(
          sizeof(T),
          before_end - before_begin,
          after_end - after_begin,
          input_temp.data() + before_begin * axis_dim * after + after_begin,
          axis_dim * after,
          output_data + output_offset + before_begin * output_channels * after +
              after_begin * sizeof(T),
          output_channels * after,
          &context_,
          input_zero.dtype().copy());
    }

    output_offset += axis_dim * after * sizeof(T);
  }

  RunOnDeviceEpilogue_();

  return true;
}

template <typename T>
void ConcatDNNLowPOp<T>::GetQuantizationParameters_() {
  using namespace dnnlowp;
  for (int i = 0; i < InputSize(); ++i) {
    in_qparams_[i] =
        GetInputTensorQuantizationParamsOf(this, i, qfactory_.get());
  }

  GetOutputQuantizationParams_();

  for (int i = 0; i < InputSize(); ++i) {
    float real_multiplier = in_qparams_[i].scale / out_qparams_.scale;
    requantization_params_[i] = qfactory_->ChooseRequantizationMultiplier(
        real_multiplier, out_qparams_);
  }
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(Concat, DNNLOWP, ConcatDNNLowPOp<uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8Concat,
    DNNLOWP,
    ConcatDNNLowPOp<uint8_t>);

} // namespace caffe2
