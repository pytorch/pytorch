#include "caffe2/operators/utility_ops.h"

#include <type_traits>

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/cudnn_wrappers.h"
#include "caffe2/utils/conversions.h"

namespace caffe2 {

class CuDNNWeightedSumOp : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  CuDNNWeightedSumOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws), cudnn_wrapper_(&context_) {
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&data_desc_));
    CUDNN_ENFORCE(cudnnCreateOpTensorDescriptor(&add_desc_));
    // Both float and at::Half require opTensorCompType to be CUDNN_DATA_FLOAT.
    CUDNN_ENFORCE(cudnnSetOpTensorDescriptor(
        add_desc_, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));
  }

  ~CuDNNWeightedSumOp() override {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(data_desc_));
    CUDNN_ENFORCE(cudnnDestroyOpTensorDescriptor(add_desc_));
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, at::Half>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    if (std::is_same<T, at::Half>::value) {
      LOG(WARNING)
          << "CuDNN only support same type for data and weight, "
             "so the weight will be cast to at::Half when data type is Half.";
    }
    const int num_inputs = InputSize();
    CAFFE_ENFORCE_EQ(num_inputs % 2, 0);
    const auto& X0 = Input(0);
    const auto& weight0 = Input(1);
    CAFFE_ENFORCE_GT(X0.numel(), 0);
    CAFFE_ENFORCE_EQ(weight0.numel(), 1);
    const int input_size = X0.numel();
    SetTensorDescriptor(cudnnTypeWrapper<T>::type, input_size);

    // Note: removed Aliasing check, since Output already has
    // caching capability
    auto* Y = Output(0, X0.sizes(), at::dtype<T>());
    T* Y_data = Y->template mutable_data<T>();
    T alpha = convert::To<float, T>(0.0f);
    T beta = convert::To<float, T>(0.0f);
    if (num_inputs == 2) {
      CopyWeightToHost<T>(weight0.template data<float>(), &alpha);
      CUDNN_ENFORCE(cudnnAddTensor(
          cudnn_wrapper_.inline_cudnn_handle(),
          &alpha,
          data_desc_,
          X0.template data<T>(),
          cudnnTypeWrapper<T>::kZero(),
          data_desc_,
          Y_data));
      return true;
    }
    const auto& X1 = Input(2);
    CAFFE_ENFORCE(
        !IsInputOutputAlias(2, 0),
        "Input #2 is the same as output. If you want to do in-place updates, "
        "put the output as input #0.");
    const auto& weight1 = Input(3);
    CAFFE_ENFORCE_EQ(X1.numel(), input_size);
    CAFFE_ENFORCE_EQ(weight1.numel(), 1);
    CopyWeightToHost<T>(weight1.template data<float>(), &alpha);
    CopyWeightToHost<T>(weight0.template data<float>(), &beta);
    if (IsInputOutputAlias(0, 0)) {
      CUDNN_ENFORCE(cudnnAddTensor(
          cudnn_wrapper_.inline_cudnn_handle(),
          &alpha,
          data_desc_,
          X1.template data<T>(),
          &beta,
          data_desc_,
          Y_data));
    } else {
      CUDNN_ENFORCE(cudnnOpTensor(
          cudnn_wrapper_.inline_cudnn_handle(),
          add_desc_,
          &alpha,
          data_desc_,
          X1.template data<T>(),
          &beta,
          data_desc_,
          X0.template data<T>(),
          cudnnTypeWrapper<T>::kZero(),
          data_desc_,
          Y_data));
    }
    for (int i = 4; i < num_inputs; i += 2) {
      const auto& Xi = Input(i);
      // Do a check: if the input is the same as output, we have a problem -
      // in-place update should always only happen with the zeroth input.
      const std::string err_msg = "Input #" + to_string(i) +
          " is the same as output. If you want to do in-place updates, "
          "put the output as input #0.";
      CAFFE_ENFORCE(!IsInputOutputAlias(i, 0), err_msg);
      const auto& weighti = Input(i + 1);
      CAFFE_ENFORCE_EQ(Xi.numel(), input_size);
      CAFFE_ENFORCE_EQ(weighti.numel(), 1);
      CopyWeightToHost<T>(weighti.template data<float>(), &alpha);
      CUDNN_ENFORCE(cudnnAddTensor(
          cudnn_wrapper_.inline_cudnn_handle(),
          &alpha,
          data_desc_,
          Xi.template data<T>(),
          cudnnTypeWrapper<T>::kOne(),
          data_desc_,
          Y_data));
    }
    return true;
  }

 private:
  void SetTensorDescriptor(
      const cudnnDataType_t data_type,
      const int input_size) {
    if (cached_input_size_ != input_size) {
      cached_input_size_ = input_size;
      // Since the best performance is obtained when the tesor is HW-packed, we
      // put X.size() to W.
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          data_desc_,
          GetCudnnTensorFormat(StorageOrder::NCHW),
          data_type,
          1,
          1,
          1,
          input_size));
    }
  }

  template <typename T>
  void CopyWeightToHost(const float* src, T* dst);

  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t data_desc_;
  cudnnOpTensorDescriptor_t add_desc_;

  int cached_input_size_ = 0;
};

template <typename T>
void CuDNNWeightedSumOp::CopyWeightToHost(const float* src, T* dst) {
  float val;
  context_.template CopyToCPU<float>(1, src, &val);
  *dst = convert::To<float, T>(val);
}

template <>
void CuDNNWeightedSumOp::CopyWeightToHost<float>(const float* src, float* dst) {
  context_.CopyToCPU<float>(1, src, dst);
}

REGISTER_CUDNN_OPERATOR(WeightedSum, CuDNNWeightedSumOp);

} // namespace caffe2
