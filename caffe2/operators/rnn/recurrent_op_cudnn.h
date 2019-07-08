#ifndef CAFFE2_OPERATORS_RECURRENT_OP_CUDNN_H_
#define CAFFE2_OPERATORS_RECURRENT_OP_CUDNN_H_

#include "caffe2/core/context.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/cudnn_wrappers.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {
namespace detail {

template <typename T>
class TensorDescriptors {
 public:
  TensorDescriptors(
      size_t n,
      const std::vector<int>& dim,
      const std::vector<int>& stride);
  ~TensorDescriptors();
  const cudnnTensorDescriptor_t* descs() const {
    return descs_.data();
  }

 private:
  std::vector<cudnnTensorDescriptor_t> descs_;
};

} // namespace detail

template <typename T>
class RecurrentBaseOp : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);
  template<class... Args> explicit RecurrentBaseOp(Args&&... args)
  : Operator<CUDAContext>(std::forward<Args>(args)...), cudnn_wrapper_(&context_) {
      CUDNN_ENFORCE(cudnnCreateDropoutDescriptor(&dropoutDesc_));
      CUDNN_ENFORCE(cudnnCreateRNNDescriptor(&rnnDesc_));
      CUDNN_ENFORCE(cudnnCreateFilterDescriptor(&wDesc_));
      CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&hxDesc_));
  }
  virtual ~RecurrentBaseOp();

 protected:
  void initialize(
      const Tensor& input,
      Tensor* dropoutStates = nullptr,
      // If passed, reshapes to the appropriate size
      Tensor* output = nullptr,
      Tensor* hiddenOutput = nullptr,
      Tensor* cellOutput = nullptr);

  CuDNNWrapper cudnn_wrapper_;
  cudnnDropoutDescriptor_t dropoutDesc_;
  cudnnRNNDescriptor_t rnnDesc_;
  cudnnFilterDescriptor_t wDesc_;
  cudnnTensorDescriptor_t hxDesc_;
  cudnnTensorDescriptor_t cxDesc_;
  cudnnTensorDescriptor_t hyDesc_;
  cudnnTensorDescriptor_t cyDesc_;

  std::unique_ptr<detail::TensorDescriptors<T>> xDesc_;
  std::unique_ptr<detail::TensorDescriptors<T>> yDesc_;

  std::vector<int64_t> cachedInputDims_;
  size_t reserveNbytes_;
  size_t cudnnWsNbytes_;

 private:
};

#define USE_RECURRENT_BASE_FUNCTIONS          \
  USE_OPERATOR_FUNCTIONS(CUDAContext);        \
  using RecurrentBaseOp<T>::cudnn_wrapper_;   \
  using RecurrentBaseOp<T>::dropoutDesc_;     \
  using RecurrentBaseOp<T>::rnnDesc_;         \
  using RecurrentBaseOp<T>::wDesc_;           \
  using RecurrentBaseOp<T>::hxDesc_;          \
  using RecurrentBaseOp<T>::cxDesc_;          \
  using RecurrentBaseOp<T>::hyDesc_;          \
  using RecurrentBaseOp<T>::cyDesc_;          \
  using RecurrentBaseOp<T>::xDesc_;           \
  using RecurrentBaseOp<T>::yDesc_;           \
  using RecurrentBaseOp<T>::cachedInputDims_; \
  using RecurrentBaseOp<T>::reserveNbytes_;   \
  using RecurrentBaseOp<T>::cudnnWsNbytes_;   \
  using RecurrentBaseOp<T>::initialize;

template <typename T>
class RecurrentOp : public RecurrentBaseOp<T> {
 public:
  USE_RECURRENT_BASE_FUNCTIONS
  template <class... Args>
  explicit RecurrentOp(Args&&... args)
      : RecurrentBaseOp<T>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override;

 protected:
  INPUT_TAGS(INPUT, HIDDEN_INPUT, CELL_INPUT, WEIGHT);
  OUTPUT_TAGS(OUTPUT, HIDDEN_OUTPUT, CELL_OUTPUT, RNN_SCRATCH, DROPOUT_STATES);
};

enum RecurrentParamOpMode { SET_PARAM, GET_PARAM };

template <typename T, RecurrentParamOpMode mode>
class RecurrentParamAccessOp : public RecurrentBaseOp<T> {
 public:
  USE_RECURRENT_BASE_FUNCTIONS
  template <class... Args>
  explicit RecurrentParamAccessOp(Args&&... args)
      : RecurrentBaseOp<T>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override;
};

template <typename T>
class RecurrentGradientOp : public RecurrentBaseOp<T> {
 public:
  USE_RECURRENT_BASE_FUNCTIONS
  template <class... Args>
  explicit RecurrentGradientOp(Args&&... args)
      : RecurrentBaseOp<T>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override;

 protected:
  INPUT_TAGS(
      INPUT,
      HIDDEN_INPUT,
      CELL_INPUT,
      WEIGHT,
      RNN_SCRATCH,
      OUTPUT,
      GRAD_OUTPUT,
      GRAD_HIDDEN_OUTPUT,
      GRAD_CELL_OUTPUT);
  OUTPUT_TAGS(
      GRAD_INPUT,
      GRAD_HIDDEN_INPUT,
      GRAD_CELL_INPUT,
      GRAD_WEIGHT,
      DROPOUT_STATES,
      RNN_SCRATCH_OUT);
};


} // namespace caffe2

#endif // CAFFE2_OPERATORS_RECURRENT_OP_CUDNN_H_
