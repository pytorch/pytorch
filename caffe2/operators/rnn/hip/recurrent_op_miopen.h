#ifndef CAFFE2_OPERATORS_RECURRENT_OP_MIOPEN_H_
#define CAFFE2_OPERATORS_RECURRENT_OP_MIOPEN_H_

#include "caffe2/core/context.h"
#include "caffe2/core/hip/context_gpu.h"
#include "caffe2/core/hip/miopen_wrapper.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {
namespace detail {

template <typename T>
class TensorDescriptors {
 public:
  TensorDescriptors(
      size_t n,
      // dim and stride are not declared as const as opposed to cuDNN
      // since miopenSetTensorDescriptor doesn't take const arguments
      std::vector<int>& dim,
      std::vector<int>& stride);
  ~TensorDescriptors();
  const miopenTensorDescriptor_t* descs() const {
    return descs_.data();
  }

 private:
  std::vector<miopenTensorDescriptor_t> descs_;
};

} // namespace detail

template <typename T>
class RecurrentBaseOp : public Operator<HIPContext> {
 public:
  USE_OPERATOR_FUNCTIONS(HIPContext);
  RecurrentBaseOp(const OperatorDef& operator_def, Workspace* ws);
  virtual ~RecurrentBaseOp();

 protected:
  void initialize(
      const Tensor& input,
      // If passed, reshapes to the appropriate size
      Tensor* output = nullptr,
      Tensor* hiddenOutput = nullptr,
      Tensor* cellOutput = nullptr);

  MIOPENWrapper miopen_wrapper_;
  miopenRNNDescriptor_t rnnDesc_;
  miopenTensorDescriptor_t wDesc_;
  miopenTensorDescriptor_t hxDesc_;
  miopenTensorDescriptor_t cxDesc_;
  miopenTensorDescriptor_t hyDesc_;
  miopenTensorDescriptor_t cyDesc_;

  std::unique_ptr<detail::TensorDescriptors<T>> xDesc_;
  std::unique_ptr<detail::TensorDescriptors<T>> yDesc_;

  std::vector<int64_t> cachedInputDims_;
  size_t reserveNbytes_;
  size_t miopenWsNbytes_;

 private:
};

#define USE_RECURRENT_BASE_FUNCTIONS          \
  USE_OPERATOR_FUNCTIONS(HIPContext);        \
  using RecurrentBaseOp<T>::miopen_wrapper_;   \
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
  using RecurrentBaseOp<T>::miopenWsNbytes_;   \
  using RecurrentBaseOp<T>::initialize;

template <typename T>
class RecurrentOp : public RecurrentBaseOp<T> {
 public:
  USE_RECURRENT_BASE_FUNCTIONS
  RecurrentOp(const OperatorDef& operator_def, Workspace* ws)
      : RecurrentBaseOp<T>(operator_def, ws) {}

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
  RecurrentParamAccessOp(const OperatorDef& operator_def, Workspace* ws)
      : RecurrentBaseOp<T>(operator_def, ws) {}

  bool RunOnDevice() override;
};

template <typename T>
class RecurrentGradientOp : public RecurrentBaseOp<T> {
 public:
  USE_RECURRENT_BASE_FUNCTIONS
  RecurrentGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : RecurrentBaseOp<T>(operator_def, ws) {}

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

#endif // CAFFE2_OPERATORS_RECURRENT_OP_MIOPEN_H_
