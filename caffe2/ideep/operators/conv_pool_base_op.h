#ifndef CAFFE2_IDEEP_OPERATORS_CONV_POOL_BASE_OP_H_
#define CAFFE2_IDEEP_OPERATORS_CONV_POOL_BASE_OP_H_

#include <vector>

#include "caffe2/ideep/ideep_utils.h"
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {

class IDEEPConvPoolOpBase : public ConvPoolOpBase<IDEEPContext> {
 public:
  IDEEPConvPoolOpBase(const OperatorDef& operator_def, Workspace* ws)
     : ConvPoolOpBase<IDEEPContext>(operator_def, ws) {}
  virtual ~IDEEPConvPoolOpBase() {}

  inline const ideep::tensor& Input(int index) {
    return OperatorBase::template Input<ideep::tensor>(index);
  }
  inline ideep::tensor* Output(int index) {
    return OperatorBase::template Output<ideep::tensor>(index);
  }

  ideep::tensor::dims pad_tl() const {
    return {pad_t(), pad_l()};
  }

  ideep::tensor::dims pad_br() const {
    return {pad_b(), pad_r()};
  }

  ideep::tensor::dims CalcOutputDims(
      const ideep::tensor& input,
      int output_channel) {
    CAFFE_ENFORCE_GT(input.get_size(), 0);
    ideep::tensor::dims output_dims;
    const auto input_dims = input.get_dims();
    std::vector<std::int64_t> input_Tdims(
        input_dims.cbegin(), input_dims.cend());
    InferOutputSize(
        input_Tdims,
        output_channel,
        StorageOrder::NCHW, //order_,
        global_pooling_,
        legacy_pad_,
        dilation_,
        stride_,
        &kernel_,
        &pads_,
        &output_dims);
    return output_dims;
  }

  bool RunOnDevice() override {
    if (!global_pooling_) {
      for (int dim = 0; dim < kernel_.size(); ++dim) {
        CAFFE_ENFORCE_GT(kernel_[dim], 0);
      }
    }

    try {
      return RunOnDeviceWithOrderNCHW();
    } catch (ideep::error& e) {
      LOG(ERROR) << "IDEEP error:" << e.message;
      throw;
    }
  }
};

#define USE_IDEEP_CONV_POOL_BASE_FUNCTIONS()             \
  USE_OPERATOR_BASE_FUNCTIONS;                           \
  /* using override */ using IDEEPConvPoolOpBase::Input; \
  /* using override */ using IDEEPConvPoolOpBase::Output;

} // namespace caffe2

#endif // CAFFE2_IDEEP_OPERATORS_CONV_POOL_BASE_OP_H_
