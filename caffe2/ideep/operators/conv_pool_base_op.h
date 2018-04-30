#pragma once

#include <caffe2/ideep/ideep_utils.h>
#include <caffe2/operators/conv_pool_op_base.h>

namespace caffe2 {

class IDEEPConvPoolOpBase : public ConvPoolOpBase<IDEEPContext> {
 public:
  IDEEPConvPoolOpBase(const OperatorDef& operator_def, Workspace* ws)
     : ConvPoolOpBase<IDEEPContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(
        order_ == StorageOrder::NCHW, "Unsupported storage order.");
  }
  virtual ~IDEEPConvPoolOpBase() {}

  inline const ideep::tensor& Input(int index) {
    return OperatorBase::template Input<ideep::tensor>(index);
  }
  inline ideep::tensor* Output(int index) {
    return OperatorBase::template Output<ideep::tensor>(index);
  }

  ideep::tensor::dims pad_tl() {
    return {pad_t(), pad_l()};
  }

  ideep::tensor::dims pad_br() {
    return {pad_b(), pad_r()};
  }

  ideep::tensor::dims CalcOutputDims(
      const ideep::tensor& input,
      int output_channel) {
    CAFFE_ENFORCE(input.get_descriptor().get_size() > 0);

    bool channel_first;
    int N = input.get_dim(0);
    ideep::tensor::dims output_dims;

    auto input_dims = input.get_dims();
    vector<TIndex> input_Tdims (input_dims.begin(), input_dims.end());
    InferOutputSize(
        input_Tdims,
        output_channel,
        order_,
        global_pooling_,
        legacy_pad_,
        N,
        kernel_,
        output_dims,
        dilation_,
        stride_,
        pads_,
        channel_first);

    if (channel_first) {
      output_dims.insert(output_dims.begin(), {N, output_channel});
    } else {
      output_dims.insert(output_dims.begin(), N);
      output_dims.push_back(output_channel);
    }

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
      VLOG(1) << "IDEEP error:" << e.message; 
      throw;
    }
  }
};

#define USE_IDEEP_CONV_POOL_BASE_FUNCTIONS()                                   \
  USE_OPERATOR_BASE_FUNCTIONS;                                                 \
  /* using override */ using IDEEPConvPoolOpBase::Input;                       \
  /* using override */ using IDEEPConvPoolOpBase::Output;

} // namespace caffe2
