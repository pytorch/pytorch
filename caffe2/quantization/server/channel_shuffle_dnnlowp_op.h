#pragma once

#include "caffe2/operators/channel_shuffle_op.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/quantization/server/conv_pool_dnnlowp_op_base.h"
#include "caffe2/quantization/server/dnnlowp.h"
#include "caffe2/quantization/server/dnnlowp_op.h"

namespace caffe2 {

namespace {

template <class Context>
using ChannelShuffleFp32Op = ChannelShuffleOp<float, Context>;

} // namespace

template <typename T>
class ChannelShuffleDNNLowPOp final
    : public DNNLowPOp<T, ChannelShuffleFp32Op<CPUContext>> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  USE_DNNLOWP_OPERATOR_BASE_FUNCTIONS(T, ChannelShuffleFp32Op<CPUContext>);

  ChannelShuffleDNNLowPOp(const OperatorDef& operator_def, Workspace* ws);

  bool RunOnDevice() override;

  bool RunOnDeviceWithOrderNCHW();
  bool RunOnDeviceWithOrderNHWC();

 private:
  const StorageOrder order_;
  const int group_;
};

} // namespace caffe2
