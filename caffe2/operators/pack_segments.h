#ifndef CAFFE2_OPERATORS_PACK_SEGMENTS_H_
#define CAFFE2_OPERATORS_PACK_SEGMENTS_H_

#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class PackSegmentsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  // USE_SIMPLE_CTOR_DTOR(PackSegmentsOp)
  USE_DISPATCH_HELPER;

  PackSegmentsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        pad_minf_(OperatorBase::GetSingleArgument<bool>("pad_minf", false)),
        return_presence_mask_(OperatorBase::GetSingleArgument<bool>(
            "return_presence_mask",
            false)) {
    if (pad_minf_) {
      padding_ = -1.0 * std::numeric_limits<float>::infinity();
    } else {
      padding_ = 0;
    }
  }

  bool RunOnDevice() {
    return DispatchHelper<TensorTypes<int, long>>::call(this, Input(LENGTHS));
  }

  template <typename T>
  bool DoRunWithType();

  template <typename T, typename Data_T>
  bool DoRunWithType2();

  INPUT_TAGS(LENGTHS, DATA);

 private:
  bool pad_minf_;
  float padding_;
  bool return_presence_mask_;

  // Scratch space required by the CUDA version
  Tensor<Context> dev_buffer_;
  Tensor<Context> dev_lengths_prefix_sum_;
  Tensor<Context> dev_max_length_;
  Tensor<CPUContext> host_max_length_;
};

template <class Context>
class UnpackSegmentsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(UnpackSegmentsOp)
  USE_DISPATCH_HELPER;

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int, long>>::call(this, Input(LENGTHS));
  }

  template <typename T>
  bool DoRunWithType();

  template <typename T, typename Data_T>
  bool DoRunWithType2();

  INPUT_TAGS(LENGTHS, DATA);

 private:
  Tensor<Context> dev_buffer_;
  Tensor<Context> dev_lengths_prefix_sum_;
  Tensor<Context> dev_max_length_;
  Tensor<Context> dev_num_cell_;
  Tensor<CPUContext> host_max_length_;
  Tensor<CPUContext> host_num_cell_;
};

} // namespace caffe2
#endif // CAFFE2_OPERATORS_PACK_SEGMENTS_H_
