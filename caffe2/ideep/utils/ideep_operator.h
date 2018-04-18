/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <ideep.hpp>
#include <caffe2/core/operator.h>
#include <caffe2/proto/caffe2.pb.h>
#include <caffe2/operators/conv_pool_op_base.h>

namespace caffe2 {

CAFFE_DECLARE_REGISTRY(
    IDEEPOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

#define REGISTER_IDEEP_OPERATOR_CREATOR(key, ...) \
  CAFFE_REGISTER_CREATOR(IDEEPOperatorRegistry, key, __VA_ARGS__)
#define REGISTER_IDEEP_OPERATOR(name, ...) \
  CAFFE_REGISTER_CLASS(IDEEPOperatorRegistry, name, __VA_ARGS__)
#define REGISTER_IDEEP_OPERATOR_STR(str_name, ...) \
  CAFFE_REGISTER_TYPED_CLASS(IDEEPOperatorRegistry, str_name, __VA_ARGS__)

#define REGISTER_IDEEP_OPERATOR_WITH_ENGINE(name, engine, ...) \
  CAFFE_REGISTER_CLASS(IDEEPOperatorRegistry, name##_ENGINE_##engine, __VA_ARGS__)

// IDEEPOperator is the base scaffolding of the operators that uses IDEEP. It
// provides a few operators that are useful to IDEEP specific implementations.
class IDEEPOperator : public OperatorBase {
 public:
  explicit IDEEPOperator(const OperatorDef& operator_def, Workspace* ws)
      : OperatorBase(operator_def, ws),
        context_(operator_def.device_option()),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    OPERATOR_NEEDS_FEATURE(
        order_ == StorageOrder::NCHW, "Unsupported storage order.");
  }
  virtual ~IDEEPOperator() {}

  inline const ideep::tensor& Input(int index) {
    return OperatorBase::template Input<ideep::tensor>(index);
  }
  inline ideep::tensor* Output(int index) {
    return OperatorBase::template Output<ideep::tensor>(index);
  }

  // The run function of Operator switches to the device, and then carries out
  // the actual computation with RunOnDevice(). You should implement RunOnDevice
  // instead of Run().
  bool Run(int /* unused */ /*stream_id*/) final {
    // Since IDEEP does not need to do SwithToDevice and
    // FinishDeviceComputation,
    // it is always just a re-route to RunOnDevice().
    try {
      return RunOnDevice();
    } catch (EnforceNotMet& err) {
      err.AppendMessage(getErrorMsg());
      throw;
    } catch (ideep::error& e) {
      VLOG(1) << "IDEEP error:" << e.message; 
      throw;
    }
  }

  // Waits for a previous event. Note that to properly wait and run
  // asynchronously, WaitEvent, RunAsync and Record should all be executed
  // on the same CPU thread.
  void WaitEvent(const Event& ev, int /* unused */) final {
    context_.WaitEvent(ev);
  }

  void WaitEvents(const std::vector<const Event*>& events, int /* unused */)
      final {
    for (const auto& ev : events) {
      context_.WaitEvent(*ev);
    }
  }

  void RecordEvent(const char* err_msg = nullptr) final {
    if (event_) {
      context_.Record(event_.get(), err_msg);
    }
  }

  virtual bool RunOnDevice() = 0;

 protected:
  std::string getErrorMsg() {
    if (has_debug_def()) {
      return "Error from operator: " + ProtoDebugString(debug_def());
    } else {
      return "Error from operator: no op def";
    }
  }

  IDEEPContext context_;
  StorageOrder order_;
};


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

#define USE_IDEEP_OPERATOR_FUNCTIONS()                                         \
  USE_OPERATOR_BASE_FUNCTIONS;                                                 \
  /* using override */ using IDEEPOperator::Input;                             \
  /* using override */ using IDEEPOperator::Output;                            \
  /* using override */ using IDEEPOperator::order_;                            \
  /* using override */ using IDEEPOperator::context_;

#define USE_IDEEP_CONV_POOL_BASE_FUNCTIONS()                                   \
  USE_OPERATOR_BASE_FUNCTIONS;                                                 \
  /* using override */ using IDEEPConvPoolOpBase::Input;                       \
  /* using override */ using IDEEPConvPoolOpBase::Output;

#define USE_SIMPLE_IDEEP_CTOR_DTOR(name)                                       \
  name(const OperatorDef& operator_def, Workspace* ws)                         \
      : IDEEPOperator(operator_def, ws) {}                                     \
  virtual ~name() {}

} // namespace caffe2
