#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"

#include "caffe2/contrib/nccl/cuda_nccl_gpu.h"

namespace caffe2 {

nccl::NCCLExecution getNCCLElements(
    OperatorBase* op,
    const CUDAContext& context) {
  // We either do an N-N op, or an N-1 op.
  CAFFE_ENFORCE(op->InputSize() == op->OutputSize() || op->OutputSize() == 1);
  nccl::NCCLExecution ex;
  ex.stream_gpu_id = context.device_id();
  ex.stream = context.cuda_stream();
  ex.root = op->template GetSingleArgument<int>("root", 0);
  ex.elements.resize(op->InputSize());
  for (auto i = 0; i < op->InputSize(); ++i) {
    auto& el = ex.elements[i];
    el.src = &(op->Input<Tensor>(i, CUDA));
    if (op->OutputSize() == 1) {
      // Reduce op
      if (i == ex.root) {
        el.dst = op->Output<Tensor>(0, CUDA);
      }
    } else if (i < op->OutputSize()) {
      el.dst = op->Output<Tensor>(i, CUDA);
    }
    // TODO - expensive (>1ms) - cache these.
    el.device = GetGPUIDForPointer(op->Input<Tensor>(i, CUDA).raw_data());
  }

  return ex;
}

namespace {

// Check if all inputs are float
template <typename T>
bool AllInputsAre(OperatorBase* op) {
  for (auto i = 0; i < op->InputSize(); ++i) {
    if (op->Input<Tensor>(i, CUDA).IsType<T>()) {
      continue;
    } else {
      return false;
    }
  }
  return true;
}

// Manual count of all instantiated NCCL ops.
// If this drops to zero after destructing the last NCCL op,
// it means we can safely destroy all lazily created NCCL contexts.
std::atomic<int> kNCCLOpCounter(0);

}; // namespace

class NCCLBaseOp : public Operator<CUDAContext> {
 public:
  using Operator::Operator;

  NCCLBaseOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws) {
    kNCCLOpCounter++;
  }

  ~NCCLBaseOp() {
    if (--kNCCLOpCounter == 0) {
      nccl::destroyContexts();
    }
  }
};

class NCCLAllreduceOp final : public NCCLBaseOp {
 public:
  using NCCLBaseOp::NCCLBaseOp;

  bool RunOnDevice() override {
    if (InputSize() == 1)
      return true;

    if (AllInputsAre<float>(this)) {
      nccl::NCCL<float>::AllReduce(getNCCLElements(this, context_));
      return true;
    } else if (AllInputsAre<at::Half>(this)) {
      nccl::NCCL<at::Half>::AllReduce(getNCCLElements(this, context_));
      return true;
    } else {
      return false;
    }
  }

  static std::vector<TensorShape> ShapeInference(
      const OperatorDef& def,
      const std::vector<TensorShape>& in) {
    auto n_outputs = def.output_size();
    CAFFE_ENFORCE(
        n_outputs == 1 || n_outputs == in.size(),
        "NCCLAllreduce only supports N-1 or N-N reductions");

    for (auto i = 0; i < in.size(); i++) {
      CAFFE_ENFORCE(
          in[0].dims_size() == in[i].dims_size(),
          "NCCLAllreduce requires inputs of same dimension");
      for (auto j = 0; j < in[0].dims_size(); j++) {
        CAFFE_ENFORCE(
            in[0].dims(j) == in[i].dims(j),
            "NCCLAllreduce requires inputs to be of same shape");
      }
    }

    std::vector<TensorShape> out(n_outputs);
    for (auto i = 0; i < out.size(); i++) {
      out[i] = in[0];
    }
    return out;
  }

  static struct OpSchema::Cost CostInference(
      const OperatorDef& def,
      const vector<TensorShape>& inputs) {
    CAFFE_ENFORCE_GE(inputs.size(), 1, "Conv requires at least 1 input");
    const TensorShape X0 = inputs[0];
    const auto nElem = nElemFromDim(inputs[0]);

    struct OpSchema::Cost c;
    c.flops = (inputs.size() - 1) * nElem;
    c.bytes_read = inputs.size() * nElem;
    c.bytes_written = def.output_size() * nElem;
    c.params_bytes = 0;
    return c;
  }
};

class NCCLBroadcastOp final : public NCCLBaseOp {
 public:
  using NCCLBaseOp::NCCLBaseOp;

  bool RunOnDevice() override {
    if (InputSize() == 1)
      return true;
    if (AllInputsAre<float>(this)) {
      nccl::NCCL<float>::Broadcast(getNCCLElements(this, context_));
      return true;
    } else if (AllInputsAre<at::Half>(this)) {
      nccl::NCCL<at::Half>::Broadcast(getNCCLElements(this, context_));
      return true;
    } else {
      return false;
    }
  }
};

class NCCLReduceOp final : public NCCLBaseOp {
 public:
  using NCCLBaseOp::NCCLBaseOp;

  bool RunOnDevice() override {
    if (InputSize() == 1)
      return true;
    const auto& ex = getNCCLElements(this, context_);

    if (AllInputsAre<float>(this)) {
      nccl::NCCL<float>::Reduce(ex);
      return true;
    } else if (AllInputsAre<at::Half>(this)) {
      nccl::NCCL<at::Half>::Reduce(ex);
      return true;
    } else {
      return false;
    }
  }
};

class NCCLAllGatherOp final : public NCCLBaseOp {
 public:
  using NCCLBaseOp::NCCLBaseOp;

  bool RunOnDevice() override {
    if (InputSize() == 1)
      return true;
    if (AllInputsAre<float>(this)) {
      nccl::NCCL<float>::AllGather(getNCCLElements(this, context_));
      return true;
    } else if (AllInputsAre<at::Half>(this)) {
      nccl::NCCL<at::Half>::AllGather(getNCCLElements(this, context_));
      return true;
    } else {
      return false;
    }
  }
};

class NCCLReduceScatterOp final : public NCCLBaseOp {
 public:
  using NCCLBaseOp::NCCLBaseOp;

  bool RunOnDevice() override {
    if (AllInputsAre<float>(this)) {
      nccl::NCCL<float>::ReduceScatter(getNCCLElements(this, context_));
      return true;
    } else if (AllInputsAre<at::Half>(this)) {
      nccl::NCCL<at::Half>::ReduceScatter(getNCCLElements(this, context_));
      return true;
    } else {
      return false;
    }
  }
};

namespace {

std::pair<std::vector<DeviceOption>, std::vector<DeviceOption>> ncclOpDevInfer(
    const OperatorDef& def) {
  std::vector<DeviceOption> opt;
  for (int i = 0; i < def.input().size(); ++i) {
    DeviceOption dev;
    dev.set_device_type(1);
    dev.set_device_id(i);
    opt.push_back(dev);
  }
  return std::make_pair(opt, opt);
}

REGISTER_CUDA_OPERATOR(NCCLAllreduce, NCCLAllreduceOp);
OPERATOR_SCHEMA(NCCLAllreduce)
    .NumInputs(1, C10_COMPILE_TIME_MAX_GPUS)
    .NumOutputs(1, C10_COMPILE_TIME_MAX_GPUS)
    .CostInferenceFunction(NCCLAllreduceOp::CostInference)
    .TensorInferenceFunction(NCCLAllreduceOp::ShapeInference)
    .IdenticalTypeAndShape()
    .InputsCanCrossDevices()
    .AllowOneToOneInplace()
    .DeviceInferenceFunction(ncclOpDevInfer);
SHOULD_NOT_DO_GRADIENT(NCCLAllreduce);

REGISTER_CUDA_OPERATOR(NCCLBroadcast, NCCLBroadcastOp);
OPERATOR_SCHEMA(NCCLBroadcast)
    .NumInputs(1, C10_COMPILE_TIME_MAX_GPUS)
    .NumOutputs(1, C10_COMPILE_TIME_MAX_GPUS)
    .IdenticalTypeAndShape()
    .InputsCanCrossDevices()
    .EnforceOneToOneInplace()
    .DeviceInferenceFunction(ncclOpDevInfer);

SHOULD_NOT_DO_GRADIENT(NCCLBroadcast);

REGISTER_CUDA_OPERATOR(NCCLReduce, NCCLReduceOp);
OPERATOR_SCHEMA(NCCLReduce)
    .NumInputs(1, C10_COMPILE_TIME_MAX_GPUS)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(0)
    .InputsCanCrossDevices()
    .AllowInplace([](int /*in*/, int out) -> bool { return (out == 0); })
    .DeviceInferenceFunction(ncclOpDevInfer);
SHOULD_NOT_DO_GRADIENT(NCCLReduce);

REGISTER_CUDA_OPERATOR(NCCLAllGather, NCCLAllGatherOp);
OPERATOR_SCHEMA(NCCLAllGather)
    .NumInputs(1, C10_COMPILE_TIME_MAX_GPUS)
    .NumOutputs(1, C10_COMPILE_TIME_MAX_GPUS)
    .InputsCanCrossDevices()
    .DeviceInferenceFunction(ncclOpDevInfer);
SHOULD_NOT_DO_GRADIENT(NCCLAllGather);

REGISTER_CUDA_OPERATOR(NCCLReduceScatter, NCCLReduceScatterOp);
OPERATOR_SCHEMA(NCCLReduceScatter)
    .NumInputs(1, C10_COMPILE_TIME_MAX_GPUS)
    .NumOutputs(1, C10_COMPILE_TIME_MAX_GPUS)
    .InputsCanCrossDevices()
    .DeviceInferenceFunction(ncclOpDevInfer);
SHOULD_NOT_DO_GRADIENT(NCCLReduceScatter);

} // namespace
} // namespace caffe2
