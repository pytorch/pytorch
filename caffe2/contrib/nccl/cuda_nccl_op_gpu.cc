#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"

#include "cuda_nccl_gpu.h"

namespace caffe2 {

nccl::NCCLExecution getNCCLElements(
    OperatorBase* op,
    const CUDAContext& context) {
  // We either do an N-N op, or an N-1 op.
  CAFFE_ENFORCE(op->InputSize() == op->OutputSize() || op->OutputSize() == 1);
  nccl::NCCLExecution ex;
  ex.stream_gpu_id = context.cuda_gpu_id();
  ex.stream = context.cuda_stream();
  ex.root = op->template GetSingleArgument<int>("root", 0);
  ex.elements.resize(op->InputSize());
  for (auto i = 0; i < op->InputSize(); ++i) {
    auto& el = ex.elements[i];
    el.src = &(op->Input<TensorCUDA>(i));
    if (op->OutputSize() == 1) {
      // Reduce op
      if (i == ex.root) {
        el.dst = op->Output<TensorCUDA>(0);
      }
    } else if (i < op->OutputSize()) {
      el.dst = op->Output<TensorCUDA>(i);
    }
    // TODO - expensive (>1ms) - cache these.
    el.device = GetGPUIDForPointer(op->Input<TensorCUDA>(i).raw_data());
  }

  return ex;
}

namespace {
// Check if all inputs are float
template <typename T>
bool AllInputsAre(OperatorBase* op) {
  for (auto i = 0; i < op->InputSize(); ++i) {
    if (op->Input<TensorCUDA>(i).IsType<T>()) {
      continue;
    } else {
      return false;
    }
  }
  return true;
}
}; // namespace

class NCCLAllreduceOp final : public Operator<CUDAContext> {
 public:
  using Operator::Operator;
  NCCLAllreduceOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws) {}
  bool RunOnDevice() override {
    if (InputSize() == 1)
      return true;

    if (AllInputsAre<float>(this)) {
      nccl::NCCL<float>::AllReduce(getNCCLElements(this, context_));
      return true;
    } else if (AllInputsAre<float16>(this)) {
      nccl::NCCL<float16>::AllReduce(getNCCLElements(this, context_));
      return true;
    } else {
      return false;
    }
  }

 protected:
};

class NCCLBroadcastOp final : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);
  using Operator::Operator;
  bool RunOnDevice() override {
    if (InputSize() == 1)
      return true;
    if (AllInputsAre<float>(this)) {
      nccl::NCCL<float>::Broadcast(getNCCLElements(this, context_));
      return true;
    } else if (AllInputsAre<float16>(this)) {
      nccl::NCCL<float16>::Broadcast(getNCCLElements(this, context_));
      return true;
    } else {
      return false;
    }
  }

 protected:
};

class NCCLReduceOp final : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);
  using Operator::Operator;
  bool RunOnDevice() override {
    if (InputSize() == 1)
      return true;
    const auto& ex = getNCCLElements(this, context_);

    if (AllInputsAre<float>(this)) {
      nccl::NCCL<float>::Reduce(ex);
      return true;
    } else if (AllInputsAre<float16>(this)) {
      nccl::NCCL<float16>::Reduce(ex);
      return true;
    } else {
      return false;
    }
  }

 protected:
};

class NCCLAllGatherOp final : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);
  using Operator::Operator;
  bool RunOnDevice() override {
    if (InputSize() == 1)
      return true;
    if (AllInputsAre<float>(this)) {
      nccl::NCCL<float>::AllGather(getNCCLElements(this, context_));
      return true;
    } else if (AllInputsAre<float16>(this)) {
      nccl::NCCL<float16>::AllGather(getNCCLElements(this, context_));
      return true;
    } else {
      return false;
    }
  }

 protected:
};

class NCCLReduceScatterOp final : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);
  using Operator::Operator;
  bool RunOnDevice() override {
    if (AllInputsAre<float>(this)) {
      nccl::NCCL<float>::ReduceScatter(getNCCLElements(this, context_));
      return true;
    } else if (AllInputsAre<float16>(this)) {
      nccl::NCCL<float16>::ReduceScatter(getNCCLElements(this, context_));
      return true;
    } else {
      return false;
    }
  }

 protected:
};

namespace {

std::pair<std::vector<DeviceOption>, std::vector<DeviceOption>> ncclOpDevInfer(
    const OperatorDef& def) {
  std::vector<DeviceOption> opt;
  for (int i = 0; i < def.input().size(); ++i) {
    DeviceOption dev;
    dev.set_device_type(1);
    dev.set_cuda_gpu_id(i);
    opt.push_back(dev);
  }
  return std::make_pair(opt, opt);
}

REGISTER_CUDA_OPERATOR(NCCLAllreduce, NCCLAllreduceOp);
OPERATOR_SCHEMA(NCCLAllreduce)
    .NumInputs(1, CAFFE2_COMPILE_TIME_MAX_GPUS)
    .NumOutputs(1, CAFFE2_COMPILE_TIME_MAX_GPUS)
    .IdenticalTypeAndShape()
    .InputsCanCrossDevices()
    .AllowOneToOneInplace()
    .DeviceInferenceFunction(ncclOpDevInfer);
SHOULD_NOT_DO_GRADIENT(NCCLAllreduce);

REGISTER_CUDA_OPERATOR(NCCLBroadcast, NCCLBroadcastOp);
OPERATOR_SCHEMA(NCCLBroadcast)
    .NumInputs(1, CAFFE2_COMPILE_TIME_MAX_GPUS)
    .NumOutputs(1, CAFFE2_COMPILE_TIME_MAX_GPUS)
    .IdenticalTypeAndShape()
    .InputsCanCrossDevices()
    .EnforceOneToOneInplace()
    .DeviceInferenceFunction(ncclOpDevInfer);

SHOULD_NOT_DO_GRADIENT(NCCLBroadcast);

REGISTER_CUDA_OPERATOR(NCCLReduce, NCCLReduceOp);
OPERATOR_SCHEMA(NCCLReduce)
    .NumInputs(1, CAFFE2_COMPILE_TIME_MAX_GPUS)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(0)
    .InputsCanCrossDevices()
    .AllowInplace([](int in, int out) -> bool { return (out == 0); })
    .DeviceInferenceFunction(ncclOpDevInfer);
SHOULD_NOT_DO_GRADIENT(NCCLReduce);

REGISTER_CUDA_OPERATOR(NCCLAllGather, NCCLAllGatherOp);
OPERATOR_SCHEMA(NCCLAllGather)
    .NumInputs(1, CAFFE2_COMPILE_TIME_MAX_GPUS)
    .NumOutputs(1, CAFFE2_COMPILE_TIME_MAX_GPUS)
    .InputsCanCrossDevices()
    .DeviceInferenceFunction(ncclOpDevInfer);
SHOULD_NOT_DO_GRADIENT(NCCLAllGather);

REGISTER_CUDA_OPERATOR(NCCLReduceScatter, NCCLReduceScatterOp);
OPERATOR_SCHEMA(NCCLReduceScatter)
    .NumInputs(1, CAFFE2_COMPILE_TIME_MAX_GPUS)
    .NumOutputs(1, CAFFE2_COMPILE_TIME_MAX_GPUS)
    .InputsCanCrossDevices()
    .DeviceInferenceFunction(ncclOpDevInfer);
SHOULD_NOT_DO_GRADIENT(NCCLReduceScatter);
} // namespace

} // namespace caffe2
