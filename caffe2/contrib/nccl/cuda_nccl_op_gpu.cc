#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"

#include "cuda_nccl_gpu.h"

namespace caffe2 {

nccl::NCCLExecution getNCCLElements(
    OperatorBase* op,
    const CUDAContext& context) {
  // We either do an N-N op, or an N-1 op.
  CHECK(
      op->def().input_size() == op->def().output_size() ||
      op->def().output_size() == 1);
  nccl::NCCLExecution ex;
  ex.stream_gpu_id = context.cuda_gpu_id();
  ex.stream = context.cuda_stream();
  ex.root = op->template GetSingleArgument<int>("root", 0);
  ex.elements.resize(op->def().input_size());
  for (auto i = 0; i < op->def().input_size(); ++i) {
    auto& el = ex.elements[i];
    el.src = &(op->Input<TensorCUDA>(i));
    if (i < op->def().output_size()) {
      el.dst = op->Output<TensorCUDA>(i);
    }
    // TODO - expensive (>1ms) - cache these.
    el.device = GetGPUIDForPointer(op->Input<TensorCUDA>(i).raw_data());
  }

  return ex;
}

template <typename T>
class NCCLAllreduceOp final : public Operator<CUDAContext> {
 public:
  using Operator::Operator;
  NCCLAllreduceOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws) {}
  bool RunOnDevice() override {
    nccl::NCCL<T>::AllReduce(getNCCLElements(this, context_));
    return true;
  }

 protected:
};

template <typename T>
class NCCLBroadcastOp final : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);
  using Operator::Operator;
  bool RunOnDevice() override {
    nccl::NCCL<T>::Broadcast(getNCCLElements(this, context_));
    return true;
  }

 protected:
};

template <typename T>
class NCCLReduceOp final : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);
  using Operator::Operator;
  bool RunOnDevice() override {
    const auto& ex = getNCCLElements(this, context_);
    CHECK_EQ(ex.root, 0)
        << "NCCLReduce has spurious deadlocks for non-zero root";
    nccl::NCCL<T>::Reduce(ex);
    return true;
  }

 protected:
};

template <typename T>
class NCCLAllGatherOp final : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);
  using Operator::Operator;
  bool RunOnDevice() override {
    nccl::NCCL<T>::AllGather(getNCCLElements(this, context_));
    return true;
  }

 protected:
};

namespace {
REGISTER_CUDA_OPERATOR(NCCLAllreduce, NCCLAllreduceOp<float>);
OPERATOR_SCHEMA(NCCLAllreduce)
    .NumInputs(2, CAFFE2_COMPILE_TIME_MAX_GPUS)
    .NumOutputs(2, CAFFE2_COMPILE_TIME_MAX_GPUS)
    .AllowOneToOneInplace();
SHOULD_NOT_DO_GRADIENT(NCCLAllreduce);

REGISTER_CUDA_OPERATOR(NCCLBroadcast, NCCLBroadcastOp<float>);
OPERATOR_SCHEMA(NCCLBroadcast)
    .NumInputs(2, CAFFE2_COMPILE_TIME_MAX_GPUS)
    .NumOutputs(2, CAFFE2_COMPILE_TIME_MAX_GPUS)
    .EnforceOneToOneInplace();
SHOULD_NOT_DO_GRADIENT(NCCLBroadcast);

REGISTER_CUDA_OPERATOR(NCCLReduce, NCCLReduceOp<float>);
OPERATOR_SCHEMA(NCCLReduce)
    .NumInputs(2, CAFFE2_COMPILE_TIME_MAX_GPUS)
    .NumOutputs(1)
    .AllowInplace({{0, 0}});
SHOULD_NOT_DO_GRADIENT(NCCLReduce);

REGISTER_CUDA_OPERATOR(NCCLAllGather, NCCLAllGatherOp<float>);
OPERATOR_SCHEMA(NCCLAllGather)
    .NumInputs(2, CAFFE2_COMPILE_TIME_MAX_GPUS)
    .NumOutputs(2, CAFFE2_COMPILE_TIME_MAX_GPUS);
SHOULD_NOT_DO_GRADIENT(NCCLAllGather);
} // namespace

} // namespace caffe2
