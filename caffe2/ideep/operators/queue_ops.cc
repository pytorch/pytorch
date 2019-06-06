#include <caffe2/ideep/ideep_utils.h>
#include <caffe2/queue/blobs_queue.h>

using namespace caffe2;

namespace {

class IDEEPCreateBlobsQueueOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPCreateBlobsQueueOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        ws_(ws),
        name(operator_def.output().Get(0)) {}

  bool RunOnDevice() override {
    const auto capacity = GetSingleArgument("capacity", 1);
    const auto numBlobs = GetSingleArgument("num_blobs", 1);
    const auto enforceUniqueName =
        GetSingleArgument("enforce_unique_name", false);
    const auto fieldNames =
        OperatorBase::template GetRepeatedArgument<std::string>("field_names");
    CAFFE_ENFORCE_EQ(this->OutputSize(), 1);
    auto queuePtr = OperatorBase::Outputs()[0]
                        ->template GetMutable<std::shared_ptr<BlobsQueue>>();

    CAFFE_ENFORCE(queuePtr);
    *queuePtr = std::make_shared<BlobsQueue>(
        ws_, name, capacity, numBlobs, enforceUniqueName, fieldNames);
    return true;
  }

 private:
  Workspace* ws_{nullptr};
  const std::string name;
};

class IDEEPSafeEnqueueBlobsOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPSafeEnqueueBlobsOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws) {}

  bool RunOnDevice() override {
    auto queue =
        OperatorBase::Inputs()[0]->template Get<std::shared_ptr<BlobsQueue>>();
    CAFFE_ENFORCE(queue);
    auto size = queue->getNumBlobs();
    CAFFE_ENFORCE(
        OutputSize() == size + 1,
        "Expected " + caffe2::to_string(size + 1) + ", " +
            " got: " + caffe2::to_string(size));
    bool status = queue->blockingWrite(OperatorBase::Outputs());

    auto st = OperatorBase::Output<TensorCPU>(1, CPU);
    st->Resize();
    auto stat = st->template mutable_data<bool>();
    stat[0] = !status;
    return true;
  }
};

REGISTER_IDEEP_OPERATOR(CreateBlobsQueue, IDEEPCreateBlobsQueueOp);
SHOULD_NOT_DO_GRADIENT(IDEEPCreateBlobsQueueOp);

REGISTER_IDEEP_OPERATOR(SafeEnqueueBlobs, IDEEPSafeEnqueueBlobsOp);
SHOULD_NOT_DO_GRADIENT(IDEEPSafeEnqueueBlobsOp);

} // namespace
