#include "caffe2/queue/blobs_queue_db.h"

#include <algorithm>
#include <chrono>
#include <random>
#include <string>

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/queue/blobs_queue.h"

#ifdef USE_MKLDNN
#include <caffe2/ideep/operators/operator_fallback_ideep.h>
#include <caffe2/ideep/utils/ideep_operator.h>
#endif

namespace caffe2 {
namespace db {

template <class Context>
class CreateBlobsQueueDBOp : public Operator<CPUContext> {
 public:
  CreateBlobsQueueDBOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    std::unique_ptr<db::DB> db = std::make_unique<BlobsQueueDB>(
        "",
        db::READ,
        OperatorBase::Input<std::shared_ptr<BlobsQueue>>(0),
        OperatorBase::template GetSingleArgument<int>("key_blob_index", -1),
        OperatorBase::template GetSingleArgument<int>("value_blob_index", 0),
        OperatorBase::template GetSingleArgument<float>("timeout_secs", 0.0));
    OperatorBase::Output<db::DBReader>(0)->Open(std::move(db), 1, 0);
    return true;
  }

 private:
  C10_DISABLE_COPY_AND_ASSIGN(CreateBlobsQueueDBOp);
};

REGISTER_CPU_OPERATOR(CreateBlobsQueueDB, CreateBlobsQueueDBOp<CPUContext>);

#ifdef USE_MKLDNN
REGISTER_IDEEP_OPERATOR(
    CreateBlobsQueueDB,
    IDEEPFallbackOp<CreateBlobsQueueDBOp<CPUContext>, SkipIndices<0>>);
#endif

OPERATOR_SCHEMA(CreateBlobsQueueDB)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg(
        "key_blob_index",
        "(default: -1 (no key)) index of blob for DB key in the BlobsQueue.")
    .Arg(
        "value_blob_index",
        "(default: 0) index of blob for DB value in the BlobsQueue.")
    .Arg(
        "timeout_secs",
        "(default: 0.0 (no timeout)) Timeout in seconds for reading from the "
        "BlobsQueue.")
    .SetDoc("Create a DBReader from a BlobsQueue")
    .Input(0, "queue", "The shared pointer to a queue containing Blobs.")
    .Output(0, "reader", "The DBReader for the given BlobsQueue");

SHOULD_NOT_DO_GRADIENT(CreateBlobsQueueDB);

} // namespace db
} // namespace caffe2
