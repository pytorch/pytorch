#include "store_ops.h"

#include "caffe2/core/blob_serialization.h"

namespace caffe2 {

constexpr auto kBlobName = "blob_name";
constexpr auto kAddValue = "add_value";

StoreSetOp::StoreSetOp(const OperatorDef& operator_def, Workspace* ws)
    : Operator<CPUContext>(operator_def, ws),
      blobName_(
          GetSingleArgument<std::string>(kBlobName, operator_def.input(DATA))) {
}

bool StoreSetOp::RunOnDevice() {
  // Serialize and pass to store
  auto* handler =
      OperatorBase::Input<std::unique_ptr<StoreHandler>>(HANDLER).get();
  handler->set(blobName_, SerializeBlob(InputBlob(DATA), blobName_));
  return true;
}

REGISTER_CPU_OPERATOR(StoreSet, StoreSetOp);
OPERATOR_SCHEMA(StoreSet)
    .NumInputs(2)
    .NumOutputs(0)
    .SetDoc(R"DOC(
Set a blob in a store. The key is the input blob's name and the value
is the data in that blob. The key can be overridden by specifying the
'blob_name' argument.
)DOC")
    .Arg("blob_name", "alternative key for the blob (optional)")
    .Input(0, "handler", "unique_ptr<StoreHandler>")
    .Input(1, "data", "data blob");

StoreGetOp::StoreGetOp(const OperatorDef& operator_def, Workspace* ws)
    : Operator<CPUContext>(operator_def, ws),
      blobName_(GetSingleArgument<std::string>(
          kBlobName,
          operator_def.output(DATA))) {}

bool StoreGetOp::RunOnDevice() {
  // Get from store and deserialize
  auto* handler =
      OperatorBase::Input<std::unique_ptr<StoreHandler>>(HANDLER).get();
  DeserializeBlob(handler->get(blobName_), OperatorBase::Outputs()[DATA]);
  return true;
}

REGISTER_CPU_OPERATOR(StoreGet, StoreGetOp);
OPERATOR_SCHEMA(StoreGet)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Get a blob from a store. The key is the output blob's name. The key
can be overridden by specifying the 'blob_name' argument.
)DOC")
    .Arg("blob_name", "alternative key for the blob (optional)")
    .Input(0, "handler", "unique_ptr<StoreHandler>")
    .Output(0, "data", "data blob");

StoreAddOp::StoreAddOp(const OperatorDef& operator_def, Workspace* ws)
    : Operator<CPUContext>(operator_def, ws),
      blobName_(GetSingleArgument<std::string>(kBlobName, "")),
      addValue_(GetSingleArgument<int64_t>(kAddValue, 1)) {
  CAFFE_ENFORCE(HasArgument(kBlobName));
}

bool StoreAddOp::RunOnDevice() {
  auto* handler =
      OperatorBase::Input<std::unique_ptr<StoreHandler>>(HANDLER).get();
  Output(VALUE)->Resize(1);
  Output(VALUE)->mutable_data<int64_t>()[0] =
      handler->add(blobName_, addValue_);
  return true;
}

REGISTER_CPU_OPERATOR(StoreAdd, StoreAddOp);
OPERATOR_SCHEMA(StoreAdd)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Add a value to a remote counter. If the key is not set, the store
initializes it to 0 and then performs the add operation. The operation
returns the resulting counter value.
)DOC")
    .Arg("blob_name", "key of the counter (required)")
    .Arg("add_value", "value that is added (optional, default: 1)")
    .Input(0, "handler", "unique_ptr<StoreHandler>")
    .Output(0, "value", "the current value of the counter");

StoreWaitOp::StoreWaitOp(const OperatorDef& operator_def, Workspace* ws)
    : Operator<CPUContext>(operator_def, ws),
      blobNames_(GetRepeatedArgument<std::string>(kBlobName)) {}

bool StoreWaitOp::RunOnDevice() {
  auto* handler =
      OperatorBase::Input<std::unique_ptr<StoreHandler>>(HANDLER).get();
  if (InputSize() == 2 && Input(1).IsType<std::string>()) {
    CAFFE_ENFORCE(
        blobNames_.empty(), "cannot specify both argument and input blob");
    std::vector<std::string> blobNames;
    auto* namesPtr = Input(1).data<std::string>();
    for (int i = 0; i < Input(1).size(); ++i) {
      blobNames.push_back(namesPtr[i]);
    }
    handler->wait(blobNames);
  } else {
    handler->wait(blobNames_);
  }
  return true;
}

REGISTER_CPU_OPERATOR(StoreWait, StoreWaitOp);
OPERATOR_SCHEMA(StoreWait)
    .NumInputs(1, 2)
    .NumOutputs(0)
    .SetDoc(R"DOC(
Wait for the specified blob names to be set. The blob names can be passed
either as an input blob with blob names or as an argument.
)DOC")
    .Arg("blob_names", "names of the blobs to wait for (optional)")
    .Input(0, "handler", "unique_ptr<StoreHandler>")
    .Input(1, "names", "names of the blobs to wait for (optional)");
}
