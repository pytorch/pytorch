#include "file_store_handler_op.h"

namespace caffe2 {

FileStoreHandlerCreateOp::FileStoreHandlerCreateOp(
    const OperatorDef& operator_def,
    Workspace* ws)
    : Operator(operator_def, ws),
      basePath_(GetSingleArgument<std::string>("path", "")) {
  CHECK_NE(basePath_, "") << "path is a required argument";
}

bool FileStoreHandlerCreateOp::RunOnDevice() {
  auto ptr = std::unique_ptr<StoreHandler>(new FileStoreHandler(basePath_));
  *OperatorBase::Output<std::unique_ptr<StoreHandler>>(HANDLER) =
      std::move(ptr);
  return true;
}

REGISTER_CPU_OPERATOR(FileStoreHandlerCreate, FileStoreHandlerCreateOp);
OPERATOR_SCHEMA(FileStoreHandlerCreate)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Creates a unique_ptr<StoreHandler> that uses the filesystem as backing
store (typically a filesystem shared between many nodes, such as NFS).
This store handler is not built to be fast. Its recommended use is for
integration tests and prototypes where extra dependencies are
cumbersome. Use an ephemeral path to ensure multiple processes or runs
don't interfere.
)DOC")
    .Arg("path", "base path used by the FileStoreHandler")
    .Output(0, "handler", "unique_ptr<StoreHandler>");

NO_GRADIENT(FileStoreHandlerCreateOp);

} // namespace caffe2
