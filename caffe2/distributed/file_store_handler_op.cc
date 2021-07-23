#include "file_store_handler_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    FileStoreHandlerCreate,
    FileStoreHandlerCreateOp<CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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
    .Arg("prefix", "prefix for all keys used by this store")
    .Output(0, "handler", "unique_ptr<StoreHandler>");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
NO_GRADIENT(FileStoreHandlerCreateOp);

} // namespace caffe2
