#include "caffe2/operators/feed_blob_op.h"

namespace caffe2 {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(FeedBlob, FeedBlobOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(FeedBlob);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(FeedBlob)
    .NumInputs(0, 0)
    .NumOutputs(1, 1)
    .SetDoc(R"DOC(
FeedBlobs the content of the blobs. The input and output blobs should be
one-to-one inplace.)DOC")
    .Arg(
        "value",
        "(string) if provided then we will use this string as the value for the"
        "provided output tensor");

} // namespace caffe2
