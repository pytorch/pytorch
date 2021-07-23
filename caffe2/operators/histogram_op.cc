#include "caffe2/operators/histogram_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Histogram, HistogramOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Histogram)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .SetDoc(
        R"DOC(
            Computes a histogram for values in the given list of tensors.
            For logging activation histograms for post-hoc analyses, consider using the
            HistogramObserver observer.
            For iteratively computing a histogram for all input tensors encountered through
            history, consider using the AccumulateHistogram operator.
            )DOC")
    .Input(0, "X1, X2, ...", "*(type: Tensor`<float>`)* List of input tensors.")
    .Output(
        0,
        "histogram",
        "1D tensor of length k, wherein the i-th element expresses the count of tensor values "
        "that fall within range [bin_edges[i], bin_edges[i + 1])")
    .Arg(
        "bin_edges",
        "length-(k + 1) sequence of float values wherein the i-th element represents the inclusive "
        "left boundary of the i-th bin for i in [0, k - 1] and the exclusive right boundary "
        "of the (i-1)-th bin for i in [1, k].");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(Histogram);
} // namespace caffe2
