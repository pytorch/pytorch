#include "caffe2/operators/self_binning_histogram_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(SelfBinningHistogram, SelfBinningHistogramOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(SelfBinningHistogram)
    .NumInputs(1, INT_MAX)
    .NumOutputs(2)
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
        "histogram_values",
        "1D tensor of edges of the bins, of dimension [num_bins+1]."
        " The range appears as: [first, ..., last), wherein the i-th element"
        " expresses the start of a bin and i+1-th value represents the exclusive"
        " end of that bin.")
    .Output(
        1,
        "histogram_counts",
        "1D tensor of counts of each bin, of dimension [num_bins+1]."
        " It is guaranteed to end with a 0 since the last edge is exclusive.")
    .Arg("num_bins", "Number of bins to use for the histogram. Must be >= 1.")
    .Arg(
        "bin_spacing",
        "A string indicating 'linear' or 'logarithmic' spacing for the bins.")
    .Arg(
        "logspace_start",
        "A float that's used as the starting point for logarithmic spacing. "
        "Since logarithmic spacing cannot contain <=0 values this value will "
        "be used to represent all such values.")
    .Arg(
        "abs",
        "Apply abs() on every input value."
    );

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(SelfBinningHistogram);
} // namespace caffe2
