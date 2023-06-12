#include "caffe2/operators/stats_put_ops.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/stats.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {
#define REGISTER_TEMPLATED_STAT_PUT_OP(OP_NAME, STAT_NAME, STAT_MACRO) \
  struct STAT_NAME {                                                   \
    CAFFE_STAT_CTOR(STAT_NAME);                                        \
    STAT_MACRO(stat_value);                                            \
  };                                                                   \
  REGISTER_CPU_OPERATOR(OP_NAME, TemplatePutOp<STAT_NAME>);

// NOLINTNEXTLINE(modernize-pass-by-value,cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_TEMPLATED_STAT_PUT_OP(
    AveragePut,
    AveragePutStat,
    CAFFE_AVG_EXPORTED_STAT)

OPERATOR_SCHEMA(AveragePut)
    .NumInputs(1)
    .NumOutputs(0)
    .Arg(
        "name",
        "(*str*): name of the stat. If not present, then uses name of input blob")
    .Arg(
        "magnitude_expand",
        "(*int64_t*): number to multiply input values by (used when inputting floats, as stats can only receive integers")
    .Arg(
        "bound",
        "(*boolean*): whether or not to clamp inputs to the max inputs allowed")
    .Arg(
        "default_value",
        "(*float*): Optionally provide a default value for receiving empty tensors")
    .SetDoc(R"DOC(
    Consume a value and pushes it to the global stat registry as an average.

    Github Links:
    - https://github.com/pytorch/pytorch/blob/main/caffe2/operators/stats_put_ops.cc

        )DOC")
    .Input(
        0,
        "value",
        "(*Tensor`<number>`*): A scalar tensor, representing any numeric value");

// NOLINTNEXTLINE(modernize-pass-by-value,cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_TEMPLATED_STAT_PUT_OP(
    IncrementPut,
    IncrementPutStat,
    CAFFE_EXPORTED_STAT)

OPERATOR_SCHEMA(IncrementPut)
    .NumInputs(1)
    .NumOutputs(0)
    .Arg(
        "name",
        "(*str*): name of the stat. If not present, then uses name of input blob")
    .Arg(
        "magnitude_expand",
        "(*int64_t*): number to multiply input values by (used when inputting floats, as stats can only receive integers")
    .Arg(
        "bound",
        "(*boolean*): whether or not to clamp inputs to the max inputs allowed")
    .Arg(
        "default_value",
        "(*float*): Optionally provide a default value for receiving empty tensors")
    .SetDoc(R"DOC(
    Consume a value and pushes it to the global stat registry as an sum.

    Github Links:
    - https://github.com/pytorch/pytorch/blob/main/caffe2/operators/stats_put_ops.cc

        )DOC")
    .Input(
        0,
        "value",
        "(*Tensor`<number>`*): A scalar tensor, representing any numeric value");

// NOLINTNEXTLINE(modernize-pass-by-value,cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_TEMPLATED_STAT_PUT_OP(
    StdDevPut,
    StdDevPutStat,
    CAFFE_STDDEV_EXPORTED_STAT)

OPERATOR_SCHEMA(StdDevPut)
    .NumInputs(1)
    .NumOutputs(0)
    .Arg(
        "name",
        "(*str*): name of the stat. If not present, then uses name of input blob")
    .Arg(
        "magnitude_expand",
        "(*int64_t*): number to multiply input values by (used when inputting floats, as stats can only receive integers")
    .Arg(
        "bound",
        "(*boolean*): whether or not to clamp inputs to the max inputs allowed")
    .Arg(
        "default_value",
        "(*float*): Optionally provide a default value for receiving empty tensors")
    .SetDoc(R"DOC(
      Consume a value and pushes it to the global stat registry as an standard deviation.

      Github Links:
      - https://github.com/pytorch/pytorch/blob/main/caffe2/operators/stats_put_ops.cc

        )DOC")
    .Input(
        0,
        "value",
        "(*Tensor`<number>`*): A scalar tensor, representing any numeric value");

} // namespace caffe2
