#include "caffe2/operators/string_ops.h"
#include "caffe2/core/operator.h"

namespace caffe2 {
namespace {

struct StartsWith {
  explicit StartsWith(OperatorBase& op)
      : prefix_(op.GetSingleArgument<std::string>("prefix", "")) {}
  bool operator()(const std::string& str) {
    return std::mismatch(prefix_.begin(), prefix_.end(), str.begin()).first ==
        prefix_.end();
  }

 private:
  std::string prefix_;
};

struct EndsWith {
  explicit EndsWith(OperatorBase& op)
      : suffix_(op.GetSingleArgument<std::string>("suffix", "")) {}
  bool operator()(const std::string& str) {
    return std::mismatch(suffix_.rbegin(), suffix_.rend(), str.rbegin())
               .first == suffix_.rend();
  }

 private:
  std::string suffix_;
};

struct Prefix {
  explicit Prefix(OperatorBase& op)
      : length_(op.GetSingleArgument<int>("length", 3)) {}
  std::string operator()(const std::string& str) {
    return std::string(str.begin(), std::min(str.end(), str.begin() + length_));
  }

 private:
  int length_;
};

struct Suffix {
  explicit Suffix(OperatorBase& op)
      : length_(op.GetSingleArgument<int>("length", 3)) {}
  std::string operator()(const std::string& str) {
    return std::string(std::max(str.begin(), str.end() - length_), str.end());
  }

 private:
  int length_;
};

template <typename ScalarFunctor, typename TypeMap = FixedType<std::string>>
using StringElementwiseOp = UnaryElementwiseWithArgsOp<
    TensorTypes<std::string>,
    CPUContext,
    ForEach<ScalarFunctor>,
    TypeMap>;

REGISTER_CPU_OPERATOR(StringPrefix, StringElementwiseOp<Prefix>);
REGISTER_CPU_OPERATOR(StringSuffix, StringElementwiseOp<Suffix>);
REGISTER_CPU_OPERATOR(
    StringStartsWith,
    StringElementwiseOp<StartsWith, FixedType<bool>>);
REGISTER_CPU_OPERATOR(
    StringEndsWith,
    StringElementwiseOp<EndsWith, FixedType<bool>>);

OPERATOR_SCHEMA(StringPrefix)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Computes the element-wise string prefix of the string tensor.
Input strings that are shorter than prefix length will be returned unchanged.
NOTE: Prefix is computed on number of bytes, which may lead to wrong behavior
and potentially invalid strings for variable-length encodings such as utf-8.
)DOC")
    .Arg("length", "Maximum size of the prefix, in bytes.")
    .Input(0, "strings", "Tensor of std::string.")
    .Output(
        0,
        "prefixes",
        "Tensor of std::string containing prefixes for each input.");

OPERATOR_SCHEMA(StringSuffix)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Computes the element-wise string suffix of the string tensor.
Input strings that are shorter than suffix length will be returned unchanged.
NOTE: Prefix is computed on number of bytes, which may lead to wrong behavior
and potentially invalid strings for variable-length encodings such as utf-8.
)DOC")
    .Input(0, "strings", "Tensor of std::string.")
    .Output(
        0,
        "suffixes",
        "Tensor of std::string containing suffixes for each output.")
    .Arg("length", "Maximum size of the suffix, in bytes.");

OPERATOR_SCHEMA(StringStartsWith)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Performs the starts-with check on each string in the input tensor.
Returns tensor of boolean of the same dimension of input.
)DOC")
    .Arg("prefix", "The prefix to check input strings against.")
    .Input(0, "strings", "Tensor of std::string.")
    .Output(0, "bools", "Tensor of bools of same shape as input.");

OPERATOR_SCHEMA(StringEndsWith)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Performs the ends-with check on each string in the input tensor.
Returns tensor of boolean of the same dimension of input.
)DOC")
    .Arg("suffix", "The suffix to check input strings against.")
    .Input(0, "strings", "Tensor of std::string.")
    .Output(0, "bools", "Tensor of bools of same shape as input.");

SHOULD_NOT_DO_GRADIENT(StringPrefix);
SHOULD_NOT_DO_GRADIENT(StringSuffix);
SHOULD_NOT_DO_GRADIENT(StringStartsWith);
SHOULD_NOT_DO_GRADIENT(StringEndsWith);
}
} // namespace caffe2
