#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <class Context>
class PackSegmentsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  // USE_SIMPLE_CTOR_DTOR(PackSegmentsOp)
  USE_DISPATCH_HELPER;

  PackSegmentsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
      pad_minf_(
        OperatorBase::GetSingleArgument<bool>("pad_minf", false)) {
          if (pad_minf_) {
            padding_ = -1.0 * std::numeric_limits<float>::infinity();
          } else {
            padding_ = 0;
          }
        }


  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int, long>>::call(this, Input(LENGTHS));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& data = Input(DATA);
    const auto& lengths = Input(LENGTHS);
    auto* output = Output(0);

    CAFFE_ENFORCE(data.ndim() >= 1, "DATA should be at least 1-D");
    CAFFE_ENFORCE(lengths.ndim() == 1, "LENGTH should be 1-D");

    // Find the length of the longest sequence.
    const T* l = lengths.template data<T>();
    T max_length = l[0];
    for (T i = 1; i < lengths.dim(0); ++i) {
      max_length = std::max(max_length, l[i]);
    }

    auto shape = data.dims(); // Shape of output is batch_size x max_len x ...
    shape[0] = max_length;
    shape.insert(shape.begin(), lengths.size());
    output->Resize(shape);

    // Do padding
    math::Set<float, Context>(
        output->size(),
        padding_,
        output->template mutable_data<float>(),
        &context_);

    int block_size = data.size() / data.dim(0);
    int block_bytesize = data.nbytes() / data.dim(0);
    const auto* d = static_cast<const char*>(data.raw_data());
    auto* out = static_cast<char*>(output->raw_mutable_data(data.meta()));
    int start = 0;
    for (int i = 0; i < lengths.dim(0); ++i) {
      context_.template CopyItems<Context, Context>(
          data.meta(),
          l[i] * block_size,
          d + block_bytesize * start,
          out + block_bytesize * max_length * i);
      start += l[i];
    }

    return true;
  }

  INPUT_TAGS(LENGTHS, DATA);
  private:
    bool pad_minf_;
    float padding_;
};

template <class Context>
class UnpackSegmentsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(UnpackSegmentsOp)
  USE_DISPATCH_HELPER;

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int, long>>::call(this, Input(LENGTHS));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& data = Input(DATA);
    const auto& lengths = Input(LENGTHS);
    auto* output = Output(0);

    CAFFE_ENFORCE(data.ndim() >= 2, "DATA should be at least 2-D");
    CAFFE_ENFORCE(lengths.ndim() == 1, "LENGTH should be 1-D");

    const T* l = lengths.template data<T>();

    T max_length = l[0];
    for (T i = 1; i < lengths.dim(0); ++i) {
      max_length = std::max(max_length, l[i]);
    }
    T total_l = std::accumulate(l, l + lengths.dim(0), 0);

    auto shape = data.dims();
    CAFFE_ENFORCE(
        shape[0] == lengths.dim(0), "LENGTH should match DATA in dimension 0");
    shape.erase(shape.begin());
    shape[0] = total_l;
    output->Resize(shape);
    int block_size = data.size() / (data.dim(0) * data.dim(1));
    int block_bytesize = data.nbytes() / (data.dim(0) * data.dim(1));
    const auto* d = static_cast<const char*>(data.raw_data());
    auto* out = static_cast<char*>(output->raw_mutable_data(data.meta()));
    int start = 0;
    for (int i = 0; i < lengths.dim(0); ++i) {
      context_.template CopyItems<Context, Context>(
          data.meta(),
          l[i] * block_size,
          d + block_bytesize * data.dim(1) * i,
          out + block_bytesize * start);
      start += l[i];
    }
    return true;
  }

  INPUT_TAGS(LENGTHS, DATA);
};

REGISTER_CPU_OPERATOR(PackSegments, PackSegmentsOp<CPUContext>);
REGISTER_CPU_OPERATOR(UnpackSegments, UnpackSegmentsOp<CPUContext>);

OPERATOR_SCHEMA(PackSegments)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(
        "Map N dim tensor to N+1 dim based on length blob. Sequences that \
    are shorter than the longest sequence are padded with zeros.")
    .Input(
        0,
        "lengths",
        "1-d int/long tensor contains the length in each of the output.")
    .Input(1, "tensor", "N dim Tensor.")
    .Output(
        0,
        "packed_tensor",
        "N + 1 dim Tesor"
        "where dim(1) is the max length"
        ", dim(0) is the batch size.")
    .Arg(
        "pad_minf", "Padding number in the packed segments. Use true to pad \
    -infinity, otherwise pad zeros");
OPERATOR_SCHEMA(UnpackSegments)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc("Map N+1 dim tensor to N dim based on length blob")
    .Input(
        0,
        "lengths",
        "1-d int/long tensor contains the length in each of the input.")
    .Input(1, "tensor", "N+1 dim Tensor.")
    .Output(0, "packed_tensor", "N dim Tesor");

class GetPackSegmentsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "UnpackSegments",
        "",
        vector<string>{I(0), GO(0)},
        vector<string>{GI(1)},
        Def().arg());
  }
};
REGISTER_GRADIENT(PackSegments, GetPackSegmentsGradient);

class GetUnpackSegmentsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "PackSegments",
        "",
        vector<string>{I(0), GO(0)},
        vector<string>{GI(1)},
        Def().arg());
  }
};
REGISTER_GRADIENT(UnpackSegments, GetUnpackSegmentsGradient);
}
} // namespace
