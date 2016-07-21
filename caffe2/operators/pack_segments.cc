#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

namespace {

template <class Context>
class PackSegmentsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(PackSegmentsOp)
  USE_DISPATCH_HELPER;

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

    const T* l = lengths.template data<T>();
    T max_l = l[0];
    for (T i = 1; i < lengths.dim(0); ++i) {
      max_l = std::max(max_l, l[i]);
    }

    auto shape = data.dims();
    shape[0] = max_l;
    shape.insert(shape.begin(), lengths.size());
    output->Resize(shape);

    int block_size = data.size() / data.dim(0);
    int block_bytesize = data.nbytes() / data.dim(0);
    const auto* d = static_cast<const char*>(data.raw_data());
    auto* out = static_cast<char*>(output->raw_mutable_data(data.meta()));
    int start = 0;
    for (int i = 0; i < lengths.dim(0); ++i) {
      if (data.meta().copy() == nullptr) {
        context_.template CopyBytes<Context, Context>(
            l[i] * block_bytesize,
            d + block_bytesize * start,
            out + block_bytesize * max_l * i);
      } else {
        data.meta().copy()(
            d + block_bytesize * start,
            out + block_bytesize * max_l * i,
            l[i] * block_size);
      }
      start += l[i];
    }
    return true;
  }

  INPUT_TAGS(LENGTHS, DATA);
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

    T max_l = l[0];
    for (T i = 1; i < lengths.dim(0); ++i) {
      max_l = std::max(max_l, l[i]);
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
      if (data.meta().copy() == nullptr) {
        context_.template CopyBytes<Context, Context>(
            l[i] * block_bytesize,
            d + block_bytesize * data.dim(1) * i,
            out + block_bytesize * start);
      } else {
        data.meta().copy()(
            d + block_bytesize * data.dim(1) * i,
            out + block_bytesize * start,
            l[i] * block_size);
      }
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
    .SetDoc("Map N dim tensor to N+1 dim based on length blob")
    .Input(
        0,
        "lengths",
        "1-d int/long tensor contains the length in each of the output.")
    .Input(1, "tensor", "N dim Tensor.")
    .Output(
        0,
        "packed_tensor",
        "N + 1 dim Tesor"
        "where dim(0) is the max length");
OPERATOR_SCHEMA(UnpackSegments)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc("Map N dim tensor to N+1 dim based on length blob")
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
