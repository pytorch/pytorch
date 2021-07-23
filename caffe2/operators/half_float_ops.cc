#include "caffe2/operators/half_float_ops.h"
#include <c10/util/Half.h>
#include "caffe2/utils/cpuid.h"
#ifdef USE_FBGEMM
#include "fbgemm/FbgemmConvert.h"
#endif

namespace caffe2 {

inline void FloatToFloat16_ref(
    const float* in,
    at::Half* out,
    size_t N,
    bool do_clip = false) {
  if (do_clip) {
    constexpr float FP16_MAX = 65504.f;
    for (size_t i = 0; i < N; ++i) {
      out[i] = std::max(-FP16_MAX, std::min(in[i], FP16_MAX));
    }
  } else {
    for (size_t i = 0; i < N; ++i) {
      out[i] = in[i];
    }
  }
}

inline void Float16ToFloat_ref(const at::Half* in, float* out, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    out[i] = in[i];
  }
}

template <>
bool FloatToHalfOp<CPUContext>::RunOnDevice() {
  auto& input = Input(0);

  auto* output = Output(0, input.sizes(), at::dtype<at::Half>());
  const float* data = input.template data<float>();
  at::Half* out = output->template mutable_data<at::Half>();
  auto N = input.numel();

#ifdef USE_FBGEMM
  // There exists a verion fbgemm::FloatToFloat16_simd which will issue avx-512
  // instructions when possible. However, this actually doesn't give perf
  // benefits, according to benchmarks on T1/T6. Hence we stick to avx2 versions
  // here.
  if (GetCpuId().avx2()) {
    fbgemm::FloatToFloat16_avx2(
        data, reinterpret_cast<fbgemm::float16*>(out), N, clip_);
  } else {
    FloatToFloat16_ref(data, out, N, clip_);
  }
#else
  FloatToFloat16_ref(data, out, N, clip_);
#endif

  return true;
}

template <>
bool HalfToFloatOp<CPUContext>::RunOnDevice() {
  auto& input = Input(0);

  auto* output = Output(0, input.sizes(), at::dtype<float>());
  const at::Half* data = input.template data<at::Half>();
  float* out = output->template mutable_data<float>();
  auto N = input.numel();

#ifdef USE_FBGEMM
  // Same reasoning of sticking to avx2
  if (GetCpuId().avx2()) {
    fbgemm::Float16ToFloat_avx2(
        reinterpret_cast<const fbgemm::float16*>(data), out, N);
  } else {
    Float16ToFloat_ref(data, out, N);
  }
#else
  Float16ToFloat_ref(data, out, N);
#endif

  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(FloatToHalf, FloatToHalfOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(HalfToFloat, HalfToFloatOp<CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(FloatToHalf)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* unused */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      const TensorShape& X = in[0];
      out.push_back(X);
      out[0].set_data_type(TensorProto_DataType_FLOAT16);

      return out;
    });

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(HalfToFloat)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* unused */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      const TensorShape& X = in[0];
      out.push_back(X);
      out[0].set_data_type(TensorProto_DataType_FLOAT);

      return out;
    });

bool Float16ConstantFillOp::RunOnDevice() {
  auto* output = Output(0, shape_, at::dtype<at::Half>());
  const float givenValue =
      this->template GetSingleArgument<float>("value", 0.0f);
  at::Half givenFp16Value = givenValue;

  if (output->numel()) {
    at::Half* out = output->template mutable_data<at::Half>();
    std::fill(out, out + output->numel(), givenFp16Value);
  }
  return true;
}

template <>
bool Float16UniformFillOp<CPUContext>::RunOnDevice() {
  auto* output = Output(0, shape_, at::dtype<at::Half>());
  at::Half* out = output->template mutable_data<at::Half>();

  // Get a batch row by row and convert
  auto leading_dim_sz = output->size(0);
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  int rowsz = output->numel() / output->size(0);

  vector<float> intermediate_data_;
  intermediate_data_.resize(rowsz);
  // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
  for (uint64_t i = 0; i < leading_dim_sz; i++) {
    math::RandUniform<float, CPUContext>(
        rowsz, min_, max_, intermediate_data_.data(), &context_);
    // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
    for (uint64_t j = 0; j < rowsz; j++) {
      out[i * rowsz + j] = intermediate_data_[j];
    }
  }
  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Float16ConstantFill, Float16ConstantFillOp);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Float16UniformFill, Float16UniformFillOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Float16UniformFill)
    .NumInputs(0)
    .NumOutputs(1)
    .TensorInferenceFunction(Float16FillerTensorInference)
    .SetDoc(
        "Fills a half float tensor of a specified shape with"
        " values from a uniform distribution[min,max]")
    .Arg("shape", "Shape of the tensor")
    .Arg("min", "Minimim value to generate")
    .Arg("max", "Maximum value to generate");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
NO_GRADIENT(Float16UniformFill);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Float16ConstantFill)
    .NumInputs(0)
    .NumOutputs(1)
    .TensorInferenceFunction(Float16FillerTensorInference)
    .Arg("value", "The value for the elements of the output tensor.")
    .Arg("shape", "The shape of the output tensor.")
    .Output(
        0,
        "output",
        "Output tensor of constant values specified by 'value'");

class GetFloatToHalfGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "HalfToFloat", "", vector<string>{GO(0)}, vector<string>{GI(0)});
  }
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(FloatToHalf, GetFloatToHalfGradient);

class GetHalfToFloatGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "FloatToHalf", "", vector<string>{GO(0)}, vector<string>{GI(0)});
  }
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(HalfToFloat, GetHalfToFloatGradient);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
NO_GRADIENT(Float16ConstantFill);
} // namespace caffe2
