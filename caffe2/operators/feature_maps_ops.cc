#include "feature_maps_ops.h"

#include "caffe2/core/context.h"

namespace caffe2 {
namespace {

const std::string doc = R"DOC(
  Single-feature representation:
  - scalar features:
    <feature full name> T
  - list features:
    <feature full name>.lengths int32
    <feature full name>.values T
  - map features:
    <feature full name>.lengths int32
    <feature full name>.keys K
    <feature full name>.values V

  Missing values are set to zero, and value presence flag is set accordingly:
    <feature full name>.presence bool

  Multi-feature representation:
  - scalar features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values T
  - list features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.values T
  - map features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.keys K
    <feature type>.values.values V

  You can read more about representing batches of lists and maps here:
  https://our.intern.facebook.com/intern/dex/caffe2/sparse-operations/
)DOC";

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    MergeDenseFeatureTensors,
    MergeDenseFeatureTensorsOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MergeDenseFeatureTensors)
    .SetDoc(
        "Merge given multi-feature dense tensors  into one "
        "multi-feature tensor." +
        doc)
    .NumInputs(2)
    .NumOutputs(3)
    .Input(0, "in1", "")
    .Input(1, "in1_presence", ".presence")
    .Output(0, "out_lengths", ".lengths")
    .Output(1, "out_keys", ".keys")
    .Output(2, "out_values", ".values")
    .Arg("feature_ids", "feature ids");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    MergeSingleScalarFeatureTensors,
    MergeSingleScalarFeatureTensorsOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MergeSingleScalarFeatureTensors)
    .SetDoc(
        "Merge given single-feature tensors with scalar features into one "
        "multi-feature tensor." +
        doc)
    .NumInputs([](int n) { return n >= 2 && n % 2 == 0; })
    .NumOutputs(3)
    .Input(0, "in1", "")
    .Input(1, "in1_presence", ".presence")
    .Output(0, "out_lengths", ".lengths")
    .Output(1, "out_keys", ".keys")
    .Output(2, "out_values", ".values")
    .Arg("feature_ids", "feature ids");

class GetMergeSingleScalarFeatureTensorsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> input_blob_names{};
    vector<string> output_blob_names{};

    for (int inputIdx = 0; inputIdx < def_.input_size() / 2; ++inputIdx) {
      input_blob_names.push_back(I(inputIdx * 2 + 1));
      output_blob_names.push_back(GI(inputIdx * 2));
    }
    input_blob_names.push_back(GO(2));

    return SingleGradientDef(
        "MergeSingleScalarFeatureTensorsGradient",
        "", /* name */
        input_blob_names,
        output_blob_names);
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    MergeSingleScalarFeatureTensorsGradient,
    MergeSingleScalarFeatureTensorsGradientOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MergeSingleScalarFeatureTensorsGradient)
    .SetDoc(
        "Explode multi-feature tensor of scalar features into one or more"
        "single-feature tensors" +
        doc)
    .NumInputs([](int n) { return n >= 2; })
    .NumOutputs([](int n) { return n >= 1; })
    .Input(0, "in1_presence", ".presence")
    .Input(1, ".values_grad", ".values_grad")
    .Output(0, "in1_grad", "_grad of inputs");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(
    MergeSingleScalarFeatureTensors,
    GetMergeSingleScalarFeatureTensorsGradient);

// ##########################################################

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    MergeSingleListFeatureTensors,
    MergeSingleListFeatureTensorsOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MergeSingleListFeatureTensors)
    .SetDoc(
        "Merge given single-feature tensors with list features into one "
        "multi-feature tensor." +
        doc)
    .NumInputs([](int n) { return n >= 3 && n % 3 == 0; })
    .NumOutputs(4)
    .Input(0, "in1_lengths", ".lengths")
    .Input(1, "in1_values", ".values")
    .Input(2, "in1_presence", ".presence")
    .Output(0, "out_lengths", ".lengths")
    .Output(1, "out_keys", ".keys")
    .Output(2, "out_values_lengths", ".values.lengths")
    .Output(3, "out_values_values", ".values.values")
    .Arg("feature_ids", "feature ids");

class GetMergeSingleListFeatureTensorsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> input_blob_names{};
    vector<string> output_blob_names{};

    for (int inputIdx = 0; inputIdx < def_.input_size() / 3; ++inputIdx) {
      input_blob_names.push_back(I(inputIdx * 3));
      input_blob_names.push_back(I(inputIdx * 3 + 2));
      output_blob_names.push_back(GI(inputIdx * 3 + 1));
    }
    input_blob_names.push_back(GO(3));

    return SingleGradientDef(
        "MergeSingleListFeatureTensorsGradient",
        "",
        input_blob_names,
        output_blob_names);
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    MergeSingleListFeatureTensorsGradient,
    MergeSingleListOrMapFeatureTensorsGradientOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MergeSingleListFeatureTensorsGradient)
    .SetDoc(
        "Explode multi-feature tensors with list features into "
        "single-feature tensors." +
        doc)
    .NumInputs([](int n) { return n >= 3 && n % 2 == 1; })
    .NumOutputs([](int n) { return n >= 1; })
    .Input(0, "in1_lengths", ".lengths")
    .Input(1, "in1_presence", ".presence")
    .Input(2, "out_values_values", ".values.values_grad")
    .Output(0, "out1_values", ".values_grad");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(
    MergeSingleListFeatureTensors,
    GetMergeSingleListFeatureTensorsGradient);

// ##########################################################

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    MergeSingleMapFeatureTensors,
    MergeSingleMapFeatureTensorsOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MergeSingleMapFeatureTensors)
    .SetDoc(
        "Merge given single-feature tensors with map features into one "
        "multi-feature tensor." +
        doc)
    .NumInputs([](int n) { return n >= 4 && n % 4 == 0; })
    .NumOutputs(5)
    .Input(0, "in1_lengths", ".lengths")
    .Input(1, "in1_keys", ".keys")
    .Input(2, "in1_values", ".values")
    .Input(3, "in1_presence", ".presence")
    .Output(0, "out_lengths", ".lengths")
    .Output(1, "out_keys", ".keys")
    .Output(2, "out_values_lengths", ".values.lengths")
    .Output(3, "out_values_keys", ".values.keys")
    .Output(4, "out_values_values", ".values.values")
    .Arg("feature_ids", "feature ids");

class GetMergeSingleMapFeatureTensorsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> input_blob_names{};
    vector<string> output_blob_names{};

    for (int inputIdx = 0; inputIdx < def_.input_size() / 4; ++inputIdx) {
      input_blob_names.push_back(I(inputIdx * 4));
      input_blob_names.push_back(I(inputIdx * 4 + 3));
      output_blob_names.push_back(GI(inputIdx * 4 + 2));
    }
    input_blob_names.push_back(GO(4));

    return SingleGradientDef(
        "MergeSingleMapFeatureTensorsGradient",
        "",
        input_blob_names,
        output_blob_names);
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    MergeSingleMapFeatureTensorsGradient,
    MergeSingleListOrMapFeatureTensorsGradientOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MergeSingleMapFeatureTensorsGradient)
    .SetDoc(
        "Explode given multi-feature tensors with map features into "
        "multiple single-feature tensor." +
        doc)
    .NumInputs([](int n) { return n >= 3 && n % 2 == 1; })
    .NumOutputs([](int n) { return n >= 1; })
    .Input(0, "in1_lengths", ".lengths")
    .Input(1, "in1_presence", ".presence")
    .Input(2, "out_values_values_grad", ".values.values_grad")
    .Output(0, "in1_values_grad", ".values_grad");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(
    MergeSingleMapFeatureTensors,
    GetMergeSingleMapFeatureTensorsGradient);

// ##########################################################

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    MergeMultiScalarFeatureTensors,
    MergeMultiScalarFeatureTensorsOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MergeMultiScalarFeatureTensors)
    .SetDoc(
        "Merge given multi-feature tensors with scalar features into one." +
        doc)
    .NumInputs([](int n) { return n >= 3 && n % 3 == 0; })
    .NumOutputs(3)
    .Input(0, "in1_lengths", ".lengths")
    .Input(1, "in1_keys", ".keys")
    .Input(2, "in1_values", ".values")
    .Output(0, "out_lengths", ".lengths")
    .Output(1, "out_keys", ".keys")
    .Output(2, "out_values", ".values");

class GetMergeMultiScalarFeatureTensorsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> input_blob_names{};
    vector<string> output_blob_names{};

    for (int inputIdx = 0; inputIdx < def_.input_size() / kNumTensorsPerInput;
         ++inputIdx) {
      input_blob_names.push_back(I(inputIdx * kNumTensorsPerInput));
      output_blob_names.push_back(GI(inputIdx * kNumTensorsPerInput + 2));
    }
    input_blob_names.push_back(GO(2));

    return SingleGradientDef(
        "MergeMultiScalarFeatureTensorsGradient",
        "",
        input_blob_names,
        output_blob_names);
  }

 private:
  const int kNumTensorsPerInput = 3;
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    MergeMultiScalarFeatureTensorsGradient,
    MergeMultiScalarFeatureTensorsGradientOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MergeMultiScalarFeatureTensorsGradient)
    .SetDoc(
        "Explode given multi-feature tensors with scalar features into many." +
        doc)
    .NumInputs([](int n) { return n >= 2; })
    .NumOutputs([](int n) { return n >= 1; })
    .Input(0, "in1_lengths", ".lengths")
    .Input(1, "out_values_grad", ".values_grad")
    .Output(0, "in1_values_grad", ".values_grad");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(
    MergeMultiScalarFeatureTensors,
    GetMergeMultiScalarFeatureTensorsGradient);

// ##########################################################

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    MergeMultiListFeatureTensors,
    MergeMultiListFeatureTensorsOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    MergeMultiListFeatureTensorsGradient,
    MergeMultiListOrMapFeatureTensorsGradientOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MergeMultiListFeatureTensors)
    .SetDoc(
        "Merge given multi-feature tensors with list features into one." + doc)
    .NumInputs([](int n) { return n >= 4 && n % 4 == 0; })
    .NumOutputs(4)
    .Input(0, "in1_lengths", ".lengths")
    .Input(1, "in1_keys", ".keys")
    .Input(2, "in1_values_lengths", ".values.lengths")
    .Input(3, "in1_values_values", ".values.values")
    .Output(0, "out_lengths", ".lengths")
    .Output(1, "out_keys", ".keys")
    .Output(2, "out_values_lengths", ".values.lengths")
    .Output(3, "out_values_values", ".values.values");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MergeMultiListFeatureTensorsGradient)
    .SetDoc(
        "Explode given multi-feature tensors with list features "
        "into many." +
        doc)
    .NumInputs([](int n) { return n >= 3 && n % 2 == 1; })
    .NumOutputs([](int n) { return n >= 1; })
    .Input(0, "in1_lengths", ".lengths")
    .Input(1, "in1_values_lengths", ".values.lengths")
    .Input(2, "out_values_values_grad", ".values.values_grad")
    .Output(0, "in1_values_values_grad", ".values.values_grad");

class GetMergeMultiListFeatureTensorsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> input_blob_names{};
    vector<string> output_blob_names{};

    for (int inputIdx = 0; inputIdx < def_.input_size() / kNumTensorsPerInput;
         ++inputIdx) {
      input_blob_names.push_back(I(inputIdx * kNumTensorsPerInput));
      input_blob_names.push_back(I(inputIdx * kNumTensorsPerInput + 2));
      output_blob_names.push_back(GI(inputIdx * kNumTensorsPerInput + 3));
    }
    input_blob_names.push_back(GO(3));

    return SingleGradientDef(
        "MergeMultiListFeatureTensorsGradient",
        "",
        input_blob_names,
        output_blob_names);
  }

 private:
  const int kNumTensorsPerInput = 4;
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(
    MergeMultiListFeatureTensors,
    GetMergeMultiListFeatureTensorsGradient);

// ##########################################################

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    MergeMultiMapFeatureTensors,
    MergeMultiMapFeatureTensorsOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MergeMultiMapFeatureTensors)
    .SetDoc(
        "Merge given multi-feature tensors with map features into one." + doc)
    .NumInputs([](int n) { return n >= 5 && n % 5 == 0; })
    .NumOutputs(5)
    .Input(0, "in1_lengths", ".lengths")
    .Input(1, "in1_keys", ".keys")
    .Input(2, "in1_values_lengths", ".values.lengths")
    .Input(3, "in1_values_keys", ".values.keys")
    .Input(4, "in1_values_values", ".values.values")
    .Output(0, "out_lengths", ".lengths")
    .Output(1, "out_keys", ".keys")
    .Output(2, "out_values_lengths", ".values_lengths")
    .Output(3, "out_values_keys", ".values.keys")
    .Output(4, "out_values_values", ".values.values");

class GetMergeMultiMapFeatureTensorsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> input_blob_names{};
    vector<string> output_blob_names{};

    for (int inputIdx = 0; inputIdx < def_.input_size() / kNumTensorsPerInput;
         ++inputIdx) {
      input_blob_names.push_back(I(inputIdx * kNumTensorsPerInput));
      input_blob_names.push_back(I(inputIdx * kNumTensorsPerInput + 2));
      output_blob_names.push_back(GI(inputIdx * kNumTensorsPerInput + 4));
    }
    input_blob_names.push_back(GO(4));

    return SingleGradientDef(
        "MergeMultiMapFeatureTensorsGradient",
        "",
        input_blob_names,
        output_blob_names);
  }

 private:
  const int kNumTensorsPerInput = 5;
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    MergeMultiMapFeatureTensorsGradient,
    MergeMultiListOrMapFeatureTensorsGradientOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MergeMultiMapFeatureTensorsGradient)
    .SetDoc(
        "Explode given multi-feature tensors with map features "
        "into many." +
        doc)
    .NumInputs([](int n) { return n >= 3 && n % 2 == 1; })
    .NumOutputs([](int n) { return n >= 1; })
    .Input(0, "in1_lengths", ".lengths")
    .Input(1, "in1_values_lengths", ".values.lengths")
    .Input(2, "out_values_values_grad", ".values.values_grad")
    .Output(0, "in1_values_values_grad", ".values.values_grad");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(
    MergeMultiMapFeatureTensors,
    GetMergeMultiMapFeatureTensorsGradient);

} // namespace
} // namespace caffe2
