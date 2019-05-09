#ifndef CAFFE2_OPERATORS_FEATURE_MAPS_OPS_H_
#define CAFFE2_OPERATORS_FEATURE_MAPS_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class MergeSingleScalarFeatureTensorsOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit MergeSingleScalarFeatureTensorsOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {
    numInputs_ = InputSize() / kNumTensorsPerInput;
    featureIDs_ = this->template GetRepeatedArgument<int64_t>("feature_ids");
  }
  virtual ~MergeSingleScalarFeatureTensorsOp() noexcept {}

  bool RunOnDevice() override {
    return DispatchHelper<
        TensorTypes<bool, int32_t, int64_t, float, double, std::string>>::
        call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    int numExamples = Input(0).numel();
    int totalNumFeatures = 0;
    for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
      const bool* inPresenceData =
          Input(kNumTensorsPerInput * inputIndex + 1).template data<bool>();
      for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
        if (inPresenceData[exampleIndex]) {
          ++totalNumFeatures;
        }
      }
    }

    auto* outLengths = Output(0, {numExamples}, at::dtype<int32_t>());
    auto* outKeys = Output(1, {totalNumFeatures}, at::dtype<int64_t>());
    auto* outValues = Output(2, {totalNumFeatures}, at::dtype<T>());

    int32_t* outLengthsData = outLengths->template mutable_data<int32_t>();
    int64_t* outKeysData = outKeys->template mutable_data<int64_t>();
    T* outValuesData = outValues->template mutable_data<T>();

    int keysOffset = 0;
    for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
      outLengthsData[exampleIndex] = 0;
      for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
        const T* inData =
            Input(kNumTensorsPerInput * inputIndex).template data<T>();
        const bool* inPresenceData =
            Input(kNumTensorsPerInput * inputIndex + 1).template data<bool>();
        if (inPresenceData[exampleIndex]) {
          ++outLengthsData[exampleIndex];
          outKeysData[keysOffset] = featureIDs_[inputIndex];
          outValuesData[keysOffset] = inData[exampleIndex];
          ++keysOffset;
        }
      }
    }
    return true;
  }

 private:
  const int kNumTensorsPerInput = 2;
  int numInputs_;
  std::vector<int64_t> featureIDs_;
};

template <class Context>
class MergeSingleScalarFeatureTensorsGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit MergeSingleScalarFeatureTensorsGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {
    numFeatureInputs_ = InputSize() - 1; // Everything other than values_grad
  }
  virtual ~MergeSingleScalarFeatureTensorsGradientOp() noexcept {}

  bool RunOnDevice() override {
    return DispatchHelper<
        TensorTypes<bool, int32_t, int64_t, float, double, std::string>>::
        call(this, Input(InputSize() - 1));
  }

  template <typename T>
  bool DoRunWithType() {
    int numExamples = Input(0).numel();
    for (int inputIndex = 0; inputIndex < numFeatureInputs_; ++inputIndex) {
      Output(inputIndex)->ResizeLike(Input(inputIndex));
    }

    const T* inValuesGradData = Input(InputSize() - 1).template data<T>();

    T default_value = T();
    int valuesOffset = 0;
    for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
      for (int inputIndex = 0; inputIndex < numFeatureInputs_; ++inputIndex) {
        const bool* inPresenceData = Input(inputIndex).template data<bool>();
        T* outFeatureData = Output(inputIndex)->template mutable_data<T>();
        if (inPresenceData[exampleIndex]) {
          outFeatureData[exampleIndex] = inValuesGradData[valuesOffset];
          ++valuesOffset;
        } else {
          outFeatureData[exampleIndex] = default_value;
        }
      }
    }
    return true;
  }

 private:
  int numFeatureInputs_;
};

template <class Context>
class MergeSingleListFeatureTensorsOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit MergeSingleListFeatureTensorsOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {
    numInputs_ = InputSize() / kNumTensorsPerInput;
    inValuesOffset_.resize(numInputs_);
    featureIDs_ = this->template GetRepeatedArgument<int64_t>("feature_ids");
  }
  virtual ~MergeSingleListFeatureTensorsOp() noexcept {}

  bool RunOnDevice() override {
    return DispatchHelper<
        TensorTypes<bool, int32_t, int64_t, float, double, std::string>>::
        call(this, Input(1));
  }

  template <typename T>
  bool DoRunWithType() {
    int numExamples = Input(0).numel();
    int totalNumFeatures = 0;
    int totalNumValues = 0;
    for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
      const int32_t* inLengthsData =
          Input(kNumTensorsPerInput * inputIndex).template data<int32_t>();
      const bool* inPresenceData =
          Input(kNumTensorsPerInput * inputIndex + 2).template data<bool>();
      for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
        if (inPresenceData[exampleIndex]) {
          ++totalNumFeatures;
          totalNumValues += inLengthsData[exampleIndex];
        }
      }
    }

    auto* outLengths = Output(0, {numExamples}, at::dtype<int32_t>());
    auto* outKeys = Output(1, {totalNumFeatures}, at::dtype<int64_t>());
    auto* outValuesLengths =
        Output(2, {totalNumFeatures}, at::dtype<int32_t>());
    auto* outValuesValues = Output(3, {totalNumValues}, at::dtype<T>());

    int32_t* outLengthsData = outLengths->template mutable_data<int32_t>();
    int64_t* outKeysData = outKeys->template mutable_data<int64_t>();
    int32_t* outValuesLengthsData =
        outValuesLengths->template mutable_data<int32_t>();
    T* outValuesValuesData = outValuesValues->template mutable_data<T>();

    int keysOffset = 0;
    int valuesOffset = 0;
    for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
      inValuesOffset_[inputIndex] = 0;
    }
    for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
      outLengthsData[exampleIndex] = 0;
      for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
        const int32_t* inLengthsData =
            Input(kNumTensorsPerInput * inputIndex).template data<int32_t>();
        const auto& inValues = Input(kNumTensorsPerInput * inputIndex + 1);
        const bool* inPresenceData =
            Input(kNumTensorsPerInput * inputIndex + 2).template data<bool>();
        if (inPresenceData[exampleIndex]) {
          ++outLengthsData[exampleIndex];
          outKeysData[keysOffset] = featureIDs_[inputIndex];
          outValuesLengthsData[keysOffset] = inLengthsData[exampleIndex];
          context_.CopyItemsSameDevice(
              inValues.dtype(),
              inLengthsData[exampleIndex],
              &inValues.template data<T>()[inValuesOffset_[inputIndex]],
              &outValuesValuesData[valuesOffset]);
          valuesOffset += inLengthsData[exampleIndex];
          inValuesOffset_[inputIndex] += inLengthsData[exampleIndex];
          ++keysOffset;
        }
      }
    }
    return true;
  }

 private:
  const int kNumTensorsPerInput = 3;
  int numInputs_;
  std::vector<int> inValuesOffset_;
  std::vector<int64_t> featureIDs_;
};

template <class Context>
class MergeSingleListOrMapFeatureTensorsGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit MergeSingleListOrMapFeatureTensorsGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {
    numFeatureInputs_ = (InputSize() - 1) / kNumTensorsPerInput;
  }
  virtual ~MergeSingleListOrMapFeatureTensorsGradientOp() noexcept {}

  bool RunOnDevice() override {
    return DispatchHelper<
        TensorTypes<bool, int32_t, int64_t, float, double, std::string>>::
        call(this, Input(InputSize() - 1));
  }

  template <typename T>
  bool DoRunWithType() {
    int numExamples = Input(0).numel();
    std::vector<int> outValuesOffset(numFeatureInputs_);
    for (int inputIndex = 0; inputIndex < numFeatureInputs_; ++inputIndex) {
      int inputNumValues = 0;
      const int32_t* inLengthsData =
          Input(kNumTensorsPerInput * inputIndex).template data<int32_t>();
      const bool* inPresenceData =
          Input(kNumTensorsPerInput * inputIndex + 1).template data<bool>();
      for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
        if (inPresenceData[exampleIndex]) {
          inputNumValues += inLengthsData[exampleIndex];
        }
      }
      Output(inputIndex)->Resize(inputNumValues);
    }

    const auto& inValuesValuesGrad = Input(InputSize() - 1);
    const T* inValuesValuesGradData = inValuesValuesGrad.template data<T>();

    int inValuesValuesOffset = 0;
    for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
      for (int inputIndex = 0; inputIndex < numFeatureInputs_; ++inputIndex) {
        const int32_t* inLengthsData =
            Input(kNumTensorsPerInput * inputIndex).template data<int32_t>();
        const bool* inPresenceData =
            Input(kNumTensorsPerInput * inputIndex + 1).template data<bool>();
        if (inPresenceData[exampleIndex]) {
          T* outFeatureValues = Output(inputIndex)->template mutable_data<T>();
          context_.CopyItemsSameDevice(
              inValuesValuesGrad.dtype(),
              inLengthsData[exampleIndex],
              &inValuesValuesGradData[inValuesValuesOffset],
              &outFeatureValues[outValuesOffset[inputIndex]]);
          outValuesOffset[inputIndex] += inLengthsData[exampleIndex];
          inValuesValuesOffset += inLengthsData[exampleIndex];
        }
      }
    }
    return true;
  }

 private:
  const int kNumTensorsPerInput = 2;
  int numFeatureInputs_;
};

template <class Context>
class MergeSingleMapFeatureTensorsOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit MergeSingleMapFeatureTensorsOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {
    numInputs_ = InputSize() / kNumTensorsPerInput;
    inValuesOffset_.resize(numInputs_);
    featureIDs_ = this->template GetRepeatedArgument<int64_t>("feature_ids");
  }
  virtual ~MergeSingleMapFeatureTensorsOp() noexcept {}

  bool RunOnDevice() override {
    return DispatchHelper<
        TensorTypes<bool, int32_t, int64_t, float, double, std::string>>::
        call(this, Input(1));
  }

  template <typename K>
  bool DoRunWithType() {
    return DispatchHelper<
        TensorTypes2<bool, int32_t, int64_t, float, double, std::string>,
        K>::call(this, Input(2));
  }

  template <typename K, typename V>
  bool DoRunWithType2() {
    int numExamples = Input(0).numel();
    int totalNumFeatures = 0;
    int totalNumValues = 0;
    for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
      const int32_t* inLengthsData =
          Input(kNumTensorsPerInput * inputIndex).template data<int32_t>();
      const bool* inPresenceData =
          Input(kNumTensorsPerInput * inputIndex + 3).template data<bool>();
      for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
        if (inPresenceData[exampleIndex]) {
          ++totalNumFeatures;
          totalNumValues += inLengthsData[exampleIndex];
        }
      }
    }

    auto* outLengths = Output(0, {numExamples}, at::dtype<int32_t>());
    auto* outKeys = Output(1, {totalNumFeatures}, at::dtype<int64_t>());
    auto* outValuesLengths =
        Output(2, {totalNumFeatures}, at::dtype<int32_t>());
    auto* outValuesKeys = Output(3, {totalNumValues}, at::dtype<K>());
    auto* outValuesValues = Output(4, {totalNumValues}, at::dtype<V>());

    int32_t* outLengthsData = outLengths->template mutable_data<int32_t>();
    int64_t* outKeysData = outKeys->template mutable_data<int64_t>();
    int32_t* outValuesLengthsData =
        outValuesLengths->template mutable_data<int32_t>();
    K* outValuesKeysData = outValuesKeys->template mutable_data<K>();
    V* outValuesValuesData = outValuesValues->template mutable_data<V>();

    int keysOffset = 0;
    int valuesOffset = 0;
    for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
      inValuesOffset_[inputIndex] = 0;
    }
    for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
      outLengthsData[exampleIndex] = 0;
      for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
        const int32_t* inLengthsData =
            Input(kNumTensorsPerInput * inputIndex).template data<int32_t>();
        const auto& inKeys = Input(kNumTensorsPerInput * inputIndex + 1);
        const auto& inValues = Input(kNumTensorsPerInput * inputIndex + 2);
        const bool* inPresenceData =
            Input(kNumTensorsPerInput * inputIndex + 3).template data<bool>();
        if (inPresenceData[exampleIndex]) {
          ++outLengthsData[exampleIndex];
          outKeysData[keysOffset] = featureIDs_[inputIndex];
          outValuesLengthsData[keysOffset] = inLengthsData[exampleIndex];
          context_.CopyItemsSameDevice(
              inKeys.dtype(),
              inLengthsData[exampleIndex],
              &inKeys.template data<K>()[inValuesOffset_[inputIndex]],
              &outValuesKeysData[valuesOffset]);
          context_.CopyItemsSameDevice(
              inValues.dtype(),
              inLengthsData[exampleIndex],
              &inValues.template data<V>()[inValuesOffset_[inputIndex]],
              &outValuesValuesData[valuesOffset]);
          valuesOffset += inLengthsData[exampleIndex];
          inValuesOffset_[inputIndex] += inLengthsData[exampleIndex];
          ++keysOffset;
        }
      }
    }
    return true;
  }

 private:
  const int kNumTensorsPerInput = 4;
  int numInputs_;
  std::vector<int> inValuesOffset_;
  std::vector<int64_t> featureIDs_;
};

template <class Context>
class MergeMultiScalarFeatureTensorsOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit MergeMultiScalarFeatureTensorsOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {
    numInputs_ = InputSize() / kNumTensorsPerInput;
    inKeysOffset_.resize(numInputs_);
  }
  virtual ~MergeMultiScalarFeatureTensorsOp() noexcept {}

  bool RunOnDevice() override {
    return DispatchHelper<
        TensorTypes<bool, int32_t, int64_t, float, double, std::string>>::
        call(this, Input(2));
  }

  template <typename T>
  bool DoRunWithType() {
    int numExamples = Input(0).numel();
    int totalNumFeatures = 0;
    for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
      totalNumFeatures += Input(kNumTensorsPerInput * inputIndex + 1).numel();
    }

    auto* outLengths = Output(0, {numExamples}, at::dtype<int32_t>());
    auto* outKeys = Output(1, {totalNumFeatures}, at::dtype<int64_t>());
    auto* outValues = Output(2, {totalNumFeatures}, at::dtype<T>());

    int32_t* outLengthsData = outLengths->template mutable_data<int32_t>();
    int64_t* outKeysData = outKeys->template mutable_data<int64_t>();
    T* outValuesData = outValues->template mutable_data<T>();

    int outKeysOffset = 0;
    for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
      inKeysOffset_[inputIndex] = 0;
    }
    for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
      outLengthsData[exampleIndex] = 0;
      for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
        const int32_t* inLengthsData =
            Input(kNumTensorsPerInput * inputIndex).template data<int32_t>();
        const int64_t* inKeysData = Input(kNumTensorsPerInput * inputIndex + 1)
                                        .template data<int64_t>();
        const T* inValuesData =
            Input(kNumTensorsPerInput * inputIndex + 2).template data<T>();
        outLengthsData[exampleIndex] += inLengthsData[exampleIndex];
        for (int featureIndex = 0; featureIndex < inLengthsData[exampleIndex];
             ++featureIndex) {
          outKeysData[outKeysOffset] = inKeysData[inKeysOffset_[inputIndex]];
          outValuesData[outKeysOffset] =
              inValuesData[inKeysOffset_[inputIndex]];
          ++outKeysOffset;
          ++inKeysOffset_[inputIndex];
        }
      }
    }

    return true;
  }

 private:
  const int kNumTensorsPerInput = 3;
  int numInputs_;
  std::vector<int> inKeysOffset_;
};

template <class Context>
class MergeMultiScalarFeatureTensorsGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit MergeMultiScalarFeatureTensorsGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {
    numFeatureInputs_ = (InputSize() - 1) / kNumTensorsPerInput;
  }
  virtual ~MergeMultiScalarFeatureTensorsGradientOp() noexcept {}

  bool RunOnDevice() override {
    return DispatchHelper<
        TensorTypes<bool, int32_t, int64_t, float, double, std::string>>::
        call(this, Input(InputSize() - 1));
  }

  template <typename T>
  bool DoRunWithType() {
    int numExamples = Input(0).numel();
    std::vector<int> outValuesOffset(numFeatureInputs_);
    for (int inputIndex = 0; inputIndex < numFeatureInputs_; ++inputIndex) {
      int inputNumValues = 0;
      const int32_t* inLengthsData =
          Input(kNumTensorsPerInput * inputIndex).template data<int32_t>();
      for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
        inputNumValues += inLengthsData[exampleIndex];
      }
      Output(inputIndex)->Resize(inputNumValues);
    }

    const auto& inValuesGrad = Input(InputSize() - 1);
    const T* inValuesGradData = inValuesGrad.template data<T>();

    int inValuesOffset = 0;
    for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
      for (int inputIndex = 0; inputIndex < numFeatureInputs_; ++inputIndex) {
        const int32_t* inLengthsData =
            Input(kNumTensorsPerInput * inputIndex).template data<int32_t>();
        if (inLengthsData[exampleIndex] > 0) {
          T* outFeatureValues = Output(inputIndex)->template mutable_data<T>();
          context_.CopyItemsSameDevice(
              inValuesGrad.dtype(),
              inLengthsData[exampleIndex],
              &inValuesGradData[inValuesOffset],
              &outFeatureValues[outValuesOffset[inputIndex]]);
          outValuesOffset[inputIndex] += inLengthsData[exampleIndex];
          inValuesOffset += inLengthsData[exampleIndex];
        }
      }
    }
    return true;
  }

 private:
  int kNumTensorsPerInput = 1;
  int numFeatureInputs_;
};

template <class Context>
class MergeMultiListFeatureTensorsOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit MergeMultiListFeatureTensorsOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {
    numInputs_ = InputSize() / kNumTensorsPerInput;
    inKeysOffset_.resize(numInputs_);
    inValuesValuesOffset_.resize(numInputs_);
  }
  virtual ~MergeMultiListFeatureTensorsOp() noexcept {}

  bool RunOnDevice() override {
    return DispatchHelper<
        TensorTypes<bool, int32_t, int64_t, float, double, std::string>>::
        call(this, Input(3));
  }

  template <typename T>
  bool DoRunWithType() {
    int numExamples = Input(0).numel();
    int totalNumFeatures = 0;
    int totalNumValues = 0;
    for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
      totalNumFeatures += Input(kNumTensorsPerInput * inputIndex + 1).numel();
      totalNumValues += Input(kNumTensorsPerInput * inputIndex + 3).numel();
    }

    auto* outLengths = Output(0, {numExamples}, at::dtype<int32_t>());
    auto* outKeys = Output(1, {totalNumFeatures}, at::dtype<int64_t>());
    auto* outValuesLengths =
        Output(2, {totalNumFeatures}, at::dtype<int32_t>());
    auto* outValuesValues = Output(3, {totalNumValues}, at::dtype<T>());

    int32_t* outLengthsData = outLengths->template mutable_data<int32_t>();
    int64_t* outKeysData = outKeys->template mutable_data<int64_t>();
    int32_t* outValuesLengthsData =
        outValuesLengths->template mutable_data<int32_t>();
    T* outValuesValuesData = outValuesValues->template mutable_data<T>();

    int outKeysOffset = 0;
    int outValuesValuesOffset = 0;
    for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
      inKeysOffset_[inputIndex] = 0;
      inValuesValuesOffset_[inputIndex] = 0;
    }
    for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
      outLengthsData[exampleIndex] = 0;
      for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
        const int32_t* inLengthsData =
            Input(kNumTensorsPerInput * inputIndex).template data<int32_t>();
        const int64_t* inKeysData = Input(kNumTensorsPerInput * inputIndex + 1)
                                        .template data<int64_t>();
        const int32_t* inValuesLengthsData =
            Input(kNumTensorsPerInput * inputIndex + 2)
                .template data<int32_t>();
        const auto& inValuesValues =
            Input(kNumTensorsPerInput * inputIndex + 3);
        outLengthsData[exampleIndex] += inLengthsData[exampleIndex];
        for (int featureIndex = 0; featureIndex < inLengthsData[exampleIndex];
             ++featureIndex) {
          outKeysData[outKeysOffset] = inKeysData[inKeysOffset_[inputIndex]];
          outValuesLengthsData[outKeysOffset] =
              inValuesLengthsData[inKeysOffset_[inputIndex]];
          context_.CopyItemsSameDevice(
              inValuesValues.dtype(),
              inValuesLengthsData[inKeysOffset_[inputIndex]],
              &inValuesValues
                   .template data<T>()[inValuesValuesOffset_[inputIndex]],
              &outValuesValuesData[outValuesValuesOffset]);
          outValuesValuesOffset +=
              inValuesLengthsData[inKeysOffset_[inputIndex]];
          inValuesValuesOffset_[inputIndex] +=
              inValuesLengthsData[inKeysOffset_[inputIndex]];
          ++outKeysOffset;
          ++inKeysOffset_[inputIndex];
        }
      }
    }

    return true;
  }

 private:
  const int kNumTensorsPerInput = 4;
  int numInputs_;
  std::vector<int> inKeysOffset_;
  std::vector<int> inValuesValuesOffset_;
};

template <class Context>
class MergeMultiMapFeatureTensorsOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit MergeMultiMapFeatureTensorsOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {
    numInputs_ = InputSize() / kNumTensorsPerInput;
    inKeysOffset_.resize(numInputs_);
    inValuesValuesOffset_.resize(numInputs_);
  }
  virtual ~MergeMultiMapFeatureTensorsOp() noexcept {}

  bool RunOnDevice() override {
    return DispatchHelper<
        TensorTypes<bool, int32_t, int64_t, float, double, std::string>>::
        call(this, Input(3));
  }

  template <typename K>
  bool DoRunWithType() {
    return DispatchHelper<
        TensorTypes2<bool, int32_t, int64_t, float, double, std::string>,
        K>::call(this, Input(4));
  }

  template <typename K, typename V>
  bool DoRunWithType2() {
    int numExamples = Input(0).numel();
    int totalNumFeatures = 0;
    int totalNumValues = 0;
    for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
      totalNumFeatures += Input(kNumTensorsPerInput * inputIndex + 1).numel();
      totalNumValues += Input(kNumTensorsPerInput * inputIndex + 4).numel();
    }

    auto* outLengths = Output(0, {numExamples}, at::dtype<int32_t>());
    auto* outKeys = Output(1, {totalNumFeatures}, at::dtype<int64_t>());
    auto* outValuesLengths =
        Output(2, {totalNumFeatures}, at::dtype<int32_t>());
    auto* outValuesKeys = Output(3, {totalNumValues}, at::dtype<K>());
    auto* outValuesValues = Output(4, {totalNumValues}, at::dtype<V>());

    int32_t* outLengthsData = outLengths->template mutable_data<int32_t>();
    int64_t* outKeysData = outKeys->template mutable_data<int64_t>();
    int32_t* outValuesLengthsData =
        outValuesLengths->template mutable_data<int32_t>();
    K* outValuesKeysData = outValuesKeys->template mutable_data<K>();
    V* outValuesValuesData = outValuesValues->template mutable_data<V>();

    int outKeysOffset = 0;
    int outValuesValuesOffset = 0;
    for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
      inKeysOffset_[inputIndex] = 0;
      inValuesValuesOffset_[inputIndex] = 0;
    }
    for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
      outLengthsData[exampleIndex] = 0;
      for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
        const int32_t* inLengthsData =
            Input(kNumTensorsPerInput * inputIndex).template data<int32_t>();
        const int64_t* inKeysData = Input(kNumTensorsPerInput * inputIndex + 1)
                                        .template data<int64_t>();
        const int32_t* inValuesLengthsData =
            Input(kNumTensorsPerInput * inputIndex + 2)
                .template data<int32_t>();
        const auto& inValuesKeys = Input(kNumTensorsPerInput * inputIndex + 3);
        const auto& inValuesValues =
            Input(kNumTensorsPerInput * inputIndex + 4);
        outLengthsData[exampleIndex] += inLengthsData[exampleIndex];
        for (int featureIndex = 0; featureIndex < inLengthsData[exampleIndex];
             ++featureIndex) {
          outKeysData[outKeysOffset] = inKeysData[inKeysOffset_[inputIndex]];
          outValuesLengthsData[outKeysOffset] =
              inValuesLengthsData[inKeysOffset_[inputIndex]];
          context_.CopyItemsSameDevice(
              inValuesKeys.dtype(),
              inValuesLengthsData[inKeysOffset_[inputIndex]],
              &inValuesKeys
                   .template data<K>()[inValuesValuesOffset_[inputIndex]],
              &outValuesKeysData[outValuesValuesOffset]);
          context_.CopyItemsSameDevice(
              inValuesValues.dtype(),
              inValuesLengthsData[inKeysOffset_[inputIndex]],
              &inValuesValues
                   .template data<V>()[inValuesValuesOffset_[inputIndex]],
              &outValuesValuesData[outValuesValuesOffset]);
          outValuesValuesOffset +=
              inValuesLengthsData[inKeysOffset_[inputIndex]];
          inValuesValuesOffset_[inputIndex] +=
              inValuesLengthsData[inKeysOffset_[inputIndex]];
          ++outKeysOffset;
          ++inKeysOffset_[inputIndex];
        }
      }
    }

    return true;
  }

 private:
  const int kNumTensorsPerInput = 5;
  int numInputs_;
  std::vector<int> inKeysOffset_;
  std::vector<int> inValuesValuesOffset_;
};

template <class Context>
class MergeMultiListOrMapFeatureTensorsGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit MergeMultiListOrMapFeatureTensorsGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {
    numFeatureInputs_ = (InputSize() - 1) / kNumTensorsPerInput;
  }
  virtual ~MergeMultiListOrMapFeatureTensorsGradientOp() noexcept {}

  bool RunOnDevice() override {
    return DispatchHelper<
        TensorTypes<bool, int32_t, int64_t, float, double, std::string>>::
        call(this, Input(InputSize() - 1));
  }

  template <typename T>
  bool DoRunWithType() {
    int numExamples = Input(0).numel();
    std::vector<int> outValuesLengthOffset(numFeatureInputs_);
    std::vector<int> outValuesValuesOffset(numFeatureInputs_);
    for (int inputIndex = 0; inputIndex < numFeatureInputs_; ++inputIndex) {
      int inputNumValues = 0;
      auto& inValuesLength = Input(kNumTensorsPerInput * inputIndex + 1);
      const int32_t* inValuesLengthsData =
          inValuesLength.template data<int32_t>();
      for (int valuesIndex = 0; valuesIndex < inValuesLength.numel();
           ++valuesIndex) {
        inputNumValues += inValuesLengthsData[valuesIndex];
      }
      Output(inputIndex)->Resize(inputNumValues);
    }

    const auto& inValuesValuesGrad = Input(InputSize() - 1);
    const T* inValuesValuesGradData = inValuesValuesGrad.template data<T>();

    int inValuesValuesOffset = 0;
    for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
      for (int inputIndex = 0; inputIndex < numFeatureInputs_; ++inputIndex) {
        const int32_t* inLengthsData =
            Input(kNumTensorsPerInput * inputIndex).template data<int32_t>();
        const int32_t* inValuesLengthsData =
            Input(kNumTensorsPerInput * inputIndex + 1)
                .template data<int32_t>();
        int valuesLengthCopy = 0;
        for (int valuesLengthIndex = 0;
             valuesLengthIndex < inLengthsData[exampleIndex];
             ++valuesLengthIndex) {
          valuesLengthCopy += inValuesLengthsData
              [outValuesLengthOffset[inputIndex] + valuesLengthIndex];
        }
        if (valuesLengthCopy > 0) {
          T* outFeatureValues = Output(inputIndex)->template mutable_data<T>();
          context_.CopyItemsSameDevice(
              inValuesValuesGrad.dtype(),
              valuesLengthCopy,
              &inValuesValuesGradData[inValuesValuesOffset],
              &outFeatureValues[outValuesValuesOffset[inputIndex]]);
        }
        outValuesLengthOffset[inputIndex] += inLengthsData[exampleIndex];
        outValuesValuesOffset[inputIndex] += valuesLengthCopy;
        inValuesValuesOffset += valuesLengthCopy;
      }
    }
    return true;
  }

 private:
  int kNumTensorsPerInput = 2;
  int numFeatureInputs_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_FEATURE_MAPS_OPS_H_
