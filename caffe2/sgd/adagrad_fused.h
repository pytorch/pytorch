#pragma once

#include "caffe2/sgd/adagrad_op.h"
#include "caffe2/sgd/math_lp.h"

namespace caffe2 {

namespace {

template <
    typename Tdata, // embedding and momentum types
    typename T, // everything else
    typename TLengths,
    typename adagradT,
    bool is_mean = false>
class SparseAdagradFusedWithSparseLengthsSumGradientOp final
    : public Operator<CPUContext> {
 public:
  SparseAdagradFusedWithSparseLengthsSumGradientOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5)),
        weight_decay_(
            this->template GetSingleArgument<float>("weight_decay", 0.f)) {
    VLOG(1) << "gradient optimization operator in use: "
            << "SparseAdagradFusedWithSparseLengthsSumGradientOp"
            << " weight_decay_=" << weight_decay_;
    const T decay = this->template GetSingleArgument<T>("decay", 1.0);
    CAFFE_ENFORCE_EQ(
        decay, 1.0, "Decay is not supported for SparseSimdAdagradOp");
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename SIndex>
  bool DoRunWithType() {
    const auto* lr = Input(LR).template data<T>();
    Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
    Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));

    auto& segmentGradsInput = Input(GRAD);
    auto& lengthsInput = Input(LENGTHS);

    CAFFE_ENFORCE_EQ(lengthsInput.dim(), 1, "LENGTHS must be a vector");
    auto numSegments = lengthsInput.size(0);
    CAFFE_ENFORCE_GT(segmentGradsInput.dim(), 0);
    CAFFE_ENFORCE_EQ(numSegments, segmentGradsInput.size(0));
    const auto* lengths = lengthsInput.template data<TLengths>();

    auto n = Input(INDICES).numel();

    const auto* indices = Input(INDICES).template data<SIndex>();
    const auto* gradIn = segmentGradsInput.template data<T>();
    const auto* paramIn = Input(PARAM).template data<Tdata>();
    const auto* momentIn = Input(MOMENT_1).template data<Tdata>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<Tdata>();
    auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<Tdata>();

    if (numSegments == 0) {
      return true;
    }
    auto block_size = segmentGradsInput.size_from_dim(1);

    // Enforce:
    // input(embedding/momentum) == outputs(embedding/momentum)
    CAFFE_ENFORCE_EQ(
        Input(PARAM).numel(),
        Input(MOMENT_1).numel(),
        "Input Param size: ",
        Input(PARAM).numel(),
        " Input Moment size: ",
        Input(MOMENT_1).numel());

    int dataIndex = 0;
    if (is_mean) {
      grad_buffer_.ResizeLike(Input(GRAD));
    }
    auto* grad_buffer_data =
        is_mean ? grad_buffer_.template mutable_data<T>() : NULL;
    if (is_mean) {
      for (const auto rangeIndex : c10::irange(numSegments)) {
        for (const auto tmpIndex : c10::irange(block_size)) {
          auto offsetI = rangeIndex * block_size;
          grad_buffer_data[offsetI + tmpIndex] = lengths[rangeIndex] > 0
              ? gradIn[offsetI + tmpIndex] / lengths[rangeIndex]
              : gradIn[offsetI + tmpIndex];
        }
      }
    }

    for (const auto rangeIndex : c10::irange(numSegments)) {
      for (auto start = dataIndex; dataIndex < start + lengths[rangeIndex];
           ++dataIndex) {
        std::size_t idx = indices[dataIndex];
        auto offsetI = rangeIndex * block_size;
        auto offsetIdx = idx * block_size;

        // Enforce:
        // access within range
        // gradient access within range
        CAFFE_ENFORCE_GE(
            Input(PARAM).numel(),
            block_size + offsetIdx,
            this->debug_def().input(PARAM),
            ", out of bound,  idx:",
            idx,
            " for input dataIndex:",
            dataIndex,
            " and block size:",
            block_size,
            " max size:",
            Input(PARAM).numel());

        if (block_size == 1) {
          float gi = std::fma(
              weight_decay_,
              paramIn[idx],
              is_mean ? grad_buffer_data[offsetI] : gradIn[offsetI]);
          float hi = momentOut[idx] = momentIn[idx] + gi * gi;
          paramOut[idx] =
              paramIn[idx] + lr[0] * gi / (std::sqrt(hi) + epsilon_);
        } else {
          // prefetching
          const int prefdist_T0 = 16;
          int i_pref = (dataIndex < n - prefdist_T0) ? dataIndex + prefdist_T0
                                                     : dataIndex;
          std::size_t idx_pref = indices[i_pref];
          kernel_(
              block_size,

              paramIn + offsetIdx,
              &paramIn[idx_pref * block_size],

              is_mean ? grad_buffer_data + offsetI : gradIn + offsetI,

              momentIn + offsetIdx,
              &momentIn[idx_pref * block_size],

              paramOut + offsetIdx,
              &paramOut[idx_pref * block_size],

              momentOut + offsetIdx,
              &momentOut[idx_pref * block_size],

              epsilon_,
              lr[0],
              weight_decay_);
        }
      }
    }
    CAFFE_ENFORCE_EQ(dataIndex, n);

    return true;
  }

 protected:
  T epsilon_;
  T weight_decay_;
  adagradT kernel_;
  Tensor grad_buffer_{CPU};

  INPUT_TAGS(PARAM, MOMENT_1, INDICES, GRAD, LR, LENGTHS);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1);
};

template <typename Tdata, typename T, typename TLengths, typename adagradT>
class SparseAdagradFusedWithSparseLengthsWeightedSumGradientOp final
    : public Operator<CPUContext> {
 public:
  SparseAdagradFusedWithSparseLengthsWeightedSumGradientOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5)),
        weight_decay_(
            this->template GetSingleArgument<float>("weight_decay", 0.f)) {
    VLOG(1) << "gradient optimization operator in use: "
            << "SparseAdagradFusedWithSparseLengthsWeightedSumGradientOp";
    const T decay = this->template GetSingleArgument<T>("decay", 1.0);
    CAFFE_ENFORCE_EQ(
        decay, 1.0, "Decay is not supported for SparseSimdAdagradOp");
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename SIndex>
  bool DoRunWithType() {
    const auto* lr = Input(LR).template data<T>();
    Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
    Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));

    auto& segmentGradsInput = Input(GRAD);
    auto& lengthsInput = Input(LENGTHS);

    CAFFE_ENFORCE_EQ(lengthsInput.dim(), 1, "LENGTHS must be a vector");
    auto numSegments = lengthsInput.size(0);
    CAFFE_ENFORCE_GT(segmentGradsInput.dim(), 0);
    CAFFE_ENFORCE_EQ(numSegments, segmentGradsInput.size(0));
    const auto* lengths = lengthsInput.template data<TLengths>();

    auto n = Input(INDICES).numel();

    const auto* indices = Input(INDICES).template data<SIndex>();
    const auto* gradIn = segmentGradsInput.template data<T>();
    const auto* paramIn = Input(PARAM).template data<Tdata>();
    const auto* momentIn = Input(MOMENT_1).template data<Tdata>();
    const auto* auxParamIn = Input(AUX_PARAM).template data<T>();

    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<Tdata>();
    auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<Tdata>();
    Output(AUX_GRAD)->Resize(n);
    auto* auxGrad = Output(AUX_GRAD)->template mutable_data<T>();

    if (numSegments == 0) {
      return true;
    }

    auto block_size = segmentGradsInput.size_from_dim(1);

    // Enforce:
    // input(embedding/momentum) == outputs(embedding/momentum)
    CAFFE_ENFORCE_EQ(
        Input(PARAM).numel(),
        Input(MOMENT_1).numel(),
        "Input Param size: ",
        Input(PARAM).numel(),
        " Input Moment size: ",
        Input(MOMENT_1).numel());

    // Cannot fuse this loop with the loop below because paramIn is updated
    // by the second loop. Specifically, there could be dataIndex1 != dataIndex2
    // s.t. indices[dataIndex1] == indices[dataIndex2], and fusing these two
    // loops would violate dependencies w.r.t.
    // paramIn[indices[dataIndex1]:block_size] The approximate version.
    // (RowWiseSparseSimdAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp)
    // ignores this dependency and fuses these two loops.
    std::vector<T> temp_grad(block_size);
    int dataIndex = 0;
    for (const auto rangeIndex : c10::irange(numSegments)) {
      for (auto start = dataIndex; dataIndex < start + lengths[rangeIndex];
           ++dataIndex) {
        std::size_t idx = indices[dataIndex];
        auto offsetI = rangeIndex * block_size;
        auto offsetIdx = idx * block_size;

        // Enforce:
        // access within range
        // gradient access within range
        CAFFE_ENFORCE_GE(
            Input(PARAM).numel(),
            block_size + offsetIdx,
            this->debug_def().input(PARAM),
            ", out of bound,  idx:",
            idx,
            " for input dataIndex:",
            dataIndex,
            " and block size:",
            block_size,
            " max size:",
            Input(PARAM).numel());

        internal::dot<T, Tdata, T>(
            block_size,
            gradIn + offsetI,
            paramIn + offsetIdx,
            auxGrad + dataIndex,
            &context_);
      }
    }
    CAFFE_ENFORCE_EQ(dataIndex, n);

    dataIndex = 0;
    for (const auto rangeIndex : c10::irange(numSegments)) {
      for (auto start = dataIndex; dataIndex < start + lengths[rangeIndex];
           ++dataIndex) {
        std::size_t idx = indices[dataIndex];
        auto offsetI = rangeIndex * block_size;
        auto offsetIdx = idx * block_size;
        auto localOffset = dataIndex - start;

        for (const auto i : c10::irange(block_size)) {
          temp_grad[i] = auxParamIn[localOffset] * gradIn[offsetI + i];
        }

        if (block_size == 1) {
          float gi = std::fma(weight_decay_, paramIn[idx], temp_grad[0]);
          float hi = momentOut[idx] = momentIn[idx] + gi * gi;
          paramOut[idx] =
              paramIn[idx] + lr[0] * gi / (std::sqrt(hi) + epsilon_);
        } else {
          // prefetching
          const int prefdist_T0 = 16;
          int i_pref = (dataIndex < n - prefdist_T0) ? dataIndex + prefdist_T0
                                                     : dataIndex;
          std::size_t idx_pref = indices[i_pref];
          kernel_(
              block_size,

              paramIn + offsetIdx,
              &paramIn[idx_pref * block_size],

              temp_grad.data(),

              momentIn + offsetIdx,
              &momentIn[idx_pref * block_size],

              paramOut + offsetIdx,
              &paramOut[idx_pref * block_size],

              momentOut + offsetIdx,
              &momentOut[idx_pref * block_size],

              epsilon_,
              lr[0],
              weight_decay_);
        }
      }
    }

    return true;
  }

 protected:
  T epsilon_;
  T weight_decay_;
  adagradT kernel_;

  INPUT_TAGS(PARAM, MOMENT_1, AUX_PARAM, INDICES, GRAD, LR, LENGTHS);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1, AUX_GRAD);
};

template <
    typename Tdata, // embedding and momentum types
    typename T, // everything else
    typename TLengths,
    typename adagradT>
class SparseAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp final
    : public Operator<CPUContext> {
 public:
  SparseAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5)),
        weight_decay_(
            this->template GetSingleArgument<float>("weight_decay", 0.f)) {
    VLOG(1) << "gradient optimization operator in use: "
            << "SparseAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp";
    const T decay = this->template GetSingleArgument<T>("decay", 1.0);
    CAFFE_ENFORCE_EQ(
        decay, 1.0, "Decay is not supported for SparseSimdAdagradOp");
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename SIndex>
  bool DoRunWithType() {
    const auto* lr = Input(LR).template data<T>();
    Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
    Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));

    auto& segmentGradsInput = Input(GRAD);
    auto& lengthsInput = Input(LENGTHS);

    CAFFE_ENFORCE_EQ(lengthsInput.dim(), 1, "LENGTHS must be a vector");
    auto numSegments = lengthsInput.size(0);
    CAFFE_ENFORCE_GT(segmentGradsInput.dim(), 0);
    CAFFE_ENFORCE_EQ(numSegments, segmentGradsInput.size(0));
    const auto* lengths = lengthsInput.template data<TLengths>();

    auto n = Input(INDICES).numel();

    const auto* indices = Input(INDICES).template data<SIndex>();
    const auto* gradIn = segmentGradsInput.template data<T>();
    const auto* paramIn = Input(PARAM).template data<Tdata>();
    const auto* momentIn = Input(MOMENT_1).template data<Tdata>();
    const auto* auxParamIn = Input(AUX_PARAM).template data<T>();

    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<Tdata>();
    auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<Tdata>();
    Output(AUX_GRAD)->Resize(n);
    auto* auxGrad = Output(AUX_GRAD)->template mutable_data<T>();

    if (numSegments == 0) {
      return true;
    }

    auto block_size = segmentGradsInput.size_from_dim(1);

    // Enforce:
    // input(embedding/momentum) == outputs(embedding/momentum)
    CAFFE_ENFORCE_EQ(
        Input(PARAM).numel(),
        Input(MOMENT_1).numel(),
        "Input Param size: ",
        Input(PARAM).numel(),
        " Input Moment size: ",
        Input(MOMENT_1).numel());

    std::vector<T> temp_grad(block_size);
    int dataIndex = 0;
    for (const auto rangeIndex : c10::irange(numSegments)) {
      for (auto start = dataIndex; dataIndex < start + lengths[rangeIndex];
           ++dataIndex) {
        std::size_t idx = indices[dataIndex];
        auto offsetI = rangeIndex * block_size;
        auto offsetIdx = idx * block_size;
        auto localOffset = dataIndex - start;

        // Enforce:
        // access within range
        // gradient access within range
        CAFFE_ENFORCE_GE(
            Input(PARAM).numel(),
            block_size + offsetIdx,
            this->debug_def().input(PARAM),
            ", out of bound,  idx:",
            idx,
            " for input dataIndex:",
            dataIndex,
            " and block size:",
            block_size,
            " max size:",
            Input(PARAM).numel());

        internal::dot<T, Tdata, T>(
            block_size,
            gradIn + offsetI,
            paramIn + offsetIdx,
            auxGrad + dataIndex,
            &context_);

        for (const auto i : c10::irange(block_size)) {
          temp_grad[i] = auxParamIn[localOffset] * gradIn[offsetI + i];
        }

        if (block_size == 1) {
          float gi = std::fma(weight_decay_, paramIn[idx], temp_grad[0]);
          float hi = momentOut[idx] = momentIn[idx] + gi * gi;
          paramOut[idx] =
              paramIn[idx] + lr[0] * gi / (std::sqrt(hi) + epsilon_);
        } else {
          // prefetching
          const int prefdist_T0 = 16;
          int i_pref = (dataIndex < n - prefdist_T0) ? dataIndex + prefdist_T0
                                                     : dataIndex;
          std::size_t idx_pref = indices[i_pref];
          kernel_(
              block_size,

              paramIn + offsetIdx,
              &paramIn[idx_pref * block_size],

              temp_grad.data(),

              momentIn + offsetIdx,
              &momentIn[idx_pref * block_size],

              paramOut + offsetIdx,
              &paramOut[idx_pref * block_size],

              momentOut + offsetIdx,
              &momentOut[idx_pref * block_size],

              epsilon_,
              lr[0],
              weight_decay_);
        }
      }
    }
    CAFFE_ENFORCE_EQ(dataIndex, n);

    return true;
  }

 protected:
  T epsilon_;
  T weight_decay_;
  adagradT kernel_;

  INPUT_TAGS(PARAM, MOMENT_1, AUX_PARAM, INDICES, GRAD, LR, LENGTHS);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1, AUX_GRAD);
};

} // namespace
} // namespace caffe2
