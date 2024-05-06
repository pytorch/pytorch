#pragma once

#include "caffe2/ideep/ideep_utils.h"
#include "caffe2/proto/caffe2_legacy.pb.h"

namespace {

class IDEEPConvTransposeUnpoolBase : public caffe2::IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPConvTransposeUnpoolBase(const caffe2::OperatorDef& operator_def, caffe2::Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        legacy_pad_(
            static_cast<caffe2::LegacyPadding>(OperatorBase::GetSingleArgument<int>(
                "legacy_pad",
                caffe2::LegacyPadding::NOTSET))),
        kernel_(OperatorBase::GetRepeatedArgument<int>("kernels")),
        stride_(OperatorBase::GetRepeatedArgument<int>("strides")),
        pads_(OperatorBase::GetRepeatedArgument<int>("pads")),
        adj_(OperatorBase::GetRepeatedArgument<int>("adjs")),
        shared_buffer_(
            OperatorBase::GetSingleArgument<int>("shared_buffer", 0)) {
    // For the padding, they should either be the legacy padding strategy
    // (VALID or SAME), or an explicit, non-negative value.
    if (legacy_pad_ == caffe2::LegacyPadding::VALID ||
        legacy_pad_ == caffe2::LegacyPadding::SAME) {
      CAFFE_ENFORCE(
          !OperatorBase::HasArgument("pads"),
          "If you use legacy padding VALID or SAME, you should not specify "
          "any specific padding values.");
    }
    // Get old arguments values.
    if (OperatorBase::HasArgument("kernel")) {
      kernel_.resize(2, OperatorBase::GetSingleArgument<int>("kernel", 0));
    } else if (
        OperatorBase::HasArgument("kernel_h") &&
        OperatorBase::HasArgument("kernel_w")) {
      kernel_.push_back(OperatorBase::GetSingleArgument<int>("kernel_h", 0));
      kernel_.push_back(OperatorBase::GetSingleArgument<int>("kernel_w", 0));
    }

    if (OperatorBase::HasArgument("stride")) {
      stride_.resize(2, OperatorBase::GetSingleArgument<int>("stride", 0));
    } else if (
        OperatorBase::HasArgument("stride_h") &&
        OperatorBase::HasArgument("stride_w")) {
      stride_.push_back(OperatorBase::GetSingleArgument<int>("stride_h", 0));
      stride_.push_back(OperatorBase::GetSingleArgument<int>("stride_w", 0));
    }

    if (OperatorBase::HasArgument("adj")) {
      adj_.resize(2, OperatorBase::GetSingleArgument<int>("adj", 0));
    } else if (
        OperatorBase::HasArgument("adj_h") &&
        OperatorBase::HasArgument("adj_w")) {
      adj_.push_back(OperatorBase::GetSingleArgument<int>("adj_h", 0));
      adj_.push_back(OperatorBase::GetSingleArgument<int>("adj_w", 0));
    }

    if (OperatorBase::HasArgument("pad")) {
      CAFFE_ENFORCE(
          legacy_pad_ != caffe2::LegacyPadding::VALID &&
              legacy_pad_ != caffe2::LegacyPadding::SAME,
          "If you use legacy padding VALID or SAME, you should not specify "
          "any specific padding values.");
      pads_.resize(4, OperatorBase::GetSingleArgument<int>("pad", 0));
    } else if (
        OperatorBase::HasArgument("pad_t") &&
        OperatorBase::HasArgument("pad_l") &&
        OperatorBase::HasArgument("pad_b") &&
        OperatorBase::HasArgument("pad_r")) {
      CAFFE_ENFORCE(
          legacy_pad_ != caffe2::LegacyPadding::VALID &&
              legacy_pad_ != caffe2::LegacyPadding::SAME,
          "If you use legacy padding VALID or SAME, you should not specify "
          "any specific padding values.");
      pads_.push_back(OperatorBase::GetSingleArgument<int>("pad_t", 0));
      pads_.push_back(OperatorBase::GetSingleArgument<int>("pad_l", 0));
      pads_.push_back(OperatorBase::GetSingleArgument<int>("pad_b", 0));
      pads_.push_back(OperatorBase::GetSingleArgument<int>("pad_r", 0));
    }

    // Fill default values.
    if (kernel_.empty()) {
      kernel_.assign({0, 0});
    }

    if (stride_.empty()) {
      stride_.assign(kernel_.size(), 1);
    }

    if (pads_.empty()) {
      pads_.assign(kernel_.size() * 2, 0);
    }

    if (adj_.empty()) {
      adj_.assign(kernel_.size(), 0);
    }

    CAFFE_ENFORCE_EQ(stride_.size(), kernel_.size());
    CAFFE_ENFORCE_EQ(adj_.size(), kernel_.size());

    if (legacy_pad_ != caffe2::LegacyPadding::VALID &&
        legacy_pad_ != caffe2::LegacyPadding::SAME) {
      CAFFE_ENFORCE_EQ(pads_.size(), 2 * kernel_.size());
    }

    for (const auto dim : c10::irange(kernel_.size())) {
      CAFFE_ENFORCE_GT(kernel_[dim], 0);
      CAFFE_ENFORCE_GT(stride_[dim], 0);
      CAFFE_ENFORCE_GE(adj_[dim], 0);
      CAFFE_ENFORCE_LE(adj_[dim], stride_[dim]);
    }
  }
  ~IDEEPConvTransposeUnpoolBase() override {}

  const ideep::tensor& Input(int index) {
    return OperatorBase::template Input<ideep::tensor>(index);
  }
  ideep::tensor* Output(int index) {
    return OperatorBase::template Output<ideep::tensor>(index);
  }

  ideep::tensor::dims pad_tl() const {
    return {pad_t(), pad_l()};
  }

  ideep::tensor::dims pad_br() const {
    return {pad_b(), pad_r()};
  }

  ideep::tensor::dims CalcOutputDims(
      const ideep::tensor& input,
      int output_channel) {
    CAFFE_ENFORCE_GT(input.get_size(), 0);

    int N = input.get_dim(0);
    ideep::tensor::dims output_dims;
    auto input_dims = input.get_dims();
    itensor::dims dims;
    dims.assign(input_dims.begin() + 2, input_dims.end());
    for (const auto dim : c10::irange(dims.size())) {
      int dim_size = 0;
      ComputeSizeAndPad(
          dims[dim],
          stride_[dim],
          kernel_[dim],
          adj_[dim],
          &pads_[dim],
          &pads_[dim + 2],
          &dim_size);
      output_dims.push_back(dim_size);
    }

    output_dims.insert(output_dims.begin(), {N, output_channel});
    return output_dims;
  }

  bool RunOnDevice() override {
    try {
      return RunOnDeviceWithOrderNCHW();
    } catch (ideep::error& e) {
      LOG(ERROR) << "IDEEP error:" << e.message;
      throw;
    }
  }

  virtual bool RunOnDeviceWithOrderNCHW() {
    CAFFE_THROW("Not implemented");
  }

 private:
  caffe2::LegacyPadding legacy_pad_;

 protected:
  std::vector<int> kernel_;
  std::vector<int> stride_;
  std::vector<int> pads_;
  std::vector<int> adj_;
  bool shared_buffer_;

  // Accessors for 2D conv params.

  inline int pad_t() const {
    return pads_[0];
  }

  inline int pad_l() const {
    return pads_[1];
  }

  inline int pad_b() const {
    return pads_[2];
  }

  inline int pad_r() const {
    return pads_[3];
  }

  inline int kernel_h() const {
    return kernel_[0];
  }

  inline int kernel_w() const {
    return kernel_[1];
  }

  inline int stride_h() const {
    return stride_[0];
  }

  inline int stride_w() const {
    return stride_[1];
  }

  inline int adj_h() const {
    return adj_[0];
  }

  inline int adj_w() const {
    return adj_[1];
  }

  inline void ComputeSizeAndPad(
      const int in_size,
      const int stride,
      const int kernel,
      const int adj,
      int* pad_head,
      int* pad_tail,
      int* out_size) {
    switch (legacy_pad_) {
      case caffe2::LegacyPadding::NOTSET:
        CAFFE_ENFORCE_GE(*pad_head, 0);
        CAFFE_ENFORCE_GE(*pad_tail, 0);
        *out_size =
            (in_size - 1) * stride + kernel + adj - *pad_head - *pad_tail;
        break;
      // We handle cases of LegacyPadding::VALID and LegacyPadding::SAME
      // the same way
      case caffe2::LegacyPadding::VALID:
      case caffe2::LegacyPadding::SAME:
        *pad_head = 0;
        *pad_tail = 0;
        *out_size = (in_size - 1) * stride + kernel + adj;
        break;
      case caffe2::LegacyPadding::CAFFE_LEGACY_POOLING:
        LOG(FATAL) << "CAFFE_LEGACY_POOLING is no longer supported.";
        break;
    }
  }
};

#define USE_IDEEP_CONV_TRANSPOSE_UNPOOL_BASE_FUNCTIONS()          \
  USE_OPERATOR_BASE_FUNCTIONS;                                    \
  /* using override */ using IDEEPConvTransposeUnpoolBase::Input; \
  /* using override */ using IDEEPConvTransposeUnpoolBase::Output;

} // namespace
