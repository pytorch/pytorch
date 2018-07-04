#include "caffe2/mkl/mkl_utils.h"

#ifdef CAFFE2_HAS_MKL_DNN

namespace caffe2 {
namespace mkl {

template <typename T>
class MKLConcatOp final : public MKLOperator<T> {
 public:
  USE_MKLOPERATOR_FUNCTIONS(T);

  MKLConcatOp(const OperatorDef& operator_def, Workspace* ws)
      : MKLOperator<T>(operator_def, ws) {
    CAFFE_ENFORCE(
        !(OperatorBase::HasArgument("axis") &&
          OperatorBase::HasArgument("order")),
        "You shouldn't specify both the dim to concat, and the order "
        "in the case of 4-D images.");

    int add_axis;
    if (OperatorBase::HasArgument("axis")) {
      axis_ = OperatorBase::GetSingleArgument<int>("axis", -1);
      add_axis = OperatorBase::GetSingleArgument<int>("add_axis", 0);
    } else {
      const auto& order = StringToStorageOrder(
          OperatorBase::GetSingleArgument<string>("order", "NCHW"));
      OPERATOR_NEEDS_FEATURE(
          order == StorageOrder::NCHW, "Only NCHW order supported.");
      axis_ = 1;
      add_axis = 0;
    }

    OPERATOR_NEEDS_FEATURE(
        axis_ == 1 || axis_ == -3, "Only channel concatenation is supported.");
    OPERATOR_NEEDS_FEATURE(add_axis == 0, "Adding axis is not supported.");
  }

  bool RunOnDevice() override {
    const auto& X0 = Input(0);
    auto* Y = Output(0);
    int nInputs = InputSize();
    int nDims = X0.ndim();
    CAFFE_ENFORCE_EQ(nDims, 4, "Only NCHW inputs are supported");

    bool dims_changed = (input_size_cache_.size() != nInputs);
    for (int i = 0; i < nInputs && !dims_changed; ++i) {
      dims_changed = (input_size_cache_[i] != Input(i).dims());
    }

    if (dims_changed || FLAGS_caffe2_mkl_memonger_in_use) {
      input_size_cache_.resize(nInputs);
      int output_channels = 0;
      int canonical_axis = canonical_axis_index_(axis_, nDims);
      vector<dnnLayout_t> input_layouts(nInputs);
      for (int i = 0; i < nInputs; ++i) {
        const auto& Xi = Input(i);
        CAFFE_ENFORCE_EQ(Xi.ndim(), nDims, "Input ", i, " has wrong ndim.");
        for (int j = 0; j < nDims; ++j) {
          if (j == canonical_axis) {
            continue;
          }
          CAFFE_ENFORCE_EQ(
              Xi.dim32(j),
              X0.dim32(j),
              "Input ",
              i,
              " has dimension mismatch at axis ",
              j);
        }
        input_size_cache_[i] = Xi.dims();
        output_channels += Xi.dim32(canonical_axis);
        input_layouts[i] = Xi.layout();
      }
      cached_output_dims_ = X0.dims();
      cached_output_dims_[canonical_axis] = output_channels;

      primitive_.Reset(
          dnnConcatCreate<T>, nullptr, nInputs, input_layouts.data());
      Y->Reset(cached_output_dims_, primitive_, dnnResourceDst);
      buffer_.Reset(cached_output_dims_, primitive_, dnnResourceDst, true);
    }
    bool shared = buffer_.ShareFrom(*Y);

    for (int i = 0; i < nInputs; ++i) {
      resources_[dnnResourceMultipleSrc + i] = Input(i).buffer();
    }
    resources_[dnnResourceDst] = buffer_.buffer();
    ExecutePrimitive();
    buffer_.CopyTo(Y, primitive_, dnnResourceDst);
    if (FLAGS_caffe2_mkl_memonger_in_use && !shared) {
      buffer_.Reset();
    }
    return true;
  }

 private:
  int axis_;
  vector<TIndex> cached_output_dims_;
};

} // namespace mkl

REGISTER_MKL_OPERATOR(Concat, mkl::MKLConcatOp<float>);

} // namespace caffe2

#endif // CAFFE2_HAS_MKL_DNN
