#include "caffe2/operators/transpose_op.h"
#ifdef CAFFE2_USE_HPTT

#include <hptt.h>

namespace caffe2 {

namespace {
template <typename T>
bool tryRunWithHPTT(const std::vector<int>& axes, const TensorCPU& input, TensorCPU* output) {
  if (!std::is_same<T, float>::value) {
    return false;
  }
  std::vector<int> axes_cm(axes.size());
  std::vector<int> idims_cm(axes.size());

  // Convert row-major index to column major.
  auto cm = [&](int i) { return axes.size() - i - 1; };

  for (auto i = 0; i < axes.size(); ++i) {
    axes_cm[i] = cm(axes[cm(i)]);
    idims_cm[i] = input.dim32(cm(i));
  }

  auto plan = hptt::create_plan(axes_cm.data(),
                                axes.size(),
                                1.0,
                                input.template data<float>(),
                                idims_cm.data(),
                                nullptr,
                                0.0,
                                output->template mutable_data<float>(),
                                nullptr,
                                hptt::ESTIMATE,
                                1);
  if (!plan) {
    return false;
  }
  plan->execute();
  return true;
}
} // namespace

} // namespace caffe2
#endif

namespace caffe2 {



#define COMPILE_TIME_MAX_TRANSPOSE_DIMS 10

template <>
template <typename T>
bool TransposeOp<CPUContext>::DoRunWithType() {
  const auto& input = Input(0);
  auto* output = Output(0);

#ifdef CAFFE2_USE_HPTT
  if (tryRunWithHPTT<T>(axes_, input, output)) {
    return true;
  }
#endif

  int from_inds[COMPILE_TIME_MAX_TRANSPOSE_DIMS] = {0};
  size_t count = input.size();
  int num_axes = axes_.size();
  const T* from_data = input.template data<T>();
  T* to_data = output->template mutable_data<T>();
  auto in_dims = input.dims();
  auto out_dims = output->dims();

  // Measure amount of contiguous data we can copy at once
  TIndex blocksize = 1;
  int n_shared_idxs = 0;
  for (int i = num_axes - 1; i >= 0; --i) {
    if (axes_[i] == i) {
      blocksize *= new_dims_[i];
      ++n_shared_idxs;
    } else {
      break;
    }
  }

  if (num_axes < 2 || n_shared_idxs == num_axes) {
    memcpy(to_data, from_data, count * sizeof(T));
    return true;
  }

  int itr_axes = num_axes - n_shared_idxs;

  // Calculate strides
  TIndex stride_x[COMPILE_TIME_MAX_TRANSPOSE_DIMS] = {0};
  for (size_t i = 0; i < itr_axes; i++) {
    stride_x[i] = 1;
    for (size_t j = axes_[i] + 1; j < itr_axes; j++) {
      stride_x[i] *= in_dims[j];
    }
  }

  TIndex itr_idxs[COMPILE_TIME_MAX_TRANSPOSE_DIMS] = {0};

  // Branch here to avoid branching within the loop
  if (blocksize > 1) {
    for (size_t index = 0; index < (count / blocksize); index++) {
      TIndex from_index = 0;
      for (int i = 0; i < itr_axes; ++i) {
        from_index += stride_x[i] * itr_idxs[i];
      }

      memcpy(
          to_data + blocksize * index,
          from_data + blocksize * from_index,
          blocksize * sizeof(T));

      ++itr_idxs[itr_axes - 1];
      for (int i = itr_axes - 1; i >= 1; --i) {
        auto expected_dim = out_dims[i];
        if (itr_idxs[i] < expected_dim) {
          break;
        }
        itr_idxs[i] %= expected_dim;
        ++itr_idxs[i - 1];
      }
    }
  } else {
    for (size_t index = 0; index < count; index++) {
      TIndex from_index = 0;
      for (int i = 0; i < itr_axes; ++i) {
        from_index += stride_x[i] * itr_idxs[i];
      }

      *(to_data + index) = *(from_data + from_index);

      ++itr_idxs[itr_axes - 1];
      for (int i = itr_axes - 1; i >= 1; --i) {
        auto expected_dim = out_dims[i];
        if (itr_idxs[i] < expected_dim) {
          break;
        }
        itr_idxs[i] %= expected_dim;
        ++itr_idxs[i - 1];
      }
    }
  }

  return true;
}

REGISTER_CPU_OPERATOR(Transpose, TransposeOp<CPUContext>);

OPERATOR_SCHEMA(Transpose)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](
        const OperatorDef& def,
        const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      vector<int> axes = helper.GetRepeatedArgument<int>("axes");
      vector<TensorShape> out(1);
      out[0].set_data_type(in[0].data_type());

      if (axes.empty()) {
        for (auto axis = in [0].dims().rbegin(); axis != in[0].dims().rend();
             ++axis) {
          out[0].add_dims(*axis);
        }
      } else {
        auto tensor_size = in[0].dims().size();
        auto valid_axes =
            std::all_of(axes.begin(), axes.end(), [&tensor_size](int& axis) {
              return axis >= 0 && axis < tensor_size;
            });

        CAFFE_ENFORCE(valid_axes, "Axes argument passed in had invalid values");
        CAFFE_ENFORCE(
            axes.size() == tensor_size,
            "Axes argument passed in had the incorrect size");

        for (auto axis = axes.begin(); axis != axes.end(); ++axis) {
          out[0].add_dims(in[0].dims().Get(*axis));
        }
      }

      return out;
    })
    .SetDoc(R"DOC(
Transpose the input tensor similar to numpy.transpose. For example, when
axes=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
will be (2, 1, 3).
)DOC")
    .Arg(
        "axes",
        "A list of integers. By default, reverse the dimensions, "
        "otherwise permute the axes according to the values given.")
    .Input(0, "data", "An input tensor.")
    .Output(0, "transposed", "Transposed output.");

class GetTransposeGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  // We will create our own arguments.
  bool CopyArguments() const override {
    return false;
  }
  vector<OperatorDef> GetGradientDefs() override {
    auto ops = SingleGradientDef(
        "Transpose", "", vector<string>{GO(0)}, vector<string>{GI(0)});
    ops[0].mutable_arg()->CopyFrom(Def().arg());
    if (ArgumentHelper::HasArgument(Def(), "axes")) {
      // If axes is specified, we will need to figure out the inverse index.
      const Argument& old_axes = GetArgument(Def(), "axes");
      const int axes_size = old_axes.ints_size();
      Argument* new_arg = GetMutableArgument("axes", false, &ops[0]);
      for (int i = 0; i < axes_size; ++i) {
        new_arg->set_ints(old_axes.ints(i), i);
      }
    }
    return ops;
  }
};
REGISTER_GRADIENT(Transpose, GetTransposeGradient);
} // namespace caffe2
