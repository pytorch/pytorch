#include "caffe2/operators/sequence_ops.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

vector<TensorShape> TensorInferenceForAddPadding(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  ArgumentHelper helper(def);
  const int padding_width = helper.GetSingleArgument<int>("padding_width", 1);
  const int end_padding_width = helper.GetSingleArgument<int>("end_padding_width", padding_width);
  CAFFE_ENFORCE_GT(in.size(), 0);
  CAFFE_ENFORCE_GE(in[0].dims_size(), 1);
  if (in.size() > 1) {
    CAFFE_ENFORCE_EQ(in[1].dims_size(), 1);
  }

  const auto num_paddings = (in.size() == 1 ? 1 : in[1].dims(0));
  vector<int> out_shape(in[0].dims().begin(), in[0].dims().end());
  out_shape[0] += (padding_width + end_padding_width) * num_paddings;

  if (def.output_size() == 1) {
    return vector<TensorShape>{CreateTensorShape(out_shape, in[0].data_type())};
  } else {
    return vector<TensorShape>{
      CreateTensorShape(out_shape, in[0].data_type()),
      CreateTensorShape(vector<int>(1, num_paddings), TensorProto::INT32)};
  }
}

template <>
template <typename T>
void GatherPaddingOp<CPUContext>::GatherPadding(
    const int outer_size,
    const int lengths_size,
    const int block_size,
    const int pad_width,
    const T* in_ptr,
    const int* lengths_ptr,
    T* padding_start_ptr,
    T* padding_end_ptr) {
  CAFFE_ENFORCE(
      (!std::is_same<bool, T>::value),
      "GatherPadding should not be executed on an input of type bool, as "
      "addition is not properly defined with booleans.");
  int64_t total_length = 0;
  for (int i = 0; i < lengths_size; ++i) {
    // check total length consistency
    const auto length = lengths_ptr[i];
    total_length += length;
    CAFFE_ENFORCE_LE(total_length, outer_size);
    // accumulate start paddings
    for (int j = 0; j < startPaddingWidth_; ++j) {
      for (int k = 0; k < block_size; ++k) {
        // Note: MSVC warns about unsafe use of type bool in operation.
        // This is now guarded by a CAFFE_ENFORCE so we can suppress it.
        #pragma warning(suppress: 4804)
        padding_start_ptr[k] += in_ptr[k];
      }
      in_ptr += block_size;
    }
    in_ptr += block_size * (length - pad_width);
    // accumulate end paddings
    for (int j = 0; j < endPaddingWidth_; ++j) {
      for (int k = 0; k < block_size; ++k) {
        #pragma warning(suppress: 4804)
        padding_end_ptr[k] += in_ptr[k];
      }
      in_ptr += block_size;
    }
  }
}

template <>
template <typename T>
bool RemovePaddingOp<CPUContext>::DoRunWithType() {
  const auto& in = Input(0);
  CAFFE_ENFORCE_GE(in.dim(), 1);
  const int32_t outer_size = in.sizes()[0];
  const auto block_size = std::accumulate(
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      in.sizes().begin() + 1, in.sizes().end(), 1, std::multiplies<int64_t>());
  const auto pad_width = startPaddingWidth_ + endPaddingWidth_;

  // if no lengths is provided, assume it is a single full-span entry
  const int32_t* lengths_ptr = &outer_size;
  int64_t lengths_size = 1;
  if (InputSize() > 1) {
    const auto& lengths = Input(1);
    lengths_ptr = lengths.data<int32_t>();
    lengths_size = lengths.numel();
  }

  auto out_dims = in.sizes().vec();
  out_dims[0] -= pad_width * lengths_size;
  auto* out = Output(0, std::move(out_dims), at::dtype<T>());

  const auto* in_ptr = in.template data<T>();
  auto* out_ptr = out->template mutable_data<T>();
  int64_t total_length = 0;
  for (int i = 0; i < lengths_size; ++i) {
    // check that total length is consistent
    const auto length = lengths_ptr[i];
    total_length += length;
    CAFFE_ENFORCE_LE(total_length, outer_size);
    std::copy(
        in_ptr + block_size * startPaddingWidth_,
        in_ptr + block_size * (length - endPaddingWidth_),
        out_ptr);
    in_ptr += block_size * length;
    out_ptr += block_size * (length - pad_width);
  }
  if (OutputSize() == 1) {
    return true;
  }

  auto* lengths_out = Output(1, {lengths_size}, at::dtype<int32_t>());
  std::transform(
      lengths_ptr,
      lengths_ptr + lengths_size,
      lengths_out->template mutable_data<int32_t>(),
      [pad_width](int32_t x) { return x - pad_width; });
  return true;
}

template <>
template <typename T>
bool AddPaddingOp<CPUContext>::MakePadding(
    const T* in_ptr,
    T* out_ptr,
    const int32_t* lengths_ptr,
    int32_t lengths_size,
    int32_t outer_size,
    const T* padding_start_ptr,
    const T* padding_end_ptr,
    int64_t block_size) {
  if (!lengths_ptr) {
    lengths_ptr = &outer_size;
  }

  int64_t total_length = 0;
  for (int i = 0; i < lengths_size; ++i) {
    // check that total length is consistent
    const auto length = lengths_ptr[i];
    total_length += length;
    CAFFE_ENFORCE_LE(total_length, outer_size);
    // copy padding before
    if (!padding_start_ptr) {
      memset(out_ptr, 0, block_size * startPaddingWidth_ * sizeof(T));
      out_ptr += block_size * startPaddingWidth_;
    } else {
      for (int j = 0; j < startPaddingWidth_; ++j) {
        std::copy(padding_start_ptr, padding_start_ptr + block_size, out_ptr);
        out_ptr += block_size;
      }
    }
    // copy payload
    const auto num_elems = block_size * length;
    std::copy(in_ptr, in_ptr + num_elems, out_ptr);
    in_ptr += num_elems;
    out_ptr += num_elems;
    // copy padding after
    if (!padding_end_ptr) {
      memset(out_ptr, 0, block_size * endPaddingWidth_ * sizeof(T));
      out_ptr += block_size * endPaddingWidth_;
    } else {
      for (int j = 0; j < endPaddingWidth_; ++j) {
        std::copy(padding_end_ptr, padding_end_ptr + block_size, out_ptr);
        out_ptr += block_size;
      }
    }
  }
  if (OutputSize() == 1) {
    return true;
  }

  auto* lengths_out = Output(1, {lengths_size}, at::dtype<int32_t>());
  const auto pad_width = startPaddingWidth_ + endPaddingWidth_;
  std::transform(
      lengths_ptr,
      lengths_ptr + lengths_size,
      lengths_out->template mutable_data<int32_t>(),
      [pad_width](int32_t x) { return x + pad_width; });
  return true;
}

template <>
bool PadEmptySamplesOp<CPUContext>::RunOnDevice() {
  auto& lengths = Input(0);
  auto* lengthsPtr = lengths.template data<int32_t>();
  CAFFE_ENFORCE(lengths.dim() == 1, "LENGTH should be 1-D");
  CAFFE_ENFORCE(InputSize() >= 1, "Input size must be no less than 1");

  int needPadding = 0;
  int sumLen = 0;
  for (int i = 0; i < lengths.numel(); ++i) {
    if (lengthsPtr[i] == 0) {
      needPadding++;
    }
    sumLen += lengthsPtr[i];
  }

  auto* out_lengths = Output(0, {lengths.numel()}, at::dtype<int32_t>());
  auto* outLengthsPtr = out_lengths->template mutable_data<int32_t>();
  for (int i = 0; i < lengths.numel(); ++i) {
    if (lengthsPtr[i] == 0) {
      outLengthsPtr[i] = 1;
    } else {
      outLengthsPtr[i] = lengthsPtr[i];
    }
  }

  for (int k = 0; k < InputSize() - 1; k++) {
    auto& features = Input(1 + k);
    CAFFE_ENFORCE(features.dim() >= 1, "FEATURE should at least 1-D");
    CAFFE_ENFORCE(
        features.size(0) == sumLen, "FEATURE and LENGTH should be consistent");
    const auto block_size = features.size_from_dim(1);

    auto* out_features = Output(1 + k);
    auto outDim = features.sizes().vec();
    outDim.at(0) += needPadding;
    out_features->Resize(outDim);
    auto dst =
        static_cast<char*>(out_features->raw_mutable_data(features.dtype()));
    auto src_base = static_cast<const char*>(features.raw_data());
    // copy data and add padding index as zero
    Tensor zero{CPU};
    zero.Resize(block_size);
    auto zeroPtr = static_cast<char*>(zero.raw_mutable_data(features.dtype()));
    // TODO Handle other composite types, such as vector<...>
    if (!features.dtype().Match<std::string>()) {
      memset(zeroPtr, 0, zero.nbytes());
    }
    int start_dest = 0;
    int start_src = 0;
    for (int i = 0; i < lengths.numel(); ++i) {
      if (lengthsPtr[i] == 0) {
        context_.CopyItemsSameDevice(
            features.dtype(),
            block_size,
            zeroPtr,
            dst + start_dest * features.dtype().itemsize());
        start_dest += block_size;
      } else {
        auto src = src_base + start_src * features.dtype().itemsize();
        context_.CopyItemsSameDevice(
            features.dtype(),
            lengthsPtr[i] * block_size,
            src,
            dst + start_dest * features.dtype().itemsize());
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        start_src += lengthsPtr[i] * block_size;
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        start_dest += lengthsPtr[i] * block_size;
      }
    }
  }
  return true;
}

REGISTER_CPU_OPERATOR(AddPadding, AddPaddingOp<CPUContext>);
REGISTER_CPU_OPERATOR(RemovePadding, RemovePaddingOp<CPUContext>);
REGISTER_CPU_OPERATOR(GatherPadding, GatherPaddingOp<CPUContext>);
REGISTER_CPU_OPERATOR(PadEmptySamples, PadEmptySamplesOp<CPUContext>);

struct GetAddPaddingGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    // whether to provide lengths as input to gradient
    vector<std::string> g_inputs{GO(0)};
    if (Def().input_size() > 1) {
      CAFFE_ENFORCE(Def().output_size() > 1);
      g_inputs.push_back(O(1));
    }

    vector<OperatorDef> ops;
    // gradient on the data
    ops.push_back(CreateOperatorDef(
        "RemovePadding", "", g_inputs, vector<string>{GI(0)}));
    // gradient on the start_padding (and end_padding)
    if (Def().input_size() >= 3) {
      std::vector<string> padding_grads{GI(2)};
      if (Def().input_size() == 4) {
        padding_grads.push_back(GI(3));
      }
      // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
      auto g_inputs2 = g_inputs;
      ops.push_back(
          CreateOperatorDef("GatherPadding", "", g_inputs2, padding_grads));
    }
    return ops;
  }
};
REGISTER_GRADIENT(AddPadding, GetAddPaddingGradient);

struct GetRemovePaddingGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    // whether to provide lengths as input to gradient
    vector<std::string> g_inputs{GO(0)};
    if (Def().input_size() > 1) {
      CAFFE_ENFORCE(Def().output_size() > 1);
      g_inputs.push_back(O(1));
    }

    return SingleGradientDef("AddPadding", "", g_inputs, vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(RemovePadding, GetRemovePaddingGradient);

OPERATOR_SCHEMA(AddPadding)
    .NumInputs(1, 4)
    .NumOutputs(1, 2)
    .TensorInferenceFunction(
        OpSchema::NeedsAllInputShapes(TensorInferenceForAddPadding))
    .SetDoc(R"DOC(
Given a partitioned tensor $T<N, D_1, ..., D_n>$, where the partitions are
defined as ranges on its outer-most (slowest varying) dimension $N$,
return a tensor $T<(N + 2 * padding\_width), D_1, ..., D_n>$ with paddings
added to the start and end of each range.

Optionally, different paddings can be provided for beginning and end.
Paddings provided must be a tensor $T<D_1, ..., D_n>$. If no padding is
provided, add zero padding. If no lengths vector is provided, add padding
only once, at the start and end of data.

Github Links:

- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/sequence_ops.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "AddPadding",
    ["X", "lengths"],
    ["Y", "lengths_out"],
    padding_width=1

)

workspace.FeedBlob("X", (np.random.rand(3,2,2).astype(np.float32)))
workspace.FeedBlob("lengths", np.array([3]).astype(np.int32))

print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))
print("lengths_out:", workspace.FetchBlob("lengths_out"))
```

**Result**

```
X: [[[0.2531572  0.4588472 ]
  [0.45140603 0.61161053]]

 [[0.92500854 0.8045306 ]
  [0.03356671 0.30233648]]

 [[0.4660227  0.6287745 ]
  [0.79372746 0.08609265]]]
Y: [[[0.         0.        ]
  [0.         0.        ]]

 [[0.2531572  0.4588472 ]
  [0.45140603 0.61161053]]

 [[0.92500854 0.8045306 ]
  [0.03356671 0.30233648]]

 [[0.4660227  0.6287745 ]
  [0.79372746 0.08609265]]

 [[0.         0.        ]
  [0.         0.        ]]]
lengths_out: [5]
```

</details>

)DOC")
    .Arg(
        "padding_width",
        "*(type: int)* Number of copies of padding to add around each range.")
    .Arg(
        "end_padding_width",
        "*(type: int)* [OPTIONAL] Specifies a different end-padding width. If "
        "this is not set, will use same as `padding_width`.")
    .Input(
        0,
        "data_in",
        "*(type: Tensor)* Input data ($T<N, D_1, ..., D_n>$).")
    .Input(
        1,
        "lengths",
        "*(type: Tensor`<int>`)* Number of elements in each range. "
        "sum(lengths) = N.")
    .Input(
        2,
        "start_padding",
        "*(type: Tensor`<int>`)* [OPTIONAL] Padding data for range start "
        "($T<D_1, ..., D_n>$).")
    .Input(
        3,
        "end_padding",
        "*(type: Tensor`<int>`)* [OPTIONAL] Padding for range end. If not "
        "provided, `start_padding` is used ($T<D_1, ..., D_n>$).")
    .Output(
        0,
        "data_out",
        "*(type: Tensor)* Padded data tensor ($T<N + 2*padding_width, "
        "D_1, ..., D_n>$).")
    .Output(
        1,
        "lengths_out",
        "*(type: Tensor`<int>`)* [OPTIONAL] Lengths for each padded range.");

OPERATOR_SCHEMA(RemovePadding)
    .NumInputs(1, 2)
    .NumOutputs(1, 2)
    .SetDoc(R"DOC(
Remove padding around the edges of each segment of the input data. This is the
reverse operation of **AddPadding**, and uses the same arguments and conventions
for input and output data format.

Github Links:

- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/sequence_ops.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

addpad_op = core.CreateOperator(
    "AddPadding",
    ["X", "lengths_add"],
    ["Y", "lengths_out_add"],
    padding_width=1
)

rmpad_op = core.CreateOperator(
    "RemovePadding",
    ["Y", "lengths_rm"],
    ["Z", "lengths_out_rm"],
    padding_width=1
)

workspace.FeedBlob("X", (np.random.randint(20, size=(3,5))))
workspace.FeedBlob("lengths_add", np.array([3]).astype(np.int32))
workspace.FeedBlob("lengths_rm", np.array([5]).astype(np.int32))

print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(addpad_op)
print("Y:", workspace.FetchBlob("Y"))
print("lengths_out_add:", workspace.FetchBlob("lengths_out_add"))

workspace.RunOperatorOnce(rmpad_op)
print("Z:", workspace.FetchBlob("Z"))
print("lengths_out_rm:", workspace.FetchBlob("lengths_out_rm"))
```

**Result**

```
X: [[17 19  1  9  1]
 [19  3  5 19  1]
 [16  0  0  0  4]]
Y: [[ 0  0  0  0  0]
 [17 19  1  9  1]
 [19  3  5 19  1]
 [16  0  0  0  4]
 [ 0  0  0  0  0]]
lengths_out_add: [5]
Z: [[17 19  1  9  1]
 [19  3  5 19  1]
 [16  0  0  0  4]]
lengths_out_rm: [3]
```

</details>

)DOC")
    .Arg(
        "padding_width",
        "*(type: int)* Outer-size of padding to remove around each range.")
    .Arg(
        "end_padding_width",
        "*(type: int)* [OPTIONAL] Specifies a different end-padding width. "
        "If this is not set, will use same as `padding_width`.")
    .Input(
        0,
        "data_in",
        "Input tensor ($T<N, D_1, ..., D_n>$).")
    .Input(
        1,
        "lengths",
        "*(type: Tensor`<int>`)* Number of elements in each range. "
        "sum(lengths) = N. If not provided, considers all data as a single "
        "segment.")
    .Output(
        0,
        "data_out",
        "*(type: Tensor)* Padded data tensor "
        "($T<N + 2*padding_width, D_1, ..., D_n>$).")
    .Output(
        1,
        "lengths_out",
        "*(type: Tensor`<int>`)* [OPTIONAL] Lengths for each padded range.");

OPERATOR_SCHEMA(GatherPadding)
    .NumInputs(2)
    .NumOutputs(1, 2)
    .SetDoc(R"DOC(
Gather the sum of start and end paddings in a padded input sequence. Used in
order to compute the gradients of AddPadding w.r.t the padding tensors.
)DOC")
    .Arg("padding_width", "Outer-size of padding present around each range.")
    .Arg(
        "end_padding_width",
        "(Optional) Specifies a different end-padding width.")
    .Input(0, "data_in", "T<N, D1..., Dn> Padded input data")
    .Input(
        1,
        "lengths",
        "(i64) Num of elements in each range. sum(lengths) = N. "
        "If not provided, considers all data as a single segment.")
    .Output(
        0,
        "padding_sum",
        "Sum of all start paddings, or of all "
        "paddings if end_padding_sum is not provided.")
    .Output(
        1,
        "end_padding_sum",
        "T<D1..., Dn> Sum of all end paddings, if provided.");

OPERATOR_SCHEMA(PadEmptySamples)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1, INT_MAX)
    .SetDoc(R"DOC(
Pad empty field given lengths and index features,

Input(0) is a blob pointing to the lengths of samples in one batch,
[Input(1),... Input(num_fields)] a list of tensors containing the data for
each field of the features.

PadEmptySamples is thread safe.
)DOC")
    .Input(0, "lengths", "A blob containing a pointer to the lengths.")
    .Output(
        0,
        "out_lengths",
        "Tensor containing lengths with empty sample padded.");

} // namespace caffe2
