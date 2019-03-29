#ifndef GATHER_OP_H_
#define GATHER_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

// This maintains index-mapping functions shared by Gather and BatchGather ops.
namespace gather_helper {

// New shape is concatenation:
//  [data dims before axis] + [indices dims] + [data dims after axis]
template <typename IndexType, typename DataDimsVec, typename IndexDimsVec>
static vector<IndexType> calc_output_shape_vector(
    const DataDimsVec& data_dims,
    const IndexDimsVec& indices_dims,
    int axis) {
  vector<IndexType> shape;
  // If the dimension we are indexing is empty, just use data_dims as shape.
  // This replicates behavior in (https://github.com/pytorch/pytorch/pull/13781)
  // needed to allow workflows with empty batch to succeed.
  if (data_dims[axis] == 0) {
    shape.insert(shape.end(), data_dims.begin(), data_dims.end());
  } else {
    shape.insert(shape.end(), data_dims.begin(), data_dims.begin() + axis);
    shape.insert(shape.end(), indices_dims.begin(), indices_dims.end());
    shape.insert(shape.end(), data_dims.begin() + axis + 1, data_dims.end());
  }
  return shape;
}

// Check that indices fall within dimension array size with CAFFE_ENFORCE.
template <typename IndexType>
static void check_indexarray_range(
    const IndexType* indices,
    int64_t n,
    IndexType indexing_axis_dim,
    bool wrap_indices) {
  //
  for (auto i = 0; i < n; ++i) {
    auto idx = indices[i];
    if (wrap_indices && idx < 0) {
      idx = idx + indexing_axis_dim;
    }
    CAFFE_ENFORCE(
        0 <= idx && idx < indexing_axis_dim,
        "INDICES element is out of DATA bounds, id=",
        idx,
        " axis_dim=",
        indexing_axis_dim);
  }
}

// Actual gather implementation - resizes output and copies indexed data.
template <typename Index, typename Context>
static bool gather_impl(
    Operator<Context>* op,
    int dataIdx,
    int indicesIdx,
    int outputIdx,
    int axis,
    bool wrap_indices) {
  // If we endup using it on GPU doing O(N) memcpy is probably not best :)
  // TODO: implement prefetching if it starts mattering (TF does it)

  const Tensor& data = op->Input(dataIdx);
  const Tensor& indices = op->Input(indicesIdx);
  const TypeMeta dataType = data.dtype();
  size_t item_bytesize = dataType.itemsize();

  // ONNX allows negative axis to index from the back, valid range: [-r, r].
  if (axis < 0) {
    axis = data.dim() + axis;
  }
  CAFFE_ENFORCE_GE(data.dim(), axis + 1, "DATA should be at least [axis+1]-D");
  CAFFE_ENFORCE_GE(axis, 0, "Axis should be non-negative");
  CAFFE_ENFORCE_LT(axis, data.dim(), "Axis out of range");

  // New shape:
  //  [data dims before axis] + [indices dims] + [data dims after axis]
  vector<int64_t> shape =
      calc_output_shape_vector<int64_t>(data.sizes(), indices.sizes(), axis);
  Tensor* output = op->Output(outputIdx, shape, at::dtype(dataType));
  auto out = static_cast<char*>(output->raw_mutable_data(dataType));

  // Succeed if size of output is zero, which can happen for empty batch which
  // would have data dimension size of 0.
  // This *must* be done AFTER output->raw_mutable_data() above as that has
  // important allocation side effect that we must see.
  if (output->numel() == 0) {
    return true;
  }

  const Index* idxs = indices.template data<Index>();
  auto src_base = static_cast<const char*>(data.raw_data());

  auto outer_dims_product = data.size_to_dim(axis);
  auto block_size = data.size_from_dim(axis + 1);
  auto block_bytesize = block_size * item_bytesize;

  auto src_indexing_axis_dim = data.size(axis);
  auto src_batch_bytesize = data.size_from_dim(axis) * item_bytesize;
  // Treat indices as a single block even if they have multiple dimensions.
  // The "gathered batch" is a cumulative result combining indexed blocks.
  auto N = indices.numel();
  auto gathered_batch_bytesize = N * block_size * item_bytesize;

  check_indexarray_range<Index>(idxs, N, src_indexing_axis_dim, wrap_indices);

  // Special-case single-float copy for efficiency
  if (data.template IsType<float>() && block_size == 1) {
    for (auto batch = 0; batch < outer_dims_product; ++batch) {
      const float* src_floats =
          (const float*)(src_base + batch * src_batch_bytesize);
      float* dst_floats = (float*)(out + batch * gathered_batch_bytesize);

      for (auto i = 0; i < N; ++i) {
        auto idx = idxs[i];
        if (wrap_indices && idx < 0) {
          idx = idx + src_indexing_axis_dim;
        }
        dst_floats[i] = src_floats[idx];
      }
    }
  } else {
    // outer_dims_product specifies how many times we repeat inner dimensions,
    // so we just iterate over it to cover all outer dimensions.
    for (auto batch = 0; batch < outer_dims_product; ++batch) {
      for (auto i = 0; i < N; ++i) {
        auto idx = idxs[i];
        if (wrap_indices && idx < 0) {
          idx = idx + src_indexing_axis_dim;
        }

        auto src = src_base + batch * src_batch_bytesize + idx * block_bytesize;
        auto dst = out + batch * gathered_batch_bytesize + i * block_bytesize;
        op->getContext()->CopyItemsSameDevice(dataType, block_size, src, dst);
      }
    }
  }
  return true;
}

} // namespace gather_helper

template <class Context>
class GatherOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit GatherOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(int, "axis", axis_, 0) {
    // TBD: We may want to fix the old index wrap behaviour once we have
    // operator versioning, to only apply it when needed as otherwise its likely
    // an error.
    // Right now, we apply index wrapping by default only to axis == 0,
    // since we have ONNX conversion code that uses it. For other ops it
    // needs to be speified explicitly with argument or you don't get it.
    if (OperatorBase::HasArgument("wrap_indices")) {
      wrap_indices_ = Operator<Context>::template GetSingleArgument<bool>(
          "wrap_indices", (false));
    } else {
      wrap_indices_ = (axis_ == 0) ? true : false;
    }
  }

  virtual ~GatherOp() noexcept {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, this->template Input<Tensor>(INDICES, CPU));
  }

  template <typename Index>
  bool DoRunWithType() {
    return gather_helper::gather_impl<Index, Context>(
        this, DATA, INDICES, 0, axis_, wrap_indices_);
  }

  INPUT_TAGS(DATA, INDICES);

 protected:
  int axis_;
  bool wrap_indices_;
};

} // namespace caffe2
#endif // GATHER_OP_H_
