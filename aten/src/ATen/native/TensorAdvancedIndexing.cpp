// Indexing tensors by by tensors
//
// This corresponds to "advanced indexing" in NumPy. The two operations are:
//
//  index(Tensor self, indices) -> Tensor
//  index_put_(Tensor self, indices, value, accumulate=false)
//
// The index is a TensorList containg kLong, kBool or kByte tensors or nulls. Byte
// tensors (boolean masks) are expanded to long tensors via nonzero(). Null
// tensors signify that the dimension is not indexed.
//
// All indexes are broadcast together and iterated as *one*. From NumPy:
//
// result[i_1, ..., i_M] == x[ind_1[i_1, ..., i_M], ind_2[i_1, ..., i_M],
//                           ..., ind_N[i_1, ..., i_M]]
//
// Note 1: ByteTensors expand to index as many dimensions as there are in the
// mask.
//
// Note 2: The behavior is more complicated when the index tensors are not all
// adjacent (e.g. x[[0, 1], :, [2, 3]]). In this case, self and the index
// tensors are transposed to the front: x.transpose(1, 2)[[0, 1], [2, 3]]
//
// The code contains two implementations of indexing. The more efficient
// implementation treats indexing like an elementwise operation over the
// tensors `result`, `x`, `ind_1`, `ind_2`, etc. This implementation does
// not work for index_put_ with accumulate=True. The other implementation
// combines the indexed tensors into a single linear index that is used
// with Tensor.put_. This is used for index_put_ with accumulate=True.
//
// The more efficient implementation takes the following steps for the
// above operation:
//
// 1) Broadcast ind_1, ind_2, ind_3 together to a common shape
// 2) Record x.stride(i) for each indexed dimension `i`
// 3) Replace the indexed subspace of `x` with the shape of the corresponding
//    subspace of `result` but with stride 0
// 4) Add dimensions of size 1 to the index tensors (ind_1, ind_2, etc.) so
//    that their shape is compatible with the result shape
//
// The CPU or CUDA kernel then computes element-wise over the broadcasted
// and restrided result, x, ind_1,  ind_2, etc.:
//
//   result[...] = *(&x[...] +
//                   ind_1[...] * x.stride(1) +
//                   ind_2[...] * x.stride(2) +
//                   ...)
//
// where & and * represent the C-style address-of and indirection operations.

#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/IndexKernel.h>
#include <ATen/native/IndexingUtils.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/core/IList.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Copy.h>
#include <ATen/native/Resize.h>
#include <ATen/native/ScatterGatherChecks.h>
#include <ATen/Parallel.h>

#include <c10/util/irange.h>
#include <c10/util/Unroll.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

namespace at {
namespace native {

static std::string shapes_as_str(TensorList tensors) {
  std::ostringstream os;
  bool first = true;
  for (auto& tensor : tensors) {
    if (tensor.defined()) {
      if (!first) {
        os << ", ";
      }
      os << tensor.sizes();
      first = false;
    }
  }
  return os.str();
}

static AdvancedIndex make_info(Tensor self, IOptTensorRefList orig) {
  checkIndexTensorTypes(orig);
  // first expand BoolTensor (masks) or ByteTensor (masks) into 1 or more LongTensors
  auto indices = expandTensors(self, orig);
  // next broadcast all index tensors together
  try {
    indices = expand_outplace(indices);
  } catch (std::exception& e) {
    TORCH_CHECK_INDEX(false, "shape mismatch: indexing tensors could not be broadcast together"
                      " with shapes ", shapes_as_str(indices));
  }
  // add missing null Tensors so that it matches self.dim()
  while (indices.size() < (size_t)self.dim()) {
    indices.emplace_back();
  }
  // if the non-null indices are not all adjacent, transpose self and indices
  // together so that they're adjacent at the front
  if (!hasContiguousSubspace(indices)) {
    std::tie(self, indices) = transposeToFront(self, indices);
  }
  // Ensure indices are on the same device as self
  for (auto & indice : indices) {
    if (indice.defined() && indice.device() != self.device()) {
      indice = indice.to(self.device());
    }
  }
  return AdvancedIndex(self, indices);
}

} // namespace native

namespace meta {

native::SCATTER_GATHER_OP get_operator_enum(const c10::string_view reduce) {
  if (reduce == "add") {
    return native::SCATTER_GATHER_OP::REDUCE_ADD;
  } else if (reduce == "multiply") {
    return native::SCATTER_GATHER_OP::REDUCE_MULTIPLY;
  } else {
    TORCH_CHECK(false, "reduce argument must be either add or multiply.");
  }
}

TORCH_META_FUNC(gather)
(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  const Tensor& result = maybe_get_output(0);
  int64_t wrapped_dim = at::maybe_wrap_dim(dim, self.dim());

  // Memory overlap checks need to be done after resizing (if required) is done.
  // But it only makes sense to do these checks when result was defined, hence
  // the boolean variable `check_result` here.
  // For more details, see: https://github.com/pytorch/pytorch/pull/63312#discussion_r694794832
  // and https://github.com/pytorch/pytorch/issues/63837
  bool check_result = result.defined();
  set_output(index.sizes(), self.options());
  if (check_result) {
    at::assert_no_internal_overlap(result);
    at::assert_no_overlap(result, self);
    at::assert_no_partial_overlap(result, index);
  }

  auto is_index_empty = index.numel() == 0;
  if (!is_index_empty) {
    TORCH_CHECK(
      index.scalar_type() == at::ScalarType::Long,
      "gather", "(): Expected dtype int64 for index"
    );
  }
  if (is_index_empty) return;
  at::native::gather_shape_check(self, wrapped_dim, index);
}

template <typename Meta>
void scatter_meta_impl(
    Meta& meta,
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const c10::optional<Tensor>& src = nullopt,
    const c10::optional<c10::string_view> reduce = nullopt) {
  int64_t wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
  at::native::scatter_gather_dtype_check("scatter", self, index, src);
  at::native::scatter_shape_check(self, wrapped_dim, index, src);
  auto output = meta.maybe_get_output(0);

  if (output.defined()) {
    at::assert_no_internal_overlap(output);
    at::assert_no_overlap(output, index);
    if (src.has_value()) {
      at::assert_no_overlap(output, src.value());
    }
  }

  meta.set_output(self.sizes(), self.options());
  if (reduce.has_value()) {
    // Check if we have a valid reduce operator.
    get_operator_enum(reduce.value());
  }
}

TORCH_META_FUNC2(scatter, src)
(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  scatter_meta_impl(*this, self, dim, index, src);
}

TORCH_META_FUNC2(scatter, value)
(const Tensor& self, int64_t dim, const Tensor& index, const Scalar& value) {
  scatter_meta_impl(*this, self, dim, index);
}

TORCH_META_FUNC2(scatter, reduce)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& src,
 const c10::string_view reduce) {
  scatter_meta_impl(*this, self, dim, index, src, reduce);
}

TORCH_META_FUNC2(scatter, value_reduce)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Scalar& src,
 const c10::string_view reduce) {
  scatter_meta_impl(*this, self, dim, index, nullopt, reduce);
}

TORCH_META_FUNC(scatter_add)
(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  scatter_meta_impl(*this, self, dim, index, src, "add");
}

TORCH_PRECOMPUTE_META_FUNC(index_add)
(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source, const Scalar& alpha) {
  dim = maybe_wrap_dim(dim, self.dim());
  auto numel = index.numel();

  TORCH_CHECK_INDEX(index.dim() <= 1, "index_add_(): Index is supposed to be a vector, but got dim: ",
                    index.dim(), " with type: ", index.scalar_type(), " and size: ", index.sizes());
  TORCH_CHECK(index.scalar_type() == ScalarType::Long || index.scalar_type() == ScalarType::Int,
              "index_add_(): Expected dtype int32/int64 for index but got: ", index.scalar_type());
  TORCH_CHECK(self.scalar_type() == source.scalar_type(),
              "index_add_(): self (", self.scalar_type(), ") and source (", source.scalar_type(),
              ") must have the same scalar type");
  TORCH_CHECK(dim == 0 || dim < source.dim(),
              "index_add_(): Indexing dim ", dim, " is out of bounds of the source tensor with dim ",
              source.dim());
  TORCH_CHECK(numel == (source.dim() == 0 ? 1 : source.size(dim)),
              "index_add_(): Number of indices (", numel, ") should be equal to source.size(dim): (",
              source.size(dim), "), for dim: ", dim);

  auto& result = maybe_get_output(0);
  bool is_defined = result.defined();
  set_output(self.sizes(), self.options());
  if (is_defined) {
    at::assert_no_internal_overlap(result);
    at::assert_no_overlap(result, index);
    at::assert_no_overlap(result, source);
  }

  // A hack to run TensorIterator checks in the meta function.
  // See comment: https://github.com/pytorch/pytorch/pull/65993#discussion_r760307417
  // TODO: (@krshrimali) Try inheriting from TensorIteratorBase instead.
  if (result.device() == kMeta) {
    auto selfSlice = result.select(dim, 0);
    auto sourceSlice = source.select(dim, 0);
    auto iter = TensorIterator::borrowing_binary_op(selfSlice, selfSlice, sourceSlice);
  }

  return TORCH_PRECOMPUTE_STRUCT(index_add)().set_dim(dim);
}

static void build_index_op(
    TensorIteratorBase& iter,
    const at::native::AdvancedIndex& info,
    const Tensor& result) {
  // 'TensorIterator' needs to own the things comming from 'info', since
  // 'info' will be destroyed after the META function.
  TensorIteratorConfig config;
  // info.src is a restrided view of result
  config.set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .add_output(result)
      .add_owned_input(info.src);
  for (auto& index : info.indices) {
    config.add_owned_input(index);
  }
  if (!result.defined()) {
    config.declare_static_dtype_and_device(info.src.scalar_type(), info.src.device());
  }
  iter.build(config);
}

void check_indices_on_cpu_or_selfdevice(
    const Tensor& self,
    at::IOptTensorRefList indices) {
  auto dev = self.device();
  bool indices_on_cpu_or_dev = std::all_of(
      indices.begin(), indices.end(), [=](const at::OptionalTensorRef& opt) {
        return opt.has_value() ? (opt->is_cpu() || opt->device() == dev) : true;
      });
  TORCH_CHECK(
      indices_on_cpu_or_dev,
      "indices should be either on ", kCPU,
      " or on the same device as the indexed tensor (", dev, ")");
}

TORCH_PRECOMPUTE_META_FUNC2(index, Tensor)
(const Tensor& self, const at::IOptTensorRefList& indices) {
  TORCH_CHECK_INDEX(
      indices.size() <= (size_t)self.dim(),
      "too many indices for tensor of dimension ",
      self.dim(), " (got ", indices.size(), ")");

  // Only allow: `dev_tensor[{cpu,dev}_tensor]`.
  // See: https://github.com/pytorch/pytorch/pull/69607
  check_indices_on_cpu_or_selfdevice(self, indices);

  const auto& result = maybe_get_output();

  if (result.defined()) {
    at::assert_no_internal_overlap(result);
    at::assert_no_overlap(result, self);
    for (const auto& index: indices) {
      if (index.has_value()) {
        at::assert_no_overlap(result, *index);
      }
    }
  }

  auto info = at::native::make_info(self, indices);
  build_index_op(*this, info, result);
  return TORCH_PRECOMPUTE_STRUCT2(index, Tensor)()
      .set_sizes(std::move(info.indexed_sizes))
      .set_strides(std::move(info.indexed_strides));
}

} // namespace meta

namespace native {

DEFINE_DISPATCH(index_stub);
DEFINE_DISPATCH(index_fill_stub);
DEFINE_DISPATCH(index_copy_stub);
DEFINE_DISPATCH(index_put_stub);
DEFINE_DISPATCH(index_put_with_sort_stub);
DEFINE_DISPATCH(put_stub);
DEFINE_DISPATCH(take_stub);
DEFINE_DISPATCH(masked_fill_stub);
REGISTER_NO_CPU_DISPATCH(index_put_with_sort_stub);
DEFINE_DISPATCH(masked_select_serial_stub);
DEFINE_DISPATCH(masked_select_stub);
DEFINE_DISPATCH(masked_scatter_stub);

DEFINE_DISPATCH(gather_stub);
DEFINE_DISPATCH(scatter_stub);
DEFINE_DISPATCH(scatter_fill_stub);
DEFINE_DISPATCH(scatter_add_stub);
DEFINE_DISPATCH(scatter_reduce_stub);
DEFINE_DISPATCH(scatter_scalar_reduce_stub);

static bool all_strides_match(TensorList tensors) {
  TORCH_CHECK(tensors.size() >= 1);
  auto strides = tensors[0].strides();
  for (auto& tensor : tensors.slice(1)) {
    if (!strides.equals(tensor.strides())) {
      return false;
    }
  }
  return true;
}

// Replace indexed dimensions in src with stride 0 and the size of the result tensor.
// The offset in these dimensions is computed by the kernel using the index tensor's
// values and the stride of src. The new shape is not meaningful. It's used to make
// the shape compatible with the result tensor.
static Tensor restride_src(const Tensor& src, int64_t dims_before, int64_t dims_indexed,
                           IntArrayRef replacement_shape) {
  auto shape = DimVector(src.sizes());
  auto strides = DimVector(src.strides());
  int64_t end = dims_before + dims_indexed;
  shape.erase(shape.begin() + dims_before, shape.begin() + end);
  strides.erase(strides.begin() + dims_before, strides.begin() + end);
  shape.insert(shape.begin() + dims_before, replacement_shape.begin(), replacement_shape.end());
  strides.insert(strides.begin() + dims_before, replacement_shape.size(), 0);
  return src.as_strided(shape, strides);
}

// Add dimensions of size 1 to an index tensor so that it can be broadcast to the result
// shape and iterated over element-wise like the result tensor and the restrided src.
static Tensor reshape_indexer(const Tensor& index, int64_t dims_before, int64_t dims_after) {
  auto orig_shape = index.sizes();
  auto shape = DimVector();
  shape.append(dims_before, 1);
  shape.append(orig_shape.begin(), orig_shape.end());
  shape.append(dims_after, 1);
  return index.reshape(shape);
}

AdvancedIndex::AdvancedIndex(const Tensor& src, TensorList indices_list)
{
  int64_t element_size_bytes = src.element_size();
  int64_t dims_before = 0, dims_after = 0, dims_indexed = 0;
  IntArrayRef replacement_shape;
  for (const auto dim : c10::irange(indices_list.size())) {
    if (!indices_list[dim].defined()) {
      if (dims_indexed == 0) {
        dims_before++;
      } else {
        dims_after++;
      }
    } else {
      dims_indexed++;
      replacement_shape = indices_list[dim].sizes();
      indexed_sizes.push_back(src.size(dim));
      indexed_strides.push_back(src.stride(dim) * element_size_bytes);
    }
  }

  // Check if the indexed subspace contains a dim of size 0, but the replacement
  // shape does not. This implies that an index is out of bounds, because there
  // is no number that's a valid index for an empty tensor. Normally, out of
  // bounds is handled in the indexing kernel, but this case fails earlier in
  // restride_src with an unhelpful error message.
  if (std::find(indexed_sizes.begin(), indexed_sizes.end(), 0) != indexed_sizes.end() &&
      std::find(replacement_shape.begin(), replacement_shape.end(), 0) == replacement_shape.end()) {
    TORCH_CHECK_INDEX(false, "index is out of bounds for dimension with size 0");
  }

  this->dims_before = dims_before;
  this->dims_after = dims_after;
  this->src = restride_src(src, dims_before, dims_indexed, replacement_shape);

  for (auto& index : indices_list) {
    if (index.defined()) {
      indices.push_back(reshape_indexer(index, dims_before, dims_after));
    }
  }

  // For CUDA tensors, force all index tensors to have the same striding to
  // simplify the CUDA kernel.
  if (indices.size() >= 2 && this->src.device().type() == kCUDA) {
    if (!all_strides_match(indices)) {
      for (auto & indice : indices) {
        indice = indice.contiguous();
      }
    }
  }
}

static std::tuple<bool, Tensor> canDispatchToMaskedFill(const Tensor& self, const torch::List<c10::optional<at::Tensor>>& indices,
const Tensor& value){
  if (!(value.numel() ==1 && value.device().is_cpu())){
    return std::make_tuple(false,Tensor());
  }
  int64_t num_ind = 0;
  Tensor mask;
  auto self_device = self.device();
  for (const c10::optional<Tensor> i: indices) {
    if (!i.has_value() || !(*i).defined()){
      num_ind++;
    } else {
      Tensor index = std::move(*i);
      if ((index.scalar_type() != kByte && index.scalar_type() != kBool) ||
          index.device() != self_device || mask.defined()){
        return std::make_tuple(false, Tensor());
      } else {
        mask = index;
        for (const auto j : c10::irange(index.dim())) {
          int64_t srcIdx = num_ind + j;
          TORCH_CHECK_INDEX(index.size(j) == self.size(srcIdx), "The shape of the mask ", index.sizes(), " at index ", j,
  " does not match the shape of the indexed tensor ", self.sizes(), " at index ", srcIdx);
        }
        num_ind += mask.ndimension();
      }
    }
  }
  for (const auto i : c10::irange(num_ind, self.ndimension())) {
    (void)i; //Suppress unused variable warning
    mask = mask.unsqueeze(-1);
  }
  return std::make_tuple(true, mask);
}



static TensorIterator make_index_put_iterator(const AdvancedIndex& info, const Tensor& value) {
  TORCH_CHECK(is_expandable_to(value.sizes(), info.src.sizes()), "shape mismatch: value tensor of shape ", value.sizes(),
             " cannot be broadcast to indexing result of shape ", info.src.sizes());
  TORCH_CHECK(value.scalar_type() == info.src.scalar_type(),
              "Index put requires the source and destination dtypes match, "
              "got ", info.src.scalar_type(), " for the destination "
              "and ", value.scalar_type(), " for the source.");
  TensorIteratorConfig config;
  // info.src is restrided by restride_src with 0 strided dimensions
  config.set_check_mem_overlap(false);
  config.resize_outputs(false);
  config.check_all_same_dtype(false);
  config.add_output(info.src);
  config.add_input(value);
  for (auto& index : info.indices) {
    config.add_input(index);
  }
  return config.build();
}

TORCH_IMPL_FUNC(index_out)
(const Tensor& self,
 DimVector sizes,
 DimVector strides,
 const Tensor& result) {
  index_stub(device_type(), *this, sizes, strides);
}

Tensor quantized_index(const Tensor & self, const at::IOptTensorRefList& indices) {
  TORCH_INTERNAL_ASSERT(
      self.qscheme() == c10::kPerTensorAffine ||
      self.qscheme() == c10::kPerTensorSymmetric,
      "Indexing is only supported for per-Tensor quantized Tensors.");

  // For now, this is a naive implementation which does dq -> index -> q.
  // TODO(future PR): improve performance by removing the copies.
  const auto& self_dq = self.dequantize();
  auto result = at::index(self_dq, indices);
  return at::quantize_per_tensor(
      result, self.q_scale(), self.q_zero_point(), self.scalar_type());
}

Tensor & put_(Tensor & self, const Tensor& index, const Tensor & source, const bool accumulate) {
  // See note [Writing Nondeterministic Operations]
  // Nondeterministic when index contains duplicate entries and we do not accumulate
  // If we accumulate on GPU, we use atomicGPUAdd, which is non-deterministic
  if (!accumulate || (accumulate && self.device().type() == DeviceType::CUDA)) {
    at::globalContext().alertNotDeterministic("put_");
  }

  // Type and device checks
  TORCH_CHECK(index.scalar_type() == ScalarType::Long, "put_(): Expected a long tensor for index, but got ", index.scalar_type())
  TORCH_CHECK(self.scalar_type() == source.scalar_type(), "put_(): self and source expected to have the same dtype, but got self.dtype = ", self.scalar_type(), " and source.dtype = ", source.scalar_type());
  TORCH_CHECK(self.device() == source.device() && self.device() == index.device(),
      "put_(): self, index and source expected to be in the same device, but got self.device = ",
      self.device(), ", index.device = ", index.device(), ", and source.device = ", source.device());

  // index checks
  TORCH_CHECK_INDEX(source.numel() == index.numel(), "put_(): Expected source and index to have the same number of elements, but got source.numel() = ", source.numel(), ", index.numel() = ", index.numel());
  TORCH_CHECK_INDEX(!(self.numel() == 0 && index.numel() != 0), "put_(): Tried to put elements into an empty tensor");

  at::assert_no_internal_overlap(self);
  at::assert_no_overlap(self, index);
  at::assert_no_overlap(self, source);

  // Early return
  if (index.numel() == 0) {
    return self;
  }

  auto index_reshaped = index.reshape(source.sizes());
  // Do not iterate over self, we will compute the offsets manually
  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)
    .check_all_same_dtype(false)
    .add_input(source)
    .add_input(index_reshaped)
    .build();

  put_stub(iter.device_type(), iter, self, accumulate);

  return self;
}

Tensor put(const Tensor & self, const Tensor& index, const Tensor & source, const bool accumulate) {
  return self.clone(at::MemoryFormat::Preserve).put_(index, source, accumulate);
}

Tensor index_put(const Tensor & self, const torch::List<c10::optional<Tensor>>& indices, const Tensor & value, bool accumulate) {
  return self.clone(at::MemoryFormat::Preserve).index_put_(indices, value, accumulate);
}

Tensor & _index_put_impl_(Tensor & self, const torch::List<c10::optional<Tensor>>& indices, const Tensor & value, const bool accumulate, const bool unsafe) {
  TORCH_CHECK_INDEX(indices.size() <= (size_t)self.dim(), "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
  if (at::has_internal_overlap(self) == MemOverlap::YES) {
    TORCH_WARN(
      "Use of index_put_ on expanded tensors is deprecated. "
      "Please clone() the tensor before performing this operation. "
      "This also applies to advanced indexing e.g. tensor[indices] = tensor");
  }
  if (!accumulate) {
    auto masked_fill_dispatch = canDispatchToMaskedFill(self, indices, value);
    if (std::get<0>(masked_fill_dispatch)) {
      return self.masked_fill_(std::get<1>(masked_fill_dispatch), value.item());
    }
  }
  auto value_ = value;
  if (value.device() != self.device() && value.numel() == 1 && value.dim() == 0) {
    value_ = value.to(self.device());
  }
  at::assert_no_overlap(self, value);
  // NOLINTNEXTLINE(performance-implicit-conversion-in-loop)
  for (const c10::optional<Tensor>& index: indices) {
    if (index.has_value()) {
      at::assert_no_overlap(self, *index);
    }
  }
  if (self.device().type() == DeviceType::CUDA && (accumulate || globalContext().deterministicAlgorithms())) {
      TORCH_CHECK(value_.device() == self.device(), "expected device ", self.device(), " but got device ",
      value_.device(), " for value tensor");
      index_put_with_sort_stub(self.device().type(), self, indices, value_, accumulate, unsafe);
      return self;
  }

  auto info = make_info(self, indices);
  auto iter = make_index_put_iterator(info, value_);
  index_put_stub(iter.device_type(), iter, info.indexed_sizes, info.indexed_strides, accumulate);
  return self;
}

Tensor& take_out(const Tensor& self, const Tensor& index, Tensor& out) {
  // Type and device checks
  TORCH_CHECK(index.scalar_type() == ScalarType::Long, "take(): Expected a long tensor for index, but got ", index.scalar_type())
  TORCH_CHECK(self.scalar_type() == out.scalar_type(), "take(): self and out expected to have the same dtype, but got self.dtype = ", self.scalar_type(), " and out.dtype = ", out.scalar_type());
  TORCH_CHECK(self.device() == out.device() && self.device() == index.device(),
      "take(): self, index and out expected to be in the same device, but got self.device = ",
      self.device(), ", index.device = ", index.device(), ", and out.device = ", out.device());

  // index checks
  TORCH_CHECK_INDEX(!(self.numel() == 0 && index.numel() != 0), "take(): tried to take from an empty tensor");

  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, index);
  at::assert_no_overlap(out, self);

  // Do not iterate over self, we will compute the offsets manually
  // out is resized inside tensor_iterator
  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)
    .check_all_same_dtype(false)
    .add_output(out)
    .add_input(index)
    .build();

  // Early return after out has been resized
  if (index.numel() == 0) {
    return out;
  }

  take_stub(iter.device_type(), iter, self);

  return out;
}

Tensor take(const Tensor& self, const Tensor& index) {
    auto out = at::empty(index.sizes(), self.options());
    at::native::take_out(self, index, out);
    return out;
}

Tensor & index_put_(Tensor & self, const torch::List<c10::optional<Tensor>>& indices, const Tensor & value, const bool accumulate) {
  return at::_index_put_impl_(self, indices, value, accumulate, /*unsafe=*/false);
}

Tensor & index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  dim = maybe_wrap_dim(dim, self.dim());

  TORCH_CHECK_INDEX(index.dim() < 2, "index_copy_(): Index should have dimension 1 or 0 (got ", index.dim(), ")");
  at::assert_no_internal_overlap(self);
  at::assert_no_overlap(self, index);
  at::assert_no_overlap(self, source);

  int64_t numIndices = index.numel();
  if (source.dim() == 0 && numIndices != 1) {
    TORCH_CHECK_INDEX(false, "index_copy_(): When source is scalar, index should have one element (got ", numIndices, ")");
  } else if ((source.dim() != self.dim()) && (source.dim() != 0 && self.dim() != 0)) {
    TORCH_CHECK_INDEX(false, "index_copy_(): When source and destination are not scalars, their dimensionality must match. Source dimensionality (",
                   source.dim(), "), destination dimensionality (", self.dim(), ")");
  }

  TORCH_CHECK(index.scalar_type() == ScalarType::Long, "index_copy_(): Expected a long tensor for index, but got ", index.scalar_type())
  TORCH_CHECK(self.scalar_type() == source.scalar_type(), "index_copy_(): self and source expected to have the same dtype, but got (self) ", self.scalar_type(), " and (source) ", source.scalar_type());
  TORCH_CHECK(self.device() == source.device() && self.device() == index.device(),
      "index_copy_(): self, index and source expected to be in the same device, but got (self) ",
      self.device(), ", (index) ", index.device(), ", and (source) ", source.device());

  // Check that source and destination slices have the same size
  auto selfSlicedSizes = self.sizes().vec();
  if (selfSlicedSizes.size() > 0) {
    selfSlicedSizes.erase(selfSlicedSizes.begin() + dim);
  }
  auto sourceSlicedSizes = source.sizes().vec();
  if (sourceSlicedSizes.size() > 0) {
    sourceSlicedSizes.erase(sourceSlicedSizes.begin() + dim);
  }
  if (selfSlicedSizes.size() != sourceSlicedSizes.size() ||
      !std::equal(selfSlicedSizes.begin(), selfSlicedSizes.end(),
                  sourceSlicedSizes.begin())) {
    std::stringstream ss;
    ss << "index_copy_(): Source/destination tensor must have same slice shapes. ";
    ss << "Destination slice shape: " << selfSlicedSizes << " at dimension " << dim;
    ss << " and source slice shape: " << sourceSlicedSizes << " at dimension 0.";
    TORCH_CHECK(false, ss.str());
  }
  TORCH_CHECK_INDEX(source.dim() == 0 || numIndices == source.size(dim),
          "index_copy_(): Number of indices (", numIndices, ") should be equal to source.size(dim) (", source.size(dim), ")");

  // See Note [Enabling Deterministic Operations]
  if (self.device().type() == DeviceType::CUDA && globalContext().deterministicAlgorithms()){
    torch::List<c10::optional<Tensor>> indices;
    indices.reserve(dim + 1);
    for (const auto i: c10::irange(dim)) {
      (void)i;
      indices.emplace_back();
    }
    indices.emplace_back(index);
    return self.index_put_(indices, source, false);
  }

  return at::_index_copy_(self, dim, index, source);
}

Tensor & _index_copy_impl_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  // Handle the case when self / source is 0-dim
  Tensor self_nonzero = self.dim() == 0 ? self.unsqueeze(0) : self;
  Tensor source_nonzero = source.dim() == 0 ? source.unsqueeze(0) : source;

  // The only different between the following  tensor iterator and that of index_fill_ is that
  // this one has also source as an input. We should refactor it when if constexpr is available (C++17)

  // Prepare `index` for TensorIterator.
  // It is restrided to be broadcastable over `self` in TensorIterator.
  auto index_sizes = std::vector<int64_t>(self_nonzero.dim(), 1);
  auto index_strides = std::vector<int64_t>(self_nonzero.dim(), 0);
  index_sizes[dim] = index.numel();
  index_strides[dim] = (index.dim() > 0) ? index.stride(0) : 1; // `index` is 1d or scalar
  auto index_restrided = index.as_strided(
    index_sizes, index_strides);

  // Prepare `self` for TensorIterator.
  // Restride `self` to not advance in dimension `dim`.
  // We do not use squash_dim here because `index` will
  // need to advance in this dimension.
  // Note that self_sizes[dim] is set to index.numel().
  // This is done so that self_sizes[dim] and index_sizes[dim]
  // match as required by TensorIterator (input shape should
  // strictly broadcast over output shape, i.e.
  // output.shape[i] >= input.shape[i] for i in range(dims)).
  auto self_sizes = self_nonzero.sizes().vec();
  auto self_strides = self_nonzero.strides().vec();
  self_sizes[dim] = index.numel();
  self_strides[dim] = 0;
  auto self_restrided = self_nonzero.as_strided(self_sizes, self_strides);

  auto iter = TensorIteratorConfig()
    // We do not check for overlap because `self` is restrided
    // with zero stride. Zero strides trigger memory overlap assert
    // within TensorIterator.
    .set_check_mem_overlap(false)
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .add_output(self_restrided)
    .add_input(index_restrided)
    .add_input(source_nonzero)
    .build();

  auto self_dim_size = self_nonzero.size(dim);
  auto self_dim_stride = self_nonzero.stride(dim);
  index_copy_stub(
    iter.device_type(),
    iter,
    dim,
    self_dim_size,
    self_dim_stride);

  return self;
}

Tensor index_copy(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  return self.clone(at::MemoryFormat::Preserve).index_copy_(dim, index, source);
}

TORCH_IMPL_FUNC(index_add_cpu_out)
(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source, const Scalar& alpha, const Tensor& result) {
  if (!result.is_same(self)) result.copy_(self);
  auto numel = index.numel();

  auto index_contig = index.contiguous();

  if (result.dim() > 1) {
    // Equivalent to:
    //   for (const auto i : c10::irange(numel)) {
    //     auto selfSlice = self.select(dim, index_data[i]);
    //     auto sourceSlice = source.select(dim, i);
    //     selfSlice.add_(sourceSlice);
    //   }
    // But much faster as this reuses the iterator from add_
    if (numel == 0) {
      return;
    }
    auto selfSlice = result.select(dim, 0);
    auto sourceSlice = source.select(dim, 0);
    auto self_stride_bytes = result.stride(dim) * elementSize(result.scalar_type());
    auto source_stride_bytes = source.stride(dim) * elementSize(source.scalar_type());
    auto self_dim_size = result.size(dim);
    auto iter = TensorIterator::borrowing_binary_op(selfSlice, selfSlice, sourceSlice);

    AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_add_cpu_", [&] () {
      auto index_data = index_contig.data_ptr<index_t>();
      for (const auto i : c10::irange(numel)) {
          auto self_i = index_data[i];
          TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_dim_size), "index out of range in self");
          auto self_data = static_cast<char*>(selfSlice.data_ptr()) + self_i * self_stride_bytes;
          auto source_data = static_cast<char*>(sourceSlice.data_ptr()) + i * source_stride_bytes;
          iter.unsafe_replace_operand(0, self_data);
          iter.unsafe_replace_operand(1, self_data);
          iter.unsafe_replace_operand(2, source_data);
          add_stub(iter.device_type(), iter, alpha);
      }
    });
  }
  else {
    TORCH_CHECK(source.dim() <= 1, "source.dim() (", source.dim(), ") must one or zero for given self.dim() (", self.dim(), ")");

    // explicitly capture all required variables to work around windows build
    // TODO: fix this when windows can correctly capture variables in nested lambda
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16,
      result.scalar_type(), "index_add_", [&result, &source, &dim, &index_contig, &numel, &alpha] {
      auto alpha_value = alpha.to<scalar_t>();
      auto result_stride = result.dim() == 0 ? 1 : result.stride(dim);
      auto source_stride = source.dim() == 0 ? 1 : source.stride(dim);
      // TODO: Maybe TensorAccessor can be used here?
      auto* result_ptr = result.data_ptr<scalar_t>();
      auto* source_ptr = source.data_ptr<scalar_t>();
      AT_DISPATCH_INDEX_TYPES(index_contig.scalar_type(), "index_add_cpu_",
        [&index_contig, &numel, &result, &result_ptr, &result_stride, &source_ptr, &source_stride, &alpha_value] {
        auto index_data = index_contig.data_ptr<index_t>();
        for (const auto i : c10::irange(numel)) {
            auto self_i = index_data[i];
            TORCH_CHECK_INDEX((self_i >= 0) && (self_i < result.numel()), "index out of range in self");
            scalar_t *self_ip = result_ptr + self_i * result_stride;
            *self_ip += *(source_ptr + i * source_stride) * alpha_value;
        }
      });
    });
  }
}

// Check that indices fall within dimension array size
// Avoid redispatch call to min/max
template <typename IndexType>
static void check_indexarray_range(
    const IndexType* indices,
    int64_t n,
    IndexType indexing_axis_dim) {
  for (const auto i : c10::irange(n)) {
    auto idx = indices[i];
    TORCH_CHECK(
        0 <= idx && idx < indexing_axis_dim,
        "INDICES element is out of DATA bounds, id=",
        idx,
        " axis_dim=",
        indexing_axis_dim);
  }
}

Tensor & index_select_out_cpu_dim1_(
    Tensor & result_contig, const Tensor & self, const Tensor & index_contig) {

  auto self_contig = self.contiguous();
  const caffe2::TypeMeta dataType = self_contig.dtype();
  size_t item_bytesize = dataType.itemsize();

  auto out = static_cast<char*>(result_contig.data_ptr());

  auto src_base = static_cast<const char*>(self_contig.data_ptr());

  auto self_sizes = self_contig.sizes();
  auto outer_dims_product = c10::size_to_dim_(1, self_sizes);
  auto block_size = c10::size_from_dim_(2, self_sizes);
  auto block_bytesize = block_size * item_bytesize;

  auto src_indexing_axis_dim = self_sizes[1];
  auto src_batch_bytesize = self_sizes[1] * block_bytesize;
  auto N = index_contig.numel();

  auto gathered_batch_bytesize = N * block_bytesize;

  AT_DISPATCH_INDEX_TYPES(
    index_contig.scalar_type(), "batch_index_select_compute", [&]() {

      const auto* idxs = index_contig.data_ptr<index_t>();
      check_indexarray_range<index_t>(idxs, N, src_indexing_axis_dim);

      // Special-case single-float copy for efficiency
      if (self.scalar_type() == ScalarType::Float && block_size == 1) {
        for (const auto batch : c10::irange(outer_dims_product)) {
          const float* src_floats =
              (const float*)(src_base + batch * src_batch_bytesize);
          float* dst_floats = (float*)(out + batch * gathered_batch_bytesize);

          for (const auto i : c10::irange(N)) {
            auto idx = idxs[i];
            if (idx < 0) {
              idx = idx + src_indexing_axis_dim;
            }
            dst_floats[i] = src_floats[idx];
          }
        }
      } else {
        // outer_dims_product specifies how many times we repeat inner dimensions,
        // so we just iterate over it to cover all outer dimensions.
        for (const auto batch : c10::irange(outer_dims_product)) {
          for (const auto i : c10::irange(N)) {
            auto idx = idxs[i];
            if (idx < 0) {
              idx = idx + src_indexing_axis_dim;
            }

            auto src = src_base + batch * src_batch_bytesize + idx * block_bytesize;
            auto dst = out + batch * gathered_batch_bytesize + i * block_bytesize;
            memcpy(dst, src, block_bytesize);
          }
        }
      }
  });
  return result_contig;
}

Tensor & index_select_out_cpu_(const Tensor & self, int64_t dim, const Tensor & index, Tensor & result) {
  if (self.is_quantized()) {
    TORCH_CHECK(
        self.qscheme() == kPerTensorAffine,
        "Only per_tensor quantized quantized tensors are supported by index_select.")
  }
  dim = maybe_wrap_dim(dim, self.dim());
  auto numel = index.numel();
  TORCH_CHECK_INDEX(index.dim() <= 1, "index_select(): Index is supposed to be a vector");
  TORCH_CHECK(index.scalar_type() == ScalarType::Long || index.scalar_type() == ScalarType::Int, "index_select(): Expected dtype int32 or int64 for index");
  TORCH_CHECK(self.scalar_type() == result.scalar_type(),
              "index_select(): self and result must have the same scalar type");
  TORCH_CHECK(dim == 0 || dim < self.dim(),
              "index_select(): Indexing dim ", dim, " is out of bounds of tensor");
  at::assert_no_internal_overlap(result);
  at::assert_no_overlap(result, self);
  at::assert_no_overlap(result, index);
  auto result_size = self.sizes().vec();
  if (self.dim() > 0) {
    result_size[dim] = numel;
  }
  at::native::resize_output(result, result_size);

  auto index_contig = index.contiguous();

  if (self.dim() > 1) {
    if (numel == 0 || self.numel() == 0) {
      return result;
    }

    if (dim == 1 && result.is_contiguous()) {
      // fast pass
      return index_select_out_cpu_dim1_(result, self, index_contig);
    }

    auto selfSlice = self.select(dim, 0);
    auto resultSlice = result.select(dim, 0);
    auto selfSlice_data = selfSlice.data_ptr();
    auto resultSlice_data = resultSlice.data_ptr();
    auto self_stride_bytes = self.stride(dim) * elementSize(self.scalar_type());
    auto result_stride_bytes = result.stride(dim) * elementSize(result.scalar_type());
    auto self_dim_size = self.size(dim);
    auto slice_size = selfSlice.numel();

    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .add_output(resultSlice)
      .add_input(selfSlice)
      .build();

    auto grain_size = at::internal::GRAIN_SIZE;
    auto outer_loop =
      // explicitly capture all required variables to work around windows build
      // TODO: fix this when windows can correctly capture variables in nested lambda
      [&index_contig, &iter, &self_dim_size, &selfSlice_data, &self_stride_bytes, &resultSlice_data,
        &result_stride_bytes](int64_t start, int64_t end) {
      auto sub_iter = TensorIterator(iter);
      AT_DISPATCH_INDEX_TYPES(index_contig.scalar_type(), "index_select_out_cpu_",
        [&index_contig, &start, &end, &sub_iter, &self_dim_size, &selfSlice_data, &self_stride_bytes,
          &resultSlice_data, &result_stride_bytes] () {
        auto index_data = index_contig.data_ptr<index_t>();
        for (const auto i : c10::irange(start, end)) {
          auto self_i = index_data[i];
          TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_dim_size), "index out of range in self");
          auto self_data = static_cast<char*>(selfSlice_data) + self_i * self_stride_bytes;
          auto result_data = static_cast<char*>(resultSlice_data) + i * result_stride_bytes;
          sub_iter.unsafe_replace_operand(0, result_data);
          sub_iter.unsafe_replace_operand(1, self_data);
          copy_stub(sub_iter.device_type(), sub_iter, false);
        };
      });
    };

    // parallel on inner loop in case the slice is large enough;
    // otherwise parallel on outer loop
    if (slice_size >= grain_size) {
      outer_loop(0, numel);
    } else {
      // use a fast loop when self and result are contiguous and of the same data type
      if (iter.is_contiguous() && self.scalar_type() == result.scalar_type()) {
        auto slice_size_bytes = slice_size * elementSize(self.scalar_type());
        // explicitly capture all required variables to work around windows build
        // TODO: fix this when windows can correctly capture variables in nested lambda
        at::parallel_for(0, numel, grain_size / slice_size,
          [&index_contig, &slice_size_bytes, &self_dim_size, &selfSlice_data,
            &self_stride_bytes, &resultSlice_data, &result_stride_bytes](int64_t start, int64_t end) {
          AT_DISPATCH_INDEX_TYPES(index_contig.scalar_type(), "index_select_out_cpu_",
            [&index_contig, &slice_size_bytes, &self_dim_size, &selfSlice_data,
              &self_stride_bytes, &resultSlice_data, &result_stride_bytes, &start, &end] () {
            auto index_data = index_contig.data_ptr<index_t>();
            for (const auto i : c10::irange(start, end)) {
              auto self_i = index_data[i];
              TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_dim_size), "index out of range in self");
              auto self_data = static_cast<char*>(selfSlice_data) + self_i * self_stride_bytes;
              auto result_data = static_cast<char*>(resultSlice_data) + i * result_stride_bytes;
              memcpy(result_data, self_data, slice_size_bytes);
            }
          });
        });
      } else {
        at::parallel_for(0, numel, grain_size / slice_size, outer_loop);
      }
    }
  } else {
    TORCH_CHECK(result.dim() <= 1, "result.dim() (", result.dim(), ") must one or zero for given self.dim() (", self.dim(), ")");
    // explicitly capture all required variables to work around windows build
    // TODO: fix this when windows can correctly capture variables in nested lambda
    if(self.is_quantized()){
      AT_DISPATCH_QINT_TYPES(self.scalar_type(), "index_select_quant", [&index_contig, &self, &result, &dim, &numel] {
        auto self_stride = self.dim() == 0 ? 1 : self.stride(dim);
        auto result_stride = result.dim() == 0 ? 1 : result.stride(dim);
        auto self_data_ptr = self.data_ptr<scalar_t>();
        auto result_data_ptr = result.data_ptr<scalar_t>();
        auto self_numel = self.numel();
        AT_DISPATCH_INDEX_TYPES(index_contig.scalar_type(), "index_select_out_cpu_quant_",
          [&index_contig, &numel, &self_numel, &self_data_ptr, &self_stride, &result_data_ptr, &result_stride] {
          auto index_data = index_contig.data_ptr<index_t>();
          for (const auto i : c10::irange(numel)) {
            auto self_i = index_data[i];
            TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_numel), "index out of range in self");
            scalar_t *self_ip = self_data_ptr + self_i * self_stride;
            *(result_data_ptr + i * result_stride) = *self_ip;
          }
        });
      });
    } else {
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16,
        self.scalar_type(), "index_select", [&index_contig, &self, &result, &dim, &numel] {
        auto self_stride = self.dim() == 0 ? 1 : self.stride(dim);
        auto result_stride = result.dim() == 0 ? 1 : result.stride(dim);

        auto self_data_ptr = self.data_ptr<scalar_t>();
        auto result_data_ptr = result.data_ptr<scalar_t>();
        auto self_numel = self.numel();
        AT_DISPATCH_INDEX_TYPES(index_contig.scalar_type(), "index_select_out_cpu_",
          [&index_contig, &numel, &self_numel, &self_data_ptr, &self_stride, &result_data_ptr, &result_stride] {
          auto index_data = index_contig.data_ptr<index_t>();
          for (const auto i : c10::irange(numel)) {
            auto self_i = index_data[i];
            TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_numel), "index out of range in self");
            scalar_t *self_ip = self_data_ptr + self_i * self_stride;
            *(result_data_ptr + i * result_stride) = *self_ip;
          }
        });
      });
    }
  }

  return result;
}

Tensor index_select_cpu_(const Tensor & self, int64_t dim, const Tensor & index) {
  Tensor result = at::empty({0}, self.options());
  return at::native::index_select_out_cpu_(self, dim, index, result);
}

Tensor index_select_quantized_cpu_(const Tensor & self, int64_t dim, const Tensor & index) {
  TORCH_CHECK(self.qscheme() == kPerTensorAffine,
              "Only per_tensor quantized quantized tensors are supported by index_select.")
  Tensor result = at::empty_quantized({0}, self);
  return at::native::index_select_out_cpu_(self, dim, index, result);
}

Tensor index_select_backward(const Tensor& grad, IntArrayRef self_sizes, int64_t dim, const Tensor& index) {
  return at::zeros(self_sizes, grad.options()).index_add_(dim, index, grad);
}

Tensor & index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Scalar& source) {
  at::NoNamesGuard guard;

  TORCH_CHECK_INDEX(
    index.scalar_type() == ScalarType::Long,
    "index_fill_(): Expected dtype int64 for index.");

  at::assert_no_overlap(self, index);
  if (at::has_internal_overlap(self) == at::MemOverlap::YES) {
    TORCH_WARN(
      "Use of index_fill_ on expanded tensors is deprecated. "
      "Please clone() the tensor before performing this operation. "
      "This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }

  if (!self.is_complex() && source.isComplex()) {
    TORCH_CHECK(false, "index_fill_(): Converting complex Scalar to non-complex type is not supported");
  }

  // Handle the case when `self` is 0-dim
  Tensor self_nonzero_dim = (self.dim() == 0) ? self.unsqueeze(-1) : self;

  dim = at::maybe_wrap_dim(dim, self_nonzero_dim);
  TORCH_CHECK(index.dim() <= 1, "Index has to be a vector/scalar");

  // Prepare `index` for TensorIterator.
  // It is restrided to be broadcastable over `self` in TensorIterator.
  auto index_sizes = std::vector<int64_t>(self_nonzero_dim.dim(), 1);
  auto index_strides = std::vector<int64_t>(self_nonzero_dim.dim(), 0);
  index_sizes[dim] = index.numel();
  index_strides[dim] = (index.dim() > 0) ? index.stride(0) : 1; // `index` is 1d or scalar
  auto index_restrided = index.as_strided(
    index_sizes, index_strides);

  // Prepare `self` for TensorIterator.
  // Restride `self` to not advance in dimension `dim`.
  // We do not use squash_dim here because `index` will
  // need to advance in this dimension.
  // Note that self_sizes[dim] is set to index.numel().
  // This is done so that self_sizes[dim] and index_sizes[dim]
  // match as required by TensorIterator (input shape should
  // strictly broadcast over output shape, i.e.
  // output.shape[i] >= input.shape[i] for i in range(dims)).
  auto self_sizes = self_nonzero_dim.sizes().vec();
  auto self_strides = self_nonzero_dim.strides().vec();
  self_sizes[dim] = index.numel();
  self_strides[dim] = 0;
  auto self_restrided = self_nonzero_dim.as_strided(self_sizes, self_strides);

  auto iter = TensorIteratorConfig()
    // We do not check for overlap because `self` is restrided
    // with zero stride. Zero strides trigger memory overlap assert
    // within TensorIterator.
    .set_check_mem_overlap(false)
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .add_output(self_restrided)
    .add_input(index_restrided)
    .build();

  auto self_dim_size = (self_nonzero_dim.sizes())[dim];
  auto self_dim_stride = (self_nonzero_dim.strides())[dim];
  index_fill_stub(
    iter.device_type(),
    iter,
    dim,
    self_dim_size,
    self_dim_stride,
    source);

  return self;
}

Tensor & index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  TORCH_CHECK(source.dim() == 0, "index_fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ", source.dim(), " dimension(s).");
  return self.index_fill_(dim, index, source.item());
}

Tensor index_fill(const Tensor & self, int64_t dim, const Tensor & index, const Scalar& source) {
  return self.clone(at::MemoryFormat::Preserve).index_fill_(dim, index, source);
}

Tensor index_fill(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  return self.clone(at::MemoryFormat::Preserve).index_fill_(dim, index, source);
}

// gather_out_cpu_cuda
TORCH_IMPL_FUNC(gather_out)
(const Tensor& self, int64_t dim, const Tensor& index, bool sparse_grad, const Tensor& result) {
  if (index.numel() == 0) return;
  dim = at::maybe_wrap_dim(dim, self.dim());
  gather_stub(result.device().type(), result, self, dim, index);
}

Tensor gather_backward(const Tensor& grad, const Tensor& self, int64_t dim, const Tensor& index, bool sparse_grad) {
  if (sparse_grad) {
    return at::_gather_sparse_backward(self, dim, index, grad);
  }
  return grad.new_zeros(self.sizes()).scatter_add_(dim, index, grad);
}

template <typename T, typename ReduceStub, typename FillStub>
void scatter_impl(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const T& src,
    const Tensor& out,
    ReduceStub& reduce_stub,
    FillStub& fill_stub,
    const c10::optional<c10::string_view> reduce = nullopt) {

  dim = at::maybe_wrap_dim(dim, self.dim());
  auto mut_out = const_cast<Tensor&>(out);

  if (!self.is_same(mut_out)) {
    mut_out.copy_(self);
  }

  if (index.numel() == 0) return;

  if (reduce.has_value()) {
    auto op = meta::get_operator_enum(reduce.value());
    reduce_stub(self.device().type(), mut_out, dim, index, src, op);
  } else {
    fill_stub(self.device().type(), mut_out, dim, index, src);
  }
}

TORCH_IMPL_FUNC(scatter_src_out)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& src,
 const Tensor& out) {
  scatter_impl(self, dim, index, src, out,
               scatter_reduce_stub,
               scatter_stub);
}

TORCH_IMPL_FUNC(scatter_value_out)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Scalar& value,
 const Tensor& out) {
  scatter_impl(self, dim, index, value, out,
               scatter_scalar_reduce_stub,
               scatter_fill_stub);
}

TORCH_IMPL_FUNC(scatter_reduce_out)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& src,
 const c10::string_view reduce,
 const Tensor& out) {
  scatter_impl(self, dim, index, src, out,
               scatter_reduce_stub,
               scatter_stub,
               reduce);
}

TORCH_IMPL_FUNC(scatter_value_reduce_out)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Scalar& value,
 const c10::string_view reduce,
 const Tensor& out) {
  scatter_impl(self, dim, index, value, out,
               scatter_scalar_reduce_stub,
               scatter_fill_stub,
               reduce);
}

TORCH_IMPL_FUNC(scatter_add)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& src,
 const Tensor& out) {
  auto mut_out = const_cast<Tensor&>(out);
  dim = maybe_wrap_dim(dim, self.dim());

  if (!self.is_same(mut_out)) {
    mut_out.copy_(self);
  }

  if (index.numel() == 0) return;

  if (globalContext().deterministicAlgorithms() && self.device().type() == DeviceType::CUDA && self.dim() == 1) {
    TORCH_CHECK(index.dim() == 1 && src.dim() == 1, "index and src should be 1D tensors when self is a 1D tensor, "
      "but their dims are ", index.dim(), " and ", src.dim(), ", respectively");
    TORCH_CHECK(index.numel() == src.numel(), "index and src should have same number of elements for 1D tensors, "
      "but got ", index.numel(), " versus ", src.numel());
    TORCH_CHECK(dim == 0, "dim should be zero for 1D self tensor, but got ", dim);
    torch::List<c10::optional<Tensor>> indices;
    indices.reserve(1);
    indices.push_back(index);
    mut_out.index_put_(indices, src, true);
  } else {
    scatter_add_stub(self.device().type(), mut_out, dim, index, src);
  }
}

Tensor scatter_reduce_two_cpu(const Tensor& self,
                              int64_t dim,
                              const Tensor& index,
                              const c10::string_view reduce,
                              const c10::optional<int64_t> output_size) {

  // TODO: Add documentation.


  TORCH_CHECK(dim >= -self.dim() && dim < self.dim(),
      "Expected `dim` to be in range ", -self.dim(), " to ", self.dim() - 1, " (got ", dim, ")");

  dim = dim < 0 ? dim + self.dim() : dim;

  auto sizes = self.sizes().vec();
  if (output_size.has_value()) {
    sizes[dim] = output_size.value();
  } else {
    sizes[dim] = index.numel() > 0 ? index.max().item<int64_t>() + 1: 0;
  }
  Tensor out = at::empty(sizes, self.options());

  TORCH_CHECK(self.dim() == index.dim(),
      "Shape mismatch between `self` (got ", self.sizes(), ") and `index` (got ", index.sizes(), ")");
  for (int64_t i = 0; i < self.dim(); i++) {
    TORCH_CHECK(self.size(i) == index.size(i),
        "Shape mismatch between `self` (got ", self.sizes(), ") and `index` (got ", index.sizes(), ")");
  }

  TORCH_CHECK(reduce == "sum" || reduce == "prod" || reduce == "mean" || reduce == "amax" || reduce =="amin",
              "`reduce` argument must be one of ('sum', 'prod', 'mean', 'amax', 'amin'");

  if (self.numel() == 0) {
    return out.zero_();
  }

  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, self.scalar_type(), "scatter_reduce", [&] {
    if (reduce == "prod") {
      out.fill_((scalar_t)1);
    } else if (reduce == "amax") {
      out.fill_(std::numeric_limits<scalar_t>::lowest());
    } else if (reduce == "amin") {
      out.fill_(std::numeric_limits<scalar_t>::max());
    } else {
      out.fill_((scalar_t)0);
    }


    auto self_cont = self.contiguous();
    auto index_cont = index.contiguous();
    auto self_data = self_cont.data_ptr<scalar_t>();
    auto index_data = index_cont.data_ptr<int64_t>();
    bool out_is_contiguous = out.is_contiguous();
    auto out_cont = out.contiguous();
    auto out_cont_data = out_cont.data_ptr<scalar_t>();

    auto counts = at::zeros_like(out_cont);
    auto counts_data = counts.data_ptr<scalar_t>();


    int64_t offset1 = 1, offset2 = 1;
    for (int64_t d = 0; d < dim; d++)
      offset1 *= self.size(d);
    for (int64_t d = dim + 1; d < self.dim(); d++)
      offset2 *= self.size(d);

    scalar_t value;
    int64_t dim_index;
    for (int64_t i = 0; i < offset1; i++) {
      for (int64_t j = 0; j < self.size(dim); j++) {
        for (int64_t k = 0; k < offset2; k++) {
          value = self_data[i * self_cont.stride(dim) * self_cont.size(dim) + j * self_cont.stride(dim) + k];
          dim_index = index_data[i * index_cont.stride(dim) * index_cont.size(dim) + j * index_cont.stride(dim) + k];
          TORCH_CHECK(dim_index >= 0 && dim_index < out.size(dim),
              "Expected `index` values to be in range ", 0, " to ", out.size(dim), " (got ", dim_index, ")");
          int64_t ind = i * out_cont.stride(dim) * out_cont.size(dim) + dim_index * out_cont.stride(dim) + k;
          if (reduce == "sum") {
            out_cont_data[ind] += value;
          } else if (reduce == "prod") {
            out_cont_data[ind] *= value;
          } else if (reduce == "mean") {
            auto n = counts_data[ind];
            out_cont_data[ind] = (out_cont_data[ind] * n + value) / (n + 1);
            counts_data[ind] += 1;
          } else if (reduce == "amax") {
            out_cont_data[ind] = std::max(out_cont_data[ind], value);
          } else {
            out_cont_data[ind] = std::min(out_cont_data[ind], value);
          }
        }
      }
    }

    if (reduce == "amin" || reduce == "amax") {
      auto val = (reduce == "amin") ? std::numeric_limits<scalar_t>::max() : std::numeric_limits<scalar_t>::lowest();
      out_cont.masked_fill_(out_cont == val, (scalar_t)0);
    }

    if (!out_is_contiguous) {
      out.copy_(out_cont);
    }

  });

  return out;
}

Tensor masked_scatter(const Tensor & self, const Tensor & mask, const Tensor & source) {
  c10::MaybeOwned<Tensor> _mask, _self;
  std::tie(_mask, _self) = expand_outplace(mask, self);
  return _self->clone(at::MemoryFormat::Contiguous).masked_scatter_(*_mask, source);
}

static Tensor & masked_fill_impl_cpu(Tensor & self, const Tensor & mask, const Scalar& value) {
  NoNamesGuard guard;
  if (mask.dtype() == ScalarType::Byte) {
    TORCH_WARN("masked_fill_ received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
  }

  if (at::has_internal_overlap(self) == MemOverlap::YES) {
    TORCH_WARN(
      "Use of masked_fill_ on expanded tensors is deprecated. "
      "Please clone() the tensor before performing this operation. "
      "This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }
  at::assert_no_partial_overlap(self, mask);

  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)  // deprecated, but not a hard error
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .add_output(self)
    .add_input(mask)
    .build();

  masked_fill_stub(iter.device_type(), iter, value);
  return self;
}

Tensor & masked_fill__cpu(Tensor& self, const Tensor & mask, const Scalar& value) {
  auto maybe_outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");

  masked_fill_impl_cpu(self, mask, value);
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;
}

Tensor & masked_fill__cpu(Tensor& self, const Tensor & mask, const Tensor & value) {
  auto maybe_outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");
  TORCH_CHECK(value.dim() == 0, "masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ", value.dim(), " dimension(s).");

  masked_fill_impl_cpu(self, mask, value.item());
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;
}

Tensor masked_fill(const Tensor & self, const Tensor & mask, const Scalar& source) {
  Tensor result;
  auto maybe_outnames = namedinference::broadcast_to_outnames(mask, self, "masked_fill");
  {
    NoNamesGuard guard;
    c10::MaybeOwned<Tensor> _mask, _self;
    std::tie(_mask, _self) = expand_outplace(mask, self);
    result = _self->clone(at::MemoryFormat::Contiguous);
    result.masked_fill_(mask, source);
  }
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

Tensor masked_fill(const Tensor & self, const Tensor & mask, const Tensor & source) {
  Tensor result;
  auto maybe_outnames = namedinference::broadcast_to_outnames(mask, self, "masked_fill");
  {
    NoNamesGuard guard;
    c10::MaybeOwned<Tensor> _mask, _self;
    std::tie(_mask, _self) = expand_outplace(mask, self);
    result = _self->clone(at::MemoryFormat::Contiguous);
    result.masked_fill_(mask, source);
  }
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

static Tensor & masked_select_out_impl_cpu(Tensor & result, const Tensor & self, const Tensor & mask) {
  NoNamesGuard guard;

  TORCH_CHECK(mask.scalar_type() == ScalarType::Byte || mask.scalar_type() == ScalarType::Bool,
              "masked_select: expected BoolTensor or ByteTensor for mask");
  TORCH_CHECK(self.scalar_type() == result.scalar_type(),
              "masked_select(): self and result must have the same scalar type");

  at::assert_no_internal_overlap(result);
  at::assert_no_overlap(result, self);
  at::assert_no_overlap(result, mask);

  if (mask.dtype() == at::ScalarType::Byte) {
    TORCH_WARN("masked_select received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
  }

  c10::MaybeOwned<Tensor> _mask, _self;
  std::tie(_mask, _self) = expand_outplace(mask, self);

  auto shape = _self->sizes();
  int64_t numel = _mask->sum().item().toLong();
  at::native::resize_output(result, {numel});
  if (numel == 0) {
    return result;
  }

  // Create strided view of result before feeding into TensorIterator
  auto strides = DimVector(shape.size(), 0);
  auto orig_stride = result.strides()[0];
  auto result_strided = result.as_strided(shape, strides);

  // serial kernel
  // serial kernel requires that src is traversed in its logical order. However, TensorIterator might
  // have reordered dimensions so that src would be traversed in its physical order, producing wrong
  // answers. A sufficient condition that no reorder happened is that both _self and _mask is contiguous.
  // If it is not satisfied, use parallel kernel that handles permutations correctly
  bool use_serial_kernel = (self.numel() < at::internal::GRAIN_SIZE || at::get_num_threads() == 1 ) &&
  _self->is_contiguous() && _mask->is_contiguous();
  if (use_serial_kernel) {
    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)  // result is intenionally zero-strided above
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .add_output(result_strided)
      .add_input(*_self)
      .add_input(*_mask)
      .build();

    masked_select_serial_stub(iter.device_type(), iter, orig_stride);
    return result;
  }

  // Use a prefix sum to record the output locations of the masked elements,
  // so as to parallel with TensorIterator.
  auto mask_long = at::empty(shape, self.options().dtype(at::kLong)).copy_(*_mask);
  auto mask_prefix_sum = at::empty(shape, self.options().dtype(at::kLong));
  auto mask_long_data = mask_long.data_ptr<int64_t>();
  auto mask_prefix_sum_data = mask_prefix_sum.data_ptr<int64_t>();
  // TODO: Here can only use std::partial_sum for C++14,
  // use std::exclusive_scan when PyTorch upgrades to C++17, which have better peformance.
  // std::exclusive_scan(mask_long_data, mask_long_data + mask_long.numel(), mask_prefix_sum_data, 0);
  std::partial_sum(mask_long_data, mask_long_data + mask_long.numel(), mask_prefix_sum_data);

  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)  // result is intenionally zero-strided above
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .add_output(result_strided)
    .add_input(*_self)
    .add_input(*_mask)
    .add_input(mask_prefix_sum)
    .build();

  masked_select_stub(iter.device_type(), iter, orig_stride);
  return result;
}

Tensor & masked_select_out_cpu(const Tensor & self, const Tensor & mask, Tensor & result) {
  namedinference::compute_broadcast_outnames(self, mask);
  return masked_select_out_impl_cpu(result, self, mask);
}

Tensor masked_select_cpu(const Tensor & self, const Tensor & mask) {
  Tensor result = at::empty({0}, self.options());
  return at::native::masked_select_out_cpu(self, mask, result);
}

Tensor masked_select_backward(const Tensor& grad, const Tensor& input, const Tensor& mask) {
  // The following could just be written as `zeros_like(input).masked_scatter(mask, grad)`.
  // However, as an optimization, we call the in-place variant of masked_scatter.
  // Unfortunately, that doesn't allow for the broadcasting of the LHS, so we need
  // to explicitly broadcast here (the out-of-place variant of masked_scatter
  // implicitly handles broadcasting).
  auto result = at::zeros_like(
      input.expand(at::infer_size(input.sizes(), mask.sizes())), at::MemoryFormat::Preserve);
  return result.masked_scatter_(mask, grad);
}

namespace {

inline std::tuple<Tensor, Tensor, int64_t> _take_along_dim_helper(
    const Tensor& self,
    const Tensor& indices,
    int64_t dim) {
  TORCH_CHECK(
      self.dim() == indices.dim(),
      "torch.take_along_dim(): input and indices should have the same number of dimensions, ",
      "but got ", self.dim(), " dimensions for input, and ", indices.dim(), " dimensions for indices")
  TORCH_CHECK(
      indices.scalar_type() == ScalarType::Long,
      "torch.take_along_dim(): dtype of indices should be Long but got ", indices.scalar_type())

  dim = at::maybe_wrap_dim(dim, self.dim());

  DimVector self_sizes{self.sizes()};
  // update number of elements at dim as per indices
  self_sizes[dim] = indices.size(dim);
  auto broadcast_shape = infer_size(self_sizes, indices.sizes());
  auto indices_broadcasted = at::broadcast_to(indices, broadcast_shape);

  DimVector indices_sizes{indices.sizes()};
  // update number of elements at dim as per self
  indices_sizes[dim] = self.size(dim);
  broadcast_shape = infer_size(indices_sizes, self.sizes());
  auto self_broadcasted = at::broadcast_to(self, broadcast_shape);

  return std::make_tuple(self_broadcasted, indices_broadcasted, dim);
}

static inline void checkDevice(CheckedFrom c, const Tensor& t, Device device) {
  TORCH_CHECK(
      !t.defined() || t.device() == device,
      "Expected tensor to have ", device,
      " Device, but got tensor with ", t.device(), " Device ",
      "(while checking arguments for ", c, ")");
}

static inline void checkDevice(CheckedFrom c, at::ArrayRef<Tensor> tensors, Device device) {
  for (auto &t : tensors) {
    checkDevice(c, t, device);
  }
}

} // anonymous namespace

Tensor take_along_dim(const Tensor& self, const Tensor& indices, c10::optional<int64_t> opt_dim) {
  checkDevice("torch.take_along_dim():", {self, indices}, self.device());
  if (opt_dim.has_value()) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t dim;
    Tensor self_broadcasted, indices_broadcasted;
    std::tie(self_broadcasted, indices_broadcasted, dim) =
        _take_along_dim_helper(self, indices, opt_dim.value());
    return self_broadcasted.gather(dim, indices_broadcasted);
  }

  // similar to `take`, but `take` doesn't support the same dtypes as `gather`.
  return self.view(-1).gather(0, indices.view(-1));
}

Tensor& take_along_dim_out(const Tensor& self, const Tensor& indices, c10::optional<int64_t> opt_dim, Tensor& result) {
  checkDevice("torch.take_along_dim():", {self, indices, result}, self.device());
  if (opt_dim.has_value()) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t dim;
    Tensor self_broadcasted, indices_broadcasted;
    std::tie(self_broadcasted, indices_broadcasted, dim) =
        _take_along_dim_helper(self, indices, opt_dim.value());
    return at::gather_out(result, self_broadcasted, dim, indices_broadcasted);
  }

  // similar to `take`, but `take` doesn't support the same dtypes as `gather`.
  return at::gather_out(result, self.view(-1), 0, indices.view(-1));
}

Tensor _gather_sparse_backward(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& grad){
// special case scalar input and/or index
    if (self.ndimension() == 0) return at::_sparse_coo_tensor_unsafe(at::empty({0,grad.numel()}, index.options()), grad, self.sizes());
    if (grad.ndimension() == 0) return at::_sparse_coo_tensor_unsafe(index.view({1,1}), grad, self.sizes());
    Tensor sparse_ind = at::empty({self.ndimension(), grad.numel()}, self.options().dtype(at::kLong));
    int64_t n_above = grad.numel();
    int64_t n_below = 1;
    if (dim < 0) dim += self.ndimension();
    for (const auto i : c10::irange(self.ndimension())) {
        n_above /= grad.size(i);
        if (i == dim) {
            sparse_ind[i] = index.reshape(-1);
        } else {
            sparse_ind[i] = at::arange(grad.size(i),self.options().dtype(at::kLong)).unsqueeze(1).expand({grad.size(i), n_above}).reshape(-1).repeat(n_below);
        }
        n_below *= grad.size(i);
    }
    return at::_sparse_coo_tensor_unsafe(sparse_ind, grad.reshape(-1), self.sizes());
}

template <typename scalar_t>
int64_t count_nonzero_impl(TensorIteratorBase& iter, Range range) {
  int64_t num_nonzero = 0;

  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    constexpr int ilp_factor = 4;
    const char* ptr = data[0];
    const auto stride = strides[0];
    int64_t nonzero[ilp_factor] = {0};

    int64_t i = 0;
    for (; i + (ilp_factor - 1) < n; i += ilp_factor) {
      c10::ForcedUnroll<ilp_factor>{}([&](int k) {
        const auto& val = *reinterpret_cast<const scalar_t*>(ptr + k * stride);
        if (val != scalar_t(0)) {
          ++nonzero[k];
        }
      });
      ptr += ilp_factor * stride;
    }
    for (; i < n; ++i) {
      const auto& val = *reinterpret_cast<const scalar_t*>(ptr);
      if (val != scalar_t(0)) {
        ++nonzero[0];
      }
      ptr += stride;
    }
    for (const auto k : c10::irange(1, ilp_factor)) {
      nonzero[0] += nonzero[k];
    }
    num_nonzero += nonzero[0];
  };
  iter.serial_for_each(loop, range);

  return num_nonzero;
}

Tensor count_nonzero_cuda(const Tensor& self, IntArrayRef dims){
  return (self != 0).sum(dims);
}

Tensor count_nonzero_cpu(const Tensor& self, IntArrayRef dims){
  if (dims.size() > 0) {
    return (self != 0).sum(dims);
  }

  // Optimized all-reduce
  auto iter = TensorIteratorConfig()
      .add_input(self)
      .build();

  const auto num_threads = at::get_num_threads();
  DimVector thread_count_nonzero(num_threads);

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBFloat16, kBool, self.scalar_type(), "nonzero_count_cpu", [&] {
    at::parallel_for(0, iter.numel(), internal::GRAIN_SIZE, [&] (int64_t begin, int64_t end) {
      const auto tid = at::get_thread_num();
      thread_count_nonzero[tid] = count_nonzero_impl<scalar_t>(iter, {begin, end});
    });
  });

  for (const auto i : c10::irange(1, num_threads)) {
    thread_count_nonzero[0] += thread_count_nonzero[i];
  }
  auto out = at::empty({}, self.options().dtype(kLong));
  *out.data_ptr<int64_t>() = thread_count_nonzero[0];
  return out;
}


Tensor count_nonzero(const Tensor& self, c10::optional<int64_t> dim) {
  if (dim) {
    return at::count_nonzero(self, IntArrayRef{*dim});
  }
  return at::count_nonzero(self, IntArrayRef{});
}


Tensor& nonzero_out_cpu(const Tensor& self, Tensor& result) {
  TORCH_CHECK(result.scalar_type() == kLong,
              "nonzero: Expected out tensor to have scalar type Long "
              "but got scalar type", result.scalar_type());
  at::assert_no_internal_overlap(result);
  at::assert_no_overlap(result, self);

  auto iter = TensorIteratorConfig()
    .add_input(self)
    .enforce_linear_iteration()
    .build();

  const auto numel = iter.numel();
  const auto num_threads = at::get_num_threads();
  DimVector thread_begin(num_threads, -1);
  DimVector thread_count_nonzero(num_threads + 1);

  // Pass 1: Count nonzero element per-thread
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBFloat16, kBool, self.scalar_type(), "nonzero_count_cpu", [&] {
    at::parallel_for(0, numel, internal::GRAIN_SIZE, [&] (int64_t begin, int64_t end) {
      const auto tid = at::get_thread_num();
      thread_begin[tid] = begin;
      thread_count_nonzero[tid + 1] = count_nonzero_impl<scalar_t>(iter, {begin, end});
    });
  });

  // Convert thread-local counts to cumulative sum
  for (const auto i : c10::irange(1, thread_count_nonzero.size())) {
    thread_count_nonzero[i] += thread_count_nonzero[i - 1];
  }

  const auto self_sizes = self.sizes();
  const auto total_nonzero = thread_count_nonzero.back();
  const int64_t ndim = self_sizes.size();
  if (resize_output(result, {total_nonzero, ndim})) {
    // Default to fortran-contiguous output (see gh-46224)
    result.as_strided_({total_nonzero, ndim}, {1, total_nonzero});
  }

  if (result.numel() == 0) {
    return result;
  }

  // Pass 2: Write indexes
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBFloat16, kBool, self.scalar_type(), "nonzero_cpu", [&] {
    at::parallel_for(0, numel, internal::GRAIN_SIZE, [&] (int64_t begin, int64_t end) {
      auto tid = at::get_thread_num();
      // Work needs to be distributed the same on both passes
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(begin == thread_begin[tid]);

      // +1 faster than additional condition check inside loop
      c10::SmallVector<int64_t, 33> sizes(ndim + 1, -1);
      std::copy(self_sizes.begin(), self_sizes.end(), sizes.begin() + 1);
      c10::SmallVector<int64_t, 33> current_idx(ndim + 1);
      if (begin > 0) {
        auto idx = begin;
        for (int64_t k = ndim; idx > 0 && k > 0; --k) {
          current_idx[k] = idx % sizes[k];
          idx /= sizes[k];
        }
      }

      auto out_accessor = result.accessor<int64_t, 2>();
      auto out_ptr = out_accessor[thread_count_nonzero[tid]].data();

      auto loop = [&](char** data, const int64_t* strides, int64_t n1, int64_t n2) {
        // Copy into local variables to improve compiler alias analysis
        int64_t* C10_RESTRICT local_idx = current_idx.data() + 1;
        const int64_t* C10_RESTRICT local_sizes = sizes.data() + 1;
        const auto in_stride = strides[0];
        const auto out_stride1 = out_accessor.stride(1);
        const auto out_stride0 = out_accessor.stride(0) - ndim * out_stride1;
        const auto ndim = out_accessor.size(1);
        int64_t* out = out_ptr;

        for (const auto i : c10::irange(n2)) {
          const char* ptr = data[0] + i * strides[1];
          for (const auto j : c10::irange(n1)) {
            (void)j; //Suppress unused variable warning
            const auto& val = *reinterpret_cast<const scalar_t*>(ptr);
            // If nonzero, write index
            if (val != scalar_t(0)) {
              for (const auto k : c10::irange(ndim)) {
                *out = local_idx[k];
                out += out_stride1;
              }
              out += out_stride0;
            }
            ptr += in_stride;

            // Advance current index
            int64_t k = ndim - 1;
            ++local_idx[k];
            while (C10_UNLIKELY(local_idx[k] == local_sizes[k])) {
              local_idx[k] = 0;
              --k;
              ++local_idx[k];
            }
          }
        }
        out_ptr = out;
      };
      iter.serial_for_each(loop, {begin, end});
      TORCH_INTERNAL_ASSERT(out_ptr == out_accessor[thread_count_nonzero[tid + 1]].data());
    });
  });
  return result;
}

Tensor nonzero_cpu(const Tensor& self) {
  auto result = at::empty({0}, self.options().dtype(kLong));
  nonzero_out_cpu(self, result);
  return result;
}

std::vector<Tensor> nonzero_numpy(const Tensor& self) {
  // special case scalar for compatibility with numpy:
  //
  // >>> np.array(5).nonzero()
  // (array([0]),)
  // >>> np.array(0).nonzero()
  // (array([], dtype=int64),)

  if (self.dim() == 0) {
    return self.unsqueeze(0).nonzero().unbind(1);
  }

  return self.nonzero().unbind(1);
}

Tensor argwhere(const Tensor& self) {
  return self.nonzero();
}

Tensor & masked_scatter__cpu(Tensor& self, const Tensor & mask, const Tensor & source) {
  at::assert_no_internal_overlap(self);
  TORCH_CHECK(
      self.scalar_type() == source.scalar_type(),
      "masked_scatter: expected self and source to have same dtypes but got",
      self.scalar_type(),
      " and ",
      source.scalar_type());

  TORCH_CHECK(self.device().type() == at::kCPU, "device type of self (", self.device().type(), ") is not CPU");
  TORCH_CHECK(mask.device().type() == at::kCPU, "device type of mask (", mask.device().type(), ") is not CPU");
  TORCH_CHECK(source.device().type() == at::kCPU, "device type of source (", source.device().type(), ") is not CPU");

  c10::MaybeOwned<Tensor> b_mask = expand_inplace(self, mask, "masked_scatter_");

  if (b_mask->dtype() == ScalarType::Byte) {
    TORCH_WARN("masked_scatter_ received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
  }

  auto src_cont = source.contiguous();

  auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .add_output(self)
      .add_input(*b_mask)
      .build();

  masked_scatter_stub(iter.device_type(), iter, src_cont);
  return self;
}

}} // at::native
