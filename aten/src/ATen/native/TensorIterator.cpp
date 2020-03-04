#include <ATen/native/TensorIterator.h>

#include <array>
#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <ATen/native/TypeProperties.h>

namespace at {

using DimMask = TensorIterator::DimMask;
using PtrVector = TensorIterator::PtrVector;
using loop_t = TensorIterator::loop_t;
using loop2d_t = TensorIterator::loop2d_t;
using StrideVector = TensorIterator::StrideVector;

void TensorIterator::reorder_dimensions() {
  // Sort the dimensions based on strides in ascending order with reduced dims
  // at the front. NOTE: that this inverts the order of C-contiguous tensors.
  // strides[0] is the fastest moving dimension instead of strides[ndim - 1].

  perm_.resize(ndim());
  if (ndim() == 1) {
    perm_[0] = 0;
    return;
  }

  // initialize perm with n-1, n-2, ..., 1, 0
  std::iota(perm_.rbegin(), perm_.rend(), 0);

  // returns 1 if the dim0 should come after dim1, -1 if dim0 should come
  // before dim1, and 0 if the comparison is ambiguous.
  auto should_swap = [&](size_t dim0, size_t dim1) {
    int ret = 0;
    for (int arg = 0; arg < ntensors(); arg++) {
      if (operands_[arg].stride_bytes.empty()) {
        continue;
      }
      int64_t stride0 = operands_[arg].stride_bytes[dim0];
      int64_t stride1 = operands_[arg].stride_bytes[dim1];
      if (is_reduction_ && operands_[arg].is_output) {
        // move reduced dimensions to the front
        if ((stride0 == 0) != (stride1 == 0)) {
          return stride1 == 0 ? 1 : -1;
        }
      }
      if (stride0 == 0 || stride1 == 0) {
        continue;
      } else if (stride0 <= stride1) {
        return -1;
      } else {
        return 1;
      }
    }
    return ret;
  };

  // insertion sort with support for ambiguous comparisons
  for (int i = 1; i < ndim(); i++) {
    int dim1 = i;
    for (int dim0 = i - 1; dim0 >= 0; dim0--) {
      int comparison = should_swap(perm_[dim0], perm_[dim1]);
      if (comparison > 0) {
        std::swap(perm_[dim0], perm_[dim1]);
        dim1 = dim0;
      } else if (comparison < 0) {
        break;
      }
    }
  }

  // perform re-ordering of shape and strides
  permute_dimensions(perm_);
}

Device compute_device(at::ArrayRef<OperandInfo> operands) {
  for (auto& op : operands) {
    if (!op.tensor.defined()) continue;
    if (op.tensor.dim() == 0) continue;
    return op.tensor.device();
  }
  for (auto& op : operands) {
    if (!op.tensor.defined()) continue;
    if (op.tensor.unsafeGetTensorImpl()->is_wrapped_number()) continue;
    return op.tensor.device();
  }
  return kCPU;
}

static std::tuple<Device, ScalarType, bool> compute_common_type_(at::ArrayRef<OperandInfo> operands) {
  // See [Result type computation] in TensorIterator.h
  auto device = compute_device(operands);
  auto common_type = ScalarType::Undefined;
  bool all_same_type = true;
  for (const auto& op: operands){
    if (!op.tensor.defined()) continue;
    //don't handle scalars
    if (op.tensor.dim() > 0){
      ScalarType current = op.current_dtype;
      if (current == ScalarType::Undefined){
        all_same_type = false;
        break;
      }
      if (common_type == ScalarType::Undefined) common_type = current;
      if (common_type != current) {
        all_same_type = false;
        break;
      }
    } else {
      all_same_type = false;
      break;
    }
  }
  if (all_same_type) {
    return std::make_tuple(device, common_type, true);
  }

  at::native::ResultTypeState state = {};
  for (const auto& op : operands) {
    state = at::native::update_result_type_state(op.tensor, state);
  }
  auto dtype = at::native::result_type(state);

  auto result = std::make_tuple(device, dtype, false);
  TORCH_INTERNAL_ASSERT(dtype != ScalarType::Undefined);
  return result;
}

std::tuple<Device, ScalarType, bool> TensorIterator::compute_common_type() {
  return compute_common_type_(operands_);
}

static void validate_dtype(OperandInfo& op, ScalarType common_dtype, CommonDTypeStrategy strategy) {
  bool check_output = strategy == CommonDTypeStrategy::CHECK || strategy == CommonDTypeStrategy::PROMOTE;
  bool promoting = strategy == CommonDTypeStrategy::PROMOTE || (strategy == CommonDTypeStrategy::PROMOTE_INPUTS && !op.is_output);
  if (op.tensor.defined()) {
    // For binary_ops, we follow casting rules. For unary/nullary types
    // we require the type to match.
    if (op.is_output && check_output) {
      TORCH_CHECK(canCast(common_dtype, op.current_dtype), "result type ", common_dtype,
          " can't be cast to the desired output type ",
          op.current_dtype);
    }
    TORCH_CHECK(promoting || op.target_dtype == op.current_dtype, "expected dtype ", op.target_dtype, " but got dtype ", op.current_dtype);
  }
}

static void maybe_copy_casting_to_common_dtype(OperandInfo& op, ScalarType common_dtype) {
  if (op.tensor.defined() && op.current_dtype != common_dtype)
  {
    op.target_dtype = common_dtype;
    op.original_tensor = op.tensor;
    if (!op.is_output) {
      op.tensor = op.tensor.to(common_dtype);
    } else {
      op.tensor =
          at::empty_like(op.tensor, op.tensor.options().dtype(common_dtype).memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT));
    }
    op.current_dtype = common_dtype;
  }
}

void TensorIterator::compute_types() {
  bool missing_dtypes = false;
  bool missing_output_dtypes = false;
  common_dtype_ = dtype();
  for (auto& op : operands_) {
    if (!op.tensor.defined() && !op.is_type_defined()) {
      missing_dtypes = true;
      if (op.is_output) {
        missing_output_dtypes = true;
      }
    }
  }

  if (common_dtype_strategy_ == CommonDTypeStrategy::PROMOTE_INPUTS) {
    TORCH_CHECK(!missing_output_dtypes, "unable to compute and promote common dtype based only on inputs if there are missing dtypes for outputs");
  }

  bool compute_common_dtype = (common_dtype_strategy_ != CommonDTypeStrategy::NONE);
  bool compute_common_dtype_only_for_inputs = (common_dtype_strategy_ == CommonDTypeStrategy::PROMOTE_INPUTS);

  bool may_have_differing_types = true;
  bool common_device_is_cuda = false;

  if (missing_dtypes || compute_common_dtype) {
    auto operands = compute_common_dtype_only_for_inputs ? at::ArrayRef<OperandInfo>(operands_).slice(noutputs()) : operands_;
    auto common_type = compute_common_type_(operands);
    auto common_device = std::get<0>(common_type);
    common_device_is_cuda = common_device.is_cuda();
    common_dtype_ = std::get<1>(common_type);
    may_have_differing_types = !std::get<2>(common_type);
    bool has_cpu_scalar = false;
    for (auto& op : operands_) {
      if (!op.is_type_defined()) {
        op.device = common_device;
        op.target_dtype = common_dtype_;
      } else if (compute_common_dtype &&
                 (op.device != common_device || op.target_dtype != common_dtype_)) {
        if (allow_cpu_scalars_ && op.tensor.defined() && op.tensor.dim() == 0 &&
            common_device_is_cuda && op.tensor.device().is_cpu() &&
            !has_cpu_scalar) {
          // don't cast CPU scalars in CUDA ops that directly support them.
          op.device = op.tensor.device();
          op.target_dtype = op.current_dtype;
          has_cpu_scalar = true;
        } else if (promote_gpu_output_dtypes_ && op.tensor.defined() &&
            !op.is_output &&
            op.current_dtype == kHalf && common_dtype_ == kFloat &&
            op.tensor.device().is_cuda() && common_device_is_cuda) {
          // allow input tensor type upcasting for fp16 to fp32 in fused kernel
          // on GPU
          op.device = op.tensor.device();
          op.target_dtype = op.current_dtype;
        } else {
          op.device = common_device;
          if (compute_common_dtype_only_for_inputs && op.is_output) {
            op.target_dtype = op.current_dtype;
          } else {
            op.target_dtype = common_dtype_;
          }
        }
      }
    }
  }

  for (auto &op : operands_) {
    bool skip_output = compute_common_dtype_only_for_inputs && op.is_output;
    bool is_different = op.tensor.defined() && op.current_dtype != common_dtype_;

    if (may_have_differing_types) {
      validate_dtype(op, common_dtype_, common_dtype_strategy_);
      bool cast_by_copy = compute_common_dtype && !common_device_is_cuda && !skip_output;
      if (cast_by_copy) {
        maybe_copy_casting_to_common_dtype(op, common_dtype_);
      }
    }

    if (op.tensor.defined() && op.device != op.tensor.device()) {
      if (op.is_output) {
        TORCH_CHECK(false, "output with device ", op.tensor.device(),
                  " doesn't match the desired device ", op.device);
      } else if (op.tensor.dim() == 0) {
        op.tensor = op.tensor.to(op.options());
      } else {
        TORCH_CHECK(false, "expected device ", op.device,
                  " but got device ", op.tensor.device());
      }
    }
  }
}

StrideVector TensorIterator::compatible_stride(int element_size) const {
  auto stride = StrideVector();
  int64_t next_stride = element_size;
  for (int dim = 0; dim < ndim(); dim++) {
    stride.push_back(next_stride);
    next_stride *= shape_[dim];
  }
  return stride;
}

DimVector TensorIterator::invert_perm(IntArrayRef input) const {
  // Invert the permutation caused by reorder_dimensions. This is not valid
  // after coalesce_dimensions is called.
  TORCH_INTERNAL_ASSERT(!has_coalesced_dimensions_);
  TORCH_INTERNAL_ASSERT(input.size()==perm_.size());
  auto res = DimVector(input.size()); //no initialization needed, every value in res should be written to.
  for (int dim = 0; dim < ndim(); dim++) {
    res[perm_[dim]] = input[dim];
  }
  return res;
}

DimVector TensorIterator::apply_perm_and_mul(IntArrayRef input, int mul) const {
  TORCH_INTERNAL_ASSERT(!has_coalesced_dimensions_);
  auto res = DimVector(input.size(), 0);
  for (size_t i = 0; i < input.size(); i++) {
    res[i] = input[perm_[i]] * mul;
  }
  return res;
}

void TensorIterator::allocate_outputs() {
  for (int i = 0; i < num_outputs_; i++) {
    auto& op = operands_[i];
    if (!op.tensor.defined()) {
      TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);
      int element_size = elementSize(op.target_dtype);
      if (requires_channels_last_output_ && ndim() == 4) {
        auto tensor_shape = invert_perm(shape_);
        op.tensor = at::empty(tensor_shape, op.options());
        op.tensor.unsafeGetTensorImpl()->empty_tensor_restride(
            MemoryFormat::ChannelsLast);
        // As we are allocating output after permutations is done, we need to
        // make sure that operand's strides are matching element size and
        // dimensions permutations which are stored in _perm
        op.stride_bytes = apply_perm_and_mul(op.tensor.strides(), element_size);
      } else {
        op.stride_bytes = compatible_stride(element_size);
        // check if permutation is just an inverted order
        bool inverted = true;
        for (int i = 1; i <= ndim(); i++) {
          if (perm_[i - 1] != ndim() - i) {
            inverted = false;
            break;
          }
        }
        auto tensor_shape = invert_perm(shape_);
        if (inverted) {
          // can just return contiguous output
          // it is faster because it avoids allocating 0 size tensor and
          // resizing and restriding it
          op.tensor = at::empty(tensor_shape, op.options());
        } else {
          auto tensor_stride = invert_perm(op.stride_bytes);
          for (int dim = 0; dim < ndim(); dim++) {
            tensor_stride[dim] /= element_size;
          }
          op.tensor =
              at::empty_strided(tensor_shape, tensor_stride, op.options());
        }
      }
      op.current_dtype = op.target_dtype;
    }
  }
}

void TensorIterator::compute_names() {
  bool should_infer_names = std::any_of(
      operands_.begin(),
      operands_.end(),
      [](const OperandInfo& op) {
        return op.tensor.defined() && op.tensor.has_names();
      });
  if (!should_infer_names) {
    return;
  }

  for (auto& op : operands_) {
    if (!op.tensor.defined()) continue;
    // don't include output tensors that are not also input tensors.
    if (resize_outputs_ && op.is_output && !op.is_read_write) continue;
    // perform name inference
    if (names_.empty()) {
      names_ = op.tensor.names();
    } else {
      names_ = NameVector(unify_from_right(names_, op.tensor.names()));
    }
  }
}

void TensorIterator::propagate_names_to_outputs() {
  // names_ can be empty for two reasons:
  // 1. We were performing ops on scalar tensors. Then there should be no names.
  // 2. All of the defined inputs/outputs had no names. Then we shouldn't
  //    run name inference.
  if (names_.empty()) {
    return;
  }

  // propagate names
  for (int i = 0; i < num_outputs_; i++) {
    auto& op = operands_[i];
    // must call propagate_names_to_outputs after outputs have been allocated.
    TORCH_INTERNAL_ASSERT(op.tensor.defined());
    if (!names_.empty()) {
      namedinference::propagate_names(op.tensor, names_);
    }
  }
}

void TensorIterator::coalesce_dimensions() {
  if (ndim() <= 1) {
    return;
  }

  // We can coalesce two adjacent dimensions if either dim has size 1 or if:
  // shape[n] * stride[n] == shape[n + 1].
  auto can_coalesce = [&](int dim0, int dim1) {
    auto shape0 = shape_[dim0];
    auto shape1 = shape_[dim1];
    if (shape0 == 1 || shape1 == 1) {
      return true;
    }
    for (int i = 0; i < ntensors(); i++) {
      auto& stride = operands_[i].stride_bytes;
      if (shape0 * stride[dim0] != stride[dim1]) {
        return false;
      }
    }
    return true;
  };

  // replace each operands stride at dim0 with its stride at dim1
  auto replace_stride = [&](int dim0, int dim1) {
    for (int i = 0; i < ntensors(); i++) {
      auto& stride = operands_[i].stride_bytes;
      stride[dim0] = stride[dim1];
    }
  };

  int prev_dim = 0;
  for (int dim = 1; dim < ndim(); dim++) {
    if (can_coalesce(prev_dim, dim)) {
      if (shape_[prev_dim] == 1) {
        replace_stride(prev_dim, dim);
      }
      shape_[prev_dim] *= shape_[dim];
    } else {
      prev_dim++;
      if (prev_dim != dim) {
        replace_stride(prev_dim, dim);
        shape_[prev_dim] = shape_[dim];
      }
    }
  }

  shape_.resize(prev_dim + 1);
  for (int i = 0; i < ntensors(); i++) {
    operands_[i].stride_bytes.resize(ndim());
  }
  has_coalesced_dimensions_ = true;
}

int64_t TensorIterator::numel() const {
  int64_t numel = 1;
  for (int64_t size : shape_) {
    numel *= size;
  }
  return numel;
}

StrideVector TensorIterator::get_dim_strides(int dim) const {
  auto dims = ndim();
  auto inner_strides = StrideVector();
  for (auto& op : operands_) {
    inner_strides.push_back(dims == 0 ? 0 : op.stride_bytes[dim]);
  }
  return inner_strides;
}

SmallVector<char*, 4> TensorIterator::get_data_ptrs(ArrayRef<char*> base, IntArrayRef counter) const {
  auto ptrs = SmallVector<char*, 4>(base);
  for (int dim = 0; dim < ndim(); dim++) {
    int64_t value = counter[dim];
    for (int arg = 0; arg < ntensors(); arg++) {
      ptrs[arg] += value * operands_[arg].stride_bytes[dim];
    }
  }
  return ptrs;
}

SmallVector<char*, 4> TensorIterator::get_base_ptrs() const {
  auto ptrs = SmallVector<char*, 4>();
  for (int i = 0; i < ntensors(); i++) {
    ptrs.push_back((char*)data_ptr(i));
  }
  return ptrs;
}

bool TensorIterator::is_dim_reduced(int dim) const {
  for (auto& op : operands_) {
    if (op.is_output && op.stride_bytes[dim] == 0 && shape_[dim] > 1) {
      return true;
    }
  }
  return false;
}

void TensorIterator::permute_dimensions(IntArrayRef perm) {
  TORCH_INTERNAL_ASSERT(perm.size() == ndim());

  auto reorder = [perm](IntArrayRef data) {
    auto res = DimVector(data.size(), 0);
    for (size_t i = 0; i < perm.size(); i++) {
      res[i] = data[perm[i]];
    }
    return res;
  };

  // Update shape and strides
  shape_ = reorder(shape_);
  for (auto& op : operands_) {
    if (op.stride_bytes.size() > 0) {
      op.stride_bytes = reorder(op.stride_bytes);
    }
  }
}

int64_t TensorIterator::num_output_elements() const {
  int64_t elem = 1;
  for (int dim = 0; dim < ndim(); dim++) {
    if (operands_[0].stride_bytes[dim] != 0 || shape_[dim] == 0)  {
      elem *= shape_[dim];
    }
  }
  return elem;
}

int TensorIterator::num_reduce_dims() const {
  int count = 0;
  for (int dim = 0; dim < ndim(); dim++) {
    if (operands_[0].stride_bytes[dim] == 0) {
      count++;
    }
  }
  return count;
}

#define LOOP_WRAPPER(ntensor, loop) \
  [=](char** base, const int64_t* strides, int64_t size0, int64_t size1) { \
    auto data = PtrVector(base, base + ntensor);                          \
    const int64_t* outer_strides = &strides[ntensor];                     \
                                                                          \
    for (int64_t i = 0; i < size1; i++) {                                 \
      if (i > 0) {                                                        \
        for (int arg = 0; arg < ntensor; arg++) {                         \
          data[arg] += outer_strides[arg];                                \
        }                                                                 \
      }                                                                   \
      loop(data.data(), strides, size0);                               \
    }                                                                     \
  }

void TensorIterator::for_each(loop_t loop) {
  for_each(LOOP_WRAPPER(ntensors(), loop));
}

void TensorIterator::for_each(loop2d_t loop) {
  int64_t numel = this->numel();
  if (numel == 0) {
    return;
  } else if (numel < internal::GRAIN_SIZE || at::get_num_threads() == 1) {
    return serial_for_each(loop, {0, numel});
  } else {
    at::parallel_for(0, numel, internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
      serial_for_each(loop, {begin, end});
    });
  }
}

StrideVector TensorIterator::get_strides() const {
  StrideVector strides;
  for (int dim = 0; dim < ndim(); dim++) {
    for (int arg = 0; arg < ntensors(); arg++) {
      strides.push_back(operands_[arg].stride_bytes[dim]);
    }
  }
  return strides;
}

void TensorIterator::serial_for_each(loop_t loop, Range range) const {
  serial_for_each(LOOP_WRAPPER(ntensors(), loop), range);
}

void TensorIterator::serial_for_each(loop2d_t loop, Range range) const {
  if (range.size() == 0) {
    return;
  }
  auto strides = get_strides();
  while (strides.size() < 2 * ntensors()) {
    strides.push_back(0);
  }

  auto base_ptrs = get_base_ptrs();
  if (ndim() <= 1) {
    auto ptrs = get_data_ptrs(base_ptrs, { range.begin });
    loop(ptrs.data(), strides.data(), range.size(), 1);
  } else {
    auto counter = DimCounter(shape_, range);
    while (!counter.is_done()) {
      auto ptrs = get_data_ptrs(base_ptrs, counter.values);
      auto step = counter.max_2d_step();
      loop(ptrs.data(), strides.data(), step[0], step[1]);
      counter.increment(step);
    }
  }
}

bool TensorIterator::is_trivial_1d() const {
  // TODO: check for casting once it's supported
  return ndim() == 1;
}

bool TensorIterator::is_contiguous() const {
  if (numel() == 1) {
    return true;
  }
  if (ndim() != 1) {
    return false;
  }
  return has_contiguous_first_dim();
}


bool TensorIterator::is_scalar(int arg) const {
  const auto& stride = operands_[arg].stride_bytes;
  for (int i = 0; i < ndim(); i++) {
    if (stride[i] != 0 && shape_[i] != 1) {
      return false;
    }
  }
  return true;
}

bool TensorIterator::is_cpu_scalar(int arg) const {
  return is_scalar(arg) && device(arg).is_cpu();
}

void* TensorIterator::data_ptr(int arg) const {
  return operands_[arg].data;
}

void TensorIterator::remove_operand(int arg) {
  operands_.erase(operands_.begin() + arg);
}

void TensorIterator::unsafe_replace_operand(int arg, void* data) {
  operands_[arg].data = data;
}

void TensorIterator::remove_dimension(int dim) {
  TORCH_INTERNAL_ASSERT(dim >= 0 && dim < ndim());
  shape_.erase(shape_.begin() + dim);
  for (auto& op : operands_) {
    op.stride_bytes.erase(op.stride_bytes.begin() + dim);
  }
}

void TensorIterator::narrow(int dim, int64_t start, int64_t size) {
  TORCH_INTERNAL_ASSERT(dim < ndim() && size >= 1);
  shape_[dim] = size;
  view_offsets_[dim] += start;
  for (auto& op : operands_) {
    op.data = ((char*)op.data) + op.stride_bytes[dim] * start;
  }
  if (size == 1) {
    coalesce_dimensions();
  }
}

void TensorIterator::select_all_keeping_dim(int start_dim, IntArrayRef indices) {
  TORCH_INTERNAL_ASSERT(start_dim <= ndim());
  for (int i = start_dim; i < ndim(); ++i) {
    for (auto& op : operands_) {
      op.data = ((char*)op.data) + op.stride_bytes[i] * indices[i - start_dim];
    }
    shape_[i] = 1;
  }
}

TensorIterator TensorIterator::binary_op(Tensor& out, const Tensor& a,
    const Tensor& b, bool check_mem_overlap) {
  auto iter = TensorIterator();
  iter.set_check_mem_overlap(check_mem_overlap);
  iter.add_output(out);
  iter.add_input(a);
  iter.add_input(b);
  iter.allow_cpu_scalars_ = true;
  iter.promote_common_dtype();
  iter.build();
  return iter;
}

TensorIterator TensorIterator::comparison_op(Tensor& out, const Tensor& a,
    const Tensor& b, bool check_mem_overlap) {
  auto iter = TensorIterator();
  iter.set_check_mem_overlap(check_mem_overlap);
  iter.add_output(out);
  iter.add_input(a);
  iter.add_input(b);
  iter.allow_cpu_scalars_ = true;
  iter.compute_common_dtype_only_for_inputs();
  iter.build();
  return iter;
}

TensorIterator TensorIterator::unary_op(Tensor& out, const Tensor& a,
    bool check_mem_overlap) {
  auto iter = TensorIterator();
  iter.set_check_mem_overlap(check_mem_overlap);
  iter.add_output(out);
  iter.add_input(a);
  iter.num_outputs_ = 1;
  iter.build();
  return iter;
}

TensorIterator TensorIterator::nullary_op(Tensor& out) {
  auto iter = TensorIterator();
  iter.add_output(out);
  // FIXME: workaround for bug: https://github.com/pytorch/pytorch/issues/20342
  iter.resize_outputs_ = false;
  iter.build();
  return iter;
}

TensorIterator TensorIterator::reduce_op(Tensor& out, const Tensor& a) {
  TORCH_INTERNAL_ASSERT(out.defined());
  auto iter = TensorIterator();
  iter.add_output(out);
  iter.add_input(a);
  iter.promote_gpu_output_dtypes_ = true;
  iter.resize_outputs_ = false;
  iter.is_reduction_ = true;
  // TODO: This is only really necessary for arg{min,max}
  iter.compute_common_dtype_only_for_inputs();
  iter.build();
  return iter;
}

TensorIterator TensorIterator::reduce_op(Tensor& out1, Tensor& out2, const Tensor& a) {
  TORCH_INTERNAL_ASSERT(out1.defined());
  TORCH_INTERNAL_ASSERT(out2.defined());
  TORCH_CHECK((!a.is_cuda() && !out1.is_cuda() && !out2.is_cuda()) || (a.device() == out1.device() && out1.device() == out2.device()),
      "reduce_op(): expected input and both outputs to be on same device, but input is on ", a.device(),
      ", output1 is on ", out1.device(), " and output2 is on", out2.device());
  TORCH_CHECK(out1.dim() == out2.dim(), "reduce_op(): expected both outputs to have same number of dims, but output1 has ", out1.dim(),
      " and output2 has ", out2.dim());
  TORCH_CHECK(out1.sizes() == out2.sizes(), "reduce_op(): expected both outputs to have same sizes, but output1 has ", out1.sizes(),
      " and output2 has ", out2.sizes());
  TORCH_CHECK(out1.strides() == out2.strides(), "reduce_op(): expected both outputs to have same strides, but output1 has ", out1.strides(),
           " and output2 has ", out2.strides());
  auto iter = TensorIterator();
  iter.add_output(out1);
  iter.add_output(out2);
  iter.add_input(a);
  iter.promote_gpu_output_dtypes_ = true;
  iter.resize_outputs_ = false;
  iter.is_reduction_ = true;
  iter.build();
  return iter;
}

void TensorIterator::mark_outputs() {
  for (int i = 0; i < num_outputs_; i++) {
    operands_[i].is_output = true;
    const auto& output = operands_[i].tensor;
    if (!output.defined()) continue;

    // check if output is also an input
    for (int arg = num_outputs_; arg < ntensors(); arg++) {
      const auto& input = operands_[arg].tensor;
      if (output.is_same(input)) {
        operands_[i].is_read_write = true;
      }
    }
  }
}

void TensorIterator::check_mem_overlaps() {
  if (!check_mem_overlap_) {
    return;
  }
  for (int i = 0; i < num_outputs_; i++) {
    const auto& output = operands_[i].tensor;
    if (!output.defined()) continue;
    assert_no_internal_overlap(output);
    for (int j = num_outputs_; j < ntensors(); j++) {
      const auto& input = operands_[j].tensor;
      assert_no_partial_overlap(output, input);
    }
  }
}

void TensorIterator::compute_shape() {
  all_ops_same_shape_ = true;
  bool has_scalars = false;
  bool has_tensors = false;
  for (auto& op : operands_) {
    if (!op.tensor.defined()) continue;

    // For now, don't include output tensors that are not also input tensors.
    // This preserves the legacy behavior where torch.add(..., out=dst) resizes
    // the destination tensor.
    if (resize_outputs_ && op.is_output && !op.is_read_write) continue;
    auto shape = op.tensor.sizes();
    if (shape.size() == 0) {
      has_scalars = true;
    } else {
      has_tensors = true;
    }
    if (has_scalars && has_tensors) {
      all_ops_same_shape_ = false;
    }
    if (shape_.empty()) {
      shape_ = shape;
    } else if (!shape.equals(shape_)) {
      all_ops_same_shape_ = false;
      shape_ = DimVector(infer_size(shape_, shape));
    }
  }
  // Outputs cannot be broadcasted. Check that the shape of the outputs matches
  // the inferred shape. There's an exception for write-only tensors to support
  // our legacy behavior that functions with `out=` arguments resize their
  // outputs.
  for (int i = 0; i < num_outputs_; i++) {
    auto& tensor = operands_[i].tensor;
    if (tensor.defined() && !tensor.sizes().equals(shape_)) {
      if (resize_outputs_ && !operands_[i].is_read_write) {
        // Preserve legacy resizing behavior of out=... arguments
        // TODO: issue warning
        tensor.resize_(shape_);
        if (requires_channels_last_output_ && tensor.dim() == 4) {
          // Temporary stick to 4d tensor, will update with arbitrary batched later on
          tensor.unsafeGetTensorImpl()->empty_tensor_restride(
              MemoryFormat::ChannelsLast);
        }
        continue;
      }
      TORCH_CHECK(is_reduction_, "output with shape ", tensor.sizes(), " doesn't match the broadcast shape ",
                 shape_);
    }
  }
}

void TensorIterator::compute_strides() {
  for (auto& op : operands_) {
    if (op.tensor.defined()) {
      auto original_shape = op.tensor.sizes();
      auto original_stride = op.tensor.strides();
      auto element_size_in_bytes = op.tensor.element_size();
      auto offset = ndim() - original_shape.size();
      if (offset > 0)
          op.stride_bytes.resize(ndim(), 0);
      else
          op.stride_bytes.resize(ndim());
      for (size_t i = 0; i < original_shape.size(); i++) {
        if (original_shape[i] == 1) {
          op.stride_bytes[offset + i] = 0;
        } else {
          op.stride_bytes[offset + i] = original_stride[i] * element_size_in_bytes;
        }
      }
    }
  }
}

void TensorIterator::analyze_memory_format() {
  for (auto& op : operands_) {
    if (op.tensor.defined() &&
        op.tensor.suggest_memory_format() == MemoryFormat::ChannelsLast) {
      requires_channels_last_output_ = true;
    }
  }
}

bool TensorIterator::can_use_32bit_indexing() const {
  int64_t max_value = std::numeric_limits<int32_t>::max();
  if (numel() > max_value) {
    return false;
  }
  for (auto& op : operands_) {
    int64_t max_offset = 1;
    for (int dim = 0; dim < ndim(); dim++) {
      max_offset += (shape_[dim] - 1) * op.stride_bytes[dim];
    }
    if (max_offset > max_value) {
      return false;
    }
  }
  return true;
}

std::unique_ptr<TensorIterator> TensorIterator::split(int dim) {
  TORCH_INTERNAL_ASSERT(dim >= 0 && dim < ndim() && shape()[dim] >= 2);
  std::unique_ptr<TensorIterator> copy(new TensorIterator(*this));

  bool overlaps = is_dim_reduced(dim);
  auto copy_size = shape_[dim] / 2;
  auto this_size = shape_[dim] - copy_size;
  copy->narrow(dim, 0, copy_size);
  copy->final_output_ &= !overlaps;
  this->narrow(dim, copy_size, this_size);
  this->accumulate_ |= overlaps;

  return copy;
}

int TensorIterator::get_dim_to_split() const {
  TORCH_INTERNAL_ASSERT(ndim() >= 1 && shape()[ndim() - 1] >= 2);
  int64_t max_extent = -1;
  int dim_to_split = -1;
  for (int dim = ndim() - 1; dim >= 0; dim--) {
    int64_t size = shape_[dim];
    for (auto& op : operands_) {
      int64_t extent = (size - 1) * op.stride_bytes[dim];
      if (extent > max_extent) {
        max_extent = extent;
        dim_to_split = dim;
      }
    }
  }
  TORCH_INTERNAL_ASSERT(max_extent >= 0);
  return dim_to_split;
}

bool TensorIterator::fast_set_up() {
  // This function tries to do a fast setup to avoid needless reordering of dimensions and tracking output strides
  // Return true if it can do fast setup or false otherwise
  // TODO enable fast handling for reductions
  FastSetupType setup_type = compute_fast_setup_type();
  if (setup_type == FastSetupType::NONE) {
    return false;
  }

  // allocate memory for output, memory format depends on setup_type
  switch (setup_type) {
    case FastSetupType::CONTIGUOUS:
      {
        for (int i = 0; i < num_outputs_; i++){
          auto& op = operands_[i];
          if (!op.tensor.defined()) {
            TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);
            op.tensor = at::empty(shape_, op.options().memory_format(MemoryFormat::Contiguous));
            op.current_dtype = op.target_dtype;
          }
        }
        break;
      }
    case FastSetupType::CHANNELS_LAST:
      {
        for (int i = 0; i < num_outputs_; i++){
          auto& op = operands_[i];
          if (!op.tensor.defined()) {
            TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);
            op.tensor = at::empty(shape_, op.options().memory_format(MemoryFormat::ChannelsLast));
            op.current_dtype = op.target_dtype;
          }
        }
        break;
      }
    case FastSetupType::NON_OVERLAPPING_DENSE:
      {
        // find the index of a defined tensor in operands_ start from input tensor
        int i_defined;
        for (i_defined = ntensors() - 1; i_defined >= 0; --i_defined) {
          if (operands_[i_defined].tensor.defined()) break;
        }
        TORCH_CHECK(i_defined >= 0, "Can not find a defined tensor when fast allocating memory to outputs");
        for (int i = 0; i < num_outputs_; i++){
          auto& op = operands_[i];
          if (!op.tensor.defined()) {
            TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);
            op.tensor = at::empty_strided(shape_, operands_[i_defined].tensor.strides(), op.options());
            op.current_dtype = op.target_dtype;
          }
          else if (resize_outputs_ && !op.is_read_write) {
            // Note: This restride logic is for some of the tensor types, eg. qtensor.
            //       These tensor class allocate a contigious output tensor directly no matter
            //       what memory format the input tensors are. So we need to restride output here for fast setup.
            //       We will revisit this logic when doing the TensorIterator refactor work and
            //       it would probably be better to move this logic out of TensorIterator.

            // Check whether output tensor needs restride, output's stride can be different than input tensors
            if (i != i_defined && !op.tensor.strides().equals(operands_[i_defined].tensor.strides())) {
              op.tensor.as_strided_(op.tensor.sizes(), operands_[i_defined].tensor.strides());
            }
          }
        }
        break;
      }
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported fast setup type", std::to_string((int)setup_type));
  }
  //coalescing dimensions consists of collapsing dimensions to 1 (we are limited to contiguous no-broadcast cases here)
  if (ndim() > 1){
    has_coalesced_dimensions_ = true;
  }
  if (ndim() >= 1) {
    shape_[0] = numel();
    shape_.resize(1);
  }
  for (auto& op : operands_ ) {
    auto element_size_in_bytes = op.tensor.element_size();
    op.stride_bytes.resize(ndim());
    if (ndim()>0) {
      op.stride_bytes[0] = element_size_in_bytes;
    }
  }
  return true;
}

FastSetupType TensorIterator::compute_fast_setup_type() {
  if (is_reduction_ || !all_ops_same_shape_) {
    return FastSetupType::NONE;
  }

  bool is_contiguous = true;
  bool is_channels_last = true;
  bool is_non_overlapping_and_dense = true;
  for (const auto& op : operands_) {
    if (op.tensor.defined()) {
      is_contiguous &= op.tensor.is_contiguous(at::MemoryFormat::Contiguous);
      is_channels_last &= op.tensor.is_contiguous(at::MemoryFormat::ChannelsLast);
      is_non_overlapping_and_dense &= op.tensor.is_non_overlapping_and_dense();
    }
  }

  if (is_contiguous) {
    return FastSetupType::CONTIGUOUS;
  }
  if (is_channels_last) {
    return FastSetupType::CHANNELS_LAST;
  }
  if (is_non_overlapping_and_dense) {
    int64_t prev = -1;
    // iterate from back to favor inputs' strides, then we check output strides.
    // if only the output's strides is inconstent but all inputs' strides are equal,
    // it is still a NON_OVERLAPPING_DENSE case and output will be restrided in fast_set_up()
    for (int64_t i = ntensors() - 1; i >= 0; --i) {
      const auto& op = operands_[i];
      if (op.tensor.defined()) {
        if (prev < 0) {
          prev = i;
          continue;
        }
        if (!operands_[prev].tensor.strides().equals(op.tensor.strides())) {
          if (!resize_outputs_ || !op.is_output || op.is_read_write) {
            return FastSetupType::NONE;
          }
        }
      }
    }
    return FastSetupType::NON_OVERLAPPING_DENSE;
  }
  return FastSetupType::NONE;
}

void TensorIterator::build() {
  // check input tensors memory format to use it during output allocation
  analyze_memory_format();
  // set is_output and is_read_write flags on appropriate tensors
  mark_outputs();
  // Check that the outputs have no internal overlap
  // and do not share memory with inputs.
  check_mem_overlaps();
  // Check that input dimensions are aligned correctly & compute outnames.
  compute_names();
  // compute the broadcasted shape
  compute_shape();
  // compute the result dtype and device
  compute_types();
  // try fast setup output tensor, if failed, fallback to normal setup
  if (!fast_set_up()) {
    // compute each tensor's stride after broadcasting
    compute_strides();
    // re-order dimensions to improve coalescing
    reorder_dimensions();
    // allocate the output tensor if it's not provided
    allocate_outputs();
    // coalesce adjacent dimensions when possible
    coalesce_dimensions();
  }
  // perform name inference
  propagate_names_to_outputs();

  for (auto& op : operands_) {
    TORCH_INTERNAL_ASSERT(op.tensor.defined());
    op.data = op.tensor.data_ptr();
  }

  // zero out offsets
  view_offsets_ = DimVector(ndim(), 0);
}

SplitUntil32Bit TensorIterator::with_32bit_indexing() const {
  return SplitUntil32Bit(*this);
}

/// SplitUntil32Bit. Recursively splits an iterator into sub-iterators that
/// can use 32-bit indexing.

SplitUntil32Bit::iterator::iterator(const TensorIterator& iter) {
  vec.emplace_back(new TensorIterator(iter));
  vec.emplace_back(nullptr); // ++ first pops the last element
  ++(*this);
}

SplitUntil32Bit::iterator& SplitUntil32Bit::iterator::operator++() {
  vec.pop_back();
  while (!vec.empty() && !vec.back()->can_use_32bit_indexing()) {
    auto& iter = *vec.back();
    int64_t split_dim = iter.get_dim_to_split();
    vec.emplace_back(iter.split(split_dim));
  }
  return *this;
}

TensorIterator& SplitUntil32Bit::iterator::operator*() const {
  return *vec.back();
}

SplitUntil32Bit::iterator SplitUntil32Bit::begin() const {
  return SplitUntil32Bit::iterator(iter);
}

SplitUntil32Bit::iterator SplitUntil32Bit::end() const {
  return SplitUntil32Bit::iterator();
}

DimCounter::DimCounter(IntArrayRef shape, Range range)
  : shape(shape)
  , range(range)
  , values(shape.size(), 0)
  , offset(range.begin) {
  int64_t linear_offset = range.begin;
  int64_t ndim = values.size();
  for (int dim = 0; dim < ndim; dim++) {
    int64_t size = shape[dim];
    if (size > 0) {
      values[dim] = linear_offset % size;
      linear_offset /= size;
    }
  }
  TORCH_INTERNAL_ASSERT(linear_offset == 0);
}

bool DimCounter::is_done() const {
  return offset >= range.end;
}

void DimCounter::increment(const std::array<int64_t, 2>& step) {
  offset += step[0] * step[1];
  int64_t ndim = values.size();
  int64_t overflow = step[0];
  int i = 0;
  if (step[1] != 1) {
    TORCH_INTERNAL_ASSERT(step[0] == shape[0] && values[0] == 0);
    i = 1;
    overflow = step[1];
  }
  for (; i < ndim && overflow > 0; i++) {
    auto size = shape[i];
    auto prev = values[i];
    auto value = prev + overflow;
    if (value >= size) {
      overflow = 1;
      value -= size;
      TORCH_INTERNAL_ASSERT(value < size);
    } else {
      overflow = 0;
    }
    values[i] = value;
  }
  TORCH_INTERNAL_ASSERT(overflow == 0 || overflow == 1);
}

std::array<int64_t, 2> DimCounter::max_2d_step() const {
  int64_t step0 = std::min(shape[0] - values[0], range.end - offset);
  int64_t step1 = 1;
  if (step0 == shape[0] && shape.size() >= 1) {
    step1 = std::min(shape[1] - values[1], (range.end - offset) / shape[0]);
  }
  return {step0, step1};
}

}  // namespace at
