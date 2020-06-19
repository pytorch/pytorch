#include <ATen/native/TensorIterator.h>

#include <array>
#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/MemoryOverlap.h>

namespace at {

using DimMask = TensorIterator::DimMask;
using PtrVector = TensorIterator::PtrVector;
using loop_t = TensorIterator::loop_t;
using loop2d_t = TensorIterator::loop2d_t;
using StrideVector = TensorIterator::StrideVector;

void TensorIterator::reorder_dimensions(const TensorIteratorConfig& config) {
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

// Computes a common dtype using type promotion
// See the [Common Dtype Computation] note
ScalarType TensorIterator::compute_common_dtype() {
  at::native::ResultTypeState state = {};
  for (const auto& op : operands_) {
    if (op.is_output) {
      continue;
    }

    state = at::native::update_result_type_state(op.tensor, state);
  }

  common_dtype_ = at::native::result_type(state);
  TORCH_INTERNAL_ASSERT(common_dtype_ != ScalarType::Undefined);

  return common_dtype_;
}

// Implements the the behavior of the following flags:
//   - check_all_same_dtype_
//   - check_all_same_device_
//   - enforce_safe_casting_to_output_
//   - promote_inputs_to_common_dtype_
//   - cast_common_dtype_to_outputs_
//
// See their descriptions in TensorIterator.h for details.
// NOTE: Checks for more specific behaviors (e.g. the first and second
//   inputs must share a dtype, but the third must have the long dtype)
//   should be implemented directly and outside of TensorIterator.
void TensorIterator::compute_types(const TensorIteratorConfig& config) {
  // Reviews operands (1/2)
  //   - validates that all input tensors are defined
  //   - computes common device
  //   - determines if there are undefined outputs
  //   - determines if there are different dtypes and attempts
  //       to quickly acquire a common dtype
  Device common_device = kCPU;
  common_dtype_ = ScalarType::Undefined;
  // NB: despite output_dtype's generic sounding name, it only is
  // used in a nontrivial way if check_all_same_dtype is true
  ScalarType output_dtype = ScalarType::Undefined;
  bool has_different_input_dtypes = false;
  bool has_different_output_dtypes = false;
  bool has_undefined_outputs = false;

  for (auto& op : operands_) {
    // Validates that all inputs have type information, and that
    //   if an output is missing type information that we can infer
    //   the device it should be allocated on.
    if (!op.is_type_defined()) {
      TORCH_INTERNAL_ASSERT(op.is_output, "Found type undefined input tensor!");
      if (config.static_dtype_and_device_.has_value()) {
        op.target_dtype = config.static_dtype_and_device_->first;
        op.device = config.static_dtype_and_device_->second;
      } else {
        TORCH_INTERNAL_ASSERT(config.check_all_same_device_);
        has_undefined_outputs = true;
        continue;
      }
    }

    // Validates input tensors are defined
    if (!op.tensor.defined()) {
      TORCH_INTERNAL_ASSERT(op.is_output, "Found undefined input tensor!");
      continue;
    }

    TORCH_INTERNAL_ASSERT(op.target_dtype == op.current_dtype)

    // Acquires the first non-CPU device (if any) as the common device
    if (common_device == kCPU && !op.tensor.device().is_cpu()) {
      common_device = op.tensor.device();
    }

    // Determines if there are varying input dtypes
    // NOTE: the common dtype is set to the first defined input dtype observed
    if (!op.is_output && op.target_dtype != common_dtype_) {
      if (common_dtype_ == ScalarType::Undefined) {
        common_dtype_ = op.target_dtype;
      } else {
        has_different_input_dtypes = true;
      }
    } else if (op.is_output && op.target_dtype != common_dtype_) {
      if (output_dtype == ScalarType::Undefined) {
        output_dtype = op.target_dtype;
      } else {
        has_different_output_dtypes = true;
      }
    }
  }

  // Checks that either the computation type is computable or unneeded
  TORCH_INTERNAL_ASSERT(!(has_different_input_dtypes && !config.promote_inputs_to_common_dtype_ &&
                        (has_undefined_outputs || config.enforce_safe_casting_to_output_ ||
                        config.cast_common_dtype_to_outputs_)));

  // Checks that all inputs and defined outputs are the same dtype, if requested
  if (config.check_all_same_dtype_ &&
      (has_different_input_dtypes || has_different_output_dtypes ||
      (common_dtype_ != output_dtype && output_dtype != ScalarType::Undefined))) {
    // Throws an informative error message
    for (auto& op : operands_) {
      if (!op.tensor.defined()) {
        continue;
      }

      TORCH_CHECK(op.target_dtype == common_dtype_,
                  "Found dtype ", op.target_dtype, " but expected ", common_dtype_);
    }
  }

  // Short-circuits if no additional work required
  if (!has_undefined_outputs && !config.check_all_same_device_ &&
      !config.promote_inputs_to_common_dtype_ && !config.cast_common_dtype_to_outputs_ &&
      !config.enforce_safe_casting_to_output_) {
    // Invalidates common_dtype_ if it could not be inferred
    common_dtype_ = has_different_input_dtypes ? ScalarType::Undefined : common_dtype_;
    return;
  }

  // Computes a common dtype, if needed
  if (has_different_input_dtypes && config.promote_inputs_to_common_dtype_) {
    common_dtype_ = compute_common_dtype();
  }

  // Reviews operands (2/2)
  //   - sets metadata for undefined outputs
  //   - checks that all tensors are on the same device, if requested
  //   - checks that the common dtype can safely cast to each output, if requested
  //   - creates temporaries for CPU operations, if needed and requested
  int max_cpu_scalars_on_cuda = config.allow_cpu_scalars_ ? 1 : 0;
  int current_cpu_scalars_on_cuda = 0;
  for (auto& op : operands_) {
    if (!op.is_type_defined()) {
      op.target_dtype = common_dtype_;
      op.device = common_device;
      continue;
    }

    // Skips undefined tensors
    if (!op.tensor.defined()) {
      continue;
    }

    // Checks all tensors are on the same device, if requested
    if (config.check_all_same_device_) {
      // Handles CPU scalars on CUDA kernels that support them
      if (common_device.is_cuda() && op.tensor.dim() == 0 && op.tensor.device().is_cpu()) {
        TORCH_CHECK(current_cpu_scalars_on_cuda < max_cpu_scalars_on_cuda,
                    "Trying to pass too many CPU scalars to CUDA kernel!");
        ++current_cpu_scalars_on_cuda;
      } else if (op.device != common_device) {
        TORCH_CHECK(false,
                    "Expected all tensors to be on the same device, but "
                    "found at least two devices, ", common_device, " and ", op.device, "!");
      }
    }

    // Checks safe casting, if requested
    if (config.enforce_safe_casting_to_output_ && op.is_output && op.current_dtype != common_dtype_) {
      TORCH_CHECK(canCast(common_dtype_, op.current_dtype),
                  "result type ", common_dtype_, " can't be cast to the "
                  "desired output type ", op.current_dtype);
    }

    // Creates temporaries for CPU operations, if needed and requested
    // TODO: reuse temporaries when possible (e.g. for inplace operations)
    if (common_device == kCPU) {
      // Casts to outputs by creating temporaries of the correct dtype (if needed)
      if (config.cast_common_dtype_to_outputs_ && op.is_output && op.current_dtype != common_dtype_) {
        op.original_tensor = op.tensor;
        op.tensor = at::empty_like(op.tensor,
                                   op.tensor.options().dtype(common_dtype_),
                                   LEGACY_CONTIGUOUS_MEMORY_FORMAT);
        op.current_dtype = common_dtype_;
    }

    // Promotes inputs by creating temporaries of the correct dtype
      if (config.promote_inputs_to_common_dtype_ && !op.is_output && op.current_dtype != common_dtype_) {
        op.original_tensor = op.tensor;
        op.tensor = op.tensor.to(common_dtype_);
        op.current_dtype = common_dtype_;
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
      if ((requires_channels_last_output_ && ndim() == 4) ||
          (requires_channels_last_3d_output_ && ndim() == 5)) {
        auto tensor_shape = invert_perm(shape_);
        op.tensor = at::empty(tensor_shape, op.options());
        if (requires_channels_last_output_) {
          op.tensor.unsafeGetTensorImpl()->empty_tensor_restride(MemoryFormat::ChannelsLast);
        } else {
          op.tensor.unsafeGetTensorImpl()->empty_tensor_restride(MemoryFormat::ChannelsLast3d);
        }
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

void TensorIterator::compute_names(const TensorIteratorConfig& config) {
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
    // Don't include output tensors if we are resizing, since we will
    // clobber their names in any case.  (If the output tensor was
    // also an input tensor, we'll pick it up when it shows up again
    // in operands).
    if (config.resize_outputs_ && op.is_output) continue;
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

void TensorIterator::for_each(loop_t loop, int64_t grain_size) {
  for_each(LOOP_WRAPPER(ntensors(), loop), grain_size);
}

void TensorIterator::for_each(loop2d_t loop, int64_t grain_size) {
  int64_t numel = this->numel();
  if (numel == 0) {
    return;
  } else if (numel < internal::GRAIN_SIZE || at::get_num_threads() == 1) {
    return serial_for_each(loop, {0, numel});
  } else {
    at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
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

void TensorIterator::cast_outputs() {
  for (auto& op : operands_) {
    if (op.is_output && op.original_tensor.defined() &&
        op.original_tensor.scalar_type() != op.current_dtype) {
      op.original_tensor.copy_(op.tensor);
      op.tensor = op.original_tensor;
    }
  }
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

void TensorIterator::narrow(int dim, int64_t start, int64_t size) {
  TORCH_INTERNAL_ASSERT(dim < ndim() && size >= 1);
  shape_[dim] = size;
  view_offsets_[dim] += start;
  for (auto& op : operands_) {
    op.data = ((char*)op.data) + op.stride_bytes[dim] * start;
  }
  if (size == 1 && !is_reduction_) {
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
  return TensorIteratorConfig()
     .set_check_mem_overlap(check_mem_overlap)
     .add_output(out)
     .add_input(a)
     .add_input(b)
     .allow_cpu_scalars(true)
     .promote_inputs_to_common_dtype(true)
     .cast_common_dtype_to_outputs(true)
     .enforce_safe_casting_to_output(true)
     .build();
}

TensorIterator TensorIterator::comparison_op(Tensor& out, const Tensor& a,
    const Tensor& b, bool check_mem_overlap) {
  return TensorIteratorConfig()
    .set_check_mem_overlap(check_mem_overlap)
    .add_output(out)
    .add_input(a)
    .add_input(b)
    .allow_cpu_scalars(true)
    .promote_inputs_to_common_dtype(true)
    .build();
}

TensorIterator TensorIterator::unary_op(Tensor& out, const Tensor& a,
    bool check_mem_overlap) {
  return TensorIteratorConfig()
    .set_check_mem_overlap(check_mem_overlap)
    .add_output(out)
    .add_input(a)
    .cast_common_dtype_to_outputs(true)
    .enforce_safe_casting_to_output(true)
    .build();
}

TensorIterator TensorIterator::nullary_op(Tensor& out) {
  return TensorIteratorConfig()
    .check_all_same_dtype(false)
    .add_output(out)
    // FIXME: workaround for bug: https://github.com/pytorch/pytorch/issues/20342
    .dont_resize_outputs()
    .build();
}

TensorIterator TensorIterator::reduce_op(Tensor& out, const Tensor& a) {
  TORCH_INTERNAL_ASSERT(out.defined());
  return TensorIteratorConfig()
    .add_output(out)
    .add_input(a)
    .dont_resize_outputs()
    .is_reduction(true)
    // TODO: not supporting casting to outputs is only really necessary for arg{min,max}
    .promote_inputs_to_common_dtype(true)
    .build();
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
  return TensorIteratorConfig()
    .add_output(out1)
    .add_output(out2)
    .add_input(a)
    .dont_resize_outputs()
    .is_reduction(true)
    .check_all_same_dtype(false)
    .build();
}

void TensorIterator::populate_operands(TensorIteratorConfig& config) {
  for (int i = 0; i < config.tensors_.size(); i++) {
    operands_.emplace_back(std::move(config.tensors_[i]));
  }
  num_outputs_ = config.num_outputs_;
}

void TensorIterator::mark_outputs() {
  // TODO: merge this into populate_operands
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

void TensorIterator::compute_mem_overlaps(const TensorIteratorConfig& config) {
  if (!config.check_mem_overlap_) {
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

void TensorIterator::compute_shape(const TensorIteratorConfig& config) {
  if (config.static_shape_.has_value()) {
    shape_ = *config.static_shape_;
    return;
  }

  all_ops_same_shape_ = true;
  bool has_scalars = false;
  bool has_tensors = false;
  for (auto& op : operands_) {
    if (!op.tensor.defined()) continue;

    // For now, don't include output tensors when we're resizing outputs.
    // These shapes don't participate in shape computation.
    // This preserves the legacy behavior where torch.add(..., out=dst) resizes
    // the destination tensor.  If the output tensor is also an input, we'll
    // pick it up later in the operands.
    if (config.resize_outputs_ && op.is_output) continue;
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
      if (config.resize_outputs_ && !operands_[i].is_read_write) {
        // Preserve legacy resizing behavior of out=... arguments
        // TODO: issue warning
        tensor.resize_(shape_);
        if (requires_channels_last_output_ && tensor.dim() == 4) {
          // Temporary stick to 4d tensor, will update with arbitrary batched later on
          tensor.unsafeGetTensorImpl()->empty_tensor_restride(MemoryFormat::ChannelsLast);
        }
        else if (requires_channels_last_3d_output_ && tensor.dim() == 5) {
          // Temporary stick to 5d tensor, will update with arbitrary batched later on
          tensor.unsafeGetTensorImpl()->empty_tensor_restride(MemoryFormat::ChannelsLast3d);
        }
        continue;
      }
      TORCH_CHECK(is_reduction_, "output with shape ", tensor.sizes(), " doesn't match the broadcast shape ",
                 shape_);
    }
  }
}

void TensorIterator::compute_strides(const TensorIteratorConfig& config) {
  for (auto& op : operands_) {
    if (op.tensor.defined()) {
      IntArrayRef original_shape = config.static_shape_ ? shape_ : op.tensor.sizes();
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
    if (op.tensor.defined()) {
      if (!requires_channels_last_output_ &&
          op.tensor.suggest_memory_format() == MemoryFormat::ChannelsLast) {
        requires_channels_last_output_ = true;
      }
      else if (!requires_channels_last_3d_output_ &&
          op.tensor.suggest_memory_format() == MemoryFormat::ChannelsLast3d) {
        requires_channels_last_3d_output_ = true;
      }
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
  TORCH_INTERNAL_ASSERT(ndim() >= 1);
  int64_t max_extent = -1;
  int dim_to_split = -1;
  for (int dim = ndim() - 1; dim >= 0; dim--) {
    if (shape_[dim] == 0) {
      continue;
    }
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

bool TensorIterator::fast_set_up(const TensorIteratorConfig& config) {
  // This function tries to do a fast setup to avoid needless reordering of dimensions and tracking output strides
  // Return true if it can do fast setup or false otherwise
  // TODO enable fast handling for reductions
  FastSetupType setup_type = compute_fast_setup_type(config);
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
            op.tensor = at::empty(shape_, op.options(), MemoryFormat::Contiguous);
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
            op.tensor = at::empty(shape_, op.options(), MemoryFormat::ChannelsLast);
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
          // defined tensors always have the same shape and strides here, no re-stride outputs happens.
          // see [Note: stride check for non contiguous tensors in fast setup]
        }
        break;
      }
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported fast setup type", c10::to_string((int)setup_type));
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

FastSetupType TensorIterator::compute_fast_setup_type(const TensorIteratorConfig& config) {
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
    // Fast setup is allowed only when all the defined tensors have the same shape and strides,
    // Iterate from back to check input tensors' strides first, then output tensors'.
    for (int64_t i = ntensors() - 1; i >= 0; --i) {
      const auto& op = operands_[i];
      if (op.tensor.defined()) {
        if (prev < 0) {
          prev = i;
          continue;
        }
        if (!operands_[prev].tensor.strides().equals(op.tensor.strides())) {
          // [Note: stride check for non contiguous tensors in fast setup]
          // We prevent 3 cases doing fast setup here:
          // 1. input tensors have different strides.
          // 2. output tensors have different strides.
          // 3. input tensors have the same strides, but output tensors have different strides with input tensors.
          //    We don't allow re-stride output tensors in this case since it is not compatible with
          //    numpy. The behavior in numpy is that if the output tensor has same shape as the input
          //    tensor but different strides, the strides of output tensor will be preserved, so we do
          //    the same in tensor iterator.
          return FastSetupType::NONE;
        }
      }
    }
    return FastSetupType::NON_OVERLAPPING_DENSE;
  }
  return FastSetupType::NONE;
}

TensorIterator::TensorIterator(TensorIteratorConfig& config) {
  build(config);
}

void TensorIterator::build(TensorIteratorConfig& config) {
  // populate some persistent configuration fields
  is_reduction_ = config.is_reduction_;

  // fill in operands_ based on configuration
  populate_operands(config);
  // check input tensors memory format to use it during output allocation
  analyze_memory_format();
  // set is_output and is_read_write flags on appropriate tensors
  mark_outputs();
  // Check that the outputs have no internal overlap
  // and do not share memory with inputs.
  compute_mem_overlaps(config);
  // Check that input dimensions are aligned correctly & compute outnames.
  compute_names(config);
  // compute the broadcasted shape
  compute_shape(config);
  // compute the result dtype and device
  compute_types(config);
  // try fast setup output tensor, if failed, fallback to normal setup
  if (!fast_set_up(config)) {
    // compute each tensor's stride after broadcasting
    compute_strides(config);
    // re-order dimensions to improve coalescing
    reorder_dimensions(config);
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
  // If the tensor is a scalar, we leave room for it
  // So index translations in reduction can access
  // a valid value for the offset
  int64_t ndim_offsets = (ndim() ? ndim() : 1);
  view_offsets_ = DimVector(ndim_offsets, 0);
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
