#include <ATen/BinaryTensorIterator.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/Parallel.h>
#include <ATen/native/TypeProperties.h>
#include <c10/util/Exception.h>

namespace at {

using StrideVector = TensorIteratorBase::StrideVector;
using PtrVector = TensorIteratorBase::PtrVector;

void BinaryTensorIteratorBase::setup(
    const Tensor& a,
    const Tensor& b,
    const Tensor& out,
    const TensorIteratorConfig& config) {
  TORCH_INTERNAL_ASSERT(
      !(a.is_meta() || b.is_meta() || out.is_meta()), "not supported -_-");

  TORCH_INTERNAL_ASSERT(
      a.defined() && b.defined(), "Input tensors has to be defined");

  a_ = a;
  b_ = b;
  out_ = out;

  // TODO: instead of reverse engineering function convention, get it from
  // structural kernel
  if (!out_.defined()) {
    convention_ = FunctionConvention::FUNCTIONAL;
  } else {
    if (a_.is_same(out_)) {
      convention_ = FunctionConvention::INPLACE;
    } else {
      convention_ = FunctionConvention::OUT;
    }
  }

  // ===================================================================

  TORCH_INTERNAL_ASSERT(!config.is_reduction_, "not supported -_-");
  TORCH_INTERNAL_ASSERT(!config.static_shape_, "not supported -_-");
  TORCH_INTERNAL_ASSERT(!config.static_dtype_and_device_, "not supported -_-");

  // ===================== Memory Overlap Check ==================
  if (config.check_mem_overlap_) {
    switch (convention_) {
      case FunctionConvention::OUT:
        assert_no_partial_overlap(a_, out_);
        assert_no_partial_overlap(b_, out_);
        // fall through
      case FunctionConvention::INPLACE:
        // TODO: if out is contiguous, this check can be skipped.
        assert_no_internal_overlap(out_);
        // fall through
      case FunctionConvention::FUNCTIONAL:;
        // nothing to do
    }
  }

  // =================== Compute Names ==============================
  bool should_infer_names = a_.has_names() || b_.has_names() || (out_.defined() && out_.has_names());
  if (should_infer_names) {
    TORCH_INTERNAL_ASSERT(false, "infer names is not not supported yet -_-");
  }

  // ==================== Shape Compute && Check Output Size && Numel
  // =============================
  if (a_.sizes().equals(b_.sizes())) {
    shape_ = a_.sizes();
    input_needs_broadcast_ = false;
  } else {
    shape_ = infer_size_dimvector(a_.sizes(), b_.sizes());
    input_needs_broadcast_ = true;
  }

  switch (convention_) {
    case FunctionConvention::OUT:
      if (config.resize_outputs_) {
        output_needs_resize_ = !out_.sizes().equals(shape_);
        break;
      }
      // if not resize output,
      // fall through
    case FunctionConvention::INPLACE:
      TORCH_INTERNAL_ASSERT(out_.sizes().equals(shape_));
      // fall through
    case FunctionConvention::FUNCTIONAL:;
      // nothing to do
  }

  // compute numel
  numel_ = 1;
  for (int64_t size : shape_) {
    numel_ *= size;
  }

  // ===================== Now it's type and device time ====================
  // =========== Behemoth, so put them into a method .... ===================
  setup_type_and_device(config);

  // Check fast setup
  if (input_needs_broadcast_ || input_needs_type_promotion_ ||
        output_needs_resize_ || output_needs_type_promotion_) {
    TORCH_INTERNAL_ASSERT(
        !input_needs_broadcast_, "broadcast is not supported");
    TORCH_INTERNAL_ASSERT(
        !input_needs_type_promotion_, "input type promotion is not supported");
    TORCH_INTERNAL_ASSERT(!output_needs_resize_, "output resize is not supported");
    TORCH_INTERNAL_ASSERT(
        !output_needs_type_promotion_, "output type promotion is not supported");
  }

  // TODO: the out check can be skipped for INPLACE, by similar switch...
  bool all_contiguous = a_.is_contiguous(at::MemoryFormat::Contiguous) &&
      b_.is_contiguous(at::MemoryFormat::Contiguous) &&
      (!out_.defined() || out_.is_contiguous(at::MemoryFormat::Contiguous));
  TORCH_CHECK(all_contiguous, "not all continous, unsupported -_-");

  // set output with contiguous
  // cannot call into a pure virtual function in constructor
  set_output(
      0,
      shape_,
      {},
      TensorOptions()
          .device(common_device_)
          .dtype(common_dtype_)
          .memory_format(MemoryFormat::Contiguous),
      names_);
}

void BinaryTensorIteratorBase::setup_type_and_device(
    const TensorIteratorConfig& config) {
  // Common device
  if (a_.device() == b_.device()) {
    common_device_ = a_.device();
  }
  else if (a_.device().is_cpu() && a_.dim() == 0) {
    common_device_ = b_.device();
  }
  else if (b_.device().is_cpu() && b_.dim() == 0) {
    common_device_ = a_.device();
  }
  else {
    TORCH_CHECK(false, "different device!");
  }

  TORCH_CHECK(!out_.defined() || out_.device() == common_device_, "output tensor device is wrong!");

  // Common dtype
  ScalarType a_scalar_type = a_.scalar_type(), b_scalar_type = b_.scalar_type();
  if (a_scalar_type == b_scalar_type) {
    common_dtype_ = a_scalar_type;
    input_needs_type_promotion_ = false;
  } else {
    compute_type_promotion();
    input_needs_type_promotion_ = true;
  }

  // TODO: out check can be skipped for inplace convention.
  if (out_.defined() && out_.scalar_type() != common_dtype_) {
    output_needs_type_promotion_ = true;
  }
}

void BinaryTensorIteratorBase::for_each(loop2d_t loop, int64_t grain_size) {
  if (numel_ == 0) {
    return;
  } else if (numel_ < grain_size || at::get_num_threads() == 1) {
    return serial_for_each(loop, {0, numel_});
  } else {
    at::parallel_for(0, numel_, grain_size, [&](int64_t begin, int64_t end) {
      serial_for_each(loop, {begin, end});
    });
  }
}

void BinaryTensorIteratorBase::serial_for_each(loop2d_t loop, Range range)
    const {
  if (range.size() == 0) {
    return;
  }
  auto strides = get_strides();
  auto base_ptrs = get_base_ptrs();

  if (true) { // TODO...
    auto ptrs = get_data_ptrs(base_ptrs, {range.begin});
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

StrideVector BinaryTensorIteratorBase::get_strides() const {
  StrideVector strides;

  // TODO: only handles fast setup...
  strides.push_back(out_.element_size());
  strides.push_back(a_.element_size());
  strides.push_back(b_.element_size());
  strides.push_back(0);
  strides.push_back(0);
  strides.push_back(0);

  return strides;
}

PtrVector BinaryTensorIteratorBase::get_base_ptrs() const {
  auto ptrs = SmallVector<char*, 4>();
  ptrs.push_back((char*)out_.data_ptr());
  ptrs.push_back((char*)a_.data_ptr());
  ptrs.push_back((char*)b_.data_ptr());
  return ptrs;
}

SmallVector<char*, 4> BinaryTensorIteratorBase::get_data_ptrs(
    ArrayRef<char*> base,
    IntArrayRef counter) const {
  TORCH_INTERNAL_ASSERT(base.size() == 3, "Only support 100% coalesced tensors -_-");

  auto ptrs = SmallVector<char*, 4>(base);
  int64_t offset = counter[0];
  ptrs[0] = base[0] + offset * out_.element_size();
  ptrs[1] = base[1] + offset * a_.element_size();
  ptrs[2] = base[2] + offset * b_.element_size();
  return ptrs;
}

// This is the structured kernels implementation of set_output.  It is
// NEVER actually called directly; instead, a subclass of TensorIteratorBase
// will override set_output to actually do the operation, and then call
// set_output on the TensorIteratorBase to setup TI's metadata.
// The precondition for this function is that maybe_get_output() now
// unconditionally returns a real Tensor (prior to output setting,
// this function may return an undefined tensor.)
void BinaryTensorIteratorBase::set_output(
    int64_t output_idx,
    IntArrayRef sizes,
    IntArrayRef strides,
    TensorOptions options,
    DimnameList names) {
  TORCH_INTERNAL_ASSERT(!output_needs_resize_, "not supported -_-");

  if (convention_ == FunctionConvention::FUNCTIONAL) {
    out_ = maybe_get_output(output_idx);
    TORCH_INTERNAL_ASSERT(out_.defined());
  }
}

void BinaryTensorIteratorBase::compute_type_promotion() {
  at::native::ResultTypeState state = {};

  state = at::native::update_result_type_state(a_, state);
  state = at::native::update_result_type_state(b_, state);
  common_dtype_ = at::native::result_type(state);
  TORCH_INTERNAL_ASSERT(common_dtype_ != ScalarType::Undefined);
}

} // namespace at
