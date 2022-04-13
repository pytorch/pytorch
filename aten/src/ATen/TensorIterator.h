#pragma once

#include <c10/util/FunctionRef.h>
#include <c10/util/MaybeOwned.h>
#include <c10/util/SmallVector.h>
#include <c10/util/TypeCast.h>
#include <c10/util/irange.h>
#include <ATen/core/Dimname.h>
#include <ATen/core/Range.h>
#include <ATen/core/TensorBase.h>
#include <ATen/TensorMeta.h>

#include <array>
#include <bitset>

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wshorten-64-to-32")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wshorten-64-to-32")
#endif
#if C10_CLANG_HAS_WARNING("-Wdeprecated-copy-dtor")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wdeprecated-copy-dtor")
#endif

namespace at {
class Tensor;
class OptionalTensorRef;
using NameVector = SmallVector<Dimname, kDimVectorStaticSize>;
}

// TensorIterator is a helper class for element-wise operations, such as
// arithmetic, comparisons, and trigonometric functions. It handles
// broadcasting and type conversions of operands.
//
// This is inspired by NumPy's Array Iterator API (NpyIter).
//
// The files Loops.h and Loops.cuh provide functions to build kernels that
// use TensorIterator.
//
// Example:
//
//   auto iter = TensorIteratorConfig()
//     .add_output(output)
//     .add_input(input)
//     .build()
//
// [MyKernel.cpp / MyKernel.cu]
//   cpu_kernel(iter, [](float a, float b) {
//     return a + b;
//   });
//
//   gpu_kernel(iter, []GPU_LAMBDA(float a, float b) -> float {
//     return a + b;
//   });
//
// Note [Order of Construction]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// When setting up the tensor iterator configuration, the output Tensors
// have to be added first via TensorIteratorConfig::add_owned_output(at::Tensor).
// After adding all outputs, the inputs can be added via
// TensorIteratorConfig::add_owned_input(at::Tensor).
// Adding another output after inputs have been added will rise an exception.
//
// Note [Common Dtype Computation]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Some operations have a natural notion of a "common dtype" or
//   "computation dtype" where all inputs are cast to one dtype, the
//   operation is performed, and then the results are cast to all outputs.
//
// TensorIterator infers a common dtype if all inputs have the same dtype,
//   and it computes one using type promotion rules on its inputs if
//   promote_inputs_to_common_dtype_ is true. Attempting to query
//   a common dtype otherwise will throw an exception.
//
// Note that the outputs are not considered when computing a common dtype.

namespace at {

namespace internal {
// This parameter is heuristically chosen to determine the minimum number of
// work that warrants parallelism. For example, when summing an array, it is
// deemed inefficient to parallelise over arrays shorter than 32768. Further,
// no parallel algorithm (such as parallel_reduce) should split work into
// smaller than GRAIN_SIZE chunks.
constexpr int64_t GRAIN_SIZE = 32768;

// Storage for a non-owning Tensor, without needing to include Tensor.h
class TORCH_API OpaqueOptionalTensorRef {
  alignas(alignof(TensorBase)) std::array<char, sizeof(TensorBase)> data_;
public:
  OpaqueOptionalTensorRef();
  ~OpaqueOptionalTensorRef();

  OptionalTensorRef* get() {
    return reinterpret_cast<OptionalTensorRef*>(data_.data());
  }
  const OptionalTensorRef* get() const {
    return reinterpret_cast<const OptionalTensorRef*>(data_.data());
  }

  OptionalTensorRef& operator*() { return *get(); }
  const OptionalTensorRef& operator*() const { return *get(); }
  OptionalTensorRef* operator->() { return get(); }
  const OptionalTensorRef* operator->() const { return get(); }

  const Tensor& getTensor() const;
};
} // namespace internal

struct TORCH_API OperandInfo {
  using StrideVector = SmallVector<int64_t, 6>;
  OperandInfo() = default;
  C10_ALWAYS_INLINE explicit OperandInfo(c10::MaybeOwned<TensorBase>&& t) {
    if (t->defined()) {
      device = t->device();
      target_dtype = t->scalar_type();
      current_dtype = target_dtype;
    }
    tensor(std::move(t));
    validate();
  }

  C10_ALWAYS_INLINE ~OperandInfo() = default;

  /// Stride after broadcasting. The stride is in bytes, not number of elements.
  StrideVector stride_bytes;

  /// The desired device and type for the operand. For inputs, this specifies that
  /// the input should be converted to this type if necessary. For outputs, this
  /// specifies which type to allocate. target_dtype and device are initialized with the dtype and device of the tensor
  /// but during type promotion target_dtype value can become different from tensor's dtype
  /// also, during type promotion target_dtype and device can be set for an undefined tensor so that tensor can be properly
  /// constructed later.
  c10::optional<Device> device = c10::nullopt;
  ScalarType target_dtype = ScalarType::Undefined;
  // Caches dtype of the tensor, because scalar_type is an expensive operation
  // If dtype of the tensor is changed (e.g. as a result of type promotion or in allocate_outputs), this
  //value should be changed too.
  ScalarType current_dtype = ScalarType::Undefined;

  bool is_device_defined() const { return device.has_value(); }
  bool is_type_defined() const { return target_dtype != ScalarType::Undefined; }
  TensorOptions options() const {
    return TensorOptions(target_dtype).device(device);
  }

  /// The data pointer. This may be different from tensor->data_ptr() if the
  /// iterator is split.
  void* data = nullptr;

  bool is_output = false;

  bool will_resize = false;

  bool is_read_write = false;

  void validate() {
    TORCH_CHECK(
        !tensor_base_->defined() || tensor_base_->layout() == kStrided,
        "unsupported tensor layout: ", tensor_base_->layout());
  }

  /// The tensor operand. Note that the strides, data pointer, and
  /// other attributes may differ due to dimension reordering and
  /// coalescing.
  const Tensor& tensor() const {
    return tensor_storage_.getTensor();
  }
  const TensorBase& tensor_base() const {
    return *tensor_base_;
  }
  void tensor(c10::MaybeOwned<TensorBase> &&tensor);

  // Save the original tensor operand in cases when an output is modified
  // (e.g. if dtype is changed)
  const Tensor& original_tensor() const {
    return original_tensor_storage_.getTensor();
  }
  const TensorBase& original_tensor_base() const {
    return *original_tensor_base_;
  }

  // Set tensor to a new value, and store the old tensor value in original_tensor
  // Should only ever be called once for the lifetime of an operand
  void exchange_tensor(c10::MaybeOwned<TensorBase> &&new_tensor);

  // Move original_tensor back into tensor, exchange_tensor must have been called before
  void restore_original_tensor();

private:
  c10::MaybeOwned<TensorBase> tensor_base_;
  c10::MaybeOwned<TensorBase> original_tensor_base_ =
      c10::MaybeOwned<TensorBase>::owned(c10::in_place);

  // We store TensorBase visibly in the header to allow inline access.
  // However, we sometimes need a genuine `const Tensor &` for the
  // TensorIterator API. So, we also store a non-owning `Tensor`
  // object in these `_storage_` variables.
  internal::OpaqueOptionalTensorRef tensor_storage_;
  internal::OpaqueOptionalTensorRef original_tensor_storage_;
};

struct SplitUntil32Bit;

enum class FastSetupType : uint8_t {
  NONE,
  CONTIGUOUS,
  CHANNELS_LAST,
  NON_OVERLAPPING_DENSE
};

class TensorIteratorConfig;
struct TensorIterator;

struct TORCH_API TensorIteratorBase : public impl::MetaBase {
  using DimMask = std::bitset<64>;
  using PtrVector = SmallVector<char*, 4>;
  using StrideVector = SmallVector<int64_t, 6>;

  TensorIteratorBase();
  void build(TensorIteratorConfig&);

  // The inner-loop function operates on the fastest moving dimension. It
  // implements element-wise operations in terms of 1-d strided tensors.
  //
  // Arguments:
  //  data: data pointers for each operand (length `ntensors`)
  //  strides: stride for each operand (length `ntensors`)
  //  size: size of inner loop
  //
  // The `size` often matches shape[0], but may be smaller due to
  // parallelization of the inner loop.
  using loop2d_t = c10::function_ref<void(char** data, const int64_t* strides, int64_t size0, int64_t size1)>;

  using loop_subiter_t = c10::function_ref<void(TensorIteratorBase& subiter)>;

  void foreach_reduced_elt(loop_subiter_t loop, bool parallelize=true);

  int ndim() const { return shape_.size(); }
  IntArrayRef shape() const { return shape_; }
  int64_t numel() const;
  int ntensors() const { return operands_.size(); }
  int noutputs() const { return num_outputs_; }
  int ninputs() const { return ntensors() - noutputs(); }
  IntArrayRef view_offsets() const { return view_offsets_; }

  /// number of elements in the output operand. this is the same as numel() for
  /// operations that are not reductions.
  int64_t num_output_elements() const;

  /// number of reduced dimensions in a reduction operation
  int num_reduce_dims() const;

  /// 1-dimensional iteration and no buffering or type conversion
  bool is_trivial_1d() const;
  /// Reducible to 1-dimensional and all operands are contiguous
  bool is_contiguous() const;
  bool is_dim_reduced(int dim) const;

  /// Accessors for each operand
  IntArrayRef strides(int arg) const { return operands_[arg].stride_bytes; }
  void* data_ptr(int arg) const;
  ScalarType dtype(int arg=0) const { return operands_[arg].current_dtype; }
  ScalarType common_dtype() const {
    TORCH_INTERNAL_ASSERT(common_dtype_ != ScalarType::Undefined, "Queried for invalid common dtype!");
    return common_dtype_;
  }
  ScalarType input_dtype(int arg=0) const { return operands_[num_outputs_ + arg].current_dtype; }
  Device device(int arg=0) const { return operands_[arg].device.value(); }
  DeviceType device_type(int arg=0) const { return device(arg).type(); }
  int64_t element_size(int arg) const { return elementSize(dtype(arg)); }
  bool is_scalar(int arg) const;
  bool is_cpu_scalar(int arg) const;

  const TensorBase& tensor_base(int arg) const {
    return operands_[arg].tensor_base();
  }
  const Tensor& tensor(int arg) const {
    return operands_[arg].tensor();
  }

  const TensorBase& output_base(int arg=0) const {
    AT_ASSERT(arg < num_outputs_);
    return tensor_base(arg);
  }

  const Tensor& output(int arg=0) const {
    AT_ASSERT(arg < num_outputs_);
    return tensor(arg);
  }

  const TensorBase& input_base(int arg=0) const {
    AT_ASSERT(arg >= 0 && arg < ntensors() - num_outputs_);
    return tensor_base(num_outputs_ + arg);
  }
  const Tensor& input(int arg=0) const {
    AT_ASSERT(arg >= 0 && arg < ntensors() - num_outputs_);
    return tensor(num_outputs_ + arg);
  }

  // Copies from temporary outputs back to the original outputs
  // NOTE: only used on CPU
  void cast_outputs();

  /// Removes an operand from this iterator
  void remove_operand(int arg);
  /// Shrinks an iterated dimension
  void narrow(int dim, int64_t start, int64_t size);
  /// Narrows every dim after and including `start_dim` to size one.
  void select_all_keeping_dim(int start_dim, IntArrayRef starts);
  /// Replaces the data pointer for the operand at index `arg`.
  /// The new pointer should have the same sizes, strides and dtype as the
  /// original
  void unsafe_replace_operand(int arg, void* data);

  /// Splits this TensorIterator into two iterators. Together they iterate over
  /// the entire operation. Used by `with_32bit_indexing()`.
  std::unique_ptr<TensorIterator> split(int dim);

  /// Returns the dimension with the largest extent: (size[dim]-1) * stride[dim]
  int get_dim_to_split() const;

  template <typename T>
  T scalar_value(int arg) {
    auto& op = operands_[arg];
    return c10::fetch_and_cast<T>(op.tensor_base().scalar_type(), op.data);
  }

private:
  template <typename loop1d_t>
  auto loop_2d_from_1d(const loop1d_t& loop) {
    return [loop, ntensor=ntensors()](
        char** base, const int64_t* strides, int64_t size0, int64_t size1) {
      PtrVector data(base, base + ntensor);
      const int64_t* outer_strides = &strides[ntensor];
      for (const auto i : c10::irange(size1)) {
        if (i > 0) {
          for (const auto arg : c10::irange(ntensor)) {
            data[arg] += outer_strides[arg];
          }
        }
        loop(data.data(), strides, size0);
      }
    };
  }

public:
  template <typename loop1d_t,
            std::enable_if_t<std::is_convertible<
              loop1d_t, c10::function_ref<void(char**, const int64_t* strides, int64_t size)>
            >::value, int> = 0>
  void for_each(loop1d_t loop, int64_t grain_size = at::internal::GRAIN_SIZE) {
    for_each(loop_2d_from_1d(loop), grain_size);
  }

  void for_each(loop2d_t loop, int64_t grain_size = at::internal::GRAIN_SIZE);

  void parallel_reduce(loop2d_t loop);

  template <typename loop1d_t,
            std::enable_if_t<std::is_convertible<
              loop1d_t, c10::function_ref<void(char**, const int64_t* strides, int64_t size)>
            >::value, int> = 0>
  void serial_for_each(loop1d_t loop, Range range) {
    serial_for_each(loop_2d_from_1d(loop), range);
  }

  void serial_for_each(loop2d_t loop, Range range) const;

  /// Create a strides array for a Tensor with shape of this iterator. The
  /// parameter `element_size` specifies the size of Tensor's data type in
  /// bytes (e.g. `4` for `float`)
  StrideVector compatible_stride(int element_size) const;

  /// Inverts the re-ordering done by reorder_dimensions. This can only be
  /// called *before* coalesce_dimensions() is called.
  DimVector invert_perm(IntArrayRef input) const;

  /// Reapply same re-ordering as it is done by reorder_dimensions. This can
  /// only be called *before* coalesce_dimensions() is called.
  DimVector apply_perm_and_mul(IntArrayRef input, int mul) const;

  /// Helper functions for CPU iteration
  StrideVector get_dim_strides(int dim) const;
  StrideVector get_strides() const;
  StrideVector get_inner_strides() const { return get_dim_strides(0); }
  PtrVector get_base_ptrs() const;

  // Helper functions for advanced stride manipulations (e.g. torch.flip)
  void _unsafe_set_arg_strides(const int arg, IntArrayRef strides) { operands_[arg].stride_bytes = std::move(strides); }
  void _unsafe_set_arg_data(const int arg, void* data) { operands_[arg].data = data; }

  /// true if the stride computation can use 32-bit arithmetic. Used by GPU kernels
  bool can_use_32bit_indexing() const;

  /// An "iteratable" object that recursively splits this iterator into sub-iterators
  /// that can use 32-bit indexing.
  SplitUntil32Bit with_32bit_indexing() const;

  /// If the kernel should accumulate into the output. Only relevant for CUDA
  /// reductions.
  bool should_accumulate() const { return accumulate_; }

  /// Whether this iterator produces the actual output,
  /// as opposed to something that will be accumulated further. Only relevant for
  /// CUDA reductions.
  bool is_final_output() const { return final_output_; }

  bool has_contiguous_first_dim() const {
    int num_tensors = ntensors();
    for (const auto i : c10::irange(num_tensors)) {
      if (strides(i)[0] != element_size(i)) {
        return false;
      }
    }
    return true;
  }

  void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options, DimnameList names) override;

#define TORCH_DISALLOW_TEMPORARIES_IMPL(methodname, maybestatic)                               \
  maybestatic void methodname(TensorBase&& out, const TensorBase& a, const TensorBase& b) = delete; \
  maybestatic void methodname(const TensorBase& out, TensorBase&& a, const TensorBase& b) = delete; \
  maybestatic void methodname(const TensorBase& out, const TensorBase& a, TensorBase&& b) = delete; \
  maybestatic void methodname(TensorBase&& out, TensorBase&& a, const TensorBase& b) = delete; \
  maybestatic void methodname(TensorBase&& out, const TensorBase& a, TensorBase&& b) = delete; \
  maybestatic void methodname(const TensorBase& out, TensorBase&& a, TensorBase&& b) = delete; \
  maybestatic void methodname(TensorBase&& out, TensorBase&& a, TensorBase&& b) = delete;

#define TORCH_DISALLOW_TEMPORARIES(methodname) TORCH_DISALLOW_TEMPORARIES_IMPL(methodname,)

  void build_binary_float_op(const TensorBase& out, const TensorBase& a, const TensorBase& b);
  void build_borrowing_binary_float_op(const TensorBase& out, const TensorBase& a, const TensorBase& b);
  TORCH_DISALLOW_TEMPORARIES(build_borrowing_binary_float_op)
  void build_binary_op(const TensorBase& out, const TensorBase& a, const TensorBase& b);
  void build_borrowing_binary_op(const TensorBase& out, const TensorBase& a, const TensorBase& b);
  TORCH_DISALLOW_TEMPORARIES(build_borrowing_binary_op)
  void build_unary_float_op(const TensorBase& out, const TensorBase& a);
  void build_borrowing_unary_float_op(const TensorBase& out, const TensorBase& a);
  TORCH_DISALLOW_TEMPORARIES(build_borrowing_unary_float_op)
  void build_unary_op(const TensorBase& out, const TensorBase& a);
  // Odd special case needed for pow. Has to borrow the output because
  // it's a structured kernel, but the argument is potentially a copy.
  void build_output_borrowing_argument_owning_unary_op(const TensorBase& out, const TensorBase& a);
  void build_borrowing_unary_op(const TensorBase& out, const TensorBase& a);
  TORCH_DISALLOW_TEMPORARIES(build_borrowing_unary_op)
  void build_borrowing_unary_force_boolean_op(const TensorBase& out, const TensorBase& a);
  TORCH_DISALLOW_TEMPORARIES(build_borrowing_unary_force_boolean_op)
  void build_comparison_op(const TensorBase& out, const TensorBase& a, const TensorBase& b);
  void build_borrowing_comparison_op(const TensorBase& out, const TensorBase& a, const TensorBase& b);
  TORCH_DISALLOW_TEMPORARIES(build_borrowing_comparison_op)
  // Another special case: we need to own the second argument for comparison ops.
  void build_borrowing_except_last_argument_comparison_op(const TensorBase& out, const TensorBase& a, const TensorBase& b);
  void build_ternary_op(const TensorBase& out, const TensorBase& a, const TensorBase& b, const TensorBase& c);

#undef TORCH_DISALLOW_TEMPORARIES
protected:
  // Mutable reference as it moves tensors out of TensorIteratorConfig
  void populate_operands(TensorIteratorConfig&);
  void mark_outputs();
  void mark_resize_outputs(const TensorIteratorConfig&);
  void compute_mem_overlaps(const TensorIteratorConfig&);
  void compute_shape(const TensorIteratorConfig&);
  void compute_strides(const TensorIteratorConfig&);
  void reorder_dimensions();
  void permute_dimensions(IntArrayRef perm);
  void compute_types(const TensorIteratorConfig&);
  ScalarType compute_common_dtype();
  void allocate_or_resize_outputs();
  bool fast_set_up(const TensorIteratorConfig&);
  FastSetupType compute_fast_setup_type(const TensorIteratorConfig&);
  void compute_names(const TensorIteratorConfig&);
  void propagate_names_to_outputs();
  void coalesce_dimensions();

protected:

  /// Records the "computation" shape of the output tensor. The computation
  /// shape is different from the regular shape in a few ways:
  ///
  ///   - The shape may be permuted (via permute_dimensions) so that we
  ///     process the dimensions in the most computationally efficient order
  ///     (rather than the logical order given to us by the users.)
  ///   - The shape may have adjacent dimensions collapsed (via
  ///     coalesce_dimensions) so that we minimize the number of
  ///     dimensions we have to explicitly iterate over.  For example,
  ///     a pointwise operation on a contiguous tensor "computationally"
  ///     consists of only a single dimension.
  ///
  /// In other words, the computation shape is the output shape as it
  /// actually matters for implementing the kernel, but not necessarily the
  /// output shape that the user will see in the end.
  ///
  /// The lifecycle of mutations to shape_ in TensorIterator:
  ///   - declare_static_shape() sets an initial shape explicitly
  ///     provided by user, otherwise
  ///   - compute_shape() computes the true (non-computational) shape
  ///     specified by the user.
  ///   - reorder_dimensions() reorders dimensions to improve coalescing.
  ///   - coalesce_dimensions() then coalesces adjacent dimensions when
  ///     possible.
  ///
  /// The shape may also be further modified if we create sub-TensorIterators,
  /// e.g., via narrow or select_all_keeping_dim.
  DimVector shape_;

  /// Temporarily records the permutation computed by reorder_dimensions.
  /// This permutation maps the computation output dimension (dim) to
  /// the original true output dimension (perm_[dim]).  It is used by
  /// invert_perm to undo the permutation.  After coalesce_dimensions is
  /// called, the permutation is no longer valid (as, in general, there
  /// is no permutation that will make computation dimensions to
  /// output dimensions); methods that manipulate perm_ are obligated
  /// to test that !has_coalesced_dimensions
  DimVector perm_;

  /// Has coalesce_dimensions() (or any moral equivalent, e.g., fast_build())
  /// been called?  This is SOLELY used to check validity of perm_.
  bool has_coalesced_dimensions_ = false;

  /// Whether iteration must be fixed. This disables dimension permuting and also
  /// changes how for_each divides work among threads.
  bool enforce_linear_iteration_ = false;

  /// The index offsets into the original tensors for each dimension.
  /// This is only non-zero when you narrow() a TensorIterator (e.g.,
  /// when you make sub-TensorIterators).
  DimVector view_offsets_;

  /// The computed names of the output tensor.  Computed by compute_names()
  NameVector names_;

  /// The operands of the TensorIterator: both the inputs and outputs.  The
  /// outputs MUST come first in the operands_ list.  There is always an
  /// operand for each output of the TensorIterator, even if TensorIterator
  /// will ultimately be responsible for allocating the output; in those
  /// cases, tensor is simply undefined (and will be populated later
  /// during build()).
  ///
  /// This list is initially populated prior to build(), but build() mutates
  /// OperandInfo to populate more information.
  SmallVector<OperandInfo, 4> operands_;

  /// Number of outputs in operands_ (the length of the outputs prefix
  /// in operands_).
  int num_outputs_ = 0;

  /// Whether or not all operands have the same shape.  Having all the same
  /// shape affects whether or not the iterator is eligible for fast setup.
  bool all_ops_same_shape_ = false;

  /// The "computation" dtype of TensorIterator, specifying what the dtype
  /// we will do the internal computation in TensorIterator.  Typically,
  /// this matches the dtype of the output tensors, but not always!
  ScalarType common_dtype_ = ScalarType::Undefined;

  /// This is currently defined as kCPU, or the device of the first non-CPU
  /// tensor argument. See TensorIteratorBase::compute_types for details.
  Device common_device_ = kCPU;

  /// Set by split(), see should_accumulate() and is_final_output()
  bool accumulate_ = false;
  bool final_output_ = true;

  // From TensorIteratorConfig
  bool is_reduction_ = false;

  /// Set by populate_operands(), says if we're handling meta tensors
  bool is_meta_ = false;
};

struct TORCH_API TensorIterator final : public TensorIteratorBase {
  TensorIterator() : TensorIteratorBase() {}
  // Slicing is OK, TensorIterator guaranteed NOT to have any fields
  TensorIterator(const TensorIteratorBase& iter) : TensorIteratorBase(iter) {}

#define TORCH_DISALLOW_TEMPORARIES(methodname) TORCH_DISALLOW_TEMPORARIES_IMPL(methodname, static)

  static TensorIterator binary_float_op(TensorBase& out, const TensorBase& a, const TensorBase& b);
  static TensorIterator binary_op(TensorBase& out, const TensorBase& a, const TensorBase& b);
  static TensorIterator borrowing_binary_op(const TensorBase& out, const TensorBase& a, const TensorBase& b);
  TORCH_DISALLOW_TEMPORARIES(borrowing_binary_op)
  static TensorIterator comparison_op(TensorBase& out, const TensorBase& a, const TensorBase& b);
  static TensorIterator unary_op(TensorBase& out, const TensorBase& a);
  static TensorIterator unary_float_op(TensorBase& out, const TensorBase& a);
  static TensorIterator nullary_op(TensorBase& out);
  static TensorIterator borrowing_nullary_op(const TensorBase& out);
  static TensorIterator borrowing_nullary_op(TensorBase&& out) = delete;
  static TensorIterator reduce_op(TensorBase& out, const TensorBase& a);
  static TensorIterator reduce_op(TensorBase& out1, TensorBase& out2, const TensorBase& a);
#undef TORCH_DISALLOW_TEMPORARIES
#undef TORCH_DISALLOW_TEMPORARIES_IMPL

  const Tensor& maybe_get_output(int64_t output_idx) override;
  void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options, DimnameList names) override;
};

class TORCH_API TensorIteratorConfig final {
public:
  friend struct TensorIteratorBase;
  friend struct TensorIterator;

  TensorIteratorConfig() {}

  C10_DISABLE_COPY_AND_ASSIGN(TensorIteratorConfig);

  /// Construction
  // Stores input/output Tensors without incrementing the reference count.
  // Important: the outputs have to be added before the inputs.
  TensorIteratorConfig& add_output(const TensorBase& output) {
    return add_borrowed_output(output);
  }
  TensorIteratorConfig& add_input(const TensorBase& input) {
    return add_borrowed_input(input);
  }

  // Borrowing from temporaries is unlikely to go well.
  TensorIteratorConfig& add_output(TensorBase&& output) = delete;
  TensorIteratorConfig& add_input(TensorBase&& input) = delete;

  // Stores input/output Tensors while incrementing the reference count.
  // Note that add_{in,out}put are nearly always what you
  // want, and the exception (adding an unnamed temporary) won't
  // compile.
  TensorIteratorConfig& add_owned_output(const TensorBase& output);
  TensorIteratorConfig& add_owned_input(const TensorBase& input);

  // Advanced API: stores input/output Tensors without incrementing
  // the reference count. The caller must ensure that these Tensors
  // live at least as long as this TensorIteratorConfig and any
  // TensorIteratorBase built from this TensorIteratorConfig.
  // Important: the outputs have to be added before the inputs.
  TensorIteratorConfig& add_borrowed_output(const TensorBase& output);
  TensorIteratorConfig& add_borrowed_input(const TensorBase& input);

  // Borrowing from temporaries is unlikely to go well.
  TensorIteratorConfig& add_borrowed_output(TensorBase&& output) = delete;
  TensorIteratorConfig& add_borrowed_input(TensorBase&& input) = delete;

  // Sets the check_mem_overlap_ flag, which is true by default.
  // If true, inputs are checked for partial overlap with the outputs and
  // outputs are checked for internal overlap (e.g. broadcasted views). An error
  // is raised if unacceptable overlap is detected.
  // If you're migrating an existing operator to using TensorIterator, please
  // consider if the previous implementation checked memory overlap. If it did
  // not, and if the operator is idempotent (for example, Tensor.fill_(0)), then
  // checking memory overlap is BC-breaking. Please don't check memory overlap
  // in that case.
  TensorIteratorConfig& set_check_mem_overlap(bool check_mem_overlap) {
    check_mem_overlap_ = check_mem_overlap;
    return *this;
  }

  // Sets the check_all_same_dtype_ flag, which is true by default
  // If true, checks that all inputs and defined outputs have the same dtype
  // Setting either of promote_inputs_to_common_dtype_
  //   or cast_common_dtype_to_outputs_ to true will set
  //   check_all_same_dtype_ to false.
  TensorIteratorConfig& check_all_same_dtype(const bool _check_all_same_dtype) {
    check_all_same_dtype_ = _check_all_same_dtype;
    return *this;
  }

  // Sets the check_all_same_device_ flag, which is true by default
  // If true, all operands must be on the same device, with the possible
  //   exception of CPU scalars, which can be passed to some CUDA kernels
  //   as kernel arguments.
  TensorIteratorConfig& check_all_same_device(const bool _check_all_same_device) {
    check_all_same_device_ = _check_all_same_device;
    return *this;
  }

  // Sets the enforce_safe_casting_to_output_ flag, which is false by default
  // If true, the iterator's "common dtype" must be computable
  //   (see the [Common Dtype Computation] note) and
  //   canCast(common dtype, output dtype) must be true for all outputs.
  TensorIteratorConfig& enforce_safe_casting_to_output(const bool _enforce_safe_casting_to_output) {
    enforce_safe_casting_to_output_ = _enforce_safe_casting_to_output;
    return *this;
  }

  // Sets the enforce_linear_iteration_ flag, which is false by default.
  // If true, iteration goes in the same order as a C-contiguous tensor
  // is layed out in memory. i.e. last dimension iterates fastest.
  //
  // This iteration order can be less efficient and may even prevent vectorization.
  // So only use if the correctness of your kernel depends on it.
  TensorIteratorConfig& enforce_linear_iteration(const bool _enforce_linear_iteration = true) {
    enforce_linear_iteration_ = _enforce_linear_iteration;
    return *this;
  }

  // Sets the promote_inputs_to_common_dtype_ flag, which is false by default
  // If true, the iterator's "common dtype" is always computed (see the
  //   [Common Dtype Computation] note) and, on the CPU, temporary copies of
  //   the inputs in the common dtype are passed as the actual inputs to
  //   the operation.
  // Setting this flag to true sets check_all_same_dtype_ to false.
  TensorIteratorConfig& promote_inputs_to_common_dtype(const bool _promote_inputs_to_common_dtype) {
    promote_inputs_to_common_dtype_ = _promote_inputs_to_common_dtype;
    if (_promote_inputs_to_common_dtype) {
      check_all_same_dtype_ = false;
    }
    return *this;
  }

  // Sets the promote_integer_inputs_to_float_ flag, which is false by default
  // NOTE: If set to true, the promote_inputs_to_common_dtype_ must also be true.
  // If true, if the iterator's "common dtype" is an integral type (including bool)
  //   then it is changed to the default float scalar type.
  TensorIteratorConfig& promote_integer_inputs_to_float(const bool _promote_integer_inputs_to_float) {
    promote_integer_inputs_to_float_ = _promote_integer_inputs_to_float;
    TORCH_INTERNAL_ASSERT(!promote_integer_inputs_to_float_ || promote_inputs_to_common_dtype_);
    return *this;
  }

  TensorIteratorConfig& is_reduction(const bool _is_reduction) {
    is_reduction_ = _is_reduction;
    return *this;
  }

  TensorIteratorConfig& allow_cpu_scalars(const bool _allow_cpu_scalars) {
    allow_cpu_scalars_ = _allow_cpu_scalars;
    return *this;
  }

  // Sets the cast_common_dtype_to_outputs_ flag, which is false by default
  // If true, the iterator's "common dtype" must be computatable
  //   (see the [Common Dtype Computation] note) and, on the CPU, temporary
  //   copies of the outputs are passed as the actual output to the operation.
  //   These temporaries are then copied to the original outputs after
  //   the operation is performed (see cast_outputs()).
  // Setting this flag to true sets check_all_same_dtype_ to false.
  TensorIteratorConfig& cast_common_dtype_to_outputs(const bool _cast_common_dtype_to_outputs) {
    cast_common_dtype_to_outputs_ = _cast_common_dtype_to_outputs;
    if (_cast_common_dtype_to_outputs) {
      check_all_same_dtype_ = false;
    }
    return *this;
  }

  TensorIteratorConfig& resize_outputs(bool resize_outputs) {
    resize_outputs_ = resize_outputs;
    return *this;
  }

  // Bypass output dtype/device computation and fix the dtype/device as specified here.
  TensorIteratorConfig& declare_static_dtype_and_device(ScalarType dtype, Device device);
  TensorIteratorConfig& declare_static_dtype(ScalarType dtype);
  TensorIteratorConfig& declare_static_device(Device device);
  TensorIteratorConfig& declare_static_shape(IntArrayRef shape);
  TensorIteratorConfig& declare_static_shape(IntArrayRef shape, IntArrayRef squash_dims);

  // It would be better if this was && qualified, but this would be at the cost
  // of a lot of boilerplate above
  TensorIterator build() {
    TensorIterator iter;
    iter.build(*this);
    return iter;
  }

private:
  SmallVector<c10::MaybeOwned<TensorBase>, 4> tensors_;
  int num_outputs_ = 0;
  int num_inputs_ = 0;

  c10::optional<DimVector> static_shape_ = c10::nullopt;
  c10::optional<ScalarType> static_dtype_ = c10::nullopt;
  c10::optional<Device> static_device_ = c10::nullopt;
  bool check_mem_overlap_ = true;
  bool allow_cpu_scalars_ = false;
  bool is_reduction_ = false;
  bool resize_outputs_ = true;
  bool check_all_same_dtype_ = true;
  bool check_all_same_device_ = true;
  bool enforce_safe_casting_to_output_ = false;
  bool enforce_linear_iteration_ = false;
  bool promote_inputs_to_common_dtype_ = false;
  bool promote_integer_inputs_to_float_ = false;
  bool cast_common_dtype_to_outputs_ = false;
};



/// A container-like struct that acts as if it contains splits of a
/// TensorIterator that can use 32-bit indexing. Taken together the splits cover
/// the original TensorIterator.
struct TORCH_API SplitUntil32Bit {
  struct TORCH_API iterator {
    iterator() {};
    iterator(const TensorIteratorBase& iter);
    iterator(iterator&&) = default;

    // Guaranteed to be a TensorIterator proper!
    TensorIterator& operator*() const;
    iterator& operator++();
    bool operator==(const iterator& other) const {
      // two iterators are equal if they are the same object or they're both empty
      return this == &other || (vec.empty() && other.vec.empty());
    }
    // needed for C++11 range-based for loop
    bool operator!=(const iterator& other) const { return !(*this == other); }

    /// stack of TensorIterators to be split
    std::vector<std::unique_ptr<TensorIterator>> vec;
  };

  SplitUntil32Bit(const TensorIteratorBase& iter) : iter(iter) {}

  iterator begin() const;
  iterator end() const;

private:
  const TensorIteratorBase& iter;
};

}  // namespace at

C10_CLANG_DIAGNOSTIC_POP()
