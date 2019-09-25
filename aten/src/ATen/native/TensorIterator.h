#pragma once

#include <ATen/ATen.h>
#include <c10/util/SmallVector.h>
#include <ATen/core/Range.h>
#include <ATen/detail/ScalarTypeConversions.h>
#include <bitset>
#include <c10/util/Optional.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/core/EnableNamedTensor.h>

// TensorIterator is a helper class for element-wise operations, such as
// arithmetic, comparisions, and trigonometric functions. It handles
// broadcasting and type conversions of operands.
//
// This is inspired by NumPy's Array Iterator API (NpyIter).
//
// The files Loops.h and Loops.cuh provide functions to build kernels that
// use TensorIterator.
//
// Example:
//
//   auto iter = TensorIterator();
//   iter.add_output(output);
//   iter.add_input(input);
//   iter.build()
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
// Note [Result type computation]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// TensorIterator handles limited mixed-type operations. The result type is
// computed using promoteTypes on the scalar types of the operands with the
// following precedence:
//
// 1) Tensors with dim 1 or higher
// 2) Tensors with dim 0 that aren't wrapped numbers (e.g. `tensor(5)`)
// 3) Tensors with dim 0 that are wrapped numbers (e.g. `5`)
//
// So if there are any tensors of dim 1 or higher, then 0-dim tensors do not
// affect the result type. This behavior was chosen to preserve backwards
// compatibility and is *likely to change* in the near future.
// (See https://github.com/pytorch/pytorch/issues/9515)
//
// Note that TensorIterator currently supports type conversions on 0-dim
// tensors and arithmetic operators. Other type conversions will raise an 
// exception.

namespace at {

struct DimCounter {
  DimCounter(IntArrayRef shape, Range range);

  void increment(const std::array<int64_t, 2>& step);
  bool is_done() const;
  std::array<int64_t, 2> max_2d_step() const;

  IntArrayRef shape;
  Range range;
  DimVector values;
  int64_t offset;
};

struct CAFFE2_API OperandInfo {
  OperandInfo() {}
  explicit OperandInfo(const Tensor& t) : tensor(t) {
    if (t.defined()) {
      device = t.device();
      dtype = t.scalar_type();
    }
    validate();
  }
  OperandInfo(const Tensor& t, Device device, ScalarType dtype)
    : tensor(t), device(device), dtype(dtype) {
    validate();
  }

  /// Stride after broadcasting. The stride is in bytes, not number of elements.
  DimVector stride_bytes;

  /// The tensor operand. Note that the strides, data pointer, and
  /// other attributes may differ due to dimension reordering and
  /// coalescing.
  Tensor tensor;

  // Save the original tensor operand in cases when an output is modified
  // (e.g. if dtype is changed)
  Tensor original_tensor;

  /// The desired device and type for the operand. For inputs, this specifies that
  /// the input should be converted to this type if necessary. For outputs, this
  /// specifies which type to allocate. Note that there is very limited support
  /// for type conversions currently: they are only allowed for zero-dim tensors.
  Device device = kCPU;
  ScalarType dtype = ScalarType::Undefined;

  bool is_type_defined() const { return dtype != ScalarType::Undefined; }
  TensorOptions options() const {
    return TensorOptions(dtype).device(device);
  }

  /// The data pointer. This may be different from tensor.data_ptr() if the
  /// iterator is split.
  void* data = nullptr;

  bool is_output = false;

  bool is_read_write = false;

  void validate() {
    TORCH_CHECK(
        !tensor.defined() || tensor.layout() == kStrided,
        "unsupported tensor layout: ", tensor.layout());
  }
};

struct SplitUntil32Bit;

enum class CommonDTypeStrategy : uint8_t {
  COMPUTE_ALL = 0, // Compute common dtype based on inputs and outputs. Try to promote common dtype to inputs and outputs
  COMPUTE_INPUTS = 1, // Compute common dtype based only on inputs. Try to promote common dtype only to inputs
  COMPUTE_NONE = 2, // Do not compute and promote common dtype
};

struct CAFFE2_API TensorIterator {
  using DimMask = std::bitset<64>;
  using PtrVector = SmallVector<char*, 4>;

  TensorIterator() {}

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
  using loop_t = std::function<void(char** data, const int64_t* strides, int64_t size)>;
  using loop2d_t = std::function<void(char** data, const int64_t* strides, int64_t size0, int64_t size1)>;

  using loop_subiter_t = std::function<void(TensorIterator& subiter)>;

  void foreach_reduced_elt(const loop_subiter_t& loop, bool parallelize=true);

  static TensorIterator binary_op(Tensor& out, const Tensor& a, const Tensor& b,
    bool check_mem_overlap = false);
  static TensorIterator unary_op(Tensor& out, const Tensor& a,
    bool check_mem_overlap = false);
  static TensorIterator nullary_op(Tensor& out);
  static TensorIterator reduce_op(Tensor& out, const Tensor& a);
  static TensorIterator reduce_op(Tensor& out1, Tensor& out2, const Tensor& a);

  int ndim() const { return shape_.size(); }
  IntArrayRef shape() const { return shape_; }
  int64_t numel() const;
  int ntensors() const { return operands_.size(); }
  int noutputs() const { return num_outputs_; }
  int ninputs() const { return ntensors() - noutputs(); }

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
  ScalarType dtype(int arg=0) const { return operands_[arg].tensor.scalar_type(); }
  ScalarType input_dtype(int arg=0) const { return operands_[num_outputs_ + arg].dtype; }
  Device device(int arg=0) const { return operands_[arg].device; }
  DeviceType device_type(int arg=0) const { return device(arg).type(); }
  int64_t element_size(int arg) const { return elementSize(dtype(arg)); }
  bool is_scalar(int arg) const;
  bool is_cpu_scalar(int arg) const;

  const Tensor& tensor(int arg) const { return operands_[arg].tensor; }
  Tensor& tensor(int arg) { return operands_[arg].tensor; }

  Tensor output(int arg=0) const {
    AT_ASSERT(arg < num_outputs_);
    return operands_[arg].tensor;
  }

  void cast_outputs() {
    if (compute_common_dtype_strategy_ == CommonDTypeStrategy::COMPUTE_ALL) {
      for(int i=0; i < noutputs(); i++) {
        if (operands_[i].original_tensor.defined() && dtype(i) != operands_[i].original_tensor.scalar_type()) {
          operands_[i].original_tensor.copy_(operands_[i].tensor);
          operands_[i].tensor = operands_[i].original_tensor;
        }
      }
    }
  }

  Tensor input(int arg=0) const {
    AT_ASSERT(arg >= 0 && arg < ntensors() - num_outputs_);
    return operands_[num_outputs_ + arg].tensor;
  }

  /// Removes an operand from this iterator
  void remove_operand(int arg);
  /// Removes a dimension from this iterator
  void remove_dimension(int dim);
  /// Shrinks an iterated dimension
  void narrow(int dim, int64_t start, int64_t size);
  /// Narrows every dim after and including `start_dim` to size one.
  void select_all_keeping_dim(int start_dim, IntArrayRef starts);
  /// Replaces the data pointer and strides for the operand at index `arg`
  void replace_operand(int arg, void* data, IntArrayRef stride);

  /// Splits this TensorIterator into two iterators. Together they iterate over
  /// the entire operation. Used by `with_32bit_indexing()`.
  std::unique_ptr<TensorIterator> split(int dim);

  /// Returns the dimension with the largest extent: (size[dim]-1) * stride[dim]
  int get_dim_to_split() const;

  template <typename T>
  T scalar_value(int arg) {
    auto& op = operands_[arg];
    return at::detail::load<T>(op.data, op.tensor.scalar_type());
  }

  void for_each(const loop_t& loop);
  void for_each(const loop2d_t& loop);

  void parallel_reduce(const loop2d_t& loop);

  void serial_for_each(const loop_t& loop, Range range) const;
  void serial_for_each(const loop2d_t& loop, Range range) const;

  /// Create a strides array for a Tensor with shape of this iterator. The
  /// parameter `element_size` specifies the size of Tensor's data type in
  /// bytes (e.g. `4` for `float`)
  DimVector compatible_stride(int element_size) const;

  /// Inverts the re-ordering done by reorder_dimensions. This can only be
  /// called *before* coalesce_dimensions() is called.
  DimVector invert_perm(IntArrayRef input) const;

  /// Helper functions for CPU iteration
  DimVector get_dim_strides(int dim) const;
  DimVector get_strides() const;
  DimVector get_inner_strides() const { return get_dim_strides(0); }
  PtrVector get_data_ptrs(ArrayRef<char*> base, IntArrayRef counter) const;
  PtrVector get_base_ptrs() const;

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

  void set_check_mem_overlap(bool check_mem_overlap) {
    check_mem_overlap_ = check_mem_overlap;
  }

  /// Construction
  void add_output(const Tensor& output) {
    operands_.emplace_back(output);
    num_outputs_++;
  }

  void add_output(const Tensor& input, Device device, ScalarType dtype) {
    operands_.emplace_back(input, device, dtype);
    num_outputs_++;
  }

  void add_input(const Tensor& input) {
    operands_.emplace_back(input);
  }

  void add_input(const Tensor& input, Device device, ScalarType dtype) {
    operands_.emplace_back(input, device, dtype);
  }

  void dont_compute_common_dtype() {
    compute_common_dtype_strategy_ = CommonDTypeStrategy::COMPUTE_NONE;
  }

  void compute_common_dtype_only_for_inputs() {
    compute_common_dtype_strategy_ = CommonDTypeStrategy::COMPUTE_INPUTS;
  }

  void dont_resize_outputs() {
    resize_outputs_ = false;
  }

  void build();

protected:
  void mark_outputs();
  void check_mem_overlaps();
  void compute_shape();
  void compute_strides();
  void reorder_dimensions();
  void permute_dimensions(IntArrayRef perm);
  void compute_types();
  std::tuple<Device, ScalarType> compute_common_type();
  void allocate_outputs();
#ifdef BUILD_NAMEDTENSOR
  void compute_names();
  void propagate_names_to_outputs();
#endif
  void coalesce_dimensions();

protected:
  DimVector shape_;
  DimVector perm_;
#ifdef BUILD_NAMEDTENSOR
  NameVector names_;
#endif
  SmallVector<OperandInfo, 4> operands_;
  int num_outputs_ = 0;
  CommonDTypeStrategy compute_common_dtype_strategy_ = CommonDTypeStrategy::COMPUTE_ALL;
  bool has_coalesced_dimensions_ = false;
  bool accumulate_ = false;
  bool resize_outputs_ = true;
  bool is_reduction_ = false;
  bool allow_cpu_scalars_ = false;
  bool promote_gpu_output_dtypes_ = false;
  bool final_output_ = true;
  bool check_mem_overlap_ = false;
};
/// A container-like struct that acts as if it contains splits of a
/// TensorIterator that can use 32-bit indexing. Taken together the splits cover
/// the original TensorIterator.
struct CAFFE2_API SplitUntil32Bit {
  struct CAFFE2_API iterator {
    iterator() {};
    iterator(const TensorIterator& iter);
    iterator(iterator&&) = default;

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

  SplitUntil32Bit(const TensorIterator& iter) : iter(iter) {}

  iterator begin() const;
  iterator end() const;

private:
  const TensorIterator& iter;
};

}  // namespace at
