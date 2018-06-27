#pragma once

#include <ATen/ATen.h>
#include <ATen/SmallVector.h>
#include <ATen/optional.h>
#include <ATen/detail/ScalarTypeConversions.h>

// TensorIterator is a helper class for element-wise operations, such as
// arithmetic, comparisions, and trigonometric functions. It handles
// broadcasting and type conversions of operands.
//
// The files Loops.h and Loops.cuh provide functions to build kernels that
// use TensorIterator.
//
// Example:
//
//   auto iter = TensorIterator::Builder()
//      .add_output(output)
//      .add_input(input)
//      .build()
//
// [MyKernel.cpp / MyKernel.cu]
//   cpu_binary_kernel(iter, [](float a, float b) {
//     return a + b;
//   });
//
//   gpu_binary_kernel(iter, []GPU_LAMBDA(float a, float b) -> float {
//     return a + b;
//   });
//
// This inspired by NumPy's Array Iterator API (NpyIter).

namespace at {

struct OperandInfo {
  OperandInfo() {}
  OperandInfo(const Tensor& t) : tensor_(const_cast<Tensor*>(&t)) {}

  /// Stride after broadcasting. The stride is in bytes, not number of elements.
  DimVector stride_;

  /// The original tensor operand. Note that the strides, data pointer, and
  /// other attributes may differ from due to dimension reordering and
  /// coalescing.
  Tensor* tensor_;

  /// The desired type for the operand. This may be different from the actual
  /// tensor type, in which case casting is necessary.
  Type* type_ = nullptr;

  /// The data pointer. This may be different from tensor.data_ptr() if the
  /// iterator is split.
  void* data_ = nullptr;

  /// True if the kernel needs to handle a cast operation for this operand.
  bool needs_cast_ = false;

  bool is_output_ = false;

  bool is_read_write_ = false;
};

enum class IteratorFlags {
  COMMON_DTYPE = 1,
  ALLOW_CPU_SCALARS = 2,
};

struct SplitUntil32Bit;

struct TensorIterator {
  struct Builder;
  friend struct Builder;

  TensorIterator() {}

  using loop_t = const std::function<void(int, char**, const int64_t*, int64_t)>&;

  static std::unique_ptr<TensorIterator> binary_op(const Tensor& a, const Tensor& b, Tensor& out);
  static TensorIterator reduce_op(const Tensor& a, IntList dims);

  int ndim() const { return shape_.size(); }
  IntList shape() const { return shape_; }
  int64_t numel() const;
  int ntensors() const { return operands_.size(); }

  /// 1-dimensional iteration and no buffering or type conversion
  bool is_trivial_1d() const;

  /// Accessors for each operand
  IntList strides(int arg) const { return operands_[arg].stride_; }
  void* data_ptr(int arg) const;
  const Type& type(int arg=0) const {
    AT_ASSERT(operands_[arg].type_);
    return *operands_[arg].type_;
  }
  ScalarType dtype(int arg) const { return type(arg).scalarType(); }
  Backend backend(int arg=0) const { return type(arg).backend(); }
  bool is_scalar(int arg) const;
  bool is_cpu_scalar(int arg) const;

  Tensor output(int arg=0) const {
    AT_ASSERT(arg < num_outputs_);
    return *operands_[arg].tensor_;
  }

  /// Removes an operand from this iterator
  void remove_operand(int arg);
  /// Removes a dimension from iteration
  void remove_dimension(int dim);
  /// Shrinks an iterated dimension
  void narrow(int dim, int64_t start, int64_t size);

  /// Splits this TensorIterator into two iterators. Together they iterate over
  /// the entire operation. Used by `with_32bit_indexing()`.
  std::unique_ptr<TensorIterator> split();

  template <typename T>
  T scalar_value(int arg) {
    auto& op = operands_[arg];
    return at::detail::load<T>(op.data_, op.tensor_->type().scalarType());
  }

  void for_each(loop_t loop);
  void serial_for_each(loop_t loop, ArrayRef<char*> base_ptrs, IntList inner_strides, int64_t start, int64_t size);

  /// Create a strides array for a Tensor with shape of this iterator. The
  /// parameter `element_size` specifies the size of Tensor's data type in
  /// bytes (e.g. `4` for `float`)
  DimVector compatible_stride(int element_size) const;

  /// Inverts the re-ordering done by reorder_dimensions. This can only be
  /// called *before* coalesce_dimensions() is called.
  DimVector invert_perm(IntList input) const;

  /// Helper functions for CPU iteration
  DimVector make_counter(int64_t linear_offset) const;
  void increment_counter(DimVector& counter, int64_t n) const;
  DimVector get_inner_strides() const;
  SmallVector<char*, 4> get_data_ptrs(ArrayRef<char*> base, IntList counter) const;
  SmallVector<char*, 4> get_base_ptrs() const;

  /// true if the stride computation can use 32-bit arithmetic. Used by GPU kernels
  bool can_use_32bit_indexing() const;

  /// An "iteratable" objet that recursively splits this iterator into sub-iterators
  /// that can use 32-bit indexing.
  SplitUntil32Bit with_32bit_indexing() const;

protected:
  void mark_outputs();
  void compute_shape();
  void compute_strides();
  void reorder_dimensions();
  void compute_common_type();
  void allocate_outputs();
  void coalesce_dimensions();
  void check_type_conversions();

private:
  DimVector shape_;
  DimVector perm_;
  SmallVector<OperandInfo, 4> operands_;
  int num_outputs_ = 0;
};

struct TensorIterator::Builder {
  Builder() : iter_(new TensorIterator()) {};

  Builder& add_output(const Tensor& output) {
    iter_->operands_.emplace_back(output);
    iter_->num_outputs_++;
    return *this;
  }

  Builder& add_input(const Tensor& input) {
    iter_->operands_.emplace_back(input);
    return *this;
  }

  std::unique_ptr<TensorIterator> build();

private:
  std::unique_ptr<TensorIterator> iter_;
};

/// A container-like struct that acts as if it contains splits of a
/// TensorIterator that can use 32-bit indexing. Taken together the splits cover
/// the original TensorIterator.
struct SplitUntil32Bit {
  struct iterator {
    iterator() {};
    iterator(const TensorIterator& iter);

    TensorIterator& operator*() const;
    iterator& operator++();
    bool operator!=(const iterator& other) {
      return !vec.empty() || !other.vec.empty();
    }

    /// stack of  queue
    std::vector<std::unique_ptr<TensorIterator>> vec;
  };

  SplitUntil32Bit(const TensorIterator& iter) : iter(iter) {}

  iterator begin() const;
  iterator end() const;

private:
  const TensorIterator& iter;
};

}  // namespace at
