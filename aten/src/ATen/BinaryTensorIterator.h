#pragma once

#include "c10/util/Exception.h"
#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorMeta.h>
#include <ATen/core/Range.h>
#include <ATen/core/TensorBody.h>
#include <c10/macros/Export.h>
#include <c10/util/FunctionRef.h>

namespace at {

namespace internal {
constexpr int64_t BINARY_TI_GRAIN_SIZE = 32768;
}

class TensorIteratorConfig;

enum class FunctionConvention : uint8_t { INPLACE, FUNCTIONAL, OUT };

class TORCH_API BinaryTensorIteratorBase : public impl::MetaBase {
 public:
  BinaryTensorIteratorBase() = default;

  // The inner-loop function operates on the fastest moving dimension. It
  // implements element-wise operations in terms of 2-d strided tensors.
  //
  // Arguments:
  //  data: data pointers for each operand (length `ntensors`)
  //  strides: stride for each operand (length `ntensors`)
  //  size0: size of inner loop
  //  size1: size of second inner-most loop
  using loop2d_t = c10::function_ref<
      void(char** data, const int64_t* strides, int64_t size0, int64_t size1)>;

  template <typename loop1d_t,
            std::enable_if_t<std::is_convertible<
              loop1d_t, c10::function_ref<void(char**, const int64_t* strides, int64_t size)>
            >::value, int> = 0>
  void for_each(loop1d_t loop, int64_t grain_size = at::internal::BINARY_TI_GRAIN_SIZE) {
    for_each(loop_2d_from_1d(loop), grain_size);
  }

  void for_each(
      loop2d_t loop,
      int64_t grain_size = at::internal::BINARY_TI_GRAIN_SIZE);

  template <typename loop1d_t,
            std::enable_if_t<std::is_convertible<
              loop1d_t, c10::function_ref<void(char**, const int64_t* strides, int64_t size)>
            >::value, int> = 0>
  void serial_for_each(loop1d_t loop, Range range) {
    serial_for_each(loop_2d_from_1d(loop), range);
  }

  void serial_for_each(loop2d_t loop, Range range) const;

  void setup(
      const Tensor& a,
      const Tensor& b,
      const Tensor& out,
      const TensorIteratorConfig& config);

  void set_output(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override;

  void cast_output_if_necessary();

  ScalarType common_dtype() const {
    return common_dtype_;
  }

 protected:
  Device common_device() const {
      return common_device_;
  }

  // used in structural kernel implementation to check types, etc
  const Tensor& output() const {
      return out_;
  }

 private:
  using StrideVector = SmallVector<int64_t, 6>;
  using PtrVector = SmallVector<char*, 4>;

  Tensor a_, b_;
  Tensor out_;

  // optional, used when output cast is done via materialization
  Tensor output_buffer_;

  FunctionConvention convention_;

  NameVector names_;
  DimVector shape_;
  ScalarType common_dtype_ = ScalarType::Undefined;

  int64_t numel_ = -1;

  bool input_needs_broadcast_ = false;
  bool input_needs_type_promotion_ = false;
  bool output_needs_resize_ = false;
  bool output_needs_type_promotion_ = false;


  // "plan" related part
  Device common_device_ = kCPU;
  bool input_needs_fused_cast_ = false;

  void setup_type_and_device(const TensorIteratorConfig& config);
  int ntensors() const {
    return 3;
  }

  void compute_type_promotion();

  // the output for for_each() to write to
  // will return output_buffer_ if output type cast is required
  const Tensor& get_foreach_output() const {
    if (output_buffer_.defined()) {
      return output_buffer_;
    }
    else {
      return out_;
    }
  }

  // output parameter is required since it may be out_ or output_buffer_
  // avoid calling get_foreach_output() for each method
  // TODO: probably don't need all the three metods (maybe one method to just set strides/data pointer up???)
  StrideVector get_strides(const Tensor& output) const;
  PtrVector get_base_ptrs(const Tensor& output) const;
  PtrVector get_data_ptrs(ArrayRef<char*> base, IntArrayRef counter, const Tensor& output) const;

  template <typename loop1d_t>
  auto loop_2d_from_1d(const loop1d_t& loop) {
    return
        [loop, ntensor = ntensors()](
            char** base, const int64_t* strides, int64_t size0, int64_t size1) {
          PtrVector data(base, base + ntensor);
          const int64_t* outer_strides = &strides[ntensor];
          for (int64_t i = 0; i < size1; i++) {
            if (i > 0) {
              for (int64_t arg = 0; arg < ntensor; arg++) {
                data[arg] += outer_strides[arg];
              }
            }
            loop(data.data(), strides, size0);
          }
        };
  }
};

} // namespace at
