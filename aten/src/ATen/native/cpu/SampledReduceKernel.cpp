#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/SampledReduceKernel.h>
#include <c10/util/irange.h>

namespace at { namespace native {

namespace {

template <typename scalar_t, typename index_t>
struct SampledReduceImpl {
  scalar_t* out_ptr; // out or grad_out
  scalar_t* grad_left_ptr;
  scalar_t* grad_right_ptr;
  scalar_t* left_ptr;
  scalar_t* right_ptr;
  index_t* left_index_ptr;
  index_t* right_index_ptr;
  int64_t left_max_size, right_max_size;
  int64_t M, K; // view out as {M, K}

  // forward
  SampledReduceImpl(
      const Tensor& output,
      const Tensor& self,
      const Tensor& other,
      const Tensor& left_index,
      const Tensor& right_index)
    : out_ptr(output.data_ptr<scalar_t>())
    , left_ptr(self.data_ptr<scalar_t>())
    , right_ptr(other.data_ptr<scalar_t>())
    , left_max_size(self.size(0))
    , right_max_size(other.size(0))
    , M(output.size(0))
    , K(output.size(1)) {

    left_index_ptr = left_index.defined() ? left_index.data_ptr<index_t>() : nullptr;
    right_index_ptr = right_index.defined() ? right_index.data_ptr<index_t>() : nullptr;

    grad_left_ptr = nullptr;
    grad_right_ptr = nullptr;
  }

  // backward
  SampledReduceImpl(
      const Tensor& grad_output,
      const Tensor& grad_left,
      const Tensor& grad_right,
      const Tensor& self,
      const Tensor& other,
      const Tensor& left_index,
      const Tensor& right_index)
    : out_ptr(grad_output.data_ptr<scalar_t>())
    , left_ptr(self.data_ptr<scalar_t>())
    , right_ptr(other.data_ptr<scalar_t>())
    , left_max_size(-1)
    , right_max_size(-1)
    , M(grad_output.size(0))
    , K(grad_output.size(1)) {

    left_index_ptr = left_index.defined() ? left_index.data_ptr<index_t>() : nullptr;
    right_index_ptr = right_index.defined() ? right_index.data_ptr<index_t>() : nullptr;

    grad_left_ptr = grad_left.defined() ? grad_left.data_ptr<scalar_t>() : nullptr;
    grad_right_ptr = grad_right.defined() ? grad_right.data_ptr<scalar_t>() : nullptr;
  }

  using Vec = vec::Vectorized<vec::vec_scalar_t<scalar_t>>;

  template <typename F>
  void forward(const F& func) {
    at::parallel_for(0, M, internal::GRAIN_SIZE / K, [&](int64_t begin, int64_t end) {
      for (const auto i: c10::irange(begin, end)) {
        index_t left_offset, right_offset;
        if (left_index_ptr != nullptr) {
          left_offset = left_index_ptr[i];
          TORCH_CHECK(left_offset >= 0 && left_offset < left_max_size,
              "sampled_reduce: Expected left index value in [0 and ", left_max_size, ") got ", left_offset);
        } else {
          left_offset = i;
        }
        if (right_index_ptr != nullptr) {
          right_offset = right_index_ptr[i];
          TORCH_CHECK(right_offset >= 0 && right_offset < right_max_size,
              "sampled_reduce: Expected right index value in [0 and ", right_max_size, ") got ", right_offset);
        } else {
          right_offset = i;
        }
        vec::map2<scalar_t>(
            [&](Vec x, Vec y) { return func(x, y); },
            out_ptr + i * K,
            left_ptr + left_offset * K,
            right_ptr + right_offset * K,
            K);
      }
    });
  }

  void backward(BinaryReductionType reduce_op) {
    at::parallel_for(0, M, internal::GRAIN_SIZE / K, [&](int64_t begin, int64_t end) {
      for (const auto i : c10::irange(begin, end)) {
        index_t left_offset, right_offset;
        left_offset = left_index_ptr != nullptr ? left_index_ptr[i] : i;
        right_offset = right_index_ptr != nullptr ? right_index_ptr[i] : i;

        if (grad_left_ptr != nullptr) {
          if (reduce_op == BinaryReductionType::MUL) {
            vec::map2<scalar_t>(
                [&](Vec grad, Vec b) { return grad * b; },
                grad_left_ptr + i * K,
                out_ptr + i * K,
                right_ptr + right_offset * K,
                K);
          } else {
            vec::map2<scalar_t>(
                [&](Vec grad, Vec b) { return grad / b; },
                grad_left_ptr + i * K,
                out_ptr + i * K,
                right_ptr + right_offset * K,
                K);
          }
        }
        if (grad_right_ptr != nullptr) {
          if (reduce_op == BinaryReductionType::MUL) {
            vec::map2<scalar_t>(
                [&](Vec grad, Vec a) { return grad * a; },
                grad_right_ptr + i * K,
                out_ptr + i * K,
                left_ptr + left_offset * K,
                K);
          } else {
            vec::map3<scalar_t>(
                [&](Vec grad, Vec a, Vec b) { return grad.neg() * a / (b * b); },
                grad_right_ptr + i * K,
                out_ptr + i * K,
                left_ptr + left_offset * K,
                right_ptr + right_offset * K,
                K);
          }
        }
      }
    });
  }
};

inline ScalarType index_type(const Tensor& left, const Tensor& right) {
  return left.defined() ? left.scalar_type()
      : right.defined() ? right.scalar_type() : ScalarType::Long;
}

void sampled_reduce_kernel(
    const Tensor& output,
    const Tensor& self,
    const Tensor& other,
    const Tensor& left_index,
    const Tensor& right_index,
    BinaryReductionType reduce_op) {

  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, self.scalar_type(), "sampled_reduce_kernel", [&]() {
    AT_DISPATCH_INDEX_TYPES(index_type(left_index, right_index), "sampled_reduce_indices", [&]() {

      SampledReduceImpl<scalar_t, index_t> op{output, self, other, left_index, right_index};

      using Vec = vec::Vectorized<vec::vec_scalar_t<scalar_t>>;
      if (reduce_op == BinaryReductionType::ADD) {
        op.forward([](Vec& x, Vec& y) {return x + y; });
      } else if (reduce_op == BinaryReductionType::SUB) {
        op.forward([](Vec& x, Vec& y) {return x - y; });
      } else if (reduce_op == BinaryReductionType::MUL) {
        op.forward([](Vec& x, Vec& y) {return x * y; });
      } else {
        op.forward([](Vec& x, Vec& y) {return x / y; });
      }
    });
  });
}

void sampled_reduce_backward_kernel(
    const Tensor& grad_left,
    const Tensor& grad_right,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& other,
    const Tensor& left_index,
    const Tensor& right_index,
    BinaryReductionType reduce_op) {
  TORCH_CHECK(reduce_op == BinaryReductionType::MUL || reduce_op == BinaryReductionType::DIV);
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, self.scalar_type(), "sampled_reduce_backward_kernel", [&]() {
    AT_DISPATCH_INDEX_TYPES(index_type(left_index, right_index), "sampled_reduce_backward_indices", [&]() {
      SampledReduceImpl<scalar_t, index_t> op{grad_output, grad_left, grad_right, self, other, left_index, right_index};
      op.backward(reduce_op);
    });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(sampled_reduce_stub, &sampled_reduce_kernel);
REGISTER_DISPATCH(sampled_reduce_backward_stub, &sampled_reduce_backward_kernel);

}} // at::native
