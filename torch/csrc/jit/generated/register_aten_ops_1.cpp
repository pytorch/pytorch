#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/custom_operator.h"

#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/jit/interned_strings.h"

#include "torch/csrc/utils/functional.h"
#include "torch/csrc/autograd/generated/variable_factories.h"

#include <ATen/ATen.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// @generated from tools/autograd/templates/register_aten_ops_1.cpp

// NOTE [Sharded File]: This file is generated in a sharded fashion to speed up
// incremental rebuilds. See the comment at the top of
// templates/VariableType.cpp for an analogous, in-depth discussion.
//
// Note that unlike VariableType.cpp, when sharding this file we take
// care to generate all overloads of a particular name in a single
// file and in a particular order. See gen_jit_dispatch.py for
// details.

namespace torch { namespace jit {

using autograd::Variable;
using autograd::variable_list;
using at::Scalar;
using at::Tensor;
using at::TensorOptions;
using at::DeviceGuard;

namespace {

inline at::optional<at::Device> deviceForInputs(Stack & stack, size_t N) {
  if(N == 0)
    return c10::nullopt;
  auto t = (stack.end() - N)->toTensor();
  return c10::make_optional(t.device());
}

template<size_t N>
std::array<bool, N> as_bool_array(at::ArrayRef<int64_t> vec) {
  std::array<bool, N> res;
  JIT_ASSERT(vec.size() == N);
  std::copy(vec.begin(), vec.end(), res.begin());
  return res;
}

RegisterOperators reg({
  Operator(
  "aten::get_device(Tensor self) -> int",
  [](Stack & stack) {
      autograd::profiler::RecordFunction record("get_device");
      auto result = at::get_device(
          (std::move(peek(stack, 0, 1))).toTensor()
      );
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
  }
  ),
  Operator(
      "aten::storage_offset(Tensor self) -> int",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("storage_offset");
          auto result = ((std::move(peek(stack, 0, 1))).toTensor()).storage_offset();
          drop(stack, 1);
          pack(stack, std::move(result));
          return 0;
      }
  ),

  // Generated operators
  Operator(
      "aten::__ilshift__(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("__ilshift__");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).__ilshift__(
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::__ilshift__(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("__ilshift__");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).__ilshift__(
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::__irshift__(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("__irshift__");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).__irshift__(
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::__irshift__(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("__irshift__");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).__irshift__(
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::__lshift__(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("__lshift__");
      
          auto result_ = at::__lshift__(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::__lshift__(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("__lshift__");
      
          auto result_ = at::__lshift__(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::__rshift__(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("__rshift__");
      
          auto result_ = at::__rshift__(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::__rshift__(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("__rshift__");
      
          auto result_ = at::__rshift__(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_argmin(Tensor self, int dim, bool keepdim=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_argmin");
      
          auto result_ = at::_argmin(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_cast_Byte(Tensor self, bool non_blocking=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_cast_Byte");
      
          auto result_ = at::_cast_Byte(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_cast_Int(Tensor self, bool non_blocking=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_cast_Int");
      
          auto result_ = at::_cast_Int(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_cholesky_helper(Tensor self, bool upper) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_cholesky_helper");
      
          auto result_ = at::_cholesky_helper(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_ctc_loss_backward(Tensor grad, Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, Tensor neg_log_likelihood, Tensor log_alpha, int blank) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_ctc_loss_backward");
      
          auto result_ = at::_ctc_loss_backward(
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toTensor(),
              (std::move(peek(stack, 3, 8))).toIntList()->elements(),
              (std::move(peek(stack, 4, 8))).toIntList()->elements(),
              (std::move(peek(stack, 5, 8))).toTensor(),
              (std::move(peek(stack, 6, 8))).toTensor(),
              (std::move(peek(stack, 7, 8))).toInt()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_cudnn_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor? weight_buf, Tensor hx, Tensor? cx, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_cudnn_rnn");
      
          auto result_ = at::_cudnn_rnn(
              (std::move(peek(stack, 0, 15))).toTensor(),
              (std::move(peek(stack, 1, 15))).toTensorList()->elements(),
              (std::move(peek(stack, 2, 15))).toInt(),
              (std::move(peek(stack, 3, 15))).toTensor(),
              (std::move(peek(stack, 4, 15))).toTensor(),
              (std::move(peek(stack, 5, 15))).toTensor(),
              (std::move(peek(stack, 6, 15))).toInt(),
              (std::move(peek(stack, 7, 15))).toInt(),
              (std::move(peek(stack, 8, 15))).toInt(),
              (std::move(peek(stack, 9, 15))).toBool(),
              (std::move(peek(stack, 10, 15))).toDouble(),
              (std::move(peek(stack, 11, 15))).toBool(),
              (std::move(peek(stack, 12, 15))).toBool(),
              (std::move(peek(stack, 13, 15))).toIntList()->elements(),
              (std::move(peek(stack, 14, 15))).toTensor()
          );
          drop(stack, 15);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_cufft_get_plan_cache_size() -> int",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_cufft_get_plan_cache_size");
      
          auto result_ = at::_cufft_get_plan_cache_size(
          
          );
          drop(stack, 0);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_dim_arange(Tensor like, int dim) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_dim_arange");
      
          auto result_ = at::_dim_arange(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_embedding_bag_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, int num_weights, bool scale_grad_by_freq, int mode, bool sparse) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_embedding_bag_backward");
      
          auto result_ = at::_embedding_bag_backward(
              (std::move(peek(stack, 0, 10))).toTensor(),
              (std::move(peek(stack, 1, 10))).toTensor(),
              (std::move(peek(stack, 2, 10))).toTensor(),
              (std::move(peek(stack, 3, 10))).toTensor(),
              (std::move(peek(stack, 4, 10))).toTensor(),
              (std::move(peek(stack, 5, 10))).toTensor(),
              (std::move(peek(stack, 6, 10))).toInt(),
              (std::move(peek(stack, 7, 10))).toBool(),
              (std::move(peek(stack, 8, 10))).toInt(),
              (std::move(peek(stack, 9, 10))).toBool()
          );
          drop(stack, 10);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_fft_with_size(Tensor self, int signal_ndim, bool complex_input, bool complex_output, bool inverse, int[] checked_signal_sizes, bool normalized, bool onesided, int[] output_sizes) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_fft_with_size");
      
          auto result_ = at::_fft_with_size(
              (std::move(peek(stack, 0, 9))).toTensor(),
              (std::move(peek(stack, 1, 9))).toInt(),
              (std::move(peek(stack, 2, 9))).toBool(),
              (std::move(peek(stack, 3, 9))).toBool(),
              (std::move(peek(stack, 4, 9))).toBool(),
              (std::move(peek(stack, 5, 9))).toIntList()->elements(),
              (std::move(peek(stack, 6, 9))).toBool(),
              (std::move(peek(stack, 7, 9))).toBool(),
              (std::move(peek(stack, 8, 9))).toIntList()->elements()
          );
          drop(stack, 9);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_gesv_helper(Tensor self, Tensor A) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_gesv_helper");
      
          auto result_ = at::_gesv_helper(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_inverse_helper(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_inverse_helper");
      
          auto result_ = at::_inverse_helper(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_log_softmax");
      
          auto result_ = at::_log_softmax(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_log_softmax_backward_data");
      
          auto result_ = at::_log_softmax_backward_data(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toInt(),
              (std::move(peek(stack, 3, 4))).toTensor()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_masked_scale(Tensor self, Tensor mask, float scale) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_masked_scale");
      
          auto result_ = at::_masked_scale(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toDouble()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_pack_padded_sequence(Tensor input, Tensor lengths, bool batch_first) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_pack_padded_sequence");
      
          auto result_ = at::_pack_padded_sequence(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_pdist_forward(Tensor self, float p=2) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_pdist_forward");
      
          auto result_ = at::_pdist_forward(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toDouble()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_s_copy_from(Tensor self, Tensor dst, bool non_blocking=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_s_copy_from");
      
          auto result_ = at::_s_copy_from(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_s_where(Tensor condition, Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_s_where");
      
          auto result_ = at::_s_where(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_shape_as_tensor(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_shape_as_tensor");
      
          auto result_ = at::_shape_as_tensor(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_softmax");
      
          auto result_ = at::_softmax(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_softmax_backward_data");
      
          auto result_ = at::_softmax_backward_data(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toInt(),
              (std::move(peek(stack, 3, 4))).toTensor()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_abs(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_abs_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_abs_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_acos(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_acos");
      
          auto result_ = at::_th_acos(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_addbmm_out");
          auto result = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::_th_addbmm_out(
              result,
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toTensor(),
              (std::move(peek(stack, 3, 6))).toScalar(),
              (std::move(peek(stack, 4, 6))).toScalar()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_addcdiv");
      
          auto result_ = at::_th_addcdiv(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toTensor(),
              (std::move(peek(stack, 3, 4))).toScalar()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_addcmul_out");
          auto result = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::_th_addcmul_out(
              result,
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toTensor(),
              (std::move(peek(stack, 3, 5))).toScalar()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_addmm");
      
          auto result_ = at::_th_addmm(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toTensor(),
              (std::move(peek(stack, 3, 5))).toScalar(),
              (std::move(peek(stack, 4, 5))).toScalar()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_addmv");
      
          auto result_ = at::_th_addmv(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toTensor(),
              (std::move(peek(stack, 3, 5))).toScalar(),
              (std::move(peek(stack, 4, 5))).toScalar()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_addr_(Tensor(a!) self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_addr_");
          auto self = (std::move(peek(stack, 0, 5))).toTensor();
          auto result_ = at::_th_addr_(
              self,
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toTensor(),
              (std::move(peek(stack, 3, 5))).toScalar(),
              (std::move(peek(stack, 4, 5))).toScalar()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_all(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_all_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_th_all_out(
              result,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toInt(),
              (std::move(peek(stack, 2, 4))).toBool()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_and(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_and_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_and_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_and(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_and_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_and_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_any(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_any_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_th_any_out(
              result,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toInt(),
              (std::move(peek(stack, 2, 4))).toBool()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_arange(Scalar end, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_arange_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_arange_out(
              result,
              (std::move(peek(stack, 0, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_arange(Scalar start, Scalar end, Scalar step, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_arange_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_th_arange_out(
              result,
              (std::move(peek(stack, 0, 4))).toScalar(),
              (std::move(peek(stack, 1, 4))).toScalar(),
              (std::move(peek(stack, 2, 4))).toScalar()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_asin(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_asin_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_asin_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_atan2(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_atan2");
      
          auto result_ = at::_th_atan2(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_bmm(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_bmm_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_bmm_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_btrisolve(Tensor self, Tensor LU_data, Tensor LU_pivots) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_btrisolve");
      
          auto result_ = at::_th_btrisolve(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_ceil(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_ceil");
      
          auto result_ = at::_th_ceil(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_clamp_max(Tensor self, Scalar max, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_clamp_max_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_clamp_max_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_clamp_min(Tensor self, Scalar min) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_clamp_min");
      
          auto result_ = at::_th_clamp_min(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_copy_ignoring_overlaps_(Tensor(a!) self, Tensor src) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_copy_ignoring_overlaps_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_copy_ignoring_overlaps_(
              self,
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_cos(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_cos_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_cos_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_cross(Tensor self, Tensor other, int dim=-1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_cross");
      
          auto result_ = at::_th_cross(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toInt()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_digamma_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_digamma_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::_th_digamma_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_eq(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_eq_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_eq_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_eq(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_eq_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_eq_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_equal(Tensor self, Tensor other) -> bool",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_equal");
      
          auto result_ = at::_th_equal(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_erf(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_erf");
      
          auto result_ = at::_th_erf(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_erfc(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_erfc");
      
          auto result_ = at::_th_erfc(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_erfinv(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_erfinv");
      
          auto result_ = at::_th_erfinv(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_expm1(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_expm1");
      
          auto result_ = at::_th_expm1(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_exponential_(Tensor(a!) self, float lambd=1, *, Generator generator=None) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_exponential_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = at::_th_exponential_(
              self,
              (std::move(peek(stack, 1, 3))).toDouble(),
              nullptr
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_fill_(Tensor(a!) self, Tensor value) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_fill_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_fill_(
              self,
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_fill_(Tensor(a!) self, Scalar value) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_fill_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_fill_(
              self,
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_fmod(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_fmod");
      
          auto result_ = at::_th_fmod(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_fmod(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_fmod");
      
          auto result_ = at::_th_fmod(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_frac(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_frac_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_frac_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_gather(Tensor self, int dim, Tensor index) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_gather");
      
          auto result_ = at::_th_gather(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt(),
              (std::move(peek(stack, 2, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_ge_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_ge_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_ge_(
              self,
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_ge_(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_ge_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_ge_(
              self,
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_geqrf(Tensor self) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_geqrf");
      
          auto result_ = at::_th_geqrf(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_getri_single(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_getri_single_out");
          auto output = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_getri_single_out(
              output,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_gt_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_gt_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_gt_(
              self,
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_gt_(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_gt_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_gt_(
              self,
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_histc");
      
          auto result_ = at::_th_histc(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toInt(),
              (std::move(peek(stack, 2, 4))).toScalar(),
              (std::move(peek(stack, 3, 4))).toScalar()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_index_copy_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = at::_th_index_copy_(
              self,
              (std::move(peek(stack, 1, 4))).toInt(),
              (std::move(peek(stack, 2, 4))).toTensor(),
              (std::move(peek(stack, 3, 4))).toTensor()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_index_select(Tensor self, int dim, Tensor index) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_index_select");
      
          auto result_ = at::_th_index_select(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt(),
              (std::move(peek(stack, 2, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_ior_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_ior_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_ior_(
              self,
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_ior_(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_ior_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_ior_(
              self,
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_ixor_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_ixor_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_ixor_(
              self,
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_ixor_(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_ixor_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_ixor_(
              self,
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_le(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_le");
      
          auto result_ = at::_th_le(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_le(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_le");
      
          auto result_ = at::_th_le(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_lerp_(Tensor(a!) self, Tensor end, Scalar weight) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_lerp_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = at::_th_lerp_(
              self,
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_lgamma(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_lgamma");
      
          auto result_ = at::_th_lgamma(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_log10(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_log10");
      
          auto result_ = at::_th_log10(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_log(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_log_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_log_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_lshift(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_lshift");
      
          auto result_ = at::_th_lshift(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_lshift(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_lshift");
      
          auto result_ = at::_th_lshift(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_lt(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_lt");
      
          auto result_ = at::_th_lt(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_lt(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_lt");
      
          auto result_ = at::_th_lt(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_masked_scatter_(Tensor(a!) self, Tensor mask, Tensor source) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_masked_scatter_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = at::_th_masked_scatter_(
              self,
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_masked_select(Tensor self, Tensor mask) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_masked_select");
      
          auto result_ = at::_th_masked_select(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_max(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_max");
      
          auto result_ = at::_th_max(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_max(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_max");
      
          auto result_ = at::_th_max(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_max(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_max");
      
          auto result_ = at::_th_max(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_mm(Tensor self, Tensor mat2) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_mm");
      
          auto result_ = at::_th_mm(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator generator=None, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_multinomial_out");
          auto result = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::_th_multinomial_out(
              result,
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toInt(),
              (std::move(peek(stack, 2, 5))).toBool(),
              nullptr
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_mv(Tensor self, Tensor vec) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_mv");
      
          auto result_ = at::_th_mv(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_ne(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_ne_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_ne_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_ne(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_ne_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_ne_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_neg(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_neg");
      
          auto result_ = at::_th_neg(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_nonzero(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_nonzero");
      
          auto result_ = at::_th_nonzero(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_normal(Tensor mean, Tensor std, *, Generator generator=None, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_normal_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_th_normal_out(
              output,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              nullptr
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_normal(float mean, Tensor std, *, Generator generator=None, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_normal_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_th_normal_out(
              output,
              (std::move(peek(stack, 0, 4))).toDouble(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              nullptr
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_normal(Tensor mean, float std=1, *, Generator generator=None, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_normal_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_th_normal_out(
              output,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toDouble(),
              nullptr
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_polygamma(int n, Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_polygamma_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_polygamma_out(
              result,
              (std::move(peek(stack, 0, 3))).toInt(),
              (std::move(peek(stack, 1, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_potrf_single(Tensor self, bool upper=True) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_potrf_single");
      
          auto result_ = at::_th_potrf_single(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_pow_(Tensor(a!) self, Tensor exponent) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_pow_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_pow_(
              self,
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_pow_(Tensor(a!) self, Scalar exponent) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_pow_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_pow_(
              self,
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_put_(Tensor(a!) self, Tensor index, Tensor source, bool accumulate=False) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_put_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = at::_th_put_(
              self,
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toTensor(),
              (std::move(peek(stack, 3, 4))).toBool()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_qr(Tensor self) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_qr");
      
          auto result_ = at::_th_qr(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_reciprocal(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_reciprocal_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_reciprocal_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_remainder_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_remainder_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_remainder_(
              self,
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_remainder_(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_remainder_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_remainder_(
              self,
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_renorm");
      
          auto result_ = at::_th_renorm(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toScalar(),
              (std::move(peek(stack, 2, 4))).toInt(),
              (std::move(peek(stack, 3, 4))).toScalar()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_rshift(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_rshift");
      
          auto result_ = at::_th_rshift(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_rshift(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_rshift");
      
          auto result_ = at::_th_rshift(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_rsqrt(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_rsqrt");
      
          auto result_ = at::_th_rsqrt(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_sigmoid(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_sigmoid_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_sigmoid_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_sign(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_sign_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_sign_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_sinh(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_sinh");
      
          auto result_ = at::_th_sinh(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_sqrt(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_sqrt");
      
          auto result_ = at::_th_sqrt(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_std(Tensor self, int dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_std_out");
          auto result = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::_th_std_out(
              result,
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toInt(),
              (std::move(peek(stack, 2, 5))).toBool(),
              (std::move(peek(stack, 3, 5))).toBool()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_take(Tensor self, Tensor index, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_take_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_take_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_tan(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_tan");
      
          auto result_ = at::_th_tan(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_tanh(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_tanh_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_tanh_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_topk");
      
          auto result_ = at::_th_topk(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toInt(),
              (std::move(peek(stack, 2, 5))).toInt(),
              (std::move(peek(stack, 3, 5))).toBool(),
              (std::move(peek(stack, 4, 5))).toBool()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_trace(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_trace");
      
          auto result_ = at::_th_trace(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_tril(Tensor self, int diagonal=0) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_tril");
      
          auto result_ = at::_th_tril(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_triu(Tensor self, int diagonal=0) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_triu");
      
          auto result_ = at::_th_triu(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_trtrs(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_trtrs");
      
          auto result_ = at::_th_trtrs(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toBool(),
              (std::move(peek(stack, 3, 5))).toBool(),
              (std::move(peek(stack, 4, 5))).toBool()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_trunc(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_trunc_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_trunc_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator generator=None) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_uniform_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = at::_th_uniform_(
              self,
              (std::move(peek(stack, 1, 4))).toDouble(),
              (std::move(peek(stack, 2, 4))).toDouble(),
              nullptr
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_var(Tensor self, bool unbiased=True) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_var");
      
          auto result_ = at::_th_var(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_var(Tensor self, int dim, bool unbiased=True, bool keepdim=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_var");
      
          auto result_ = at::_th_var(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toInt(),
              (std::move(peek(stack, 2, 4))).toBool(),
              (std::move(peek(stack, 3, 4))).toBool()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_view(Tensor self, int[] size) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_view");
      
          auto result_ = at::_th_view(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_adaptive_avg_pool3d_forward(Tensor self, int[3] output_size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_adaptive_avg_pool3d_forward_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_thnn_adaptive_avg_pool3d_forward_out(
              output,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toIntList()->elements()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_adaptive_max_pool2d_backward(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_adaptive_max_pool2d_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_adaptive_max_pool2d_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toTensor()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_adaptive_max_pool2d_forward(Tensor self, int[2] output_size) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_adaptive_max_pool2d_forward");
      
          auto result_ = at::_thnn_adaptive_max_pool2d_forward(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_adaptive_max_pool3d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_adaptive_max_pool3d_backward");
      
          auto result_ = at::_thnn_adaptive_max_pool3d_backward(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_avg_pool2d_forward(Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_avg_pool2d_forward_out");
          auto output = (std::move(peek(stack, 6, 7))).toTensor();
          auto result_ = at::_thnn_avg_pool2d_forward_out(
              output,
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toIntList()->elements(),
              (std::move(peek(stack, 2, 7))).toIntList()->elements(),
              (std::move(peek(stack, 3, 7))).toIntList()->elements(),
              (std::move(peek(stack, 4, 7))).toBool(),
              (std::move(peek(stack, 5, 7))).toBool()
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_avg_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_avg_pool3d_backward_out");
          auto grad_input = (std::move(peek(stack, 7, 8))).toTensor();
          auto result_ = at::_thnn_avg_pool3d_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toIntList()->elements(),
              (std::move(peek(stack, 3, 8))).toIntList()->elements(),
              (std::move(peek(stack, 4, 8))).toIntList()->elements(),
              (std::move(peek(stack, 5, 8))).toBool(),
              (std::move(peek(stack, 6, 8))).toBool()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_avg_pool3d_forward(Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_avg_pool3d_forward");
      
          auto result_ = at::_thnn_avg_pool3d_forward(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toIntList()->elements(),
              (std::move(peek(stack, 2, 6))).toIntList()->elements(),
              (std::move(peek(stack, 3, 6))).toIntList()->elements(),
              (std::move(peek(stack, 4, 6))).toBool(),
              (std::move(peek(stack, 5, 6))).toBool()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_binary_cross_entropy_backward");
      
          auto result_ = at::_thnn_binary_cross_entropy_backward(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toTensor(),
              (std::move(peek(stack, 3, 5))).toTensor(),
              (std::move(peek(stack, 4, 5))).toInt()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_col2im_forward(Tensor self, int[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_col2im_forward_out");
          auto output = (std::move(peek(stack, 6, 7))).toTensor();
          auto result_ = at::_thnn_col2im_forward_out(
              output,
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toIntList()->elements(),
              (std::move(peek(stack, 2, 7))).toIntList()->elements(),
              (std::move(peek(stack, 3, 7))).toIntList()->elements(),
              (std::move(peek(stack, 4, 7))).toIntList()->elements(),
              (std::move(peek(stack, 5, 7))).toIntList()->elements()
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_conv3d_forward(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, int[3] padding) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_conv3d_forward");
      
          auto result_ = at::_thnn_conv3d_forward(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toIntList()->elements(),
              (std::move(peek(stack, 3, 6))).toTensor(),
              (std::move(peek(stack, 4, 6))).toIntList()->elements(),
              (std::move(peek(stack, 5, 6))).toIntList()->elements()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_conv_depthwise2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_conv_depthwise2d_forward_out");
          auto output = (std::move(peek(stack, 7, 8))).toTensor();
          auto result_ = at::_thnn_conv_depthwise2d_forward_out(
              output,
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toIntList()->elements(),
              (std::move(peek(stack, 3, 8))).toTensor(),
              (std::move(peek(stack, 4, 8))).toIntList()->elements(),
              (std::move(peek(stack, 5, 8))).toIntList()->elements(),
              (std::move(peek(stack, 6, 8))).toIntList()->elements()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_conv_dilated3d_forward(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, int[3] padding, int[3] dilation) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_conv_dilated3d_forward");
      
          auto result_ = at::_thnn_conv_dilated3d_forward(
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              (std::move(peek(stack, 2, 7))).toIntList()->elements(),
              (std::move(peek(stack, 3, 7))).toTensor(),
              (std::move(peek(stack, 4, 7))).toIntList()->elements(),
              (std::move(peek(stack, 5, 7))).toIntList()->elements(),
              (std::move(peek(stack, 6, 7))).toIntList()->elements()
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_conv_transpose3d_forward(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, int[3] padding, int[3] output_padding, int[3] dilation) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_conv_transpose3d_forward");
      
          auto result_ = at::_thnn_conv_transpose3d_forward(
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toIntList()->elements(),
              (std::move(peek(stack, 3, 8))).toTensor(),
              (std::move(peek(stack, 4, 8))).toIntList()->elements(),
              (std::move(peek(stack, 5, 8))).toIntList()->elements(),
              (std::move(peek(stack, 6, 8))).toIntList()->elements(),
              (std::move(peek(stack, 7, 8))).toIntList()->elements()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_elu_(Tensor(a!) self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_elu_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = at::_thnn_elu_(
              self,
              (std::move(peek(stack, 1, 4))).toScalar(),
              (std::move(peek(stack, 2, 4))).toScalar(),
              (std::move(peek(stack, 3, 4))).toScalar()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_elu_backward(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, Tensor output) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_elu_backward");
      
          auto result_ = at::_thnn_elu_backward(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toScalar(),
              (std::move(peek(stack, 2, 5))).toScalar(),
              (std::move(peek(stack, 3, 5))).toScalar(),
              (std::move(peek(stack, 4, 5))).toTensor()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_elu_forward_(Tensor(a!) self, Scalar alpha, Scalar scale, Scalar input_scale) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_elu_forward_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = at::_thnn_elu_forward_(
              self,
              (std::move(peek(stack, 1, 4))).toScalar(),
              (std::move(peek(stack, 2, 4))).toScalar(),
              (std::move(peek(stack, 3, 4))).toScalar()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_fractional_max_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] output_size, Tensor indices, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_fractional_max_pool2d_backward_out");
          auto grad_input = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::_thnn_fractional_max_pool2d_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toIntList()->elements(),
              (std::move(peek(stack, 3, 6))).toIntList()->elements(),
              (std::move(peek(stack, 4, 6))).toTensor()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_fractional_max_pool2d_forward(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_fractional_max_pool2d_forward");
      
          auto result_ = at::_thnn_fractional_max_pool2d_forward(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toIntList()->elements(),
              (std::move(peek(stack, 2, 4))).toIntList()->elements(),
              (std::move(peek(stack, 3, 4))).toTensor()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_fused_gru_cell_backward(Tensor grad_hy, Tensor workspace, bool has_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_fused_gru_cell_backward");
      
          auto result_ = at::_thnn_fused_gru_cell_backward(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_fused_lstm_cell_backward(Tensor? grad_hy, Tensor? grad_cy, Tensor cx, Tensor cy, Tensor workspace, bool has_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_fused_lstm_cell_backward");
      
          auto result_ = at::_thnn_fused_lstm_cell_backward(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toTensor(),
              (std::move(peek(stack, 3, 6))).toTensor(),
              (std::move(peek(stack, 4, 6))).toTensor(),
              (std::move(peek(stack, 5, 6))).toBool()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_glu_backward(Tensor grad_output, Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_glu_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_glu_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toInt()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_glu_forward(Tensor self, int dim) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_glu_forward");
      
          auto result_ = at::_thnn_glu_forward(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_hardtanh_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = at::_thnn_hardtanh_(
              self,
              (std::move(peek(stack, 1, 3))).toScalar(),
              (std::move(peek(stack, 2, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_hardtanh_backward");
      
          auto result_ = at::_thnn_hardtanh_backward(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toScalar(),
              (std::move(peek(stack, 3, 4))).toScalar()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_hardtanh_forward_(Tensor(a!) self, Scalar min_val, Scalar max_val) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_hardtanh_forward_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = at::_thnn_hardtanh_forward_(
              self,
              (std::move(peek(stack, 1, 3))).toScalar(),
              (std::move(peek(stack, 2, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_im2col_forward(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_im2col_forward_out");
          auto output = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::_thnn_im2col_forward_out(
              output,
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toIntList()->elements(),
              (std::move(peek(stack, 2, 6))).toIntList()->elements(),
              (std::move(peek(stack, 3, 6))).toIntList()->elements(),
              (std::move(peek(stack, 4, 6))).toIntList()->elements()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_l1_loss_backward");
      
          auto result_ = at::_thnn_l1_loss_backward(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toTensor(),
              (std::move(peek(stack, 3, 4))).toInt()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_leaky_relu_backward(Tensor grad_output, Tensor self, Scalar negative_slope, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_leaky_relu_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_leaky_relu_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toScalar()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_leaky_relu_forward(Tensor self, Scalar negative_slope) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_leaky_relu_forward");
      
          auto result_ = at::_thnn_leaky_relu_forward(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_log_sigmoid_backward(Tensor grad_output, Tensor self, Tensor buffer, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_log_sigmoid_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_log_sigmoid_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toTensor()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_log_sigmoid_forward(Tensor self) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_log_sigmoid_forward");
      
          auto result_ = at::_thnn_log_sigmoid_forward(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_max_pool3d_with_indices_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_max_pool3d_with_indices_backward_out");
          auto grad_input = (std::move(peek(stack, 8, 9))).toTensor();
          auto result_ = at::_thnn_max_pool3d_with_indices_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 9))).toTensor(),
              (std::move(peek(stack, 1, 9))).toTensor(),
              (std::move(peek(stack, 2, 9))).toIntList()->elements(),
              (std::move(peek(stack, 3, 9))).toIntList()->elements(),
              (std::move(peek(stack, 4, 9))).toIntList()->elements(),
              (std::move(peek(stack, 5, 9))).toIntList()->elements(),
              (std::move(peek(stack, 6, 9))).toBool(),
              (std::move(peek(stack, 7, 9))).toTensor()
          );
          drop(stack, 9);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_max_pool3d_with_indices_forward(Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_max_pool3d_with_indices_forward");
      
          auto result_ = at::_thnn_max_pool3d_with_indices_forward(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toIntList()->elements(),
              (std::move(peek(stack, 2, 6))).toIntList()->elements(),
              (std::move(peek(stack, 3, 6))).toIntList()->elements(),
              (std::move(peek(stack, 4, 6))).toIntList()->elements(),
              (std::move(peek(stack, 5, 6))).toBool()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_max_unpool2d_backward(Tensor grad_output, Tensor self, Tensor indices, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_max_unpool2d_backward_out");
          auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::_thnn_max_unpool2d_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toTensor(),
              (std::move(peek(stack, 3, 5))).toIntList()->elements()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_max_unpool2d_forward(Tensor self, Tensor indices, int[2] output_size) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_max_unpool2d_forward");
      
          auto result_ = at::_thnn_max_unpool2d_forward(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toIntList()->elements()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_max_unpool3d_backward(Tensor grad_output, Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_max_unpool3d_backward");
      
          auto result_ = at::_thnn_max_unpool3d_backward(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toTensor(),
              (std::move(peek(stack, 3, 6))).toIntList()->elements(),
              (std::move(peek(stack, 4, 6))).toIntList()->elements(),
              (std::move(peek(stack, 5, 6))).toIntList()->elements()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_mse_loss_backward");
      
          auto result_ = at::_thnn_mse_loss_backward(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toTensor(),
              (std::move(peek(stack, 3, 4))).toInt()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_multi_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight, int reduction) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_multi_margin_loss_backward");
      
          auto result_ = at::_thnn_multi_margin_loss_backward(
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              (std::move(peek(stack, 2, 7))).toTensor(),
              (std::move(peek(stack, 3, 7))).toScalar(),
              (std::move(peek(stack, 4, 7))).toScalar(),
              (std::move(peek(stack, 5, 7))).toTensor(),
              (std::move(peek(stack, 6, 7))).toInt()
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_multilabel_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_multilabel_margin_loss_backward_out");
          auto grad_input = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::_thnn_multilabel_margin_loss_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toTensor(),
              (std::move(peek(stack, 3, 6))).toInt(),
              (std::move(peek(stack, 4, 6))).toTensor()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_multilabel_margin_loss_forward(Tensor self, Tensor target, int reduction) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_multilabel_margin_loss_forward");
      
          auto result_ = at::_thnn_multilabel_margin_loss_forward(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toInt()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_reflection_pad1d_forward(Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_reflection_pad1d_forward_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_thnn_reflection_pad1d_forward_out(
              output,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toIntList()->elements()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_reflection_pad2d_backward(Tensor grad_output, Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_reflection_pad2d_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_reflection_pad2d_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toIntList()->elements()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_reflection_pad2d_forward(Tensor self, int[4] padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_reflection_pad2d_forward");
      
          auto result_ = at::_thnn_reflection_pad2d_forward(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_replication_pad1d_forward(Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_replication_pad1d_forward_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_thnn_replication_pad1d_forward_out(
              output,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toIntList()->elements()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_replication_pad2d_backward(Tensor grad_output, Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_replication_pad2d_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_replication_pad2d_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toIntList()->elements()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_replication_pad2d_forward(Tensor self, int[4] padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_replication_pad2d_forward");
      
          auto result_ = at::_thnn_replication_pad2d_forward(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_replication_pad3d_backward(Tensor grad_output, Tensor self, int[6] padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_replication_pad3d_backward");
      
          auto result_ = at::_thnn_replication_pad3d_backward(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toIntList()->elements()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_rrelu_with_noise_(Tensor(a!) self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator generator=None) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_rrelu_with_noise_");
          auto self = (std::move(peek(stack, 0, 6))).toTensor();
          auto result_ = at::_thnn_rrelu_with_noise_(
              self,
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toScalar(),
              (std::move(peek(stack, 3, 6))).toScalar(),
              (std::move(peek(stack, 4, 6))).toBool(),
              nullptr
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_rrelu_with_noise_backward(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_rrelu_with_noise_backward");
      
          auto result_ = at::_thnn_rrelu_with_noise_backward(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toTensor(),
              (std::move(peek(stack, 3, 6))).toScalar(),
              (std::move(peek(stack, 4, 6))).toScalar(),
              (std::move(peek(stack, 5, 6))).toBool()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_sigmoid_backward(Tensor grad_output, Tensor output, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_sigmoid_backward_out");
          auto grad_input = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_thnn_sigmoid_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_sigmoid_forward(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_sigmoid_forward");
      
          auto result_ = at::_thnn_sigmoid_forward(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_smooth_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_smooth_l1_loss_backward_out");
          auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::_thnn_smooth_l1_loss_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toTensor(),
              (std::move(peek(stack, 3, 5))).toInt()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_smooth_l1_loss_forward(Tensor self, Tensor target, int reduction) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_smooth_l1_loss_forward");
      
          auto result_ = at::_thnn_smooth_l1_loss_forward(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toInt()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_soft_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_soft_margin_loss_backward");
      
          auto result_ = at::_thnn_soft_margin_loss_backward(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toTensor(),
              (std::move(peek(stack, 3, 4))).toInt()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_softplus_backward(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, Tensor output) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_softplus_backward");
      
          auto result_ = at::_thnn_softplus_backward(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toScalar(),
              (std::move(peek(stack, 3, 5))).toScalar(),
              (std::move(peek(stack, 4, 5))).toTensor()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_softshrink_backward(Tensor grad_output, Tensor self, Scalar lambd, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_softshrink_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_softshrink_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toScalar()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_softshrink_forward(Tensor self, Scalar lambd) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_softshrink_forward");
      
          auto result_ = at::_thnn_softshrink_forward(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_tanh_backward(Tensor grad_output, Tensor output, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_tanh_backward_out");
          auto grad_input = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_thnn_tanh_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_tanh_forward(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_tanh_forward");
      
          auto result_ = at::_thnn_tanh_forward(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_upsample_bicubic2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_bicubic2d_backward_out");
          auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::_thnn_upsample_bicubic2d_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toIntList()->elements(),
              (std::move(peek(stack, 2, 5))).toIntList()->elements(),
              (std::move(peek(stack, 3, 5))).toBool()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_upsample_bicubic2d_forward(Tensor self, int[2] output_size, bool align_corners) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_bicubic2d_forward");
      
          auto result_ = at::_thnn_upsample_bicubic2d_forward(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toIntList()->elements(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_upsample_bilinear2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_bilinear2d_backward_out");
          auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::_thnn_upsample_bilinear2d_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toIntList()->elements(),
              (std::move(peek(stack, 2, 5))).toIntList()->elements(),
              (std::move(peek(stack, 3, 5))).toBool()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_upsample_bilinear2d_forward(Tensor self, int[2] output_size, bool align_corners) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_bilinear2d_forward");
      
          auto result_ = at::_thnn_upsample_bilinear2d_forward(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toIntList()->elements(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_upsample_linear1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_linear1d_backward_out");
          auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::_thnn_upsample_linear1d_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toIntList()->elements(),
              (std::move(peek(stack, 2, 5))).toIntList()->elements(),
              (std::move(peek(stack, 3, 5))).toBool()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_upsample_linear1d_forward(Tensor self, int[1] output_size, bool align_corners) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_linear1d_forward");
      
          auto result_ = at::_thnn_upsample_linear1d_forward(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toIntList()->elements(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_upsample_nearest1d_forward(Tensor self, int[1] output_size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_nearest1d_forward_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_thnn_upsample_nearest1d_forward_out(
              output,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toIntList()->elements()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_upsample_nearest2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_nearest2d_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_upsample_nearest2d_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toIntList()->elements(),
              (std::move(peek(stack, 2, 4))).toIntList()->elements()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_upsample_nearest2d_forward(Tensor self, int[2] output_size) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_nearest2d_forward");
      
          auto result_ = at::_thnn_upsample_nearest2d_forward(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_upsample_nearest3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_nearest3d_backward");
      
          auto result_ = at::_thnn_upsample_nearest3d_backward(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toIntList()->elements(),
              (std::move(peek(stack, 2, 3))).toIntList()->elements()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_upsample_trilinear3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size, bool align_corners) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_trilinear3d_backward");
      
          auto result_ = at::_thnn_upsample_trilinear3d_backward(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toIntList()->elements(),
              (std::move(peek(stack, 2, 4))).toIntList()->elements(),
              (std::move(peek(stack, 3, 4))).toBool()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_values(Tensor(a) self) -> Tensor(a)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_values");
      
          auto result_ = ((std::move(peek(stack, 0, 1))).toTensor())._values(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_weight_norm(Tensor v, Tensor g, int dim=0) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_weight_norm");
      
          auto result_ = at::_weight_norm(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toInt()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_weight_norm_cuda_interface(Tensor v, Tensor g, int dim=0) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_weight_norm_cuda_interface");
      
          auto result_ = at::_weight_norm_cuda_interface(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toInt()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_weight_norm_differentiable_backward(Tensor grad_w, Tensor saved_v, Tensor saved_g, Tensor saved_norms, int dim) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_weight_norm_differentiable_backward");
      
          auto result_ = at::_weight_norm_differentiable_backward(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toTensor(),
              (std::move(peek(stack, 3, 5))).toTensor(),
              (std::move(peek(stack, 4, 5))).toInt()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::abs(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("abs");
      
          auto result_ = at::abs(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::acos_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("acos_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::acos_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::adaptive_avg_pool1d(Tensor self, int[1] output_size) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("adaptive_avg_pool1d");
      
          auto result_ = at::adaptive_avg_pool1d(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("adaptive_avg_pool2d_backward");
      
          auto result_ = at::adaptive_avg_pool2d_backward(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::adaptive_avg_pool3d(Tensor self, int[3] output_size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("adaptive_avg_pool3d_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::adaptive_avg_pool3d_out(
              output,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toIntList()->elements()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::adaptive_max_pool2d(Tensor self, int[2] output_size) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("adaptive_max_pool2d");
      
          auto result_ = at::adaptive_max_pool2d(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::adaptive_max_pool2d_backward(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("adaptive_max_pool2d_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::adaptive_max_pool2d_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toTensor()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::adaptive_max_pool3d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("adaptive_max_pool3d_backward");
      
          auto result_ = at::adaptive_max_pool3d_backward(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::add(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("add_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::add_out(
              result,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toScalar()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("addbmm");
      
          auto result_ = at::addbmm(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toTensor(),
              (std::move(peek(stack, 3, 5))).toScalar(),
              (std::move(peek(stack, 4, 5))).toScalar()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::addcdiv_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("addcdiv_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = (self).addcdiv_(
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toTensor(),
              (std::move(peek(stack, 3, 4))).toScalar()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("addcmul");
      
          auto result_ = at::addcmul(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toTensor(),
              (std::move(peek(stack, 3, 4))).toScalar()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("addmm_");
          auto self = (std::move(peek(stack, 0, 5))).toTensor();
          auto result_ = (self).addmm_(
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toTensor(),
              (std::move(peek(stack, 3, 5))).toScalar(),
              (std::move(peek(stack, 4, 5))).toScalar()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("addmv_");
          auto self = (std::move(peek(stack, 0, 5))).toTensor();
          auto result_ = at::addmv_(
              self,
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toTensor(),
              (std::move(peek(stack, 3, 5))).toScalar(),
              (std::move(peek(stack, 4, 5))).toScalar()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("addr_out");
          auto result = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::addr_out(
              result,
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toTensor(),
              (std::move(peek(stack, 3, 6))).toScalar(),
              (std::move(peek(stack, 4, 6))).toScalar()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::all(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("all");
      
          auto result_ = at::all(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::all(Tensor self, int dim, bool keepdim=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("all");
      
          auto result_ = at::all(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::allclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> bool",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("allclose");
      
          auto result_ = at::allclose(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toDouble(),
              (std::move(peek(stack, 3, 5))).toDouble(),
              (std::move(peek(stack, 4, 5))).toBool()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("alpha_dropout_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = at::alpha_dropout_(
              self,
              (std::move(peek(stack, 1, 3))).toDouble(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::any(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("any");
      
          auto result_ = at::any(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::any(Tensor self, int dim, bool keepdim=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("any");
      
          auto result_ = at::any(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::arange(Scalar end, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("arange");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 1, 4))).toScalarType())
                  .layout((std::move(peek(stack, 2, 4))).toLayout())
                  .device((std::move(peek(stack, 3, 4))).toDevice());
          auto result_ = torch::arange((std::move(peek(stack, 0, 4))).toScalar(),
          options);
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::arange(Scalar start, Scalar end, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("arange");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 2, 5))).toScalarType())
                  .layout((std::move(peek(stack, 3, 5))).toLayout())
                  .device((std::move(peek(stack, 4, 5))).toDevice());
          auto result_ = torch::arange((std::move(peek(stack, 0, 5))).toScalar(),
          (std::move(peek(stack, 1, 5))).toScalar(),
          options);
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::arange(Scalar start, Scalar end, Scalar step, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("arange");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 3, 6))).toScalarType())
                  .layout((std::move(peek(stack, 4, 6))).toLayout())
                  .device((std::move(peek(stack, 5, 6))).toDevice());
          auto result_ = torch::arange((std::move(peek(stack, 0, 6))).toScalar(),
          (std::move(peek(stack, 1, 6))).toScalar(),
          (std::move(peek(stack, 2, 6))).toScalar(),
          options);
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::argmax(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("argmax");
      
          auto result_ = at::argmax(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::argmax(Tensor self, int dim, bool keepdim=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("argmax");
      
          auto result_ = at::argmax(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::as_strided_(Tensor(a!) self, int[] size, int[] stride) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("as_strided_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = at::as_strided_(
              self,
              (std::move(peek(stack, 1, 3))).toIntList()->elements(),
              (std::move(peek(stack, 2, 3))).toIntList()->elements()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::as_strided_(Tensor(a!) self, int[] size, int[] stride, int storage_offset) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("as_strided_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = at::as_strided_(
              self,
              (std::move(peek(stack, 1, 4))).toIntList()->elements(),
              (std::move(peek(stack, 2, 4))).toIntList()->elements(),
              (std::move(peek(stack, 3, 4))).toInt()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::asin(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("asin");
      
          auto result_ = at::asin(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::atan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("atan2_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).atan2_(
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::atan(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("atan_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::atan_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("avg_pool2d_out");
          auto output = (std::move(peek(stack, 6, 7))).toTensor();
          auto result_ = at::avg_pool2d_out(
              output,
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toIntList()->elements(),
              (std::move(peek(stack, 2, 7))).toIntList()->elements(),
              (std::move(peek(stack, 3, 7))).toIntList()->elements(),
              (std::move(peek(stack, 4, 7))).toBool(),
              (std::move(peek(stack, 5, 7))).toBool()
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("avg_pool3d");
      
          auto result_ = at::avg_pool3d(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toIntList()->elements(),
              (std::move(peek(stack, 2, 6))).toIntList()->elements(),
              (std::move(peek(stack, 3, 6))).toIntList()->elements(),
              (std::move(peek(stack, 4, 6))).toBool(),
              (std::move(peek(stack, 5, 6))).toBool()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::avg_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("avg_pool3d_backward_out");
          auto grad_input = (std::move(peek(stack, 7, 8))).toTensor();
          auto result_ = at::avg_pool3d_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toIntList()->elements(),
              (std::move(peek(stack, 3, 8))).toIntList()->elements(),
              (std::move(peek(stack, 4, 8))).toIntList()->elements(),
              (std::move(peek(stack, 5, 8))).toBool(),
              (std::move(peek(stack, 6, 8))).toBool()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("baddbmm_out");
          auto result = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::baddbmm_out(
              result,
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toTensor(),
              (std::move(peek(stack, 3, 6))).toScalar(),
              (std::move(peek(stack, 4, 6))).toScalar()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::bernoulli(Tensor self, *, Generator generator=None, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("bernoulli_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::bernoulli_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              nullptr
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("bilinear");
      
          auto result_ = at::bilinear(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toTensor(),
              (std::move(peek(stack, 3, 4))).toTensor()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor weight, int reduction) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("binary_cross_entropy_backward");
      
          auto result_ = at::binary_cross_entropy_backward(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toTensor(),
              (std::move(peek(stack, 3, 5))).toTensor(),
              (std::move(peek(stack, 4, 5))).toInt()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::binary_cross_entropy_with_logits(Tensor self, Tensor target, Tensor? weight, Tensor? pos_weight, int reduction) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("binary_cross_entropy_with_logits");
      
          auto result_ = at::binary_cross_entropy_with_logits(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toTensor(),
              (std::move(peek(stack, 3, 5))).toTensor(),
              (std::move(peek(stack, 4, 5))).toInt()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::bmm(Tensor self, Tensor mat2) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("bmm");
      
          auto result_ = at::bmm(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::btrifact(Tensor self, *, bool pivot=True) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("btrifact");
      
          auto result_ = at::btrifact(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::btrifact_with_info(Tensor self, *, bool pivot=True) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("btrifact_with_info");
      
          auto result_ = at::btrifact_with_info(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cat(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cat_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::cat_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensorList()->elements(),
              (std::move(peek(stack, 1, 3))).toInt()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::ceil_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ceil_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::ceil_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::celu_(Tensor(a!) self, Scalar alpha=1.0) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("celu_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::celu_(
              self,
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::clamp_max(Tensor self, Scalar max) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("clamp_max");
      
          auto result_ = at::clamp_max(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::clamp_min_(Tensor(a!) self, Scalar min) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("clamp_min_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::clamp_min_(
              self,
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("clamp_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::clamp_out(
              result,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toOptional<Scalar>(),
              (std::move(peek(stack, 2, 4))).toOptional<Scalar>()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::clone(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("clone");
      
          auto result_ = at::clone(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::conv3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("conv3d");
      
          auto result_ = at::conv3d(
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              (std::move(peek(stack, 2, 7))).toTensor(),
              (std::move(peek(stack, 3, 7))).toIntList()->elements(),
              (std::move(peek(stack, 4, 7))).toIntList()->elements(),
              (std::move(peek(stack, 5, 7))).toIntList()->elements(),
              (std::move(peek(stack, 6, 7))).toInt()
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::conv_transpose3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int groups=1, int[3] dilation=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("conv_transpose3d");
      
          auto result_ = at::conv_transpose3d(
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toTensor(),
              (std::move(peek(stack, 3, 8))).toIntList()->elements(),
              (std::move(peek(stack, 4, 8))).toIntList()->elements(),
              (std::move(peek(stack, 5, 8))).toIntList()->elements(),
              (std::move(peek(stack, 6, 8))).toInt(),
              (std::move(peek(stack, 7, 8))).toIntList()->elements()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("convolution");
      
          auto result_ = at::convolution(
              (std::move(peek(stack, 0, 9))).toTensor(),
              (std::move(peek(stack, 1, 9))).toTensor(),
              (std::move(peek(stack, 2, 9))).toTensor(),
              (std::move(peek(stack, 3, 9))).toIntList()->elements(),
              (std::move(peek(stack, 4, 9))).toIntList()->elements(),
              (std::move(peek(stack, 5, 9))).toIntList()->elements(),
              (std::move(peek(stack, 6, 9))).toBool(),
              (std::move(peek(stack, 7, 9))).toIntList()->elements(),
              (std::move(peek(stack, 8, 9))).toInt()
          );
          drop(stack, 9);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cos(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cos");
      
          auto result_ = at::cos(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cosh(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cosh_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::cosh_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cosine_similarity(Tensor x1, Tensor x2, int dim=1, float eps=1e-08) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cosine_similarity");
      
          auto result_ = at::cosine_similarity(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toInt(),
              (std::move(peek(stack, 3, 4))).toDouble()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cudnn_affine_grid_generator(Tensor theta, int N, int C, int H, int W) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cudnn_affine_grid_generator");
      
          auto result_ = at::cudnn_affine_grid_generator(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toInt(),
              (std::move(peek(stack, 2, 5))).toInt(),
              (std::move(peek(stack, 3, 5))).toInt(),
              (std::move(peek(stack, 4, 5))).toInt()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cudnn_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cudnn_batch_norm");
      
          auto result_ = at::cudnn_batch_norm(
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toTensor(),
              (std::move(peek(stack, 3, 8))).toTensor(),
              (std::move(peek(stack, 4, 8))).toTensor(),
              (std::move(peek(stack, 5, 8))).toBool(),
              (std::move(peek(stack, 6, 8))).toDouble(),
              (std::move(peek(stack, 7, 8))).toDouble()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cudnn_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cudnn_convolution_backward");
      
          auto result_ = at::cudnn_convolution_backward(
              (std::move(peek(stack, 0, 10))).toTensor(),
              (std::move(peek(stack, 1, 10))).toTensor(),
              (std::move(peek(stack, 2, 10))).toTensor(),
              (std::move(peek(stack, 3, 10))).toIntList()->elements(),
              (std::move(peek(stack, 4, 10))).toIntList()->elements(),
              (std::move(peek(stack, 5, 10))).toIntList()->elements(),
              (std::move(peek(stack, 6, 10))).toInt(),
              (std::move(peek(stack, 7, 10))).toBool(),
              (std::move(peek(stack, 8, 10))).toBool(),
              as_bool_array<3>((std::move(peek(stack, 9, 10))).toIntList()->elements())
          );
          drop(stack, 10);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cudnn_convolution_backward_bias(Tensor grad_output) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cudnn_convolution_backward_bias");
      
          auto result_ = at::cudnn_convolution_backward_bias(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cudnn_convolution_transpose_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cudnn_convolution_transpose_backward");
      
          auto result_ = at::cudnn_convolution_transpose_backward(
              (std::move(peek(stack, 0, 11))).toTensor(),
              (std::move(peek(stack, 1, 11))).toTensor(),
              (std::move(peek(stack, 2, 11))).toTensor(),
              (std::move(peek(stack, 3, 11))).toIntList()->elements(),
              (std::move(peek(stack, 4, 11))).toIntList()->elements(),
              (std::move(peek(stack, 5, 11))).toIntList()->elements(),
              (std::move(peek(stack, 6, 11))).toIntList()->elements(),
              (std::move(peek(stack, 7, 11))).toInt(),
              (std::move(peek(stack, 8, 11))).toBool(),
              (std::move(peek(stack, 9, 11))).toBool(),
              as_bool_array<3>((std::move(peek(stack, 10, 11))).toIntList()->elements())
          );
          drop(stack, 11);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cudnn_convolution_transpose_backward_bias(Tensor grad_output) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cudnn_convolution_transpose_backward_bias");
      
          auto result_ = at::cudnn_convolution_transpose_backward_bias(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cudnn_grid_sampler_backward(Tensor self, Tensor grid, Tensor grad_output) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cudnn_grid_sampler_backward");
      
          auto result_ = at::cudnn_grid_sampler_backward(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cumprod(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cumprod_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::cumprod_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cumprod(Tensor self, int dim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cumprod_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::cumprod_out(
              result,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toInt(),
              (std::move(peek(stack, 2, 4))).toScalarType()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cumsum(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cumsum_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::cumsum_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cumsum(Tensor self, int dim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cumsum_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::cumsum_out(
              result,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toInt(),
              (std::move(peek(stack, 2, 4))).toScalarType()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::detach_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("detach_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::detach_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("diag_embed");
      
          auto result_ = at::diag_embed(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toInt(),
              (std::move(peek(stack, 2, 4))).toInt(),
              (std::move(peek(stack, 3, 4))).toInt()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::diag(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("diag_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::diag_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::digamma(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("digamma_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::digamma_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::dist(Tensor self, Tensor other, Scalar p=2) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("dist");
      
          auto result_ = at::dist(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::div_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("div_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).div_(
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::div_(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("div_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).div_(
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::dot(Tensor self, Tensor tensor, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("dot_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::dot_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::dropout(Tensor input, float p, bool train) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("dropout");
      
          auto result_ = at::dropout(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toDouble(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::elu_(Tensor(a!) self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("elu_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = at::elu_(
              self,
              (std::move(peek(stack, 1, 4))).toScalar(),
              (std::move(peek(stack, 2, 4))).toScalar(),
              (std::move(peek(stack, 3, 4))).toScalar()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::elu_backward(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, Tensor output) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("elu_backward");
      
          auto result_ = at::elu_backward(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toScalar(),
              (std::move(peek(stack, 2, 5))).toScalar(),
              (std::move(peek(stack, 3, 5))).toScalar(),
              (std::move(peek(stack, 4, 5))).toTensor()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::embedding_sparse_backward(Tensor grad, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("embedding_sparse_backward");
      
          auto result_ = at::embedding_sparse_backward(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toInt(),
              (std::move(peek(stack, 3, 5))).toInt(),
              (std::move(peek(stack, 4, 5))).toBool()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::empty(int[] size, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("empty");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 1, 4))).toScalarType())
                  .layout((std::move(peek(stack, 2, 4))).toLayout())
                  .device((std::move(peek(stack, 3, 4))).toDevice());
          auto result_ = torch::empty((std::move(peek(stack, 0, 4))).toIntList()->elements(),
          options);
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::empty_like(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("empty_like");
      
          auto result_ = at::empty_like(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::empty_like(Tensor self, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("empty_like");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 1, 4))).toScalarType())
                  .layout((std::move(peek(stack, 2, 4))).toLayout())
                  .device((std::move(peek(stack, 3, 4))).toDevice());
          auto result_ = torch::empty_like((std::move(peek(stack, 0, 4))).toTensor(),
          options);
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::empty_strided(int[] size, int[] stride, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("empty_strided");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 2, 5))).toScalarType())
                  .layout((std::move(peek(stack, 3, 5))).toLayout())
                  .device((std::move(peek(stack, 4, 5))).toDevice());
          auto result_ = torch::empty_strided((std::move(peek(stack, 0, 5))).toIntList()->elements(),
          (std::move(peek(stack, 1, 5))).toIntList()->elements(),
          options);
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::eq(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("eq");
      
          auto result_ = at::eq(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::eq(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("eq");
      
          auto result_ = at::eq(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::erf_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("erf_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::erf_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::erfc_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("erfc_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::erfc_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::erfinv_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("erfinv_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = (self).erfinv_(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::exp(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("exp_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::exp_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("expand");
      
          auto result_ = ((std::move(peek(stack, 0, 3))).toTensor()).expand(
              (std::move(peek(stack, 1, 3))).toIntList()->elements(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::expm1_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("expm1_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::expm1_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::feature_alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("feature_alpha_dropout_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = at::feature_alpha_dropout_(
              self,
              (std::move(peek(stack, 1, 3))).toDouble(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::feature_dropout(Tensor input, float p, bool train) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("feature_dropout");
      
          auto result_ = at::feature_dropout(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toDouble(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::flip(Tensor self, int[] dims) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("flip");
      
          auto result_ = at::flip(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::floor(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("floor_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::floor_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::fmod_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("fmod_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).fmod_(
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::fmod_(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("fmod_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).fmod_(
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::frac(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("frac");
      
          auto result_ = at::frac(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::fractional_max_pool2d(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("fractional_max_pool2d");
      
          auto result_ = at::fractional_max_pool2d(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toIntList()->elements(),
              (std::move(peek(stack, 2, 4))).toIntList()->elements(),
              (std::move(peek(stack, 3, 4))).toTensor()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::fractional_max_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] output_size, Tensor indices, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("fractional_max_pool2d_backward_out");
          auto grad_input = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::fractional_max_pool2d_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toIntList()->elements(),
              (std::move(peek(stack, 3, 6))).toIntList()->elements(),
              (std::move(peek(stack, 4, 6))).toTensor()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::frobenius_norm(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("frobenius_norm_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::frobenius_norm_out(
              result,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toIntList()->elements(),
              (std::move(peek(stack, 2, 4))).toBool()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::full(int[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("full_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::full_out(
              result,
              (std::move(peek(stack, 0, 3))).toIntList()->elements(),
              (std::move(peek(stack, 1, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::ge(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ge_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::ge_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::ge(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ge_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::ge_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::gels(Tensor self, Tensor A) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("gels");
      
          auto result_ = at::gels(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::geometric_(Tensor(a!) self, float p, *, Generator generator=None) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("geometric_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = (self).geometric_(
              (std::move(peek(stack, 1, 3))).toDouble(),
              nullptr
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::ger(Tensor self, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ger_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::ger_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::glu(Tensor self, int dim=-1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("glu");
      
          auto result_ = at::glu(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::glu_backward(Tensor grad_output, Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("glu_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::glu_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toInt()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::grid_sampler(Tensor input, Tensor grid, int interpolation_mode, int padding_mode) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("grid_sampler");
      
          auto result_ = at::grid_sampler(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toInt(),
              (std::move(peek(stack, 3, 4))).toInt()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::grid_sampler_3d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("grid_sampler_3d");
      
          auto result_ = at::grid_sampler_3d(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toInt(),
              (std::move(peek(stack, 3, 4))).toInt()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("group_norm");
      
          auto result_ = at::group_norm(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toInt(),
              (std::move(peek(stack, 2, 6))).toTensor(),
              (std::move(peek(stack, 3, 6))).toTensor(),
              (std::move(peek(stack, 4, 6))).toDouble(),
              (std::move(peek(stack, 5, 6))).toBool()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::gru(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("gru");
      
          auto result_ = at::gru(
              (std::move(peek(stack, 0, 9))).toTensor(),
              (std::move(peek(stack, 1, 9))).toTensor(),
              (std::move(peek(stack, 2, 9))).toTensor(),
              (std::move(peek(stack, 3, 9))).toTensorList()->elements(),
              (std::move(peek(stack, 4, 9))).toBool(),
              (std::move(peek(stack, 5, 9))).toInt(),
              (std::move(peek(stack, 6, 9))).toDouble(),
              (std::move(peek(stack, 7, 9))).toBool(),
              (std::move(peek(stack, 8, 9))).toBool()
          );
          drop(stack, 9);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::gru(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("gru");
      
          auto result_ = at::gru(
              (std::move(peek(stack, 0, 9))).toTensor(),
              (std::move(peek(stack, 1, 9))).toTensor(),
              (std::move(peek(stack, 2, 9))).toTensorList()->elements(),
              (std::move(peek(stack, 3, 9))).toBool(),
              (std::move(peek(stack, 4, 9))).toInt(),
              (std::move(peek(stack, 5, 9))).toDouble(),
              (std::move(peek(stack, 6, 9))).toBool(),
              (std::move(peek(stack, 7, 9))).toBool(),
              (std::move(peek(stack, 8, 9))).toBool()
          );
          drop(stack, 9);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::gt(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("gt_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::gt_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::gt(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("gt_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::gt_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::hann_window(int window_length, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("hann_window");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 1, 4))).toScalarType())
                  .layout((std::move(peek(stack, 2, 4))).toLayout())
                  .device((std::move(peek(stack, 3, 4))).toDevice());
          auto result_ = torch::hann_window((std::move(peek(stack, 0, 4))).toInt(),
          options);
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::hann_window(int window_length, bool periodic, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("hann_window");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 2, 5))).toScalarType())
                  .layout((std::move(peek(stack, 3, 5))).toLayout())
                  .device((std::move(peek(stack, 4, 5))).toDevice());
          auto result_ = torch::hann_window((std::move(peek(stack, 0, 5))).toInt(),
          (std::move(peek(stack, 1, 5))).toBool(),
          options);
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::hardshrink_backward(Tensor grad_out, Tensor self, Scalar lambd) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("hardshrink_backward");
      
          auto result_ = at::hardshrink_backward(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("hardtanh_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = at::hardtanh_(
              self,
              (std::move(peek(stack, 1, 3))).toScalar(),
              (std::move(peek(stack, 2, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("hardtanh_backward");
      
          auto result_ = at::hardtanh_backward(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toScalar(),
              (std::move(peek(stack, 3, 4))).toScalar()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::hinge_embedding_loss(Tensor self, Tensor target, float margin=1.0, int reduction=Mean) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("hinge_embedding_loss");
      
          auto result_ = at::hinge_embedding_loss(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toDouble(),
              (std::move(peek(stack, 3, 4))).toInt()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::hspmm(Tensor mat1, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("hspmm_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::hspmm_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::index_put(Tensor self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("index_put");
      
          auto result_ = at::index_put(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensorList()->elements(),
              (std::move(peek(stack, 2, 4))).toTensor(),
              (std::move(peek(stack, 3, 4))).toBool()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::is_coalesced(Tensor self) -> bool",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("is_coalesced");
      
          auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).is_coalesced(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::is_complex(Tensor self) -> bool",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("is_complex");
      
          auto result_ = at::is_complex(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::is_floating_point(Tensor self) -> bool",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("is_floating_point");
      
          auto result_ = at::is_floating_point(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::is_signed(Tensor self) -> bool",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("is_signed");
      
          auto result_ = at::is_signed(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::isclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("isclose");
      
          auto result_ = at::isclose(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toDouble(),
              (std::move(peek(stack, 3, 5))).toDouble(),
              (std::move(peek(stack, 4, 5))).toBool()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::kthvalue(Tensor self, int k, int dim=-1, bool keepdim=False) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("kthvalue");
      
          auto result_ = at::kthvalue(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toInt(),
              (std::move(peek(stack, 2, 4))).toInt(),
              (std::move(peek(stack, 3, 4))).toBool()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("l1_loss_backward");
      
          auto result_ = at::l1_loss_backward(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toTensor(),
              (std::move(peek(stack, 3, 4))).toInt()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::le_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("le_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).le_(
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::le_(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("le_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).le_(
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("leaky_relu");
      
          auto result_ = at::leaky_relu(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::leaky_relu_backward(Tensor grad_output, Tensor self, Scalar negative_slope, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("leaky_relu_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::leaky_relu_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toScalar()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::lerp(Tensor self, Tensor end, Scalar weight, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("lerp_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::lerp_out(
              result,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toScalar()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::lgamma_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("lgamma_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = (self).lgamma_(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::linspace(Scalar start, Scalar end, int steps=100, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("linspace");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 3, 6))).toScalarType())
                  .layout((std::move(peek(stack, 4, 6))).toLayout())
                  .device((std::move(peek(stack, 5, 6))).toDevice());
          auto result_ = torch::linspace((std::move(peek(stack, 0, 6))).toScalar(),
          (std::move(peek(stack, 1, 6))).toScalar(),
          (std::move(peek(stack, 2, 6))).toInt(),
          options);
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::log(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("log");
      
          auto result_ = at::log(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::log10_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("log10_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::log10_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::log1p(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("log1p_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::log1p_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::log2(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("log2_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::log2_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::log_sigmoid(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("log_sigmoid");
      
          auto result_ = at::log_sigmoid(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::log_sigmoid_backward(Tensor grad_output, Tensor self, Tensor buffer, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("log_sigmoid_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::log_sigmoid_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toTensor()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::log_sigmoid_forward(Tensor self) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("log_sigmoid_forward");
      
          auto result_ = at::log_sigmoid_forward(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::logspace(Scalar start, Scalar end, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("logspace_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::logspace_out(
              result,
              (std::move(peek(stack, 0, 3))).toScalar(),
              (std::move(peek(stack, 1, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::logspace(Scalar start, Scalar end, int steps, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("logspace_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::logspace_out(
              result,
              (std::move(peek(stack, 0, 4))).toScalar(),
              (std::move(peek(stack, 1, 4))).toScalar(),
              (std::move(peek(stack, 2, 4))).toInt()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::logsumexp(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("logsumexp_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::logsumexp_out(
              result,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toInt(),
              (std::move(peek(stack, 2, 4))).toBool()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::lstm(Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("lstm");
      
          auto result_ = at::lstm(
              (std::move(peek(stack, 0, 9))).toTensor(),
              (std::move(peek(stack, 1, 9))).toTensor(),
              (std::move(peek(stack, 2, 9))).toTensorList()->elements(),
              (std::move(peek(stack, 3, 9))).toTensorList()->elements(),
              (std::move(peek(stack, 4, 9))).toBool(),
              (std::move(peek(stack, 5, 9))).toInt(),
              (std::move(peek(stack, 6, 9))).toDouble(),
              (std::move(peek(stack, 7, 9))).toBool(),
              (std::move(peek(stack, 8, 9))).toBool()
          );
          drop(stack, 9);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::lstm(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("lstm");
      
          auto result_ = at::lstm(
              (std::move(peek(stack, 0, 9))).toTensor(),
              (std::move(peek(stack, 1, 9))).toTensorList()->elements(),
              (std::move(peek(stack, 2, 9))).toTensorList()->elements(),
              (std::move(peek(stack, 3, 9))).toBool(),
              (std::move(peek(stack, 4, 9))).toInt(),
              (std::move(peek(stack, 5, 9))).toDouble(),
              (std::move(peek(stack, 6, 9))).toBool(),
              (std::move(peek(stack, 7, 9))).toBool(),
              (std::move(peek(stack, 8, 9))).toBool()
          );
          drop(stack, 9);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::lt_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("lt_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).lt_(
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::lt_(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("lt_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).lt_(
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("margin_ranking_loss");
      
          auto result_ = at::margin_ranking_loss(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toTensor(),
              (std::move(peek(stack, 3, 5))).toDouble(),
              (std::move(peek(stack, 4, 5))).toInt()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("max_pool1d");
      
          auto result_ = at::max_pool1d(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toIntList()->elements(),
              (std::move(peek(stack, 2, 6))).toIntList()->elements(),
              (std::move(peek(stack, 3, 6))).toIntList()->elements(),
              (std::move(peek(stack, 4, 6))).toIntList()->elements(),
              (std::move(peek(stack, 5, 6))).toBool()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::max_pool3d_with_indices(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("max_pool3d_with_indices");
      
          auto result_ = at::max_pool3d_with_indices(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toIntList()->elements(),
              (std::move(peek(stack, 2, 6))).toIntList()->elements(),
              (std::move(peek(stack, 3, 6))).toIntList()->elements(),
              (std::move(peek(stack, 4, 6))).toIntList()->elements(),
              (std::move(peek(stack, 5, 6))).toBool()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::max_pool3d_with_indices_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("max_pool3d_with_indices_backward_out");
          auto grad_input = (std::move(peek(stack, 8, 9))).toTensor();
          auto result_ = at::max_pool3d_with_indices_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 9))).toTensor(),
              (std::move(peek(stack, 1, 9))).toTensor(),
              (std::move(peek(stack, 2, 9))).toIntList()->elements(),
              (std::move(peek(stack, 3, 9))).toIntList()->elements(),
              (std::move(peek(stack, 4, 9))).toIntList()->elements(),
              (std::move(peek(stack, 5, 9))).toIntList()->elements(),
              (std::move(peek(stack, 6, 9))).toBool(),
              (std::move(peek(stack, 7, 9))).toTensor()
          );
          drop(stack, 9);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::max_unpool2d(Tensor self, Tensor indices, int[2] output_size) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("max_unpool2d");
      
          auto result_ = at::max_unpool2d(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toIntList()->elements()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::max_unpool2d_backward(Tensor grad_output, Tensor self, Tensor indices, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("max_unpool2d_backward_out");
          auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::max_unpool2d_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toTensor(),
              (std::move(peek(stack, 3, 5))).toIntList()->elements()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::max_unpool3d_backward(Tensor grad_output, Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("max_unpool3d_backward");
      
          auto result_ = at::max_unpool3d_backward(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toTensor(),
              (std::move(peek(stack, 3, 6))).toIntList()->elements(),
              (std::move(peek(stack, 4, 6))).toIntList()->elements(),
              (std::move(peek(stack, 5, 6))).toIntList()->elements()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::mean(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mean_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::mean_out(
              result,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toIntList()->elements(),
              (std::move(peek(stack, 2, 4))).toBool()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::mean(Tensor self, int[1] dim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mean_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::mean_out(
              result,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toIntList()->elements(),
              (std::move(peek(stack, 2, 4))).toScalarType()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::mean(Tensor self, int[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mean_out");
          auto result = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::mean_out(
              result,
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toIntList()->elements(),
              (std::move(peek(stack, 2, 5))).toBool(),
              (std::move(peek(stack, 3, 5))).toScalarType()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::median(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("median");
      
          auto result_ = at::median(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::median(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("median");
      
          auto result_ = at::median(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::min(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("min_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::min_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::min_values(Tensor self, int dim, bool keepdim=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("min_values");
      
          auto result_ = at::min_values(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::miopen_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("miopen_batch_norm_backward");
      
          auto result_ = at::miopen_batch_norm_backward(
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toTensor(),
              (std::move(peek(stack, 3, 8))).toTensor(),
              (std::move(peek(stack, 4, 8))).toTensor(),
              (std::move(peek(stack, 5, 8))).toTensor(),
              (std::move(peek(stack, 6, 8))).toTensor(),
              (std::move(peek(stack, 7, 8))).toDouble()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::miopen_convolution_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("miopen_convolution_backward_weight");
      
          auto result_ = at::miopen_convolution_backward_weight(
              (std::move(peek(stack, 0, 9))).toIntList()->elements(),
              (std::move(peek(stack, 1, 9))).toTensor(),
              (std::move(peek(stack, 2, 9))).toTensor(),
              (std::move(peek(stack, 3, 9))).toIntList()->elements(),
              (std::move(peek(stack, 4, 9))).toIntList()->elements(),
              (std::move(peek(stack, 5, 9))).toIntList()->elements(),
              (std::move(peek(stack, 6, 9))).toInt(),
              (std::move(peek(stack, 7, 9))).toBool(),
              (std::move(peek(stack, 8, 9))).toBool()
          );
          drop(stack, 9);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::miopen_convolution_transpose_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("miopen_convolution_transpose_backward_weight");
      
          auto result_ = at::miopen_convolution_transpose_backward_weight(
              (std::move(peek(stack, 0, 9))).toIntList()->elements(),
              (std::move(peek(stack, 1, 9))).toTensor(),
              (std::move(peek(stack, 2, 9))).toTensor(),
              (std::move(peek(stack, 3, 9))).toIntList()->elements(),
              (std::move(peek(stack, 4, 9))).toIntList()->elements(),
              (std::move(peek(stack, 5, 9))).toIntList()->elements(),
              (std::move(peek(stack, 6, 9))).toInt(),
              (std::move(peek(stack, 7, 9))).toBool(),
              (std::move(peek(stack, 8, 9))).toBool()
          );
          drop(stack, 9);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::mkldnn_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mkldnn_convolution_backward");
      
          auto result_ = at::mkldnn_convolution_backward(
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toTensor(),
              (std::move(peek(stack, 3, 8))).toIntList()->elements(),
              (std::move(peek(stack, 4, 8))).toIntList()->elements(),
              (std::move(peek(stack, 5, 8))).toIntList()->elements(),
              (std::move(peek(stack, 6, 8))).toInt(),
              as_bool_array<3>((std::move(peek(stack, 7, 8))).toIntList()->elements())
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::mkldnn_convolution_backward_weights(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool bias_defined) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mkldnn_convolution_backward_weights");
      
          auto result_ = at::mkldnn_convolution_backward_weights(
              (std::move(peek(stack, 0, 8))).toIntList()->elements(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toTensor(),
              (std::move(peek(stack, 3, 8))).toIntList()->elements(),
              (std::move(peek(stack, 4, 8))).toIntList()->elements(),
              (std::move(peek(stack, 5, 8))).toIntList()->elements(),
              (std::move(peek(stack, 6, 8))).toInt(),
              (std::move(peek(stack, 7, 8))).toBool()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::mode(Tensor self, int dim=-1, bool keepdim=False) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mode");
      
          auto result_ = at::mode(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mse_loss_backward");
      
          auto result_ = at::mse_loss_backward(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toTensor(),
              (std::move(peek(stack, 3, 4))).toInt()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::mul(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mul");
      
          auto result_ = at::mul(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::mul(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mul");
      
          auto result_ = at::mul(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::multi_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor weight, int reduction) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("multi_margin_loss_backward");
      
          auto result_ = at::multi_margin_loss_backward(
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              (std::move(peek(stack, 2, 7))).toTensor(),
              (std::move(peek(stack, 3, 7))).toScalar(),
              (std::move(peek(stack, 4, 7))).toScalar(),
              (std::move(peek(stack, 5, 7))).toTensor(),
              (std::move(peek(stack, 6, 7))).toInt()
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::multilabel_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("multilabel_margin_loss");
      
          auto result_ = at::multilabel_margin_loss(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toInt()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::multilabel_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("multilabel_margin_loss_backward_out");
          auto grad_input = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::multilabel_margin_loss_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toTensor(),
              (std::move(peek(stack, 3, 6))).toInt(),
              (std::move(peek(stack, 4, 6))).toTensor()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::multilabel_margin_loss_forward(Tensor self, Tensor target, int reduction) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("multilabel_margin_loss_forward");
      
          auto result_ = at::multilabel_margin_loss_forward(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toInt()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator generator=None) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("multinomial");
      
          auto result_ = at::multinomial(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toInt(),
              (std::move(peek(stack, 2, 4))).toBool(),
              nullptr
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::mvlgamma(Tensor self, int p) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mvlgamma");
      
          auto result_ = at::mvlgamma(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("native_batch_norm");
      
          auto result_ = at::native_batch_norm(
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toTensor(),
              (std::move(peek(stack, 3, 8))).toTensor(),
              (std::move(peek(stack, 4, 8))).toTensor(),
              (std::move(peek(stack, 5, 8))).toBool(),
              (std::move(peek(stack, 6, 8))).toDouble(),
              (std::move(peek(stack, 7, 8))).toDouble()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::native_norm(Tensor self, Scalar p=2) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("native_norm");
      
          auto result_ = at::native_norm(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::native_pow(Tensor self, Scalar exponent) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("native_pow");
      
          auto result_ = at::native_pow(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::native_zero_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("native_zero_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::native_zero_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::ne(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ne");
      
          auto result_ = at::ne(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::ne(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ne");
      
          auto result_ = at::ne(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::neg_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("neg_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = (self).neg_(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::nll_loss2d(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("nll_loss2d_out");
          auto output = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::nll_loss2d_out(
              output,
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toTensor(),
              (std::move(peek(stack, 3, 6))).toInt(),
              (std::move(peek(stack, 4, 6))).toInt()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::nll_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("nll_loss_out");
          auto output = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::nll_loss_out(
              output,
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toTensor(),
              (std::move(peek(stack, 3, 6))).toInt(),
              (std::move(peek(stack, 4, 6))).toInt()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::norm_except_dim(Tensor v, int pow=2, int dim=0) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("norm_except_dim");
      
          auto result_ = at::norm_except_dim(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt(),
              (std::move(peek(stack, 2, 3))).toInt()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::norm(Tensor self, Scalar p, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("norm_out");
          auto result = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::norm_out(
              result,
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toScalar(),
              (std::move(peek(stack, 2, 5))).toInt(),
              (std::move(peek(stack, 3, 5))).toBool()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::normal(Tensor mean, Tensor std, *, Generator generator=None) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("normal");
      
          auto result_ = at::normal(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              nullptr
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::normal(float mean, Tensor std, *, Generator generator=None) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("normal");
      
          auto result_ = at::normal(
              (std::move(peek(stack, 0, 3))).toDouble(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              nullptr
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::normal(Tensor mean, float std=1, *, Generator generator=None) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("normal");
      
          auto result_ = at::normal(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toDouble(),
              nullptr
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::nuclear_norm(Tensor self, bool keepdim=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("nuclear_norm");
      
          auto result_ = at::nuclear_norm(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::orgqr(Tensor self, Tensor input2, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("orgqr_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::orgqr_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ormqr_out");
          auto result = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::ormqr_out(
              result,
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toTensor(),
              (std::move(peek(stack, 3, 6))).toBool(),
              (std::move(peek(stack, 4, 6))).toBool()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::pin_memory(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("pin_memory");
      
          auto result_ = at::pin_memory(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::pixel_shuffle(Tensor self, int upscale_factor) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("pixel_shuffle");
      
          auto result_ = at::pixel_shuffle(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::polygamma(int n, Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("polygamma");
      
          auto result_ = at::polygamma(
              (std::move(peek(stack, 0, 2))).toInt(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::potri(Tensor self, bool upper=True, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("potri_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::potri_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::potrs(Tensor self, Tensor input2, bool upper=True) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("potrs");
      
          auto result_ = at::potrs(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::pow(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("pow_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::pow_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::pow(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("pow_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::pow_out(
              result,
              (std::move(peek(stack, 0, 3))).toScalar(),
              (std::move(peek(stack, 1, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::pow(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("pow_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::pow_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::pstrf(Tensor self, bool upper=True, Scalar tol=-1) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("pstrf");
      
          auto result_ = at::pstrf(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toBool(),
              (std::move(peek(stack, 2, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::rand(int[] size, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("rand");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 1, 4))).toScalarType())
                  .layout((std::move(peek(stack, 2, 4))).toLayout())
                  .device((std::move(peek(stack, 3, 4))).toDevice());
          auto result_ = torch::rand((std::move(peek(stack, 0, 4))).toIntList()->elements(),
          options);
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::rand_like(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("rand_like");
      
          auto result_ = at::rand_like(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::rand_like(Tensor self, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("rand_like");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 1, 4))).toScalarType())
                  .layout((std::move(peek(stack, 2, 4))).toLayout())
                  .device((std::move(peek(stack, 3, 4))).toDevice());
          auto result_ = torch::rand_like((std::move(peek(stack, 0, 4))).toTensor(),
          options);
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::randn(int[] size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("randn_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::randn_out(
              result,
              (std::move(peek(stack, 0, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::random_(Tensor(a!) self, *, Generator generator=None) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("random_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).random_(
              nullptr
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::random_(Tensor(a!) self, int to, *, Generator generator=None) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("random_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = (self).random_(
              (std::move(peek(stack, 1, 3))).toInt(),
              nullptr
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::random_(Tensor(a!) self, int from, int to, *, Generator generator=None) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("random_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = (self).random_(
              (std::move(peek(stack, 1, 4))).toInt(),
              (std::move(peek(stack, 2, 4))).toInt(),
              nullptr
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::range(Scalar start, Scalar end, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("range_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::range_out(
              result,
              (std::move(peek(stack, 0, 3))).toScalar(),
              (std::move(peek(stack, 1, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::range(Scalar start, Scalar end, Scalar step, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("range_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::range_out(
              result,
              (std::move(peek(stack, 0, 4))).toScalar(),
              (std::move(peek(stack, 1, 4))).toScalar(),
              (std::move(peek(stack, 2, 4))).toScalar()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::reciprocal(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("reciprocal");
      
          auto result_ = at::reciprocal(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::reflection_pad1d(Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("reflection_pad1d_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::reflection_pad1d_out(
              output,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toIntList()->elements()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::reflection_pad2d(Tensor self, int[4] padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("reflection_pad2d");
      
          auto result_ = at::reflection_pad2d(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::reflection_pad2d_backward(Tensor grad_output, Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("reflection_pad2d_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::reflection_pad2d_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toIntList()->elements()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::relu_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("relu_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::relu_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::remainder(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("remainder_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::remainder_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::remainder(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("remainder_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::remainder_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::renorm_(Tensor(a!) self, Scalar p, int dim, Scalar maxnorm) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("renorm_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = (self).renorm_(
              (std::move(peek(stack, 1, 4))).toScalar(),
              (std::move(peek(stack, 2, 4))).toInt(),
              (std::move(peek(stack, 3, 4))).toScalar()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::replication_pad1d(Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("replication_pad1d_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::replication_pad1d_out(
              output,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toIntList()->elements()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::replication_pad2d(Tensor self, int[4] padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("replication_pad2d");
      
          auto result_ = at::replication_pad2d(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::replication_pad2d_backward(Tensor grad_output, Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("replication_pad2d_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::replication_pad2d_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toIntList()->elements()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::replication_pad3d_backward(Tensor grad_output, Tensor self, int[6] padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("replication_pad3d_backward");
      
          auto result_ = at::replication_pad3d_backward(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toIntList()->elements()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::reshape_as(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("reshape_as");
      
          auto result_ = ((std::move(peek(stack, 0, 2))).toTensor()).reshape_as(
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::resize_as_(Tensor(a!) self, Tensor the_template) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("resize_as_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::resize_as_(
              self,
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::rnn_tanh(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("rnn_tanh");
      
          auto result_ = at::rnn_tanh(
              (std::move(peek(stack, 0, 9))).toTensor(),
              (std::move(peek(stack, 1, 9))).toTensor(),
              (std::move(peek(stack, 2, 9))).toTensor(),
              (std::move(peek(stack, 3, 9))).toTensorList()->elements(),
              (std::move(peek(stack, 4, 9))).toBool(),
              (std::move(peek(stack, 5, 9))).toInt(),
              (std::move(peek(stack, 6, 9))).toDouble(),
              (std::move(peek(stack, 7, 9))).toBool(),
              (std::move(peek(stack, 8, 9))).toBool()
          );
          drop(stack, 9);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::rnn_tanh(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("rnn_tanh");
      
          auto result_ = at::rnn_tanh(
              (std::move(peek(stack, 0, 9))).toTensor(),
              (std::move(peek(stack, 1, 9))).toTensor(),
              (std::move(peek(stack, 2, 9))).toTensorList()->elements(),
              (std::move(peek(stack, 3, 9))).toBool(),
              (std::move(peek(stack, 4, 9))).toInt(),
              (std::move(peek(stack, 5, 9))).toDouble(),
              (std::move(peek(stack, 6, 9))).toBool(),
              (std::move(peek(stack, 7, 9))).toBool(),
              (std::move(peek(stack, 8, 9))).toBool()
          );
          drop(stack, 9);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::round(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("round_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::round_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::rrelu_(Tensor(a!) self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator generator=None) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("rrelu_");
          auto self = (std::move(peek(stack, 0, 5))).toTensor();
          auto result_ = at::rrelu_(
              self,
              (std::move(peek(stack, 1, 5))).toScalar(),
              (std::move(peek(stack, 2, 5))).toScalar(),
              (std::move(peek(stack, 3, 5))).toBool(),
              nullptr
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::rrelu_with_noise_(Tensor(a!) self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator generator=None) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("rrelu_with_noise_");
          auto self = (std::move(peek(stack, 0, 6))).toTensor();
          auto result_ = at::rrelu_with_noise_(
              self,
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toScalar(),
              (std::move(peek(stack, 3, 6))).toScalar(),
              (std::move(peek(stack, 4, 6))).toBool(),
              nullptr
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::rrelu_with_noise_backward(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("rrelu_with_noise_backward");
      
          auto result_ = at::rrelu_with_noise_backward(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toTensor(),
              (std::move(peek(stack, 3, 6))).toScalar(),
              (std::move(peek(stack, 4, 6))).toScalar(),
              (std::move(peek(stack, 5, 6))).toBool()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::rsqrt_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("rsqrt_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::rsqrt_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::s_copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("s_copy_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = at::s_copy_(
              self,
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::s_native_addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("s_native_addmm_out");
          auto result = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::s_native_addmm_out(
              result,
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toTensor(),
              (std::move(peek(stack, 3, 6))).toScalar(),
              (std::move(peek(stack, 4, 6))).toScalar()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::scatter_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("scatter_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = (self).scatter_(
              (std::move(peek(stack, 1, 4))).toInt(),
              (std::move(peek(stack, 2, 4))).toTensor(),
              (std::move(peek(stack, 3, 4))).toTensor()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::scatter_(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("scatter_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = (self).scatter_(
              (std::move(peek(stack, 1, 4))).toInt(),
              (std::move(peek(stack, 2, 4))).toTensor(),
              (std::move(peek(stack, 3, 4))).toScalar()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::select(Tensor(a) self, int dim, int index) -> Tensor(a)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("select");
      
          auto result_ = at::select(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt(),
              (std::move(peek(stack, 2, 3))).toInt()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::set_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("set_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = (self).set_(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::set_(Tensor(a!) self, Tensor source) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("set_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).set_(
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sigmoid(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sigmoid");
      
          auto result_ = at::sigmoid(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sigmoid_backward(Tensor grad_output, Tensor output, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sigmoid_backward_out");
          auto grad_input = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::sigmoid_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sign(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sign");
      
          auto result_ = at::sign(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sin(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sin_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::sin_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sinh_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sinh_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::sinh_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::slogdet(Tensor self) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("slogdet");
      
          auto result_ = at::slogdet(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::smooth_l1_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("smooth_l1_loss");
      
          auto result_ = at::smooth_l1_loss(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toInt()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::smooth_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("smooth_l1_loss_backward_out");
          auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::smooth_l1_loss_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toTensor(),
              (std::move(peek(stack, 3, 5))).toInt()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::soft_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("soft_margin_loss_backward");
      
          auto result_ = at::soft_margin_loss_backward(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toTensor(),
              (std::move(peek(stack, 3, 4))).toInt()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::softplus_backward(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, Tensor output) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("softplus_backward");
      
          auto result_ = at::softplus_backward(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toScalar(),
              (std::move(peek(stack, 3, 5))).toScalar(),
              (std::move(peek(stack, 4, 5))).toTensor()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::softshrink(Tensor self, Scalar lambd=0.5) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("softshrink");
      
          auto result_ = at::softshrink(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::softshrink_backward(Tensor grad_output, Tensor self, Scalar lambd, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("softshrink_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::softshrink_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toScalar()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sparse_dim(Tensor self) -> int",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sparse_dim");
      
          auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).sparse_dim(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sparse_resize_and_clear_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sparse_resize_and_clear_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = (self).sparse_resize_and_clear_(
              (std::move(peek(stack, 1, 4))).toIntList()->elements(),
              (std::move(peek(stack, 2, 4))).toInt(),
              (std::move(peek(stack, 3, 4))).toInt()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::split(Tensor(a) self, int split_size, int dim=0) -> Tensor(a)[]",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("split");
      
          auto result_ = at::split(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt(),
              (std::move(peek(stack, 2, 3))).toInt()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sqrt_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sqrt_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::sqrt_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::squeeze_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("squeeze_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = (self).squeeze_(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::squeeze_(Tensor(a!) self, int dim) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("squeeze_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).squeeze_(
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::stack(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("stack_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::stack_out(
              result,
              (std::move(peek(stack, 0, 3))).toTensorList()->elements(),
              (std::move(peek(stack, 1, 3))).toInt()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::std(Tensor self, bool unbiased=True) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("std");
      
          auto result_ = at::std(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::std(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("std");
      
          auto result_ = at::std(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toIntList()->elements(),
              (std::move(peek(stack, 2, 4))).toBool(),
              (std::move(peek(stack, 3, 4))).toBool()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::strides(Tensor self) -> int",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("strides");
      
          auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).strides(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sub(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sub_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::sub_out(
              result,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toScalar()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::t_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("t_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = (self).t_(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::take(Tensor self, Tensor index) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("take");
      
          auto result_ = at::take(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::tan_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("tan_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::tan_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::tanh(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("tanh");
      
          auto result_ = at::tanh(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::tanh_backward(Tensor grad_output, Tensor output, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("tanh_backward_out");
          auto grad_input = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::tanh_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::tensordot(Tensor self, Tensor other, int[] dims_self, int[] dims_other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("tensordot");
      
          auto result_ = at::tensordot(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toIntList()->elements(),
              (std::move(peek(stack, 3, 4))).toIntList()->elements()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::thnn_col2im(Tensor self, int[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_col2im");
      
          auto result_ = at::thnn_col2im(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toIntList()->elements(),
              (std::move(peek(stack, 2, 6))).toIntList()->elements(),
              (std::move(peek(stack, 3, 6))).toIntList()->elements(),
              (std::move(peek(stack, 4, 6))).toIntList()->elements(),
              (std::move(peek(stack, 5, 6))).toIntList()->elements()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::thnn_conv2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv2d");
      
          auto result_ = at::thnn_conv2d(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toIntList()->elements(),
              (std::move(peek(stack, 3, 6))).toTensor(),
              (std::move(peek(stack, 4, 6))).toIntList()->elements(),
              (std::move(peek(stack, 5, 6))).toIntList()->elements()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::thnn_conv2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv2d_forward");
      
          auto result_ = at::thnn_conv2d_forward(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toIntList()->elements(),
              (std::move(peek(stack, 3, 6))).toTensor(),
              (std::move(peek(stack, 4, 6))).toIntList()->elements(),
              (std::move(peek(stack, 5, 6))).toIntList()->elements()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::thnn_conv3d_backward(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, Tensor finput, Tensor fgrad_input, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv3d_backward");
      
          auto result_ = at::thnn_conv3d_backward(
              (std::move(peek(stack, 0, 9))).toTensor(),
              (std::move(peek(stack, 1, 9))).toTensor(),
              (std::move(peek(stack, 2, 9))).toTensor(),
              (std::move(peek(stack, 3, 9))).toIntList()->elements(),
              (std::move(peek(stack, 4, 9))).toIntList()->elements(),
              (std::move(peek(stack, 5, 9))).toIntList()->elements(),
              (std::move(peek(stack, 6, 9))).toTensor(),
              (std::move(peek(stack, 7, 9))).toTensor(),
              as_bool_array<3>((std::move(peek(stack, 8, 9))).toIntList()->elements())
          );
          drop(stack, 9);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::thnn_conv_depthwise2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv_depthwise2d");
      
          auto result_ = at::thnn_conv_depthwise2d(
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              (std::move(peek(stack, 2, 7))).toIntList()->elements(),
              (std::move(peek(stack, 3, 7))).toTensor(),
              (std::move(peek(stack, 4, 7))).toIntList()->elements(),
              (std::move(peek(stack, 5, 7))).toIntList()->elements(),
              (std::move(peek(stack, 6, 7))).toIntList()->elements()
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::thnn_conv_depthwise2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv_depthwise2d_forward");
      
          auto result_ = at::thnn_conv_depthwise2d_forward(
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              (std::move(peek(stack, 2, 7))).toIntList()->elements(),
              (std::move(peek(stack, 3, 7))).toTensor(),
              (std::move(peek(stack, 4, 7))).toIntList()->elements(),
              (std::move(peek(stack, 5, 7))).toIntList()->elements(),
              (std::move(peek(stack, 6, 7))).toIntList()->elements()
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::thnn_conv_dilated2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv_dilated2d");
      
          auto result_ = at::thnn_conv_dilated2d(
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              (std::move(peek(stack, 2, 7))).toIntList()->elements(),
              (std::move(peek(stack, 3, 7))).toTensor(),
              (std::move(peek(stack, 4, 7))).toIntList()->elements(),
              (std::move(peek(stack, 5, 7))).toIntList()->elements(),
              (std::move(peek(stack, 6, 7))).toIntList()->elements()
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::thnn_conv_dilated2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv_dilated2d_forward");
      
          auto result_ = at::thnn_conv_dilated2d_forward(
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              (std::move(peek(stack, 2, 7))).toIntList()->elements(),
              (std::move(peek(stack, 3, 7))).toTensor(),
              (std::move(peek(stack, 4, 7))).toIntList()->elements(),
              (std::move(peek(stack, 5, 7))).toIntList()->elements(),
              (std::move(peek(stack, 6, 7))).toIntList()->elements()
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::thnn_conv_dilated3d_backward(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, Tensor columns, Tensor ones, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv_dilated3d_backward");
      
          auto result_ = at::thnn_conv_dilated3d_backward(
              (std::move(peek(stack, 0, 10))).toTensor(),
              (std::move(peek(stack, 1, 10))).toTensor(),
              (std::move(peek(stack, 2, 10))).toTensor(),
              (std::move(peek(stack, 3, 10))).toIntList()->elements(),
              (std::move(peek(stack, 4, 10))).toIntList()->elements(),
              (std::move(peek(stack, 5, 10))).toIntList()->elements(),
              (std::move(peek(stack, 6, 10))).toIntList()->elements(),
              (std::move(peek(stack, 7, 10))).toTensor(),
              (std::move(peek(stack, 8, 10))).toTensor(),
              as_bool_array<3>((std::move(peek(stack, 9, 10))).toIntList()->elements())
          );
          drop(stack, 10);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::thnn_conv_transpose2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int[2] dilation=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv_transpose2d");
      
          auto result_ = at::thnn_conv_transpose2d(
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toIntList()->elements(),
              (std::move(peek(stack, 3, 8))).toTensor(),
              (std::move(peek(stack, 4, 8))).toIntList()->elements(),
              (std::move(peek(stack, 5, 8))).toIntList()->elements(),
              (std::move(peek(stack, 6, 8))).toIntList()->elements(),
              (std::move(peek(stack, 7, 8))).toIntList()->elements()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::thnn_conv_transpose2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, int[2] output_padding, int[2] dilation) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv_transpose2d_forward");
      
          auto result_ = at::thnn_conv_transpose2d_forward(
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toIntList()->elements(),
              (std::move(peek(stack, 3, 8))).toTensor(),
              (std::move(peek(stack, 4, 8))).toIntList()->elements(),
              (std::move(peek(stack, 5, 8))).toIntList()->elements(),
              (std::move(peek(stack, 6, 8))).toIntList()->elements(),
              (std::move(peek(stack, 7, 8))).toIntList()->elements()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::thnn_conv_transpose3d_backward(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] output_padding, int[3] dilation, Tensor finput, Tensor fgrad_input, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv_transpose3d_backward");
      
          auto result_ = at::thnn_conv_transpose3d_backward(
              (std::move(peek(stack, 0, 11))).toTensor(),
              (std::move(peek(stack, 1, 11))).toTensor(),
              (std::move(peek(stack, 2, 11))).toTensor(),
              (std::move(peek(stack, 3, 11))).toIntList()->elements(),
              (std::move(peek(stack, 4, 11))).toIntList()->elements(),
              (std::move(peek(stack, 5, 11))).toIntList()->elements(),
              (std::move(peek(stack, 6, 11))).toIntList()->elements(),
              (std::move(peek(stack, 7, 11))).toIntList()->elements(),
              (std::move(peek(stack, 8, 11))).toTensor(),
              (std::move(peek(stack, 9, 11))).toTensor(),
              as_bool_array<3>((std::move(peek(stack, 10, 11))).toIntList()->elements())
          );
          drop(stack, 11);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::thnn_im2col(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_im2col");
      
          auto result_ = at::thnn_im2col(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toIntList()->elements(),
              (std::move(peek(stack, 2, 5))).toIntList()->elements(),
              (std::move(peek(stack, 3, 5))).toIntList()->elements(),
              (std::move(peek(stack, 4, 5))).toIntList()->elements()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("threshold");
      
          auto result_ = at::threshold(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toScalar(),
              (std::move(peek(stack, 2, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::to_sparse(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("to_sparse");
      
          auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).to_sparse(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::to_sparse(Tensor self, int sparse_dim) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("to_sparse");
      
          auto result_ = ((std::move(peek(stack, 0, 2))).toTensor()).to_sparse(
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::transpose(Tensor(a) self, int dim0, int dim1) -> Tensor(a)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("transpose");
      
          auto result_ = at::transpose(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt(),
              (std::move(peek(stack, 2, 3))).toInt()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::tril_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("tril_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).tril_(
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::tril_indices(int row, int col, int offset=0, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("tril_indices");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 3, 6))).toScalarType())
                  .layout((std::move(peek(stack, 4, 6))).toLayout())
                  .device((std::move(peek(stack, 5, 6))).toDevice());
          auto result_ = torch::tril_indices((std::move(peek(stack, 0, 6))).toInt(),
          (std::move(peek(stack, 1, 6))).toInt(),
          (std::move(peek(stack, 2, 6))).toInt(),
          options);
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::triu_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("triu_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).triu_(
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::triu_indices(int row, int col, int offset=0, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("triu_indices");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 3, 6))).toScalarType())
                  .layout((std::move(peek(stack, 4, 6))).toLayout())
                  .device((std::move(peek(stack, 5, 6))).toDevice());
          auto result_ = torch::triu_indices((std::move(peek(stack, 0, 6))).toInt(),
          (std::move(peek(stack, 1, 6))).toInt(),
          (std::move(peek(stack, 2, 6))).toInt(),
          options);
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::trunc(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("trunc");
      
          auto result_ = at::trunc(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::type_as(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("type_as");
      
          auto result_ = ((std::move(peek(stack, 0, 2))).toTensor()).type_as(
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::unbind(Tensor(a) self, int dim=0) -> Tensor(a)[]",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("unbind");
      
          auto result_ = at::unbind(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("unsqueeze");
      
          auto result_ = at::unsqueeze(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::upsample_bicubic2d(Tensor self, int[2] output_size, bool align_corners) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_bicubic2d");
      
          auto result_ = at::upsample_bicubic2d(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toIntList()->elements(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::upsample_bicubic2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_bicubic2d_backward_out");
          auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::upsample_bicubic2d_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toIntList()->elements(),
              (std::move(peek(stack, 2, 5))).toIntList()->elements(),
              (std::move(peek(stack, 3, 5))).toBool()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::upsample_bilinear2d(Tensor self, int[2] output_size, bool align_corners) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_bilinear2d");
      
          auto result_ = at::upsample_bilinear2d(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toIntList()->elements(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::upsample_bilinear2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_bilinear2d_backward_out");
          auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::upsample_bilinear2d_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toIntList()->elements(),
              (std::move(peek(stack, 2, 5))).toIntList()->elements(),
              (std::move(peek(stack, 3, 5))).toBool()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::upsample_linear1d(Tensor self, int[1] output_size, bool align_corners) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_linear1d");
      
          auto result_ = at::upsample_linear1d(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toIntList()->elements(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::upsample_linear1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_linear1d_backward_out");
          auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::upsample_linear1d_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toIntList()->elements(),
              (std::move(peek(stack, 2, 5))).toIntList()->elements(),
              (std::move(peek(stack, 3, 5))).toBool()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::upsample_nearest1d(Tensor self, int[1] output_size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_nearest1d_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::upsample_nearest1d_out(
              output,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toIntList()->elements()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::upsample_nearest2d(Tensor self, int[2] output_size) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_nearest2d");
      
          auto result_ = at::upsample_nearest2d(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::upsample_nearest2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_nearest2d_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::upsample_nearest2d_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toIntList()->elements(),
              (std::move(peek(stack, 2, 4))).toIntList()->elements()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::upsample_nearest3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_nearest3d_backward");
      
          auto result_ = at::upsample_nearest3d_backward(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toIntList()->elements(),
              (std::move(peek(stack, 2, 3))).toIntList()->elements()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::upsample_trilinear3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size, bool align_corners) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_trilinear3d_backward");
      
          auto result_ = at::upsample_trilinear3d_backward(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toIntList()->elements(),
              (std::move(peek(stack, 2, 4))).toIntList()->elements(),
              (std::move(peek(stack, 3, 4))).toBool()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
});

} // anon namespace


}} // namespace torch::jit
