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

// @generated from tools/autograd/templates/register_aten_ops_2.cpp

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
      "aten::RoiPooling2d_backward(Tensor input, Tensor rois, int pooledHeight, int pooledWidth, float spatialScale, Tensor gradOutput, Tensor argmaxes) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("RoiPooling2d_backward");
      
          auto result_ = at::RoiPooling2d_backward(
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              (std::move(peek(stack, 2, 7))).toInt(),
              (std::move(peek(stack, 3, 7))).toInt(),
              (std::move(peek(stack, 4, 7))).toDouble(),
              (std::move(peek(stack, 5, 7))).toTensor(),
              (std::move(peek(stack, 6, 7))).toTensor()
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::__ior__(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("__ior__");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).__ior__(
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::__ior__(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("__ior__");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).__ior__(
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::__ixor__(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("__ixor__");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).__ixor__(
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::__ixor__(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("__ixor__");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).__ixor__(
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::__or__(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("__or__");
      
          auto result_ = at::__or__(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::__or__(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("__or__");
      
          auto result_ = at::__or__(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::__xor__(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("__xor__");
      
          auto result_ = at::__xor__(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::__xor__(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("__xor__");
      
          auto result_ = at::__xor__(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_cast_Double(Tensor self, bool non_blocking=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_cast_Double");
      
          auto result_ = at::_cast_Double(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_cast_Short(Tensor self, bool non_blocking=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_cast_Short");
      
          auto result_ = at::_cast_Short(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_coalesced_(Tensor(a!) self, bool coalesced) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_coalesced_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self)._coalesced_(
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_ctc_loss");
      
          auto result_ = at::_ctc_loss(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toIntList()->elements(),
              (std::move(peek(stack, 3, 5))).toIntList()->elements(),
              (std::move(peek(stack, 4, 5))).toInt()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_cudnn_init_dropout_state(float dropout, bool train, int dropout_seed, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_cudnn_init_dropout_state");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 3, 6))).toScalarType())
                  .layout((std::move(peek(stack, 4, 6))).toLayout())
                  .device((std::move(peek(stack, 5, 6))).toDevice());
          auto result_ = torch::_cudnn_init_dropout_state((std::move(peek(stack, 0, 6))).toDouble(),
          (std::move(peek(stack, 1, 6))).toBool(),
          (std::move(peek(stack, 2, 6))).toInt(),
          options);
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_cudnn_rnn_flatten_weight(Tensor[] weight_arr, int weight_stride0, int input_size, int mode, int hidden_size, int num_layers, bool batch_first, bool bidirectional) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_cudnn_rnn_flatten_weight");
      
          auto result_ = at::_cudnn_rnn_flatten_weight(
              (std::move(peek(stack, 0, 8))).toTensorList()->elements(),
              (std::move(peek(stack, 1, 8))).toInt(),
              (std::move(peek(stack, 2, 8))).toInt(),
              (std::move(peek(stack, 3, 8))).toInt(),
              (std::move(peek(stack, 4, 8))).toInt(),
              (std::move(peek(stack, 5, 8))).toInt(),
              (std::move(peek(stack, 6, 8))).toBool(),
              (std::move(peek(stack, 7, 8))).toBool()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_cufft_get_plan_cache_max_size() -> int",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_cufft_get_plan_cache_max_size");
      
          auto result_ = at::_cufft_get_plan_cache_max_size(
          
          );
          drop(stack, 0);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_dimI(Tensor self) -> int",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_dimI");
      
          auto result_ = ((std::move(peek(stack, 0, 1))).toTensor())._dimI(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_dirichlet_grad(Tensor x, Tensor alpha, Tensor total) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_dirichlet_grad");
      
          auto result_ = at::_dirichlet_grad(
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
      "aten::_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False) -> (Tensor, Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_embedding_bag");
      
          auto result_ = at::_embedding_bag(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toTensor(),
              (std::move(peek(stack, 3, 6))).toBool(),
              (std::move(peek(stack, 4, 6))).toInt(),
              (std::move(peek(stack, 5, 6))).toBool()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_embedding_bag_dense_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, int num_weights, bool scale_grad_by_freq, int mode) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_embedding_bag_dense_backward");
      
          auto result_ = at::_embedding_bag_dense_backward(
              (std::move(peek(stack, 0, 9))).toTensor(),
              (std::move(peek(stack, 1, 9))).toTensor(),
              (std::move(peek(stack, 2, 9))).toTensor(),
              (std::move(peek(stack, 3, 9))).toTensor(),
              (std::move(peek(stack, 4, 9))).toTensor(),
              (std::move(peek(stack, 5, 9))).toTensor(),
              (std::move(peek(stack, 6, 9))).toInt(),
              (std::move(peek(stack, 7, 9))).toBool(),
              (std::move(peek(stack, 8, 9))).toInt()
          );
          drop(stack, 9);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_indices(Tensor(a) self) -> Tensor(a)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_indices");
      
          auto result_ = ((std::move(peek(stack, 0, 1))).toTensor())._indices(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_nnz(Tensor self) -> int",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_nnz");
      
          auto result_ = ((std::move(peek(stack, 0, 1))).toTensor())._nnz(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_reshape_from_tensor(Tensor self, Tensor shape) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_reshape_from_tensor");
      
          auto result_ = at::_reshape_from_tensor(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_sparse_add(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_sparse_add_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_sparse_add_out(
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
      "aten::_sparse_coo_tensor_with_dims(int sparse_dim, int dense_dim, int[] size, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_sparse_coo_tensor_with_dims");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 3, 6))).toScalarType())
                  .layout((std::move(peek(stack, 4, 6))).toLayout())
                  .device((std::move(peek(stack, 5, 6))).toDevice());
          auto result_ = torch::_sparse_coo_tensor_with_dims((std::move(peek(stack, 0, 6))).toInt(),
          (std::move(peek(stack, 1, 6))).toInt(),
          (std::move(peek(stack, 2, 6))).toIntList()->elements(),
          options);
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_sparse_mul_scalar(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_sparse_mul_scalar_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_sparse_mul_scalar_out(
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
      "aten::_sparse_mul_zerodim(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_sparse_mul_zerodim_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_sparse_mul_zerodim_out(
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
      "aten::_sparse_sum_backward(Tensor grad, Tensor self, int[] dim) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_sparse_sum_backward");
      
          auto result_ = at::_sparse_sum_backward(
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
      "aten::_standard_gamma_grad(Tensor self, Tensor output) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_standard_gamma_grad");
      
          auto result_ = at::_standard_gamma_grad(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_acos(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_acos_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_acos_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_addbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_addbmm_");
          auto self = (std::move(peek(stack, 0, 5))).toTensor();
          auto result_ = at::_th_addbmm_(
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
      "aten::_th_addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_addcdiv_out");
          auto result = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::_th_addcdiv_out(
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
      "aten::_th_addcmul_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_addcmul_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = at::_th_addcmul_(
              self,
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
      "aten::_th_addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_addmm_out");
          auto result = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::_th_addmm_out(
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
      "aten::_th_addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_addmv_out");
          auto result = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::_th_addmv_out(
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
      "aten::_th_addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_addr");
      
          auto result_ = at::_th_addr(
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
      "aten::_th_alias(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_alias");
      
          auto result_ = at::_th_alias(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_atan(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_atan");
      
          auto result_ = at::_th_atan(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_atan2(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_atan2_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_atan2_out(
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
      "aten::_th_baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_baddbmm");
      
          auto result_ = at::_th_baddbmm(
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
      "aten::_th_btrisolve(Tensor self, Tensor LU_data, Tensor LU_pivots, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_btrisolve_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_th_btrisolve_out(
              result,
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
      "aten::_th_cat(Tensor[] tensors, int dim=0) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_cat");
      
          auto result_ = at::_th_cat(
              (std::move(peek(stack, 0, 2))).toTensorList()->elements(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator generator=None) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_cauchy_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = at::_th_cauchy_(
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
      "aten::_th_ceil(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_ceil_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_ceil_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_clamp(Tensor self, Scalar min, Scalar max) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_clamp");
      
          auto result_ = at::_th_clamp(
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
      "aten::_th_clamp_min(Tensor self, Scalar min, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_clamp_min_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_clamp_min_out(
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
      "aten::_th_cosh(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_cosh");
      
          auto result_ = at::_th_cosh(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_cross(Tensor self, Tensor other, int dim=-1, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_cross_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_th_cross_out(
              result,
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
      "aten::_th_cumprod(Tensor self, int dim) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_cumprod");
      
          auto result_ = at::_th_cumprod(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_cumsum(Tensor self, int dim) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_cumsum");
      
          auto result_ = at::_th_cumsum(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_diag(Tensor self, int diagonal=0) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_diag");
      
          auto result_ = at::_th_diag(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_digamma(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_digamma");
      
          auto result_ = at::_th_digamma(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_dirichlet_grad(Tensor x, Tensor alpha, Tensor total) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_dirichlet_grad");
      
          auto result_ = at::_th_dirichlet_grad(
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
      "aten::_th_dot(Tensor self, Tensor tensor) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_dot");
      
          auto result_ = at::_th_dot(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_eig(Tensor self, bool eigenvectors=False) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_eig");
      
          auto result_ = at::_th_eig(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_eq_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_eq_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_eq_(
              self,
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_eq_(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_eq_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_eq_(
              self,
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_erf(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_erf_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_erf_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_erfc(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_erfc_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_erfc_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_erfinv(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_erfinv_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_erfinv_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_exp(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_exp");
      
          auto result_ = at::_th_exp(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_expm1(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_expm1_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_expm1_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_floor(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_floor");
      
          auto result_ = at::_th_floor(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_fmod(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_fmod_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_fmod_out(
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
      "aten::_th_fmod(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_fmod_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_fmod_out(
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
      "aten::_th_frac_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_frac_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::_th_frac_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_gather(Tensor self, int dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_gather_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_th_gather_out(
              result,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toInt(),
              (std::move(peek(stack, 2, 4))).toTensor()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_ge(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_ge");
      
          auto result_ = at::_th_ge(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_ge(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_ge");
      
          auto result_ = at::_th_ge(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_ger(Tensor self, Tensor vec2) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_ger");
      
          auto result_ = at::_th_ger(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_gt(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_gt");
      
          auto result_ = at::_th_gt(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_gt(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_gt");
      
          auto result_ = at::_th_gt(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_histc_out");
          auto result = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::_th_histc_out(
              result,
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toInt(),
              (std::move(peek(stack, 2, 5))).toScalar(),
              (std::move(peek(stack, 3, 5))).toScalar()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_iand_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_iand_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_iand_(
              self,
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_iand_(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_iand_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_iand_(
              self,
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_index_add_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = at::_th_index_add_(
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
      "aten::_th_index_fill_(Tensor(a!) self, int dim, Tensor index, Tensor value) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_index_fill_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = at::_th_index_fill_(
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
      "aten::_th_index_fill_(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_index_fill_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = at::_th_index_fill_(
              self,
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
      "aten::_th_index_select(Tensor self, int dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_index_select_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_th_index_select_out(
              result,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toInt(),
              (std::move(peek(stack, 2, 4))).toTensor()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_is_set_to(Tensor self, Tensor tensor) -> bool",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_is_set_to");
      
          auto result_ = at::_th_is_set_to(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_le(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_le_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_le_out(
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
      "aten::_th_le(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_le_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_le_out(
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
      "aten::_th_lerp(Tensor self, Tensor end, Scalar weight) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_lerp");
      
          auto result_ = at::_th_lerp(
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
      "aten::_th_lgamma(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_lgamma_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_lgamma_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_log10(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_log10_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_log10_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_log1p(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_log1p");
      
          auto result_ = at::_th_log1p(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_log2(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_log2");
      
          auto result_ = at::_th_log2(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_log_normal_(Tensor(a!) self, float mean=1, float std=2, *, Generator generator=None) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_log_normal_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = at::_th_log_normal_(
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
      "aten::_th_lshift(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_lshift_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_lshift_out(
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
      "aten::_th_lshift(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_lshift_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_lshift_out(
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
      "aten::_th_lt(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_lt_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_lt_out(
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
      "aten::_th_lt(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_lt_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_lt_out(
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
      "aten::_th_masked_fill_(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_masked_fill_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = at::_th_masked_fill_(
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
      "aten::_th_masked_fill_(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_masked_fill_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = at::_th_masked_fill_(
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
      "aten::_th_masked_select(Tensor self, Tensor mask, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_masked_select_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_masked_select_out(
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
      "aten::_th_max(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_max_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_max_out(
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
      "aten::_th_min(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_min");
      
          auto result_ = at::_th_min(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_min(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_min");
      
          auto result_ = at::_th_min(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_min(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_min");
      
          auto result_ = at::_th_min(
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
      "aten::_th_mm(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_mm_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_mm_out(
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
      "aten::_th_mv(Tensor self, Tensor vec, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_mv_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_mv_out(
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
      "aten::_th_ne_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_ne_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_ne_(
              self,
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_ne_(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_ne_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_ne_(
              self,
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_neg(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_neg_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_neg_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_nonzero(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_nonzero_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_nonzero_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_norm(Tensor self, Scalar p=2) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_norm");
      
          auto result_ = at::_th_norm(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_norm(Tensor self, Scalar p, int dim, bool keepdim=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_norm");
      
          auto result_ = at::_th_norm(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toScalar(),
              (std::move(peek(stack, 2, 4))).toInt(),
              (std::move(peek(stack, 3, 4))).toBool()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator generator=None) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_normal_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = at::_th_normal_(
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
      "aten::_th_or(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_or");
      
          auto result_ = at::_th_or(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_or(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_or");
      
          auto result_ = at::_th_or(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_orgqr(Tensor self, Tensor input2) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_orgqr");
      
          auto result_ = at::_th_orgqr(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_ormqr");
      
          auto result_ = at::_th_ormqr(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toTensor(),
              (std::move(peek(stack, 3, 5))).toBool(),
              (std::move(peek(stack, 4, 5))).toBool()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_polygamma_(Tensor(a!) self, int n) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_polygamma_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_polygamma_(
              self,
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_potrf_single(Tensor self, bool upper=True, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_potrf_single_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_potrf_single_out(
              output,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_potri(Tensor self, bool upper=True) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_potri");
      
          auto result_ = at::_th_potri(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_potrs_single(Tensor self, Tensor input2, bool upper=True) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_potrs_single");
      
          auto result_ = at::_th_potrs_single(
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
      "aten::_th_pow(Tensor self, Tensor exponent) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_pow");
      
          auto result_ = at::_th_pow(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_pow(Scalar self, Tensor exponent) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_pow");
      
          auto result_ = at::_th_pow(
              (std::move(peek(stack, 0, 2))).toScalar(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_pow(Tensor self, Scalar exponent) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_pow");
      
          auto result_ = at::_th_pow(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_reciprocal_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_reciprocal_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::_th_reciprocal_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_remainder(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_remainder");
      
          auto result_ = at::_th_remainder(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_remainder(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_remainder");
      
          auto result_ = at::_th_remainder(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_renorm(Tensor self, Scalar p, int dim, Scalar maxnorm, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_renorm_out");
          auto result = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::_th_renorm_out(
              result,
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toScalar(),
              (std::move(peek(stack, 2, 5))).toInt(),
              (std::move(peek(stack, 3, 5))).toScalar()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_round(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_round");
      
          auto result_ = at::_th_round(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_rshift(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_rshift_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_rshift_out(
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
      "aten::_th_rshift(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_rshift_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_rshift_out(
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
      "aten::_th_rsqrt(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_rsqrt_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_rsqrt_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_scatter_add_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = at::_th_scatter_add_(
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
      "aten::_th_sign_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_sign_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::_th_sign_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_sin(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_sin");
      
          auto result_ = at::_th_sin(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_sinh(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_sinh_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_sinh_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_sort");
      
          auto result_ = at::_th_sort(
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
      "aten::_th_sqrt(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_sqrt_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_sqrt_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_svd(Tensor self, bool some=True, bool compute_uv=True) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_svd");
      
          auto result_ = at::_th_svd(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toBool(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_symeig(Tensor self, bool eigenvectors=False, bool upper=True) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_symeig");
      
          auto result_ = at::_th_symeig(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toBool(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_tan(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_tan_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_tan_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_tril(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_tril_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_tril_out(
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
      "aten::_th_triu(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_triu_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_triu_out(
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
      "aten::_th_unfold(Tensor self, int dimension, int size, int step) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_unfold");
      
          auto result_ = at::_th_unfold(
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
      "aten::_th_var(Tensor self, int dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_var_out");
          auto result = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::_th_var_out(
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
      "aten::_th_xor(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_xor");
      
          auto result_ = at::_th_xor(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_xor(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_xor");
      
          auto result_ = at::_th_xor(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_zero_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_zero_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::_th_zero_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_adaptive_avg_pool3d_backward(Tensor grad_output, Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_adaptive_avg_pool3d_backward");
      
          auto result_ = at::_thnn_adaptive_avg_pool3d_backward(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_adaptive_max_pool3d_backward(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_adaptive_max_pool3d_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_adaptive_max_pool3d_backward_out(
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
      "aten::_thnn_adaptive_max_pool3d_forward(Tensor self, int[3] output_size) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_adaptive_max_pool3d_forward");
      
          auto result_ = at::_thnn_adaptive_max_pool3d_forward(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_avg_pool2d_backward");
      
          auto result_ = at::_thnn_avg_pool2d_backward(
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              (std::move(peek(stack, 2, 7))).toIntList()->elements(),
              (std::move(peek(stack, 3, 7))).toIntList()->elements(),
              (std::move(peek(stack, 4, 7))).toIntList()->elements(),
              (std::move(peek(stack, 5, 7))).toBool(),
              (std::move(peek(stack, 6, 7))).toBool()
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_avg_pool3d_forward(Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_avg_pool3d_forward_out");
          auto output = (std::move(peek(stack, 6, 7))).toTensor();
          auto result_ = at::_thnn_avg_pool3d_forward_out(
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
      "aten::_thnn_binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_binary_cross_entropy_backward_out");
          auto grad_input = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::_thnn_binary_cross_entropy_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toTensor(),
              (std::move(peek(stack, 3, 6))).toTensor(),
              (std::move(peek(stack, 4, 6))).toInt()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_binary_cross_entropy_forward(Tensor self, Tensor target, Tensor? weight, int reduction) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_binary_cross_entropy_forward");
      
          auto result_ = at::_thnn_binary_cross_entropy_forward(
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
      "aten::_thnn_col2im_backward(Tensor grad_output, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_col2im_backward");
      
          auto result_ = at::_thnn_col2im_backward(
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
      "aten::_thnn_conv2d_backward(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, Tensor finput, Tensor fgrad_input, bool[3] output_mask=[True, True, True]) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_conv2d_backward");
      
          auto result_ = at::_thnn_conv2d_backward(
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
      "aten::_thnn_conv_depthwise2d_backward(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool[2] output_mask=[True, True]) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_conv_depthwise2d_backward");
      
          auto result_ = at::_thnn_conv_depthwise2d_backward(
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toTensor(),
              (std::move(peek(stack, 3, 8))).toIntList()->elements(),
              (std::move(peek(stack, 4, 8))).toIntList()->elements(),
              (std::move(peek(stack, 5, 8))).toIntList()->elements(),
              (std::move(peek(stack, 6, 8))).toIntList()->elements(),
              as_bool_array<2>((std::move(peek(stack, 7, 8))).toIntList()->elements())
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_conv_dilated2d_backward(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, Tensor columns, Tensor ones, bool[3] output_mask=[True, True, True]) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_conv_dilated2d_backward");
      
          auto result_ = at::_thnn_conv_dilated2d_backward(
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
      "aten::_thnn_conv_transpose2d_backward(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] output_padding, int[2] dilation, Tensor columns, Tensor ones, bool[3] output_mask=[True, True, True]) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_conv_transpose2d_backward");
      
          auto result_ = at::_thnn_conv_transpose2d_backward(
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
      "aten::_thnn_elu_backward(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, Tensor output, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_elu_backward_out");
          auto grad_input = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::_thnn_elu_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toScalar(),
              (std::move(peek(stack, 2, 6))).toScalar(),
              (std::move(peek(stack, 3, 6))).toScalar(),
              (std::move(peek(stack, 4, 6))).toTensor()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_elu_forward(Tensor self, Scalar alpha, Scalar scale, Scalar input_scale) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_elu_forward");
      
          auto result_ = at::_thnn_elu_forward(
              (std::move(peek(stack, 0, 4))).toTensor(),
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
      "aten::_thnn_fused_gru_cell(Tensor input_gates, Tensor hidden_gates, Tensor hx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_fused_gru_cell");
      
          auto result_ = at::_thnn_fused_gru_cell(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toTensor(),
              (std::move(peek(stack, 3, 5))).toTensor(),
              (std::move(peek(stack, 4, 5))).toTensor()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_fused_lstm_cell(Tensor input_gates, Tensor hidden_gates, Tensor cx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_fused_lstm_cell");
      
          auto result_ = at::_thnn_fused_lstm_cell(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toTensor(),
              (std::move(peek(stack, 3, 5))).toTensor(),
              (std::move(peek(stack, 4, 5))).toTensor()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_glu_forward(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_glu_forward_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_thnn_glu_forward_out(
              output,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_hardtanh_backward_out");
          auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::_thnn_hardtanh_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toScalar(),
              (std::move(peek(stack, 3, 5))).toScalar()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_hardtanh_forward(Tensor self, Scalar min_val, Scalar max_val) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_hardtanh_forward");
      
          auto result_ = at::_thnn_hardtanh_forward(
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
      "aten::_thnn_im2col_backward(Tensor grad_output, int[2] input_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_im2col_backward");
      
          auto result_ = at::_thnn_im2col_backward(
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
      "aten::_thnn_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_l1_loss_backward_out");
          auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::_thnn_l1_loss_backward_out(
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
      "aten::_thnn_l1_loss_forward(Tensor self, Tensor target, int reduction) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_l1_loss_forward");
      
          auto result_ = at::_thnn_l1_loss_forward(
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
      "aten::_thnn_leaky_relu_forward(Tensor self, Scalar negative_slope, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_leaky_relu_forward_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_thnn_leaky_relu_forward_out(
              output,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_max_pool2d_with_indices_backward");
      
          auto result_ = at::_thnn_max_pool2d_with_indices_backward(
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toIntList()->elements(),
              (std::move(peek(stack, 3, 8))).toIntList()->elements(),
              (std::move(peek(stack, 4, 8))).toIntList()->elements(),
              (std::move(peek(stack, 5, 8))).toIntList()->elements(),
              (std::move(peek(stack, 6, 8))).toBool(),
              (std::move(peek(stack, 7, 8))).toTensor()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_max_unpool2d_forward(Tensor self, Tensor indices, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_max_unpool2d_forward_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_max_unpool2d_forward_out(
              output,
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
      "aten::_thnn_max_unpool3d_backward(Tensor grad_output, Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_max_unpool3d_backward_out");
          auto grad_input = (std::move(peek(stack, 6, 7))).toTensor();
          auto result_ = at::_thnn_max_unpool3d_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              (std::move(peek(stack, 2, 7))).toTensor(),
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
      "aten::_thnn_max_unpool3d_forward(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_max_unpool3d_forward");
      
          auto result_ = at::_thnn_max_unpool3d_forward(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
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
      "aten::_thnn_mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_mse_loss_backward_out");
          auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::_thnn_mse_loss_backward_out(
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
      "aten::_thnn_mse_loss_forward(Tensor self, Tensor target, int reduction) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_mse_loss_forward");
      
          auto result_ = at::_thnn_mse_loss_forward(
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
      "aten::_thnn_multi_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight, int reduction, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_multi_margin_loss_backward_out");
          auto grad_input = (std::move(peek(stack, 7, 8))).toTensor();
          auto result_ = at::_thnn_multi_margin_loss_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toTensor(),
              (std::move(peek(stack, 3, 8))).toScalar(),
              (std::move(peek(stack, 4, 8))).toScalar(),
              (std::move(peek(stack, 5, 8))).toTensor(),
              (std::move(peek(stack, 6, 8))).toInt()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_multi_margin_loss_forward(Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight, int reduction) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_multi_margin_loss_forward");
      
          auto result_ = at::_thnn_multi_margin_loss_forward(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toScalar(),
              (std::move(peek(stack, 3, 6))).toScalar(),
              (std::move(peek(stack, 4, 6))).toTensor(),
              (std::move(peek(stack, 5, 6))).toInt()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_nll_loss2d_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_nll_loss2d_backward");
      
          auto result_ = at::_thnn_nll_loss2d_backward(
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              (std::move(peek(stack, 2, 7))).toTensor(),
              (std::move(peek(stack, 3, 7))).toTensor(),
              (std::move(peek(stack, 4, 7))).toInt(),
              (std::move(peek(stack, 5, 7))).toInt(),
              (std::move(peek(stack, 6, 7))).toTensor()
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_nll_loss_backward");
      
          auto result_ = at::_thnn_nll_loss_backward(
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              (std::move(peek(stack, 2, 7))).toTensor(),
              (std::move(peek(stack, 3, 7))).toTensor(),
              (std::move(peek(stack, 4, 7))).toInt(),
              (std::move(peek(stack, 5, 7))).toInt(),
              (std::move(peek(stack, 6, 7))).toTensor()
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_reflection_pad1d_backward(Tensor grad_output, Tensor self, int[2] padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_reflection_pad1d_backward");
      
          auto result_ = at::_thnn_reflection_pad1d_backward(
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
      "aten::_thnn_reflection_pad2d_forward(Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_reflection_pad2d_forward_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_thnn_reflection_pad2d_forward_out(
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
      "aten::_thnn_replication_pad1d_backward(Tensor grad_output, Tensor self, int[2] padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_replication_pad1d_backward");
      
          auto result_ = at::_thnn_replication_pad1d_backward(
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
      "aten::_thnn_replication_pad2d_forward(Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_replication_pad2d_forward_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_thnn_replication_pad2d_forward_out(
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
      "aten::_thnn_replication_pad3d_backward(Tensor grad_output, Tensor self, int[6] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_replication_pad3d_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_replication_pad3d_backward_out(
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
      "aten::_thnn_replication_pad3d_forward(Tensor self, int[6] padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_replication_pad3d_forward");
      
          auto result_ = at::_thnn_replication_pad3d_forward(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_rrelu_with_noise_backward(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_rrelu_with_noise_backward_out");
          auto grad_input = (std::move(peek(stack, 6, 7))).toTensor();
          auto result_ = at::_thnn_rrelu_with_noise_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              (std::move(peek(stack, 2, 7))).toTensor(),
              (std::move(peek(stack, 3, 7))).toScalar(),
              (std::move(peek(stack, 4, 7))).toScalar(),
              (std::move(peek(stack, 5, 7))).toBool()
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_sigmoid_forward(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_sigmoid_forward_out");
          auto output = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_thnn_sigmoid_forward_out(
              output,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_smooth_l1_loss_forward(Tensor self, Tensor target, int reduction, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_smooth_l1_loss_forward_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_smooth_l1_loss_forward_out(
              output,
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
      "aten::_thnn_soft_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_soft_margin_loss_backward_out");
          auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::_thnn_soft_margin_loss_backward_out(
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
      "aten::_thnn_soft_margin_loss_forward(Tensor self, Tensor target, int reduction) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_soft_margin_loss_forward");
      
          auto result_ = at::_thnn_soft_margin_loss_forward(
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
      "aten::_thnn_softplus_backward(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, Tensor output, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_softplus_backward_out");
          auto grad_input = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::_thnn_softplus_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toScalar(),
              (std::move(peek(stack, 3, 6))).toScalar(),
              (std::move(peek(stack, 4, 6))).toTensor()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_softplus_forward(Tensor self, Scalar beta, Scalar threshold) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_softplus_forward");
      
          auto result_ = at::_thnn_softplus_forward(
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
      "aten::_thnn_softshrink_forward(Tensor self, Scalar lambd, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_softshrink_forward_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_thnn_softshrink_forward_out(
              output,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_tanh_forward(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_tanh_forward_out");
          auto output = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_thnn_tanh_forward_out(
              output,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_upsample_bicubic2d_forward(Tensor self, int[2] output_size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_bicubic2d_forward_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_upsample_bicubic2d_forward_out(
              output,
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
      "aten::_thnn_upsample_bilinear2d_forward(Tensor self, int[2] output_size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_bilinear2d_forward_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_upsample_bilinear2d_forward_out(
              output,
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
      "aten::_thnn_upsample_linear1d_forward(Tensor self, int[1] output_size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_linear1d_forward_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_upsample_linear1d_forward_out(
              output,
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
      "aten::_thnn_upsample_nearest1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_nearest1d_backward");
      
          auto result_ = at::_thnn_upsample_nearest1d_backward(
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
      "aten::_thnn_upsample_nearest2d_forward(Tensor self, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_nearest2d_forward_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_thnn_upsample_nearest2d_forward_out(
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
      "aten::_thnn_upsample_nearest3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_nearest3d_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_upsample_nearest3d_backward_out(
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
      "aten::_thnn_upsample_nearest3d_forward(Tensor self, int[3] output_size) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_nearest3d_forward");
      
          auto result_ = at::_thnn_upsample_nearest3d_forward(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_upsample_trilinear3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_trilinear3d_backward_out");
          auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::_thnn_upsample_trilinear3d_backward_out(
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
      "aten::_thnn_upsample_trilinear3d_forward(Tensor self, int[3] output_size, bool align_corners) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_trilinear3d_forward");
      
          auto result_ = at::_thnn_upsample_trilinear3d_forward(
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
      "aten::_unique(Tensor self, bool sorted=False, bool return_inverse=False) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_unique");
      
          auto result_ = at::_unique(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toBool(),
              (std::move(peek(stack, 2, 3))).toBool()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::abs(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("abs_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::abs_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::acos(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("acos");
      
          auto result_ = at::acos(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("adaptive_avg_pool2d");
      
          auto result_ = at::adaptive_avg_pool2d(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("adaptive_avg_pool2d_backward_out");
          auto grad_input = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::adaptive_avg_pool2d_backward_out(
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
      "aten::adaptive_avg_pool3d_backward(Tensor grad_output, Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("adaptive_avg_pool3d_backward");
      
          auto result_ = at::adaptive_avg_pool3d_backward(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::adaptive_max_pool3d(Tensor self, int[3] output_size) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("adaptive_max_pool3d");
      
          auto result_ = at::adaptive_max_pool3d(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::adaptive_max_pool3d_backward(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("adaptive_max_pool3d_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::adaptive_max_pool3d_backward_out(
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
      "aten::add_(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("add_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = (self).add_(
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::add_(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("add_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = (self).add_(
              (std::move(peek(stack, 1, 3))).toScalar(),
              (std::move(peek(stack, 2, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("addbmm_out");
          auto result = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::addbmm_out(
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
      "aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("addcdiv");
      
          auto result_ = at::addcdiv(
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
      "aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("addcmul_out");
          auto result = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::addcmul_out(
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
      "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("addmm");
      
          auto result_ = at::addmm(
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
      "aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("addmv");
      
          auto result_ = at::addmv(
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
      "aten::addr_(Tensor(a!) self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("addr_");
          auto self = (std::move(peek(stack, 0, 5))).toTensor();
          auto result_ = (self).addr_(
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
      "aten::affine_grid_generator_backward(Tensor grad, int[] size) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("affine_grid_generator_backward");
      
          auto result_ = at::affine_grid_generator_backward(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::all(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("all_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::all_out(
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
      "aten::alpha_dropout(Tensor input, float p, bool train) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("alpha_dropout");
      
          auto result_ = at::alpha_dropout(
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
      "aten::any(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("any_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::any_out(
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
      "aten::arange(Scalar end, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("arange_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::arange_out(
              result,
              (std::move(peek(stack, 0, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::arange(Scalar start, Scalar end, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("arange_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::arange_out(
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
      "aten::arange(Scalar start, Scalar end, Scalar step, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("arange_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::arange_out(
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
      "aten::argmin(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("argmin");
      
          auto result_ = at::argmin(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::argmin(Tensor self, int dim, bool keepdim=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("argmin");
      
          auto result_ = at::argmin(
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
      "aten::as_strided(Tensor(a) self, int[] size, int[] stride) -> Tensor(a)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("as_strided");
      
          auto result_ = at::as_strided(
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
      "aten::as_strided(Tensor(a) self, int[] size, int[] stride, int storage_offset) -> Tensor(a)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("as_strided");
      
          auto result_ = at::as_strided(
              (std::move(peek(stack, 0, 4))).toTensor(),
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
      "aten::asin(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("asin_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::asin_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::atan2(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("atan2");
      
          auto result_ = at::atan2(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::atan_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("atan_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::atan_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("avg_pool1d");
      
          auto result_ = at::avg_pool1d(
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
      "aten::avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("avg_pool2d_backward");
      
          auto result_ = at::avg_pool2d_backward(
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              (std::move(peek(stack, 2, 7))).toIntList()->elements(),
              (std::move(peek(stack, 3, 7))).toIntList()->elements(),
              (std::move(peek(stack, 4, 7))).toIntList()->elements(),
              (std::move(peek(stack, 5, 7))).toBool(),
              (std::move(peek(stack, 6, 7))).toBool()
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("avg_pool3d_out");
          auto output = (std::move(peek(stack, 6, 7))).toTensor();
          auto result_ = at::avg_pool3d_out(
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
      "aten::baddbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("baddbmm_");
          auto self = (std::move(peek(stack, 0, 5))).toTensor();
          auto result_ = (self).baddbmm_(
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
      "aten::bartlett_window(int window_length, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("bartlett_window");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 1, 4))).toScalarType())
                  .layout((std::move(peek(stack, 2, 4))).toLayout())
                  .device((std::move(peek(stack, 3, 4))).toDevice());
          auto result_ = torch::bartlett_window((std::move(peek(stack, 0, 4))).toInt(),
          options);
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::bartlett_window(int window_length, bool periodic, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("bartlett_window");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 2, 5))).toScalarType())
                  .layout((std::move(peek(stack, 3, 5))).toLayout())
                  .device((std::move(peek(stack, 4, 5))).toDevice());
          auto result_ = torch::bartlett_window((std::move(peek(stack, 0, 5))).toInt(),
          (std::move(peek(stack, 1, 5))).toBool(),
          options);
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::bernoulli_(Tensor(a!) self, Tensor p, *, Generator generator=None) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("bernoulli_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = (self).bernoulli_(
              (std::move(peek(stack, 1, 3))).toTensor(),
              nullptr
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::bernoulli_(Tensor(a!) self, float p=0.5, *, Generator generator=None) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("bernoulli_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = (self).bernoulli_(
              (std::move(peek(stack, 1, 3))).toDouble(),
              nullptr
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("binary_cross_entropy");
      
          auto result_ = at::binary_cross_entropy(
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
      "aten::binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor weight, int reduction, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("binary_cross_entropy_backward_out");
          auto grad_input = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::binary_cross_entropy_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toTensor(),
              (std::move(peek(stack, 3, 6))).toTensor(),
              (std::move(peek(stack, 4, 6))).toInt()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::bincount(Tensor self, Tensor? weights=None, int minlength=0) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("bincount");
      
          auto result_ = at::bincount(
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
      "aten::bmm(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("bmm_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::bmm_out(
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
      "aten::btrisolve(Tensor self, Tensor LU_data, Tensor LU_pivots) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("btrisolve");
      
          auto result_ = at::btrisolve(
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
      "aten::ceil(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ceil");
      
          auto result_ = at::ceil(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::celu(Tensor self, Scalar alpha=1.0) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("celu");
      
          auto result_ = at::celu(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cholesky(Tensor self, bool upper=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cholesky");
      
          auto result_ = at::cholesky(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("clamp_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = at::clamp_(
              self,
              (std::move(peek(stack, 1, 3))).toOptional<Scalar>(),
              (std::move(peek(stack, 2, 3))).toOptional<Scalar>()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::clamp_max(Tensor self, Scalar max, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("clamp_max_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::clamp_max_out(
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
      "aten::clamp_min(Tensor self, Scalar min) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("clamp_min");
      
          auto result_ = at::clamp_min(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("constant_pad_nd");
      
          auto result_ = at::constant_pad_nd(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toIntList()->elements(),
              (std::move(peek(stack, 2, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("conv1d");
      
          auto result_ = at::conv1d(
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
      "aten::conv_tbc_backward(Tensor self, Tensor input, Tensor weight, Tensor bias, int pad) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("conv_tbc_backward");
      
          auto result_ = at::conv_tbc_backward(
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
      "aten::conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] output_padding=0, int groups=1, int[1] dilation=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("conv_transpose1d");
      
          auto result_ = at::conv_transpose1d(
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
      "aten::cos(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cos_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::cos_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cosh_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cosh_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::cosh_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cosine_embedding_loss");
      
          auto result_ = at::cosine_embedding_loss(
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
      "aten::cross(Tensor self, Tensor other, int dim=-1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cross");
      
          auto result_ = at::cross(
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
      "aten::cudnn_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cudnn_convolution");
      
          auto result_ = at::cudnn_convolution(
              (std::move(peek(stack, 0, 9))).toTensor(),
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
      "aten::cudnn_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cudnn_convolution_backward_input");
      
          auto result_ = at::cudnn_convolution_backward_input(
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
      "aten::cudnn_convolution_transpose(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cudnn_convolution_transpose");
      
          auto result_ = at::cudnn_convolution_transpose(
              (std::move(peek(stack, 0, 10))).toTensor(),
              (std::move(peek(stack, 1, 10))).toTensor(),
              (std::move(peek(stack, 2, 10))).toTensor(),
              (std::move(peek(stack, 3, 10))).toIntList()->elements(),
              (std::move(peek(stack, 4, 10))).toIntList()->elements(),
              (std::move(peek(stack, 5, 10))).toIntList()->elements(),
              (std::move(peek(stack, 6, 10))).toIntList()->elements(),
              (std::move(peek(stack, 7, 10))).toInt(),
              (std::move(peek(stack, 8, 10))).toBool(),
              (std::move(peek(stack, 9, 10))).toBool()
          );
          drop(stack, 10);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cudnn_convolution_transpose_backward_input(Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cudnn_convolution_transpose_backward_input");
      
          auto result_ = at::cudnn_convolution_transpose_backward_input(
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toIntList()->elements(),
              (std::move(peek(stack, 3, 8))).toIntList()->elements(),
              (std::move(peek(stack, 4, 8))).toIntList()->elements(),
              (std::move(peek(stack, 5, 8))).toInt(),
              (std::move(peek(stack, 6, 8))).toBool(),
              (std::move(peek(stack, 7, 8))).toBool()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cudnn_grid_sampler(Tensor self, Tensor grid) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cudnn_grid_sampler");
      
          auto result_ = at::cudnn_grid_sampler(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::det(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("det");
      
          auto result_ = at::det(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::detach(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("detach");
      
          auto result_ = at::detach(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::digamma_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("digamma_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = (self).digamma_(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::dim(Tensor self) -> int",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("dim");
      
          auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).dim(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::div(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("div");
      
          auto result_ = at::div(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::div(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("div");
      
          auto result_ = at::div(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("elu");
      
          auto result_ = at::elu(
              (std::move(peek(stack, 0, 4))).toTensor(),
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
      "aten::elu_backward(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, Tensor output, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("elu_backward_out");
          auto grad_input = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::elu_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toScalar(),
              (std::move(peek(stack, 2, 6))).toScalar(),
              (std::move(peek(stack, 3, 6))).toScalar(),
              (std::move(peek(stack, 4, 6))).toTensor()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::embedding_backward(Tensor grad, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq, bool sparse) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("embedding_backward");
      
          auto result_ = at::embedding_backward(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toInt(),
              (std::move(peek(stack, 3, 6))).toInt(),
              (std::move(peek(stack, 4, 6))).toBool(),
              (std::move(peek(stack, 5, 6))).toBool()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::empty(int[] size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("empty_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::empty_out(
              result,
              (std::move(peek(stack, 0, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::eq(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("eq_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::eq_out(
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
      "aten::eq(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("eq_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::eq_out(
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
      "aten::equal(Tensor self, Tensor other) -> bool",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("equal");
      
          auto result_ = at::equal(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::erf(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("erf");
      
          auto result_ = at::erf(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::erfc(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("erfc");
      
          auto result_ = at::erfc(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::erfinv(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("erfinv");
      
          auto result_ = at::erfinv(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::exp_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("exp_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::exp_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::expand_as(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("expand_as");
      
          auto result_ = ((std::move(peek(stack, 0, 2))).toTensor()).expand_as(
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::expm1(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("expm1");
      
          auto result_ = at::expm1(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::exponential_(Tensor(a!) self, float lambd=1, *, Generator generator=None) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("exponential_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = (self).exponential_(
              (std::move(peek(stack, 1, 3))).toDouble(),
              nullptr
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::eye(int n, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("eye");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 1, 4))).toScalarType())
                  .layout((std::move(peek(stack, 2, 4))).toLayout())
                  .device((std::move(peek(stack, 3, 4))).toDevice());
          auto result_ = torch::eye((std::move(peek(stack, 0, 4))).toInt(),
          options);
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::eye(int n, int m, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("eye");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 2, 5))).toScalarType())
                  .layout((std::move(peek(stack, 3, 5))).toLayout())
                  .device((std::move(peek(stack, 4, 5))).toDevice());
          auto result_ = torch::eye((std::move(peek(stack, 0, 5))).toInt(),
          (std::move(peek(stack, 1, 5))).toInt(),
          options);
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::feature_alpha_dropout(Tensor input, float p, bool train) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("feature_alpha_dropout");
      
          auto result_ = at::feature_alpha_dropout(
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
      "aten::fft(Tensor self, int signal_ndim, bool normalized=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("fft");
      
          auto result_ = at::fft(
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
      "aten::fill_(Tensor(a!) self, Tensor value) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("fill_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::fill_(
              self,
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::fill_(Tensor(a!) self, Scalar value) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("fill_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::fill_(
              self,
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::floor_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("floor_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::floor_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::fmod(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("fmod");
      
          auto result_ = at::fmod(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::fmod(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("fmod");
      
          auto result_ = at::fmod(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::frac(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("frac_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::frac_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::gather(Tensor self, int dim, Tensor index) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("gather");
      
          auto result_ = at::gather(
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
      "aten::ge_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ge_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).ge_(
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::ge_(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ge_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).ge_(
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::geqrf(Tensor self) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("geqrf");
      
          auto result_ = at::geqrf(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::gesv(Tensor self, Tensor A) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("gesv");
      
          auto result_ = at::gesv(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::glu(Tensor self, int dim=-1, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("glu_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::glu_out(
              output,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::grid_sampler_2d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("grid_sampler_2d_backward");
      
          auto result_ = at::grid_sampler_2d_backward(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toTensor(),
              (std::move(peek(stack, 3, 5))).toInt(),
              (std::move(peek(stack, 4, 5))).toInt()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("gru_cell");
      
          auto result_ = at::gru_cell(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toTensor(),
              (std::move(peek(stack, 3, 6))).toTensor(),
              (std::move(peek(stack, 4, 6))).toTensor(),
              (std::move(peek(stack, 5, 6))).toTensor()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::gt_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("gt_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).gt_(
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::gt_(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("gt_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).gt_(
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::hamming_window(int window_length, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("hamming_window");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 1, 4))).toScalarType())
                  .layout((std::move(peek(stack, 2, 4))).toLayout())
                  .device((std::move(peek(stack, 3, 4))).toDevice());
          auto result_ = torch::hamming_window((std::move(peek(stack, 0, 4))).toInt(),
          options);
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::hamming_window(int window_length, bool periodic, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("hamming_window");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 2, 5))).toScalarType())
                  .layout((std::move(peek(stack, 3, 5))).toLayout())
                  .device((std::move(peek(stack, 4, 5))).toDevice());
          auto result_ = torch::hamming_window((std::move(peek(stack, 0, 5))).toInt(),
          (std::move(peek(stack, 1, 5))).toBool(),
          options);
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::hamming_window(int window_length, bool periodic, float alpha, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("hamming_window");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 3, 6))).toScalarType())
                  .layout((std::move(peek(stack, 4, 6))).toLayout())
                  .device((std::move(peek(stack, 5, 6))).toDevice());
          auto result_ = torch::hamming_window((std::move(peek(stack, 0, 6))).toInt(),
          (std::move(peek(stack, 1, 6))).toBool(),
          (std::move(peek(stack, 2, 6))).toDouble(),
          options);
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::hamming_window(int window_length, bool periodic, float alpha, float beta, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("hamming_window");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 4, 7))).toScalarType())
                  .layout((std::move(peek(stack, 5, 7))).toLayout())
                  .device((std::move(peek(stack, 6, 7))).toDevice());
          auto result_ = torch::hamming_window((std::move(peek(stack, 0, 7))).toInt(),
          (std::move(peek(stack, 1, 7))).toBool(),
          (std::move(peek(stack, 2, 7))).toDouble(),
          (std::move(peek(stack, 3, 7))).toDouble(),
          options);
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("hardshrink");
      
          auto result_ = at::hardshrink(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("hardtanh");
      
          auto result_ = at::hardtanh(
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
      "aten::hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("hardtanh_backward_out");
          auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::hardtanh_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toScalar(),
              (std::move(peek(stack, 3, 5))).toScalar()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("histc");
      
          auto result_ = at::histc(
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
      "aten::ifft(Tensor self, int signal_ndim, bool normalized=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ifft");
      
          auto result_ = at::ifft(
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
      "aten::index(Tensor self, Tensor[] indices) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("index");
      
          auto result_ = at::index(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensorList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("index_copy_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = (self).index_copy_(
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
      "aten::index_select(Tensor self, int dim, Tensor index) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("index_select");
      
          auto result_ = at::index_select(
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
      "aten::inverse(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("inverse");
      
          auto result_ = at::inverse(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::irfft(Tensor self, int signal_ndim, bool normalized=False, bool onesided=True, int[] signal_sizes=[]) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("irfft");
      
          auto result_ = at::irfft(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toInt(),
              (std::move(peek(stack, 2, 5))).toBool(),
              (std::move(peek(stack, 3, 5))).toBool(),
              (std::move(peek(stack, 4, 5))).toIntList()->elements()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::is_distributed(Tensor self) -> bool",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("is_distributed");
      
          auto result_ = at::is_distributed(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::is_nonzero(Tensor self) -> bool",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("is_nonzero");
      
          auto result_ = at::is_nonzero(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::item(Tensor self) -> Scalar",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("item");
      
          auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).item(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::kl_div_backward(Tensor grad_output, Tensor self, Tensor target, int reduction=Mean) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("kl_div_backward");
      
          auto result_ = at::kl_div_backward(
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
      "aten::l1_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("l1_loss");
      
          auto result_ = at::l1_loss(
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
      "aten::l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("l1_loss_backward_out");
          auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::l1_loss_backward_out(
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
      "aten::le(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("le");
      
          auto result_ = at::le(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::le(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("le");
      
          auto result_ = at::le(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::leaky_relu(Tensor self, Scalar negative_slope=0.01, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("leaky_relu_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::leaky_relu_out(
              output,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::lerp_(Tensor(a!) self, Tensor end, Scalar weight) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("lerp_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = (self).lerp_(
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::lgamma(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("lgamma");
      
          auto result_ = at::lgamma(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("linear");
      
          auto result_ = at::linear(
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
      "aten::linspace(Scalar start, Scalar end, int steps=100, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("linspace_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::linspace_out(
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
      "aten::log10(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("log10");
      
          auto result_ = at::log10(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::log1p_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("log1p_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::log1p_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::log2_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("log2_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::log2_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::log(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("log_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::log_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::log_sigmoid(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("log_sigmoid_out");
          auto output = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::log_sigmoid_out(
              output,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::log_softmax(Tensor self, int dim) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("log_softmax");
      
          auto result_ = at::log_softmax(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::log_softmax(Tensor self, int dim, ScalarType dtype) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("log_softmax");
      
          auto result_ = at::log_softmax(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt(),
              (std::move(peek(stack, 2, 3))).toScalarType()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("lstm_cell");
      
          auto result_ = at::lstm_cell(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensorList()->elements(),
              (std::move(peek(stack, 2, 6))).toTensor(),
              (std::move(peek(stack, 3, 6))).toTensor(),
              (std::move(peek(stack, 4, 6))).toTensor(),
              (std::move(peek(stack, 5, 6))).toTensor()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::lt(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("lt");
      
          auto result_ = at::lt(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::lt(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("lt");
      
          auto result_ = at::lt(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::masked_scatter_(Tensor(a!) self, Tensor mask, Tensor source) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("masked_scatter_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = (self).masked_scatter_(
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::masked_select(Tensor self, Tensor mask) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("masked_select");
      
          auto result_ = at::masked_select(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::matmul(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("matmul");
      
          auto result_ = at::matmul(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::matrix_power(Tensor self, int n) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("matrix_power");
      
          auto result_ = at::matrix_power(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::matrix_rank(Tensor self, bool symmetric=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("matrix_rank");
      
          auto result_ = at::matrix_rank(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::matrix_rank(Tensor self, float tol, bool symmetric=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("matrix_rank");
      
          auto result_ = at::matrix_rank(
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
      "aten::max(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("max");
      
          auto result_ = at::max(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::max(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("max");
      
          auto result_ = at::max(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::max(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("max");
      
          auto result_ = at::max(
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
      "aten::max_pool1d_with_indices(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("max_pool1d_with_indices");
      
          auto result_ = at::max_pool1d_with_indices(
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
      "aten::max_pool2d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("max_pool2d");
      
          auto result_ = at::max_pool2d(
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
      "aten::max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("max_pool2d_with_indices_backward");
      
          auto result_ = at::max_pool2d_with_indices_backward(
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toIntList()->elements(),
              (std::move(peek(stack, 3, 8))).toIntList()->elements(),
              (std::move(peek(stack, 4, 8))).toIntList()->elements(),
              (std::move(peek(stack, 5, 8))).toIntList()->elements(),
              (std::move(peek(stack, 6, 8))).toBool(),
              (std::move(peek(stack, 7, 8))).toTensor()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::max_unpool2d(Tensor self, Tensor indices, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("max_unpool2d_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::max_unpool2d_out(
              output,
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
      "aten::max_unpool3d(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("max_unpool3d");
      
          auto result_ = at::max_unpool3d(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
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
      "aten::max_unpool3d_backward(Tensor grad_output, Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("max_unpool3d_backward_out");
          auto grad_input = (std::move(peek(stack, 6, 7))).toTensor();
          auto result_ = at::max_unpool3d_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              (std::move(peek(stack, 2, 7))).toTensor(),
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
      "aten::meshgrid(Tensor[] tensors) -> Tensor[]",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("meshgrid");
      
          auto result_ = at::meshgrid(
              (std::move(peek(stack, 0, 1))).toTensorList()->elements()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::miopen_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("miopen_batch_norm");
      
          auto result_ = at::miopen_batch_norm(
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
      "aten::miopen_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("miopen_convolution_backward");
      
          auto result_ = at::miopen_convolution_backward(
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
      "aten::miopen_convolution_backward_bias(Tensor grad_output) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("miopen_convolution_backward_bias");
      
          auto result_ = at::miopen_convolution_backward_bias(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::miopen_convolution_transpose_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("miopen_convolution_transpose_backward");
      
          auto result_ = at::miopen_convolution_transpose_backward(
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
      "aten::mkldnn_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mkldnn_convolution");
      
          auto result_ = at::mkldnn_convolution(
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
      "aten::mkldnn_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool bias_defined) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mkldnn_convolution_backward_input");
      
          auto result_ = at::mkldnn_convolution_backward_input(
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
      "aten::mm(Tensor self, Tensor mat2) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mm");
      
          auto result_ = at::mm(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::mse_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mse_loss");
      
          auto result_ = at::mse_loss(
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
      "aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mse_loss_backward_out");
          auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::mse_loss_backward_out(
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
      "aten::mul(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mul_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::mul_out(
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
      "aten::multi_margin_loss(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("multi_margin_loss");
      
          auto result_ = at::multi_margin_loss(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toScalar(),
              (std::move(peek(stack, 3, 6))).toScalar(),
              (std::move(peek(stack, 4, 6))).toTensor(),
              (std::move(peek(stack, 5, 6))).toInt()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::multi_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor weight, int reduction, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("multi_margin_loss_backward_out");
          auto grad_input = (std::move(peek(stack, 7, 8))).toTensor();
          auto result_ = at::multi_margin_loss_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toTensor(),
              (std::move(peek(stack, 3, 8))).toScalar(),
              (std::move(peek(stack, 4, 8))).toScalar(),
              (std::move(peek(stack, 5, 8))).toTensor(),
              (std::move(peek(stack, 6, 8))).toInt()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::multilabel_margin_loss(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("multilabel_margin_loss_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::multilabel_margin_loss_out(
              output,
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
      "aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator generator=None, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("multinomial_out");
          auto result = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::multinomial_out(
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
      "aten::mv(Tensor self, Tensor vec) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mv");
      
          auto result_ = at::mv(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("narrow");
      
          auto result_ = at::narrow(
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
      "aten::native_clone(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("native_clone");
      
          auto result_ = at::native_clone(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::native_pow(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("native_pow_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::native_pow_out(
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
      "aten::native_resize_as_(Tensor(a!) self, Tensor the_template) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("native_resize_as_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::native_resize_as_(
              self,
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::ne(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ne_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::ne_out(
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
      "aten::ne(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ne_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::ne_out(
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
      "aten::neg(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("neg");
      
          auto result_ = at::neg(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::nll_loss2d_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("nll_loss2d_backward");
      
          auto result_ = at::nll_loss2d_backward(
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              (std::move(peek(stack, 2, 7))).toTensor(),
              (std::move(peek(stack, 3, 7))).toTensor(),
              (std::move(peek(stack, 4, 7))).toInt(),
              (std::move(peek(stack, 5, 7))).toInt(),
              (std::move(peek(stack, 6, 7))).toTensor()
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("nll_loss_backward");
      
          auto result_ = at::nll_loss_backward(
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              (std::move(peek(stack, 2, 7))).toTensor(),
              (std::move(peek(stack, 3, 7))).toTensor(),
              (std::move(peek(stack, 4, 7))).toInt(),
              (std::move(peek(stack, 5, 7))).toInt(),
              (std::move(peek(stack, 6, 7))).toTensor()
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::nonzero(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("nonzero");
      
          auto result_ = at::nonzero(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::normal(Tensor mean, Tensor std, *, Generator generator=None, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("normal_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::normal_out(
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
      "aten::normal(float mean, Tensor std, *, Generator generator=None, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("normal_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::normal_out(
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
      "aten::normal(Tensor mean, float std=1, *, Generator generator=None, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("normal_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::normal_out(
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
      "aten::nuclear_norm(Tensor self, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("nuclear_norm_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::nuclear_norm_out(
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
      "aten::numel(Tensor self) -> int",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("numel");
      
          auto result_ = at::numel(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::ones(int[] size, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ones");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 1, 4))).toScalarType())
                  .layout((std::move(peek(stack, 2, 4))).toLayout())
                  .device((std::move(peek(stack, 3, 4))).toDevice());
          auto result_ = torch::ones((std::move(peek(stack, 0, 4))).toIntList()->elements(),
          options);
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::ones_like(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ones_like");
      
          auto result_ = at::ones_like(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::ones_like(Tensor self, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ones_like");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 1, 4))).toScalarType())
                  .layout((std::move(peek(stack, 2, 4))).toLayout())
                  .device((std::move(peek(stack, 3, 4))).toDevice());
          auto result_ = torch::ones_like((std::move(peek(stack, 0, 4))).toTensor(),
          options);
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::pdist(Tensor self, float p=2) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("pdist");
      
          auto result_ = at::pdist(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toDouble()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("permute");
      
          auto result_ = ((std::move(peek(stack, 0, 2))).toTensor()).permute(
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::poisson(Tensor self, Generator generator=None) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("poisson");
      
          auto result_ = at::poisson(
              (std::move(peek(stack, 0, 2))).toTensor(),
              nullptr
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::polygamma(int n, Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("polygamma_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::polygamma_out(
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
      "aten::potrs(Tensor self, Tensor input2, bool upper=True, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("potrs_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::potrs_out(
              result,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toBool()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::pow_(Tensor(a!) self, Tensor exponent) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("pow_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).pow_(
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::pow_(Tensor(a!) self, Scalar exponent) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("pow_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).pow_(
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::prelu_backward(Tensor grad_output, Tensor self, Tensor weight) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("prelu_backward");
      
          auto result_ = at::prelu_backward(
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
      "aten::prod(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("prod");
      
          auto result_ = at::prod(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::prod(Tensor self, *, ScalarType dtype) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("prod");
      
          auto result_ = at::prod(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalarType()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::prod(Tensor self, int dim, bool keepdim=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("prod");
      
          auto result_ = at::prod(
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
      "aten::prod(Tensor self, int dim, *, ScalarType dtype) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("prod");
      
          auto result_ = at::prod(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt(),
              (std::move(peek(stack, 2, 3))).toScalarType()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::prod(Tensor self, int dim, bool keepdim, *, ScalarType dtype) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("prod");
      
          auto result_ = at::prod(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toInt(),
              (std::move(peek(stack, 2, 4))).toBool(),
              (std::move(peek(stack, 3, 4))).toScalarType()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::put_(Tensor(a!) self, Tensor index, Tensor source, bool accumulate=False) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("put_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = (self).put_(
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
      "aten::qr(Tensor self) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("qr");
      
          auto result_ = at::qr(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::rand(int[] size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("rand_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::rand_out(
              result,
              (std::move(peek(stack, 0, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::randint(int high, int[] size, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("randint");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 2, 5))).toScalarType())
                  .layout((std::move(peek(stack, 3, 5))).toLayout())
                  .device((std::move(peek(stack, 4, 5))).toDevice());
          auto result_ = torch::randint((std::move(peek(stack, 0, 5))).toInt(),
          (std::move(peek(stack, 1, 5))).toIntList()->elements(),
          options);
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::randint(int low, int high, int[] size, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("randint");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 3, 6))).toScalarType())
                  .layout((std::move(peek(stack, 4, 6))).toLayout())
                  .device((std::move(peek(stack, 5, 6))).toDevice());
          auto result_ = torch::randint((std::move(peek(stack, 0, 6))).toInt(),
          (std::move(peek(stack, 1, 6))).toInt(),
          (std::move(peek(stack, 2, 6))).toIntList()->elements(),
          options);
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::randint_like(Tensor self, int high) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("randint_like");
      
          auto result_ = at::randint_like(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::randint_like(Tensor self, int low, int high) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("randint_like");
      
          auto result_ = at::randint_like(
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
      "aten::randint_like(Tensor self, int high, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("randint_like");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 2, 5))).toScalarType())
                  .layout((std::move(peek(stack, 3, 5))).toLayout())
                  .device((std::move(peek(stack, 4, 5))).toDevice());
          auto result_ = torch::randint_like((std::move(peek(stack, 0, 5))).toTensor(),
          (std::move(peek(stack, 1, 5))).toInt(),
          options);
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::randint_like(Tensor self, int low, int high, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("randint_like");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 3, 6))).toScalarType())
                  .layout((std::move(peek(stack, 4, 6))).toLayout())
                  .device((std::move(peek(stack, 5, 6))).toDevice());
          auto result_ = torch::randint_like((std::move(peek(stack, 0, 6))).toTensor(),
          (std::move(peek(stack, 1, 6))).toInt(),
          (std::move(peek(stack, 2, 6))).toInt(),
          options);
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::randperm(int n, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("randperm");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 1, 4))).toScalarType())
                  .layout((std::move(peek(stack, 2, 4))).toLayout())
                  .device((std::move(peek(stack, 3, 4))).toDevice());
          auto result_ = torch::randperm((std::move(peek(stack, 0, 4))).toInt(),
          options);
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::reciprocal(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("reciprocal_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::reciprocal_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::reflection_pad1d_backward(Tensor grad_output, Tensor self, int[2] padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("reflection_pad1d_backward");
      
          auto result_ = at::reflection_pad1d_backward(
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
      "aten::reflection_pad2d(Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("reflection_pad2d_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::reflection_pad2d_out(
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
      "aten::relu(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("relu");
      
          auto result_ = at::relu(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::remainder_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("remainder_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).remainder_(
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::remainder_(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("remainder_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).remainder_(
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("renorm");
      
          auto result_ = at::renorm(
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
      "aten::repeat(Tensor self, int[] repeats) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("repeat");
      
          auto result_ = ((std::move(peek(stack, 0, 2))).toTensor()).repeat(
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::replication_pad1d_backward(Tensor grad_output, Tensor self, int[2] padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("replication_pad1d_backward");
      
          auto result_ = at::replication_pad1d_backward(
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
      "aten::replication_pad2d(Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("replication_pad2d_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::replication_pad2d_out(
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
      "aten::replication_pad3d(Tensor self, int[6] padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("replication_pad3d");
      
          auto result_ = at::replication_pad3d(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::replication_pad3d_backward(Tensor grad_output, Tensor self, int[6] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("replication_pad3d_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::replication_pad3d_backward_out(
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
      "aten::rfft(Tensor self, int signal_ndim, bool normalized=False, bool onesided=True) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("rfft");
      
          auto result_ = at::rfft(
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
      "aten::rnn_relu(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("rnn_relu");
      
          auto result_ = at::rnn_relu(
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
      "aten::rnn_relu(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("rnn_relu");
      
          auto result_ = at::rnn_relu(
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
      "aten::rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("rnn_tanh_cell");
      
          auto result_ = at::rnn_tanh_cell(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toTensor(),
              (std::move(peek(stack, 3, 6))).toTensor(),
              (std::move(peek(stack, 4, 6))).toTensor(),
              (std::move(peek(stack, 5, 6))).toTensor()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::rot90(Tensor self, int k=1, int[] dims=[0,1]) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("rot90");
      
          auto result_ = at::rot90(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt(),
              (std::move(peek(stack, 2, 3))).toIntList()->elements()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::round_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("round_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::round_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::rrelu(Tensor self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator generator=None) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("rrelu");
      
          auto result_ = at::rrelu(
              (std::move(peek(stack, 0, 5))).toTensor(),
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
      "aten::rrelu_with_noise(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator generator=None) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("rrelu_with_noise");
      
          auto result_ = at::rrelu_with_noise(
              (std::move(peek(stack, 0, 6))).toTensor(),
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
      "aten::rrelu_with_noise_backward(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("rrelu_with_noise_backward_out");
          auto grad_input = (std::move(peek(stack, 6, 7))).toTensor();
          auto result_ = at::rrelu_with_noise_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              (std::move(peek(stack, 2, 7))).toTensor(),
              (std::move(peek(stack, 3, 7))).toScalar(),
              (std::move(peek(stack, 4, 7))).toScalar(),
              (std::move(peek(stack, 5, 7))).toBool()
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::rsqrt(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("rsqrt");
      
          auto result_ = at::rsqrt(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::s_native_addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("s_native_addmm_");
          auto self = (std::move(peek(stack, 0, 5))).toTensor();
          auto result_ = at::s_native_addmm_(
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
      "aten::selu_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("selu_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::selu_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sigmoid(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sigmoid_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::sigmoid_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sign(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sign_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::sign_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sin_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sin_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::sin_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sinh(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sinh");
      
          auto result_ = at::sinh(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::size(Tensor self, int dim) -> int",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("size");
      
          auto result_ = at::size(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::smooth_l1_loss(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("smooth_l1_loss_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::smooth_l1_loss_out(
              output,
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
      "aten::soft_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("soft_margin_loss");
      
          auto result_ = at::soft_margin_loss(
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
      "aten::soft_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("soft_margin_loss_backward_out");
          auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::soft_margin_loss_backward_out(
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
      "aten::softmax(Tensor self, int dim) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("softmax");
      
          auto result_ = at::softmax(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::softmax(Tensor self, int dim, ScalarType dtype) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("softmax");
      
          auto result_ = at::softmax(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toInt(),
              (std::move(peek(stack, 2, 3))).toScalarType()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("softplus");
      
          auto result_ = at::softplus(
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
      "aten::softplus_backward(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, Tensor output, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("softplus_backward_out");
          auto grad_input = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::softplus_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toScalar(),
              (std::move(peek(stack, 3, 6))).toScalar(),
              (std::move(peek(stack, 4, 6))).toTensor()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::softshrink(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("softshrink_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::softshrink_out(
              output,
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sparse_coo_tensor(int[] size, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sparse_coo_tensor");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 1, 4))).toScalarType())
                  .layout((std::move(peek(stack, 2, 4))).toLayout())
                  .device((std::move(peek(stack, 3, 4))).toDevice());
          auto result_ = torch::sparse_coo_tensor((std::move(peek(stack, 0, 4))).toIntList()->elements(),
          options);
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sparse_coo_tensor(Tensor indices, Tensor values, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sparse_coo_tensor");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 2, 5))).toScalarType())
                  .layout((std::move(peek(stack, 3, 5))).toLayout())
                  .device((std::move(peek(stack, 4, 5))).toDevice());
          auto result_ = torch::sparse_coo_tensor((std::move(peek(stack, 0, 5))).toTensor(),
          (std::move(peek(stack, 1, 5))).toTensor(),
          options);
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sparse_coo_tensor(Tensor indices, Tensor values, int[] size, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sparse_coo_tensor");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 3, 6))).toScalarType())
                  .layout((std::move(peek(stack, 4, 6))).toLayout())
                  .device((std::move(peek(stack, 5, 6))).toDevice());
          auto result_ = torch::sparse_coo_tensor((std::move(peek(stack, 0, 6))).toTensor(),
          (std::move(peek(stack, 1, 6))).toTensor(),
          (std::move(peek(stack, 2, 6))).toIntList()->elements(),
          options);
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sparse_resize_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sparse_resize_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = (self).sparse_resize_(
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
      "aten::split_with_sizes(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("split_with_sizes");
      
          auto result_ = at::split_with_sizes(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toIntList()->elements(),
              (std::move(peek(stack, 2, 3))).toInt()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sqrt(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sqrt");
      
          auto result_ = at::sqrt(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::squeeze(Tensor(a) self) -> Tensor(a)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("squeeze");
      
          auto result_ = at::squeeze(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::squeeze(Tensor(a) self, int dim) -> Tensor(a)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("squeeze");
      
          auto result_ = at::squeeze(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sspaddmm");
      
          auto result_ = at::sspaddmm(
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
      "aten::std(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("std_out");
          auto result = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::std_out(
              result,
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toIntList()->elements(),
              (std::move(peek(stack, 2, 5))).toBool(),
              (std::move(peek(stack, 3, 5))).toBool()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::stft(Tensor self, int n_fft, int hop_length, int win_length, Tensor? window=None, bool normalized=False, bool onesided=True) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("stft");
      
          auto result_ = at::stft(
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toInt(),
              (std::move(peek(stack, 2, 7))).toInt(),
              (std::move(peek(stack, 3, 7))).toInt(),
              (std::move(peek(stack, 4, 7))).toTensor(),
              (std::move(peek(stack, 5, 7))).toBool(),
              (std::move(peek(stack, 6, 7))).toBool()
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sub_(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sub_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = (self).sub_(
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sub_(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sub_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = (self).sub_(
              (std::move(peek(stack, 1, 3))).toScalar(),
              (std::move(peek(stack, 2, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sum(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sum");
      
          auto result_ = at::sum(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sum(Tensor self, *, ScalarType dtype) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sum");
      
          auto result_ = at::sum(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalarType()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sum(Tensor self, int[1] dim, bool keepdim=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sum");
      
          auto result_ = at::sum(
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
      "aten::sum(Tensor self, int[1] dim, *, ScalarType dtype) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sum");
      
          auto result_ = at::sum(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toIntList()->elements(),
              (std::move(peek(stack, 2, 3))).toScalarType()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sum(Tensor self, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sum");
      
          auto result_ = at::sum(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toIntList()->elements(),
              (std::move(peek(stack, 2, 4))).toBool(),
              (std::move(peek(stack, 3, 4))).toScalarType()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::t(Tensor(a) self) -> Tensor(a)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("t");
      
          auto result_ = at::t(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::take(Tensor self, Tensor index, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("take_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::take_out(
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
      "aten::tan(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("tan");
      
          auto result_ = at::tan(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::tanh(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("tanh_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::tanh_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::thnn_conv2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv2d_out");
          auto output = (std::move(peek(stack, 6, 7))).toTensor();
          auto result_ = at::thnn_conv2d_out(
              output,
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              (std::move(peek(stack, 2, 7))).toIntList()->elements(),
              (std::move(peek(stack, 3, 7))).toTensor(),
              (std::move(peek(stack, 4, 7))).toIntList()->elements(),
              (std::move(peek(stack, 5, 7))).toIntList()->elements()
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::thnn_conv3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv3d");
      
          auto result_ = at::thnn_conv3d(
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
      "aten::thnn_conv3d_forward(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, int[3] padding) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv3d_forward");
      
          auto result_ = at::thnn_conv3d_forward(
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
      "aten::thnn_conv_depthwise2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv_depthwise2d_forward_out");
          auto output = (std::move(peek(stack, 7, 8))).toTensor();
          auto result_ = at::thnn_conv_depthwise2d_forward_out(
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
      "aten::thnn_conv_depthwise2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv_depthwise2d_out");
          auto output = (std::move(peek(stack, 7, 8))).toTensor();
          auto result_ = at::thnn_conv_depthwise2d_out(
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
      "aten::thnn_conv_dilated2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv_dilated2d_out");
          auto output = (std::move(peek(stack, 7, 8))).toTensor();
          auto result_ = at::thnn_conv_dilated2d_out(
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
      "aten::thnn_conv_dilated3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv_dilated3d");
      
          auto result_ = at::thnn_conv_dilated3d(
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
      "aten::thnn_conv_dilated3d_forward(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, int[3] padding, int[3] dilation) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv_dilated3d_forward");
      
          auto result_ = at::thnn_conv_dilated3d_forward(
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
      "aten::thnn_conv_transpose2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int[2] dilation=1, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv_transpose2d_out");
          auto output = (std::move(peek(stack, 8, 9))).toTensor();
          auto result_ = at::thnn_conv_transpose2d_out(
              output,
              (std::move(peek(stack, 0, 9))).toTensor(),
              (std::move(peek(stack, 1, 9))).toTensor(),
              (std::move(peek(stack, 2, 9))).toIntList()->elements(),
              (std::move(peek(stack, 3, 9))).toTensor(),
              (std::move(peek(stack, 4, 9))).toIntList()->elements(),
              (std::move(peek(stack, 5, 9))).toIntList()->elements(),
              (std::move(peek(stack, 6, 9))).toIntList()->elements(),
              (std::move(peek(stack, 7, 9))).toIntList()->elements()
          );
          drop(stack, 9);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::thnn_conv_transpose3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int[3] dilation=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv_transpose3d");
      
          auto result_ = at::thnn_conv_transpose3d(
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
      "aten::thnn_conv_transpose3d_forward(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, int[3] padding, int[3] output_padding, int[3] dilation) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv_transpose3d_forward");
      
          auto result_ = at::thnn_conv_transpose3d_forward(
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
      "aten::threshold(Tensor self, Scalar threshold, Scalar value, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("threshold_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::threshold_out(
              result,
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toScalar(),
              (std::move(peek(stack, 2, 4))).toScalar()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::to(Tensor self, Tensor other, bool non_blocking=False, bool copy=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("to");
      
          auto result_ = ((std::move(peek(stack, 0, 4))).toTensor()).to(
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toBool(),
              (std::move(peek(stack, 3, 4))).toBool()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::to(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("to");
      
          auto result_ = ((std::move(peek(stack, 0, 4))).toTensor()).to(
              (std::move(peek(stack, 1, 4))).toScalarType(),
              (std::move(peek(stack, 2, 4))).toBool(),
              (std::move(peek(stack, 3, 4))).toBool()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::to(Tensor self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("to");
      
          auto result_ = ((std::move(peek(stack, 0, 5))).toTensor()).to(
              (std::move(peek(stack, 1, 5))).toDevice(),
              (std::move(peek(stack, 2, 5))).toScalarType(),
              (std::move(peek(stack, 3, 5))).toBool(),
              (std::move(peek(stack, 4, 5))).toBool()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::to(Tensor self, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\", bool non_blocking=False, bool copy=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("to");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 1, 6))).toScalarType())
                  .layout((std::move(peek(stack, 2, 6))).toLayout())
                  .device((std::move(peek(stack, 3, 6))).toDevice());
          auto result_ = ((std::move(peek(stack, 0, 6))).toTensor()).to(options,
          (std::move(peek(stack, 4, 6))).toBool(),
          (std::move(peek(stack, 5, 6))).toBool());
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("topk");
      
          auto result_ = at::topk(
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
      "aten::trace(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("trace");
      
          auto result_ = at::trace(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::tril(Tensor self, int diagonal=0) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("tril");
      
          auto result_ = at::tril(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::triu(Tensor self, int diagonal=0) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("triu");
      
          auto result_ = at::triu(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::trtrs(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("trtrs");
      
          auto result_ = at::trtrs(
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
      "aten::trunc(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("trunc_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::trunc_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator generator=None) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("uniform_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = (self).uniform_(
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
      "aten::upsample_bicubic2d(Tensor self, int[2] output_size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_bicubic2d_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::upsample_bicubic2d_out(
              output,
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
      "aten::upsample_bilinear2d(Tensor self, int[2] output_size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_bilinear2d_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::upsample_bilinear2d_out(
              output,
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
      "aten::upsample_linear1d(Tensor self, int[1] output_size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_linear1d_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::upsample_linear1d_out(
              output,
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
      "aten::upsample_nearest1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_nearest1d_backward");
      
          auto result_ = at::upsample_nearest1d_backward(
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
      "aten::upsample_nearest2d(Tensor self, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_nearest2d_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::upsample_nearest2d_out(
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
      "aten::upsample_nearest3d(Tensor self, int[3] output_size) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_nearest3d");
      
          auto result_ = at::upsample_nearest3d(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::upsample_nearest3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_nearest3d_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::upsample_nearest3d_backward_out(
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
      "aten::upsample_trilinear3d(Tensor self, int[3] output_size, bool align_corners) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_trilinear3d");
      
          auto result_ = at::upsample_trilinear3d(
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
      "aten::upsample_trilinear3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_trilinear3d_backward_out");
          auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::upsample_trilinear3d_backward_out(
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
      "aten::values(Tensor(a) self) -> Tensor(a)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("values");
      
          auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).values(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::var(Tensor self, bool unbiased=True) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("var");
      
          auto result_ = at::var(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::var(Tensor self, int dim, bool unbiased=True, bool keepdim=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("var");
      
          auto result_ = at::var(
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
      "aten::view(Tensor(a) self, int[] size) -> Tensor(a)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("view");
      
          auto result_ = ((std::move(peek(stack, 0, 2))).toTensor()).view(
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::where(Tensor condition, Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("where");
      
          auto result_ = at::where(
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
      "aten::zeros(int[] size, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("zeros");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 1, 4))).toScalarType())
                  .layout((std::move(peek(stack, 2, 4))).toLayout())
                  .device((std::move(peek(stack, 3, 4))).toDevice());
          auto result_ = torch::zeros((std::move(peek(stack, 0, 4))).toIntList()->elements(),
          options);
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::zeros_like(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("zeros_like");
      
          auto result_ = at::zeros_like(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::zeros_like(Tensor self, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("zeros_like");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 1, 4))).toScalarType())
                  .layout((std::move(peek(stack, 2, 4))).toLayout())
                  .device((std::move(peek(stack, 3, 4))).toDevice());
          auto result_ = torch::zeros_like((std::move(peek(stack, 0, 4))).toTensor(),
          options);
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
});

} // anon namespace


}} // namespace torch::jit
