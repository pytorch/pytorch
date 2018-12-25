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

// @generated from tools/autograd/templates/register_aten_ops_0.cpp

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
      "aten::RoiPooling2d_forward(Tensor input, Tensor rois, int pooledHeight, int pooledWidth, float spatialScale) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("RoiPooling2d_forward");
      
          auto result_ = at::RoiPooling2d_forward(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toInt(),
              (std::move(peek(stack, 3, 5))).toInt(),
              (std::move(peek(stack, 4, 5))).toDouble()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::__and__(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("__and__");
      
          auto result_ = at::__and__(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::__and__(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("__and__");
      
          auto result_ = at::__and__(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::__iand__(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("__iand__");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).__iand__(
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::__iand__(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("__iand__");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).__iand__(
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_argmax(Tensor self, int dim, bool keepdim=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_argmax");
      
          auto result_ = at::_argmax(
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
      "aten::_baddbmm_mkl_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_baddbmm_mkl_");
          auto self = (std::move(peek(stack, 0, 5))).toTensor();
          auto result_ = at::_baddbmm_mkl_(
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
      "aten::_cast_Char(Tensor self, bool non_blocking=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_cast_Char");
      
          auto result_ = at::_cast_Char(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_cast_Float(Tensor self, bool non_blocking=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_cast_Float");
      
          auto result_ = at::_cast_Float(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_cast_Half(Tensor self, bool non_blocking=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_cast_Half");
      
          auto result_ = at::_cast_Half(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_cast_Long(Tensor self, bool non_blocking=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_cast_Long");
      
          auto result_ = at::_cast_Long(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_convolution");
      
          auto result_ = at::_convolution(
              (std::move(peek(stack, 0, 12))).toTensor(),
              (std::move(peek(stack, 1, 12))).toTensor(),
              (std::move(peek(stack, 2, 12))).toTensor(),
              (std::move(peek(stack, 3, 12))).toIntList()->elements(),
              (std::move(peek(stack, 4, 12))).toIntList()->elements(),
              (std::move(peek(stack, 5, 12))).toIntList()->elements(),
              (std::move(peek(stack, 6, 12))).toBool(),
              (std::move(peek(stack, 7, 12))).toIntList()->elements(),
              (std::move(peek(stack, 8, 12))).toInt(),
              (std::move(peek(stack, 9, 12))).toBool(),
              (std::move(peek(stack, 10, 12))).toBool(),
              (std::move(peek(stack, 11, 12))).toBool()
          );
          drop(stack, 12);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_convolution_double_backward(Tensor? ggI, Tensor? ggW, Tensor? ggb, Tensor gO, Tensor weight, Tensor self, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_convolution_double_backward");
      
          auto result_ = at::_convolution_double_backward(
              (std::move(peek(stack, 0, 16))).toTensor(),
              (std::move(peek(stack, 1, 16))).toTensor(),
              (std::move(peek(stack, 2, 16))).toTensor(),
              (std::move(peek(stack, 3, 16))).toTensor(),
              (std::move(peek(stack, 4, 16))).toTensor(),
              (std::move(peek(stack, 5, 16))).toTensor(),
              (std::move(peek(stack, 6, 16))).toIntList()->elements(),
              (std::move(peek(stack, 7, 16))).toIntList()->elements(),
              (std::move(peek(stack, 8, 16))).toIntList()->elements(),
              (std::move(peek(stack, 9, 16))).toBool(),
              (std::move(peek(stack, 10, 16))).toIntList()->elements(),
              (std::move(peek(stack, 11, 16))).toInt(),
              (std::move(peek(stack, 12, 16))).toBool(),
              (std::move(peek(stack, 13, 16))).toBool(),
              (std::move(peek(stack, 14, 16))).toBool(),
              as_bool_array<3>((std::move(peek(stack, 15, 16))).toIntList()->elements())
          );
          drop(stack, 16);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_convolution_nogroup(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_convolution_nogroup");
      
          auto result_ = at::_convolution_nogroup(
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toTensor(),
              (std::move(peek(stack, 3, 8))).toIntList()->elements(),
              (std::move(peek(stack, 4, 8))).toIntList()->elements(),
              (std::move(peek(stack, 5, 8))).toIntList()->elements(),
              (std::move(peek(stack, 6, 8))).toBool(),
              (std::move(peek(stack, 7, 8))).toIntList()->elements()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_cudnn_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank, bool deterministic) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_cudnn_ctc_loss");
      
          auto result_ = at::_cudnn_ctc_loss(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toIntList()->elements(),
              (std::move(peek(stack, 3, 6))).toIntList()->elements(),
              (std::move(peek(stack, 4, 6))).toInt(),
              (std::move(peek(stack, 5, 6))).toBool()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_cudnn_rnn_backward(Tensor input, Tensor[] weight, int weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state, Tensor reserve, bool[4] output_mask) -> (Tensor, Tensor, Tensor, Tensor[])",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_cudnn_rnn_backward");
      
          auto result_ = at::_cudnn_rnn_backward(
              (std::move(peek(stack, 0, 21))).toTensor(),
              (std::move(peek(stack, 1, 21))).toTensorList()->elements(),
              (std::move(peek(stack, 2, 21))).toInt(),
              (std::move(peek(stack, 3, 21))).toTensor(),
              (std::move(peek(stack, 4, 21))).toTensor(),
              (std::move(peek(stack, 5, 21))).toTensor(),
              (std::move(peek(stack, 6, 21))).toTensor(),
              (std::move(peek(stack, 7, 21))).toTensor(),
              (std::move(peek(stack, 8, 21))).toTensor(),
              (std::move(peek(stack, 9, 21))).toTensor(),
              (std::move(peek(stack, 10, 21))).toInt(),
              (std::move(peek(stack, 11, 21))).toInt(),
              (std::move(peek(stack, 12, 21))).toInt(),
              (std::move(peek(stack, 13, 21))).toBool(),
              (std::move(peek(stack, 14, 21))).toDouble(),
              (std::move(peek(stack, 15, 21))).toBool(),
              (std::move(peek(stack, 16, 21))).toBool(),
              (std::move(peek(stack, 17, 21))).toIntList()->elements(),
              (std::move(peek(stack, 18, 21))).toTensor(),
              (std::move(peek(stack, 19, 21))).toTensor(),
              as_bool_array<4>((std::move(peek(stack, 20, 21))).toIntList()->elements())
          );
          drop(stack, 21);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_dimV(Tensor self) -> int",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_dimV");
      
          auto result_ = ((std::move(peek(stack, 0, 1))).toTensor())._dimV(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_dirichlet_grad(Tensor x, Tensor alpha, Tensor total, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_dirichlet_grad_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_dirichlet_grad_out(
              output,
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
      "aten::_embedding_bag_sparse_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, int num_weights, bool scale_grad_by_freq, int mode) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_embedding_bag_sparse_backward");
      
          auto result_ = at::_embedding_bag_sparse_backward(
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toTensor(),
              (std::move(peek(stack, 3, 8))).toTensor(),
              (std::move(peek(stack, 4, 8))).toTensor(),
              (std::move(peek(stack, 5, 8))).toInt(),
              (std::move(peek(stack, 6, 8))).toBool(),
              (std::move(peek(stack, 7, 8))).toInt()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_fused_dropout(Tensor self, float p, Generator generator=None) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_fused_dropout");
      
          auto result_ = at::_fused_dropout(
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
      "aten::_local_scalar_dense(Tensor self) -> Scalar",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_local_scalar_dense");
      
          auto result_ = at::_local_scalar_dense(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_pack_padded_sequence_backward(Tensor grad, int[] input_size, Tensor batch_sizes, bool batch_first) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_pack_padded_sequence_backward");
      
          auto result_ = at::_pack_padded_sequence_backward(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toIntList()->elements(),
              (std::move(peek(stack, 2, 4))).toTensor(),
              (std::move(peek(stack, 3, 4))).toBool()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_pad_packed_sequence(Tensor data, Tensor batch_sizes, bool batch_first, Scalar padding_value, int total_length) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_pad_packed_sequence");
      
          auto result_ = at::_pad_packed_sequence(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toBool(),
              (std::move(peek(stack, 3, 5))).toScalar(),
              (std::move(peek(stack, 4, 5))).toInt()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_pdist_backward(Tensor grad, Tensor self, float p, Tensor pdist) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_pdist_backward");
      
          auto result_ = at::_pdist_backward(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toDouble(),
              (std::move(peek(stack, 3, 4))).toTensor()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_potrs_helper(Tensor self, Tensor A, bool upper) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_potrs_helper");
      
          auto result_ = at::_potrs_helper(
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
      "aten::_sparse_addmm(Tensor self, Tensor sparse, Tensor dense, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_sparse_addmm");
      
          auto result_ = at::_sparse_addmm(
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
      "aten::_sparse_coo_tensor_unsafe(Tensor indices, Tensor values, int[] size, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_sparse_coo_tensor_unsafe");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 3, 6))).toScalarType())
                  .layout((std::move(peek(stack, 4, 6))).toLayout())
                  .device((std::move(peek(stack, 5, 6))).toDevice());
          auto result_ = torch::_sparse_coo_tensor_unsafe((std::move(peek(stack, 0, 6))).toTensor(),
          (std::move(peek(stack, 1, 6))).toTensor(),
          (std::move(peek(stack, 2, 6))).toIntList()->elements(),
          options);
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_sparse_coo_tensor_with_dims_and_tensors(int sparse_dim, int dense_dim, int[] size, Tensor indices, Tensor values, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_sparse_coo_tensor_with_dims_and_tensors");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 5, 8))).toScalarType())
                  .layout((std::move(peek(stack, 6, 8))).toLayout())
                  .device((std::move(peek(stack, 7, 8))).toDevice());
          auto result_ = torch::_sparse_coo_tensor_with_dims_and_tensors((std::move(peek(stack, 0, 8))).toInt(),
          (std::move(peek(stack, 1, 8))).toInt(),
          (std::move(peek(stack, 2, 8))).toIntList()->elements(),
          (std::move(peek(stack, 3, 8))).toTensor(),
          (std::move(peek(stack, 4, 8))).toTensor(),
          options);
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_sparse_div_scalar(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_sparse_div_scalar_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_sparse_div_scalar_out(
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
      "aten::_sparse_div_zerodim(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_sparse_div_zerodim_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_sparse_div_zerodim_out(
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
      "aten::_sparse_mm(Tensor sparse, Tensor dense) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_sparse_mm");
      
          auto result_ = at::_sparse_mm(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_sparse_mul(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_sparse_mul_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_sparse_mul_out(
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
      "aten::_sparse_sum(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_sparse_sum");
      
          auto result_ = at::_sparse_sum(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_sparse_sum(Tensor self, *, ScalarType dtype) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_sparse_sum");
      
          auto result_ = at::_sparse_sum(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalarType()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_sparse_sum(Tensor self, int[1] dim) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_sparse_sum");
      
          auto result_ = at::_sparse_sum(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_sparse_sum(Tensor self, int[1] dim, *, ScalarType dtype) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_sparse_sum");
      
          auto result_ = at::_sparse_sum(
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
      "aten::_standard_gamma(Tensor self, Generator generator=None) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_standard_gamma");
      
          auto result_ = at::_standard_gamma(
              (std::move(peek(stack, 0, 2))).toTensor(),
              nullptr
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_abs(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_abs");
      
          auto result_ = at::_th_abs(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_addbmm");
      
          auto result_ = at::_th_addbmm(
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
      "aten::_th_addcdiv_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_addcdiv_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = at::_th_addcdiv_(
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
      "aten::_th_addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_addcmul");
      
          auto result_ = at::_th_addcmul(
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
      "aten::_th_addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_addmm_");
          auto self = (std::move(peek(stack, 0, 5))).toTensor();
          auto result_ = at::_th_addmm_(
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
      "aten::_th_addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_addmv_");
          auto self = (std::move(peek(stack, 0, 5))).toTensor();
          auto result_ = at::_th_addmv_(
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
      "aten::_th_addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_addr_out");
          auto result = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::_th_addr_out(
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
      "aten::_th_all(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_all");
      
          auto result_ = at::_th_all(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_all(Tensor self, int dim, bool keepdim=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_all");
      
          auto result_ = at::_th_all(
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
      "aten::_th_and(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_and");
      
          auto result_ = at::_th_and(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_and(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_and");
      
          auto result_ = at::_th_and(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_any(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_any");
      
          auto result_ = at::_th_any(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_any(Tensor self, int dim, bool keepdim=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_any");
      
          auto result_ = at::_th_any(
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
      "aten::_th_asin(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_asin");
      
          auto result_ = at::_th_asin(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_atan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_atan2_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_atan2_(
              self,
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_atan(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_atan_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_atan_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_baddbmm_out");
          auto result = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::_th_baddbmm_out(
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
      "aten::_th_bmm(Tensor self, Tensor mat2) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_bmm");
      
          auto result_ = at::_th_bmm(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_btrifact(Tensor self, *, bool pivot=True) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_btrifact");
      
          auto result_ = at::_th_btrifact(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_btrifact_with_info(Tensor self, *, bool pivot=True) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_btrifact_with_info");
      
          auto result_ = at::_th_btrifact_with_info(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_cat(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_cat_out");
          auto self = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_cat_out(
              self,
              (std::move(peek(stack, 0, 3))).toTensorList()->elements(),
              (std::move(peek(stack, 1, 3))).toInt()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_clamp_max(Tensor self, Scalar max) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_clamp_max");
      
          auto result_ = at::_th_clamp_max(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_clamp(Tensor self, Scalar min, Scalar max, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_clamp_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_th_clamp_out(
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
      "aten::_th_clone(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_clone");
      
          auto result_ = at::_th_clone(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_cos(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_cos");
      
          auto result_ = at::_th_cos(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_cosh(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_cosh_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_cosh_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_cumprod(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_cumprod_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_cumprod_out(
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
      "aten::_th_cumsum(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_cumsum_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_cumsum_out(
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
      "aten::_th_diag(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_diag_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_diag_out(
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
      "aten::_th_digamma(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_digamma_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_digamma_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_dirichlet_grad(Tensor x, Tensor alpha, Tensor total, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_dirichlet_grad_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_th_dirichlet_grad_out(
              output,
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
      "aten::_th_dist(Tensor self, Tensor other, Scalar p=2) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_dist");
      
          auto result_ = at::_th_dist(
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
      "aten::_th_eq(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_eq");
      
          auto result_ = at::_th_eq(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_eq(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_eq");
      
          auto result_ = at::_th_eq(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_erfinv_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_erfinv_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::_th_erfinv_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_exp(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_exp_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_exp_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_floor(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_floor_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_floor_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_fmod_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_fmod_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_fmod_(
              self,
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_fmod_(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_fmod_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_fmod_(
              self,
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_frac(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_frac");
      
          auto result_ = at::_th_frac(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_ge(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_ge_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_ge_out(
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
      "aten::_th_ge(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_ge_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_ge_out(
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
      "aten::_th_gels(Tensor self, Tensor A) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_gels");
      
          auto result_ = at::_th_gels(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_geometric_(Tensor(a!) self, float p, *, Generator generator=None) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_geometric_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = at::_th_geometric_(
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
      "aten::_th_ger(Tensor self, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_ger_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_ger_out(
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
      "aten::_th_gesv_single(Tensor self, Tensor A) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_gesv_single");
      
          auto result_ = at::_th_gesv_single(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_getri_single(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_getri_single");
      
          auto result_ = at::_th_getri_single(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_gt(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_gt_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_gt_out(
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
      "aten::_th_gt(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_gt_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_gt_out(
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
      "aten::_th_ilshift_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_ilshift_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_ilshift_(
              self,
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_ilshift_(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_ilshift_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_ilshift_(
              self,
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_irshift_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_irshift_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_irshift_(
              self,
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_irshift_(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_irshift_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_irshift_(
              self,
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_kthvalue(Tensor self, int k, int dim=-1, bool keepdim=False) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_kthvalue");
      
          auto result_ = at::_th_kthvalue(
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
      "aten::_th_le_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_le_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_le_(
              self,
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_le_(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_le_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_le_(
              self,
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_lerp(Tensor self, Tensor end, Scalar weight, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_lerp_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_th_lerp_out(
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
      "aten::_th_lgamma_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_lgamma_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::_th_lgamma_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_log(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_log");
      
          auto result_ = at::_th_log(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_log1p(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_log1p_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_log1p_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_log2(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_log2_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_log2_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_logspace(Scalar start, Scalar end, int steps, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_logspace_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_th_logspace_out(
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
      "aten::_th_lt_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_lt_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_lt_(
              self,
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_lt_(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_lt_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_lt_(
              self,
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_median(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_median");
      
          auto result_ = at::_th_median(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_median(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_median");
      
          auto result_ = at::_th_median(
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
      "aten::_th_min(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_min_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_min_out(
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
      "aten::_th_mode(Tensor self, int dim=-1, bool keepdim=False) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_mode");
      
          auto result_ = at::_th_mode(
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
      "aten::_th_multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator generator=None) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_multinomial");
      
          auto result_ = at::_th_multinomial(
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
      "aten::_th_ne(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_ne");
      
          auto result_ = at::_th_ne(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_ne(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_ne");
      
          auto result_ = at::_th_ne(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_neg_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_neg_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::_th_neg_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_norm(Tensor self, Scalar p, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_norm_out");
          auto result = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::_th_norm_out(
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
      "aten::_th_normal(Tensor mean, Tensor std, *, Generator generator=None) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_normal");
      
          auto result_ = at::_th_normal(
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
      "aten::_th_normal(float mean, Tensor std, *, Generator generator=None) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_normal");
      
          auto result_ = at::_th_normal(
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
      "aten::_th_normal(Tensor mean, float std=1, *, Generator generator=None) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_normal");
      
          auto result_ = at::_th_normal(
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
      "aten::_th_or(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_or_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_or_out(
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
      "aten::_th_or(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_or_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_or_out(
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
      "aten::_th_orgqr(Tensor self, Tensor input2, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_orgqr_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_orgqr_out(
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
      "aten::_th_ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_ormqr_out");
          auto result = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::_th_ormqr_out(
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
      "aten::_th_polygamma(int n, Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_polygamma");
      
          auto result_ = at::_th_polygamma(
              (std::move(peek(stack, 0, 2))).toInt(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_potri(Tensor self, bool upper=True, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_potri_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_potri_out(
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
      "aten::_th_potrs_single(Tensor self, Tensor input2, bool upper=True, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_potrs_single_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_th_potrs_single_out(
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
      "aten::_th_pow(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_pow_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_pow_out(
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
      "aten::_th_pow(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_pow_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_pow_out(
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
      "aten::_th_pow(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_pow_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_pow_out(
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
      "aten::_th_pstrf(Tensor self, bool upper=True, Scalar tol=-1) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_pstrf");
      
          auto result_ = at::_th_pstrf(
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
      "aten::_th_random_(Tensor(a!) self, *, Generator generator=None) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_random_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_random_(
              self,
              nullptr
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_random_(Tensor(a!) self, int to, *, Generator generator=None) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_random_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = at::_th_random_(
              self,
              (std::move(peek(stack, 1, 3))).toInt(),
              nullptr
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_random_(Tensor(a!) self, int from, int to, *, Generator generator=None) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_random_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = at::_th_random_(
              self,
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
      "aten::_th_range(Scalar start, Scalar end, Scalar step, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_range_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_th_range_out(
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
      "aten::_th_reciprocal(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_reciprocal");
      
          auto result_ = at::_th_reciprocal(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_remainder(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_remainder_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_remainder_out(
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
      "aten::_th_remainder(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_remainder_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_remainder_out(
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
      "aten::_th_renorm_(Tensor(a!) self, Scalar p, int dim, Scalar maxnorm) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_renorm_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = at::_th_renorm_(
              self,
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
      "aten::_th_resize_as_(Tensor(a!) self, Tensor the_template) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_resize_as_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_resize_as_(
              self,
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_round(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_round_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_round_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_scatter_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_scatter_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = at::_th_scatter_(
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
      "aten::_th_scatter_(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_scatter_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = at::_th_scatter_(
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
      "aten::_th_set_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_set_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::_th_set_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_set_(Tensor(a!) self, Tensor source) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_set_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_set_(
              self,
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_sigmoid(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_sigmoid");
      
          auto result_ = at::_th_sigmoid(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_sign(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_sign");
      
          auto result_ = at::_th_sign(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_sin(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_sin_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::_th_sin_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_std(Tensor self, bool unbiased=True) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_std");
      
          auto result_ = at::_th_std(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_std(Tensor self, int dim, bool unbiased=True, bool keepdim=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_std");
      
          auto result_ = at::_th_std(
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
      "aten::_th_take(Tensor self, Tensor index) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_take");
      
          auto result_ = at::_th_take(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_tanh(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_tanh");
      
          auto result_ = at::_th_tanh(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_tril_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_tril_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_tril_(
              self,
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_triu_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_triu_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_th_triu_(
              self,
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_trunc(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_trunc");
      
          auto result_ = at::_th_trunc(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_unfold(Tensor self, int dimension, int size, int step, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_unfold_out");
          auto result = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::_th_unfold_out(
              result,
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toInt(),
              (std::move(peek(stack, 2, 5))).toInt(),
              (std::move(peek(stack, 3, 5))).toInt()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_th_xor(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_xor_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_xor_out(
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
      "aten::_th_xor(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_th_xor_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_th_xor_out(
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
      "aten::_thnn_adaptive_avg_pool3d_backward(Tensor grad_output, Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_adaptive_avg_pool3d_backward_out");
          auto grad_input = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_thnn_adaptive_avg_pool3d_backward_out(
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
      "aten::_thnn_adaptive_avg_pool3d_forward(Tensor self, int[3] output_size) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_adaptive_avg_pool3d_forward");
      
          auto result_ = at::_thnn_adaptive_avg_pool3d_forward(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_adaptive_max_pool2d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_adaptive_max_pool2d_backward");
      
          auto result_ = at::_thnn_adaptive_max_pool2d_backward(
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
      "aten::_thnn_avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_avg_pool2d_backward_out");
          auto grad_input = (std::move(peek(stack, 7, 8))).toTensor();
          auto result_ = at::_thnn_avg_pool2d_backward_out(
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
      "aten::_thnn_avg_pool2d_forward(Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_avg_pool2d_forward");
      
          auto result_ = at::_thnn_avg_pool2d_forward(
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
      "aten::_thnn_avg_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_avg_pool3d_backward");
      
          auto result_ = at::_thnn_avg_pool3d_backward(
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
      "aten::_thnn_binary_cross_entropy_forward(Tensor self, Tensor target, Tensor? weight, int reduction, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_binary_cross_entropy_forward_out");
          auto output = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::_thnn_binary_cross_entropy_forward_out(
              output,
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
      "aten::_thnn_col2im_backward(Tensor grad_output, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_col2im_backward_out");
          auto grad_input = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::_thnn_col2im_backward_out(
              grad_input,
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
      "aten::_thnn_col2im_forward(Tensor self, int[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_col2im_forward");
      
          auto result_ = at::_thnn_col2im_forward(
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
      "aten::_thnn_conv2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_conv2d_forward");
      
          auto result_ = at::_thnn_conv2d_forward(
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
      "aten::_thnn_conv3d_backward(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, Tensor finput, Tensor fgrad_input, bool[3] output_mask=[True, True, True]) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_conv3d_backward");
      
          auto result_ = at::_thnn_conv3d_backward(
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
      "aten::_thnn_conv_depthwise2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_conv_depthwise2d_forward");
      
          auto result_ = at::_thnn_conv_depthwise2d_forward(
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
      "aten::_thnn_conv_dilated2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_conv_dilated2d_forward");
      
          auto result_ = at::_thnn_conv_dilated2d_forward(
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
      "aten::_thnn_conv_dilated3d_backward(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, Tensor columns, Tensor ones, bool[3] output_mask=[True, True, True]) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_conv_dilated3d_backward");
      
          auto result_ = at::_thnn_conv_dilated3d_backward(
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
      "aten::_thnn_conv_transpose2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, int[2] output_padding, int[2] dilation) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_conv_transpose2d_forward");
      
          auto result_ = at::_thnn_conv_transpose2d_forward(
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
      "aten::_thnn_conv_transpose3d_backward(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] output_padding, int[3] dilation, Tensor finput, Tensor fgrad_input, bool[3] output_mask=[True, True, True]) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_conv_transpose3d_backward");
      
          auto result_ = at::_thnn_conv_transpose3d_backward(
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
      "aten::_thnn_elu_forward(Tensor self, Scalar alpha, Scalar scale, Scalar input_scale, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_elu_forward_out");
          auto output = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::_thnn_elu_forward_out(
              output,
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toScalar(),
              (std::move(peek(stack, 2, 5))).toScalar(),
              (std::move(peek(stack, 3, 5))).toScalar()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_fractional_max_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] output_size, Tensor indices) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_fractional_max_pool2d_backward");
      
          auto result_ = at::_thnn_fractional_max_pool2d_backward(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toIntList()->elements(),
              (std::move(peek(stack, 3, 5))).toIntList()->elements(),
              (std::move(peek(stack, 4, 5))).toTensor()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_glu_backward(Tensor grad_output, Tensor self, int dim) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_glu_backward");
      
          auto result_ = at::_thnn_glu_backward(
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
      "aten::_thnn_hardtanh_forward(Tensor self, Scalar min_val, Scalar max_val, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_hardtanh_forward_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_hardtanh_forward_out(
              output,
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
      "aten::_thnn_im2col_backward(Tensor grad_output, int[2] input_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_im2col_backward_out");
          auto grad_input = (std::move(peek(stack, 6, 7))).toTensor();
          auto result_ = at::_thnn_im2col_backward_out(
              grad_input,
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
      "aten::_thnn_im2col_forward(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_im2col_forward");
      
          auto result_ = at::_thnn_im2col_forward(
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
      "aten::_thnn_l1_loss_forward(Tensor self, Tensor target, int reduction, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_l1_loss_forward_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_l1_loss_forward_out(
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
      "aten::_thnn_leaky_relu_(Tensor(a!) self, Scalar negative_slope=0.01) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_leaky_relu_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_thnn_leaky_relu_(
              self,
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_leaky_relu_backward(Tensor grad_output, Tensor self, Scalar negative_slope) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_leaky_relu_backward");
      
          auto result_ = at::_thnn_leaky_relu_backward(
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
      "aten::_thnn_leaky_relu_forward_(Tensor(a!) self, Scalar negative_slope) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_leaky_relu_forward_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::_thnn_leaky_relu_forward_(
              self,
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_log_sigmoid_backward(Tensor grad_output, Tensor self, Tensor buffer) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_log_sigmoid_backward");
      
          auto result_ = at::_thnn_log_sigmoid_backward(
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
      "aten::_thnn_max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_max_pool2d_with_indices_backward_out");
          auto grad_input = (std::move(peek(stack, 8, 9))).toTensor();
          auto result_ = at::_thnn_max_pool2d_with_indices_backward_out(
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
      "aten::_thnn_max_pool2d_with_indices_forward(Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_max_pool2d_with_indices_forward");
      
          auto result_ = at::_thnn_max_pool2d_with_indices_forward(
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
      "aten::_thnn_max_pool3d_with_indices_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_max_pool3d_with_indices_backward");
      
          auto result_ = at::_thnn_max_pool3d_with_indices_backward(
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
      "aten::_thnn_max_unpool2d_backward(Tensor grad_output, Tensor self, Tensor indices, int[2] output_size) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_max_unpool2d_backward");
      
          auto result_ = at::_thnn_max_unpool2d_backward(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toTensor(),
              (std::move(peek(stack, 3, 4))).toIntList()->elements()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_max_unpool3d_forward(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_max_unpool3d_forward_out");
          auto output = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::_thnn_max_unpool3d_forward_out(
              output,
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
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
      "aten::_thnn_mse_loss_forward(Tensor self, Tensor target, int reduction, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_mse_loss_forward_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_mse_loss_forward_out(
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
      "aten::_thnn_multi_margin_loss_forward(Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight, int reduction, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_multi_margin_loss_forward_out");
          auto output = (std::move(peek(stack, 6, 7))).toTensor();
          auto result_ = at::_thnn_multi_margin_loss_forward_out(
              output,
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              (std::move(peek(stack, 2, 7))).toScalar(),
              (std::move(peek(stack, 3, 7))).toScalar(),
              (std::move(peek(stack, 4, 7))).toTensor(),
              (std::move(peek(stack, 5, 7))).toInt()
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_multilabel_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_multilabel_margin_loss_backward");
      
          auto result_ = at::_thnn_multilabel_margin_loss_backward(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toTensor(),
              (std::move(peek(stack, 3, 5))).toInt(),
              (std::move(peek(stack, 4, 5))).toTensor()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_nll_loss2d_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_nll_loss2d_backward_out");
          auto grad_input = (std::move(peek(stack, 7, 8))).toTensor();
          auto result_ = at::_thnn_nll_loss2d_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toTensor(),
              (std::move(peek(stack, 3, 8))).toTensor(),
              (std::move(peek(stack, 4, 8))).toInt(),
              (std::move(peek(stack, 5, 8))).toInt(),
              (std::move(peek(stack, 6, 8))).toTensor()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_nll_loss2d_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_nll_loss2d_forward");
      
          auto result_ = at::_thnn_nll_loss2d_forward(
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
      "aten::_thnn_nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_nll_loss_backward_out");
          auto grad_input = (std::move(peek(stack, 7, 8))).toTensor();
          auto result_ = at::_thnn_nll_loss_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toTensor(),
              (std::move(peek(stack, 3, 8))).toTensor(),
              (std::move(peek(stack, 4, 8))).toInt(),
              (std::move(peek(stack, 5, 8))).toInt(),
              (std::move(peek(stack, 6, 8))).toTensor()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_nll_loss_forward");
      
          auto result_ = at::_thnn_nll_loss_forward(
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
      "aten::_thnn_reflection_pad1d_backward(Tensor grad_output, Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_reflection_pad1d_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_reflection_pad1d_backward_out(
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
      "aten::_thnn_reflection_pad1d_forward(Tensor self, int[2] padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_reflection_pad1d_forward");
      
          auto result_ = at::_thnn_reflection_pad1d_forward(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_reflection_pad2d_backward(Tensor grad_output, Tensor self, int[4] padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_reflection_pad2d_backward");
      
          auto result_ = at::_thnn_reflection_pad2d_backward(
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
      "aten::_thnn_replication_pad1d_backward(Tensor grad_output, Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_replication_pad1d_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_replication_pad1d_backward_out(
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
      "aten::_thnn_replication_pad1d_forward(Tensor self, int[2] padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_replication_pad1d_forward");
      
          auto result_ = at::_thnn_replication_pad1d_forward(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_replication_pad2d_backward(Tensor grad_output, Tensor self, int[4] padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_replication_pad2d_backward");
      
          auto result_ = at::_thnn_replication_pad2d_backward(
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
      "aten::_thnn_replication_pad3d_forward(Tensor self, int[6] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_replication_pad3d_forward_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_thnn_replication_pad3d_forward_out(
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
      "aten::_thnn_sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_sigmoid_backward");
      
          auto result_ = at::_thnn_sigmoid_backward(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_smooth_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_smooth_l1_loss_backward");
      
          auto result_ = at::_thnn_smooth_l1_loss_backward(
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
      "aten::_thnn_soft_margin_loss_forward(Tensor self, Tensor target, int reduction, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_soft_margin_loss_forward_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_soft_margin_loss_forward_out(
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
      "aten::_thnn_softplus_forward(Tensor self, Scalar beta, Scalar threshold, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_softplus_forward_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_softplus_forward_out(
              output,
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
      "aten::_thnn_softshrink_backward(Tensor grad_output, Tensor self, Scalar lambd) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_softshrink_backward");
      
          auto result_ = at::_thnn_softshrink_backward(
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
      "aten::_thnn_tanh_backward(Tensor grad_output, Tensor output) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_tanh_backward");
      
          auto result_ = at::_thnn_tanh_backward(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_upsample_bicubic2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_bicubic2d_backward");
      
          auto result_ = at::_thnn_upsample_bicubic2d_backward(
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
      "aten::_thnn_upsample_bilinear2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_bilinear2d_backward");
      
          auto result_ = at::_thnn_upsample_bilinear2d_backward(
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
      "aten::_thnn_upsample_linear1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size, bool align_corners) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_linear1d_backward");
      
          auto result_ = at::_thnn_upsample_linear1d_backward(
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
      "aten::_thnn_upsample_nearest1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_nearest1d_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_upsample_nearest1d_backward_out(
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
      "aten::_thnn_upsample_nearest1d_forward(Tensor self, int[1] output_size) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_nearest1d_forward");
      
          auto result_ = at::_thnn_upsample_nearest1d_forward(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_thnn_upsample_nearest2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_nearest2d_backward");
      
          auto result_ = at::_thnn_upsample_nearest2d_backward(
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
      "aten::_thnn_upsample_nearest3d_forward(Tensor self, int[3] output_size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_nearest3d_forward_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::_thnn_upsample_nearest3d_forward_out(
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
      "aten::_thnn_upsample_trilinear3d_forward(Tensor self, int[3] output_size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_thnn_upsample_trilinear3d_forward_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::_thnn_upsample_trilinear3d_forward_out(
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
      "aten::_trilinear(Tensor i1, Tensor i2, Tensor i3, int[] expand1, int[] expand2, int[] expand3, int[] sumdim, int unroll_dim=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_trilinear");
      
          auto result_ = at::_trilinear(
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toTensor(),
              (std::move(peek(stack, 3, 8))).toIntList()->elements(),
              (std::move(peek(stack, 4, 8))).toIntList()->elements(),
              (std::move(peek(stack, 5, 8))).toIntList()->elements(),
              (std::move(peek(stack, 6, 8))).toIntList()->elements(),
              (std::move(peek(stack, 7, 8))).toInt()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_unique_dim(Tensor self, int dim, bool sorted=False, bool return_inverse=False) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_unique_dim");
      
          auto result_ = at::_unique_dim(
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
      "aten::_unsafe_view(Tensor self, int[] size) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_unsafe_view");
      
          auto result_ = at::_unsafe_view(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::_weight_norm_cuda_interface_backward(Tensor grad_w, Tensor saved_v, Tensor saved_g, Tensor saved_norms, int dim) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("_weight_norm_cuda_interface_backward");
      
          auto result_ = at::_weight_norm_cuda_interface_backward(
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
      "aten::abs_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("abs_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::abs_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::acos(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("acos_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::acos_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::adaptive_avg_pool2d(Tensor self, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("adaptive_avg_pool2d_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::adaptive_avg_pool2d_out(
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
      "aten::adaptive_avg_pool3d(Tensor self, int[3] output_size) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("adaptive_avg_pool3d");
      
          auto result_ = at::adaptive_avg_pool3d(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::adaptive_avg_pool3d_backward(Tensor grad_output, Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("adaptive_avg_pool3d_backward_out");
          auto grad_input = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::adaptive_avg_pool3d_backward_out(
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
      "aten::adaptive_max_pool1d(Tensor self, int[1] output_size) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("adaptive_max_pool1d");
      
          auto result_ = at::adaptive_max_pool1d(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::adaptive_max_pool2d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("adaptive_max_pool2d_backward");
      
          auto result_ = at::adaptive_max_pool2d_backward(
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
      "aten::add(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("add");
      
          auto result_ = at::add(
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
      "aten::add(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("add");
      
          auto result_ = at::add(
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
      "aten::addbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("addbmm_");
          auto self = (std::move(peek(stack, 0, 5))).toTensor();
          auto result_ = (self).addbmm_(
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
      "aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("addcdiv_out");
          auto result = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::addcdiv_out(
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
      "aten::addcmul_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("addcmul_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = (self).addcmul_(
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
      "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("addmm_out");
          auto result = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::addmm_out(
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
      "aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("addmv_out");
          auto result = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::addmv_out(
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
      "aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("addr");
      
          auto result_ = at::addr(
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
      "aten::affine_grid_generator(Tensor theta, int[] size) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("affine_grid_generator");
      
          auto result_ = at::affine_grid_generator(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::alias(Tensor(a) self) -> Tensor(a)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("alias");
      
          auto result_ = at::alias(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::asin_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("asin_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::asin_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::atan(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("atan");
      
          auto result_ = at::atan(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::atan2(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("atan2_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::atan2_out(
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
      "aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("avg_pool2d");
      
          auto result_ = at::avg_pool2d(
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
      "aten::avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("avg_pool2d_backward_out");
          auto grad_input = (std::move(peek(stack, 7, 8))).toTensor();
          auto result_ = at::avg_pool2d_backward_out(
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
      "aten::avg_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("avg_pool3d_backward");
      
          auto result_ = at::avg_pool3d_backward(
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
      "aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("baddbmm");
      
          auto result_ = at::baddbmm(
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
      "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("batch_norm");
      
          auto result_ = at::batch_norm(
              (std::move(peek(stack, 0, 9))).toTensor(),
              (std::move(peek(stack, 1, 9))).toTensor(),
              (std::move(peek(stack, 2, 9))).toTensor(),
              (std::move(peek(stack, 3, 9))).toTensor(),
              (std::move(peek(stack, 4, 9))).toTensor(),
              (std::move(peek(stack, 5, 9))).toBool(),
              (std::move(peek(stack, 6, 9))).toDouble(),
              (std::move(peek(stack, 7, 9))).toDouble(),
              (std::move(peek(stack, 8, 9))).toBool()
          );
          drop(stack, 9);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::bernoulli(Tensor self, *, Generator generator=None) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("bernoulli");
      
          auto result_ = at::bernoulli(
              (std::move(peek(stack, 0, 2))).toTensor(),
              nullptr
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::bernoulli(Tensor self, float p, *, Generator generator=None) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("bernoulli");
      
          auto result_ = at::bernoulli(
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
      "aten::binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("binary_cross_entropy_out");
          auto output = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::binary_cross_entropy_out(
              output,
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
      "aten::binary_cross_entropy_with_logits_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, Tensor? pos_weight, int reduction) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("binary_cross_entropy_with_logits_backward");
      
          auto result_ = at::binary_cross_entropy_with_logits_backward(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toTensor(),
              (std::move(peek(stack, 3, 6))).toTensor(),
              (std::move(peek(stack, 4, 6))).toTensor(),
              (std::move(peek(stack, 5, 6))).toInt()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::blackman_window(int window_length, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("blackman_window");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 1, 4))).toScalarType())
                  .layout((std::move(peek(stack, 2, 4))).toLayout())
                  .device((std::move(peek(stack, 3, 4))).toDevice());
          auto result_ = torch::blackman_window((std::move(peek(stack, 0, 4))).toInt(),
          options);
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::blackman_window(int window_length, bool periodic, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("blackman_window");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 2, 5))).toScalarType())
                  .layout((std::move(peek(stack, 3, 5))).toLayout())
                  .device((std::move(peek(stack, 4, 5))).toDevice());
          auto result_ = torch::blackman_window((std::move(peek(stack, 0, 5))).toInt(),
          (std::move(peek(stack, 1, 5))).toBool(),
          options);
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::broadcast_tensors(Tensor[] tensors) -> Tensor[]",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("broadcast_tensors");
      
          auto result_ = at::broadcast_tensors(
              (std::move(peek(stack, 0, 1))).toTensorList()->elements()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::btrisolve(Tensor self, Tensor LU_data, Tensor LU_pivots, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("btrisolve_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::btrisolve_out(
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
      "aten::cat(Tensor[] tensors, int dim=0) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cat");
      
          auto result_ = at::cat(
              (std::move(peek(stack, 0, 2))).toTensorList()->elements(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator generator=None) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cauchy_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = (self).cauchy_(
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
      "aten::ceil(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ceil_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::ceil_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::chain_matmul(Tensor[] matrices) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("chain_matmul");
      
          auto result_ = at::chain_matmul(
              (std::move(peek(stack, 0, 1))).toTensorList()->elements()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cholesky(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cholesky_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::cholesky_out(
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
      "aten::chunk(Tensor(a) self, int chunks, int dim=0) -> Tensor(a)[]",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("chunk");
      
          auto result_ = at::chunk(
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
      "aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("clamp");
      
          auto result_ = at::clamp(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toOptional<Scalar>(),
              (std::move(peek(stack, 2, 3))).toOptional<Scalar>()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::clamp_max_(Tensor(a!) self, Scalar max) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("clamp_max_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::clamp_max_(
              self,
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::clamp_min(Tensor self, Scalar min, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("clamp_min_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::clamp_min_out(
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
      "aten::coalesce(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("coalesce");
      
          auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).coalesce(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::contiguous(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("contiguous");
      
          auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).contiguous(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("conv2d");
      
          auto result_ = at::conv2d(
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
      "aten::conv_tbc(Tensor self, Tensor weight, Tensor bias, int pad=0) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("conv_tbc");
      
          auto result_ = at::conv_tbc(
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
      "aten::conv_transpose2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int groups=1, int[2] dilation=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("conv_transpose2d");
      
          auto result_ = at::conv_transpose2d(
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
      "aten::copy_sparse_to_sparse_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("copy_sparse_to_sparse_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = at::copy_sparse_to_sparse_(
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
      "aten::cos_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cos_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::cos_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cosh(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cosh");
      
          auto result_ = at::cosh(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cross(Tensor self, Tensor other, int dim=-1, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cross_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::cross_out(
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
      "aten::ctc_loss(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank=0, int reduction=Mean) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ctc_loss");
      
          auto result_ = at::ctc_loss(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toTensor(),
              (std::move(peek(stack, 3, 6))).toTensor(),
              (std::move(peek(stack, 4, 6))).toInt(),
              (std::move(peek(stack, 5, 6))).toInt()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, int reduction=Mean) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ctc_loss");
      
          auto result_ = at::ctc_loss(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
              (std::move(peek(stack, 2, 6))).toIntList()->elements(),
              (std::move(peek(stack, 3, 6))).toIntList()->elements(),
              (std::move(peek(stack, 4, 6))).toInt(),
              (std::move(peek(stack, 5, 6))).toInt()
          );
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cudnn_affine_grid_generator_backward(Tensor grad, int N, int C, int H, int W) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cudnn_affine_grid_generator_backward");
      
          auto result_ = at::cudnn_affine_grid_generator_backward(
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
      "aten::cudnn_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cudnn_batch_norm_backward");
      
          auto result_ = at::cudnn_batch_norm_backward(
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
      "aten::cudnn_convolution_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cudnn_convolution_backward_weight");
      
          auto result_ = at::cudnn_convolution_backward_weight(
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
      "aten::cudnn_convolution_transpose_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cudnn_convolution_transpose_backward_weight");
      
          auto result_ = at::cudnn_convolution_transpose_backward_weight(
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
      "aten::cudnn_is_acceptable(Tensor self) -> bool",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cudnn_is_acceptable");
      
          auto result_ = at::cudnn_is_acceptable(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cumprod(Tensor self, int dim) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cumprod");
      
          auto result_ = at::cumprod(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cumprod(Tensor self, int dim, *, ScalarType dtype) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cumprod");
      
          auto result_ = at::cumprod(
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
      "aten::cumsum(Tensor self, int dim) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cumsum");
      
          auto result_ = at::cumsum(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::cumsum(Tensor self, int dim, *, ScalarType dtype) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("cumsum");
      
          auto result_ = at::cumsum(
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
      "aten::dense_dim(Tensor self) -> int",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("dense_dim");
      
          auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).dense_dim(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::diag(Tensor self, int diagonal=0) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("diag");
      
          auto result_ = at::diag(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::diagflat(Tensor self, int offset=0) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("diagflat");
      
          auto result_ = at::diagflat(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("diagonal");
      
          auto result_ = at::diagonal(
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
      "aten::digamma(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("digamma");
      
          auto result_ = at::digamma(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::div(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("div_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::div_out(
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
      "aten::dot(Tensor self, Tensor tensor) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("dot");
      
          auto result_ = at::dot(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("dropout_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = at::dropout_(
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
      "aten::eig(Tensor self, bool eigenvectors=False) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("eig");
      
          auto result_ = at::eig(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::einsum(str equation, Tensor[] tensors) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("einsum");
      
          auto result_ = at::einsum(
              (std::move(peek(stack, 0, 2))).toString()->string(),
              (std::move(peek(stack, 1, 2))).toTensorList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("elu_out");
          auto output = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::elu_out(
              output,
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toScalar(),
              (std::move(peek(stack, 2, 5))).toScalar(),
              (std::move(peek(stack, 3, 5))).toScalar()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("embedding");
      
          auto result_ = at::embedding(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
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
      "aten::embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False) -> (Tensor, Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("embedding_bag");
      
          auto result_ = at::embedding_bag(
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
      "aten::embedding_dense_backward(Tensor grad, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("embedding_dense_backward");
      
          auto result_ = at::embedding_dense_backward(
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
      "aten::embedding_renorm_(Tensor(a!) self, Tensor indices, float max_norm, float norm_type) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("embedding_renorm_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = at::embedding_renorm_(
              self,
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toDouble(),
              (std::move(peek(stack, 3, 4))).toDouble()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::eq_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("eq_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).eq_(
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::eq_(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("eq_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).eq_(
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::erf(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("erf_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::erf_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::erfc(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("erfc_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::erfc_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::erfinv(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("erfinv_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::erfinv_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::exp(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("exp");
      
          auto result_ = at::exp(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::expm1(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("expm1_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::expm1_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::eye(int n, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("eye_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::eye_out(
              result,
              (std::move(peek(stack, 0, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::eye(int n, int m, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("eye_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::eye_out(
              result,
              (std::move(peek(stack, 0, 3))).toInt(),
              (std::move(peek(stack, 1, 3))).toInt()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::feature_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("feature_dropout_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = at::feature_dropout_(
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
      "aten::flatten(Tensor self, int start_dim=0, int end_dim=-1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("flatten");
      
          auto result_ = at::flatten(
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
      "aten::floor(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("floor");
      
          auto result_ = at::floor(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::fmod(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("fmod_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::fmod_out(
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
      "aten::fmod(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("fmod_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::fmod_out(
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
      "aten::frac_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("frac_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = (self).frac_(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::fractional_max_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] output_size, Tensor indices) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("fractional_max_pool2d_backward");
      
          auto result_ = at::fractional_max_pool2d_backward(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toIntList()->elements(),
              (std::move(peek(stack, 3, 5))).toIntList()->elements(),
              (std::move(peek(stack, 4, 5))).toTensor()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::frobenius_norm(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("frobenius_norm");
      
          auto result_ = at::frobenius_norm(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::frobenius_norm(Tensor self, int[1] dim, bool keepdim=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("frobenius_norm");
      
          auto result_ = at::frobenius_norm(
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
      "aten::full(int[] size, Scalar fill_value, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("full");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 2, 5))).toScalarType())
                  .layout((std::move(peek(stack, 3, 5))).toLayout())
                  .device((std::move(peek(stack, 4, 5))).toDevice());
          auto result_ = torch::full((std::move(peek(stack, 0, 5))).toIntList()->elements(),
          (std::move(peek(stack, 1, 5))).toScalar(),
          options);
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::full_like(Tensor self, Scalar fill_value) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("full_like");
      
          auto result_ = at::full_like(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::full_like(Tensor self, Scalar fill_value, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("full_like");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 2, 5))).toScalarType())
                  .layout((std::move(peek(stack, 3, 5))).toLayout())
                  .device((std::move(peek(stack, 4, 5))).toDevice());
          auto result_ = torch::full_like((std::move(peek(stack, 0, 5))).toTensor(),
          (std::move(peek(stack, 1, 5))).toScalar(),
          options);
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::gather(Tensor self, int dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("gather_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::gather_out(
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
      "aten::ge(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ge");
      
          auto result_ = at::ge(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::ge(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ge");
      
          auto result_ = at::ge(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::ger(Tensor self, Tensor vec2) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ger");
      
          auto result_ = at::ger(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::glu_backward(Tensor grad_output, Tensor self, int dim) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("glu_backward");
      
          auto result_ = at::glu_backward(
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
      "aten::grid_sampler_2d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("grid_sampler_2d");
      
          auto result_ = at::grid_sampler_2d(
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
      "aten::grid_sampler_3d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("grid_sampler_3d_backward");
      
          auto result_ = at::grid_sampler_3d_backward(
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
      "aten::gt(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("gt");
      
          auto result_ = at::gt(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::gt(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("gt");
      
          auto result_ = at::gt(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("hardtanh_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::hardtanh_out(
              output,
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
      "aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("histc_out");
          auto result = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::histc_out(
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
      "aten::hspmm(Tensor mat1, Tensor mat2) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("hspmm");
      
          auto result_ = at::hspmm(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("index_add_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = (self).index_add_(
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
      "aten::index_fill_(Tensor(a!) self, int dim, Tensor index, Tensor value) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("index_fill_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = (self).index_fill_(
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
      "aten::index_fill_(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("index_fill_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = (self).index_fill_(
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
      "aten::index_put_(Tensor(a!) self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("index_put_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = at::index_put_(
              self,
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
      "aten::index_select(Tensor self, int dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("index_select_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::index_select_out(
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
      "aten::indices(Tensor(a) self) -> Tensor(a)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("indices");
      
          auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).indices(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("instance_norm");
      
          auto result_ = at::instance_norm(
              (std::move(peek(stack, 0, 9))).toTensor(),
              (std::move(peek(stack, 1, 9))).toTensor(),
              (std::move(peek(stack, 2, 9))).toTensor(),
              (std::move(peek(stack, 3, 9))).toTensor(),
              (std::move(peek(stack, 4, 9))).toTensor(),
              (std::move(peek(stack, 5, 9))).toBool(),
              (std::move(peek(stack, 6, 9))).toDouble(),
              (std::move(peek(stack, 7, 9))).toDouble(),
              (std::move(peek(stack, 8, 9))).toBool()
          );
          drop(stack, 9);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::inverse(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("inverse_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::inverse_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::is_same_size(Tensor self, Tensor other) -> bool",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("is_same_size");
      
          auto result_ = at::is_same_size(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::is_set_to(Tensor self, Tensor tensor) -> bool",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("is_set_to");
      
          auto result_ = ((std::move(peek(stack, 0, 2))).toTensor()).is_set_to(
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::kl_div(Tensor self, Tensor target, int reduction=Mean) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("kl_div");
      
          auto result_ = at::kl_div(
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
      "aten::l1_loss(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("l1_loss_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::l1_loss_out(
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
      "aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("layer_norm");
      
          auto result_ = at::layer_norm(
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toIntList()->elements(),
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
      "aten::le(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("le_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::le_out(
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
      "aten::le(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("le_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::le_out(
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
      "aten::leaky_relu_(Tensor(a!) self, Scalar negative_slope=0.01) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("leaky_relu_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = at::leaky_relu_(
              self,
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::leaky_relu_backward(Tensor grad_output, Tensor self, Scalar negative_slope) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("leaky_relu_backward");
      
          auto result_ = at::leaky_relu_backward(
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
      "aten::lerp(Tensor self, Tensor end, Scalar weight) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("lerp");
      
          auto result_ = at::lerp(
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
      "aten::lgamma(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("lgamma_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::lgamma_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::log10(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("log10_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::log10_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::log1p(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("log1p");
      
          auto result_ = at::log1p(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::log2(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("log2");
      
          auto result_ = at::log2(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::log_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("log_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::log_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::log_normal_(Tensor(a!) self, float mean=1, float std=2, *, Generator generator=None) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("log_normal_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = (self).log_normal_(
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
      "aten::log_sigmoid_backward(Tensor grad_output, Tensor self, Tensor buffer) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("log_sigmoid_backward");
      
          auto result_ = at::log_sigmoid_backward(
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
      "aten::logdet(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("logdet");
      
          auto result_ = at::logdet(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::logspace(Scalar start, Scalar end, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("logspace");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 2, 5))).toScalarType())
                  .layout((std::move(peek(stack, 3, 5))).toLayout())
                  .device((std::move(peek(stack, 4, 5))).toDevice());
          auto result_ = torch::logspace((std::move(peek(stack, 0, 5))).toScalar(),
          (std::move(peek(stack, 1, 5))).toScalar(),
          options);
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::logspace(Scalar start, Scalar end, int steps, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("logspace");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 3, 6))).toScalarType())
                  .layout((std::move(peek(stack, 4, 6))).toLayout())
                  .device((std::move(peek(stack, 5, 6))).toDevice());
          auto result_ = torch::logspace((std::move(peek(stack, 0, 6))).toScalar(),
          (std::move(peek(stack, 1, 6))).toScalar(),
          (std::move(peek(stack, 2, 6))).toInt(),
          options);
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::logsumexp(Tensor self, int dim, bool keepdim=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("logsumexp");
      
          auto result_ = at::logsumexp(
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
      "aten::lt(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("lt_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::lt_out(
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
      "aten::lt(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("lt_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::lt_out(
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
      "aten::masked_fill_(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("masked_fill_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = (self).masked_fill_(
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toTensor()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::masked_fill_(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("masked_fill_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = (self).masked_fill_(
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toScalar()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::masked_select(Tensor self, Tensor mask, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("masked_select_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::masked_select_out(
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
      "aten::matmul(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("matmul_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::matmul_out(
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
      "aten::max(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("max_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::max_out(
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
      "aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("max_pool2d_with_indices");
      
          auto result_ = at::max_pool2d_with_indices(
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
      "aten::max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("max_pool2d_with_indices_backward_out");
          auto grad_input = (std::move(peek(stack, 8, 9))).toTensor();
          auto result_ = at::max_pool2d_with_indices_backward_out(
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
      "aten::max_pool3d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("max_pool3d");
      
          auto result_ = at::max_pool3d(
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
      "aten::max_pool3d_with_indices_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("max_pool3d_with_indices_backward");
      
          auto result_ = at::max_pool3d_with_indices_backward(
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
      "aten::max_unpool2d_backward(Tensor grad_output, Tensor self, Tensor indices, int[2] output_size) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("max_unpool2d_backward");
      
          auto result_ = at::max_unpool2d_backward(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toTensor(),
              (std::move(peek(stack, 3, 4))).toIntList()->elements()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::max_unpool3d(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("max_unpool3d_out");
          auto output = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::max_unpool3d_out(
              output,
              (std::move(peek(stack, 0, 6))).toTensor(),
              (std::move(peek(stack, 1, 6))).toTensor(),
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
      "aten::max_values(Tensor self, int dim, bool keepdim=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("max_values");
      
          auto result_ = at::max_values(
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
      "aten::mean(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mean");
      
          auto result_ = at::mean(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::mean(Tensor self, *, ScalarType dtype) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mean");
      
          auto result_ = at::mean(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalarType()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::mean(Tensor self, int[1] dim, bool keepdim=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mean");
      
          auto result_ = at::mean(
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
      "aten::mean(Tensor self, int[1] dim, *, ScalarType dtype) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mean");
      
          auto result_ = at::mean(
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
      "aten::mean(Tensor self, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mean");
      
          auto result_ = at::mean(
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
      "aten::min(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("min");
      
          auto result_ = at::min(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::min(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("min");
      
          auto result_ = at::min(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::min(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("min");
      
          auto result_ = at::min(
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
      "aten::miopen_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("miopen_convolution");
      
          auto result_ = at::miopen_convolution(
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
      "aten::miopen_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("miopen_convolution_backward_input");
      
          auto result_ = at::miopen_convolution_backward_input(
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
      "aten::miopen_convolution_transpose(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("miopen_convolution_transpose");
      
          auto result_ = at::miopen_convolution_transpose(
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
      "aten::miopen_convolution_transpose_backward_input(Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("miopen_convolution_transpose_backward_input");
      
          auto result_ = at::miopen_convolution_transpose_backward_input(
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
      "aten::mm(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mm_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::mm_out(
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
      "aten::mse_loss(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mse_loss_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::mse_loss_out(
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
      "aten::mul_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mul_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).mul_(
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::mul_(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mul_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).mul_(
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::multi_margin_loss(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("multi_margin_loss_out");
          auto output = (std::move(peek(stack, 6, 7))).toTensor();
          auto result_ = at::multi_margin_loss_out(
              output,
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              (std::move(peek(stack, 2, 7))).toScalar(),
              (std::move(peek(stack, 3, 7))).toScalar(),
              (std::move(peek(stack, 4, 7))).toTensor(),
              (std::move(peek(stack, 5, 7))).toInt()
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::multilabel_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("multilabel_margin_loss_backward");
      
          auto result_ = at::multilabel_margin_loss_backward(
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toTensor(),
              (std::move(peek(stack, 2, 5))).toTensor(),
              (std::move(peek(stack, 3, 5))).toInt(),
              (std::move(peek(stack, 4, 5))).toTensor()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::mv(Tensor self, Tensor vec, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mv_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::mv_out(
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
      "aten::mvlgamma_(Tensor(a!) self, int p) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("mvlgamma_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).mvlgamma_(
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::narrow_copy(Tensor self, int dim, int start, int length) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("narrow_copy");
      
          auto result_ = ((std::move(peek(stack, 0, 4))).toTensor()).narrow_copy(
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
      "aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("native_batch_norm_backward");
      
          auto result_ = at::native_batch_norm_backward(
              (std::move(peek(stack, 0, 10))).toTensor(),
              (std::move(peek(stack, 1, 10))).toTensor(),
              (std::move(peek(stack, 2, 10))).toTensor(),
              (std::move(peek(stack, 3, 10))).toTensor(),
              (std::move(peek(stack, 4, 10))).toTensor(),
              (std::move(peek(stack, 5, 10))).toTensor(),
              (std::move(peek(stack, 6, 10))).toTensor(),
              (std::move(peek(stack, 7, 10))).toBool(),
              (std::move(peek(stack, 8, 10))).toDouble(),
              as_bool_array<3>((std::move(peek(stack, 9, 10))).toIntList()->elements())
          );
          drop(stack, 10);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::ne_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ne_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).ne_(
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::ne_(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ne_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).ne_(
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::neg(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("neg_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::neg_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::nll_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("nll_loss");
      
          auto result_ = at::nll_loss(
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
      "aten::nll_loss2d(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("nll_loss2d");
      
          auto result_ = at::nll_loss2d(
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
      "aten::nll_loss2d_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("nll_loss2d_backward_out");
          auto grad_input = (std::move(peek(stack, 7, 8))).toTensor();
          auto result_ = at::nll_loss2d_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toTensor(),
              (std::move(peek(stack, 3, 8))).toTensor(),
              (std::move(peek(stack, 4, 8))).toInt(),
              (std::move(peek(stack, 5, 8))).toInt(),
              (std::move(peek(stack, 6, 8))).toTensor()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::nll_loss2d_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("nll_loss2d_forward");
      
          auto result_ = at::nll_loss2d_forward(
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
      "aten::nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("nll_loss_backward_out");
          auto grad_input = (std::move(peek(stack, 7, 8))).toTensor();
          auto result_ = at::nll_loss_backward_out(
              grad_input,
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toTensor(),
              (std::move(peek(stack, 3, 8))).toTensor(),
              (std::move(peek(stack, 4, 8))).toInt(),
              (std::move(peek(stack, 5, 8))).toInt(),
              (std::move(peek(stack, 6, 8))).toTensor()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("nll_loss_forward");
      
          auto result_ = at::nll_loss_forward(
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
      "aten::nonzero(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("nonzero_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::nonzero_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::norm(Tensor self, Scalar p=2) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("norm");
      
          auto result_ = at::norm(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::norm(Tensor self, Scalar p, int dim, bool keepdim=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("norm");
      
          auto result_ = at::norm(
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
      "aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator generator=None) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("normal_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = (self).normal_(
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
      "aten::ones(int[] size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ones_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::ones_out(
              result,
              (std::move(peek(stack, 0, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::orgqr(Tensor self, Tensor input2) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("orgqr");
      
          auto result_ = at::orgqr(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("ormqr");
      
          auto result_ = at::ormqr(
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
      "aten::pairwise_distance(Tensor x1, Tensor x2, float p=2, float eps=1e-06, bool keepdim=False) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("pairwise_distance");
      
          auto result_ = at::pairwise_distance(
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
      "aten::pinverse(Tensor self, float rcond=1e-15) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("pinverse");
      
          auto result_ = at::pinverse(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toDouble()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::polygamma_(Tensor(a!) self, int n) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("polygamma_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).polygamma_(
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::potri(Tensor self, bool upper=True) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("potri");
      
          auto result_ = at::potri(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toBool()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::pow(Tensor self, Tensor exponent) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("pow");
      
          auto result_ = at::pow(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::pow(Scalar self, Tensor exponent) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("pow");
      
          auto result_ = at::pow(
              (std::move(peek(stack, 0, 2))).toScalar(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::pow(Tensor self, Scalar exponent) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("pow");
      
          auto result_ = at::pow(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::prelu(Tensor self, Tensor weight) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("prelu");
      
          auto result_ = at::prelu(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::prod(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("prod_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::prod_out(
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
      "aten::prod(Tensor self, int dim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("prod_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::prod_out(
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
      "aten::prod(Tensor self, int dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("prod_out");
          auto result = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::prod_out(
              result,
              (std::move(peek(stack, 0, 5))).toTensor(),
              (std::move(peek(stack, 1, 5))).toInt(),
              (std::move(peek(stack, 2, 5))).toBool(),
              (std::move(peek(stack, 3, 5))).toScalarType()
          );
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::randint(int high, int[] size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("randint_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::randint_out(
              result,
              (std::move(peek(stack, 0, 3))).toInt(),
              (std::move(peek(stack, 1, 3))).toIntList()->elements()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::randint(int low, int high, int[] size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("randint_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::randint_out(
              result,
              (std::move(peek(stack, 0, 4))).toInt(),
              (std::move(peek(stack, 1, 4))).toInt(),
              (std::move(peek(stack, 2, 4))).toIntList()->elements()
          );
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::randn(int[] size, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("randn");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 1, 4))).toScalarType())
                  .layout((std::move(peek(stack, 2, 4))).toLayout())
                  .device((std::move(peek(stack, 3, 4))).toDevice());
          auto result_ = torch::randn((std::move(peek(stack, 0, 4))).toIntList()->elements(),
          options);
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::randn_like(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("randn_like");
      
          auto result_ = at::randn_like(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::randn_like(Tensor self, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("randn_like");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 1, 4))).toScalarType())
                  .layout((std::move(peek(stack, 2, 4))).toLayout())
                  .device((std::move(peek(stack, 3, 4))).toDevice());
          auto result_ = torch::randn_like((std::move(peek(stack, 0, 4))).toTensor(),
          options);
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::randperm(int n, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("randperm_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::randperm_out(
              result,
              (std::move(peek(stack, 0, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::range(Scalar start, Scalar end, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("range");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 2, 5))).toScalarType())
                  .layout((std::move(peek(stack, 3, 5))).toLayout())
                  .device((std::move(peek(stack, 4, 5))).toDevice());
          auto result_ = torch::range((std::move(peek(stack, 0, 5))).toScalar(),
          (std::move(peek(stack, 1, 5))).toScalar(),
          options);
          drop(stack, 5);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::range(Scalar start, Scalar end, Scalar step, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("range");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 3, 6))).toScalarType())
                  .layout((std::move(peek(stack, 4, 6))).toLayout())
                  .device((std::move(peek(stack, 5, 6))).toDevice());
          auto result_ = torch::range((std::move(peek(stack, 0, 6))).toScalar(),
          (std::move(peek(stack, 1, 6))).toScalar(),
          (std::move(peek(stack, 2, 6))).toScalar(),
          options);
          drop(stack, 6);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::reciprocal_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("reciprocal_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = (self).reciprocal_(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::reflection_pad1d(Tensor self, int[2] padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("reflection_pad1d");
      
          auto result_ = at::reflection_pad1d(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::reflection_pad1d_backward(Tensor grad_output, Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("reflection_pad1d_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::reflection_pad1d_backward_out(
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
      "aten::reflection_pad2d_backward(Tensor grad_output, Tensor self, int[4] padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("reflection_pad2d_backward");
      
          auto result_ = at::reflection_pad2d_backward(
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
      "aten::remainder(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("remainder");
      
          auto result_ = at::remainder(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::remainder(Tensor self, Scalar other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("remainder");
      
          auto result_ = at::remainder(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toScalar()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("renorm_out");
          auto result = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::renorm_out(
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
      "aten::replication_pad1d(Tensor self, int[2] padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("replication_pad1d");
      
          auto result_ = at::replication_pad1d(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::replication_pad1d_backward(Tensor grad_output, Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("replication_pad1d_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::replication_pad1d_backward_out(
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
      "aten::replication_pad2d_backward(Tensor grad_output, Tensor self, int[4] padding) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("replication_pad2d_backward");
      
          auto result_ = at::replication_pad2d_backward(
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
      "aten::replication_pad3d(Tensor self, int[6] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("replication_pad3d_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::replication_pad3d_out(
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
      "aten::reshape(Tensor self, int[] shape) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("reshape");
      
          auto result_ = at::reshape(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::resize_(Tensor(a!) self, int[] size) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("resize_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).resize_(
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("rnn_relu_cell");
      
          auto result_ = at::rnn_relu_cell(
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
      "aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("roll");
      
          auto result_ = at::roll(
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
      "aten::round(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("round");
      
          auto result_ = at::round(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::rrelu_with_noise(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator generator=None, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("rrelu_with_noise_out");
          auto output = (std::move(peek(stack, 6, 7))).toTensor();
          auto result_ = at::rrelu_with_noise_out(
              output,
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              (std::move(peek(stack, 2, 7))).toScalar(),
              (std::move(peek(stack, 3, 7))).toScalar(),
              (std::move(peek(stack, 4, 7))).toBool(),
              nullptr
          );
          drop(stack, 7);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::rsqrt(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("rsqrt_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::rsqrt_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::rsub(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("rsub");
      
          auto result_ = at::rsub(
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
      "aten::rsub(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("rsub");
      
          auto result_ = at::rsub(
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
      "aten::s_native_addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("s_native_addmm");
      
          auto result_ = at::s_native_addmm(
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
      "aten::scalar_tensor(Scalar s, *, ScalarType dtype=float, Layout layout=strided, Device device=\"cpu\") -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("scalar_tensor");
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(stack, 1, 4))).toScalarType())
                  .layout((std::move(peek(stack, 2, 4))).toLayout())
                  .device((std::move(peek(stack, 3, 4))).toDevice());
          auto result_ = torch::scalar_tensor((std::move(peek(stack, 0, 4))).toScalar(),
          options);
          drop(stack, 4);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("scatter_add_");
          auto self = (std::move(peek(stack, 0, 4))).toTensor();
          auto result_ = (self).scatter_add_(
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
      "aten::selu(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("selu");
      
          auto result_ = at::selu(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sigmoid_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::sigmoid_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sigmoid_backward");
      
          auto result_ = at::sigmoid_backward(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sign_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sign_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = (self).sign_(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sin(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sin");
      
          auto result_ = at::sin(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sinh(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sinh_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::sinh_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sizes(Tensor self) -> int",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sizes");
      
          auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).sizes(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::slice(Tensor(a) self, int dim=0, int start=0, int end=9223372036854775807, int step=1) -> Tensor(a)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("slice");
      
          auto result_ = at::slice(
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
      "aten::smm(Tensor self, Tensor mat2) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("smm");
      
          auto result_ = at::smm(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::smooth_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("smooth_l1_loss_backward");
      
          auto result_ = at::smooth_l1_loss_backward(
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
      "aten::soft_margin_loss(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("soft_margin_loss_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::soft_margin_loss_out(
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
      "aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("softplus_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::softplus_out(
              output,
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
      "aten::softshrink_backward(Tensor grad_output, Tensor self, Scalar lambd) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("softshrink_backward");
      
          auto result_ = at::softshrink_backward(
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
      "aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sort");
      
          auto result_ = at::sort(
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
      "aten::sqrt(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sqrt_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::sqrt_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sspaddmm_out");
          auto result = (std::move(peek(stack, 5, 6))).toTensor();
          auto result_ = at::sspaddmm_out(
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
      "aten::stack(Tensor[] tensors, int dim=0) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("stack");
      
          auto result_ = at::stack(
              (std::move(peek(stack, 0, 2))).toTensorList()->elements(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::stride(Tensor self, int dim) -> int",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("stride");
      
          auto result_ = at::stride(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::sub(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sub");
      
          auto result_ = at::sub(
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
      "aten::sub(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sub");
      
          auto result_ = at::sub(
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
      "aten::sum(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sum_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::sum_out(
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
      "aten::sum(Tensor self, int[1] dim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sum_out");
          auto result = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::sum_out(
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
      "aten::sum(Tensor self, int[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("sum_out");
          auto result = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::sum_out(
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
      "aten::svd(Tensor self, bool some=True, bool compute_uv=True) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("svd");
      
          auto result_ = at::svd(
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
      "aten::symeig(Tensor self, bool eigenvectors=False, bool upper=True) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("symeig");
      
          auto result_ = at::symeig(
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
      "aten::tan(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("tan_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::tan_out(
              result,
              (std::move(peek(stack, 0, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::tanh_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("tanh_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::tanh_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::tanh_backward(Tensor grad_output, Tensor output) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("tanh_backward");
      
          auto result_ = at::tanh_backward(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::thnn_col2im_backward(Tensor grad_output, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_col2im_backward");
      
          auto result_ = at::thnn_col2im_backward(
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
      "aten::thnn_conv2d_backward(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, Tensor finput, Tensor fgrad_input, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv2d_backward");
      
          auto result_ = at::thnn_conv2d_backward(
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
      "aten::thnn_conv3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv3d_out");
          auto output = (std::move(peek(stack, 6, 7))).toTensor();
          auto result_ = at::thnn_conv3d_out(
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
      "aten::thnn_conv_depthwise2d_backward(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool[2] output_mask) -> (Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv_depthwise2d_backward");
      
          auto result_ = at::thnn_conv_depthwise2d_backward(
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
      "aten::thnn_conv_dilated2d_backward(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, Tensor columns, Tensor ones, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv_dilated2d_backward");
      
          auto result_ = at::thnn_conv_dilated2d_backward(
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
      "aten::thnn_conv_dilated3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv_dilated3d_out");
          auto output = (std::move(peek(stack, 7, 8))).toTensor();
          auto result_ = at::thnn_conv_dilated3d_out(
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
      "aten::thnn_conv_transpose2d_backward(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] output_padding, int[2] dilation, Tensor columns, Tensor ones, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv_transpose2d_backward");
      
          auto result_ = at::thnn_conv_transpose2d_backward(
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
      "aten::thnn_conv_transpose3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int[3] dilation=1, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_conv_transpose3d_out");
          auto output = (std::move(peek(stack, 8, 9))).toTensor();
          auto result_ = at::thnn_conv_transpose3d_out(
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
      "aten::thnn_im2col_backward(Tensor grad_output, int[2] input_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("thnn_im2col_backward");
      
          auto result_ = at::thnn_im2col_backward(
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
      "aten::threshold_(Tensor(a!) self, Scalar threshold, Scalar value) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("threshold_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = at::threshold_(
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
      "aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("threshold_backward");
      
          auto result_ = at::threshold_backward(
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
      "aten::to_dense(Tensor self) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("to_dense");
      
          auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).to_dense(
          
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("transpose_");
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          auto result_ = (self).transpose_(
              (std::move(peek(stack, 1, 3))).toInt(),
              (std::move(peek(stack, 2, 3))).toInt()
          );
          drop(stack, 3);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::tril(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("tril_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::tril_out(
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
      "aten::triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, float margin=1.0, float p=2, float eps=1e-06, bool swap=False, int reduction=Mean) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("triplet_margin_loss");
      
          auto result_ = at::triplet_margin_loss(
              (std::move(peek(stack, 0, 8))).toTensor(),
              (std::move(peek(stack, 1, 8))).toTensor(),
              (std::move(peek(stack, 2, 8))).toTensor(),
              (std::move(peek(stack, 3, 8))).toDouble(),
              (std::move(peek(stack, 4, 8))).toDouble(),
              (std::move(peek(stack, 5, 8))).toDouble(),
              (std::move(peek(stack, 6, 8))).toBool(),
              (std::move(peek(stack, 7, 8))).toInt()
          );
          drop(stack, 8);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::triu(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("triu_out");
          auto result = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::triu_out(
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
      "aten::trunc_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("trunc_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::trunc_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("unfold");
      
          auto result_ = ((std::move(peek(stack, 0, 4))).toTensor()).unfold(
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
      "aten::unsqueeze_(Tensor(a!) self, int dim) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("unsqueeze_");
          auto self = (std::move(peek(stack, 0, 2))).toTensor();
          auto result_ = (self).unsqueeze_(
              (std::move(peek(stack, 1, 2))).toInt()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::upsample_bicubic2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_bicubic2d_backward");
      
          auto result_ = at::upsample_bicubic2d_backward(
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
      "aten::upsample_bilinear2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_bilinear2d_backward");
      
          auto result_ = at::upsample_bilinear2d_backward(
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
      "aten::upsample_linear1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size, bool align_corners) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_linear1d_backward");
      
          auto result_ = at::upsample_linear1d_backward(
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
      "aten::upsample_nearest1d(Tensor self, int[1] output_size) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_nearest1d");
      
          auto result_ = at::upsample_nearest1d(
              (std::move(peek(stack, 0, 2))).toTensor(),
              (std::move(peek(stack, 1, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::upsample_nearest1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_nearest1d_backward_out");
          auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::upsample_nearest1d_backward_out(
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
      "aten::upsample_nearest2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_nearest2d_backward");
      
          auto result_ = at::upsample_nearest2d_backward(
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
      "aten::upsample_nearest3d(Tensor self, int[3] output_size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_nearest3d_out");
          auto output = (std::move(peek(stack, 2, 3))).toTensor();
          auto result_ = at::upsample_nearest3d_out(
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
      "aten::upsample_trilinear3d(Tensor self, int[3] output_size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("upsample_trilinear3d_out");
          auto output = (std::move(peek(stack, 3, 4))).toTensor();
          auto result_ = at::upsample_trilinear3d_out(
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
      "aten::var(Tensor self, int dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("var_out");
          auto result = (std::move(peek(stack, 4, 5))).toTensor();
          auto result_ = at::var_out(
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
      "aten::view_as(Tensor self, Tensor other) -> Tensor",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("view_as");
      
          auto result_ = ((std::move(peek(stack, 0, 2))).toTensor()).view_as(
              (std::move(peek(stack, 1, 2))).toTensor()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::zero_(Tensor(a!) self) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("zero_");
          auto self = (std::move(peek(stack, 0, 1))).toTensor();
          auto result_ = at::zero_(
              self
          );
          drop(stack, 1);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
  Operator(
      "aten::zeros(int[] size, *, Tensor(a!) out) -> Tensor(a!)",
      [](Stack & stack) {
          autograd::profiler::RecordFunction record("zeros_out");
          auto result = (std::move(peek(stack, 1, 2))).toTensor();
          auto result_ = at::zeros_out(
              result,
              (std::move(peek(stack, 0, 2))).toIntList()->elements()
          );
          drop(stack, 2);
          pack(stack, std::move(result_));
          return 0;
      }
  ),
});

} // anon namespace


}} // namespace torch::jit
