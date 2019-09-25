#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/custom_operator.h"

#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/autograd/generated/variable_factories.h"

#include <ATen/ATen.h>
#include <ATen/core/functional.h>
#include <ATen/core/interned_strings.h>

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

// @generated from tools/jit/templates/register_aten_ops.cpp

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
using at::ScalarType;
using at::Tensor;
using at::TensorOptions;
using at::DeviceGuard;
using at::MemoryFormat;

using ::c10::fmap;
using ::c10::filter;

namespace {

// TODO: remove the toOptionalTensor and toListOfOptionalTensor
// when we remove the undefined tensor semantic from TH

// XXX: This function is to specialize IValue for tensor type in
// interpreter, it should only be used in this file
at::Tensor toOptionalTensor(const IValue& v) {
  if (v.isNone()) {
    return at::Tensor();
  }
  return v.toTensor();
}

// XXX: This function is to specialize IValue for list of optional
// tensor type in interpreter, it should only be used in this file
std::vector<Tensor> toListOfOptionalTensor(const IValue& v) {
  // v is a list of optional tensor, loop over as generic list
  auto vlist = v.toGenericListRef();
  std::vector<Tensor> res;

  for (const IValue &v: vlist) {
    res.emplace_back(toOptionalTensor(v));
  }
  return res;
}

template<size_t N>
std::array<bool, N> as_bool_array(const c10::List<bool>& list) {
  std::array<bool, N> res;
  AT_ASSERT(list.size() == N);
  std::copy(list.begin(), list.end(), res.begin());
  return res;
}

c10::OperatorOptions atenOperatorOptions() {
  c10::OperatorOptions result;
  result.setAliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA);
  return result;
}

RegisterOperators reg(
    {Operator(
         "aten::get_device(Tensor self) -> int",
         [](Stack& stack) {
           RECORD_FUNCTION("get_device", std::vector<c10::IValue>());
           auto result =
               at::get_device((std::move(peek(stack, 0, 1))).toTensor());
           drop(stack, 1);
           pack(stack, std::move(result));
           return 0;
         },
         atenOperatorOptions()),
     Operator(
         "aten::storage_offset(Tensor self) -> int",
         [](Stack& stack) {
           RECORD_FUNCTION("storage_offset", std::vector<c10::IValue>());
           auto result =
               ((std::move(peek(stack, 0, 1))).toTensor()).storage_offset();
           drop(stack, 1);
           pack(stack, std::move(result));
           return 0;
         },
         atenOperatorOptions()),
     Operator(
         "aten::is_contiguous(Tensor self) -> bool",
         [](Stack& stack) {
           RECORD_FUNCTION("is_contiguous", std::vector<c10::IValue>());
           auto result =
               ((std::move(peek(stack, 0, 1))).toTensor()).is_contiguous();
           drop(stack, 1);
           pack(stack, std::move(result));
           return 0;
         },
         atenOperatorOptions()),

     // Generated operators
     Operator(
         "aten::__ior__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).__ior__(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::__ior__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).__ior__(
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::__ixor__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).__ixor__(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::__ixor__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).__ixor__(
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::__or__.Tensor(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::__or__(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::__or__.Scalar(Tensor self, Scalar other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::__or__(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::__xor__.Tensor(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::__xor__(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::__xor__.Scalar(Tensor self, Scalar other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::__xor__(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_addr(
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toTensor(),
                 (std::move(peek(stack, 2, 5))).toTensor(),
                 (std::move(peek(stack, 3, 5))).toScalar(),
                 (std::move(peek(stack, 4, 5))).toScalar()
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_cast_Double(Tensor self, bool non_blocking=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_cast_Double(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toBool()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_cast_Short(Tensor self, bool non_blocking=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_cast_Short(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toBool()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_cat(Tensor[] tensors, int dim=0) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_cat(
                 (std::move(peek(stack, 0, 2))).toTensorListRef(),
                 (std::move(peek(stack, 1, 2))).toInt()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_cdist_backward(Tensor grad, Tensor x1, Tensor x2, float p, Tensor cdist) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_cdist_backward(
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toTensor(),
                 (std::move(peek(stack, 2, 5))).toTensor(),
                 (std::move(peek(stack, 3, 5))).toDouble(),
                 (std::move(peek(stack, 4, 5))).toTensor()
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_coalesced_(Tensor(a!) self, bool coalesced) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self)._coalesced_(
                 (std::move(peek(stack, 1, 2))).toBool()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, bool zero_infinity=False) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_ctc_loss(
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toTensor(),
                 (std::move(peek(stack, 2, 6))).toIntListRef(),
                 (std::move(peek(stack, 3, 6))).toIntListRef(),
                 (std::move(peek(stack, 4, 6))).toInt(),
                 (std::move(peek(stack, 5, 6))).toBool()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_cudnn_init_dropout_state(float dropout, bool train, int dropout_seed, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 3, 7))).toScalarType())
                     .layout((std::move(peek(stack, 4, 7))).toLayout())
                     .device((std::move(peek(stack, 5, 7))).toDevice())
                     .pinned_memory((std::move(peek(stack, 6, 7))).toBool());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::_cudnn_init_dropout_state((std::move(peek(stack, 0, 7))).toDouble(),
             (std::move(peek(stack, 1, 7))).toBool(),
             (std::move(peek(stack, 2, 7))).toInt(),
             options);
             #else
                 auto result_ = torch::_cudnn_init_dropout_state((std::move(peek(stack, 0, 7))).toDouble(),
             (std::move(peek(stack, 1, 7))).toBool(),
             (std::move(peek(stack, 2, 7))).toInt(),
             options);
             #endif
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_cudnn_rnn_flatten_weight(Tensor[] weight_arr, int weight_stride0, int input_size, int mode, int hidden_size, int num_layers, bool batch_first, bool bidirectional) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_cudnn_rnn_flatten_weight(
                 (std::move(peek(stack, 0, 8))).toTensorListRef(),
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_cufft_get_plan_cache_max_size(int device_index) -> int",
         [](Stack & stack) {
         
             auto result_ = at::_cufft_get_plan_cache_max_size(
                 (std::move(peek(stack, 0, 1))).toInt()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_cumprod(Tensor self, int dim) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_cumprod(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toInt()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_cumsum(Tensor self, int dim) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_cumsum(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toInt()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_debug_has_internal_overlap(Tensor self) -> int",
         [](Stack & stack) {
         
             auto result_ = at::_debug_has_internal_overlap(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_dimI(Tensor self) -> int",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor())._dimI(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_dirichlet_grad(Tensor x, Tensor alpha, Tensor total) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_dirichlet_grad(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor(),
                 (std::move(peek(stack, 2, 3))).toTensor()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None) -> (Tensor, Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_embedding_bag(
                 (std::move(peek(stack, 0, 7))).toTensor(),
                 (std::move(peek(stack, 1, 7))).toTensor(),
                 (std::move(peek(stack, 2, 7))).toTensor(),
                 (std::move(peek(stack, 3, 7))).toBool(),
                 (std::move(peek(stack, 4, 7))).toInt(),
                 (std::move(peek(stack, 5, 7))).toBool(),
                 toOptionalTensor((std::move(peek(stack, 6, 7))))
             );
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_embedding_bag_dense_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, int num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_embedding_bag_dense_backward(
                 (std::move(peek(stack, 0, 10))).toTensor(),
                 (std::move(peek(stack, 1, 10))).toTensor(),
                 (std::move(peek(stack, 2, 10))).toTensor(),
                 (std::move(peek(stack, 3, 10))).toTensor(),
                 (std::move(peek(stack, 4, 10))).toTensor(),
                 (std::move(peek(stack, 5, 10))).toTensor(),
                 (std::move(peek(stack, 6, 10))).toInt(),
                 (std::move(peek(stack, 7, 10))).toBool(),
                 (std::move(peek(stack, 8, 10))).toInt(),
                 toOptionalTensor((std::move(peek(stack, 9, 10))))
             );
             drop(stack, 10);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_embedding_bag_per_sample_weights_backward(Tensor grad, Tensor weight, Tensor indices, Tensor offsets, Tensor offset2bag, int mode) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_embedding_bag_per_sample_weights_backward(
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_gather_sparse_backward(Tensor self, int dim, Tensor index, Tensor grad) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_gather_sparse_backward(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toInt(),
                 (std::move(peek(stack, 2, 4))).toTensor(),
                 (std::move(peek(stack, 3, 4))).toTensor()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_indices(Tensor(a) self) -> Tensor(a)",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor())._indices(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_lu_solve_helper(Tensor self, Tensor LU_data, Tensor LU_pivots) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_lu_solve_helper(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor(),
                 (std::move(peek(stack, 2, 3))).toTensor()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_lu_with_info(Tensor self, bool pivot=True, bool check_errors=True) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_lu_with_info(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toBool(),
                 (std::move(peek(stack, 2, 3))).toBool()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_make_per_channel_quantized_tensor(Tensor self, Tensor scale, Tensor zero_point, int axis) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_make_per_channel_quantized_tensor(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 (std::move(peek(stack, 2, 4))).toTensor(),
                 (std::move(peek(stack, 3, 4))).toInt()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_min(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_min(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toInt(),
                 (std::move(peek(stack, 2, 3))).toBool()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_multinomial_alias_draw(Tensor J, Tensor q, int num_samples, *, Generator? generator=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_multinomial_alias_draw(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 (std::move(peek(stack, 2, 4))).toInt(),
                 nullptr
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_nnpack_available() -> bool",
         [](Stack & stack) {
         
             auto result_ = at::_nnpack_available(
             
             );
             drop(stack, 0);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_nnpack_spatial_convolution_backward(Tensor input, Tensor grad_output, Tensor weight, int[2] padding, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_nnpack_spatial_convolution_backward(
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toTensor(),
                 (std::move(peek(stack, 2, 5))).toTensor(),
                 (std::move(peek(stack, 3, 5))).toIntListRef(),
                 as_bool_array<3>((std::move(peek(stack, 4, 5))).toBoolList())
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_nnz(Tensor self) -> int",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor())._nnz(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_reshape_from_tensor(Tensor self, Tensor shape) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_reshape_from_tensor(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_sample_dirichlet(Tensor self, Generator? generator=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_sample_dirichlet(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 nullptr
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_sobol_engine_ff_(Tensor(a!) self, int n, Tensor sobolstate, int dimension, int num_generated) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 5))).toTensor();
             auto result_ = at::_sobol_engine_ff_(
                 self,
                 (std::move(peek(stack, 1, 5))).toInt(),
                 (std::move(peek(stack, 2, 5))).toTensor(),
                 (std::move(peek(stack, 3, 5))).toInt(),
                 (std::move(peek(stack, 4, 5))).toInt()
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_sparse_coo_tensor_with_dims(int sparse_dim, int dense_dim, int[] size, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 3, 7))).toScalarType())
                     .layout((std::move(peek(stack, 4, 7))).toLayout())
                     .device((std::move(peek(stack, 5, 7))).toDevice())
                     .pinned_memory((std::move(peek(stack, 6, 7))).toBool());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::_sparse_coo_tensor_with_dims((std::move(peek(stack, 0, 7))).toInt(),
             (std::move(peek(stack, 1, 7))).toInt(),
             (std::move(peek(stack, 2, 7))).toIntListRef(),
             options);
             #else
                 auto result_ = torch::_sparse_coo_tensor_with_dims((std::move(peek(stack, 0, 7))).toInt(),
             (std::move(peek(stack, 1, 7))).toInt(),
             (std::move(peek(stack, 2, 7))).toIntListRef(),
             options);
             #endif
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_sparse_sum_backward(Tensor grad, Tensor self, int[] dim) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_sparse_sum_backward(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor(),
                 (std::move(peek(stack, 2, 3))).toIntListRef()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_standard_gamma_grad(Tensor self, Tensor output) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_standard_gamma_grad(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_svd_helper(Tensor self, bool some, bool compute_uv) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_svd_helper(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toBool(),
                 (std::move(peek(stack, 2, 3))).toBool()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_symeig_helper(Tensor self, bool eigenvectors, bool upper) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_symeig_helper(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toBool(),
                 (std::move(peek(stack, 2, 3))).toBool()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_thnn_fused_gru_cell(Tensor input_gates, Tensor hidden_gates, Tensor hx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_thnn_fused_gru_cell(
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toTensor(),
                 (std::move(peek(stack, 2, 5))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 3, 5)))),
                 toOptionalTensor((std::move(peek(stack, 4, 5))))
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_thnn_fused_lstm_cell(Tensor input_gates, Tensor hidden_gates, Tensor cx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_thnn_fused_lstm_cell(
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toTensor(),
                 (std::move(peek(stack, 2, 5))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 3, 5)))),
                 toOptionalTensor((std::move(peek(stack, 4, 5))))
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_unique(Tensor self, bool sorted=True, bool return_inverse=False) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_unique(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toBool(),
                 (std::move(peek(stack, 2, 3))).toBool()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_version(Tensor self) -> int",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor())._version(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::abs_out(
                 out,
                 (std::move(peek(stack, 0, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::acos(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::acos(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::adaptive_avg_pool2d(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toIntListRef()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::adaptive_avg_pool3d_backward(Tensor grad_output, Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::adaptive_avg_pool3d_backward(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::adaptive_max_pool3d(Tensor self, int[3] output_size) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::adaptive_max_pool3d(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toIntListRef()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::adaptive_max_pool3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = (self).add_(
                 (std::move(peek(stack, 1, 3))).toTensor(),
                 (std::move(peek(stack, 2, 3))).toScalar()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = (self).add_(
                 (std::move(peek(stack, 1, 3))).toScalar(),
                 (std::move(peek(stack, 2, 3))).toScalar()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::addbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 5, 6))).toTensor();
             auto result_ = at::addbmm_out(
                 out,
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toTensor(),
                 (std::move(peek(stack, 2, 6))).toTensor(),
                 (std::move(peek(stack, 3, 6))).toScalar(),
                 (std::move(peek(stack, 4, 6))).toScalar()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::addcdiv(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 (std::move(peek(stack, 2, 4))).toTensor(),
                 (std::move(peek(stack, 3, 4))).toScalar()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::addcmul.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 4, 5))).toTensor();
             auto result_ = at::addcmul_out(
                 out,
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toTensor(),
                 (std::move(peek(stack, 2, 5))).toTensor(),
                 (std::move(peek(stack, 3, 5))).toScalar()
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::addr_(Tensor(a!) self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::affine_grid_generator_backward(Tensor grad, int[] size, bool align_corners) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::affine_grid_generator_backward(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toIntListRef(),
                 (std::move(peek(stack, 2, 3))).toBool()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::align_as(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 2))).toTensor()).align_as(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::align_tensors(Tensor[] tensors) -> Tensor[]",
         [](Stack & stack) {
         
             auto result_ = at::align_tensors(
                 (std::move(peek(stack, 0, 1))).toTensorListRef()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::all.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::all_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toInt(),
                 (std::move(peek(stack, 2, 4))).toBool()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::alpha_dropout(Tensor input, float p, bool train) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::alpha_dropout(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toDouble(),
                 (std::move(peek(stack, 2, 3))).toBool()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::any.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::any_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toInt(),
                 (std::move(peek(stack, 2, 4))).toBool()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::arange.out(Scalar end, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::arange_out(
                 out,
                 (std::move(peek(stack, 0, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::arange.start_out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::arange_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toScalar(),
                 (std::move(peek(stack, 1, 4))).toScalar(),
                 (std::move(peek(stack, 2, 4))).toScalar()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::argmin(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toOptional<int64_t>(),
                 (std::move(peek(stack, 2, 3))).toBool()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::argsort(Tensor self, int dim=-1, bool descending=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::argsort(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toInt(),
                 (std::move(peek(stack, 2, 3))).toBool()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a)",
         [](Stack & stack) {
         
             auto result_ = at::as_strided(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toIntListRef(),
                 (std::move(peek(stack, 2, 4))).toIntListRef(),
                 (std::move(peek(stack, 3, 4))).toOptional<int64_t>()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::asin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::asin_out(
                 out,
                 (std::move(peek(stack, 0, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::atan2(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::atan2(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::atan_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::atan_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::avg_pool1d(
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toIntListRef(),
                 (std::move(peek(stack, 2, 6))).toIntListRef(),
                 (std::move(peek(stack, 3, 6))).toIntListRef(),
                 (std::move(peek(stack, 4, 6))).toBool(),
                 (std::move(peek(stack, 5, 6))).toBool()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::avg_pool2d_backward(
                 (std::move(peek(stack, 0, 8))).toTensor(),
                 (std::move(peek(stack, 1, 8))).toTensor(),
                 (std::move(peek(stack, 2, 8))).toIntListRef(),
                 (std::move(peek(stack, 3, 8))).toIntListRef(),
                 (std::move(peek(stack, 4, 8))).toIntListRef(),
                 (std::move(peek(stack, 5, 8))).toBool(),
                 (std::move(peek(stack, 6, 8))).toBool(),
                 (std::move(peek(stack, 7, 8))).toOptional<int64_t>()
             );
             drop(stack, 8);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::avg_pool3d.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 7, 8))).toTensor();
             auto result_ = at::avg_pool3d_out(
                 out,
                 (std::move(peek(stack, 0, 8))).toTensor(),
                 (std::move(peek(stack, 1, 8))).toIntListRef(),
                 (std::move(peek(stack, 2, 8))).toIntListRef(),
                 (std::move(peek(stack, 3, 8))).toIntListRef(),
                 (std::move(peek(stack, 4, 8))).toBool(),
                 (std::move(peek(stack, 5, 8))).toBool(),
                 (std::move(peek(stack, 6, 8))).toOptional<int64_t>()
             );
             drop(stack, 8);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::baddbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::bartlett_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 1, 5))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 2, 5))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 3, 5))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 4, 5))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::bartlett_window((std::move(peek(stack, 0, 5))).toInt(),
             options);
             #else
                 auto result_ = torch::bartlett_window((std::move(peek(stack, 0, 5))).toInt(),
             options);
             #endif
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::bartlett_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 2, 6))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 3, 6))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 4, 6))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 5, 6))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::bartlett_window((std::move(peek(stack, 0, 6))).toInt(),
             (std::move(peek(stack, 1, 6))).toBool(),
             options);
             #else
                 auto result_ = torch::bartlett_window((std::move(peek(stack, 0, 6))).toInt(),
             (std::move(peek(stack, 1, 6))).toBool(),
             options);
             #endif
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::batch_norm_backward_elemt(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, Tensor mean_dy, Tensor mean_dy_xmu) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::batch_norm_backward_elemt(
                 (std::move(peek(stack, 0, 7))).toTensor(),
                 (std::move(peek(stack, 1, 7))).toTensor(),
                 (std::move(peek(stack, 2, 7))).toTensor(),
                 (std::move(peek(stack, 3, 7))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 4, 7)))),
                 (std::move(peek(stack, 5, 7))).toTensor(),
                 (std::move(peek(stack, 6, 7))).toTensor()
             );
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::bernoulli_.Tensor(Tensor(a!) self, Tensor p, *, Generator? generator=None) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = (self).bernoulli_(
                 (std::move(peek(stack, 1, 3))).toTensor(),
                 nullptr
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = (self).bernoulli_(
                 (std::move(peek(stack, 1, 3))).toDouble(),
                 nullptr
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::binary_cross_entropy(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 2, 4)))),
                 (std::move(peek(stack, 3, 4))).toInt()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::binary_cross_entropy_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 5, 6))).toTensor();
             auto result_ = at::binary_cross_entropy_backward_out(
                 grad_input,
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toTensor(),
                 (std::move(peek(stack, 2, 6))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 3, 6)))),
                 (std::move(peek(stack, 4, 6))).toInt()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::bincount(Tensor self, Tensor? weights=None, int minlength=0) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::bincount(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 1, 3)))),
                 (std::move(peek(stack, 2, 3))).toInt()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::bitwise_not_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = (self).bitwise_not_(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::bmm_out(
                 out,
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::ceil(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::ceil(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::celu(Tensor self, Scalar alpha=1.0) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::celu(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cholesky(Tensor self, bool upper=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::cholesky(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toBool()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cholesky_solve(Tensor self, Tensor input2, bool upper=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::cholesky_solve(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor(),
                 (std::move(peek(stack, 2, 3))).toBool()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = at::clamp_(
                 self,
                 (std::move(peek(stack, 1, 3))).toOptional<Scalar>(),
                 (std::move(peek(stack, 2, 3))).toOptional<Scalar>()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::clamp_max.out(Tensor self, Scalar max, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::clamp_max_out(
                 out,
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toScalar()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::clamp_min(Tensor self, Scalar min) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::clamp_min(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::col2im_backward(Tensor grad_output, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::col2im_backward(
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toIntListRef(),
                 (std::move(peek(stack, 2, 5))).toIntListRef(),
                 (std::move(peek(stack, 3, 5))).toIntListRef(),
                 (std::move(peek(stack, 4, 5))).toIntListRef()
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::combinations(Tensor self, int r=2, bool with_replacement=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::combinations(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toInt(),
                 (std::move(peek(stack, 2, 3))).toBool()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::constant_pad_nd(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toIntListRef(),
                 (std::move(peek(stack, 2, 3))).toScalar()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::conv1d(
                 (std::move(peek(stack, 0, 7))).toTensor(),
                 (std::move(peek(stack, 1, 7))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 2, 7)))),
                 (std::move(peek(stack, 3, 7))).toIntListRef(),
                 (std::move(peek(stack, 4, 7))).toIntListRef(),
                 (std::move(peek(stack, 5, 7))).toIntListRef(),
                 (std::move(peek(stack, 6, 7))).toInt()
             );
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::conv_tbc_backward(Tensor self, Tensor input, Tensor weight, Tensor bias, int pad) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] output_padding=0, int groups=1, int[1] dilation=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::conv_transpose1d(
                 (std::move(peek(stack, 0, 8))).toTensor(),
                 (std::move(peek(stack, 1, 8))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 2, 8)))),
                 (std::move(peek(stack, 3, 8))).toIntListRef(),
                 (std::move(peek(stack, 4, 8))).toIntListRef(),
                 (std::move(peek(stack, 5, 8))).toIntListRef(),
                 (std::move(peek(stack, 6, 8))).toInt(),
                 (std::move(peek(stack, 7, 8))).toIntListRef()
             );
             drop(stack, 8);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::convolution_overrideable(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::convolution_overrideable(
                 (std::move(peek(stack, 0, 9))).toTensor(),
                 (std::move(peek(stack, 1, 9))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 2, 9)))),
                 (std::move(peek(stack, 3, 9))).toIntListRef(),
                 (std::move(peek(stack, 4, 9))).toIntListRef(),
                 (std::move(peek(stack, 5, 9))).toIntListRef(),
                 (std::move(peek(stack, 6, 9))).toBool(),
                 (std::move(peek(stack, 7, 9))).toIntListRef(),
                 (std::move(peek(stack, 8, 9))).toInt()
             );
             drop(stack, 9);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::cos_out(
                 out,
                 (std::move(peek(stack, 0, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cosh_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::cosh_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cross(Tensor self, Tensor other, int? dim=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::cross(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor(),
                 (std::move(peek(stack, 2, 3))).toOptional<int64_t>()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cudnn_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::cudnn_convolution(
                 (std::move(peek(stack, 0, 9))).toTensor(),
                 (std::move(peek(stack, 1, 9))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 2, 9)))),
                 (std::move(peek(stack, 3, 9))).toIntListRef(),
                 (std::move(peek(stack, 4, 9))).toIntListRef(),
                 (std::move(peek(stack, 5, 9))).toIntListRef(),
                 (std::move(peek(stack, 6, 9))).toInt(),
                 (std::move(peek(stack, 7, 9))).toBool(),
                 (std::move(peek(stack, 8, 9))).toBool()
             );
             drop(stack, 9);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cudnn_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::cudnn_convolution_backward_input(
                 (std::move(peek(stack, 0, 9))).toIntListRef(),
                 (std::move(peek(stack, 1, 9))).toTensor(),
                 (std::move(peek(stack, 2, 9))).toTensor(),
                 (std::move(peek(stack, 3, 9))).toIntListRef(),
                 (std::move(peek(stack, 4, 9))).toIntListRef(),
                 (std::move(peek(stack, 5, 9))).toIntListRef(),
                 (std::move(peek(stack, 6, 9))).toInt(),
                 (std::move(peek(stack, 7, 9))).toBool(),
                 (std::move(peek(stack, 8, 9))).toBool()
             );
             drop(stack, 9);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cudnn_convolution_transpose(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::cudnn_convolution_transpose(
                 (std::move(peek(stack, 0, 10))).toTensor(),
                 (std::move(peek(stack, 1, 10))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 2, 10)))),
                 (std::move(peek(stack, 3, 10))).toIntListRef(),
                 (std::move(peek(stack, 4, 10))).toIntListRef(),
                 (std::move(peek(stack, 5, 10))).toIntListRef(),
                 (std::move(peek(stack, 6, 10))).toIntListRef(),
                 (std::move(peek(stack, 7, 10))).toInt(),
                 (std::move(peek(stack, 8, 10))).toBool(),
                 (std::move(peek(stack, 9, 10))).toBool()
             );
             drop(stack, 10);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cudnn_convolution_transpose_backward_input(Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::cudnn_convolution_transpose_backward_input(
                 (std::move(peek(stack, 0, 8))).toTensor(),
                 (std::move(peek(stack, 1, 8))).toTensor(),
                 (std::move(peek(stack, 2, 8))).toIntListRef(),
                 (std::move(peek(stack, 3, 8))).toIntListRef(),
                 (std::move(peek(stack, 4, 8))).toIntListRef(),
                 (std::move(peek(stack, 5, 8))).toInt(),
                 (std::move(peek(stack, 6, 8))).toBool(),
                 (std::move(peek(stack, 7, 8))).toBool()
             );
             drop(stack, 8);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cudnn_grid_sampler(Tensor self, Tensor grid) -> Tensor output",
         [](Stack & stack) {
         
             auto result_ = at::cudnn_grid_sampler(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::data(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).data(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::dequantize(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::dequantize(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::det(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::det(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::detach(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::detach(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::digamma_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = (self).digamma_(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::dim(Tensor self) -> int",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).dim(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::div.Tensor(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::div(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::div.Scalar(Tensor self, Scalar other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::div(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::elu(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toScalar(),
                 (std::move(peek(stack, 2, 4))).toScalar(),
                 (std::move(peek(stack, 3, 4))).toScalar()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::elu_backward.grad_input(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::embedding_backward(Tensor grad, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq, bool sparse) -> Tensor",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::empty.out(int[] size, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::empty_out(
                 out,
                 (std::move(peek(stack, 0, 3))).toIntListRef(),
                 (std::move(peek(stack, 1, 3))).toOptional<c10::MemoryFormat>()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::eq_out(
                 out,
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::eq_out(
                 out,
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toScalar()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::equal(Tensor self, Tensor other) -> bool",
         [](Stack & stack) {
         
             auto result_ = at::equal(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::erf(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::erf(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::erfc(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::erfc(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::erfinv(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::erfinv(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::exp_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::exp_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::expand_as(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 2))).toTensor()).expand_as(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::expm1(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::expm1(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::exponential_(Tensor(a!) self, float lambd=1, *, Generator? generator=None) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = (self).exponential_(
                 (std::move(peek(stack, 1, 3))).toDouble(),
                 nullptr
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::eye(int n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 1, 5))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 2, 5))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 3, 5))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 4, 5))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::eye((std::move(peek(stack, 0, 5))).toInt(),
             options);
             #else
                 auto result_ = torch::eye((std::move(peek(stack, 0, 5))).toInt(),
             options);
             #endif
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::eye.m(int n, int m, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 2, 6))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 3, 6))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 4, 6))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 5, 6))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::eye((std::move(peek(stack, 0, 6))).toInt(),
             (std::move(peek(stack, 1, 6))).toInt(),
             options);
             #else
                 auto result_ = torch::eye((std::move(peek(stack, 0, 6))).toInt(),
             (std::move(peek(stack, 1, 6))).toInt(),
             options);
             #endif
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::fake_quantize_per_tensor_affine_backward(Tensor grad, Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::fake_quantize_per_tensor_affine_backward(
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toTensor(),
                 (std::move(peek(stack, 2, 6))).toDouble(),
                 (std::move(peek(stack, 3, 6))).toInt(),
                 (std::move(peek(stack, 4, 6))).toInt(),
                 (std::move(peek(stack, 5, 6))).toInt()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::fbgemm_is_cpu_supported() -> bool",
         [](Stack & stack) {
         
             auto result_ = at::fbgemm_is_cpu_supported(
             
             );
             drop(stack, 0);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::fbgemm_linear_fp16_weight(Tensor input, Tensor packed_weight, Tensor bias) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::fbgemm_linear_fp16_weight(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor(),
                 (std::move(peek(stack, 2, 3))).toTensor()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::fbgemm_linear_quantize_weight(Tensor input) -> (Tensor, Tensor, float, int)",
         [](Stack & stack) {
         
             auto result_ = at::fbgemm_linear_quantize_weight(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::feature_alpha_dropout(Tensor input, float p, bool train) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::feature_alpha_dropout(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toDouble(),
                 (std::move(peek(stack, 2, 3))).toBool()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::fft(Tensor self, int signal_ndim, bool normalized=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::fft(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toInt(),
                 (std::move(peek(stack, 2, 3))).toBool()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = at::fill_(
                 self,
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = at::fill_(
                 self,
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::floor_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::floor_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::fmod.Tensor(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::fmod(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::fmod.Scalar(Tensor self, Scalar other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::fmod(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::frac.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::frac_out(
                 out,
                 (std::move(peek(stack, 0, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::fractional_max_pool3d(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::fractional_max_pool3d(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toIntListRef(),
                 (std::move(peek(stack, 2, 4))).toIntListRef(),
                 (std::move(peek(stack, 3, 4))).toTensor()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::fractional_max_pool3d_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] output_size, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 5, 6))).toTensor();
             auto result_ = at::fractional_max_pool3d_backward_out(
                 grad_input,
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toTensor(),
                 (std::move(peek(stack, 2, 6))).toIntListRef(),
                 (std::move(peek(stack, 3, 6))).toIntListRef(),
                 (std::move(peek(stack, 4, 6))).toTensor()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::from_file(str filename, bool? shared=None, int? size=0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 3, 7))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 4, 7))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 5, 7))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 6, 7))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::from_file((std::move(peek(stack, 0, 7))).toStringRef(),
             (std::move(peek(stack, 1, 7))).toOptional<bool>(),
             (std::move(peek(stack, 2, 7))).toOptional<int64_t>(),
             options);
             #else
                 auto result_ = torch::from_file((std::move(peek(stack, 0, 7))).toStringRef(),
             (std::move(peek(stack, 1, 7))).toOptional<bool>(),
             (std::move(peek(stack, 2, 7))).toOptional<int64_t>(),
             options);
             #endif
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::gather(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toInt(),
                 (std::move(peek(stack, 2, 4))).toTensor(),
                 (std::move(peek(stack, 3, 4))).toBool()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::ge_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).ge_(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::ge_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).ge_(
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::gelu_backward(Tensor grad, Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::gelu_backward(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::geqrf(Tensor self) -> (Tensor a, Tensor tau)",
         [](Stack & stack) {
         
             auto result_ = at::geqrf(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::glu.out(Tensor self, int dim=-1, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::glu_out(
                 out,
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toInt()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::grid_sampler_2d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::grid_sampler_2d_backward(
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toTensor(),
                 (std::move(peek(stack, 2, 6))).toTensor(),
                 (std::move(peek(stack, 3, 6))).toInt(),
                 (std::move(peek(stack, 4, 6))).toInt(),
                 (std::move(peek(stack, 5, 6))).toBool()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::gru_cell(
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toTensor(),
                 (std::move(peek(stack, 2, 6))).toTensor(),
                 (std::move(peek(stack, 3, 6))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 4, 6)))),
                 toOptionalTensor((std::move(peek(stack, 5, 6))))
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::gt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).gt_(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::gt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).gt_(
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::hamming_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 1, 5))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 2, 5))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 3, 5))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 4, 5))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::hamming_window((std::move(peek(stack, 0, 5))).toInt(),
             options);
             #else
                 auto result_ = torch::hamming_window((std::move(peek(stack, 0, 5))).toInt(),
             options);
             #endif
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::hamming_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 2, 6))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 3, 6))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 4, 6))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 5, 6))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::hamming_window((std::move(peek(stack, 0, 6))).toInt(),
             (std::move(peek(stack, 1, 6))).toBool(),
             options);
             #else
                 auto result_ = torch::hamming_window((std::move(peek(stack, 0, 6))).toInt(),
             (std::move(peek(stack, 1, 6))).toBool(),
             options);
             #endif
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::hamming_window.periodic_alpha(int window_length, bool periodic, float alpha, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 3, 7))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 4, 7))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 5, 7))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 6, 7))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::hamming_window((std::move(peek(stack, 0, 7))).toInt(),
             (std::move(peek(stack, 1, 7))).toBool(),
             (std::move(peek(stack, 2, 7))).toDouble(),
             options);
             #else
                 auto result_ = torch::hamming_window((std::move(peek(stack, 0, 7))).toInt(),
             (std::move(peek(stack, 1, 7))).toBool(),
             (std::move(peek(stack, 2, 7))).toDouble(),
             options);
             #endif
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::hamming_window.periodic_alpha_beta(int window_length, bool periodic, float alpha, float beta, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 4, 8))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 5, 8))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 6, 8))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 7, 8))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::hamming_window((std::move(peek(stack, 0, 8))).toInt(),
             (std::move(peek(stack, 1, 8))).toBool(),
             (std::move(peek(stack, 2, 8))).toDouble(),
             (std::move(peek(stack, 3, 8))).toDouble(),
             options);
             #else
                 auto result_ = torch::hamming_window((std::move(peek(stack, 0, 8))).toInt(),
             (std::move(peek(stack, 1, 8))).toBool(),
             (std::move(peek(stack, 2, 8))).toDouble(),
             (std::move(peek(stack, 3, 8))).toDouble(),
             options);
             #endif
             drop(stack, 8);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::hardshrink(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::hardtanh(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toScalar(),
                 (std::move(peek(stack, 2, 3))).toScalar()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::hardtanh_backward.grad_input(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::histc(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toInt(),
                 (std::move(peek(stack, 2, 4))).toScalar(),
                 (std::move(peek(stack, 3, 4))).toScalar()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::ifft(Tensor self, int signal_ndim, bool normalized=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::ifft(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toInt(),
                 (std::move(peek(stack, 2, 3))).toBool()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::im2col_backward(Tensor grad_output, int[2] input_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::im2col_backward(
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toIntListRef(),
                 (std::move(peek(stack, 2, 6))).toIntListRef(),
                 (std::move(peek(stack, 3, 6))).toIntListRef(),
                 (std::move(peek(stack, 4, 6))).toIntListRef(),
                 (std::move(peek(stack, 5, 6))).toIntListRef()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::index(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 toListOfOptionalTensor((std::move(peek(stack, 1, 2))))
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::index.Tensor(Tensor self, Tensor[] indices) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::index(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toTensorListRef()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 4))).toTensor();
             auto result_ = (self).index_copy_(
                 (std::move(peek(stack, 1, 4))).toInt(),
                 (std::move(peek(stack, 2, 4))).toTensor(),
                 (std::move(peek(stack, 3, 4))).toTensor()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::index_select(Tensor self, int dim, Tensor index) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::index_select(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toInt(),
                 (std::move(peek(stack, 2, 3))).toTensor()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::inverse(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::inverse(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::irfft(Tensor self, int signal_ndim, bool normalized=False, bool onesided=True, int[] signal_sizes=[]) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::irfft(
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toInt(),
                 (std::move(peek(stack, 2, 5))).toBool(),
                 (std::move(peek(stack, 3, 5))).toBool(),
                 (std::move(peek(stack, 4, 5))).toIntListRef()
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::is_distributed(Tensor self) -> bool",
         [](Stack & stack) {
         
             auto result_ = at::is_distributed(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::is_nonzero(Tensor self) -> bool",
         [](Stack & stack) {
         
             auto result_ = at::is_nonzero(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::is_pinned(Tensor self) -> bool",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).is_pinned(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::item(Tensor self) -> Scalar",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).item(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::kl_div_backward(Tensor grad_output, Tensor self, Tensor target, int reduction=Mean) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::kl_div_backward(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 (std::move(peek(stack, 2, 4))).toTensor(),
                 (std::move(peek(stack, 3, 4))).toInt()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::l1_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::l1_loss(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor(),
                 (std::move(peek(stack, 2, 3))).toInt()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::l1_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::le.Tensor(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::le(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::le.Scalar(Tensor self, Scalar other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::le(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::leaky_relu.out(Tensor self, Scalar negative_slope=0.01, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::leaky_relu_out(
                 out,
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toScalar()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::lerp_.Tensor(Tensor(a!) self, Tensor end, Tensor weight) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = (self).lerp_(
                 (std::move(peek(stack, 1, 3))).toTensor(),
                 (std::move(peek(stack, 2, 3))).toTensor()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::lerp_.Scalar(Tensor(a!) self, Tensor end, Scalar weight) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = (self).lerp_(
                 (std::move(peek(stack, 1, 3))).toTensor(),
                 (std::move(peek(stack, 2, 3))).toScalar()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::lgamma(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::lgamma(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::linear(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 2, 3))))
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::linspace.out(Scalar start, Scalar end, int steps=100, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::linspace_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toScalar(),
                 (std::move(peek(stack, 1, 4))).toScalar(),
                 (std::move(peek(stack, 2, 4))).toInt()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::log10(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::log10(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::log1p_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::log1p_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::log2_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::log2_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::log.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::log_out(
                 out,
                 (std::move(peek(stack, 0, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::log_sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::log_sigmoid_out(
                 out,
                 (std::move(peek(stack, 0, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::log_softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::log_softmax(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toInt(),
                 (std::move(peek(stack, 2, 3))).toOptional<ScalarType>()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::logical_not(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::logical_not(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::logical_xor.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::logical_xor_out(
                 out,
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::lstm_cell(
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toTensorListRef(),
                 (std::move(peek(stack, 2, 6))).toTensor(),
                 (std::move(peek(stack, 3, 6))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 4, 6)))),
                 toOptionalTensor((std::move(peek(stack, 5, 6))))
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::lt.Tensor(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::lt(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::lt.Scalar(Tensor self, Scalar other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::lt(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::masked_scatter_(Tensor(a!) self, Tensor mask, Tensor source) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = (self).masked_scatter_(
                 (std::move(peek(stack, 1, 3))).toTensor(),
                 (std::move(peek(stack, 2, 3))).toTensor()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::masked_select(Tensor self, Tensor mask) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::masked_select(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::matmul(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::matmul(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::matrix_power(Tensor self, int n) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::matrix_power(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toInt()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::matrix_rank(Tensor self, bool symmetric=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::matrix_rank(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toBool()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::matrix_rank.tol(Tensor self, float tol, bool symmetric=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::matrix_rank(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toDouble(),
                 (std::move(peek(stack, 2, 3))).toBool()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::max(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::max(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::max.other(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::max(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)",
         [](Stack & stack) {
         
             auto result_ = at::max(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toInt(),
                 (std::move(peek(stack, 2, 3))).toBool()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::max_pool1d_with_indices(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::max_pool1d_with_indices(
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toIntListRef(),
                 (std::move(peek(stack, 2, 6))).toIntListRef(),
                 (std::move(peek(stack, 3, 6))).toIntListRef(),
                 (std::move(peek(stack, 4, 6))).toIntListRef(),
                 (std::move(peek(stack, 5, 6))).toBool()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::max_pool2d(
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toIntListRef(),
                 (std::move(peek(stack, 2, 6))).toIntListRef(),
                 (std::move(peek(stack, 3, 6))).toIntListRef(),
                 (std::move(peek(stack, 4, 6))).toIntListRef(),
                 (std::move(peek(stack, 5, 6))).toBool()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::max_pool2d_with_indices_backward(
                 (std::move(peek(stack, 0, 8))).toTensor(),
                 (std::move(peek(stack, 1, 8))).toTensor(),
                 (std::move(peek(stack, 2, 8))).toIntListRef(),
                 (std::move(peek(stack, 3, 8))).toIntListRef(),
                 (std::move(peek(stack, 4, 8))).toIntListRef(),
                 (std::move(peek(stack, 5, 8))).toIntListRef(),
                 (std::move(peek(stack, 6, 8))).toBool(),
                 (std::move(peek(stack, 7, 8))).toTensor()
             );
             drop(stack, 8);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::max_unpool2d.out(Tensor self, Tensor indices, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::max_unpool2d_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 (std::move(peek(stack, 2, 4))).toIntListRef()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::max_unpool3d(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::max_unpool3d(
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toTensor(),
                 (std::move(peek(stack, 2, 5))).toIntListRef(),
                 (std::move(peek(stack, 3, 5))).toIntListRef(),
                 (std::move(peek(stack, 4, 5))).toIntListRef()
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::max_unpool3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 6, 7))).toTensor();
             auto result_ = at::max_unpool3d_backward_out(
                 grad_input,
                 (std::move(peek(stack, 0, 7))).toTensor(),
                 (std::move(peek(stack, 1, 7))).toTensor(),
                 (std::move(peek(stack, 2, 7))).toTensor(),
                 (std::move(peek(stack, 3, 7))).toIntListRef(),
                 (std::move(peek(stack, 4, 7))).toIntListRef(),
                 (std::move(peek(stack, 5, 7))).toIntListRef()
             );
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::meshgrid(Tensor[] tensors) -> Tensor[]",
         [](Stack & stack) {
         
             auto result_ = at::meshgrid(
                 (std::move(peek(stack, 0, 1))).toTensorListRef()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::miopen_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::miopen_batch_norm(
                 (std::move(peek(stack, 0, 8))).toTensor(),
                 (std::move(peek(stack, 1, 8))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 2, 8)))),
                 toOptionalTensor((std::move(peek(stack, 3, 8)))),
                 toOptionalTensor((std::move(peek(stack, 4, 8)))),
                 (std::move(peek(stack, 5, 8))).toBool(),
                 (std::move(peek(stack, 6, 8))).toDouble(),
                 (std::move(peek(stack, 7, 8))).toDouble()
             );
             drop(stack, 8);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::miopen_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::miopen_convolution_backward(
                 (std::move(peek(stack, 0, 10))).toTensor(),
                 (std::move(peek(stack, 1, 10))).toTensor(),
                 (std::move(peek(stack, 2, 10))).toTensor(),
                 (std::move(peek(stack, 3, 10))).toIntListRef(),
                 (std::move(peek(stack, 4, 10))).toIntListRef(),
                 (std::move(peek(stack, 5, 10))).toIntListRef(),
                 (std::move(peek(stack, 6, 10))).toInt(),
                 (std::move(peek(stack, 7, 10))).toBool(),
                 (std::move(peek(stack, 8, 10))).toBool(),
                 as_bool_array<3>((std::move(peek(stack, 9, 10))).toBoolList())
             );
             drop(stack, 10);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::miopen_convolution_backward_bias(Tensor grad_output) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::miopen_convolution_backward_bias(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::miopen_convolution_transpose_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::miopen_convolution_transpose_backward(
                 (std::move(peek(stack, 0, 11))).toTensor(),
                 (std::move(peek(stack, 1, 11))).toTensor(),
                 (std::move(peek(stack, 2, 11))).toTensor(),
                 (std::move(peek(stack, 3, 11))).toIntListRef(),
                 (std::move(peek(stack, 4, 11))).toIntListRef(),
                 (std::move(peek(stack, 5, 11))).toIntListRef(),
                 (std::move(peek(stack, 6, 11))).toIntListRef(),
                 (std::move(peek(stack, 7, 11))).toInt(),
                 (std::move(peek(stack, 8, 11))).toBool(),
                 (std::move(peek(stack, 9, 11))).toBool(),
                 as_bool_array<3>((std::move(peek(stack, 10, 11))).toBoolList())
             );
             drop(stack, 11);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::miopen_depthwise_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::miopen_depthwise_convolution_backward(
                 (std::move(peek(stack, 0, 10))).toTensor(),
                 (std::move(peek(stack, 1, 10))).toTensor(),
                 (std::move(peek(stack, 2, 10))).toTensor(),
                 (std::move(peek(stack, 3, 10))).toIntListRef(),
                 (std::move(peek(stack, 4, 10))).toIntListRef(),
                 (std::move(peek(stack, 5, 10))).toIntListRef(),
                 (std::move(peek(stack, 6, 10))).toInt(),
                 (std::move(peek(stack, 7, 10))).toBool(),
                 (std::move(peek(stack, 8, 10))).toBool(),
                 as_bool_array<3>((std::move(peek(stack, 9, 10))).toBoolList())
             );
             drop(stack, 10);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::miopen_rnn_backward(Tensor input, Tensor[] weight, int weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state, Tensor reserve, bool[4] output_mask) -> (Tensor, Tensor, Tensor, Tensor[])",
         [](Stack & stack) {
         
             auto result_ = at::miopen_rnn_backward(
                 (std::move(peek(stack, 0, 21))).toTensor(),
                 (std::move(peek(stack, 1, 21))).toTensorListRef(),
                 (std::move(peek(stack, 2, 21))).toInt(),
                 (std::move(peek(stack, 3, 21))).toTensor(),
                 (std::move(peek(stack, 4, 21))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 5, 21)))),
                 (std::move(peek(stack, 6, 21))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 7, 21)))),
                 toOptionalTensor((std::move(peek(stack, 8, 21)))),
                 toOptionalTensor((std::move(peek(stack, 9, 21)))),
                 (std::move(peek(stack, 10, 21))).toInt(),
                 (std::move(peek(stack, 11, 21))).toInt(),
                 (std::move(peek(stack, 12, 21))).toInt(),
                 (std::move(peek(stack, 13, 21))).toBool(),
                 (std::move(peek(stack, 14, 21))).toDouble(),
                 (std::move(peek(stack, 15, 21))).toBool(),
                 (std::move(peek(stack, 16, 21))).toBool(),
                 (std::move(peek(stack, 17, 21))).toIntListRef(),
                 toOptionalTensor((std::move(peek(stack, 18, 21)))),
                 (std::move(peek(stack, 19, 21))).toTensor(),
                 as_bool_array<4>((std::move(peek(stack, 20, 21))).toBoolList())
             );
             drop(stack, 21);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::mkldnn_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::mkldnn_convolution(
                 (std::move(peek(stack, 0, 7))).toTensor(),
                 (std::move(peek(stack, 1, 7))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 2, 7)))),
                 (std::move(peek(stack, 3, 7))).toIntListRef(),
                 (std::move(peek(stack, 4, 7))).toIntListRef(),
                 (std::move(peek(stack, 5, 7))).toIntListRef(),
                 (std::move(peek(stack, 6, 7))).toInt()
             );
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::mkldnn_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool bias_defined) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::mkldnn_convolution_backward_input(
                 (std::move(peek(stack, 0, 8))).toIntListRef(),
                 (std::move(peek(stack, 1, 8))).toTensor(),
                 (std::move(peek(stack, 2, 8))).toTensor(),
                 (std::move(peek(stack, 3, 8))).toIntListRef(),
                 (std::move(peek(stack, 4, 8))).toIntListRef(),
                 (std::move(peek(stack, 5, 8))).toIntListRef(),
                 (std::move(peek(stack, 6, 8))).toInt(),
                 (std::move(peek(stack, 7, 8))).toBool()
             );
             drop(stack, 8);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::mm(Tensor self, Tensor mat2) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::mm(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::mse_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::mse_loss(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor(),
                 (std::move(peek(stack, 2, 3))).toInt()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::mse_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::mul_out(
                 out,
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::multi_margin_loss(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::multi_margin_loss(
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toTensor(),
                 (std::move(peek(stack, 2, 6))).toScalar(),
                 (std::move(peek(stack, 3, 6))).toScalar(),
                 toOptionalTensor((std::move(peek(stack, 4, 6)))),
                 (std::move(peek(stack, 5, 6))).toInt()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::multi_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 7, 8))).toTensor();
             auto result_ = at::multi_margin_loss_backward_out(
                 grad_input,
                 (std::move(peek(stack, 0, 8))).toTensor(),
                 (std::move(peek(stack, 1, 8))).toTensor(),
                 (std::move(peek(stack, 2, 8))).toTensor(),
                 (std::move(peek(stack, 3, 8))).toScalar(),
                 (std::move(peek(stack, 4, 8))).toScalar(),
                 toOptionalTensor((std::move(peek(stack, 5, 8)))),
                 (std::move(peek(stack, 6, 8))).toInt()
             );
             drop(stack, 8);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::multilabel_margin_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::multilabel_margin_loss_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 (std::move(peek(stack, 2, 4))).toInt()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::multinomial.out(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 4, 5))).toTensor();
             auto result_ = at::multinomial_out(
                 out,
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toInt(),
                 (std::move(peek(stack, 2, 5))).toBool(),
                 nullptr
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::mv(Tensor self, Tensor vec) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::mv(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)",
         [](Stack & stack) {
         
             auto result_ = at::narrow(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toInt(),
                 (std::move(peek(stack, 2, 4))).toInt(),
                 (std::move(peek(stack, 3, 4))).toInt()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::ne.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::ne_out(
                 out,
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::ne.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::ne_out(
                 out,
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toScalar()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::neg(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::neg(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::new_full(Tensor self, int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 3, 7))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 4, 7))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 5, 7))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 6, 7))).toOptional<bool>());;
             auto result_ = ((std::move(peek(stack, 0, 7))).toTensor()).new_full((std::move(peek(stack, 1, 7))).toIntListRef(),
             (std::move(peek(stack, 2, 7))).toScalar(),
             options);
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::nll_loss2d_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::nll_loss2d_backward(
                 (std::move(peek(stack, 0, 7))).toTensor(),
                 (std::move(peek(stack, 1, 7))).toTensor(),
                 (std::move(peek(stack, 2, 7))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 3, 7)))),
                 (std::move(peek(stack, 4, 7))).toInt(),
                 (std::move(peek(stack, 5, 7))).toInt(),
                 (std::move(peek(stack, 6, 7))).toTensor()
             );
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::nll_loss_backward(
                 (std::move(peek(stack, 0, 7))).toTensor(),
                 (std::move(peek(stack, 1, 7))).toTensor(),
                 (std::move(peek(stack, 2, 7))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 3, 7)))),
                 (std::move(peek(stack, 4, 7))).toInt(),
                 (std::move(peek(stack, 5, 7))).toInt(),
                 (std::move(peek(stack, 6, 7))).toTensor()
             );
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::nonzero(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::nonzero(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::normal.Tensor_Tensor_out(Tensor mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::normal_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 nullptr
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::normal.float_Tensor_out(float mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::normal_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toDouble(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 nullptr
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::normal.Tensor_float_out(Tensor mean, float std=1, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::normal_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toDouble(),
                 nullptr
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::normal.float_float_out(float mean, float std, int[] size, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 4, 5))).toTensor();
             auto result_ = at::normal_out(
                 out,
                 (std::move(peek(stack, 0, 5))).toDouble(),
                 (std::move(peek(stack, 1, 5))).toDouble(),
                 (std::move(peek(stack, 2, 5))).toIntListRef(),
                 nullptr
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::nuclear_norm.out(Tensor self, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::nuclear_norm_out(
                 out,
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toBool()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::nuclear_norm.dim_out(Tensor self, int[2] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::nuclear_norm_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toIntListRef(),
                 (std::move(peek(stack, 2, 4))).toBool()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::numel(Tensor self) -> int",
         [](Stack & stack) {
         
             auto result_ = at::numel(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::ones(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 1, 5))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 2, 5))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 3, 5))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 4, 5))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::ones((std::move(peek(stack, 0, 5))).toIntListRef(),
             options);
             #else
                 auto result_ = torch::ones((std::move(peek(stack, 0, 5))).toIntListRef(),
             options);
             #endif
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::ones_like(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::ones_like(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::ones_like.dtype(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 1, 5))).toScalarType())
                     .layout((std::move(peek(stack, 2, 5))).toLayout())
                     .device((std::move(peek(stack, 3, 5))).toDevice())
                     .pinned_memory((std::move(peek(stack, 4, 5))).toBool());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::ones_like((std::move(peek(stack, 0, 5))).toTensor(),
             options);
             #else
                 auto result_ = torch::ones_like((std::move(peek(stack, 0, 5))).toTensor(),
             options);
             #endif
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::pdist(Tensor self, float p=2) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::pdist(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toDouble()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 2))).toTensor()).permute(
                 (std::move(peek(stack, 1, 2))).toIntListRef()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::poisson(Tensor self, Generator? generator=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::poisson(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 nullptr
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::polygamma.out(int n, Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::polygamma_out(
                 out,
                 (std::move(peek(stack, 0, 3))).toInt(),
                 (std::move(peek(stack, 1, 3))).toTensor()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::pow_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).pow_(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::pow_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).pow_(
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::prelu_backward(Tensor grad_output, Tensor self, Tensor weight) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::prelu_backward(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor(),
                 (std::move(peek(stack, 2, 3))).toTensor()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::prod(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toOptional<ScalarType>()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::prod(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toInt(),
                 (std::move(peek(stack, 2, 4))).toBool(),
                 (std::move(peek(stack, 3, 4))).toOptional<ScalarType>()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::put_(Tensor(a!) self, Tensor index, Tensor source, bool accumulate=False) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 4))).toTensor();
             auto result_ = (self).put_(
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 (std::move(peek(stack, 2, 4))).toTensor(),
                 (std::move(peek(stack, 3, 4))).toBool()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::q_per_channel_zero_points(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::q_per_channel_zero_points(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::q_scale(Tensor self) -> float",
         [](Stack & stack) {
         
             auto result_ = at::q_scale(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::qr(Tensor self, bool some=True) -> (Tensor Q, Tensor R)",
         [](Stack & stack) {
         
             auto result_ = at::qr(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toBool()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::quantized_rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::quantized_rnn_relu_cell(
                 (std::move(peek(stack, 0, 14))).toTensor(),
                 (std::move(peek(stack, 1, 14))).toTensor(),
                 (std::move(peek(stack, 2, 14))).toTensor(),
                 (std::move(peek(stack, 3, 14))).toTensor(),
                 (std::move(peek(stack, 4, 14))).toTensor(),
                 (std::move(peek(stack, 5, 14))).toTensor(),
                 (std::move(peek(stack, 6, 14))).toTensor(),
                 (std::move(peek(stack, 7, 14))).toTensor(),
                 (std::move(peek(stack, 8, 14))).toTensor(),
                 (std::move(peek(stack, 9, 14))).toTensor(),
                 (std::move(peek(stack, 10, 14))).toScalar(),
                 (std::move(peek(stack, 11, 14))).toScalar(),
                 (std::move(peek(stack, 12, 14))).toScalar(),
                 (std::move(peek(stack, 13, 14))).toScalar()
             );
             drop(stack, 14);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::rand.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::rand_out(
                 out,
                 (std::move(peek(stack, 0, 2))).toIntListRef()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::randint(int high, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 2, 6))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 3, 6))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 4, 6))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 5, 6))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::randint((std::move(peek(stack, 0, 6))).toInt(),
             (std::move(peek(stack, 1, 6))).toIntListRef(),
             options);
             #else
                 auto result_ = torch::randint((std::move(peek(stack, 0, 6))).toInt(),
             (std::move(peek(stack, 1, 6))).toIntListRef(),
             options);
             #endif
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::randint.low(int low, int high, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 3, 7))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 4, 7))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 5, 7))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 6, 7))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::randint((std::move(peek(stack, 0, 7))).toInt(),
             (std::move(peek(stack, 1, 7))).toInt(),
             (std::move(peek(stack, 2, 7))).toIntListRef(),
             options);
             #else
                 auto result_ = torch::randint((std::move(peek(stack, 0, 7))).toInt(),
             (std::move(peek(stack, 1, 7))).toInt(),
             (std::move(peek(stack, 2, 7))).toIntListRef(),
             options);
             #endif
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::randint_like(Tensor self, int high) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::randint_like(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toInt()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::randint_like.low(Tensor self, int low, int high) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::randint_like(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toInt(),
                 (std::move(peek(stack, 2, 3))).toInt()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::randint_like.dtype(Tensor self, int high, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 2, 6))).toScalarType())
                     .layout((std::move(peek(stack, 3, 6))).toLayout())
                     .device((std::move(peek(stack, 4, 6))).toDevice())
                     .pinned_memory((std::move(peek(stack, 5, 6))).toBool());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::randint_like((std::move(peek(stack, 0, 6))).toTensor(),
             (std::move(peek(stack, 1, 6))).toInt(),
             options);
             #else
                 auto result_ = torch::randint_like((std::move(peek(stack, 0, 6))).toTensor(),
             (std::move(peek(stack, 1, 6))).toInt(),
             options);
             #endif
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::randint_like.low_dtype(Tensor self, int low, int high, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 3, 7))).toScalarType())
                     .layout((std::move(peek(stack, 4, 7))).toLayout())
                     .device((std::move(peek(stack, 5, 7))).toDevice())
                     .pinned_memory((std::move(peek(stack, 6, 7))).toBool());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::randint_like((std::move(peek(stack, 0, 7))).toTensor(),
             (std::move(peek(stack, 1, 7))).toInt(),
             (std::move(peek(stack, 2, 7))).toInt(),
             options);
             #else
                 auto result_ = torch::randint_like((std::move(peek(stack, 0, 7))).toTensor(),
             (std::move(peek(stack, 1, 7))).toInt(),
             (std::move(peek(stack, 2, 7))).toInt(),
             options);
             #endif
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::randperm(int n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 1, 5))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 2, 5))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 3, 5))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 4, 5))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::randperm((std::move(peek(stack, 0, 5))).toInt(),
             options);
             #else
                 auto result_ = torch::randperm((std::move(peek(stack, 0, 5))).toInt(),
             options);
             #endif
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::reciprocal.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::reciprocal_out(
                 out,
                 (std::move(peek(stack, 0, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::reflection_pad1d_backward(Tensor grad_output, Tensor self, int[2] padding) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::reflection_pad1d_backward(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor(),
                 (std::move(peek(stack, 2, 3))).toIntListRef()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::reflection_pad2d.out(Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::reflection_pad2d_out(
                 out,
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toIntListRef()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::relu(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::relu(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::remainder_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).remainder_(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::remainder_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).remainder_(
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::renorm(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toScalar(),
                 (std::move(peek(stack, 2, 4))).toInt(),
                 (std::move(peek(stack, 3, 4))).toScalar()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::repeat(Tensor self, int[] repeats) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 2))).toTensor()).repeat(
                 (std::move(peek(stack, 1, 2))).toIntListRef()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::replication_pad1d_backward(Tensor grad_output, Tensor self, int[2] padding) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::replication_pad1d_backward(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor(),
                 (std::move(peek(stack, 2, 3))).toIntListRef()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::replication_pad2d.out(Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::replication_pad2d_out(
                 out,
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toIntListRef()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::replication_pad3d(Tensor self, int[6] padding) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::replication_pad3d(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toIntListRef()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::replication_pad3d_backward.grad_input(Tensor grad_output, Tensor self, int[6] padding, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::replication_pad3d_backward_out(
                 grad_input,
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 (std::move(peek(stack, 2, 4))).toIntListRef()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::rfft(Tensor self, int signal_ndim, bool normalized=False, bool onesided=True) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::rfft(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toInt(),
                 (std::move(peek(stack, 2, 4))).toBool(),
                 (std::move(peek(stack, 3, 4))).toBool()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::rnn_relu.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::rnn_relu(
                 (std::move(peek(stack, 0, 9))).toTensor(),
                 (std::move(peek(stack, 1, 9))).toTensor(),
                 (std::move(peek(stack, 2, 9))).toTensor(),
                 (std::move(peek(stack, 3, 9))).toTensorListRef(),
                 (std::move(peek(stack, 4, 9))).toBool(),
                 (std::move(peek(stack, 5, 9))).toInt(),
                 (std::move(peek(stack, 6, 9))).toDouble(),
                 (std::move(peek(stack, 7, 9))).toBool(),
                 (std::move(peek(stack, 8, 9))).toBool()
             );
             drop(stack, 9);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::rnn_relu.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::rnn_relu(
                 (std::move(peek(stack, 0, 9))).toTensor(),
                 (std::move(peek(stack, 1, 9))).toTensor(),
                 (std::move(peek(stack, 2, 9))).toTensorListRef(),
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::rnn_tanh_cell(
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toTensor(),
                 (std::move(peek(stack, 2, 6))).toTensor(),
                 (std::move(peek(stack, 3, 6))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 4, 6)))),
                 toOptionalTensor((std::move(peek(stack, 5, 6))))
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::rot90(Tensor self, int k=1, int[] dims=[0,1]) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::rot90(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toInt(),
                 (std::move(peek(stack, 2, 3))).toIntListRef()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::round_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::round_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::rrelu(Tensor self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::rrelu_with_noise(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::rrelu_with_noise_backward.grad_input(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::rsqrt(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::rsqrt(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::scatter(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toInt(),
                 (std::move(peek(stack, 2, 4))).toTensor(),
                 (std::move(peek(stack, 3, 4))).toTensor()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::scatter(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toInt(),
                 (std::move(peek(stack, 2, 4))).toTensor(),
                 (std::move(peek(stack, 3, 4))).toScalar()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::selu_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::selu_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::sigmoid_out(
                 out,
                 (std::move(peek(stack, 0, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sign.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::sign_out(
                 out,
                 (std::move(peek(stack, 0, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sin_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::sin_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sinh(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::sinh(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::size.int(Tensor self, int dim) -> int",
         [](Stack & stack) {
         
             auto result_ = at::size(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toInt()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::slow_conv_dilated2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::slow_conv_dilated2d(
                 (std::move(peek(stack, 0, 7))).toTensor(),
                 (std::move(peek(stack, 1, 7))).toTensor(),
                 (std::move(peek(stack, 2, 7))).toIntListRef(),
                 toOptionalTensor((std::move(peek(stack, 3, 7)))),
                 (std::move(peek(stack, 4, 7))).toIntListRef(),
                 (std::move(peek(stack, 5, 7))).toIntListRef(),
                 (std::move(peek(stack, 6, 7))).toIntListRef()
             );
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::slow_conv_dilated3d_backward(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)",
         [](Stack & stack) {
         
             auto result_ = at::slow_conv_dilated3d_backward(
                 (std::move(peek(stack, 0, 8))).toTensor(),
                 (std::move(peek(stack, 1, 8))).toTensor(),
                 (std::move(peek(stack, 2, 8))).toTensor(),
                 (std::move(peek(stack, 3, 8))).toIntListRef(),
                 (std::move(peek(stack, 4, 8))).toIntListRef(),
                 (std::move(peek(stack, 5, 8))).toIntListRef(),
                 (std::move(peek(stack, 6, 8))).toIntListRef(),
                 as_bool_array<3>((std::move(peek(stack, 7, 8))).toBoolList())
             );
             drop(stack, 8);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::slow_conv_transpose2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int[2] dilation=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::slow_conv_transpose2d(
                 (std::move(peek(stack, 0, 8))).toTensor(),
                 (std::move(peek(stack, 1, 8))).toTensor(),
                 (std::move(peek(stack, 2, 8))).toIntListRef(),
                 toOptionalTensor((std::move(peek(stack, 3, 8)))),
                 (std::move(peek(stack, 4, 8))).toIntListRef(),
                 (std::move(peek(stack, 5, 8))).toIntListRef(),
                 (std::move(peek(stack, 6, 8))).toIntListRef(),
                 (std::move(peek(stack, 7, 8))).toIntListRef()
             );
             drop(stack, 8);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::slow_conv_transpose3d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] output_padding, int[3] dilation, Tensor finput, Tensor fgrad_input, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)",
         [](Stack & stack) {
         
             auto result_ = at::slow_conv_transpose3d_backward(
                 (std::move(peek(stack, 0, 11))).toTensor(),
                 (std::move(peek(stack, 1, 11))).toTensor(),
                 (std::move(peek(stack, 2, 11))).toTensor(),
                 (std::move(peek(stack, 3, 11))).toIntListRef(),
                 (std::move(peek(stack, 4, 11))).toIntListRef(),
                 (std::move(peek(stack, 5, 11))).toIntListRef(),
                 (std::move(peek(stack, 6, 11))).toIntListRef(),
                 (std::move(peek(stack, 7, 11))).toIntListRef(),
                 (std::move(peek(stack, 8, 11))).toTensor(),
                 (std::move(peek(stack, 9, 11))).toTensor(),
                 as_bool_array<3>((std::move(peek(stack, 10, 11))).toBoolList())
             );
             drop(stack, 11);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::smooth_l1_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::smooth_l1_loss_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 (std::move(peek(stack, 2, 4))).toInt()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::soft_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::soft_margin_loss(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor(),
                 (std::move(peek(stack, 2, 3))).toInt()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::soft_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::softmax(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toInt(),
                 (std::move(peek(stack, 2, 3))).toOptional<ScalarType>()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::softplus(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toScalar(),
                 (std::move(peek(stack, 2, 3))).toScalar()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::softplus_backward.grad_input(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::softshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::softshrink_out(
                 out,
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toScalar()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sparse_coo_tensor.size(int[] size, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 1, 5))).toScalarType())
                     .layout((std::move(peek(stack, 2, 5))).toLayout())
                     .device((std::move(peek(stack, 3, 5))).toDevice())
                     .pinned_memory((std::move(peek(stack, 4, 5))).toBool());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::sparse_coo_tensor((std::move(peek(stack, 0, 5))).toIntListRef(),
             options);
             #else
                 auto result_ = torch::sparse_coo_tensor((std::move(peek(stack, 0, 5))).toIntListRef(),
             options);
             #endif
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sparse_coo_tensor.indices(Tensor indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 2, 6))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 3, 6))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 4, 6))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 5, 6))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::sparse_coo_tensor((std::move(peek(stack, 0, 6))).toTensor(),
             (std::move(peek(stack, 1, 6))).toTensor(),
             options);
             #else
                 auto result_ = torch::sparse_coo_tensor((std::move(peek(stack, 0, 6))).toTensor(),
             (std::move(peek(stack, 1, 6))).toTensor(),
             options);
             #endif
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sparse_coo_tensor.indices_size(Tensor indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 3, 7))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 4, 7))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 5, 7))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 6, 7))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::sparse_coo_tensor((std::move(peek(stack, 0, 7))).toTensor(),
             (std::move(peek(stack, 1, 7))).toTensor(),
             (std::move(peek(stack, 2, 7))).toIntListRef(),
             options);
             #else
                 auto result_ = torch::sparse_coo_tensor((std::move(peek(stack, 0, 7))).toTensor(),
             (std::move(peek(stack, 1, 7))).toTensor(),
             (std::move(peek(stack, 2, 7))).toIntListRef(),
             options);
             #endif
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sparse_resize_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 4))).toTensor();
             auto result_ = (self).sparse_resize_(
                 (std::move(peek(stack, 1, 4))).toIntListRef(),
                 (std::move(peek(stack, 2, 4))).toInt(),
                 (std::move(peek(stack, 3, 4))).toInt()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::split_with_sizes(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]",
         [](Stack & stack) {
         
             auto result_ = at::split_with_sizes(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toIntListRef(),
                 (std::move(peek(stack, 2, 3))).toInt()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sqrt(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::sqrt(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::squeeze(Tensor(a) self) -> Tensor(a)",
         [](Stack & stack) {
         
             auto result_ = at::squeeze(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)",
         [](Stack & stack) {
         
             auto result_ = at::squeeze(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toInt()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::std.out(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 4, 5))).toTensor();
             auto result_ = at::std_out(
                 out,
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toIntListRef(),
                 (std::move(peek(stack, 2, 5))).toBool(),
                 (std::move(peek(stack, 3, 5))).toBool()
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::stft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool normalized=False, bool onesided=True) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::stft(
                 (std::move(peek(stack, 0, 7))).toTensor(),
                 (std::move(peek(stack, 1, 7))).toInt(),
                 (std::move(peek(stack, 2, 7))).toOptional<int64_t>(),
                 (std::move(peek(stack, 3, 7))).toOptional<int64_t>(),
                 toOptionalTensor((std::move(peek(stack, 4, 7)))),
                 (std::move(peek(stack, 5, 7))).toBool(),
                 (std::move(peek(stack, 6, 7))).toBool()
             );
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = (self).sub_(
                 (std::move(peek(stack, 1, 3))).toTensor(),
                 (std::move(peek(stack, 2, 3))).toScalar()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = (self).sub_(
                 (std::move(peek(stack, 1, 3))).toScalar(),
                 (std::move(peek(stack, 2, 3))).toScalar()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::sum(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toOptional<ScalarType>()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::sum(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toIntListRef(),
                 (std::move(peek(stack, 2, 4))).toBool(),
                 (std::move(peek(stack, 3, 4))).toOptional<ScalarType>()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::t(Tensor(a) self) -> Tensor(a)",
         [](Stack & stack) {
         
             auto result_ = at::t(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::take.out(Tensor self, Tensor index, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::take_out(
                 out,
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::tan(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::tan(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::tanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::tanh_out(
                 out,
                 (std::move(peek(stack, 0, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::thnn_conv2d.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 6, 7))).toTensor();
             auto result_ = at::thnn_conv2d_out(
                 out,
                 (std::move(peek(stack, 0, 7))).toTensor(),
                 (std::move(peek(stack, 1, 7))).toTensor(),
                 (std::move(peek(stack, 2, 7))).toIntListRef(),
                 toOptionalTensor((std::move(peek(stack, 3, 7)))),
                 (std::move(peek(stack, 4, 7))).toIntListRef(),
                 (std::move(peek(stack, 5, 7))).toIntListRef()
             );
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::thnn_conv3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::thnn_conv3d(
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toTensor(),
                 (std::move(peek(stack, 2, 6))).toIntListRef(),
                 toOptionalTensor((std::move(peek(stack, 3, 6)))),
                 (std::move(peek(stack, 4, 6))).toIntListRef(),
                 (std::move(peek(stack, 5, 6))).toIntListRef()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::thnn_conv3d_forward(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, int[3] padding) -> (Tensor output, Tensor finput, Tensor fgrad_input)",
         [](Stack & stack) {
         
             auto result_ = at::thnn_conv3d_forward(
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toTensor(),
                 (std::move(peek(stack, 2, 6))).toIntListRef(),
                 toOptionalTensor((std::move(peek(stack, 3, 6)))),
                 (std::move(peek(stack, 4, 6))).toIntListRef(),
                 (std::move(peek(stack, 5, 6))).toIntListRef()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::thnn_conv_depthwise2d_forward.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 7, 8))).toTensor();
             auto result_ = at::thnn_conv_depthwise2d_forward_out(
                 out,
                 (std::move(peek(stack, 0, 8))).toTensor(),
                 (std::move(peek(stack, 1, 8))).toTensor(),
                 (std::move(peek(stack, 2, 8))).toIntListRef(),
                 toOptionalTensor((std::move(peek(stack, 3, 8)))),
                 (std::move(peek(stack, 4, 8))).toIntListRef(),
                 (std::move(peek(stack, 5, 8))).toIntListRef(),
                 (std::move(peek(stack, 6, 8))).toIntListRef()
             );
             drop(stack, 8);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::thnn_conv_depthwise2d.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 7, 8))).toTensor();
             auto result_ = at::thnn_conv_depthwise2d_out(
                 out,
                 (std::move(peek(stack, 0, 8))).toTensor(),
                 (std::move(peek(stack, 1, 8))).toTensor(),
                 (std::move(peek(stack, 2, 8))).toIntListRef(),
                 toOptionalTensor((std::move(peek(stack, 3, 8)))),
                 (std::move(peek(stack, 4, 8))).toIntListRef(),
                 (std::move(peek(stack, 5, 8))).toIntListRef(),
                 (std::move(peek(stack, 6, 8))).toIntListRef()
             );
             drop(stack, 8);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::threshold.out(Tensor self, Scalar threshold, Scalar value, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::threshold_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toScalar(),
                 (std::move(peek(stack, 2, 4))).toScalar()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::to.other(Tensor self, Tensor other, bool non_blocking=False, bool copy=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 4))).toTensor()).to(
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 (std::move(peek(stack, 2, 4))).toBool(),
                 (std::move(peek(stack, 3, 4))).toBool()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::to.dtype(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 4))).toTensor()).to(
                 (std::move(peek(stack, 1, 4))).toScalarType(),
                 (std::move(peek(stack, 2, 4))).toBool(),
                 (std::move(peek(stack, 3, 4))).toBool()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::to.device(Tensor self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 5))).toTensor()).to(
                 (std::move(peek(stack, 1, 5))).toDevice(),
                 (std::move(peek(stack, 2, 5))).toScalarType(),
                 (std::move(peek(stack, 3, 5))).toBool(),
                 (std::move(peek(stack, 4, 5))).toBool()
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::to.dtype_layout(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, bool non_blocking=False, bool copy=False) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 1, 7))).toScalarType())
                     .layout((std::move(peek(stack, 2, 7))).toLayout())
                     .device((std::move(peek(stack, 3, 7))).toDevice())
                     .pinned_memory((std::move(peek(stack, 4, 7))).toBool());;
             auto result_ = ((std::move(peek(stack, 0, 7))).toTensor()).to(options,
             (std::move(peek(stack, 5, 7))).toBool(),
             (std::move(peek(stack, 6, 7))).toBool());
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::to_dense_backward(Tensor grad, Tensor input) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::to_dense_backward(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::to_mkldnn_backward(Tensor grad, Tensor input) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::to_mkldnn_backward(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::trace(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::trace(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::tril(Tensor self, int diagonal=0) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::tril(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toInt()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::triu(Tensor self, int diagonal=0) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::triu(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toInt()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::trunc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::trunc_out(
                 out,
                 (std::move(peek(stack, 0, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 4))).toTensor();
             auto result_ = (self).uniform_(
                 (std::move(peek(stack, 1, 4))).toDouble(),
                 (std::move(peek(stack, 2, 4))).toDouble(),
                 nullptr
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::upsample_bicubic2d.out(Tensor self, int[2] output_size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::upsample_bicubic2d_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toIntListRef(),
                 (std::move(peek(stack, 2, 4))).toBool()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::upsample_bilinear2d.out(Tensor self, int[2] output_size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::upsample_bilinear2d_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toIntListRef(),
                 (std::move(peek(stack, 2, 4))).toBool()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::upsample_linear1d.out(Tensor self, int[1] output_size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::upsample_linear1d_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toIntListRef(),
                 (std::move(peek(stack, 2, 4))).toBool()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::upsample_nearest1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::upsample_nearest1d_backward(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toIntListRef(),
                 (std::move(peek(stack, 2, 3))).toIntListRef()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::upsample_nearest2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::upsample_nearest2d_out(
                 out,
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toIntListRef()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::upsample_nearest3d(Tensor self, int[3] output_size) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::upsample_nearest3d(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toIntListRef()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::upsample_nearest3d_backward.grad_input(Tensor grad_output, int[3] output_size, int[5] input_size, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::upsample_nearest3d_backward_out(
                 grad_input,
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toIntListRef(),
                 (std::move(peek(stack, 2, 4))).toIntListRef()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::upsample_trilinear3d(Tensor self, int[3] output_size, bool align_corners) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::upsample_trilinear3d(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toIntListRef(),
                 (std::move(peek(stack, 2, 3))).toBool()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::upsample_trilinear3d_backward.grad_input(Tensor grad_output, int[3] output_size, int[5] input_size, bool align_corners, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
             auto result_ = at::upsample_trilinear3d_backward_out(
                 grad_input,
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toIntListRef(),
                 (std::move(peek(stack, 2, 5))).toIntListRef(),
                 (std::move(peek(stack, 3, 5))).toBool()
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::values(Tensor(a) self) -> Tensor(a)",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).values(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::var(Tensor self, bool unbiased=True) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::var(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toBool()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::var.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::var(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toIntListRef(),
                 (std::move(peek(stack, 2, 4))).toBool(),
                 (std::move(peek(stack, 3, 4))).toBool()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::view(Tensor(a) self, int[] size) -> Tensor(a)",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 2))).toTensor()).view(
                 (std::move(peek(stack, 1, 2))).toIntListRef()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::where(Tensor condition) -> Tensor[]",
         [](Stack & stack) {
         
             auto result_ = at::where(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::where.self(Tensor condition, Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::where(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor(),
                 (std::move(peek(stack, 2, 3))).toTensor()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::zeros(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 1, 5))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 2, 5))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 3, 5))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 4, 5))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::zeros((std::move(peek(stack, 0, 5))).toIntListRef(),
             options);
             #else
                 auto result_ = torch::zeros((std::move(peek(stack, 0, 5))).toIntListRef(),
             options);
             #endif
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::zeros_like(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::zeros_like(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::zeros_like.dtype(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 1, 5))).toScalarType())
                     .layout((std::move(peek(stack, 2, 5))).toLayout())
                     .device((std::move(peek(stack, 3, 5))).toDevice())
                     .pinned_memory((std::move(peek(stack, 4, 5))).toBool());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::zeros_like((std::move(peek(stack, 0, 5))).toTensor(),
             options);
             #else
                 auto result_ = torch::zeros_like((std::move(peek(stack, 0, 5))).toTensor(),
             options);
             #endif
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),});

} // anon namespace


}} // namespace torch::jit
