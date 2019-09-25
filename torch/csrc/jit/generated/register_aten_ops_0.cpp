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
         "aten::__and__.Tensor(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::__and__(
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
         "aten::__and__.Scalar(Tensor self, Scalar other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::__and__(
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
         "aten::__iand__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).__iand__(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::__iand__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).__iand__(
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_adaptive_avg_pool2d_backward(
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
         "aten::_addr.out(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 5, 6))).toTensor();
             auto result_ = at::_addr_out(
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
         "aten::_baddbmm_mkl_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_batch_norm_impl_index_backward(int impl_index, Tensor input, Tensor grad_output, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var_transform, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_batch_norm_impl_index_backward(
                 (std::move(peek(stack, 0, 11))).toInt(),
                 (std::move(peek(stack, 1, 11))).toTensor(),
                 (std::move(peek(stack, 2, 11))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 3, 11)))),
                 toOptionalTensor((std::move(peek(stack, 4, 11)))),
                 toOptionalTensor((std::move(peek(stack, 5, 11)))),
                 toOptionalTensor((std::move(peek(stack, 6, 11)))),
                 toOptionalTensor((std::move(peek(stack, 7, 11)))),
                 (std::move(peek(stack, 8, 11))).toBool(),
                 (std::move(peek(stack, 9, 11))).toDouble(),
                 as_bool_array<3>((std::move(peek(stack, 10, 11))).toBoolList())
             );
             drop(stack, 11);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_cast_Char(Tensor self, bool non_blocking=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_cast_Char(
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
         "aten::_cast_Float(Tensor self, bool non_blocking=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_cast_Float(
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
         "aten::_cast_Half(Tensor self, bool non_blocking=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_cast_Half(
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
         "aten::_cast_Long(Tensor self, bool non_blocking=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_cast_Long(
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
         "aten::_cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::_cat_out(
                 out,
                 (std::move(peek(stack, 0, 3))).toTensorListRef(),
                 (std::move(peek(stack, 1, 3))).toInt()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_convolution(
                 (std::move(peek(stack, 0, 12))).toTensor(),
                 (std::move(peek(stack, 1, 12))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 2, 12)))),
                 (std::move(peek(stack, 3, 12))).toIntListRef(),
                 (std::move(peek(stack, 4, 12))).toIntListRef(),
                 (std::move(peek(stack, 5, 12))).toIntListRef(),
                 (std::move(peek(stack, 6, 12))).toBool(),
                 (std::move(peek(stack, 7, 12))).toIntListRef(),
                 (std::move(peek(stack, 8, 12))).toInt(),
                 (std::move(peek(stack, 9, 12))).toBool(),
                 (std::move(peek(stack, 10, 12))).toBool(),
                 (std::move(peek(stack, 11, 12))).toBool()
             );
             drop(stack, 12);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_convolution_double_backward(Tensor? ggI, Tensor? ggW, Tensor? ggb, Tensor gO, Tensor weight, Tensor self, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_convolution_double_backward(
                 toOptionalTensor((std::move(peek(stack, 0, 16)))),
                 toOptionalTensor((std::move(peek(stack, 1, 16)))),
                 toOptionalTensor((std::move(peek(stack, 2, 16)))),
                 (std::move(peek(stack, 3, 16))).toTensor(),
                 (std::move(peek(stack, 4, 16))).toTensor(),
                 (std::move(peek(stack, 5, 16))).toTensor(),
                 (std::move(peek(stack, 6, 16))).toIntListRef(),
                 (std::move(peek(stack, 7, 16))).toIntListRef(),
                 (std::move(peek(stack, 8, 16))).toIntListRef(),
                 (std::move(peek(stack, 9, 16))).toBool(),
                 (std::move(peek(stack, 10, 16))).toIntListRef(),
                 (std::move(peek(stack, 11, 16))).toInt(),
                 (std::move(peek(stack, 12, 16))).toBool(),
                 (std::move(peek(stack, 13, 16))).toBool(),
                 (std::move(peek(stack, 14, 16))).toBool(),
                 as_bool_array<3>((std::move(peek(stack, 15, 16))).toBoolList())
             );
             drop(stack, 16);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_convolution_nogroup(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_convolution_nogroup(
                 (std::move(peek(stack, 0, 8))).toTensor(),
                 (std::move(peek(stack, 1, 8))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 2, 8)))),
                 (std::move(peek(stack, 3, 8))).toIntListRef(),
                 (std::move(peek(stack, 4, 8))).toIntListRef(),
                 (std::move(peek(stack, 5, 8))).toIntListRef(),
                 (std::move(peek(stack, 6, 8))).toBool(),
                 (std::move(peek(stack, 7, 8))).toIntListRef()
             );
             drop(stack, 8);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_cudnn_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank, bool deterministic, bool zero_infinity) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_cudnn_ctc_loss(
                 (std::move(peek(stack, 0, 7))).toTensor(),
                 (std::move(peek(stack, 1, 7))).toTensor(),
                 (std::move(peek(stack, 2, 7))).toIntListRef(),
                 (std::move(peek(stack, 3, 7))).toIntListRef(),
                 (std::move(peek(stack, 4, 7))).toInt(),
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
         "aten::_cudnn_rnn_backward(Tensor input, Tensor[] weight, int weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state, Tensor reserve, bool[4] output_mask) -> (Tensor, Tensor, Tensor, Tensor[])",
         [](Stack & stack) {
         
             auto result_ = at::_cudnn_rnn_backward(
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
         "aten::_cumprod.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::_cumprod_out(
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
         "aten::_cumsum.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::_cumsum_out(
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
         "aten::_dimV(Tensor self) -> int",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor())._dimV(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_embedding_bag_sparse_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, int num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_embedding_bag_sparse_backward(
                 (std::move(peek(stack, 0, 9))).toTensor(),
                 (std::move(peek(stack, 1, 9))).toTensor(),
                 (std::move(peek(stack, 2, 9))).toTensor(),
                 (std::move(peek(stack, 3, 9))).toTensor(),
                 (std::move(peek(stack, 4, 9))).toTensor(),
                 (std::move(peek(stack, 5, 9))).toInt(),
                 (std::move(peek(stack, 6, 9))).toBool(),
                 (std::move(peek(stack, 7, 9))).toInt(),
                 toOptionalTensor((std::move(peek(stack, 8, 9))))
             );
             drop(stack, 9);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_empty_affine_quantized(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, float scale=1, int zero_point=0, MemoryFormat? memory_format=contiguous_format) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 1, 8))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 2, 8))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 3, 8))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 4, 8))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::_empty_affine_quantized((std::move(peek(stack, 0, 8))).toIntListRef(),
             options,
             (std::move(peek(stack, 5, 8))).toDouble(),
             (std::move(peek(stack, 6, 8))).toInt(),
             (std::move(peek(stack, 7, 8))).toOptional<c10::MemoryFormat>());
             #else
                 auto result_ = torch::_empty_affine_quantized((std::move(peek(stack, 0, 8))).toIntListRef(),
             options,
             (std::move(peek(stack, 5, 8))).toDouble(),
             (std::move(peek(stack, 6, 8))).toInt(),
             (std::move(peek(stack, 7, 8))).toOptional<c10::MemoryFormat>());
             #endif
             drop(stack, 8);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_fused_dropout(Tensor self, float p, Generator? generator=None) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_fused_dropout(
                 (std::move(peek(stack, 0, 3))).toTensor(),
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
         "aten::_index_put_impl_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 5))).toTensor();
             auto result_ = at::_index_put_impl_(
                 self,
                 toListOfOptionalTensor((std::move(peek(stack, 1, 5)))),
                 (std::move(peek(stack, 2, 5))).toTensor(),
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
         "aten::_index_put_impl_(Tensor(a!) self, Tensor[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 5))).toTensor();
             auto result_ = at::_index_put_impl_(
                 self,
                 (std::move(peek(stack, 1, 5))).toTensorListRef(),
                 (std::move(peek(stack, 2, 5))).toTensor(),
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
         "aten::_local_scalar_dense(Tensor self) -> Scalar",
         [](Stack & stack) {
         
             auto result_ = at::_local_scalar_dense(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_make_per_tensor_quantized_tensor(Tensor self, float scale, int zero_point) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_make_per_tensor_quantized_tensor(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toDouble(),
                 (std::move(peek(stack, 2, 3))).toInt()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_mkldnn_reshape(Tensor self, int[] shape) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_mkldnn_reshape(
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
         "aten::_mkldnn_transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = at::_mkldnn_transpose_(
                 self,
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
         "aten::_mode(Tensor self, int dim=-1, bool keepdim=False) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_mode(
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
         "aten::_nnpack_spatial_convolution(Tensor input, Tensor weight, Tensor? bias, int[2] padding) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_nnpack_spatial_convolution(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 2, 4)))),
                 (std::move(peek(stack, 3, 4))).toIntListRef()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_nnpack_spatial_convolution_backward_input(Tensor input, Tensor grad_output, Tensor weight, int[2] padding) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_nnpack_spatial_convolution_backward_input(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 (std::move(peek(stack, 2, 4))).toTensor(),
                 (std::move(peek(stack, 3, 4))).toIntListRef()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_pack_padded_sequence_backward(Tensor grad, int[] input_size, Tensor batch_sizes, bool batch_first) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_pack_padded_sequence_backward(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toIntListRef(),
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
         "aten::_pad_packed_sequence(Tensor data, Tensor batch_sizes, bool batch_first, Scalar padding_value, int total_length) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_pdist_backward(Tensor grad, Tensor self, float p, Tensor pdist) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_pdist_backward(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 (std::move(peek(stack, 2, 4))).toDouble(),
                 (std::move(peek(stack, 3, 4))).toTensor()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_sobol_engine_initialize_state_(Tensor(a!) self, int dimension) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = at::_sobol_engine_initialize_state_(
                 self,
                 (std::move(peek(stack, 1, 2))).toInt()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_sobol_engine_scramble_(Tensor(a!) self, Tensor ltm, int dimension) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = at::_sobol_engine_scramble_(
                 self,
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
         "aten::_solve_helper(Tensor self, Tensor A) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_solve_helper(
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
         "aten::_sparse_addmm(Tensor self, Tensor sparse, Tensor dense, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_sparse_coo_tensor_unsafe(Tensor indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 3, 7))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 4, 7))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 5, 7))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 6, 7))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::_sparse_coo_tensor_unsafe((std::move(peek(stack, 0, 7))).toTensor(),
             (std::move(peek(stack, 1, 7))).toTensor(),
             (std::move(peek(stack, 2, 7))).toIntListRef(),
             options);
             #else
                 auto result_ = torch::_sparse_coo_tensor_unsafe((std::move(peek(stack, 0, 7))).toTensor(),
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
         "aten::_sparse_coo_tensor_with_dims_and_tensors(int sparse_dim, int dense_dim, int[] size, Tensor indices, Tensor values, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 5, 9))).toScalarType())
                     .layout((std::move(peek(stack, 6, 9))).toLayout())
                     .device((std::move(peek(stack, 7, 9))).toDevice())
                     .pinned_memory((std::move(peek(stack, 8, 9))).toBool());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::_sparse_coo_tensor_with_dims_and_tensors((std::move(peek(stack, 0, 9))).toInt(),
             (std::move(peek(stack, 1, 9))).toInt(),
             (std::move(peek(stack, 2, 9))).toIntListRef(),
             (std::move(peek(stack, 3, 9))).toTensor(),
             (std::move(peek(stack, 4, 9))).toTensor(),
             options);
             #else
                 auto result_ = torch::_sparse_coo_tensor_with_dims_and_tensors((std::move(peek(stack, 0, 9))).toInt(),
             (std::move(peek(stack, 1, 9))).toInt(),
             (std::move(peek(stack, 2, 9))).toIntListRef(),
             (std::move(peek(stack, 3, 9))).toTensor(),
             (std::move(peek(stack, 4, 9))).toTensor(),
             options);
             #endif
             drop(stack, 9);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_sparse_mm(Tensor sparse, Tensor dense) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_sparse_mm(
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
         "aten::_sparse_sum(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_sparse_sum(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_sparse_sum.dtype(Tensor self, *, ScalarType dtype) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_sparse_sum(
                 (std::move(peek(stack, 0, 2))).toTensor(),
                 (std::move(peek(stack, 1, 2))).toScalarType()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_sparse_sum.dim(Tensor self, int[1] dim) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_sparse_sum(
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
         "aten::_sparse_sum.dim_dtype(Tensor self, int[1] dim, *, ScalarType dtype) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_sparse_sum(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toIntListRef(),
                 (std::move(peek(stack, 2, 3))).toScalarType()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_standard_gamma(Tensor self, Generator? generator=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_standard_gamma(
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
         "aten::_std(Tensor self, bool unbiased=True) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_std(
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
         "aten::_triangular_solve_helper(Tensor self, Tensor A, bool upper, bool transpose, bool unitriangular) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_triangular_solve_helper(
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toTensor(),
                 (std::move(peek(stack, 2, 5))).toBool(),
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
         "aten::_trilinear(Tensor i1, Tensor i2, Tensor i3, int[] expand1, int[] expand2, int[] expand3, int[] sumdim, int unroll_dim=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_trilinear(
                 (std::move(peek(stack, 0, 8))).toTensor(),
                 (std::move(peek(stack, 1, 8))).toTensor(),
                 (std::move(peek(stack, 2, 8))).toTensor(),
                 (std::move(peek(stack, 3, 8))).toIntListRef(),
                 (std::move(peek(stack, 4, 8))).toIntListRef(),
                 (std::move(peek(stack, 5, 8))).toIntListRef(),
                 (std::move(peek(stack, 6, 8))).toIntListRef(),
                 (std::move(peek(stack, 7, 8))).toInt()
             );
             drop(stack, 8);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_unsafe_view(Tensor self, int[] size) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_unsafe_view(
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
         "aten::_weight_norm_cuda_interface_backward(Tensor grad_w, Tensor saved_v, Tensor saved_g, Tensor saved_norms, int dim) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::abs_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::abs_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::acos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::acos_out(
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
         "aten::adaptive_avg_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::adaptive_avg_pool2d_out(
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
         "aten::adaptive_avg_pool3d(Tensor self, int[3] output_size) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::adaptive_avg_pool3d(
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
         "aten::adaptive_avg_pool3d_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::adaptive_avg_pool3d_backward_out(
                 grad_input,
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
         "aten::adaptive_max_pool1d(Tensor self, int[1] output_size) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::adaptive_max_pool1d(
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
         "aten::adaptive_max_pool2d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::adaptive_max_pool2d_backward(
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
         "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::add(
                 (std::move(peek(stack, 0, 3))).toTensor(),
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
         "aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::add(
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
         "aten::addbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::addcdiv.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 4, 5))).toTensor();
             auto result_ = at::addcdiv_out(
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
         "aten::addcmul_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 4))).toTensor();
             auto result_ = (self).addcmul_(
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
         "aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 5, 6))).toTensor();
             auto result_ = at::addmm_out(
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
         "aten::addmv.out(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 5, 6))).toTensor();
             auto result_ = at::addmv_out(
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
         "aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::affine_grid_generator(Tensor theta, int[] size, bool align_corners) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::affine_grid_generator(
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
         "aten::alias(Tensor(a) self) -> Tensor(a)",
         [](Stack & stack) {
         
             auto result_ = at::alias(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::asin_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::asin_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::atan(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::atan(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::atan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::atan2_out(
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
         "aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::avg_pool2d(
                 (std::move(peek(stack, 0, 7))).toTensor(),
                 (std::move(peek(stack, 1, 7))).toIntListRef(),
                 (std::move(peek(stack, 2, 7))).toIntListRef(),
                 (std::move(peek(stack, 3, 7))).toIntListRef(),
                 (std::move(peek(stack, 4, 7))).toBool(),
                 (std::move(peek(stack, 5, 7))).toBool(),
                 (std::move(peek(stack, 6, 7))).toOptional<int64_t>()
             );
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::avg_pool2d_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 8, 9))).toTensor();
             auto result_ = at::avg_pool2d_backward_out(
                 grad_input,
                 (std::move(peek(stack, 0, 9))).toTensor(),
                 (std::move(peek(stack, 1, 9))).toTensor(),
                 (std::move(peek(stack, 2, 9))).toIntListRef(),
                 (std::move(peek(stack, 3, 9))).toIntListRef(),
                 (std::move(peek(stack, 4, 9))).toIntListRef(),
                 (std::move(peek(stack, 5, 9))).toBool(),
                 (std::move(peek(stack, 6, 9))).toBool(),
                 (std::move(peek(stack, 7, 9))).toOptional<int64_t>()
             );
             drop(stack, 9);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::avg_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::avg_pool3d_backward(
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
         "aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::batch_norm(
                 (std::move(peek(stack, 0, 9))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 1, 9)))),
                 toOptionalTensor((std::move(peek(stack, 2, 9)))),
                 toOptionalTensor((std::move(peek(stack, 3, 9)))),
                 toOptionalTensor((std::move(peek(stack, 4, 9)))),
                 (std::move(peek(stack, 5, 9))).toBool(),
                 (std::move(peek(stack, 6, 9))).toDouble(),
                 (std::move(peek(stack, 7, 9))).toDouble(),
                 (std::move(peek(stack, 8, 9))).toBool()
             );
             drop(stack, 9);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::batch_norm_backward_reduce(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, bool input_g, bool weight_g, bool bias_g) -> (Tensor, Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::batch_norm_backward_reduce(
                 (std::move(peek(stack, 0, 8))).toTensor(),
                 (std::move(peek(stack, 1, 8))).toTensor(),
                 (std::move(peek(stack, 2, 8))).toTensor(),
                 (std::move(peek(stack, 3, 8))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 4, 8)))),
                 (std::move(peek(stack, 5, 8))).toBool(),
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
         "aten::batch_norm_elemt(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor invstd, float eps) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::batch_norm_elemt(
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 1, 6)))),
                 toOptionalTensor((std::move(peek(stack, 2, 6)))),
                 (std::move(peek(stack, 3, 6))).toTensor(),
                 (std::move(peek(stack, 4, 6))).toTensor(),
                 (std::move(peek(stack, 5, 6))).toDouble()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::batch_norm_stats(Tensor input, float eps) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::batch_norm_stats(
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
         "aten::batch_norm_update_stats(Tensor input, Tensor? running_mean, Tensor? running_var, float momentum) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::batch_norm_update_stats(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 1, 4)))),
                 toOptionalTensor((std::move(peek(stack, 2, 4)))),
                 (std::move(peek(stack, 3, 4))).toDouble()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::bernoulli(Tensor self, *, Generator? generator=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::bernoulli(
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
         "aten::bernoulli.p(Tensor self, float p, *, Generator? generator=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::bernoulli(
                 (std::move(peek(stack, 0, 3))).toTensor(),
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
         "aten::binary_cross_entropy.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 4, 5))).toTensor();
             auto result_ = at::binary_cross_entropy_out(
                 out,
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 2, 5)))),
                 (std::move(peek(stack, 3, 5))).toInt()
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::binary_cross_entropy_with_logits_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=Mean) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::binary_cross_entropy_with_logits_backward(
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toTensor(),
                 (std::move(peek(stack, 2, 6))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 3, 6)))),
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
         "aten::bitwise_not(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::bitwise_not(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::blackman_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 1, 5))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 2, 5))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 3, 5))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 4, 5))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::blackman_window((std::move(peek(stack, 0, 5))).toInt(),
             options);
             #else
                 auto result_ = torch::blackman_window((std::move(peek(stack, 0, 5))).toInt(),
             options);
             #endif
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::blackman_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 2, 6))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 3, 6))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 4, 6))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 5, 6))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::blackman_window((std::move(peek(stack, 0, 6))).toInt(),
             (std::move(peek(stack, 1, 6))).toBool(),
             options);
             #else
                 auto result_ = torch::blackman_window((std::move(peek(stack, 0, 6))).toInt(),
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
         "aten::broadcast_tensors(Tensor[] tensors) -> Tensor[]",
         [](Stack & stack) {
         
             auto result_ = at::broadcast_tensors(
                 (std::move(peek(stack, 0, 1))).toTensorListRef()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cat(Tensor[] tensors, int dim=0) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::cat(
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
         "aten::cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator? generator=None) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 4))).toTensor();
             auto result_ = (self).cauchy_(
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
         "aten::ceil.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::ceil_out(
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
         "aten::chain_matmul(Tensor[] matrices) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::chain_matmul(
                 (std::move(peek(stack, 0, 1))).toTensorListRef()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cholesky_inverse(Tensor self, bool upper=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::cholesky_inverse(
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
         "aten::cholesky.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::cholesky_out(
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
         "aten::cholesky_solve.out(Tensor self, Tensor input2, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::cholesky_solve_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 (std::move(peek(stack, 2, 4))).toBool()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::chunk(Tensor(a) self, int chunks, int dim=0) -> Tensor(a)[]",
         [](Stack & stack) {
         
             auto result_ = at::chunk(
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
         "aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::clamp(
                 (std::move(peek(stack, 0, 3))).toTensor(),
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
         "aten::clamp_max_(Tensor(a!) self, Scalar max) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = at::clamp_max_(
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
         "aten::clamp_min.out(Tensor self, Scalar min, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::clamp_min_out(
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
         "aten::coalesce(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).coalesce(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::col2im(Tensor self, int[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::col2im(
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
         "aten::col2im_backward.grad_input(Tensor grad_output, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 5, 6))).toTensor();
             auto result_ = at::col2im_backward_out(
                 grad_input,
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toIntListRef(),
                 (std::move(peek(stack, 2, 6))).toIntListRef(),
                 (std::move(peek(stack, 3, 6))).toIntListRef(),
                 (std::move(peek(stack, 4, 6))).toIntListRef()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::contiguous(Tensor self, *, MemoryFormat memory_format=contiguous_format) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 2))).toTensor()).contiguous(
                 (std::move(peek(stack, 1, 2))).toMemoryFormat()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::conv2d(
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
         "aten::conv_tbc(Tensor self, Tensor weight, Tensor bias, int pad=0) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::conv_tbc(
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
         "aten::conv_transpose2d.input(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int groups=1, int[2] dilation=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::conv_transpose2d(
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
         "aten::copy_sparse_to_sparse_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = at::copy_sparse_to_sparse_(
                 self,
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
         "aten::cos_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::cos_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cosh(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::cosh(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cross.out(Tensor self, Tensor other, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::cross_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 (std::move(peek(stack, 2, 4))).toOptional<int64_t>()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::ctc_loss.Tensor(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank=0, int reduction=Mean, bool zero_infinity=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::ctc_loss(
                 (std::move(peek(stack, 0, 7))).toTensor(),
                 (std::move(peek(stack, 1, 7))).toTensor(),
                 (std::move(peek(stack, 2, 7))).toTensor(),
                 (std::move(peek(stack, 3, 7))).toTensor(),
                 (std::move(peek(stack, 4, 7))).toInt(),
                 (std::move(peek(stack, 5, 7))).toInt(),
                 (std::move(peek(stack, 6, 7))).toBool()
             );
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::ctc_loss.IntList(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, int reduction=Mean, bool zero_infinity=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::ctc_loss(
                 (std::move(peek(stack, 0, 7))).toTensor(),
                 (std::move(peek(stack, 1, 7))).toTensor(),
                 (std::move(peek(stack, 2, 7))).toIntListRef(),
                 (std::move(peek(stack, 3, 7))).toIntListRef(),
                 (std::move(peek(stack, 4, 7))).toInt(),
                 (std::move(peek(stack, 5, 7))).toInt(),
                 (std::move(peek(stack, 6, 7))).toBool()
             );
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cudnn_affine_grid_generator_backward(Tensor grad, int N, int C, int H, int W) -> Tensor grad_theta",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cudnn_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::cudnn_batch_norm_backward(
                 (std::move(peek(stack, 0, 8))).toTensor(),
                 (std::move(peek(stack, 1, 8))).toTensor(),
                 (std::move(peek(stack, 2, 8))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 3, 8)))),
                 toOptionalTensor((std::move(peek(stack, 4, 8)))),
                 toOptionalTensor((std::move(peek(stack, 5, 8)))),
                 toOptionalTensor((std::move(peek(stack, 6, 8)))),
                 (std::move(peek(stack, 7, 8))).toDouble()
             );
             drop(stack, 8);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cudnn_convolution_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::cudnn_convolution_backward_weight(
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
         "aten::cudnn_convolution_transpose_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::cudnn_convolution_transpose_backward_weight(
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
         "aten::cudnn_is_acceptable(Tensor self) -> bool",
         [](Stack & stack) {
         
             auto result_ = at::cudnn_is_acceptable(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::cumprod(
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
         "aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::cumsum(
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
         "aten::dense_dim(Tensor self) -> int",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).dense_dim(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::diag(Tensor self, int diagonal=0) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::diag(
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
         "aten::diagflat(Tensor self, int offset=0) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::diagflat(
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
         "aten::diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)",
         [](Stack & stack) {
         
             auto result_ = at::diagonal(
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
         "aten::digamma(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::digamma(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::div_out(
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
         "aten::dot(Tensor self, Tensor tensor) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::dot(
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
         "aten::dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = at::dropout_(
                 self,
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
         "aten::eig(Tensor self, bool eigenvectors=False) -> (Tensor eigenvalues, Tensor eigenvectors)",
         [](Stack & stack) {
         
             auto result_ = at::eig(
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
         "aten::einsum(str equation, Tensor[] tensors) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::einsum(
                 (std::move(peek(stack, 0, 2))).toStringRef(),
                 (std::move(peek(stack, 1, 2))).toTensorListRef()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::elu.out(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 4, 5))).toTensor();
             auto result_ = at::elu_out(
                 out,
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toScalar(),
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
         "aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None) -> (Tensor, Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::embedding_bag(
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
         "aten::embedding_dense_backward(Tensor grad_output, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::embedding_renorm_(Tensor(a!) self, Tensor indices, float max_norm, float norm_type) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::eq_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).eq_(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::eq_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).eq_(
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::erf_out(
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
         "aten::erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::erfc_out(
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
         "aten::erfinv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::erfinv_out(
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
         "aten::exp(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::exp(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::expm1_out(
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
         "aten::eye.out(int n, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::eye_out(
                 out,
                 (std::move(peek(stack, 0, 2))).toInt()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::eye.m_out(int n, int m, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::eye_out(
                 out,
                 (std::move(peek(stack, 0, 3))).toInt(),
                 (std::move(peek(stack, 1, 3))).toInt()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::fake_quantize_per_tensor_affine(Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::fake_quantize_per_tensor_affine(
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toDouble(),
                 (std::move(peek(stack, 2, 5))).toInt(),
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
         "aten::fbgemm_linear_fp16_weight_fp32_activation(Tensor input, Tensor packed_weight, Tensor bias) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::fbgemm_linear_fp16_weight_fp32_activation(
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
         "aten::fbgemm_linear_int8_weight(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::fbgemm_linear_int8_weight(
                 (std::move(peek(stack, 0, 7))).toTensor(),
                 (std::move(peek(stack, 1, 7))).toTensor(),
                 (std::move(peek(stack, 2, 7))).toTensor(),
                 (std::move(peek(stack, 3, 7))).toTensor(),
                 (std::move(peek(stack, 4, 7))).toScalar(),
                 (std::move(peek(stack, 5, 7))).toScalar(),
                 (std::move(peek(stack, 6, 7))).toTensor()
             );
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::fbgemm_pack_gemm_matrix_fp16(Tensor input) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::fbgemm_pack_gemm_matrix_fp16(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::fbgemm_pack_quantized_matrix(Tensor input) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::fbgemm_pack_quantized_matrix(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::fbgemm_pack_quantized_matrix.KN(Tensor input, int K, int N) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::fbgemm_pack_quantized_matrix(
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
         "aten::feature_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = at::feature_dropout_(
                 self,
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
         "aten::flatten.using_ints(Tensor self, int start_dim=0, int end_dim=-1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::flatten(
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
         "aten::floor(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::floor(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::fmod.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::fmod_out(
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
         "aten::fmod.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::fmod_out(
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
         "aten::frac_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::frac_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::fractional_max_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] output_size, Tensor indices) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::fractional_max_pool2d_backward(
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toTensor(),
                 (std::move(peek(stack, 2, 5))).toIntListRef(),
                 (std::move(peek(stack, 3, 5))).toIntListRef(),
                 (std::move(peek(stack, 4, 5))).toTensor()
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::frobenius_norm(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::frobenius_norm(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::frobenius_norm.dim(Tensor self, int[1] dim, bool keepdim=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::frobenius_norm(
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
         "aten::full(int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 2, 6))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 3, 6))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 4, 6))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 5, 6))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::full((std::move(peek(stack, 0, 6))).toIntListRef(),
             (std::move(peek(stack, 1, 6))).toScalar(),
             options);
             #else
                 auto result_ = torch::full((std::move(peek(stack, 0, 6))).toIntListRef(),
             (std::move(peek(stack, 1, 6))).toScalar(),
             options);
             #endif
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::full_like(Tensor self, Scalar fill_value) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::full_like(
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
         "aten::full_like.dtype(Tensor self, Scalar fill_value, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 2, 6))).toScalarType())
                     .layout((std::move(peek(stack, 3, 6))).toLayout())
                     .device((std::move(peek(stack, 4, 6))).toDevice())
                     .pinned_memory((std::move(peek(stack, 5, 6))).toBool());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::full_like((std::move(peek(stack, 0, 6))).toTensor(),
             (std::move(peek(stack, 1, 6))).toScalar(),
             options);
             #else
                 auto result_ = torch::full_like((std::move(peek(stack, 0, 6))).toTensor(),
             (std::move(peek(stack, 1, 6))).toScalar(),
             options);
             #endif
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::gather.out(Tensor self, int dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 4, 5))).toTensor();
             auto result_ = at::gather_out(
                 out,
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toInt(),
                 (std::move(peek(stack, 2, 5))).toTensor(),
                 (std::move(peek(stack, 3, 5))).toBool()
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::ge.Tensor(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::ge(
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
         "aten::ge.Scalar(Tensor self, Scalar other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::ge(
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
         "aten::gelu(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::gelu(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::ger(Tensor self, Tensor vec2) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::ger(
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
         "aten::glu_backward(Tensor grad_output, Tensor self, int dim) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::glu_backward(
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
         "aten::grid_sampler_2d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::grid_sampler_2d(
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toTensor(),
                 (std::move(peek(stack, 2, 5))).toInt(),
                 (std::move(peek(stack, 3, 5))).toInt(),
                 (std::move(peek(stack, 4, 5))).toBool()
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::grid_sampler_3d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::grid_sampler_3d_backward(
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
         "aten::gt.Tensor(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::gt(
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
         "aten::gt.Scalar(Tensor self, Scalar other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::gt(
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
         "aten::hardtanh.out(Tensor self, Scalar min_val=-1, Scalar max_val=1, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::hardtanh_out(
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
         "aten::histc.out(Tensor self, int bins=100, Scalar min=0, Scalar max=0, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 4, 5))).toTensor();
             auto result_ = at::histc_out(
                 out,
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toInt(),
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
         "aten::hspmm(Tensor mat1, Tensor mat2) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::hspmm(
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
         "aten::im2col(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::im2col(
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
         "aten::im2col_backward.grad_input(Tensor grad_output, int[2] input_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 6, 7))).toTensor();
             auto result_ = at::im2col_backward_out(
                 grad_input,
                 (std::move(peek(stack, 0, 7))).toTensor(),
                 (std::move(peek(stack, 1, 7))).toIntListRef(),
                 (std::move(peek(stack, 2, 7))).toIntListRef(),
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
         "aten::index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 4))).toTensor();
             auto result_ = (self).index_add_(
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
         "aten::index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::index_copy(
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
         "aten::index_fill_.Tensor(Tensor(a!) self, int dim, Tensor index, Tensor value) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 4))).toTensor();
             auto result_ = (self).index_fill_(
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
         "aten::index_fill_.Scalar(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 4))).toTensor();
             auto result_ = (self).index_fill_(
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
         "aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 4))).toTensor();
             auto result_ = at::index_put_(
                 self,
                 toListOfOptionalTensor((std::move(peek(stack, 1, 4)))),
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
         "aten::index_put_(Tensor(a!) self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 4))).toTensor();
             auto result_ = at::index_put_(
                 self,
                 (std::move(peek(stack, 1, 4))).toTensorListRef(),
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
         "aten::index_select.out(Tensor self, int dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::index_select_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toInt(),
                 (std::move(peek(stack, 2, 4))).toTensor()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::indices(Tensor(a) self) -> Tensor(a)",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).indices(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::instance_norm(
                 (std::move(peek(stack, 0, 9))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 1, 9)))),
                 toOptionalTensor((std::move(peek(stack, 2, 9)))),
                 toOptionalTensor((std::move(peek(stack, 3, 9)))),
                 toOptionalTensor((std::move(peek(stack, 4, 9)))),
                 (std::move(peek(stack, 5, 9))).toBool(),
                 (std::move(peek(stack, 6, 9))).toDouble(),
                 (std::move(peek(stack, 7, 9))).toDouble(),
                 (std::move(peek(stack, 8, 9))).toBool()
             );
             drop(stack, 9);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::int_repr(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::int_repr(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::inverse.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::inverse_out(
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
         "aten::is_leaf(Tensor self) -> bool",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).is_leaf(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::is_same_size(Tensor self, Tensor other) -> bool",
         [](Stack & stack) {
         
             auto result_ = at::is_same_size(
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
         "aten::is_set_to(Tensor self, Tensor tensor) -> bool",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 2))).toTensor()).is_set_to(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::isnan(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::isnan(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::kl_div(Tensor self, Tensor target, int reduction=Mean) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::kl_div(
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
         "aten::l1_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::l1_loss_out(
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
         "aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::layer_norm(
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toIntListRef(),
                 toOptionalTensor((std::move(peek(stack, 2, 6)))),
                 toOptionalTensor((std::move(peek(stack, 3, 6)))),
                 (std::move(peek(stack, 4, 6))).toDouble(),
                 (std::move(peek(stack, 5, 6))).toBool()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::le.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::le_out(
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
         "aten::le.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::le_out(
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
         "aten::leaky_relu_(Tensor(a!) self, Scalar negative_slope=0.01) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = at::leaky_relu_(
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
         "aten::leaky_relu_backward(Tensor grad_output, Tensor self, Scalar negative_slope) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::leaky_relu_backward(
                 (std::move(peek(stack, 0, 3))).toTensor(),
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
         "aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::lerp(
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
         "aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::lerp(
                 (std::move(peek(stack, 0, 3))).toTensor(),
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
         "aten::lgamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::lgamma_out(
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
         "aten::log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::log10_out(
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
         "aten::log1p(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::log1p(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::log2(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::log2(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::log_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::log_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::log_normal_(Tensor(a!) self, float mean=1, float std=2, *, Generator? generator=None) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 4))).toTensor();
             auto result_ = (self).log_normal_(
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
         "aten::log_sigmoid_backward(Tensor grad_output, Tensor self, Tensor buffer) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::log_sigmoid_backward(
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
         "aten::logdet(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::logdet(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::logical_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::logical_not_out(
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
         "aten::logical_xor_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).logical_xor_(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::logspace(Scalar start, Scalar end, int steps=100, float base=10.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 4, 8))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 5, 8))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 6, 8))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 7, 8))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::logspace((std::move(peek(stack, 0, 8))).toScalar(),
             (std::move(peek(stack, 1, 8))).toScalar(),
             (std::move(peek(stack, 2, 8))).toInt(),
             (std::move(peek(stack, 3, 8))).toDouble(),
             options);
             #else
                 auto result_ = torch::logspace((std::move(peek(stack, 0, 8))).toScalar(),
             (std::move(peek(stack, 1, 8))).toScalar(),
             (std::move(peek(stack, 2, 8))).toInt(),
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
         "aten::logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::logsumexp(
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
         "aten::lstsq(Tensor self, Tensor A) -> (Tensor solution, Tensor QR)",
         [](Stack & stack) {
         
             auto result_ = at::lstsq(
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
         "aten::lt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::lt_out(
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
         "aten::lt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::lt_out(
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
         "aten::lu_solve(Tensor self, Tensor LU_data, Tensor LU_pivots) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::lu_solve(
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
         "aten::masked_fill_.Tensor(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = (self).masked_fill_(
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
         "aten::masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = (self).masked_fill_(
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
         "aten::masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::masked_scatter(
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
         "aten::masked_select.out(Tensor self, Tensor mask, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::masked_select_out(
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
         "aten::matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::matmul_out(
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
         "aten::max.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::max_out(
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
         "aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::max_pool2d_with_indices(
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
         "aten::max_pool2d_with_indices_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 8, 9))).toTensor();
             auto result_ = at::max_pool2d_with_indices_backward_out(
                 grad_input,
                 (std::move(peek(stack, 0, 9))).toTensor(),
                 (std::move(peek(stack, 1, 9))).toTensor(),
                 (std::move(peek(stack, 2, 9))).toIntListRef(),
                 (std::move(peek(stack, 3, 9))).toIntListRef(),
                 (std::move(peek(stack, 4, 9))).toIntListRef(),
                 (std::move(peek(stack, 5, 9))).toIntListRef(),
                 (std::move(peek(stack, 6, 9))).toBool(),
                 (std::move(peek(stack, 7, 9))).toTensor()
             );
             drop(stack, 9);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::max_pool3d(
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
         "aten::max_pool3d_with_indices_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::max_pool3d_with_indices_backward(
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
         "aten::max_unpool2d_backward(Tensor grad_output, Tensor self, Tensor indices, int[2] output_size) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::max_unpool2d_backward(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 (std::move(peek(stack, 2, 4))).toTensor(),
                 (std::move(peek(stack, 3, 4))).toIntListRef()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::max_unpool3d.out(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 5, 6))).toTensor();
             auto result_ = at::max_unpool3d_out(
                 out,
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toTensor(),
                 (std::move(peek(stack, 2, 6))).toIntListRef(),
                 (std::move(peek(stack, 3, 6))).toIntListRef(),
                 (std::move(peek(stack, 4, 6))).toIntListRef()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::max_values(Tensor self, int[1] dim, bool keepdim=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::max_values(
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
         "aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::mean(
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
         "aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::mean(
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
         "aten::min(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::min(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::min.other(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::min(
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
         "aten::min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)",
         [](Stack & stack) {
         
             auto result_ = at::min(
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
         "aten::miopen_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::miopen_convolution(
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
         "aten::miopen_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::miopen_convolution_backward_input(
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
         "aten::miopen_convolution_transpose(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::miopen_convolution_transpose(
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
         "aten::miopen_convolution_transpose_backward_input(Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::miopen_convolution_transpose_backward_input(
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
         "aten::miopen_depthwise_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::miopen_depthwise_convolution(
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
         "aten::miopen_depthwise_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::miopen_depthwise_convolution_backward_input(
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
         "aten::miopen_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor hx, Tensor? cx, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::miopen_rnn(
                 (std::move(peek(stack, 0, 14))).toTensor(),
                 (std::move(peek(stack, 1, 14))).toTensorListRef(),
                 (std::move(peek(stack, 2, 14))).toInt(),
                 (std::move(peek(stack, 3, 14))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 4, 14)))),
                 (std::move(peek(stack, 5, 14))).toInt(),
                 (std::move(peek(stack, 6, 14))).toInt(),
                 (std::move(peek(stack, 7, 14))).toInt(),
                 (std::move(peek(stack, 8, 14))).toBool(),
                 (std::move(peek(stack, 9, 14))).toDouble(),
                 (std::move(peek(stack, 10, 14))).toBool(),
                 (std::move(peek(stack, 11, 14))).toBool(),
                 (std::move(peek(stack, 12, 14))).toIntListRef(),
                 toOptionalTensor((std::move(peek(stack, 13, 14))))
             );
             drop(stack, 14);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::mkldnn_adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::mkldnn_adaptive_avg_pool2d(
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
         "aten::mkldnn_linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::mkldnn_linear(
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
         "aten::mkldnn_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::mkldnn_max_pool2d(
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
         "aten::mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::mm_out(
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
         "aten::mse_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::mse_loss_out(
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
         "aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).mul_(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).mul_(
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::multi_margin_loss.out(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 6, 7))).toTensor();
             auto result_ = at::multi_margin_loss_out(
                 out,
                 (std::move(peek(stack, 0, 7))).toTensor(),
                 (std::move(peek(stack, 1, 7))).toTensor(),
                 (std::move(peek(stack, 2, 7))).toScalar(),
                 (std::move(peek(stack, 3, 7))).toScalar(),
                 toOptionalTensor((std::move(peek(stack, 4, 7)))),
                 (std::move(peek(stack, 5, 7))).toInt()
             );
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::multilabel_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target) -> Tensor",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::mv.out(Tensor self, Tensor vec, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::mv_out(
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
         "aten::mvlgamma_(Tensor(a!) self, int p) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).mvlgamma_(
                 (std::move(peek(stack, 1, 2))).toInt()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::narrow_copy(Tensor self, int dim, int start, int length) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 4))).toTensor()).narrow_copy(
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
         "aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::native_batch_norm_backward(
                 (std::move(peek(stack, 0, 10))).toTensor(),
                 (std::move(peek(stack, 1, 10))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 2, 10)))),
                 toOptionalTensor((std::move(peek(stack, 3, 10)))),
                 toOptionalTensor((std::move(peek(stack, 4, 10)))),
                 toOptionalTensor((std::move(peek(stack, 5, 10)))),
                 toOptionalTensor((std::move(peek(stack, 6, 10)))),
                 (std::move(peek(stack, 7, 10))).toBool(),
                 (std::move(peek(stack, 8, 10))).toDouble(),
                 as_bool_array<3>((std::move(peek(stack, 9, 10))).toBoolList())
             );
             drop(stack, 10);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::native_layer_norm_backward(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, int M, int N, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::native_layer_norm_backward(
                 (std::move(peek(stack, 0, 8))).toTensor(),
                 (std::move(peek(stack, 1, 8))).toTensor(),
                 (std::move(peek(stack, 2, 8))).toTensor(),
                 (std::move(peek(stack, 3, 8))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 4, 8)))),
                 (std::move(peek(stack, 5, 8))).toInt(),
                 (std::move(peek(stack, 6, 8))).toInt(),
                 as_bool_array<3>((std::move(peek(stack, 7, 8))).toBoolList())
             );
             drop(stack, 8);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::ne_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).ne_(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::ne_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).ne_(
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::neg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::neg_out(
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
         "aten::new_empty(Tensor self, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 2, 6))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 3, 6))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 4, 6))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 5, 6))).toOptional<bool>());;
             auto result_ = ((std::move(peek(stack, 0, 6))).toTensor()).new_empty((std::move(peek(stack, 1, 6))).toIntListRef(),
             options);
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::nll_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::nll_loss(
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 2, 5)))),
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
         "aten::nll_loss2d(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::nll_loss2d(
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 2, 5)))),
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
         "aten::nll_loss2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 7, 8))).toTensor();
             auto result_ = at::nll_loss2d_backward_out(
                 grad_input,
                 (std::move(peek(stack, 0, 8))).toTensor(),
                 (std::move(peek(stack, 1, 8))).toTensor(),
                 (std::move(peek(stack, 2, 8))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 3, 8)))),
                 (std::move(peek(stack, 4, 8))).toInt(),
                 (std::move(peek(stack, 5, 8))).toInt(),
                 (std::move(peek(stack, 6, 8))).toTensor()
             );
             drop(stack, 8);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::nll_loss2d_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor output, Tensor total_weight)",
         [](Stack & stack) {
         
             auto result_ = at::nll_loss2d_forward(
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 2, 5)))),
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
         "aten::nll_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 7, 8))).toTensor();
             auto result_ = at::nll_loss_backward_out(
                 grad_input,
                 (std::move(peek(stack, 0, 8))).toTensor(),
                 (std::move(peek(stack, 1, 8))).toTensor(),
                 (std::move(peek(stack, 2, 8))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 3, 8)))),
                 (std::move(peek(stack, 4, 8))).toInt(),
                 (std::move(peek(stack, 5, 8))).toInt(),
                 (std::move(peek(stack, 6, 8))).toTensor()
             );
             drop(stack, 8);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor output, Tensor total_weight)",
         [](Stack & stack) {
         
             auto result_ = at::nll_loss_forward(
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 2, 5)))),
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
         "aten::nonzero_numpy(Tensor self) -> Tensor[]",
         [](Stack & stack) {
         
             auto result_ = at::nonzero_numpy(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::nonzero.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::nonzero_out(
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
         "aten::norm.Scalar(Tensor self, Scalar p=2) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::norm(
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
         "aten::norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::norm(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toOptional<Scalar>(),
                 (std::move(peek(stack, 2, 3))).toScalarType()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::norm(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toOptional<Scalar>(),
                 (std::move(peek(stack, 2, 4))).toIntListRef(),
                 (std::move(peek(stack, 3, 4))).toBool()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::norm(
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toOptional<Scalar>(),
                 (std::move(peek(stack, 2, 5))).toIntListRef(),
                 (std::move(peek(stack, 3, 5))).toBool(),
                 (std::move(peek(stack, 4, 5))).toScalarType()
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 4))).toTensor();
             auto result_ = (self).normal_(
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
         "aten::ones.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::ones_out(
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
         "aten::orgqr(Tensor self, Tensor input2) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::orgqr(
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
         "aten::ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::output_nr(Tensor self) -> int",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).output_nr(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::pairwise_distance(Tensor x1, Tensor x2, float p=2, float eps=1e-06, bool keepdim=False) -> Tensor",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::pinverse(Tensor self, float rcond=1e-15) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::pinverse(
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
         "aten::polygamma_(Tensor(a!) self, int n) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).polygamma_(
                 (std::move(peek(stack, 1, 2))).toInt()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::pow(
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
         "aten::pow.Scalar(Scalar self, Tensor exponent) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::pow(
                 (std::move(peek(stack, 0, 2))).toScalar(),
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::pow(
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
         "aten::prelu(Tensor self, Tensor weight) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::prelu(
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
         "aten::prod.int_out(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 4, 5))).toTensor();
             auto result_ = at::prod_out(
                 out,
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toInt(),
                 (std::move(peek(stack, 2, 5))).toBool(),
                 (std::move(peek(stack, 3, 5))).toOptional<ScalarType>()
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::q_zero_point(Tensor self) -> int",
         [](Stack & stack) {
         
             auto result_ = at::q_zero_point(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::quantize_per_channel(Tensor self, Tensor scales, Tensor zero_points, int axis, ScalarType dtype) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::quantize_per_channel(
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toTensor(),
                 (std::move(peek(stack, 2, 5))).toTensor(),
                 (std::move(peek(stack, 3, 5))).toInt(),
                 (std::move(peek(stack, 4, 5))).toScalarType()
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::quantized_gru.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::quantized_gru(
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
         "aten::quantized_gru.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::quantized_gru(
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
         "aten::quantized_lstm(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first, *, ScalarType? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::quantized_lstm(
                 (std::move(peek(stack, 0, 11))).toTensor(),
                 (std::move(peek(stack, 1, 11))).toTensorListRef(),
                 (std::move(peek(stack, 2, 11))).toTensorListRef(),
                 (std::move(peek(stack, 3, 11))).toBool(),
                 (std::move(peek(stack, 4, 11))).toInt(),
                 (std::move(peek(stack, 5, 11))).toDouble(),
                 (std::move(peek(stack, 6, 11))).toBool(),
                 (std::move(peek(stack, 7, 11))).toBool(),
                 (std::move(peek(stack, 8, 11))).toBool(),
                 (std::move(peek(stack, 9, 11))).toOptional<ScalarType>(),
                 (std::move(peek(stack, 10, 11))).toBool()
             );
             drop(stack, 11);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::randint.out(int high, int[] size, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::randint_out(
                 out,
                 (std::move(peek(stack, 0, 3))).toInt(),
                 (std::move(peek(stack, 1, 3))).toIntListRef()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::randint.low_out(int low, int high, int[] size, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::randint_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toInt(),
                 (std::move(peek(stack, 1, 4))).toInt(),
                 (std::move(peek(stack, 2, 4))).toIntListRef()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::randn(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 1, 5))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 2, 5))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 3, 5))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 4, 5))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::randn((std::move(peek(stack, 0, 5))).toIntListRef(),
             options);
             #else
                 auto result_ = torch::randn((std::move(peek(stack, 0, 5))).toIntListRef(),
             options);
             #endif
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::randn_like(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::randn_like(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::randn_like.dtype(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 1, 5))).toScalarType())
                     .layout((std::move(peek(stack, 2, 5))).toLayout())
                     .device((std::move(peek(stack, 3, 5))).toDevice())
                     .pinned_memory((std::move(peek(stack, 4, 5))).toBool());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::randn_like((std::move(peek(stack, 0, 5))).toTensor(),
             options);
             #else
                 auto result_ = torch::randn_like((std::move(peek(stack, 0, 5))).toTensor(),
             options);
             #endif
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::randperm.out(int n, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::randperm_out(
                 out,
                 (std::move(peek(stack, 0, 2))).toInt()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::range(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 2, 6))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 3, 6))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 4, 6))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 5, 6))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::range((std::move(peek(stack, 0, 6))).toScalar(),
             (std::move(peek(stack, 1, 6))).toScalar(),
             options);
             #else
                 auto result_ = torch::range((std::move(peek(stack, 0, 6))).toScalar(),
             (std::move(peek(stack, 1, 6))).toScalar(),
             options);
             #endif
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::range.step(Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 3, 7))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 4, 7))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 5, 7))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 6, 7))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::range((std::move(peek(stack, 0, 7))).toScalar(),
             (std::move(peek(stack, 1, 7))).toScalar(),
             (std::move(peek(stack, 2, 7))).toScalar(),
             options);
             #else
                 auto result_ = torch::range((std::move(peek(stack, 0, 7))).toScalar(),
             (std::move(peek(stack, 1, 7))).toScalar(),
             (std::move(peek(stack, 2, 7))).toScalar(),
             options);
             #endif
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::reciprocal_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::reciprocal_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::reflection_pad1d(Tensor self, int[2] padding) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::reflection_pad1d(
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
         "aten::reflection_pad1d_backward.grad_input(Tensor grad_output, Tensor self, int[2] padding, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::reflection_pad1d_backward_out(
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
         "aten::reflection_pad2d_backward(Tensor grad_output, Tensor self, int[4] padding) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::reflection_pad2d_backward(
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
         "aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::remainder(
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
         "aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::remainder(
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
         "aten::renorm.out(Tensor self, Scalar p, int dim, Scalar maxnorm, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 4, 5))).toTensor();
             auto result_ = at::renorm_out(
                 out,
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toScalar(),
                 (std::move(peek(stack, 2, 5))).toInt(),
                 (std::move(peek(stack, 3, 5))).toScalar()
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::replication_pad1d(Tensor self, int[2] padding) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::replication_pad1d(
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
         "aten::replication_pad1d_backward.grad_input(Tensor grad_output, Tensor self, int[2] padding, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::replication_pad1d_backward_out(
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
         "aten::replication_pad2d_backward(Tensor grad_output, Tensor self, int[4] padding) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::replication_pad2d_backward(
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
         "aten::replication_pad3d.out(Tensor self, int[6] padding, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::replication_pad3d_out(
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
         "aten::reshape(Tensor self, int[] shape) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::reshape(
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
         "aten::resize_(Tensor(a!) self, int[] size) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).resize_(
                 (std::move(peek(stack, 1, 2))).toIntListRef()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::rnn_relu_cell(
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
         "aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::roll(
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
         "aten::round(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::round(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::rrelu_with_noise.out(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 6, 7))).toTensor();
             auto result_ = at::rrelu_with_noise_out(
                 out,
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::rsqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::rsqrt_out(
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
         "aten::rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::rsub(
                 (std::move(peek(stack, 0, 3))).toTensor(),
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
         "aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::rsub(
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
         "aten::scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 1, 5))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 2, 5))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 3, 5))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 4, 5))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::scalar_tensor((std::move(peek(stack, 0, 5))).toScalar(),
             options);
             #else
                 auto result_ = torch::scalar_tensor((std::move(peek(stack, 0, 5))).toScalar(),
             options);
             #endif
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 4))).toTensor();
             auto result_ = (self).scatter_add_(
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
         "aten::selu(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::selu(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::sigmoid_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::sigmoid_backward(
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
         "aten::sign_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = (self).sign_(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sin(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::sin(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::sinh_out(
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
         "aten::sizes(Tensor self) -> int",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).sizes(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::slice.Tensor(Tensor(a) self, int dim=0, int start=0, int end=9223372036854775807, int step=1) -> Tensor(a)",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::slow_conv_dilated3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::slow_conv_dilated3d(
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
         "aten::slow_conv_transpose2d.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int[2] dilation=1, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 8, 9))).toTensor();
             auto result_ = at::slow_conv_transpose2d_out(
                 out,
                 (std::move(peek(stack, 0, 9))).toTensor(),
                 (std::move(peek(stack, 1, 9))).toTensor(),
                 (std::move(peek(stack, 2, 9))).toIntListRef(),
                 toOptionalTensor((std::move(peek(stack, 3, 9)))),
                 (std::move(peek(stack, 4, 9))).toIntListRef(),
                 (std::move(peek(stack, 5, 9))).toIntListRef(),
                 (std::move(peek(stack, 6, 9))).toIntListRef(),
                 (std::move(peek(stack, 7, 9))).toIntListRef()
             );
             drop(stack, 9);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::slow_conv_transpose3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int[3] dilation=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::slow_conv_transpose3d(
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
         "aten::smm(Tensor self, Tensor mat2) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::smm(
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
         "aten::smooth_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::smooth_l1_loss_backward(
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
         "aten::soft_margin_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::soft_margin_loss_out(
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
         "aten::softplus.out(Tensor self, Scalar beta=1, Scalar threshold=20, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::softplus_out(
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
         "aten::softshrink_backward(Tensor grad_output, Tensor self, Scalar lambd) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::softshrink_backward(
                 (std::move(peek(stack, 0, 3))).toTensor(),
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
         "aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)",
         [](Stack & stack) {
         
             auto result_ = at::sort(
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
         "aten::sqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::sqrt_out(
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
         "aten::sspaddmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 5, 6))).toTensor();
             auto result_ = at::sspaddmm_out(
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
         "aten::stack(Tensor[] tensors, int dim=0) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::stack(
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
         "aten::std_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::std_mean(
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
         "aten::std_mean.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::std_mean(
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
         "aten::stride.int(Tensor self, int dim) -> int",
         [](Stack & stack) {
         
             auto result_ = at::stride(
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
         "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::sub(
                 (std::move(peek(stack, 0, 3))).toTensor(),
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
         "aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::sub(
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
         "aten::sum.IntList_out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 4, 5))).toTensor();
             auto result_ = at::sum_out(
                 out,
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toIntListRef(),
                 (std::move(peek(stack, 2, 5))).toBool(),
                 (std::move(peek(stack, 3, 5))).toOptional<ScalarType>()
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::svd(Tensor self, bool some=True, bool compute_uv=True) -> (Tensor U, Tensor S, Tensor V)",
         [](Stack & stack) {
         
             auto result_ = at::svd(
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
         "aten::symeig(Tensor self, bool eigenvectors=False, bool upper=True) -> (Tensor eigenvalues, Tensor eigenvectors)",
         [](Stack & stack) {
         
             auto result_ = at::symeig(
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
         "aten::tan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::tan_out(
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
         "aten::tanh_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::tanh_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::tanh_backward(Tensor grad_output, Tensor output) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::tanh_backward(
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
         "aten::thnn_conv2d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, Tensor finput, Tensor fgrad_input, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)",
         [](Stack & stack) {
         
             auto result_ = at::thnn_conv2d_backward(
                 (std::move(peek(stack, 0, 9))).toTensor(),
                 (std::move(peek(stack, 1, 9))).toTensor(),
                 (std::move(peek(stack, 2, 9))).toTensor(),
                 (std::move(peek(stack, 3, 9))).toIntListRef(),
                 (std::move(peek(stack, 4, 9))).toIntListRef(),
                 (std::move(peek(stack, 5, 9))).toIntListRef(),
                 (std::move(peek(stack, 6, 9))).toTensor(),
                 (std::move(peek(stack, 7, 9))).toTensor(),
                 as_bool_array<3>((std::move(peek(stack, 8, 9))).toBoolList())
             );
             drop(stack, 9);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::thnn_conv3d.out(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 6, 7))).toTensor();
             auto result_ = at::thnn_conv3d_out(
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
         "aten::thnn_conv_depthwise2d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool[2] output_mask) -> (Tensor grad_input, Tensor grad_weight)",
         [](Stack & stack) {
         
             auto result_ = at::thnn_conv_depthwise2d_backward(
                 (std::move(peek(stack, 0, 8))).toTensor(),
                 (std::move(peek(stack, 1, 8))).toTensor(),
                 (std::move(peek(stack, 2, 8))).toTensor(),
                 (std::move(peek(stack, 3, 8))).toIntListRef(),
                 (std::move(peek(stack, 4, 8))).toIntListRef(),
                 (std::move(peek(stack, 5, 8))).toIntListRef(),
                 (std::move(peek(stack, 6, 8))).toIntListRef(),
                 as_bool_array<2>((std::move(peek(stack, 7, 8))).toBoolList())
             );
             drop(stack, 8);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::threshold_(Tensor(a!) self, Scalar threshold, Scalar value) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = at::threshold_(
                 self,
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
         "aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::threshold_backward(
                 (std::move(peek(stack, 0, 3))).toTensor(),
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
         "aten::to_dense(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).to_dense(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::to_mkldnn(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).to_mkldnn(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = (self).transpose_(
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
         "aten::trapz.x(Tensor y, Tensor x, *, int dim=-1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::trapz(
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
         "aten::trapz.dx(Tensor y, *, float dx=1, int dim=-1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::trapz(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toDouble(),
                 (std::move(peek(stack, 2, 3))).toInt()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::tril.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::tril_out(
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
         "aten::triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, float margin=1.0, float p=2, float eps=1e-06, bool swap=False, int reduction=Mean) -> Tensor",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::triu.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::triu_out(
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
         "aten::trunc_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::trunc_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 4))).toTensor()).unfold(
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
         "aten::unique_consecutive(Tensor self, bool return_inverse=False, bool return_counts=False, int? dim=None) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::unique_consecutive(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toBool(),
                 (std::move(peek(stack, 2, 4))).toBool(),
                 (std::move(peek(stack, 3, 4))).toOptional<int64_t>()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::unsqueeze_(Tensor(a!) self, int dim) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).unsqueeze_(
                 (std::move(peek(stack, 1, 2))).toInt()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::upsample_bicubic2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::upsample_bicubic2d_backward(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toIntListRef(),
                 (std::move(peek(stack, 2, 4))).toIntListRef(),
                 (std::move(peek(stack, 3, 4))).toBool()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::upsample_bilinear2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::upsample_bilinear2d_backward(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toIntListRef(),
                 (std::move(peek(stack, 2, 4))).toIntListRef(),
                 (std::move(peek(stack, 3, 4))).toBool()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::upsample_linear1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size, bool align_corners) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::upsample_linear1d_backward(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toIntListRef(),
                 (std::move(peek(stack, 2, 4))).toIntListRef(),
                 (std::move(peek(stack, 3, 4))).toBool()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::upsample_nearest1d(Tensor self, int[1] output_size) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::upsample_nearest1d(
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
         "aten::upsample_nearest1d_backward.grad_input(Tensor grad_output, int[1] output_size, int[3] input_size, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::upsample_nearest1d_backward_out(
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
         "aten::upsample_nearest2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::upsample_nearest2d_backward(
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
         "aten::upsample_nearest3d.out(Tensor self, int[3] output_size, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::upsample_nearest3d_out(
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
         "aten::upsample_trilinear3d.out(Tensor self, int[3] output_size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::upsample_trilinear3d_out(
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
         "aten::var.out(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 4, 5))).toTensor();
             auto result_ = at::var_out(
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
         "aten::view_as(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 2))).toTensor()).view_as(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::zero_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::zero_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::zeros.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::zeros_out(
                 out,
                 (std::move(peek(stack, 0, 2))).toIntListRef()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),});

} // anon namespace


}} // namespace torch::jit
