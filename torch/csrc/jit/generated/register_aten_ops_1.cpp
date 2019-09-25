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
         "aten::__ilshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).__ilshift__(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::__ilshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).__ilshift__(
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::__irshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).__irshift__(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::__irshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).__irshift__(
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::__lshift__.Tensor(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::__lshift__(
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
         "aten::__lshift__.Scalar(Tensor self, Scalar other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::__lshift__(
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
         "aten::__rshift__.Tensor(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::__rshift__(
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
         "aten::__rshift__.Scalar(Tensor self, Scalar other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::__rshift__(
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
         "aten::_adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_adaptive_avg_pool2d(
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
         "aten::_addr_(Tensor(a!) self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 5))).toTensor();
             auto result_ = at::_addr_(
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
         "aten::_batch_norm_impl_index(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> (Tensor, Tensor, Tensor, int)",
         [](Stack & stack) {
         
             auto result_ = at::_batch_norm_impl_index(
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
         "aten::_cast_Byte(Tensor self, bool non_blocking=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_cast_Byte(
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
         "aten::_cast_Int(Tensor self, bool non_blocking=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_cast_Int(
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
         "aten::_cholesky_helper(Tensor self, bool upper) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_cholesky_helper(
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
         "aten::_cholesky_solve_helper(Tensor self, Tensor A, bool upper) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_cholesky_solve_helper(
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
         "aten::_copy_from(Tensor self, Tensor dst, bool non_blocking=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_copy_from(
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
         "aten::_ctc_loss_backward(Tensor grad, Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, Tensor neg_log_likelihood, Tensor log_alpha, int blank, bool zero_infinity=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_ctc_loss_backward(
                 (std::move(peek(stack, 0, 9))).toTensor(),
                 (std::move(peek(stack, 1, 9))).toTensor(),
                 (std::move(peek(stack, 2, 9))).toTensor(),
                 (std::move(peek(stack, 3, 9))).toIntListRef(),
                 (std::move(peek(stack, 4, 9))).toIntListRef(),
                 (std::move(peek(stack, 5, 9))).toTensor(),
                 (std::move(peek(stack, 6, 9))).toTensor(),
                 (std::move(peek(stack, 7, 9))).toInt(),
                 (std::move(peek(stack, 8, 9))).toBool()
             );
             drop(stack, 9);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_cudnn_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor? weight_buf, Tensor hx, Tensor? cx, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_cudnn_rnn(
                 (std::move(peek(stack, 0, 15))).toTensor(),
                 (std::move(peek(stack, 1, 15))).toTensorListRef(),
                 (std::move(peek(stack, 2, 15))).toInt(),
                 toOptionalTensor((std::move(peek(stack, 3, 15)))),
                 (std::move(peek(stack, 4, 15))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 5, 15)))),
                 (std::move(peek(stack, 6, 15))).toInt(),
                 (std::move(peek(stack, 7, 15))).toInt(),
                 (std::move(peek(stack, 8, 15))).toInt(),
                 (std::move(peek(stack, 9, 15))).toBool(),
                 (std::move(peek(stack, 10, 15))).toDouble(),
                 (std::move(peek(stack, 11, 15))).toBool(),
                 (std::move(peek(stack, 12, 15))).toBool(),
                 (std::move(peek(stack, 13, 15))).toIntListRef(),
                 toOptionalTensor((std::move(peek(stack, 14, 15))))
             );
             drop(stack, 15);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_cufft_get_plan_cache_size(int device_index) -> int",
         [](Stack & stack) {
         
             auto result_ = at::_cufft_get_plan_cache_size(
                 (std::move(peek(stack, 0, 1))).toInt()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_dim_arange(Tensor like, int dim) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_dim_arange(
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
         "aten::_embedding_bag_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, int num_weights, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_embedding_bag_backward(
                 (std::move(peek(stack, 0, 11))).toTensor(),
                 (std::move(peek(stack, 1, 11))).toTensor(),
                 (std::move(peek(stack, 2, 11))).toTensor(),
                 (std::move(peek(stack, 3, 11))).toTensor(),
                 (std::move(peek(stack, 4, 11))).toTensor(),
                 (std::move(peek(stack, 5, 11))).toTensor(),
                 (std::move(peek(stack, 6, 11))).toInt(),
                 (std::move(peek(stack, 7, 11))).toBool(),
                 (std::move(peek(stack, 8, 11))).toInt(),
                 (std::move(peek(stack, 9, 11))).toBool(),
                 toOptionalTensor((std::move(peek(stack, 10, 11))))
             );
             drop(stack, 11);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_empty_per_channel_affine_quantized(int[] size, *, Tensor scales, Tensor zero_points, int axis, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=contiguous_format) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 4, 9))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 5, 9))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 6, 9))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 7, 9))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::_empty_per_channel_affine_quantized((std::move(peek(stack, 0, 9))).toIntListRef(),
             (std::move(peek(stack, 1, 9))).toTensor(),
             (std::move(peek(stack, 2, 9))).toTensor(),
             (std::move(peek(stack, 3, 9))).toInt(),
             options,
             (std::move(peek(stack, 8, 9))).toOptional<c10::MemoryFormat>());
             #else
                 auto result_ = torch::_empty_per_channel_affine_quantized((std::move(peek(stack, 0, 9))).toIntListRef(),
             (std::move(peek(stack, 1, 9))).toTensor(),
             (std::move(peek(stack, 2, 9))).toTensor(),
             (std::move(peek(stack, 3, 9))).toInt(),
             options,
             (std::move(peek(stack, 8, 9))).toOptional<c10::MemoryFormat>());
             #endif
             drop(stack, 9);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_fft_with_size(Tensor self, int signal_ndim, bool complex_input, bool complex_output, bool inverse, int[] checked_signal_sizes, bool normalized, bool onesided, int[] output_sizes) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_fft_with_size(
                 (std::move(peek(stack, 0, 9))).toTensor(),
                 (std::move(peek(stack, 1, 9))).toInt(),
                 (std::move(peek(stack, 2, 9))).toBool(),
                 (std::move(peek(stack, 3, 9))).toBool(),
                 (std::move(peek(stack, 4, 9))).toBool(),
                 (std::move(peek(stack, 5, 9))).toIntListRef(),
                 (std::move(peek(stack, 6, 9))).toBool(),
                 (std::move(peek(stack, 7, 9))).toBool(),
                 (std::move(peek(stack, 8, 9))).toIntListRef()
             );
             drop(stack, 9);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_has_compatible_shallow_copy_type(Tensor self, Tensor from) -> bool",
         [](Stack & stack) {
         
             auto result_ = at::_has_compatible_shallow_copy_type(
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
         "aten::_index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 4))).toTensor();
             auto result_ = at::_index_copy_(
                 self,
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
         "aten::_inverse_helper(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_inverse_helper(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_log_softmax(
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
         "aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_log_softmax_backward_data(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 (std::move(peek(stack, 2, 4))).toInt(),
                 (std::move(peek(stack, 3, 4))).toTensor()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_masked_scale(Tensor self, Tensor mask, float scale) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_masked_scale(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor(),
                 (std::move(peek(stack, 2, 3))).toDouble()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_max(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_max(
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
         "aten::_mkldnn_transpose(Tensor self, int dim0, int dim1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_mkldnn_transpose(
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
         "aten::_multinomial_alias_setup(Tensor probs) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_multinomial_alias_setup(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_nnpack_spatial_convolution_backward_weight(Tensor input, int[] weightsize, Tensor grad_output, int[2] padding) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_nnpack_spatial_convolution_backward_weight(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toIntListRef(),
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
         "aten::_pack_padded_sequence(Tensor input, Tensor lengths, bool batch_first) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_pack_padded_sequence(
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
         "aten::_pdist_forward(Tensor self, float p=2) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_pdist_forward(
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
         "aten::_qr_helper(Tensor self, bool some) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_qr_helper(
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
         "aten::_s_where(Tensor condition, Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_s_where(
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
         "aten::_shape_as_tensor(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_shape_as_tensor(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_sobol_engine_draw(Tensor quasi, int n, Tensor sobolstate, int dimension, int num_generated, ScalarType? dtype) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_sobol_engine_draw(
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toInt(),
                 (std::move(peek(stack, 2, 6))).toTensor(),
                 (std::move(peek(stack, 3, 6))).toInt(),
                 (std::move(peek(stack, 4, 6))).toInt(),
                 (std::move(peek(stack, 5, 6))).toOptional<ScalarType>()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_softmax(
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
         "aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_softmax_backward_data(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 (std::move(peek(stack, 2, 4))).toInt(),
                 (std::move(peek(stack, 3, 4))).toTensor()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_thnn_fused_gru_cell_backward(Tensor grad_hy, Tensor workspace, bool has_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_thnn_fused_gru_cell_backward(
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
         "aten::_thnn_fused_lstm_cell_backward(Tensor? grad_hy, Tensor? grad_cy, Tensor cx, Tensor cy, Tensor workspace, bool has_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_thnn_fused_lstm_cell_backward(
                 toOptionalTensor((std::move(peek(stack, 0, 6)))),
                 toOptionalTensor((std::move(peek(stack, 1, 6)))),
                 (std::move(peek(stack, 2, 6))).toTensor(),
                 (std::move(peek(stack, 3, 6))).toTensor(),
                 (std::move(peek(stack, 4, 6))).toTensor(),
                 (std::move(peek(stack, 5, 6))).toBool()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_unique2(Tensor self, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_unique2(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toBool(),
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
         "aten::_values(Tensor(a) self) -> Tensor(a)",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor())._values(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::_var(Tensor self, bool unbiased=True) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_var(
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
         "aten::_weight_norm(Tensor v, Tensor g, int dim=0) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::_weight_norm(
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
         "aten::_weight_norm_cuda_interface(Tensor v, Tensor g, int dim=0) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::_weight_norm_cuda_interface(
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
         "aten::_weight_norm_differentiable_backward(Tensor grad_w, Tensor saved_v, Tensor saved_g, Tensor saved_norms, int dim) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::abs(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::abs(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::acos_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::acos_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::adaptive_avg_pool1d(Tensor self, int[1] output_size) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::adaptive_avg_pool1d(
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
         "aten::adaptive_avg_pool3d.out(Tensor self, int[3] output_size, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::adaptive_avg_pool3d_out(
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
         "aten::adaptive_max_pool2d(Tensor self, int[2] output_size) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::adaptive_max_pool2d(
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
         "aten::adaptive_max_pool2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::adaptive_max_pool3d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::adaptive_max_pool3d_backward(
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
         "aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::add_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 (std::move(peek(stack, 2, 4))).toScalar()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::addcdiv_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 4))).toTensor();
             auto result_ = (self).addcdiv_(
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
         "aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::addcmul(
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
         "aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::addr.out(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 5, 6))).toTensor();
             auto result_ = at::addr_out(
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
         "aten::all(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::all(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::all(
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
         "aten::allclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> bool",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = at::alpha_dropout_(
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
         "aten::any(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::any(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::any(
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
         "aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 1, 5))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 2, 5))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 3, 5))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 4, 5))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::arange((std::move(peek(stack, 0, 5))).toScalar(),
             options);
             #else
                 auto result_ = torch::arange((std::move(peek(stack, 0, 5))).toScalar(),
             options);
             #endif
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 2, 6))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 3, 6))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 4, 6))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 5, 6))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::arange((std::move(peek(stack, 0, 6))).toScalar(),
             (std::move(peek(stack, 1, 6))).toScalar(),
             options);
             #else
                 auto result_ = torch::arange((std::move(peek(stack, 0, 6))).toScalar(),
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
         "aten::arange.start_step(Scalar start, Scalar end, Scalar step, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 3, 7))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 4, 7))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 5, 7))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 6, 7))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::arange((std::move(peek(stack, 0, 7))).toScalar(),
             (std::move(peek(stack, 1, 7))).toScalar(),
             (std::move(peek(stack, 2, 7))).toScalar(),
             options);
             #else
                 auto result_ = torch::arange((std::move(peek(stack, 0, 7))).toScalar(),
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
         "aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::argmax(
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
         "aten::as_strided_(Tensor(a!) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 4))).toTensor();
             auto result_ = at::as_strided_(
                 self,
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
         "aten::asin(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::asin(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::atan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).atan2_(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::atan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::atan_out(
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
         "aten::avg_pool2d.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 7, 8))).toTensor();
             auto result_ = at::avg_pool2d_out(
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
         "aten::avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::avg_pool3d(
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
         "aten::avg_pool3d_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 8, 9))).toTensor();
             auto result_ = at::avg_pool3d_backward_out(
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
         "aten::baddbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 5, 6))).toTensor();
             auto result_ = at::baddbmm_out(
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
         "aten::batch_norm_gather_stats(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, int count) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::batch_norm_gather_stats(
                 (std::move(peek(stack, 0, 8))).toTensor(),
                 (std::move(peek(stack, 1, 8))).toTensor(),
                 (std::move(peek(stack, 2, 8))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 3, 8)))),
                 toOptionalTensor((std::move(peek(stack, 4, 8)))),
                 (std::move(peek(stack, 5, 8))).toDouble(),
                 (std::move(peek(stack, 6, 8))).toDouble(),
                 (std::move(peek(stack, 7, 8))).toInt()
             );
             drop(stack, 8);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::batch_norm_gather_stats_with_counts(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, int[] counts) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::batch_norm_gather_stats_with_counts(
                 (std::move(peek(stack, 0, 8))).toTensor(),
                 (std::move(peek(stack, 1, 8))).toTensor(),
                 (std::move(peek(stack, 2, 8))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 3, 8)))),
                 toOptionalTensor((std::move(peek(stack, 4, 8)))),
                 (std::move(peek(stack, 5, 8))).toDouble(),
                 (std::move(peek(stack, 6, 8))).toDouble(),
                 (std::move(peek(stack, 7, 8))).toIntListRef()
             );
             drop(stack, 8);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::bernoulli.out(Tensor self, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::bernoulli_out(
                 out,
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 nullptr
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::bilinear(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 (std::move(peek(stack, 2, 4))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 3, 4))))
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::binary_cross_entropy_backward(
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toTensor(),
                 (std::move(peek(stack, 2, 5))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 3, 5)))),
                 (std::move(peek(stack, 4, 5))).toInt()
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::binary_cross_entropy_with_logits(Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=Mean) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::binary_cross_entropy_with_logits(
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 2, 5)))),
                 toOptionalTensor((std::move(peek(stack, 3, 5)))),
                 (std::move(peek(stack, 4, 5))).toInt()
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::bitwise_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::bitwise_not_out(
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
         "aten::bmm(Tensor self, Tensor mat2) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::bmm(
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
         "aten::cartesian_prod(Tensor[] tensors) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::cartesian_prod(
                 (std::move(peek(stack, 0, 1))).toTensorListRef()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::cat_out(
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
         "aten::cdist(Tensor x1, Tensor x2, float p=2) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::cdist(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor(),
                 (std::move(peek(stack, 2, 3))).toDouble()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::ceil_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::ceil_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::celu_(Tensor(a!) self, Scalar alpha=1.0) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = at::celu_(
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
         "aten::cholesky_inverse.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::cholesky_inverse_out(
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
         "aten::clamp_max(Tensor self, Scalar max) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::clamp_max(
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
         "aten::clamp_min_(Tensor(a!) self, Scalar min) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = at::clamp_min_(
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
         "aten::clamp.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::clamp_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toOptional<Scalar>(),
                 (std::move(peek(stack, 2, 4))).toOptional<Scalar>()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::clone(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::clone(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::col2im.out(Tensor self, int[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 6, 7))).toTensor();
             auto result_ = at::col2im_out(
                 out,
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
         "aten::conv3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::conv3d(
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
         "aten::conv_transpose3d.input(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int groups=1, int[3] dilation=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::conv_transpose3d(
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
         "aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::convolution(
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
         "aten::convolution_backward_overrideable(Tensor grad_output, Tensor input, Tensor weight, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)",
         [](Stack & stack) {
         
             auto result_ = at::convolution_backward_overrideable(
                 (std::move(peek(stack, 0, 10))).toTensor(),
                 (std::move(peek(stack, 1, 10))).toTensor(),
                 (std::move(peek(stack, 2, 10))).toTensor(),
                 (std::move(peek(stack, 3, 10))).toIntListRef(),
                 (std::move(peek(stack, 4, 10))).toIntListRef(),
                 (std::move(peek(stack, 5, 10))).toIntListRef(),
                 (std::move(peek(stack, 6, 10))).toBool(),
                 (std::move(peek(stack, 7, 10))).toIntListRef(),
                 (std::move(peek(stack, 8, 10))).toInt(),
                 as_bool_array<3>((std::move(peek(stack, 9, 10))).toBoolList())
             );
             drop(stack, 10);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = (self).copy_(
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
         "aten::cos(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::cos(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::cosh_out(
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
         "aten::cosine_similarity(Tensor x1, Tensor x2, int dim=1, float eps=1e-08) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::cosine_similarity(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 (std::move(peek(stack, 2, 4))).toInt(),
                 (std::move(peek(stack, 3, 4))).toDouble()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cudnn_affine_grid_generator(Tensor theta, int N, int C, int H, int W) -> Tensor grid",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cudnn_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::cudnn_batch_norm(
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
         "aten::cudnn_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::cudnn_convolution_backward(
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
         "aten::cudnn_convolution_backward_bias(Tensor grad_output) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::cudnn_convolution_backward_bias(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cudnn_convolution_transpose_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::cudnn_convolution_transpose_backward(
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
         "aten::cudnn_convolution_transpose_backward_bias(Tensor grad_output) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::cudnn_convolution_transpose_backward_bias(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cudnn_grid_sampler_backward(Tensor self, Tensor grid, Tensor grad_output) -> (Tensor grad_self, Tensor grad_grid)",
         [](Stack & stack) {
         
             auto result_ = at::cudnn_grid_sampler_backward(
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
         "aten::cumprod.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::cumprod_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toInt(),
                 (std::move(peek(stack, 2, 4))).toOptional<ScalarType>()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::cumsum.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::cumsum_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toInt(),
                 (std::move(peek(stack, 2, 4))).toOptional<ScalarType>()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::detach_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::detach_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::diag_embed(
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
         "aten::diag.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::diag_out(
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
         "aten::digamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::digamma_out(
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
         "aten::dist(Tensor self, Tensor other, Scalar p=2) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::dist(
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
         "aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).div_(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).div_(
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::dot.out(Tensor self, Tensor tensor, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::dot_out(
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
         "aten::dropout(Tensor input, float p, bool train) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::dropout(
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
         "aten::elu_(Tensor(a!) self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::elu_backward(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, Tensor output) -> Tensor",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::embedding_sparse_backward(Tensor grad, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 1, 6))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 2, 6))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 3, 6))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 4, 6))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::empty((std::move(peek(stack, 0, 6))).toIntListRef(),
             options,
             (std::move(peek(stack, 5, 6))).toOptional<c10::MemoryFormat>());
             #else
                 auto result_ = torch::empty((std::move(peek(stack, 0, 6))).toIntListRef(),
             options,
             (std::move(peek(stack, 5, 6))).toOptional<c10::MemoryFormat>());
             #endif
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::empty_like(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::empty_like(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::empty_like.dtype(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, MemoryFormat? memory_format=contiguous_format) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 1, 6))).toScalarType())
                     .layout((std::move(peek(stack, 2, 6))).toLayout())
                     .device((std::move(peek(stack, 3, 6))).toDevice())
                     .pinned_memory((std::move(peek(stack, 4, 6))).toBool());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::empty_like((std::move(peek(stack, 0, 6))).toTensor(),
             options,
             (std::move(peek(stack, 5, 6))).toOptional<c10::MemoryFormat>());
             #else
                 auto result_ = torch::empty_like((std::move(peek(stack, 0, 6))).toTensor(),
             options,
             (std::move(peek(stack, 5, 6))).toOptional<c10::MemoryFormat>());
             #endif
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::empty_strided(int[] size, int[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 2, 6))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 3, 6))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 4, 6))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 5, 6))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::empty_strided((std::move(peek(stack, 0, 6))).toIntListRef(),
             (std::move(peek(stack, 1, 6))).toIntListRef(),
             options);
             #else
                 auto result_ = torch::empty_strided((std::move(peek(stack, 0, 6))).toIntListRef(),
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
         "aten::eq.Tensor(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::eq(
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
         "aten::eq.Scalar(Tensor self, Scalar other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::eq(
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
         "aten::erf_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::erf_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::erfc_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::erfc_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::erfinv_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = (self).erfinv_(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::exp_out(
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
         "aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 3))).toTensor()).expand(
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
         "aten::expm1_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::expm1_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::fbgemm_linear_int8_weight_fp32_activation(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::fbgemm_linear_int8_weight_fp32_activation(
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
         "aten::feature_alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = at::feature_alpha_dropout_(
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
         "aten::feature_dropout(Tensor input, float p, bool train) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::feature_dropout(
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
         "aten::fill_diagonal_(Tensor(a!) self, Scalar fill_value, bool wrap=False) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = (self).fill_diagonal_(
                 (std::move(peek(stack, 1, 3))).toScalar(),
                 (std::move(peek(stack, 2, 3))).toBool()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::flip(Tensor self, int[] dims) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::flip(
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
         "aten::floor.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::floor_out(
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
         "aten::fmod_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).fmod_(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::fmod_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).fmod_(
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::frac(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::frac(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::fractional_max_pool2d(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::fractional_max_pool2d(
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
         "aten::fractional_max_pool2d_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] output_size, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 5, 6))).toTensor();
             auto result_ = at::fractional_max_pool2d_backward_out(
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
         "aten::fractional_max_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] output_size, Tensor indices) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::fractional_max_pool3d_backward(
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
         "aten::frobenius_norm.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::frobenius_norm_out(
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
         "aten::full.out(int[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::full_out(
                 out,
                 (std::move(peek(stack, 0, 3))).toIntListRef(),
                 (std::move(peek(stack, 1, 3))).toScalar()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::ge.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::ge_out(
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
         "aten::ge.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::ge_out(
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
         "aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = (self).geometric_(
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
         "aten::ger.out(Tensor self, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::ger_out(
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
         "aten::glu(Tensor self, int dim=-1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::glu(
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
         "aten::glu_backward.grad_input(Tensor grad_output, Tensor self, int dim, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::grid_sampler(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::grid_sampler(
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
         "aten::grid_sampler_3d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::grid_sampler_3d(
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
         "aten::group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::group_norm(
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toInt(),
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
         "aten::gru.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::gru(
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
         "aten::gru.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::gru(
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
         "aten::gt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::gt_out(
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
         "aten::gt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::gt_out(
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
         "aten::hann_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 1, 5))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 2, 5))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 3, 5))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 4, 5))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::hann_window((std::move(peek(stack, 0, 5))).toInt(),
             options);
             #else
                 auto result_ = torch::hann_window((std::move(peek(stack, 0, 5))).toInt(),
             options);
             #endif
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::hann_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 2, 6))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 3, 6))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 4, 6))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 5, 6))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::hann_window((std::move(peek(stack, 0, 6))).toInt(),
             (std::move(peek(stack, 1, 6))).toBool(),
             options);
             #else
                 auto result_ = torch::hann_window((std::move(peek(stack, 0, 6))).toInt(),
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
         "aten::hardshrink_backward(Tensor grad_out, Tensor self, Scalar lambd) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::hardshrink_backward(
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
         "aten::hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = at::hardtanh_(
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
         "aten::hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::hardtanh_backward(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
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
         "aten::hinge_embedding_loss(Tensor self, Tensor target, float margin=1.0, int reduction=Mean) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::hinge_embedding_loss(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 (std::move(peek(stack, 2, 4))).toDouble(),
                 (std::move(peek(stack, 3, 4))).toInt()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::hspmm.out(Tensor mat1, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::hspmm_out(
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
         "aten::im2col.out(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 5, 6))).toTensor();
             auto result_ = at::im2col_out(
                 out,
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
         "aten::index_add(Tensor self, int dim, Tensor index, Tensor source) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::index_add(
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
         "aten::index_fill.Tensor(Tensor self, int dim, Tensor index, Tensor value) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::index_fill(
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
         "aten::index_fill.Scalar(Tensor self, int dim, Tensor index, Scalar value) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::index_fill(
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
         "aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::index_put(
                 (std::move(peek(stack, 0, 4))).toTensor(),
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
         "aten::index_put(Tensor self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::index_put(
                 (std::move(peek(stack, 0, 4))).toTensor(),
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
         "aten::is_coalesced(Tensor self) -> bool",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).is_coalesced(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::is_complex(Tensor self) -> bool",
         [](Stack & stack) {
         
             auto result_ = at::is_complex(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::is_floating_point(Tensor self) -> bool",
         [](Stack & stack) {
         
             auto result_ = at::is_floating_point(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::is_signed(Tensor self) -> bool",
         [](Stack & stack) {
         
             auto result_ = at::is_signed(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::isclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> Tensor",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::kthvalue(Tensor self, int k, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)",
         [](Stack & stack) {
         
             auto result_ = at::kthvalue(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toInt(),
                 (std::move(peek(stack, 2, 4))).toInt(),
                 (std::move(peek(stack, 3, 4))).toBool()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::l1_loss_backward(
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
         "aten::le_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).le_(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::le_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).le_(
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::leaky_relu(
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
         "aten::leaky_relu_backward.grad_input(Tensor grad_output, Tensor self, Scalar negative_slope, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::lerp.Tensor_out(Tensor self, Tensor end, Tensor weight, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::lerp_out(
                 out,
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
         "aten::lerp.Scalar_out(Tensor self, Tensor end, Scalar weight, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::lerp_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 (std::move(peek(stack, 2, 4))).toScalar()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::lgamma_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = (self).lgamma_(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::linspace(Scalar start, Scalar end, int steps=100, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 3, 7))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 4, 7))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 5, 7))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 6, 7))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::linspace((std::move(peek(stack, 0, 7))).toScalar(),
             (std::move(peek(stack, 1, 7))).toScalar(),
             (std::move(peek(stack, 2, 7))).toInt(),
             options);
             #else
                 auto result_ = torch::linspace((std::move(peek(stack, 0, 7))).toScalar(),
             (std::move(peek(stack, 1, 7))).toScalar(),
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
         "aten::log(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::log(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::log10_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::log10_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::log1p_out(
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
         "aten::log2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::log2_out(
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
         "aten::log_sigmoid(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::log_sigmoid(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::log_sigmoid_backward.grad_input(Tensor grad_output, Tensor self, Tensor buffer, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::log_sigmoid_forward(Tensor self) -> (Tensor output, Tensor buffer)",
         [](Stack & stack) {
         
             auto result_ = at::log_sigmoid_forward(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::logical_not_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = (self).logical_not_(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::logical_xor(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::logical_xor(
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
         "aten::logspace.out(Scalar start, Scalar end, int steps=100, float base=10.0, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 4, 5))).toTensor();
             auto result_ = at::logspace_out(
                 out,
                 (std::move(peek(stack, 0, 5))).toScalar(),
                 (std::move(peek(stack, 1, 5))).toScalar(),
                 (std::move(peek(stack, 2, 5))).toInt(),
                 (std::move(peek(stack, 3, 5))).toDouble()
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::logsumexp.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::logsumexp_out(
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
         "aten::lstm.data(Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::lstm(
                 (std::move(peek(stack, 0, 9))).toTensor(),
                 (std::move(peek(stack, 1, 9))).toTensor(),
                 (std::move(peek(stack, 2, 9))).toTensorListRef(),
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
         "aten::lstm.input(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::lstm(
                 (std::move(peek(stack, 0, 9))).toTensor(),
                 (std::move(peek(stack, 1, 9))).toTensorListRef(),
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
         "aten::lt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).lt_(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::lt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).lt_(
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::lu_solve.out(Tensor self, Tensor LU_data, Tensor LU_pivots, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::lu_solve_out(
                 out,
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
         "aten::margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::masked_fill(
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
         "aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::masked_fill(
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
         "aten::max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::max_pool1d(
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
         "aten::max_pool3d_with_indices(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::max_pool3d_with_indices(
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
         "aten::max_pool3d_with_indices_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 8, 9))).toTensor();
             auto result_ = at::max_pool3d_with_indices_backward_out(
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
         "aten::max_unpool2d(Tensor self, Tensor indices, int[2] output_size) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::max_unpool2d(
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
         "aten::max_unpool2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, int[2] output_size, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
             auto result_ = at::max_unpool2d_backward_out(
                 grad_input,
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toTensor(),
                 (std::move(peek(stack, 2, 5))).toTensor(),
                 (std::move(peek(stack, 3, 5))).toIntListRef()
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::max_unpool3d_backward(Tensor grad_output, Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::max_unpool3d_backward(
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toTensor(),
                 (std::move(peek(stack, 2, 6))).toTensor(),
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
         "aten::mean.out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 4, 5))).toTensor();
             auto result_ = at::mean_out(
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
         "aten::median(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::median(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::median.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)",
         [](Stack & stack) {
         
             auto result_ = at::median(
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
         "aten::min.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::min_out(
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
         "aten::min_values(Tensor self, int[1] dim, bool keepdim=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::min_values(
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
         "aten::miopen_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::miopen_batch_norm_backward(
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
         "aten::miopen_convolution_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::miopen_convolution_backward_weight(
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
         "aten::miopen_convolution_transpose_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::miopen_convolution_transpose_backward_weight(
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
         "aten::miopen_depthwise_convolution_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::miopen_depthwise_convolution_backward_weight(
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
         "aten::mkldnn_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::mkldnn_convolution_backward(
                 (std::move(peek(stack, 0, 8))).toTensor(),
                 (std::move(peek(stack, 1, 8))).toTensor(),
                 (std::move(peek(stack, 2, 8))).toTensor(),
                 (std::move(peek(stack, 3, 8))).toIntListRef(),
                 (std::move(peek(stack, 4, 8))).toIntListRef(),
                 (std::move(peek(stack, 5, 8))).toIntListRef(),
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
         "aten::mkldnn_convolution_backward_weights(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool bias_defined) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::mkldnn_convolution_backward_weights(
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
         "aten::mkldnn_reorder_conv2d_weight(Tensor self, int[2] padding=0, int[2] stride=1, int[2] dilation=1, int groups=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::mkldnn_reorder_conv2d_weight(
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toIntListRef(),
                 (std::move(peek(stack, 2, 5))).toIntListRef(),
                 (std::move(peek(stack, 3, 5))).toIntListRef(),
                 (std::move(peek(stack, 4, 5))).toInt()
             );
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::mode(Tensor self, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)",
         [](Stack & stack) {
         
             auto result_ = at::mode(
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
         "aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::mse_loss_backward(
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
         "aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::mul(
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
         "aten::mul.Scalar(Tensor self, Scalar other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::mul(
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
         "aten::multi_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight=None, int reduction=Mean) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::multi_margin_loss_backward(
                 (std::move(peek(stack, 0, 7))).toTensor(),
                 (std::move(peek(stack, 1, 7))).toTensor(),
                 (std::move(peek(stack, 2, 7))).toTensor(),
                 (std::move(peek(stack, 3, 7))).toScalar(),
                 (std::move(peek(stack, 4, 7))).toScalar(),
                 toOptionalTensor((std::move(peek(stack, 5, 7)))),
                 (std::move(peek(stack, 6, 7))).toInt()
             );
             drop(stack, 7);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::multilabel_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::multilabel_margin_loss(
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
         "aten::multilabel_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::multilabel_margin_loss_forward(Tensor self, Tensor target, int reduction) -> (Tensor output, Tensor is_target)",
         [](Stack & stack) {
         
             auto result_ = at::multilabel_margin_loss_forward(
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
         "aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::multinomial(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toInt(),
                 (std::move(peek(stack, 2, 4))).toBool(),
                 nullptr
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::mvlgamma(Tensor self, int p) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::mvlgamma(
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
         "aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::native_batch_norm(
                 (std::move(peek(stack, 0, 8))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 1, 8)))),
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
         "aten::native_layer_norm(Tensor input, Tensor? weight, Tensor? bias, int M, int N, float eps) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::native_layer_norm(
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 1, 6)))),
                 toOptionalTensor((std::move(peek(stack, 2, 6)))),
                 (std::move(peek(stack, 3, 6))).toInt(),
                 (std::move(peek(stack, 4, 6))).toInt(),
                 (std::move(peek(stack, 5, 6))).toDouble()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::native_layer_norm_double_backward(Tensor? ggI, Tensor? ggW, Tensor? ggb, Tensor gO, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, int M, int N, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::native_layer_norm_double_backward(
                 toOptionalTensor((std::move(peek(stack, 0, 11)))),
                 toOptionalTensor((std::move(peek(stack, 1, 11)))),
                 toOptionalTensor((std::move(peek(stack, 2, 11)))),
                 (std::move(peek(stack, 3, 11))).toTensor(),
                 (std::move(peek(stack, 4, 11))).toTensor(),
                 (std::move(peek(stack, 5, 11))).toTensor(),
                 (std::move(peek(stack, 6, 11))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 7, 11)))),
                 (std::move(peek(stack, 8, 11))).toInt(),
                 (std::move(peek(stack, 9, 11))).toInt(),
                 as_bool_array<3>((std::move(peek(stack, 10, 11))).toBoolList())
             );
             drop(stack, 11);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::native_norm(Tensor self, Scalar p=2) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::native_norm(
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
         "aten::ne.Tensor(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::ne(
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
         "aten::ne.Scalar(Tensor self, Scalar other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::ne(
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
         "aten::neg_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::neg_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::nll_loss2d.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 5, 6))).toTensor();
             auto result_ = at::nll_loss2d_out(
                 out,
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 2, 6)))),
                 (std::move(peek(stack, 3, 6))).toInt(),
                 (std::move(peek(stack, 4, 6))).toInt()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::nll_loss.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 5, 6))).toTensor();
             auto result_ = at::nll_loss_out(
                 out,
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toTensor(),
                 toOptionalTensor((std::move(peek(stack, 2, 6)))),
                 (std::move(peek(stack, 3, 6))).toInt(),
                 (std::move(peek(stack, 4, 6))).toInt()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::norm_except_dim(Tensor v, int pow=2, int dim=0) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::norm_except_dim(
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
         "aten::norm.out(Tensor self, Scalar? p, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 4, 5))).toTensor();
             auto result_ = at::norm_out(
                 out,
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toOptional<Scalar>(),
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
         "aten::norm.dtype_out(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 5, 6))).toTensor();
             auto result_ = at::norm_out(
                 out,
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toOptional<Scalar>(),
                 (std::move(peek(stack, 2, 6))).toIntListRef(),
                 (std::move(peek(stack, 3, 6))).toBool(),
                 (std::move(peek(stack, 4, 6))).toScalarType()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::normal.Tensor_Tensor(Tensor mean, Tensor std, *, Generator? generator=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::normal(
                 (std::move(peek(stack, 0, 3))).toTensor(),
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
         "aten::normal.float_Tensor(float mean, Tensor std, *, Generator? generator=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::normal(
                 (std::move(peek(stack, 0, 3))).toDouble(),
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
         "aten::normal.Tensor_float(Tensor mean, float std=1, *, Generator? generator=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::normal(
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
         "aten::normal.float_float(float mean, float std, int[] size, *, Generator? generator=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 4, 8))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 5, 8))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 6, 8))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 7, 8))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::normal((std::move(peek(stack, 0, 8))).toDouble(),
             (std::move(peek(stack, 1, 8))).toDouble(),
             (std::move(peek(stack, 2, 8))).toIntListRef(),
             nullptr,
             options);
             #else
                 auto result_ = torch::normal((std::move(peek(stack, 0, 8))).toDouble(),
             (std::move(peek(stack, 1, 8))).toDouble(),
             (std::move(peek(stack, 2, 8))).toIntListRef(),
             nullptr,
             options);
             #endif
             drop(stack, 8);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::nuclear_norm(Tensor self, bool keepdim=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::nuclear_norm(
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
         "aten::nuclear_norm.dim(Tensor self, int[2] dim, bool keepdim=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::nuclear_norm(
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
         "aten::numpy_T(Tensor(a) self) -> Tensor(a)",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).numpy_T(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::one_hot(Tensor self, int num_classes=-1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::one_hot(
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
         "aten::orgqr.out(Tensor self, Tensor input2, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::orgqr_out(
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
         "aten::ormqr.out(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 5, 6))).toTensor();
             auto result_ = at::ormqr_out(
                 out,
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toTensor(),
                 (std::move(peek(stack, 2, 6))).toTensor(),
                 (std::move(peek(stack, 3, 6))).toBool(),
                 (std::move(peek(stack, 4, 6))).toBool()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::pin_memory(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).pin_memory(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::pixel_shuffle(Tensor self, int upscale_factor) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::pixel_shuffle(
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
         "aten::poisson_nll_loss(Tensor input, Tensor target, bool log_input, bool full, float eps, int reduction) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::poisson_nll_loss(
                 (std::move(peek(stack, 0, 6))).toTensor(),
                 (std::move(peek(stack, 1, 6))).toTensor(),
                 (std::move(peek(stack, 2, 6))).toBool(),
                 (std::move(peek(stack, 3, 6))).toBool(),
                 (std::move(peek(stack, 4, 6))).toDouble(),
                 (std::move(peek(stack, 5, 6))).toInt()
             );
             drop(stack, 6);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::polygamma(int n, Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::polygamma(
                 (std::move(peek(stack, 0, 2))).toInt(),
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::pow.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::pow_out(
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
         "aten::pow.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::pow_out(
                 out,
                 (std::move(peek(stack, 0, 3))).toScalar(),
                 (std::move(peek(stack, 1, 3))).toTensor()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::pow.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::pow_out(
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
         "aten::q_per_channel_axis(Tensor self) -> int",
         [](Stack & stack) {
         
             auto result_ = at::q_per_channel_axis(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::q_per_channel_scales(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::q_per_channel_scales(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::qscheme(Tensor self) -> QScheme",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).qscheme(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::quantize_per_tensor(Tensor self, float scale, int zero_point, ScalarType dtype) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::quantize_per_tensor(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toDouble(),
                 (std::move(peek(stack, 2, 4))).toInt(),
                 (std::move(peek(stack, 3, 4))).toScalarType()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::quantized_gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::quantized_gru_cell(
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
         "aten::quantized_lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::quantized_lstm_cell(
                 (std::move(peek(stack, 0, 14))).toTensor(),
                 (std::move(peek(stack, 1, 14))).toTensorListRef(),
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
         "aten::quantized_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::quantized_max_pool2d(
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
         "aten::quantized_rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::quantized_rnn_tanh_cell(
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
         "aten::rand(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 1, 5))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 2, 5))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 3, 5))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 4, 5))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::rand((std::move(peek(stack, 0, 5))).toIntListRef(),
             options);
             #else
                 auto result_ = torch::rand((std::move(peek(stack, 0, 5))).toIntListRef(),
             options);
             #endif
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::rand_like(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::rand_like(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::rand_like.dtype(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 1, 5))).toScalarType())
                     .layout((std::move(peek(stack, 2, 5))).toLayout())
                     .device((std::move(peek(stack, 3, 5))).toDevice())
                     .pinned_memory((std::move(peek(stack, 4, 5))).toBool());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::rand_like((std::move(peek(stack, 0, 5))).toTensor(),
             options);
             #else
                 auto result_ = torch::rand_like((std::move(peek(stack, 0, 5))).toTensor(),
             options);
             #endif
             drop(stack, 5);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::randn.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::randn_out(
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
         "aten::random_(Tensor(a!) self, *, Generator? generator=None) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).random_(
                 nullptr
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::random_.to(Tensor(a!) self, int to, *, Generator? generator=None) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 3))).toTensor();
             auto result_ = (self).random_(
                 (std::move(peek(stack, 1, 3))).toInt(),
                 nullptr
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::random_.from(Tensor(a!) self, int from, int to, *, Generator? generator=None) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 4))).toTensor();
             auto result_ = (self).random_(
                 (std::move(peek(stack, 1, 4))).toInt(),
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
         "aten::range.out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::range_out(
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
         "aten::reciprocal(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::reciprocal(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::reflection_pad1d.out(Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::reflection_pad1d_out(
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
         "aten::reflection_pad2d(Tensor self, int[4] padding) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::reflection_pad2d(
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
         "aten::reflection_pad2d_backward.grad_input(Tensor grad_output, Tensor self, int[4] padding, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::reflection_pad2d_backward_out(
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
         "aten::relu_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::relu_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::remainder.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::remainder_out(
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
         "aten::remainder.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::remainder_out(
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
         "aten::renorm_(Tensor(a!) self, Scalar p, int dim, Scalar maxnorm) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 4))).toTensor();
             auto result_ = (self).renorm_(
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
         "aten::repeat_interleave.Tensor(Tensor repeats) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::repeat_interleave(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::repeat_interleave.self_Tensor(Tensor self, Tensor repeats, int? dim=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::repeat_interleave(
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
         "aten::repeat_interleave.self_int(Tensor self, int repeats, int? dim=None) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::repeat_interleave(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toInt(),
                 (std::move(peek(stack, 2, 3))).toOptional<int64_t>()
             );
             drop(stack, 3);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::replication_pad1d.out(Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::replication_pad1d_out(
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
         "aten::replication_pad2d(Tensor self, int[4] padding) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::replication_pad2d(
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
         "aten::replication_pad2d_backward.grad_input(Tensor grad_output, Tensor self, int[4] padding, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::replication_pad2d_backward_out(
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
         "aten::replication_pad3d_backward(Tensor grad_output, Tensor self, int[6] padding) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::replication_pad3d_backward(
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
         "aten::reshape_as(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 2))).toTensor()).reshape_as(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::resize_as_(Tensor(a!) self, Tensor the_template) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = at::resize_as_(
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
         "aten::result_type.Tensor(Tensor tensor, Tensor other) -> ScalarType",
         [](Stack & stack) {
         
             auto result_ = at::result_type(
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
         "aten::result_type.Scalar_Tensor(Scalar scalar, Tensor tensor) -> ScalarType",
         [](Stack & stack) {
         
             auto result_ = at::result_type(
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
         "aten::result_type.Scalar(Tensor tensor, Scalar other) -> ScalarType",
         [](Stack & stack) {
         
             auto result_ = at::result_type(
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
         "aten::result_type.Scalar_Scalar(Scalar scalar1, Scalar scalar2) -> ScalarType",
         [](Stack & stack) {
         
             auto result_ = at::result_type(
                 (std::move(peek(stack, 0, 2))).toScalar(),
                 (std::move(peek(stack, 1, 2))).toScalar()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::rnn_tanh.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::rnn_tanh(
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
         "aten::rnn_tanh.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::rnn_tanh(
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
         "aten::round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::round_out(
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
         "aten::rrelu_(Tensor(a!) self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::rrelu_with_noise_(Tensor(a!) self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::rrelu_with_noise_backward(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training) -> Tensor",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::rsqrt_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::rsqrt_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::scatter_.src(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 4))).toTensor();
             auto result_ = (self).scatter_(
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
         "aten::scatter_.value(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 4))).toTensor();
             auto result_ = (self).scatter_(
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
         "aten::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::scatter_add(
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
         "aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)",
         [](Stack & stack) {
         
             auto result_ = at::select(
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
         "aten::set_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = (self).set_(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::set_.source_Tensor(Tensor(a!) self, Tensor source) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).set_(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sigmoid(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::sigmoid(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sigmoid_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::sigmoid_backward_out(
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
         "aten::sign(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::sign(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 1, 2))).toTensor();
             auto result_ = at::sin_out(
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
         "aten::sinh_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::sinh_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)",
         [](Stack & stack) {
         
             auto result_ = at::slogdet(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::slow_conv_dilated2d_backward(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)",
         [](Stack & stack) {
         
             auto result_ = at::slow_conv_dilated2d_backward(
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
         "aten::slow_conv_transpose2d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] output_padding, int[2] dilation, Tensor columns, Tensor ones, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)",
         [](Stack & stack) {
         
             auto result_ = at::slow_conv_transpose2d_backward(
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
         "aten::slow_conv_transpose3d.out(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int[3] dilation=1, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 8, 9))).toTensor();
             auto result_ = at::slow_conv_transpose3d_out(
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
         "aten::smooth_l1_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::smooth_l1_loss(
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
         "aten::smooth_l1_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::soft_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::soft_margin_loss_backward(
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
         "aten::softplus_backward(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, Tensor output) -> Tensor",
         [](Stack & stack) {
         
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::softshrink(Tensor self, Scalar lambd=0.5) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::softshrink(
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
         "aten::softshrink_backward.grad_input(Tensor grad_output, Tensor self, Scalar lambd, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
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
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::solve(Tensor self, Tensor A) -> (Tensor solution, Tensor LU)",
         [](Stack & stack) {
         
             auto result_ = at::solve(
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
         "aten::sparse_dim(Tensor self) -> int",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).sparse_dim(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sparse_mask(Tensor self, Tensor mask) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 2))).toTensor()).sparse_mask(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sparse_resize_and_clear_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 4))).toTensor();
             auto result_ = (self).sparse_resize_and_clear_(
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
         "aten::split.Tensor(Tensor(a) self, int split_size, int dim=0) -> Tensor(a)[]",
         [](Stack & stack) {
         
             auto result_ = at::split(
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
         "aten::sqrt_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::sqrt_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::squeeze_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = (self).squeeze_(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::squeeze_.dim(Tensor(a!) self, int dim) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).squeeze_(
                 (std::move(peek(stack, 1, 2))).toInt()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::stack.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::stack_out(
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
         "aten::std(Tensor self, bool unbiased=True) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::std(
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
         "aten::std.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::std(
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
         "aten::strides(Tensor self) -> int",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).strides(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::sub_out(
                 out,
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 (std::move(peek(stack, 2, 4))).toScalar()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::sum_to_size(Tensor self, int[] size) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 2))).toTensor()).sum_to_size(
                 (std::move(peek(stack, 1, 2))).toIntListRef()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::t_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = (self).t_(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::take(Tensor self, Tensor index) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::take(
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
         "aten::tan_(Tensor(a!) self) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 1))).toTensor();
             auto result_ = at::tan_(
                 self
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::tanh(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::tanh(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::tanh_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::tanh_backward_out(
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
         "aten::tensordot(Tensor self, Tensor other, int[] dims_self, int[] dims_other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::tensordot(
                 (std::move(peek(stack, 0, 4))).toTensor(),
                 (std::move(peek(stack, 1, 4))).toTensor(),
                 (std::move(peek(stack, 2, 4))).toIntListRef(),
                 (std::move(peek(stack, 3, 4))).toIntListRef()
             );
             drop(stack, 4);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::thnn_conv2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::thnn_conv2d(
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
         "aten::thnn_conv2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding) -> (Tensor output, Tensor finput, Tensor fgrad_input)",
         [](Stack & stack) {
         
             auto result_ = at::thnn_conv2d_forward(
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
         "aten::thnn_conv3d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, Tensor finput, Tensor fgrad_input, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)",
         [](Stack & stack) {
         
             auto result_ = at::thnn_conv3d_backward(
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
         "aten::thnn_conv_depthwise2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::thnn_conv_depthwise2d(
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
         "aten::thnn_conv_depthwise2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::thnn_conv_depthwise2d_forward(
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
         "aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::threshold(
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
         "aten::to_sparse(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 1))).toTensor()).to_sparse(
             
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::to_sparse.sparse_dim(Tensor self, int sparse_dim) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 2))).toTensor()).to_sparse(
                 (std::move(peek(stack, 1, 2))).toInt()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)",
         [](Stack & stack) {
         
             auto result_ = at::transpose(
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
         "aten::triangular_solve(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False) -> (Tensor solution, Tensor cloned_coefficient)",
         [](Stack & stack) {
         
             auto result_ = at::triangular_solve(
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
         "aten::tril_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).tril_(
                 (std::move(peek(stack, 1, 2))).toInt()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 3, 7))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 4, 7))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 5, 7))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 6, 7))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::tril_indices((std::move(peek(stack, 0, 7))).toInt(),
             (std::move(peek(stack, 1, 7))).toInt(),
             (std::move(peek(stack, 2, 7))).toInt(),
             options);
             #else
                 auto result_ = torch::tril_indices((std::move(peek(stack, 0, 7))).toInt(),
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
         "aten::triu_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)",
         [](Stack & stack) {
             auto self = (std::move(peek(stack, 0, 2))).toTensor();
             auto result_ = (self).triu_(
                 (std::move(peek(stack, 1, 2))).toInt()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::triu_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
         [](Stack & stack) {
         
             const auto options = TensorOptions()
                     .dtype((std::move(peek(stack, 3, 7))).toOptional<ScalarType>())
                     .layout((std::move(peek(stack, 4, 7))).toOptional<c10::Layout>())
                     .device((std::move(peek(stack, 5, 7))).toOptional<c10::Device>())
                     .pinned_memory((std::move(peek(stack, 6, 7))).toOptional<bool>());
             #ifdef USE_STATIC_DISPATCH
                 auto result_ = at::triu_indices((std::move(peek(stack, 0, 7))).toInt(),
             (std::move(peek(stack, 1, 7))).toInt(),
             (std::move(peek(stack, 2, 7))).toInt(),
             options);
             #else
                 auto result_ = torch::triu_indices((std::move(peek(stack, 0, 7))).toInt(),
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
         "aten::trunc(Tensor self) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::trunc(
                 (std::move(peek(stack, 0, 1))).toTensor()
             );
             drop(stack, 1);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::type_as(Tensor self, Tensor other) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = ((std::move(peek(stack, 0, 2))).toTensor()).type_as(
                 (std::move(peek(stack, 1, 2))).toTensor()
             );
             drop(stack, 2);
             pack(stack, std::move(result_));
             return 0;
         },
         atenOperatorOptions()
     ),
     Operator(
         "aten::unbind.int(Tensor(a) self, int dim=0) -> Tensor(a)[]",
         [](Stack & stack) {
         
             auto result_ = at::unbind(
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
         "aten::unique_dim(Tensor self, int dim, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::unique_dim(
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toInt(),
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
         "aten::unique_dim_consecutive(Tensor self, int dim, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::unique_dim_consecutive(
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
         "aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)",
         [](Stack & stack) {
         
             auto result_ = at::unsqueeze(
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
         "aten::upsample_bicubic2d(Tensor self, int[2] output_size, bool align_corners) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::upsample_bicubic2d(
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
         "aten::upsample_bicubic2d_backward.grad_input(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
             auto result_ = at::upsample_bicubic2d_backward_out(
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
         "aten::upsample_bilinear2d(Tensor self, int[2] output_size, bool align_corners) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::upsample_bilinear2d(
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
         "aten::upsample_bilinear2d_backward.grad_input(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
             auto result_ = at::upsample_bilinear2d_backward_out(
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
         "aten::upsample_linear1d(Tensor self, int[1] output_size, bool align_corners) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::upsample_linear1d(
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
         "aten::upsample_linear1d_backward.grad_input(Tensor grad_output, int[1] output_size, int[3] input_size, bool align_corners, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 4, 5))).toTensor();
             auto result_ = at::upsample_linear1d_backward_out(
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
         "aten::upsample_nearest1d.out(Tensor self, int[1] output_size, *, Tensor(a!) out) -> Tensor(a!)",
         [](Stack & stack) {
             auto out = (std::move(peek(stack, 2, 3))).toTensor();
             auto result_ = at::upsample_nearest1d_out(
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
         "aten::upsample_nearest2d(Tensor self, int[2] output_size) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::upsample_nearest2d(
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
         "aten::upsample_nearest2d_backward.grad_input(Tensor grad_output, int[2] output_size, int[4] input_size, *, Tensor(a!) grad_input) -> Tensor(a!)",
         [](Stack & stack) {
             auto grad_input = (std::move(peek(stack, 3, 4))).toTensor();
             auto result_ = at::upsample_nearest2d_backward_out(
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
         "aten::upsample_nearest3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::upsample_nearest3d_backward(
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
         "aten::upsample_trilinear3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size, bool align_corners) -> Tensor",
         [](Stack & stack) {
         
             auto result_ = at::upsample_trilinear3d_backward(
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
         "aten::var_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::var_mean(
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
         "aten::var_mean.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)",
         [](Stack & stack) {
         
             auto result_ = at::var_mean(
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
     ),});

} // anon namespace


}} // namespace torch::jit
