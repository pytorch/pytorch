#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/RNN.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/native/cpu/moments_utils.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#endif

namespace at { namespace native {

namespace {

template<typename Vec>
inline Vec internal_sigmoid(Vec& in_vec, Vec& one_vec, Vec& zero_vec) {
  Vec sig_vec = zero_vec - in_vec;
  sig_vec = sig_vec.exp();
  sig_vec = one_vec + sig_vec;
  return sig_vec.reciprocal();
}

template <typename T, typename T_ACC>
void CalcHyCy(
  const T_ACC* ingate_ptr,
  const T_ACC* forgetgate_ptr,
  const T_ACC* cellgate_ptr,
  const T_ACC* outgate_ptr,
  const T* cx_ptr,
  T* hy_ptr,
  T* cy_ptr,
  int64_t l) {
  using Vec = vec::Vectorized<T>;
  auto cy_vec = (Vec::loadu(cellgate_ptr, l) * Vec::loadu(ingate_ptr, l)) +
                      (Vec::loadu(forgetgate_ptr, l)) * Vec::loadu(cx_ptr, l);
 
  auto hy_vec = Vec::loadu(outgate_ptr, l) * cy_vec.tanh();
  cy_vec.store(cy_ptr, l);
  hy_vec.store(hy_ptr, l);
}

template <>
void CalcHyCy(
  const float* ingate_ptr,
  const float* forgetgate_ptr,
  const float* cellgate_ptr,
  const float* outgate_ptr,
  const BFloat16* cx_ptr,
  BFloat16* hy_ptr,
  BFloat16* cy_ptr,
  int64_t l) {
  using fVec = vec::Vectorized<float>;
  using bVec = vec::Vectorized<BFloat16>;
  auto fsize = fVec::size();
  fVec cx_vec0, cx_vec1;
  std::tie(cx_vec0, cx_vec1) = convert_bfloat16_float(bVec::loadu(cx_ptr, l));
  auto ingate_vec0 = fVec::loadu(ingate_ptr, l > fsize ? fsize : l);
  auto ingate_vec1 = fVec::loadu(ingate_ptr + fsize, l > fsize ? (l - fsize) : 0);
  auto forgetgate_vec0 = fVec::loadu(forgetgate_ptr, l > fsize ? fsize : l);
  auto forgetgate_vec1 = fVec::loadu(forgetgate_ptr + fsize, l > fsize ? (l - fsize) : 0);
  auto cellgate_vec0 = fVec::loadu(cellgate_ptr, l > fsize ? fsize : l);
  auto cellgate_vec1 = fVec::loadu(cellgate_ptr + fsize, l > fsize ? (l - fsize) : 0);
  auto outgate_vec0 = fVec::loadu(outgate_ptr, l > fsize ? fsize : l);
  auto outgate_vec1 = fVec::loadu(outgate_ptr + fsize, l > fsize ? (l - fsize) : 0);


  auto cy_vec0 = cellgate_vec0 * ingate_vec0 +
                forgetgate_vec0 * cx_vec0;
  auto cy_vec1 = cellgate_vec1 * ingate_vec1 +
                forgetgate_vec1 * cx_vec1;

  auto hy_vec0 = outgate_vec0 * cy_vec0.tanh();
  auto hy_vec1 = outgate_vec1 * cy_vec1.tanh();
  convert_float_bfloat16(cy_vec0, cy_vec1).store(cy_ptr, l);
  convert_float_bfloat16(hy_vec0, hy_vec1).store(hy_ptr, l);
}

template <typename T, typename T_ACC>
inline void CalcForward(const T* in_ptr, const T* h_ptr,
T_ACC* buffer_ptr, T* wp_ptr,
vec::Vectorized<T_ACC> &one_vec,
vec::Vectorized<T_ACC> &zero_vec,
int64_t d, int64_t l) {
  using Vec = vec::Vectorized<T>;
  auto in_vec = Vec::loadu(in_ptr, l) + Vec::loadu(h_ptr, l);
  if (d == 2) {
    in_vec = in_vec.tanh();
  } else {
    in_vec = internal_sigmoid(in_vec, one_vec, zero_vec);
  }
  in_vec.store(buffer_ptr, l);
  in_vec.store(wp_ptr, l);
}

template <>
inline void CalcForward(const BFloat16* in_ptr, const BFloat16* h_ptr,
float*buffer_ptr, BFloat16* wp_ptr, vec::Vectorized<float> &one_vec, vec::Vectorized<float> &zero_vec, int64_t d, int64_t l) {
  using fVec = vec::Vectorized<float>;
  using bVec = vec::Vectorized<BFloat16>;
  auto fsize = fVec::size();
  auto in_vec = bVec::loadu(in_ptr, l);
  auto h_vec = bVec::loadu(h_ptr, l);
  fVec in_vec0, in_vec1, h_vec0, h_vec1;
  std::tie(in_vec0, in_vec1) = convert_bfloat16_float(in_vec);
  std::tie(h_vec0, h_vec1) = convert_bfloat16_float(h_vec);
  in_vec0 = in_vec0 + h_vec0;
  in_vec1 = in_vec1 + h_vec1;
  if (d == 2) {
    in_vec0 = in_vec0.tanh();
    in_vec1 = in_vec1.tanh();
  } else {
    in_vec0 = internal_sigmoid(in_vec0, one_vec, zero_vec);
    in_vec1 = internal_sigmoid(in_vec1, one_vec, zero_vec);
  }
  in_vec0.store(buffer_ptr, l > fsize ? fsize: l);
  in_vec1.store(buffer_ptr + fsize, l > fsize ? (l - fsize): 0);
  convert_float_bfloat16(in_vec0, in_vec1).store(wp_ptr, l);
}

template <typename T>
void _thnn_fused_lstm_cell_forward_cpu_internal_impl(
  const Tensor& input_gates,
  const Tensor& hidden_gates,
  const Tensor& cx,
  Tensor& workspace,
  Tensor& hy,
  Tensor& cy) {
  auto N = cx.size(0);
  auto H = cx.size(1);
  int64_t D = 4;
  auto grain_size = (at::internal::GRAIN_SIZE + H * D - 1) / (H * D);

  const T* input_data = input_gates.data_ptr<T>();
  const T* h_data = hidden_gates.data_ptr<T>();
  const T* cx_data = cx.data_ptr<T>();
  T* cy_data = cy.data_ptr<T>();
  T* hy_data = hy.data_ptr<T>();
  T* workspace_data = workspace.data_ptr<T>();
  using Vec = vec::Vectorized<T>;
  using T_ACC = vec::vec_scalar_t<T>;
  constexpr int64_t K = Vec::size();
  const int64_t inner_size = H / K * K;
  auto zero_vec = vec::Vectorized<T_ACC>(static_cast<T_ACC>(0));
  auto one_vec = vec::Vectorized<T_ACC>(static_cast<T_ACC>(1));
  Tensor buffer = at::empty({D * H}, workspace.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
  T_ACC * buffer_data = buffer.data_ptr<T_ACC>();
  // For lstm_cell, H is usually not very large. Parallel on N.
  at::parallel_for(0, N, grain_size, [&](int64_t begin, int64_t end) {

    for (const auto n : c10::irange(begin, end)) {
      const T* cx_ptr = cx_data + n * H;
      T* cy_ptr = cy_data + n * H;
      T* hy_ptr = hy_data + n * H;

      // First loop
      for (const auto d : c10::irange(0, D)) {
        const T* in_ptr = input_data + n * H * D + d * H;
        const T* h_ptr = h_data + n * H * D + d * H;
        T* wp_ptr = workspace_data + n * H * D + d * H;
        T_ACC* buffer_ptr = buffer_data + d * H;

        int64_t i = 0;
        for (; i < inner_size; i += K) {
          CalcForward<T, T_ACC>(in_ptr + i, h_ptr + i, buffer_ptr + i, wp_ptr + i, one_vec, zero_vec, d, K);
        }
        if (i < H) {
          CalcForward<T, T_ACC>(in_ptr + i, h_ptr + i, buffer_ptr + i, wp_ptr + i, one_vec, zero_vec, d, H - i);
        }
      }

      // Second loop
      const T_ACC* buffer_ptr0 = buffer_data;
      const T_ACC* buffer_ptr1 = buffer_data + H;
      const T_ACC* buffer_ptr2 = buffer_data + 2 * H;
      const T_ACC* buffer_ptr3 = buffer_data + 3 * H;
      int64_t i = 0;
      for (; i < inner_size; i += K) {
        CalcHyCy<T, T_ACC>(buffer_ptr0 + i, buffer_ptr1 + i, buffer_ptr2 + i, buffer_ptr3 + i,
                           cx_ptr + i, hy_ptr + i, cy_ptr + i, K);
      }
      if (i < H) {
        CalcHyCy<T, T_ACC>(buffer_ptr0 + i, buffer_ptr1 + i, buffer_ptr2 + i, buffer_ptr3 + i,
                           cx_ptr + i, hy_ptr + i, cy_ptr + i, H - i);
      }
    }
  });

}

std::tuple<Tensor, Tensor, Tensor> _thnn_fused_lstm_cell_forward_cpu_kernel(
      const Tensor& input_gates, const Tensor& hidden_gates,
      const Tensor& cx, const c10::optional<Tensor>& input_bias_opt, const c10::optional<Tensor>& hidden_bias_opt) {
  c10::MaybeOwned<Tensor> input_bias_maybe_owned = at::borrow_from_optional_tensor(input_bias_opt);
  const Tensor& input_bias = *input_bias_maybe_owned;
  const Tensor& hidden_bias = c10::value_or_else(hidden_bias_opt, [] {return Tensor();});
  TORCH_CHECK(!input_bias.defined())
  TORCH_CHECK(!hidden_bias.defined())

  auto hy = at::empty_like(cx);
  auto cy = at::empty_like(cx);
  auto workspace = at::empty_like(input_gates);

  AT_DISPATCH_FLOATING_TYPES_AND(
  ScalarType::BFloat16, input_gates.scalar_type(), "_thnn_fused_lstm_cell_cpu", [&]() {
    _thnn_fused_lstm_cell_forward_cpu_internal_impl<scalar_t>(input_gates, hidden_gates, cx, workspace, hy, cy);
  });
  return std::make_tuple(hy, cy, workspace);
}

template <typename T, typename T_ACC>
inline void CalcBckward(const T* wp_ptr0, const T* wp_ptr1, const T* wp_ptr2, const T* wp_ptr3,
const T* cx_ptr, const T* cy_ptr, const T* dhy_ptr, const T* dcy_ptr, T* dcx_ptr, T* dgates_ptr0, T* dgates_ptr1, T* dgates_ptr2, T* dgates_ptr3, vec::Vectorized<T_ACC> &one_vec, int64_t l, bool has_gradhy, bool has_gradcy) {
  using Vec = vec::Vectorized<T>;
  auto g0_vec = Vec::loadu(wp_ptr0, l);
  auto g1_vec = Vec::loadu(wp_ptr1, l);
  auto g2_vec = Vec::loadu(wp_ptr2, l);
  auto g3_vec = Vec::loadu(wp_ptr3, l);
  auto dcx_vec = Vec::loadu(cy_ptr, l).tanh();
  auto dhy_vec = has_gradhy ? Vec::loadu(dhy_ptr, l) : Vec(0.f);
  auto dcy_vec = has_gradcy ? Vec::loadu(dcy_ptr, l) : Vec(0.f);

  auto dg3_vec = dhy_vec * dcx_vec;
  dcx_vec = dhy_vec * g3_vec * (one_vec - dcx_vec * dcx_vec) + dcy_vec;
  auto cx_vec = Vec::loadu(cx_ptr, l);

  auto dg0_vec = dcx_vec * g2_vec;
  auto dg1_vec = dcx_vec * cx_vec;
  auto dg2_vec = dcx_vec * g0_vec;

  dcx_vec = dcx_vec * g1_vec;

  dg0_vec = dg0_vec * (one_vec - g0_vec) * g0_vec;
  dg1_vec = dg1_vec * (one_vec - g1_vec) * g1_vec;
  dg2_vec = dg2_vec * (one_vec - g2_vec * g2_vec);
  dg3_vec = dg3_vec * (one_vec - g3_vec) * g3_vec;

  dg0_vec.store(dgates_ptr0, l);
  dg1_vec.store(dgates_ptr1, l);
  dg2_vec.store(dgates_ptr2, l);
  dg3_vec.store(dgates_ptr3, l);
  dcx_vec.store(dcx_ptr, l);
}

template <>
inline void CalcBckward(const BFloat16* wp_ptr0, const BFloat16* wp_ptr1, const BFloat16* wp_ptr2, const BFloat16* wp_ptr3,
const BFloat16* cx_ptr, const BFloat16*cy_ptr, const BFloat16* dhy_ptr, const BFloat16* dcy_ptr, BFloat16* dcx_ptr, BFloat16* dgates_ptr0, BFloat16* dgates_ptr1, BFloat16* dgates_ptr2, BFloat16* dgates_ptr3, vec::Vectorized<float> &one_vec, int64_t l, bool has_gradhy, bool has_gradcy) {
  using fVec = vec::Vectorized<float>;
  using bVec = vec::Vectorized<BFloat16>;
  fVec g0_vec0, g0_vec1, g1_vec0, g1_vec1, g2_vec0, g2_vec1, g3_vec0, g3_vec1;
  auto g0_vec = bVec::loadu(wp_ptr0, l);
  std::tie(g0_vec0, g0_vec1) = convert_bfloat16_float(g0_vec);
  auto g1_vec = bVec::loadu(wp_ptr1, l);
  std::tie(g1_vec0, g1_vec1) = convert_bfloat16_float(g1_vec);
  auto g2_vec = bVec::loadu(wp_ptr2, l);
  std::tie(g2_vec0, g2_vec1) = convert_bfloat16_float(g2_vec);
  auto g3_vec = bVec::loadu(wp_ptr3, l);
  std::tie(g3_vec0, g3_vec1) = convert_bfloat16_float(g3_vec);
  auto dcx_vec = bVec::loadu(cy_ptr, l);
  fVec dcx_vec0, dcx_vec1;
  std::tie(dcx_vec0, dcx_vec1) = convert_bfloat16_float(dcx_vec);
  dcx_vec0 = dcx_vec0.tanh();
  dcx_vec1 = dcx_vec1.tanh();
  auto dhy_vec = has_gradhy ? bVec::loadu(dhy_ptr, l) : bVec(0.f);
  auto dcy_vec = has_gradcy ? bVec::loadu(dcy_ptr, l) : bVec(0.f);
  fVec dhy_vec0, dhy_vec1, dcy_vec0, dcy_vec1;
  std::tie(dhy_vec0, dhy_vec1) = convert_bfloat16_float(dhy_vec);
  std::tie(dcy_vec0, dcy_vec1) = convert_bfloat16_float(dcy_vec);

  auto dg3_vec0 = dhy_vec0 * dcx_vec0;
  dcx_vec0 = dhy_vec0 * g3_vec0 * (one_vec - dcx_vec0 * dcx_vec0) + dcy_vec0;
  auto dg3_vec1 = dhy_vec1 * dcx_vec1;
  dcx_vec1 = dhy_vec1 * g3_vec1 * (one_vec - dcx_vec1 * dcx_vec1) + dcy_vec1;
  auto cx_vec = bVec::loadu(cx_ptr, l);
  fVec cx_vec0, cx_vec1;
  std::tie(cx_vec0, cx_vec1) = convert_bfloat16_float(cx_vec);

  auto dg0_vec0 = dcx_vec0 * g2_vec0;
  auto dg0_vec1 = dcx_vec1 * g2_vec1;
  auto dg1_vec0 = dcx_vec0 * cx_vec0;
  auto dg1_vec1 = dcx_vec1 * cx_vec1;
  auto dg2_vec0 = dcx_vec0 * g0_vec0;
  auto dg2_vec1 = dcx_vec1 * g0_vec1;

  dcx_vec0 = dcx_vec0 * g1_vec0;
  dcx_vec1 = dcx_vec1 * g1_vec1;

  dg0_vec0 = dg0_vec0 * (one_vec - g0_vec0) * g0_vec0;
  dg0_vec1 = dg0_vec1 * (one_vec - g0_vec1) * g0_vec1;
  dg1_vec0 = dg1_vec0 * (one_vec - g1_vec0) * g1_vec0;
  dg1_vec1 = dg1_vec1 * (one_vec - g1_vec1) * g1_vec1;
  dg2_vec0 = dg2_vec0 * (one_vec - g2_vec0 * g2_vec0);
  dg2_vec1 = dg2_vec1 * (one_vec - g2_vec1 * g2_vec1);
  dg3_vec0 = dg3_vec0 * (one_vec - g3_vec0) * g3_vec0;
  dg3_vec1 = dg3_vec1 * (one_vec - g3_vec1) * g3_vec1;

  convert_float_bfloat16(dg0_vec0, dg0_vec1).store(dgates_ptr0, l);
  convert_float_bfloat16(dg1_vec0, dg1_vec1).store(dgates_ptr1, l);
  convert_float_bfloat16(dg2_vec0, dg2_vec1).store(dgates_ptr2, l);
  convert_float_bfloat16(dg3_vec0, dg3_vec1).store(dgates_ptr3, l);
  convert_float_bfloat16(dcx_vec0, dcx_vec1).store(dcx_ptr, l);
}

template <typename T>
void _thnn_fused_lstm_cell_backward_cpu_internal_internal_impl(
  const Tensor& grad_hy, const Tensor& grad_cy,
  const Tensor& cx, const Tensor& cy,
  const Tensor& workspace, bool has_bias,
  Tensor& grad_gates, Tensor& grad_cx) {
    auto N = cx.size(0);
  auto H = cx.size(1);
  int64_t D = 4;
  auto grain_size = (at::internal::GRAIN_SIZE + H * D - 1) / (H * D);

  const T* dhy_data = grad_hy.defined() ? grad_hy.data_ptr<T>() : nullptr;
  const T* dcy_data = grad_cy.defined() ? grad_cy.data_ptr<T>() : nullptr;
  bool has_gradhy = dhy_data != nullptr;
  bool has_gradcy = dcy_data != nullptr;
  const T* cx_data = cx.data_ptr<T>();
  const T* cy_data = cy.data_ptr<T>();
  const T* workspace_data = workspace.data_ptr<T>();

  T* dgates_data = grad_gates.data_ptr<T>();
  T* dcx_data = grad_cx.data_ptr<T>();
  using Vec = vec::Vectorized<T>;
  using T_ACC = vec::vec_scalar_t<T>;
  constexpr int64_t K = Vec::size();
  const int64_t inner_size = H / K * K;
  auto one_vec = vec::Vectorized<T_ACC>(static_cast<T_ACC>(1));
  at::parallel_for(0, N, grain_size, [&](int64_t begin, int64_t end) {
    for (const auto n : c10::irange(begin, end)) {
      const T* cx_ptr = cx_data + n * H;
      const T* cy_ptr = cy_data + n * H;
      const T* dhy_ptr = has_gradhy ? dhy_data + n * H : nullptr;
      const T* dcy_ptr = has_gradcy ? dcy_data + n * H : nullptr;
      T* dcx_ptr = dcx_data + n * H;

      T* dgates_ptr0 = dgates_data + n * H * D;
      T* dgates_ptr1 = dgates_data + n * H * D + H;
      T* dgates_ptr2 = dgates_data + n * H * D + 2 * H;
      T* dgates_ptr3 = dgates_data + n * H * D + 3 * H;

      const T* wp_ptr0 = workspace_data + n * H * D;
      const T* wp_ptr1 = workspace_data + n * H * D + H;
      const T* wp_ptr2 = workspace_data + n * H * D + 2 * H;
      const T* wp_ptr3 = workspace_data + n * H * D + 3 * H;

      int64_t i = 0;
      for (; i < inner_size; i += K) {
        CalcBckward<T, T_ACC>(wp_ptr0 + i, wp_ptr1 + i, wp_ptr2 + i, wp_ptr3 + i,
                              cx_ptr + i, cy_ptr + i, dhy_ptr + i, dcy_ptr + i, dcx_ptr + i,
                              dgates_ptr0 + i, dgates_ptr1 + i, dgates_ptr2 + i, dgates_ptr3 + i,
                              one_vec, K, has_gradhy, has_gradcy);
      }
      if (i < H) {
        CalcBckward<T, T_ACC>(wp_ptr0 + i, wp_ptr1 + i, wp_ptr2 + i, wp_ptr3 + i,
                              cx_ptr + i, cy_ptr + i, dhy_ptr + i, dcy_ptr + i, dcx_ptr + i,
                              dgates_ptr0 + i, dgates_ptr1 + i, dgates_ptr2 + i, dgates_ptr3 + i,
                              one_vec, H - i, has_gradhy, has_gradcy);
      }

    }
  });

}

std::tuple<Tensor, Tensor, Tensor> _thnn_fused_lstm_cell_backward_cpu_internal_kernel(
  const Tensor& grad_hy, const Tensor& grad_cy,
  const Tensor& cx, const Tensor& cy,
  const Tensor& workspace, bool has_bias) {
  auto grad_gates = at::empty_like(workspace, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_cx = at::empty_like(cx, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  AT_DISPATCH_FLOATING_TYPES_AND(
    ScalarType::BFloat16, workspace.scalar_type(), "_thnn_fused_lstm_cell_backward_cpu", [&]() {
      _thnn_fused_lstm_cell_backward_cpu_internal_internal_impl<scalar_t>(grad_hy, grad_cy, cx, cy, workspace, has_bias, grad_gates, grad_cx);
    });

  auto grad_bias = has_bias ? grad_gates.sum(0, /*keepdim=*/false) : at::Tensor{};
  return std::make_tuple(grad_gates, grad_cx, grad_bias);
}

} // namespace

REGISTER_DISPATCH(fused_lstm_cell_forward_stub, _thnn_fused_lstm_cell_forward_cpu_kernel);
REGISTER_DISPATCH(fused_lstm_cell_backward_stub, _thnn_fused_lstm_cell_backward_cpu_internal_kernel);
} // namespace native
} // namespace at
