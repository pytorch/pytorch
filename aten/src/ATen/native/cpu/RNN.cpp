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

std::tuple<Tensor, Tensor, Tensor> _thnn_fused_lstm_cell_backward_cpu_internal_impl(
  const Tensor& grad_hy, const Tensor& grad_cy,
  const Tensor& cx, const Tensor& cy,
  const Tensor& workspace, bool has_bias) {
  auto grad_gates = at::empty_like(workspace, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_cx = at::empty_like(cx, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto N = cx.size(0);
  auto H = cx.size(1);
  int64_t D = 4;
  auto grain_size = (at::internal::GRAIN_SIZE + H * D - 1) / (H * D);
  AT_DISPATCH_FLOATING_TYPES_AND(
    ScalarType::BFloat16, workspace.scalar_type(), "_thnn_fused_lstm_cell_backward_cpu", [&]() {
      const scalar_t* dhy_data = grad_hy.defined() ? grad_hy.data_ptr<scalar_t>() : nullptr;
      const scalar_t* dcy_data = grad_cy.defined() ? grad_cy.data_ptr<scalar_t>() : nullptr;
      bool has_gradhy = dhy_data != nullptr;
      bool has_gradcy = dcy_data != nullptr;
      const scalar_t* cx_data = cx.data_ptr<scalar_t>();
      const scalar_t* cy_data = cy.data_ptr<scalar_t>();
      const scalar_t* workspace_data = workspace.data_ptr<scalar_t>();

      scalar_t* dgates_data = grad_gates.data_ptr<scalar_t>();
      scalar_t* dcx_data = grad_cx.data_ptr<scalar_t>();
      using Vec = vec::Vectorized<scalar_t>;
      constexpr int64_t K = Vec::size();
      const int64_t inner_size = H / K * K;
      auto zero_vec = Vec(static_cast<scalar_t>(0));
      auto one_vec = Vec(static_cast<scalar_t>(1));
      at::parallel_for(0, N, grain_size, [&](int64_t begin, int64_t end) {
        for (const auto n : c10::irange(begin, end)) {
          const scalar_t* cx_ptr = cx_data + n * H;
          const scalar_t* cy_ptr = cy_data + n * H;
          const scalar_t* dhy_ptr = has_gradhy ? dhy_data + n * H : nullptr;
          const scalar_t* dcy_ptr = has_gradcy ? dcy_data + n * H : nullptr;
          scalar_t* dcx_ptr = dcx_data + n * H;

          scalar_t* dgates_ptr0 = dgates_data + n * H * D;
          scalar_t* dgates_ptr1 = dgates_data + n * H * D + H;
          scalar_t* dgates_ptr2 = dgates_data + n * H * D + 2 * H;
          scalar_t* dgates_ptr3 = dgates_data + n * H * D + 3 * H;

          const scalar_t* wp_ptr0 = workspace_data + n * H * D;
          const scalar_t* wp_ptr1 = workspace_data + n * H * D + H;
          const scalar_t* wp_ptr2 = workspace_data + n * H * D + 2 * H;
          const scalar_t* wp_ptr3 = workspace_data + n * H * D + 3 * H;

          int64_t i = 0;
          for (; i < inner_size; i += K) {
            auto g0_vec = Vec::loadu(wp_ptr0 + i);
            auto g1_vec = Vec::loadu(wp_ptr1 + i);
            auto g2_vec = Vec::loadu(wp_ptr2 + i);
            auto g3_vec = Vec::loadu(wp_ptr3 + i);
            auto dcx_vec = Vec::loadu(cy_ptr + i).tanh();
            auto dhy_vec = has_gradhy ? Vec::loadu(dhy_ptr + i) : Vec(0.f);
            auto dcy_vec = has_gradcy ? Vec::loadu(dcy_ptr + i) : Vec(0.f);

            auto dg3_vec = dhy_vec * dcx_vec;
            dcx_vec = dhy_vec * g3_vec * (one_vec - dcx_vec * dcx_vec) + dcy_vec;
            auto cx_dev = Vec::loadu(cx_ptr + i);
            auto cy_dev = Vec::loadu(cy_ptr + i);

            auto dg0_vec = dcx_vec * g2_vec;
            auto dg1_vec = dcx_vec * cx_dev;
            auto dg2_vec = dcx_vec * g0_vec;

            dcx_vec = dcx_vec * g1_vec;

            dg0_vec = dg0_vec * (one_vec - g0_vec) * g0_vec;
            dg1_vec = dg1_vec * (one_vec - g1_vec) * g1_vec;
            dg2_vec = dg2_vec * (one_vec - g2_vec * g2_vec);
            dg3_vec = dg3_vec * (one_vec - g3_vec) * g3_vec;

            dg0_vec.store(dgates_ptr0 + i);
            dg1_vec.store(dgates_ptr1 + i);
            dg2_vec.store(dgates_ptr2 + i);
            dg3_vec.store(dgates_ptr3 + i);
            dcx_vec.store(dcx_ptr + i);
          }
          if (i < H) {
            auto g0_vec = Vec::loadu(wp_ptr0 + i, H - i);
            auto g1_vec = Vec::loadu(wp_ptr1 + i, H - i);
            auto g2_vec = Vec::loadu(wp_ptr2 + i, H - i);
            auto g3_vec = Vec::loadu(wp_ptr3 + i, H - i);
            auto dcx_vec = Vec::loadu(cy_ptr + i, H - i).tanh();
            auto dhy_vec = has_gradhy ? Vec::loadu(dhy_ptr + i, H - i) : Vec(0.f);
            auto dcy_vec = has_gradcy ? Vec::loadu(dcy_ptr + i, H - i) : Vec(0.f);

            auto dg3_vec = dhy_vec * dcx_vec;
            dcx_vec = dhy_vec * g3_vec * (one_vec - dcx_vec * dcx_vec) + dcy_vec;
            auto cx_dev = Vec::loadu(cx_ptr + i, H - i);
            auto cy_dev = Vec::loadu(cy_ptr + i, H - i);

            auto dg0_vec = dcx_vec * g2_vec;
            auto dg1_vec = dcx_vec * cx_dev;
            auto dg2_vec = dcx_vec * g0_vec;

            dcx_vec = dcx_vec * g1_vec;

            dg0_vec = dg0_vec * (one_vec - g0_vec) * g0_vec;
            dg1_vec = dg1_vec * (one_vec - g1_vec) * g1_vec;
            dg2_vec = dg2_vec * (one_vec - g2_vec * g2_vec);
            dg3_vec = dg3_vec * (one_vec - g3_vec) * g3_vec;

            dg0_vec.store(dgates_ptr0 + i, H - i);
            dg1_vec.store(dgates_ptr1 + i, H - i);
            dg2_vec.store(dgates_ptr2 + i, H - i);
            dg3_vec.store(dgates_ptr3 + i, H - i);
            dcx_vec.store(dcx_ptr + i, H - i);
          }

        }
      });
    });

    auto grad_bias = has_bias ? grad_gates.sum(0, /*keepdim=*/false) : at::Tensor{};
    return std::make_tuple(grad_gates, grad_cx, grad_bias);
  
}

std::tuple<Tensor, Tensor, Tensor> _thnn_fused_lstm_cell_forward_cpu_impl(
      const Tensor& input_gates, const Tensor& hidden_gates,
      const Tensor& cx, const c10::optional<Tensor>& input_bias_opt, const c10::optional<Tensor>& hidden_bias_opt) {
  c10::MaybeOwned<Tensor> input_bias_maybe_owned = at::borrow_from_optional_tensor(input_bias_opt);
  const Tensor& input_bias = *input_bias_maybe_owned;
  const Tensor& hidden_bias = c10::value_or_else(hidden_bias_opt, [] {return Tensor();});
  TORCH_CHECK(!input_bias.defined())
  TORCH_CHECK(!hidden_bias.defined())
  auto hy = at::empty_like(cx, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto cy = at::empty_like(cx, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto workspace = at::empty_like(input_gates, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto N = cx.size(0);
    auto H = cx.size(1);
    int64_t D = 4;
    auto grain_size = (at::internal::GRAIN_SIZE + H * D - 1) / (H * D);
    AT_DISPATCH_FLOATING_TYPES_AND(
    ScalarType::BFloat16, input_gates.scalar_type(), "_thnn_fused_lstm_cell_cpu", [&]() {
      const scalar_t* input_data = input_gates.data_ptr<scalar_t>();
      const scalar_t* h_data = hidden_gates.data_ptr<scalar_t>();
      const scalar_t* cx_data = cx.data_ptr<scalar_t>();
      scalar_t* cy_data = cy.data_ptr<scalar_t>();
      scalar_t* hy_data = hy.data_ptr<scalar_t>();
      scalar_t* workspace_data = workspace.data_ptr<scalar_t>();
      using Vec = vec::Vectorized<scalar_t>;
      constexpr int64_t K = Vec::size();
      const int64_t inner_size = H / K * K;
      auto zero_vec = Vec(static_cast<scalar_t>(0));
      auto one_vec = Vec(static_cast<scalar_t>(1));
      at::parallel_for(0, N, grain_size, [&](int64_t begin, int64_t end) {

        for (const auto n : c10::irange(begin, end)) {
          const scalar_t* cx_ptr = cx_data + n * H;
          scalar_t* cy_ptr = cy_data + n * H;
          scalar_t* hy_ptr = hy_data + n * H;

          // for (const auto d : c10::irange(0, D)) {

            const scalar_t* in_ptr0 = input_data + n * H * D;
            const scalar_t* in_ptr1 = input_data + n * H * D + H;
            const scalar_t* in_ptr2 = input_data + n * H * D + 2 * H;
            const scalar_t* in_ptr3 = input_data + n * H * D + 3 * H;

            const scalar_t* h_ptr0 = h_data + n * H * D;
            const scalar_t* h_ptr1 = h_data + n * H * D + H;
            const scalar_t* h_ptr2 = h_data + n * H * D + 2 * H;
            const scalar_t* h_ptr3 = h_data + n * H * D + 3 * H;

            scalar_t* wp_ptr0 = workspace_data + n * H * D;
            scalar_t* wp_ptr1 = workspace_data + n * H * D + H;
            scalar_t* wp_ptr2 = workspace_data + n * H * D + 2 * H;
            scalar_t* wp_ptr3 = workspace_data + n * H * D + 3 * H;

            int64_t i = 0;
            for (; i < inner_size; i += K) {
              auto in0_vec = Vec::loadu(in_ptr0 + i) + Vec::loadu(h_ptr0 + i);
              auto ingate_vec = internal_sigmoid(in0_vec, one_vec, zero_vec);
              auto in1_vec = Vec::loadu(in_ptr1 + i) + Vec::loadu(h_ptr1 + i);
              auto forgetgate_vec = internal_sigmoid(in1_vec, one_vec, zero_vec);
              auto in2_vec = Vec::loadu(in_ptr2 + i) + Vec::loadu(h_ptr2 + i);
              auto cellgate_vec = in2_vec.tanh();
              auto in3_vec = Vec::loadu(in_ptr3 + i) + Vec::loadu(h_ptr3 + i);
              auto outgate_vec = internal_sigmoid(in3_vec, one_vec, zero_vec);
              auto cy_vec = (forgetgate_vec * Vec::loadu(cx_ptr + i)) +
                            (ingate_vec * cellgate_vec);
              auto hy_vec = outgate_vec * cy_vec.tanh();
              cy_vec.store(cy_ptr + i);
              hy_vec.store(hy_ptr + i);
              ingate_vec.store(wp_ptr0 + i);
              forgetgate_vec.store(wp_ptr1 + i);
              cellgate_vec.store(wp_ptr2 + i);
              outgate_vec.store(wp_ptr3 + i);
            }
            if (i < H) {
              auto in0_vec = Vec::loadu(in_ptr0 + i, H - i) + Vec::loadu(h_ptr0 + i, H - i);
              auto ingate_vec = internal_sigmoid(in0_vec, one_vec, zero_vec);
              auto in1_vec = Vec::loadu(in_ptr1 + i, H - i) + Vec::loadu(h_ptr1 + i, H - i);
              auto forgetgate_vec = internal_sigmoid(in1_vec, one_vec, zero_vec);
              auto in2_vec = Vec::loadu(in_ptr2 + i, H - i) + Vec::loadu(h_ptr2 + i, H - i);
              auto cellgate_vec = in2_vec.tanh();
              auto in3_vec = Vec::loadu(in_ptr3 + i, H - i) + Vec::loadu(h_ptr3 + i, H - i);
              auto outgate_vec = internal_sigmoid(in3_vec, one_vec, zero_vec);
              auto cy_vec = (forgetgate_vec * Vec::loadu(cx_ptr + i, H - i)) +
                            (ingate_vec * cellgate_vec);
              auto hy_vec = outgate_vec * cy_vec.tanh();
              cy_vec.store(cy_ptr + i, H - i);
              hy_vec.store(hy_ptr + i, H - i);
              ingate_vec.store(wp_ptr0 + i, H - i);
              forgetgate_vec.store(wp_ptr1 + i, H - i);
              cellgate_vec.store(wp_ptr2 + i, H - i);
              outgate_vec.store(wp_ptr3 + i, H - i);
            }

          // }
        }
      });
    });
    return std::make_tuple(hy, cy, workspace);
}

} // namespace

REGISTER_DISPATCH(fused_lstm_cell_forward_stub, _thnn_fused_lstm_cell_forward_cpu_impl);
REGISTER_DISPATCH(fused_lstm_cell_backward_stub, _thnn_fused_lstm_cell_backward_cpu_internal_impl);
} // namespace native
} // namespace at
