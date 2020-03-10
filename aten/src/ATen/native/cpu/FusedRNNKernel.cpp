#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/native/RNN.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>

namespace at { namespace native {

namespace {

using namespace vec256;

template <typename scalar_t>
static inline scalar_t sigmoid(scalar_t a)  {
  scalar_t one = static_cast<scalar_t>(1.0);
  return one / (one + ::exp(-a));
}

template <typename scalar_t>
static inline Vec256<scalar_t> sigmoid(Vec256<scalar_t> a) {
  auto one_vec = Vec256<scalar_t>(scalar_t(1));
  return one_vec / (one_vec + a.neg().exp());
}

template <typename scalar_t>
void fused_lstm_cell_kernel(
    Tensor& hy,
    Tensor& cy,
    Tensor& workspace,
    const Tensor& input_gates_,
    const Tensor& hidden_gates_,
    const Tensor& cx_) {
  using Vec = Vec256<scalar_t>;
  int64_t batch_size = cx_.size(0);
  int64_t hidden_size = cx_.size(1);
  int64_t gate_size = hidden_size * 4;

  TORCH_CHECK(input_gates_.sizes().equals({batch_size, gate_size}),
              "expect input_gates size of [", batch_size, ", ", gate_size,
              "], got ", input_gates_.sizes());
  TORCH_CHECK(hidden_gates_.sizes().equals({batch_size, gate_size}),
              "expect hidden_gates size of [", batch_size, ", ", gate_size,
              "], got ", hidden_gates_.sizes());

  auto input_gates = input_gates_.contiguous();
  auto hidden_gates = hidden_gates_.contiguous();
  auto cx = cx_.contiguous();

  auto input_gates_data = input_gates.data_ptr<scalar_t>();
  auto hidden_gates_data = hidden_gates.data_ptr<scalar_t>();
  auto cx_data = cx.data_ptr<scalar_t>();
  auto hy_data = hy.data_ptr<scalar_t>();
  auto cy_data = cy.data_ptr<scalar_t>();
  auto workspace_data = workspace.data_ptr<scalar_t>();

  at::parallel_for(0, batch_size, 1,  [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; i++) {
      scalar_t* input_gates_ptr = input_gates_data + i * gate_size;
      scalar_t* hidden_gates_ptr = hidden_gates_data + i * gate_size;
      scalar_t* workspace_ptr = workspace_data + i * gate_size;

      // input_gates: input, forget, cell, output
      scalar_t* iig = input_gates_ptr;
      scalar_t* ifg = input_gates_ptr + hidden_size;
      scalar_t* icg = input_gates_ptr + 2 * hidden_size;
      scalar_t* iog = input_gates_ptr + 3 * hidden_size;
      // hidden_gates: input, forget, cell, output
      scalar_t* hig = hidden_gates_ptr;
      scalar_t* hfg = hidden_gates_ptr + hidden_size;
      scalar_t* hcg = hidden_gates_ptr + 2 * hidden_size;
      scalar_t* hog = hidden_gates_ptr + 3 * hidden_size;
      // workspace: input, forget, cell, output
      scalar_t* wig = workspace_ptr;
      scalar_t* wfg = workspace_ptr + hidden_size;
      scalar_t* wcg = workspace_ptr + 2 * hidden_size;
      scalar_t* wog = workspace_ptr + 3 * hidden_size;

      scalar_t* cx_ptr = cx_data + i * hidden_size;
      scalar_t* hy_ptr = hy_data + i * hidden_size;
      scalar_t* cy_ptr = cy_data + i * hidden_size;

      int64_t size = hidden_size;
      int64_t d = 0;
      for (; d < size - (size % Vec::size()); d += Vec::size()) {
        Vec ig_vec = sigmoid<scalar_t>(Vec::loadu(iig + d) + Vec::loadu(hig + d));
        Vec fg_vec = sigmoid<scalar_t>(Vec::loadu(ifg + d) + Vec::loadu(hfg + d));
        Vec cg_vec = (Vec::loadu(icg + d) + Vec::loadu(hcg + d)).tanh();
        Vec og_vec = sigmoid<scalar_t>(Vec::loadu(iog + d) + Vec::loadu(hog + d));
        Vec cy_vec = fg_vec * Vec::loadu(cx_ptr + d) + ig_vec * cg_vec;
        Vec hy_vec = og_vec * cy_vec.tanh();
        cy_vec.store(cy_ptr + d);
        hy_vec.store(hy_ptr + d);
        ig_vec.store(wig + d);
        fg_vec.store(wfg + d);
        cg_vec.store(wcg + d);
        og_vec.store(wog + d);
      }
      for (; d < size; d++) {
        scalar_t ig = sigmoid(iig[d] + hig[d]);
        scalar_t fg = sigmoid(ifg[d] + hfg[d]);
        scalar_t cg = ::tanh(icg[d] + hcg[d]);
        scalar_t og = sigmoid(iog[d] + hog[d]);
        scalar_t _cy = fg * cx_ptr[d] + ig * cg;
        scalar_t _hy = og * ::tanh(_cy);
        cy_ptr[d] = _cy;
        hy_ptr[d] = _hy;
        wig[d] = ig;
        wfg[d] = fg;
        wcg[d] = cg;
        wog[d] = og;
      }
    }
  });
}

template <typename scalar_t>
void fused_lstm_cell_backward_kernel(
    Tensor& grad_gates,
    Tensor& grad_cx,
    const Tensor& grad_hy_,
    const Tensor& grad_cy_,
    const Tensor& cx_,
    const Tensor& cy_,
    const Tensor& workspace) {
  using Vec = Vec256<scalar_t>;
  int64_t batch_size = cx_.size(0);
  int64_t hidden_size = cx_.size(1);
  int64_t gate_size = hidden_size * 4;

  bool has_grad_hy = grad_hy_.defined();
  bool has_grad_cy = grad_cy_.defined();
  // undefined tensor gradient support should already be handled at 
  // _fused_lstm_cell_backward_cpu
  TORCH_INTERNAL_ASSERT((has_grad_hy || has_grad_cy));
  if (has_grad_hy) {
    TORCH_CHECK(grad_hy_.sizes().equals({batch_size, hidden_size}),
                "expect grad_hy size of ", batch_size, ", ", hidden_size,
                "], got ", grad_hy_.sizes());
  }
  if (has_grad_cy) {
    TORCH_CHECK(grad_cy_.sizes().equals({batch_size, hidden_size}),
                "expect grad_cy size of ", batch_size, ", ", hidden_size,
                "], got ", grad_cy_.sizes());
  }
  TORCH_CHECK(cy_.sizes().equals({batch_size, hidden_size}),
              "expect cy size of ", batch_size, ", ", hidden_size,
              "], got ", cy_.sizes());
  TORCH_CHECK(workspace.sizes().equals({batch_size, 4 * hidden_size}),
              "expect workspace size of ", batch_size, ", ", 4 * hidden_size,
              "], got", workspace.sizes());

  auto grad_hy = has_grad_hy ? grad_hy_.contiguous() : Tensor();
  auto grad_cy = has_grad_cy ? grad_cy_.contiguous() : Tensor();
  auto cx = cx_.contiguous();
  auto cy = cy_.contiguous();

  auto grad_hy_data = has_grad_hy ? grad_hy.data_ptr<scalar_t>() : nullptr;
  auto grad_cy_data = has_grad_cy ? grad_cy.data_ptr<scalar_t>() : nullptr;
  auto cx_data = cx.data_ptr<scalar_t>();
  auto cy_data = cy.data_ptr<scalar_t>();
  auto workspace_data = workspace.data_ptr<scalar_t>();
  auto grad_gates_data = grad_gates.data_ptr<scalar_t>();
  auto grad_cx_data = grad_cx.data_ptr<scalar_t>();

  at::parallel_for(0, batch_size, 1,  [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; i++) {
      scalar_t* workspace_ptr = workspace_data + i * gate_size;
      scalar_t* grad_gates_ptr = grad_gates_data + i * gate_size;

      // input_gates: input, forget, cell, output
      scalar_t* ig = workspace_ptr;
      scalar_t* fg = workspace_ptr + hidden_size;
      scalar_t* cg = workspace_ptr + 2 * hidden_size;
      scalar_t* og = workspace_ptr + 3 * hidden_size;
      // grad_gates: input, forget, cell, output
      scalar_t* ih = grad_gates_ptr;
      scalar_t* fh = grad_gates_ptr + hidden_size;
      scalar_t* ch = grad_gates_ptr + 2 * hidden_size;
      scalar_t* oh = grad_gates_ptr + 3 * hidden_size;

      scalar_t* grad_hy_ptr = has_grad_hy ? grad_hy_data + i * hidden_size : nullptr;
      scalar_t* grad_cy_ptr = has_grad_cy ? grad_cy_data + i * hidden_size : nullptr;
      scalar_t* cx_ptr = cx_data + i * hidden_size;
      scalar_t* cy_ptr = cy_data + i * hidden_size;
      scalar_t* grad_cx_ptr = grad_cx_data + i * hidden_size;

      int64_t size = hidden_size;
      int64_t d = 0;
      auto one_vec = Vec(scalar_t(1));
      auto zero_vec = Vec(scalar_t(0));
      for (; d < size - (size % Vec::size()); d += Vec::size()) {
        Vec ig_vec = Vec::loadu(ig + d);
        Vec fg_vec = Vec::loadu(fg + d);
        Vec cg_vec = Vec::loadu(cg + d);
        Vec og_vec = Vec::loadu(og + d);
        Vec ghy_vec = has_grad_hy ? Vec::loadu(grad_hy_ptr + d) : zero_vec;
        Vec gcy_vec = has_grad_cy ? Vec::loadu(grad_cy_ptr + d) : zero_vec;

        Vec gcx_vec = Vec::loadu(cy_ptr + d).tanh();
        Vec gog_vec = ghy_vec * gcx_vec;
        gcx_vec = ghy_vec * og_vec * (Vec(scalar_t(1)) - gcx_vec * gcx_vec) + gcy_vec;

        Vec gig_vec = gcx_vec * cg_vec;
        Vec gfg_vec = gcx_vec * Vec::loadu(cx_ptr + d);
        Vec gcg_vec = gcx_vec * ig_vec;
        gcx_vec = gcx_vec * fg_vec;

        gig_vec = gig_vec * (one_vec - ig_vec) * ig_vec;
        gfg_vec = gfg_vec * (one_vec - fg_vec) * fg_vec;
        gcg_vec = gcg_vec * (one_vec - cg_vec * cg_vec);
        gog_vec = gog_vec * (one_vec - og_vec) * og_vec;

        gig_vec.store(ih + d);
        gfg_vec.store(fh + d);
        gcg_vec.store(ch + d);
        gog_vec.store(oh + d);
        gcx_vec.store(grad_cx_ptr + d);
      }
      for (; d < size; d++) {
        scalar_t ghy = has_grad_hy ? grad_hy_ptr[d] : scalar_t(0);
        scalar_t gcy = has_grad_cy ? grad_cy_ptr[d] : scalar_t(0);

        scalar_t gcx = ::tanh(cy_ptr[d]);
        scalar_t gog = ghy * gcx;
        gcx = ghy * og[d] * (1 - gcx * gcx) + gcy;

        scalar_t gig = gcx * cg[d];
        scalar_t gfg = gcx * cx_ptr[d];
        scalar_t gcg = gcx * ig[d];
        gcx = gcx * fg[d];

        gig = gig * (1 - ig[d]) * ig[d];
        gfg = gfg * (1 - fg[d]) * fg[d];
        gcg = gcg * (1 - cg[d] * cg[d]);
        gog = gog * (1 - og[d]) * og[d];

        ih[d] = gig;
        fh[d] = gfg;
        ch[d] = gcg;
        oh[d] = gog;
        grad_cx_ptr[d] = gcx;
      }
    }
  });
}

static constexpr int64_t GRU_WORKSPACE_MULTIPLIER = 5;

template <typename scalar_t>
void fused_gru_cell_kernel(
    Tensor& hy,
    Tensor& workspace,
    const Tensor& input_gates_,
    const Tensor& hidden_gates_,
    const Tensor& hx_) {
  using Vec = Vec256<scalar_t>;
  int64_t batch_size = hx_.size(0);
  int64_t hidden_size = hx_.size(1);
  int64_t gate_size = hidden_size * 3;

  TORCH_CHECK(input_gates_.sizes().equals({batch_size, gate_size}),
              "expect input_gates size of ", batch_size, ", ", gate_size,
              "], got ", input_gates_.sizes());
  TORCH_CHECK(hidden_gates_.sizes().equals({batch_size, gate_size}),
              "expect hidden_gates size of ", batch_size, ", ", gate_size,
              "], got ", hidden_gates_.sizes());

  auto input_gates = input_gates_.contiguous();
  auto hidden_gates = hidden_gates_.contiguous();
  auto hx = hx_.contiguous();

  auto input_gates_data = input_gates.data_ptr<scalar_t>();
  auto hidden_gates_data = hidden_gates.data_ptr<scalar_t>();
  auto hx_data = hx.data_ptr<scalar_t>();
  auto hy_data = hy.data_ptr<scalar_t>();
  auto workspace_data = workspace.data_ptr<scalar_t>();

  at::parallel_for(0, batch_size, 1,  [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; i++) {
      scalar_t* input_gates_ptr = input_gates_data + i * gate_size;
      scalar_t* hidden_gates_ptr = hidden_gates_data + i * gate_size;
      scalar_t* workspace_ptr = workspace_data + i * hidden_size * GRU_WORKSPACE_MULTIPLIER;

      // input_gates: reset, input, new
      scalar_t* irg = input_gates_ptr;
      scalar_t* iig = input_gates_ptr + hidden_size;
      scalar_t* ing = input_gates_ptr + 2 * hidden_size;
      // hidden_gates: reset, input, new
      scalar_t* hrg = hidden_gates_ptr;
      scalar_t* hig = hidden_gates_ptr + hidden_size;
      scalar_t* hng = hidden_gates_ptr + 2 * hidden_size;
      // workspac: reset, input, new, hx, hn
      scalar_t* wrg = workspace_ptr;
      scalar_t* wig = workspace_ptr + hidden_size;
      scalar_t* wng = workspace_ptr + 2 * hidden_size;
      scalar_t* whx = workspace_ptr + 3 * hidden_size;
      scalar_t* whn = workspace_ptr + 4 * hidden_size;

      scalar_t* hx_ptr = hx_data + i * hidden_size;
      scalar_t* hy_ptr = hy_data + i * hidden_size;

      int64_t size = hidden_size;
      int64_t d = 0;
      for (; d < size - (size % Vec::size()); d += Vec::size()) {
        Vec rg_vec = sigmoid<scalar_t>(Vec::loadu(irg + d) + Vec::loadu(hrg + d));
        Vec ig_vec = sigmoid<scalar_t>(Vec::loadu(iig + d) + Vec::loadu(hig + d));
        Vec hn_vec = Vec::loadu(hng + d);
        Vec ng_vec = (Vec::loadu(ing + d) + rg_vec * hn_vec).tanh();
        Vec hx_vec = Vec::loadu(hx_ptr + d);
        Vec hy_vec = ng_vec + ig_vec * (hx_vec - ng_vec);
        hy_vec.store(hy_ptr + d);
        rg_vec.store(wrg + d);
        ig_vec.store(wig + d);
        ng_vec.store(wng + d);
        hx_vec.store(whx + d);
        hn_vec.store(whn + d);
      }
      for (; d < size; d++) {
        scalar_t rg = sigmoid(irg[d] + hrg[d]);
        scalar_t ig = sigmoid(iig[d] + hig[d]);
        scalar_t hn = hng[d];
        scalar_t ng = ::tanh(ing[d] + rg * hn);
        scalar_t hx = hx_ptr[d];
        scalar_t hy = ng + ig * (hx - ng);
        hy_ptr[d] = hy;
        wrg[d] = rg;
        wig[d] = ig;
        wng[d] = ng;
        whx[d] = hx;
        whn[d] = hn;
      }
    }
  });
}

template <typename scalar_t>
void fused_gru_cell_backward_kernel(
    Tensor& grad_input_gates,
    Tensor& grad_hidden_gates,
    Tensor& grad_hx,
    const Tensor& grad_hy_,
    const Tensor& workspace) {
  using Vec = Vec256<scalar_t>;
  int64_t batch_size = grad_hy_.size(0);
  int64_t hidden_size = grad_hy_.size(1);
  int64_t gate_size = hidden_size * 3;

  TORCH_CHECK(grad_hy_.sizes().equals({batch_size, hidden_size}),
              "expect grad_hy size of ", batch_size, ", ", hidden_size,
              "], got ", grad_hy_.sizes());
  TORCH_CHECK(workspace.sizes().equals({batch_size, hidden_size * GRU_WORKSPACE_MULTIPLIER}),
              "expect workspace size of ", batch_size, ", ", hidden_size * GRU_WORKSPACE_MULTIPLIER,
              "], got ", workspace.sizes());

  auto grad_hy = grad_hy_.contiguous();
  auto grad_hy_data = grad_hy.data_ptr<scalar_t>();
  auto workspace_data = workspace.data_ptr<scalar_t>();
  auto grad_ingates_data = grad_input_gates.data_ptr<scalar_t>();
  auto grad_higates_data = grad_hidden_gates.data_ptr<scalar_t>();
  auto grad_hx_data = grad_hx.data_ptr<scalar_t>();

  at::parallel_for(0, batch_size, 1,  [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; i++) {
      scalar_t* workspace_ptr = workspace_data + i * hidden_size * GRU_WORKSPACE_MULTIPLIER;
      scalar_t* grad_ingates_ptr = grad_ingates_data + i * gate_size;
      scalar_t* grad_higates_ptr = grad_higates_data + i * gate_size;

      // workspace: reset, input, new, hx, hn
      scalar_t* rg = workspace_ptr;
      scalar_t* ig = workspace_ptr + hidden_size;
      scalar_t* ng = workspace_ptr + 2 * hidden_size;
      scalar_t* hx = workspace_ptr + 3 * hidden_size;
      scalar_t* hn = workspace_ptr + 4 * hidden_size;
      // input_gates: reset, input, new
      scalar_t* gir = grad_ingates_ptr;
      scalar_t* gii = grad_ingates_ptr + hidden_size;
      scalar_t* gin = grad_ingates_ptr + 2 * hidden_size;
      // hidden_gates: reset, input, new
      scalar_t* ghr = grad_higates_ptr;
      scalar_t* ghi = grad_higates_ptr + hidden_size;
      scalar_t* ghn = grad_higates_ptr + 2 * hidden_size;

      scalar_t* grad_hy_ptr = grad_hy_data + i * hidden_size;
      scalar_t* grad_hx_ptr = grad_hx_data + i * hidden_size;

      int64_t size = hidden_size;
      int64_t d = 0;
      auto one_vec = Vec(scalar_t(1));
      auto zero_vec = Vec(scalar_t(0));
      for (; d < size - (size % Vec::size()); d += Vec::size()) {
        Vec rg_vec = Vec::loadu(rg + d);
        Vec ig_vec = Vec::loadu(ig + d);
        Vec ng_vec = Vec::loadu(ng + d);
        Vec hx_vec = Vec::loadu(hx + d);
        Vec hn_vec = Vec::loadu(hn + d);
        Vec ghy_vec = Vec::loadu(grad_hy_ptr + d);

        Vec gig_vec = ghy_vec * (hx_vec - ng_vec) * (one_vec - ig_vec) * ig_vec;
        Vec ghx_vec = ghy_vec * ig_vec;
        Vec gin_vec = ghy_vec * (one_vec - ig_vec) * (one_vec - ng_vec * ng_vec);
        Vec ghn_vec = gin_vec * rg_vec;
        Vec grg_vec = gin_vec * hn_vec * (one_vec - rg_vec) * rg_vec;

        grg_vec.store(gir + d);
        gig_vec.store(gii + d);
        gin_vec.store(gin + d);
        grg_vec.store(ghr + d);
        gig_vec.store(ghi + d);
        ghn_vec.store(ghn + d);
        ghx_vec.store(grad_hx_ptr + d);
      }
      for (; d < size; d++) {
        scalar_t ghy = grad_hy_ptr[d];

        scalar_t gig = ghy * (hx[d] - ng[d]) * (1 - ig[d]) * ig[d];
        scalar_t ghx = ghy * ig[d];
        scalar_t gin_ = ghy * (1 - ig[d]) * (1 - ng[d] * ng[d]);
        scalar_t ghn_ = gin_ * rg[d];
        scalar_t grg = gin_ * hn[d] * (1 - rg[d]) * rg[d];

        gir[d] = grg;
        gii[d] = gig;
        gin[d] = gin_;
        ghr[d] = grg;
        ghi[d] = gig;
        ghn[d] = ghn_;
        grad_hx_ptr[d] = ghx;
      }
    }
  });
}

void fused_lstm_cell_kernel_impl(
    Tensor& hy,
    Tensor& cy,
    Tensor& workspace,
    const Tensor& input_gates,
    const Tensor& hidden_gates,
    const Tensor& cx) {
  AT_DISPATCH_FLOATING_TYPES(input_gates.scalar_type(), "fused_lstm_cell", [&] {
    fused_lstm_cell_kernel<scalar_t>(hy, cy, workspace, input_gates, hidden_gates, cx);
  });
}

void fused_lstm_cell_backward_kernel_impl(
    Tensor& grad_gates,
    Tensor& grad_cx,
    const Tensor& grad_hy,
    const Tensor& grad_cy,
    const Tensor& cx,
    const Tensor& cy,
    const Tensor& workspace) {
  AT_DISPATCH_FLOATING_TYPES(workspace.scalar_type(), "fused_lstm_cell_backward", [&] {
    fused_lstm_cell_backward_kernel<scalar_t>(grad_gates, grad_cx, grad_hy, grad_cy, cx, cy, workspace);
  });
}

void fused_gru_cell_kernel_impl(
    Tensor& hy,
    Tensor& workspace,
    const Tensor& input_gates,
    const Tensor& hidden_gates,
    const Tensor& hx) {
  AT_DISPATCH_FLOATING_TYPES(input_gates.scalar_type(), "fused_gru_cell", [&] {
    fused_gru_cell_kernel<scalar_t>(hy, workspace, input_gates, hidden_gates, hx);
  });
}

void fused_gru_cell_backward_kernel_impl(
    Tensor& grad_input_gates,
    Tensor& grad_hidden_gates,
    Tensor& grad_hx,
    const Tensor& grad_hy,
    const Tensor& workspace) {
  AT_DISPATCH_FLOATING_TYPES(grad_hy.scalar_type(), "fused_gru_cell_backward", [&] {
    fused_gru_cell_backward_kernel<scalar_t>(grad_input_gates, grad_hidden_gates, grad_hx, grad_hy, workspace);
  });
}

} // anonymous namespace

REGISTER_DISPATCH(fused_lstm_cell_stub, &fused_lstm_cell_kernel_impl);
REGISTER_DISPATCH(fused_lstm_cell_backward_stub, &fused_lstm_cell_backward_kernel_impl);
REGISTER_DISPATCH(fused_gru_cell_stub, &fused_gru_cell_kernel_impl);
REGISTER_DISPATCH(fused_gru_cell_backward_stub, &fused_gru_cell_backward_kernel_impl);
}} // at::native
