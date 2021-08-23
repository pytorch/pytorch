#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/operators/conv2d.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

namespace {

void assert_dims_constant(const BufHandle& buf) {
  for (auto const& dim : buf.node()->dims()) {
    TORCH_INTERNAL_ASSERT(dim->isConstant());
  }
}

using InitFunc = std::function<ExprHandle(const std::vector<VarHandle>&)>;

// schedule for kernel size 3x3
Tensor* conv2d_depthwise_static_schedule1(
    BufHandle input,
    BufHandle weight,
    const InitFunc& init_func,
    int stride,
    int pad,
    int groups) {
  TORCH_INTERNAL_ASSERT(input.ndim() == 4);
  TORCH_INTERNAL_ASSERT(weight.ndim() == 4);

  assert_dims_constant(input);
  assert_dims_constant(weight);

  auto const& N = immediateAs<int>(input.dim(0));
  auto const& C = immediateAs<int>(input.dim(1));
  auto const& H = immediateAs<int>(input.dim(2));
  auto const& W = immediateAs<int>(input.dim(3));

  auto const& K = immediateAs<int>(weight.dim(0));
  auto const& CperG = immediateAs<int>(weight.dim(1));
  auto const& R = immediateAs<int>(weight.dim(2));
  auto const& S = immediateAs<int>(weight.dim(3));

  TORCH_INTERNAL_ASSERT(C == K && K == groups && CperG == 1);
  TORCH_INTERNAL_ASSERT(R == S);

  auto OH = (H - R + 2 * pad) / stride + 1;
  auto OW = (W - S + 2 * pad) / stride + 1;

  Tensor* conv = Reduce(
      "conv2d_depthwise",
      {{N, "n"}, {K, "k"}, {OH, "oh"}, {OW, "ow"}},
      Sum(),
      [&](const std::vector<VarHandle>& v) { return init_func(v); },
      [&](const std::vector<VarHandle>& v) {
        auto const& n = v[0];
        auto const& k = v[1];
        auto const& oh = v[2];
        auto const& ow = v[3];
        auto const& c = v[4];
        auto const& r = v[5];
        auto const& s = v[6];
        auto cond = CompareSelect::make(oh * stride - pad + r, 0, 1, 0, kLT);
        cond = CompareSelect::make(ow * stride - pad + s, 0, 1, cond, kLT);
        cond = CompareSelect::make(oh * stride - pad + r, H, 1, cond, kGE);
        cond = CompareSelect::make(ow * stride - pad + s, W, 1, cond, kGE);
        auto in = ifThenElse(
            cond,
            0.f,
            input.load(n, k, oh * stride - pad + r, ow * stride - pad + s));
        return in * weight.load(k, c, r, s);
      },
      {{C / groups, "c"}, {R, "r"}, {S, "s"}});

  LoopNest nest({conv});

  constexpr int kLoopH = 2, kLoopW = 3;
  if (R == 3 && stride == 2 && pad == 1) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    ForPtr head, tail;
    auto loops = nest.getLoopStmtsFor(conv);
    nest.sliceHead(loops[kLoopW], 2, &head, &tail);
    loops = nest.getLoopStmtsFor(conv);
    nest.sliceHead(loops[kLoopH], 2, &head, &tail);
  } else if (R == 3 && stride == 1 && pad == 1) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    ForPtr main, peeled;
    auto loops = nest.getAllLoopNestsWritingToBuf(conv->buf());
    main = loops[1][kLoopW];
    nest.sliceHead(main, 1, &peeled, &main);
    nest.sliceTail(main, 1, &main, &peeled);
    main = LoopNest::getParentLoop(main);
    nest.sliceHead(main, 1, &peeled, &main);
    nest.sliceTail(main, 1, &main, &peeled);
  }

  return new Tensor(conv->buf(), nest.root_stmt());
}

// schedule for kernel size 5x5 + batch size 1
Tensor* conv2d_depthwise_static_schedule2(
    BufHandle input,
    BufHandle weight,
    const InitFunc& init_func,
    int stride,
    int pad,
    int groups) {
  TORCH_INTERNAL_ASSERT(input.ndim() == 4);
  TORCH_INTERNAL_ASSERT(weight.ndim() == 4);

  assert_dims_constant(input);
  assert_dims_constant(weight);

  auto const& N = immediateAs<int>(input.dim(0));
  auto const& C = immediateAs<int>(input.dim(1));
  auto const& H = immediateAs<int>(input.dim(2));
  auto const& W = immediateAs<int>(input.dim(3));

  auto const& K = immediateAs<int>(weight.dim(0));
  auto const& CperG = immediateAs<int>(weight.dim(1));
  auto const& R = immediateAs<int>(weight.dim(2));
  auto const& S = immediateAs<int>(weight.dim(3));

  TORCH_INTERNAL_ASSERT(C == K && K == groups && CperG == 1);
  TORCH_INTERNAL_ASSERT(R == S);
  TORCH_INTERNAL_ASSERT(N == 1);

  // check if #channel is a multiplier of 8
  TORCH_INTERNAL_ASSERT(C % 8 == 0);

  // pack image into NCHWc8 format
  auto Cblock = ExprHandle(8);
  auto Cchunck = C / 8;
  auto input_NCHWc = Compute(
      "input_NCHWc",
      {{N, "n"}, {Cchunck, "cc"}, {H, "h"}, {W, "w"}, {Cblock, "cb"}},
      [&](const std::vector<VarHandle>& v) {
        auto const& n = v[0];
        auto const& cc = v[1];
        auto const& h = v[2];
        auto const& w = v[3];
        auto const& cb = v[4];
        return input.load(n, cc * Cblock + cb, h, w);
      });
      input_NCHWc->buf()->disable_inline();

  auto input_NCHWc_padded = Compute(
      "input_NCHWc_padded",
      {{N, "n"},
       {Cchunck, "cc"},
       {H + pad * 2, "h"},
       {W + pad * 2, "w"},
       {Cblock, "cb"}},
      [&](const std::vector<VarHandle>& v) {
        auto const& n = v[0];
        auto const& cc = v[1];
        auto const& h = v[2];
        auto const& w = v[3];
        auto const& cb = v[4];
        auto cond = CompareSelect::make(w, W + pad, 1, 0, kGE);
        cond = CompareSelect::make(w, pad, 1, cond, kLT);
        cond = CompareSelect::make(h, H + pad, 1, cond, kGE);
        cond = CompareSelect::make(h, pad, 1, cond, kLT);
        return ifThenElse(
            cond, 0.f, input_NCHWc->load(n, cc, h - pad, w - pad, cb));
      });
      input_NCHWc_padded->buf()->disable_inline();

  // pack kernel into NCHWc8 format
  auto weight_NCHWc = Compute(
      "weight_NCHWc",
      {{Cchunck, "cc"}, {C / groups, "c"}, {R, "r"}, {S, "s"}, {Cblock, "cb"}},
      [&](const std::vector<VarHandle>& v) {
        auto const& cc = v[0];
        auto const& c = v[1];
        auto const& r = v[2];
        auto const& s = v[3];
        auto const& cb = v[4];
        return weight.load(cc * Cblock + cb, c, r, s);
      });
      weight_NCHWc->buf()->disable_inline();

  // compute conv_NCHWc8
  auto OH = (H - R + pad * 2) / stride + 1;
  auto OW = (W - S + pad * 2) / stride + 1;
  auto conv_depthwise_NCHWc = Reduce(
      "conv2d_depthwise_NCHWc",
      {{N, "n"}, {Cchunck, "cc"}, {OH, "oh"}, {OW, "ow"}, {Cblock, "cb"}},
      Sum(),
      //[&](const std::vector<VarHandle>& v) { return init_func(v); },
      [&](const std::vector<VarHandle>& v) { return FloatImm::make(0.f); },
      [&](const std::vector<VarHandle>& v) {
        auto const& n = v[0];
        auto const& cc = v[1];
        auto const& oh = v[2];
        auto const& ow = v[3];
        auto const& cb = v[4];
        auto const& c = v[5];
        auto const& r = v[6];
        auto const& s = v[7];
        return input_NCHWc_padded->load(
                   n, cc, oh * stride + r, ow * stride + s, cb) *
            weight_NCHWc->load(cc, c, r, s, cb);
      },
      {{C / groups, "c"}, {R, "r"}, {S, "s"}});
      conv_depthwise_NCHWc->buf()->disable_inline();

  // unpack conv from NCHWc8 to NCHW
  auto conv_depthwise = Compute(
      "conv2d_depthwise",
      {{N, "n"}, {K, "k"}, {OH, "oh"}, {OW, "ow"}},
      [&](const std::vector<VarHandle>& v) {
        auto const& n = v[0];
        auto const& k = v[1];
        auto const& oh = v[2];
        auto const& ow = v[3];
        return conv_depthwise_NCHWc->load(n, k / Cblock, oh, ow, k % Cblock);
      });
      conv_depthwise->buf()->disable_inline();

  auto stmt = new Block(
      {input_NCHWc->stmt(),
       input_NCHWc_padded->stmt(),
       weight_NCHWc->stmt(),
       conv_depthwise_NCHWc->stmt(),
       conv_depthwise->stmt()});
  GRAPH_DEBUG("Conv2d Original Stmt:\n", std::to_string(stmt), "\n");
  auto nest = LoopNest(stmt, {conv_depthwise->buf()});

  /*** scheduling ***/
  // padded image: flatten `n`, `cc` and `h` and vectorize `cb`
  // TODO: the flattened loop should be parallelized when NNC multi-threading is
  // enabled.
  auto loops = nest.getLoopStmtsFor(input_NCHWc_padded);
  For* flattened;
  nest.flatten({loops[0], loops[1], loops[2]}, &flattened);
  GRAPH_DEBUG(
      "flattened image padding stmts:\n",
      std::to_string(nest.root_stmt()),
      "\n");

  // conv NCHWc8: cache access on `w` axis
  // TODO: fix `cacheAccesses` to either use thread-private memory to avoid
  // Write-after-Write data dependency or use glocal memory whose accesses
  // through multiple threads are controlled by locking/unlocking.
  loops = nest.getLoopStmtsFor(conv_depthwise_NCHWc);
  auto cache_ref = nest.cacheAccesses(
      conv_depthwise_NCHWc->buf(), "conv_NCHWc_cache", loops[3]);
  nest.simplify();
  nest.eliminateDeadStores();
  cache_ref.first->disable_inline();
  GRAPH_DEBUG(
      "cache access conv NCHWc8:\n", std::to_string(nest.root_stmt()), "\n");

  // conv: compute conv at `w` of conv NCHWc8
  // TODO: fix `computeAt` and use it to replace
  // `distribute->split->reorder->fuse->compressBuf`.
  loops = nest.getLoopStmtsFor(cache_ref.first);
  auto cc = loops[0], h = loops[1], w = loops[2], cb = loops[3], kh = loops[4],
       kw = loops[5];
  loops = nest.distributeLoop(cb);
  GRAPH_DEBUG("distribute:\n", std::to_string(nest.root_stmt()), "\n");
  cb = loops[1];
  kh = nest.getLoopAt(cb, {0});
  kw = nest.getLoopAt(cb, {0, 0});
  // TODO: fix `reorder` to use `nest.reorder({cb, kh, kw}, {2, 0, 1})`
  nest.reorder({cb, kh}, {1, 0});
  nest.reorder({cb, kw}, {1, 0});
  GRAPH_DEBUG("reorder:\n", std::to_string(nest.root_stmt()), "\n");

  loops = nest.getLoopStmtsFor(conv_depthwise);
  For *cinner, *ctail;
  nest.splitWithTail(loops[0], 8, &cinner, &ctail);
  nest.simplify();
  loops = nest.getLoopStmtsFor(conv_depthwise);
  // TODO: fix `reorder` to use `nest.reorder({loops[1], loops[2], loops[3]},
  // {2, 0, 1})`
  nest.reorder({loops[1], loops[2]}, {1, 0});
  nest.reorder({loops[1], loops[3]}, {1, 0});
  GRAPH_DEBUG("split:\n", std::to_string(nest.root_stmt()), "\n");

  loops = nest.getLoopStmtsFor(conv_depthwise);
  auto loops_NCHWc = nest.getLoopStmtsFor(conv_depthwise_NCHWc);
  For* fused_conv;
  LoopNest::unsafeFuseLoops({loops_NCHWc[0], loops[0]}, &fused_conv);
  auto a = nest.getLoopAt(fused_conv, {0});
  auto b = nest.getLoopAt(fused_conv, {1});
  LoopNest::unsafeFuseLoops({a, b}, &fused_conv);
  GRAPH_DEBUG("fuse:\n", std::to_string(nest.root_stmt()), "\n");
  nest.compressBuffer(conv_depthwise_NCHWc->buf(), fused_conv);
  GRAPH_DEBUG("buffer compression:\n", std::to_string(nest.root_stmt()), "\n");
  GRAPH_DEBUG(
      "compute conv at w axis of conv NCHWc8:\n",
      std::to_string(nest.root_stmt()),
      "\n");

  loops = nest.getLoopStmtsFor(conv_depthwise);
  // conv NCHWc8: vectorize cache initialization
  nest.vectorize(nest.getLoopAt(loops[0], {0, 0, 0}));
  // conv NCHWc8: vectorize computation
  nest.vectorize(nest.getLoopAt(loops[0], {0, 0, 1, 0, 0}));
  // conv:  vectorization
  nest.vectorize(nest.getLoopAt(loops[0], {0, 2, 0}));
  GRAPH_DEBUG("vectorize c8:\n", std::to_string(nest.root_stmt()), "\n");

  // conv NCHWc8: unroll `kw` reduction axis
  nest.unroll(nest.getLoopAt(loops[0], {0, 0, 1, 0}));
  // conv: unroll `w` axis
  nest.unroll(nest.getLoopAt(loops[0], {0, 2}));
  GRAPH_DEBUG("unrolled:\n", std::to_string(nest.root_stmt()), "\n");

  // conv: flatten `cc` and `h`
  // TODO: the flattened loop should be parallelized when NNC multi-threading is
  // enabled.
  For* flattened_conv;
  nest.flatten({loops[0], loops[1]}, &flattened_conv);
  GRAPH_DEBUG("flattened conv:\n", std::to_string(nest.root_stmt()), "\n");

  // padded input NCHWc8: vectorization
  loops = nest.getLoopStmtsFor(input_NCHWc_padded);
  nest.vectorize(loops[2]);
  GRAPH_DEBUG("final conv stmt:\n", std::to_string(nest.root_stmt()), "\n");

  return new Tensor(conv_depthwise->buf(), nest.root_stmt());
}

Tensor* conv2d_depthwise_dynamic(
    BufHandle input,
    BufHandle weight,
    const InitFunc& init_func,
    ExprHandle N,
    ExprHandle C,
    ExprHandle H,
    ExprHandle W,
    ExprHandle K,
    ExprHandle CperG,
    ExprHandle R,
    ExprHandle S,
    ExprHandle stride,
    ExprHandle pad,
    ExprHandle groups) {
  TORCH_INTERNAL_ASSERT(input.ndim() == 4);
  TORCH_INTERNAL_ASSERT(weight.ndim() == 4);

  auto OH = (H - R + pad * 2) / stride + 1;
  auto OW = (W - S + pad * 2) / stride + 1;

  return Reduce(
      "conv2d_depthwise",
      {{N, "n"}, {K, "k"}, {OH, "oh"}, {OW, "ow"}},
      Sum(),
      [&](const std::vector<VarHandle>& v) { return init_func(v); },
      [&](const std::vector<VarHandle>& v) {
        auto const& n = v[0];
        auto const& k = v[1];
        auto const& oh = v[2];
        auto const& ow = v[3];
        auto const& c = v[4];
        auto const& r = v[5];
        auto const& s = v[6];
        auto cond = CompareSelect::make(oh * stride - pad + r, 0, 1, 0, kLT);
        cond = CompareSelect::make(ow * stride - pad + s, 0, 1, cond, kLT);
        cond = CompareSelect::make(oh * stride - pad + r, H, 1, cond, kGE);
        cond = CompareSelect::make(ow * stride - pad + s, W, 1, cond, kGE);
        auto in = ifThenElse(
            cond,
            0.f,
            input.load(n, k, oh * stride - pad + r, ow * stride - pad + s));
        return in * weight.load(k, c, r, s);
      },
      {{C / groups, "c"}, {R, "r"}, {S, "s"}});
}

} // namespace

Tensor* conv2d_depthwise(
    BufHandle input,
    BufHandle weight,
    BufHandle bias,
    int stride,
    int pad,
    int groups) {
  assert_dims_constant(bias);
  auto init_func = [&](const std::vector<VarHandle>& v) {
    return bias.load(v[1]);
  };

  auto KH = immediateAs<int>(weight.dim(2));
  auto KW = immediateAs<int>(weight.dim(3));
  if ((KH == 3) && (KW == 3)) {
    return conv2d_depthwise_static_schedule1(
        input, weight, init_func, stride, pad, groups);
  }
  if ((KH == 5) && (KW == 5)) {
    return conv2d_depthwise_static_schedule2(
        input, weight, init_func, stride, pad, groups);
  }
  TORCH_INTERNAL_ASSERT(0);
}

Tensor* conv2d_depthwise(
    BufHandle input,
    BufHandle weight,
    int stride,
    int pad,
    int groups) {
  auto init_func = [](const std::vector<VarHandle>& v) {
    return ExprHandle(Sum().initializer());
  };

  auto KH = immediateAs<int>(weight.dim(2));
  auto KW = immediateAs<int>(weight.dim(3));
  if ((KH == 3) && (KW == 3)) {
    return conv2d_depthwise_static_schedule1(
        input, weight, init_func, stride, pad, groups);
  }
  if ((KH == 5) && (KW == 5)) {
    return conv2d_depthwise_static_schedule2(
        input, weight, init_func, stride, pad, groups);
  }
  TORCH_INTERNAL_ASSERT(0);
}

Tensor* conv2d_depthwise(
    BufHandle input,
    BufHandle weight,
    BufHandle bias,
    ExprHandle N,
    ExprHandle C,
    ExprHandle H,
    ExprHandle W,
    ExprHandle K,
    ExprHandle CperG,
    ExprHandle R,
    ExprHandle S,
    ExprHandle stride,
    ExprHandle pad,
    ExprHandle groups) {
  assert_dims_constant(bias);
  auto init_func = [&](const std::vector<VarHandle>& v) {
    return bias.load(v[1]);
  };
  return conv2d_depthwise_dynamic(
      input,
      weight,
      init_func,
      N,
      C,
      H,
      W,
      K,
      CperG,
      R,
      S,
      stride,
      pad,
      groups);
}

Tensor* conv2d_depthwise(
    BufHandle input,
    BufHandle weight,
    ExprHandle N,
    ExprHandle C,
    ExprHandle H,
    ExprHandle W,
    ExprHandle K,
    ExprHandle CperG,
    ExprHandle R,
    ExprHandle S,
    ExprHandle stride,
    ExprHandle pad,
    ExprHandle groups) {
  auto init_func = [](const std::vector<VarHandle>& v) {
    return ExprHandle(Sum().initializer());
  };
  return conv2d_depthwise_dynamic(
      input,
      weight,
      init_func,
      N,
      C,
      H,
      W,
      K,
      CperG,
      R,
      S,
      stride,
      pad,
      groups);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
