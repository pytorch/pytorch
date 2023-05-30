#include <torch/csrc/jit/runtime/static/te_wrapper.h>

#include <ATen/CPUFunctions.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/operators/misc.h>
#include <torch/csrc/jit/tensorexpr/operators/operators.h>

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

// Use the width of an AVX-512 vector by default; this happens to work OK for
// AVX2 as well. Some ops benefit from using multiple AVX ports, in which case
// they are vectorized by twice this constant.  An exception is logit, since it
// contains FP divide, which is single-ported.
static constexpr int kVectorWidth = 16;

#ifdef TORCH_ENABLE_LLVM

void TEWrapper::update(std::unique_ptr<LLVMCodeGen>&& cg_) {
  cg = std::move(cg_);
}

void TEWrapper::call(const std::vector<void*>& args) {
  cg->call_raw(args);
}

void optimizePointwise(LoopNest* ln, Tensor target, int width) {
  std::vector<ForPtr> loops = ln->getLoopStmtsFor(target);
  ForPtr inner, tail;
  TORCH_CHECK(loops.size() > 0, "No loops created for pointwise op");
  ln->splitWithTail(loops[0], width, &inner, &tail);
  ln->vectorize(inner);
}

std::shared_ptr<TEWrapper> wrapTECompute(
    std::shared_ptr<TEWrapper> wrap,
    Tensor out,
    std::vector<CodeGen::BufferArg> args,
    int width = kVectorWidth) {
  LoopNest ln({out});
  optimizePointwise(&ln, out, width);
  ln.prepareForCodegen();
  StmtPtr s = ln.root_stmt();
  s = IRSimplifier::simplify(s);
  args.insert(args.begin(), out);
  auto cg = std::make_unique<LLVMCodeGen>(s, args);
  cg->cleanup_memory();
  wrap->update(std::move(cg));
  return wrap;
}

std::shared_ptr<TEWrapper> wrapTECompute(
    std::shared_ptr<TEWrapper> wrap,
    LoopNest* ln,
    std::vector<CodeGen::BufferArg> args) {
  auto cg = std::make_unique<LLVMCodeGen>(ln->root_stmt(), args);
  wrap->update(std::move(cg));
  return wrap;
}

#else

void TEWrapper::call(const std::vector<void*>& args) {
  DCHECK(0 && "Invalid call");
}

std::shared_ptr<TEWrapper> wrapTECompute(
    std::shared_ptr<TEWrapper> wrap,
    Tensor out,
    std::vector<CodeGen::BufferArg> args,
    int width = kVectorWidth) {
  return wrap;
}

std::shared_ptr<TEWrapper> wrapTECompute(
    std::shared_ptr<TEWrapper> wrap,
    LoopNest* ln,
    std::vector<CodeGen::BufferArg> args) {
  return wrap;
}

#endif

namespace {

std::mutex& getNNCCacheMutex() {
  static std::mutex nncCacheMutex;
  return nncCacheMutex;
}

c10::FastMap<NodeKind, std::shared_ptr<TEWrapper>>& getNNCCache() {
  static c10::FastMap<NodeKind, std::shared_ptr<TEWrapper>> nncCache;
  return nncCache;
}

std::shared_ptr<TEWrapper> lookupNNCCache(NodeKind kind) {
  std::lock_guard<std::mutex> lock(getNNCCacheMutex());
  auto it = getNNCCache().find(kind);
  if (it != getNNCCache().end()) {
    return it->second;
  }
  return nullptr;
}

void updateNNCCache(NodeKind kind, std::shared_ptr<TEWrapper> code) {
  std::lock_guard<std::mutex> lock(getNNCCacheMutex());
  getNNCCache()[kind] = code;
}

} // namespace

std::shared_ptr<TEWrapper> createDiv() {
  auto wrap = lookupNNCCache(aten::div);
  if (wrap) {
    return wrap;
  }
  wrap = std::make_shared<TEWrapper>();

  auto dim = VarHandle("dim", kInt);
  auto mode = VarHandle("mode", kInt);
  BufHandle A("A", {dim}, kFloat);
  BufHandle B("B", {dim}, kFloat);

  using axis = const VarHandle&;
  Tensor C = Compute("C", {dim}, [&](axis x) {
    auto true_div_result = A.load(x) / B.load(x);

    auto mode_default = IntImm::make(0);
    auto mode_trunc = IntImm::make(1);
    auto mode_floor = IntImm::make(2);

    // this is a glorified ternary choice operator train
    return CompareSelect::make(
        mode,
        mode_default,
        true_div_result,
        CompareSelect::make(
            mode,
            mode_trunc,
            trunc(true_div_result),
            floor(true_div_result),
            kEQ),
        kEQ);
  });

  wrap = wrapTECompute(wrap, C, {A, B, mode, dim});

  updateNNCCache(aten::div, wrap);
  return wrap;
}

std::shared_ptr<TEWrapper> createLogit() {
  auto wrap = lookupNNCCache(aten::logit);
  if (wrap) {
    return wrap;
  }
  wrap = std::make_shared<TEWrapper>();
  auto N = VarHandle("N", kInt);
  auto C = VarHandle("C", kFloat);
  BufHandle A("A", {N}, kFloat);
  Tensor B = Compute("B", {N}, [&](const VarHandle& i) {
    auto A_elem = [&]() {
      auto elem = A.load(i);
      auto one = FloatImm::make(1.0f);
      const auto& min = C;
      auto max = one - C;
      elem = CompareSelect::make(elem, min, min, elem, kLT);
      return CompareSelect::make(elem, max, max, elem, kGT);
    }();
    return log_vml(A_elem / (FloatImm::make(1.0f) - A_elem));
  });
  wrap = wrapTECompute(wrap, B, {A, N, C});
  updateNNCCache(aten::logit, wrap);
  return wrap;
}

std::shared_ptr<TEWrapper> createRelu() {
  auto wrap = lookupNNCCache(aten::relu);
  if (wrap) {
    return wrap;
  }
  wrap = std::make_shared<TEWrapper>();
  auto N = VarHandle("N", kInt);
  BufHandle A("A", {N}, kFloat);
  Tensor B = Compute("B", {N}, [&](const VarHandle& i) {
    auto zero = FloatImm::make(0.f);
    auto a = A.load(i);
    return CompareSelect::make(a, zero, zero, a, kLT);
  });
  wrap = wrapTECompute(wrap, B, {A, N});
  updateNNCCache(aten::relu, wrap);
  return wrap;
}

std::shared_ptr<TEWrapper> createTanh() {
  auto wrap = lookupNNCCache(aten::tanh);
  if (wrap) {
    return wrap;
  }
  wrap = std::make_shared<TEWrapper>();
  auto N = VarHandle("N", kInt);
  BufHandle A("A", {N}, kFloat);
  Tensor B = Compute("B", {N}, [&](const VarHandle& i) {
    auto a = A.load(i);
    return fast_tanh(a);
  });
  wrap = wrapTECompute(wrap, B, {A, N});
  updateNNCCache(aten::tanh, wrap);
  return wrap;
}

std::shared_ptr<TEWrapper> createSigmoid() {
  auto wrap = lookupNNCCache(aten::sigmoid);
  if (wrap) {
    return wrap;
  }
  wrap = std::make_shared<TEWrapper>();
  auto N = VarHandle("N", kInt);
  BufHandle A("A", {N}, kFloat);
  Tensor B = Compute(
      "B", {N}, [&](const VarHandle& i) { return fast_sigmoid(A.load(i)); });
  wrap = wrapTECompute(wrap, B, {A, N});
  updateNNCCache(aten::sigmoid, wrap);
  return wrap;
}

std::shared_ptr<TEWrapper> createClamp() {
  static auto clamp_symbol = c10::Symbol::fromQualString("aten::clamp");
  auto wrap = lookupNNCCache(clamp_symbol);
  if (wrap) {
    return wrap;
  }
  wrap = std::make_shared<TEWrapper>();
  auto N = VarHandle("N", kInt);
  auto min_handle = VarHandle("min", kFloat);
  auto max_handle = VarHandle("max", kFloat);

  BufHandle A("A", {N}, kFloat);
  Tensor result = Compute("aten_clamp", {N}, [&](const VarHandle& i) {
    auto a = A.load(i);
    return tensorexpr::clamp(min_handle, max_handle, a);
  });
  wrap = wrapTECompute(wrap, result, {A, min_handle, max_handle, N});
  updateNNCCache(clamp_symbol, wrap);
  return wrap;
}

std::shared_ptr<TEWrapper> createClampNanToNum() {
  static auto symbol =
      c10::Symbol::fromQualString("static_runtime::clamp_nan_to_num");
  auto wrap = lookupNNCCache(symbol);
  if (wrap) {
    return wrap;
  }
  wrap = std::make_shared<TEWrapper>();
  auto N = VarHandle("N", kInt);
  auto min_handle = VarHandle("min", kFloat);
  auto max_handle = VarHandle("max", kFloat);
  auto nan_replace_val = VarHandle("nan_replace_val", kFloat);

  BufHandle A("A", {N}, kFloat);
  Tensor result = Compute("aten_clamp", {N}, [&](const VarHandle& i) {
    auto a = A.load(i);
    auto clamp = tensorexpr::clamp(min_handle, max_handle, a);
    auto is_nan = tensorexpr::isnan(clamp);
    auto nans_replaced =
        tensorexpr::CompareSelect::make(is_nan, 1, nan_replace_val, clamp, kEQ);
    return nans_replaced;
  });
  wrap = wrapTECompute(
      wrap, result, {A, min_handle, max_handle, nan_replace_val, N});
  updateNNCCache(symbol, wrap);
  return wrap;
}

std::shared_ptr<TEWrapper> createSignedLog1p() {
  static auto signed_log1p_symbol =
      c10::Symbol::fromQualString("static_runtime::signed_log1p");
  auto wrap = lookupNNCCache(signed_log1p_symbol);
  if (wrap) {
    return wrap;
  }
  wrap = std::make_shared<TEWrapper>();
  auto N = VarHandle("N", kInt);
  BufHandle A("A", {N}, kFloat);
  Tensor abs_result = Compute("aten_abs", {N}, [&](const VarHandle& i) {
    return tensorexpr::abs(A.load(i));
  });
  Tensor log1p_result = Compute("aten_log1p", {N}, [&](const VarHandle& i) {
    return log1p(abs_result.load(i));
  });
  Tensor sign = computeSign({A}, {N});
  Tensor output = Compute("aten_mul", {N}, [&](const VarHandle& i) {
    return sign.load(i) * log1p_result.load(i);
  });
  LoopNest ln({output}, {abs_result, log1p_result, sign, output});
  GRAPH_DEBUG("Original stmt: ", *ln.root_stmt());
  ln.inlineIntermediateBufs(true);
  ln.prepareForCodegen();
  ln.simplify();
  ln.vectorizeInnerLoops();
  ln.simplify();
  GRAPH_DEBUG("Final stmt: ", *ln.root_stmt());
  wrap = wrapTECompute(wrap, &ln, {output, A, N});
  updateNNCCache(signed_log1p_symbol, wrap);
  return wrap;
}

} // namespace jit
} // namespace torch
