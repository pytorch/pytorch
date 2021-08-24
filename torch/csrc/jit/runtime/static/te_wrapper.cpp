#include <torch/csrc/jit/runtime/static/te_wrapper.h>

#include <ATen/CPUFunctions.h>
#include <torch/csrc/jit/ir/ir.h>

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

bool TEWrapper::supports(const at::Tensor& t) {
  return t.is_contiguous() && t.dtype().Match<float>();
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
    Placeholder& in,
    Tensor out,
    VarHandle& dim,
    int width = kVectorWidth) {
  LoopNest ln({out});
  optimizePointwise(&ln, out, width);
  ln.prepareForCodegen();
  StmtPtr s = ln.root_stmt();
  s = IRSimplifier::simplify(s);
  std::vector<CodeGen::BufferArg> args;
  args.emplace_back(out);
  args.emplace_back(in);
  args.emplace_back(dim);
  auto cg = std::make_unique<LLVMCodeGen>(s, args);
  wrap->update(std::move(cg));
  return wrap;
};

#else

void TEWrapper::call(const std::vector<void*>& args) {
  DCHECK(0 && "Invalid call");
}

bool TEWrapper::supports(const at::Tensor& t) {
  return false;
}

std::shared_ptr<TEWrapper> wrapTECompute(
    std::shared_ptr<TEWrapper> wrap,
    Placeholder& in,
    Tensor out,
    VarHandle& dim,
    int width = kVectorWidth) {
  return wrap;
};

#endif

namespace {

std::mutex& getNNCCacheMutex() {
  static std::mutex nncCacheMutex;
  return nncCacheMutex;
}

std::unordered_map<NodeKind, std::shared_ptr<TEWrapper>>& getNNCCache() {
  static std::unordered_map<NodeKind, std::shared_ptr<TEWrapper>> nncCache;
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

std::shared_ptr<TEWrapper> createLogit(c10::optional<float> clamp) {
  // TODO: Use NNC cache for this op.
  auto wrap = std::make_shared<TEWrapper>();
  auto N = VarHandle("N", kInt);
  Placeholder A("A", kFloat, {N});
  Tensor B = Compute("B", {N}, [&](const VarHandle& i) {
    auto A_elem = [&]() {
      if (!clamp) {
        return A.load(i);
      } else {
        auto elem = A.load(i);
        auto min = FloatImm::make(*clamp);
        auto max = FloatImm::make(1.0f - *clamp);
        elem = CompareSelect::make(elem, min, min, elem, kLT);
        return CompareSelect::make(elem, max, max, elem, kGT);
      }
    }();
    return log_vml(A_elem / (FloatImm::make(1.0f) - A_elem));
  });
  return wrapTECompute(wrap, A, B, N);
}

std::shared_ptr<TEWrapper> createRelu() {
  auto wrap = lookupNNCCache(aten::relu);
  if (wrap) {
    return wrap;
  }
  wrap = std::make_shared<TEWrapper>();
  auto N = VarHandle("N", kInt);
  Placeholder A("A", kFloat, {N});
  Tensor B = Compute("B", {N}, [&](const VarHandle& i) {
    auto zero = FloatImm::make(0.f);
    auto a = A.load(i);
    return ifThenElse(a < zero, zero, a);
  });
  wrap = wrapTECompute(wrap, A, B, N);
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
  Placeholder A("A", kFloat, {N});
  Tensor B = Compute("B", {N}, [&](const VarHandle& i) {
    auto a = A.load(i);
    return fast_tanh(a);
  });
  wrap = wrapTECompute(wrap, A, B, N);
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
  Placeholder A("A", kFloat, {N});
  Tensor B =
      Compute("B", {N}, [&](const VarHandle& i) { return sigmoid(A.load(i)); });
  // NNC uses sleef for vectorizing sigmoid, which comes in an 8-wide flavor
  // (Sleef_expf8).
  constexpr int kSleefWidth = 8;
  wrap = wrapTECompute(wrap, A, B, N, kSleefWidth);
  updateNNCCache(aten::sigmoid, wrap);
  return wrap;
}

} // namespace jit
} // namespace torch
