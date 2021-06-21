#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/operators/matmul.h>

namespace torch {
namespace jit {
namespace tensorexpr {

Tensor* computeMatmul(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType) {
  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);
  }
  BufHandle ResultBuf("matmul", outputShape, dtype);
  const BufHandle a = c10::get<BufHandle>(inputs[0]);
  const BufHandle b = c10::get<BufHandle>(inputs[1]);

  auto size_a = a.dims();
  auto size_b = b.dims();
  // We currently only support rank 2 matmuls
  TORCH_INTERNAL_ASSERT(size_a.size() == 2 && size_b.size() == 2);
  auto total_size = dynamic_cast<LongImm*>(
      IRSimplifier::simplify(
          cast<int64_t>(size_a[0]) * cast<int64_t>(size_a[1]) *
          cast<int64_t>(size_b[1]))
          .node());

  // For small sizes, where N*M*K < 1000, lower matmul to a naive 3-level
  // loopnest. The number is not tuned very carefully, and in future we should
  // fine-tune it as well as we should add more advanced native TE lowerings for
  // matmuls. For bigger sizes we generate a TE ExternalCall, which would call
  // an aten::matmul.
  // Native, even naive, lowering is beneficial when the sizes are small because
  // it allows to eliminate dispatch overhead.
  if (total_size && total_size->value() < 1000) {
    return Reduce(
        "nnc_matmul",
        {{size_a[0], "M"}, {size_b[1], "N"}},
        Sum(),
        [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
          return Load::make(a, {m, k}) * Load::make(b, {k, n});
        },
        {{size_a[1], "K"}});
  } else {
    return new Tensor(
        ResultBuf.node(),
        ExternalCall::make(ResultBuf, "nnc_aten_matmul", {a, b}, {}));
  }
}

Tensor* computeAddMM(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType) {
  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);
  }
  BufHandle ResultBuf("addmm", outputShape, dtype);
  return new Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf,
          "nnc_aten_addmm",
          {c10::get<BufHandle>(inputs[0]),
           c10::get<BufHandle>(inputs[1]),
           c10::get<BufHandle>(inputs[2])},
          {c10::get<int64_t>(inputs[3]),
           c10::get<int64_t>(
               inputs[4])})); // TODO: handle other dtypes of alpha and beta
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
