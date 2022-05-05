
/**
 * @generated
 * This is an auto-generated file. Please do not modify it by hand.
 * To re-generate, please run:
 * cd ~/pytorch && python torchgen/decompositions/gen_jit_decompositions.py
 */
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/decomposition_registry_util.h>
#include <torch/csrc/jit/runtime/operator.h>

namespace torch {
namespace jit {

const std::string decomp_funcs =
    R"(def var_decomposition(input: Tensor,
    dim: Optional[List[int]]=None,
    correction: Optional[int]=None,
    keepdim: bool=False) -> Tensor:
  if torch.__is__(dim, None):
    dim0 = annotate(List[int], [])
  else:
    dim0 = unchecked_cast(List[int], dim)
  if torch.eq(torch.len(dim0), 0):
    n = torch.numel(input)
  else:
    n0 = 1
    for _0 in range(torch.len(dim0)):
      dim_i = dim0[_0]
      n1 = torch.mul(n0, (torch.size(input))[dim_i])
      n0 = n1
    n = n0
  mean = torch.mean(input, dim0, True)
  sub = torch.sub(input, mean)
  sq = torch.mul(sub, sub)
  sum = torch.sum(sq, dim0, keepdim)
  if torch.__isnot__(correction, None):
    correction0 = unchecked_cast(int, correction)
    n2 = torch.sub(n, correction0)
  else:
    n2 = n
  return torch.div(sum, n2)

def var(input: Tensor,
    unbiased: bool=True) -> Tensor:
  if unbiased:
    _0 = 1
  else:
    _0 = 0
  n = torch.numel(input)
  mean = torch.mean(input, annotate(List[int], []), True)
  sub = torch.sub(input, mean)
  sq = torch.mul(sub, sub)
  sum = torch.sum(sq, annotate(List[int], []))
  n0 = torch.sub(n, _0)
  return torch.div(sum, n0)

)";

const std::string& GetSerializedDecompositions() {
  return decomp_funcs;
}

const OperatorMap<std::string>& GetDecompositionMapping() {
  // clang-format off
 static const OperatorMap<std::string> decomposition_mapping {
    {"aten::var.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> (Tensor)", "var_decomposition"},
    {"aten::var(Tensor self, bool unbiased=True) -> (Tensor)", "var"},
  };
  // clang-format on

  return decomposition_mapping;
}

} // namespace jit
} // namespace torch
