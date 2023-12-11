
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

namespace torch::jit {

const std::string decomp_funcs =
    R"(def var_decomposition(input: Tensor,
    dim: Optional[List[int]]=None,
    correction: Union[float, int, NoneType, bool]=None,
    keepdim: bool=False) -> Tensor:
  _0 = uninitialized(float)
  if torch.__is__(dim, None):
    dim0 = annotate(List[int], [])
  else:
    dim0 = unchecked_cast(List[int], dim)
  if torch.eq(torch.len(dim0), 0):
    n = torch.numel(input)
  else:
    n0 = 1
    for _1 in range(torch.len(dim0)):
      dim_i = dim0[_1]
      n1 = torch.mul(n0, (torch.size(input))[dim_i])
      n0 = n1
    n = n0
  mean = torch.mean(input, dim0, True)
  sub = torch.sub(input, mean)
  sq = torch.mul(sub, sub)
  sum = torch.sum(sq, dim0, keepdim)
  if torch.__is__(correction, None):
    denom = float(torch.sub(n, 1))
  else:
    correction0 = unchecked_cast(Union[float, int, bool], correction)
    _2 = isinstance(correction0, int)
    if _2:
      correction1 = unchecked_cast(int, correction0)
      denom0 = float(torch.sub(n, correction1))
    else:
      correction2 = unchecked_cast(Union[float, bool], correction0)
      _3 = isinstance(correction2, float)
      if _3:
        correction3 = unchecked_cast(float, correction2)
        denom2 = torch.sub(float(n), correction3)
        denom1 = denom2
      else:
        ops.prim.RaiseException("correction must be int or float", "builtins.RuntimeError")
        denom1 = _0
      denom0 = denom1
    denom = denom0
  _4 = torch.div(sum, ops.prim.max(0, denom))
  return _4

def var(input: Tensor,
    unbiased: bool=True) -> Tensor:
  if unbiased:
    _0 = 1
  else:
    _0 = 0
  _1 = uninitialized(float)
  n = torch.numel(input)
  mean = torch.mean(input, annotate(List[int], []), True)
  sub = torch.sub(input, mean)
  sq = torch.mul(sub, sub)
  sum = torch.sum(sq, annotate(List[int], []))
  _2 = isinstance(_0, int)
  if _2:
    denom = float(torch.sub(n, _0))
  else:
    correction = unchecked_cast(Union[float, bool], _0)
    _3 = isinstance(correction, float)
    if _3:
      correction0 = unchecked_cast(float, correction)
      denom0 = torch.sub(float(n), correction0)
    else:
      ops.prim.RaiseException("correction must be int or float", "builtins.RuntimeError")
      denom0 = _1
    denom = denom0
  _4 = torch.div(sum, ops.prim.max(0, denom))
  return _4

)";

const std::string& GetSerializedDecompositions() {
  return decomp_funcs;
}

const OperatorMap<std::string>& GetDecompositionMapping() {
  // clang-format off
 static const OperatorMap<std::string> decomposition_mapping {
    {"aten::var.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor", "var_decomposition"},
    {"aten::var(Tensor self, bool unbiased=True) -> Tensor", "var"},
  };
  // clang-format on

  return decomposition_mapping;
}

} // namespace torch::jit
