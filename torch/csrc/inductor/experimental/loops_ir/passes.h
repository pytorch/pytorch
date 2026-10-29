#pragma once
// Logical micro-passes for LoopsIR. Strict numerics: preserve all FP op
// order/rounding. No FP constant folding, reassociation, neutral identities,
// FMA/FTZ. Integer-only simplifications allowed with checked arithmetic.

#include <map>
#include <string>
#include <vector>

#include "ir.h"

namespace torch::inductor::loops_ir {

// Inline view reads into explicit source-buffer flat offsets, composing
// alias offsets. Rewrites ViewRead -> Load, retargets compute inputs.
Program view_inline(const Program& program);

// Simplify integer index expressions with checked arithmetic, constant
// folding, and valid integer identities (add 0, mul 0/1, div 1, mod 1).
Program index_simplify(const Program& program);

// Eliminate unit pointwise dimensions (extent == 1) with explicit index
// substitution; reduction dimensions unchanged.
Program unit_range_removal(const Program& program);

// Local typed SSA CSE + DCE per body. CSE uses structural Index equality
// (indexEquals), ordered operand matching, bit-exact constant matching.
// No FP reassociation or folding. DCE removes defs not transitively used
// by stores/accumulates.
Program local_cse_dce(const Program& program);

// Prune dead views and computes using output, alias, and extern liveness.
// Extern calls are always preserved (conservative effects).
Program dead_prune(const Program& program);

// Full pipeline with verification before/after each pass.
Program optimize_program(const Program& program);

// Registry + deterministic per-pass dumps.
std::vector<std::string> pass_names();
Program run_pass(const std::string& name, const Program& program);
std::map<std::string, std::string> optimize_program_dumps(const Program& program);

// Narrow dense/injective store proof for fusion. Returns true only when the
// result store is proven to cover exactly [0, elements) injectively.
// Conservative: false means "no proof", never "proven false".
bool prove_dense_injective_store(const Compute& compute, const std::string& result);

} // namespace torch::inductor::loops_ir
