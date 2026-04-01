// slang-ir-variable-scope-correction.h
#ifndef SLANG_IR_VARIABLE_SCOPE_CORRECTION_H
#define SLANG_IR_VARIABLE_SCOPE_CORRECTION_H

namespace Slang
{

struct IRModule;
class TargetRequest;

/// This pass correct the scope of variables in loop regions
///
/// In the IR optimization pass, we turn all the loop to do-while loop form.
///    But in the do-while loop form, the loop body block is dominating the
///    blocks after the loop break block. E.g.
///
///    do {
///     A
///    } while (cond);
///    B
///
///    In the above example, the block A is dominating block B. This assumption
///    is fine for SPIRV and IR code, however, it's incorrect for all the other
///    language targets (e.g. c/c++/cuda/glsl/hlsl) because the instructions defined
///    in the block A are not visible from block B. Therefore, when translating to
///    other textual language, there could be issue for the variables scope.
///
///    To fix this issue, we first detect the instructions that are defined
///    inside the loop block (block A), then check if these instructions are used after
///    the break block (block B). If so, we duplicate these instructions right before
///    their users such that we can make those instructions available globally.
void applyVariableScopeCorrection(IRModule* module, TargetRequest* targetReq);

} // namespace Slang

#endif // SLANG_IR_VARIABLE_SCOPE_CORRECTION_H
