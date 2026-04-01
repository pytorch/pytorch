// slang-ir-sccp.h
#pragma once

namespace Slang
{
struct IRModule;
struct IRInst;
class DiagnosticSink;

/// Apply Sparse Conditional Constant Propagation (SCCP) to a module.
///
/// This optimization replaces instructions that can only ever evaluate
/// to a single (well-defined) value with that constant value, and
/// also eliminates conditional branches where the condition will
/// always evaluate to a constant (which can lead to entire blocks
/// becoming dead code)
/// Returns true if IR is changed.
bool applySparseConditionalConstantPropagation(IRModule* module, DiagnosticSink* sink);
bool applySparseConditionalConstantPropagationForGlobalScope(
    IRModule* module,
    DiagnosticSink* sink);

bool applySparseConditionalConstantPropagation(IRInst* func, DiagnosticSink* sink);

IRInst* tryConstantFoldInst(IRModule* module, IRInst* inst);
} // namespace Slang
