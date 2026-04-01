// slang-ir-specialize-buffer-load-arg.h
#pragma once

namespace Slang
{
struct CodeGenContext;
struct IRModule;


/// Specialize functions in `module` that are called with direct loads from buffers.
///
/// For example:
///
///     struct Params { /* many fields */ }
///     int helper(Params p, int x) { return p.justOneField + x; }
///     ...
///     ConstantBuffer<Params> gParams;
///     ...
///     int z = helper(gParams, y);
///
/// In this case, the function `helper` declares a very large structure type as
/// a by-value argument. Depending on the final code-generation target, this could
/// result in output code that loads the entire contents of `gParams` before passing
/// it to `helper`, which then uses only a single field (rendering the rest of the load
/// operations wasted).
///
/// This pass is designed to specialize a callee function like `helper` based on call
/// sites in this form, so that the output code is:
///
///     struct Params { /* as before */ }
///     ConstantBuffer<Params> gParams;
///     int helper_1(int x) { return gParams.justOneField + x; }
///     ...
///     int z = helper_1(y);
///
/// Note how in the transformed code, there is no longer any attempt to load the rest
/// of the contents of `gParams`.
///
void specializeFuncsForBufferLoadArgs(CodeGenContext* codeGenContext, IRModule* module);
} // namespace Slang
