// slang-ir-wrap-structured-buffers.h
#pragma once

namespace Slang
{
struct IRModule;

/// Wrap structured buffer types of matrices for
/// targets that compute incorrect layouts for such
/// types.
///
void wrapStructuredBuffersOfMatrices(IRModule* module);
} // namespace Slang
