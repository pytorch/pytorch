// slang-ir-liveness.h
#ifndef SLANG_IR_GLSL_LIVENESS_H
#define SLANG_IR_GLSL_LIVENESS_H

namespace Slang
{

struct IRModule;

/// Converts liveness marker instructions in a module into SPIR-V lifetime ops.
///
/// It does this by using the GL_EXT_spirv_intrinsics extension.
///
/// The transformation takes place at the IR level, inserting new functions as needed for
/// the types referenced via liveness markers.
void applyGLSLLiveness(IRModule* module);

} // namespace Slang

#endif // SLANG_IR_LIVENESS_H