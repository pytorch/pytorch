// slang-ir-glsl-legalize.h
#pragma once
#include "../core/slang-list.h"
#include "slang-compiler.h"

namespace Slang
{

class DiagnosticSink;
class Session;

class ShaderExtensionTracker;

struct IRFunc;
struct IRModule;

void legalizeEntryPointsForGLSL(
    Session* session,
    IRModule* module,
    const List<IRFunc*>& func,
    CodeGenContext* context,
    ShaderExtensionTracker* glslExtensionTracker);

void legalizeConstantBufferLoadForGLSL(IRModule* module);

void legalizeDispatchMeshPayloadForGLSL(IRModule* module);

void legalizeDynamicResourcesForGLSL(CodeGenContext* context, IRModule* module);
} // namespace Slang
