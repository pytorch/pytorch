// slang-ir-entry-point-decorations.h
#pragma once

#include "slang-ir.h"

namespace Slang
{
enum class CodeGenTarget;
class DiagnosticSink;

/// Checks entry point decoration values to ensure that they are valid for
/// the shader stage and target.
void checkEntryPointDecorations(IRModule* module, CodeGenTarget target, DiagnosticSink* sink);


// OutputTopologyType member definition macro
#define OUTPUT_TOPOLOGY_TYPES(M) \
    M(Point, point)              \
    M(Line, line)                \
    M(Triangle, triangle)        \
    M(TriangleCW, triangle_cw)   \
    M(TriangleCCW, triangle_ccw) \
    /* end */

enum class OutputTopologyType
{
    Unknown = 0,
#define CASE(ID, NAME) ID,
    OUTPUT_TOPOLOGY_TYPES(CASE)
#undef CASE
};


OutputTopologyType convertOutputTopologyStringToEnum(String rawOutputTopology);

} // namespace Slang
