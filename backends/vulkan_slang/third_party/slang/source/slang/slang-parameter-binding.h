#ifndef SLANG_PARAMETER_BINDING_H
#define SLANG_PARAMETER_BINDING_H

#include "../core/slang-basic.h"
#include "slang-syntax.h"
#include "slang.h"

namespace Slang
{

class Program;
class TargetRequest;

// The parameter-binding interface is responsible for assigning
// binding locations/registers to every parameter of a shader
// program. This can include both parameters declared on a
// particular entry point, as well as parameters declared at
// global scope.
//


// Generate binding information for the given program,
// represented as a collection of different translation units,
// and attach that information to the syntax nodes
// of the program.

void generateParameterBindings(Program* program, TargetRequest* targetReq, DiagnosticSink* sink);

/// Given a string that specifies a name and index (e.g., `COLOR0`),
/// split it into slices for the name part and the index part.
///
bool splitNameAndIndex(
    UnownedStringSlice const& text,
    UnownedStringSlice& outName,
    UnownedStringSlice& outDigits);

} // namespace Slang

#endif
