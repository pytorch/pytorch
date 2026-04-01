#pragma once

namespace Slang
{
struct IRModule;
class DiagnosticSink;

// - [ ] Lower 0 length vectors to unit
// - [x] Lower 1 length vectors to scalar
// - [ ] Lower too long vectors to tuples
void legalizeVectorTypes(IRModule* module, DiagnosticSink* sink);

} // namespace Slang
