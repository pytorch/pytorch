// slang-ir-bind-existentials.h
#pragma once

namespace Slang
{

class DiagnosticSink;
struct IRModule;

/// Bind concrete types to paameters that use existential slots.
void bindExistentialSlots(IRModule* module, DiagnosticSink* sink);

} // namespace Slang
