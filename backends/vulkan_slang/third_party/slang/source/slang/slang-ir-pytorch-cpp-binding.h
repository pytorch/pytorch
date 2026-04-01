#pragma once

namespace Slang
{
struct IRModule;
class DiagnosticSink;

void generatePyTorchCppBinding(IRModule* module, DiagnosticSink* sink);
void generateHostFunctionsForAutoBindCuda(IRModule* module, DiagnosticSink* sink);
void removeTorchKernels(IRModule* module);
void handleAutoBindNames(IRModule* module);
void generateDerivativeWrappers(IRModule* module, DiagnosticSink* sink);
void lowerBuiltinTypesForKernelEntryPoints(IRModule* module, DiagnosticSink* sink);
void removeTorchAndCUDAEntryPoints(IRModule* module);

} // namespace Slang
