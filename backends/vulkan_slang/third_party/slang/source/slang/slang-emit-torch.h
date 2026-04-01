// slang-emit-torch.h
#ifndef SLANG_EMIT_TORCH_H
#define SLANG_EMIT_TORCH_H

#include "slang-emit-cpp.h"

namespace Slang
{

class TorchCppSourceEmitter : public CPPSourceEmitter
{
public:
    typedef CPPSourceEmitter Super;

    TorchCppSourceEmitter(const Desc& desc)
        : Super(desc)
    {
    }

protected:
    // CPPSourceEmitter overrides
    virtual bool tryEmitInstStmtImpl(IRInst* inst) override;

    virtual bool tryEmitInstExprImpl(IRInst* inst, const EmitOpInfo& inOuterPrec) override;
    virtual SlangResult calcTypeName(IRType* type, CodeGenTarget target, StringBuilder& out)
        override;
    virtual void emitModuleImpl(IRModule* module, DiagnosticSink* sink) override;
};

} // namespace Slang
#endif
