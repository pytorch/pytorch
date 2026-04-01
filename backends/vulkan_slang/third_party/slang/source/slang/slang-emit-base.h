// slang-emit-base.h
#ifndef SLANG_EMIT_BASE_H
#define SLANG_EMIT_BASE_H

#include "../core/slang-basic.h"
#include "slang-ir-insts.h"
#include "slang-ir-restructure.h"
#include "slang-ir.h"

namespace Slang
{

class SourceEmitterBase : public RefObject
{
public:
    IRInst* getSpecializedValue(IRSpecialize* specInst);

    /// Inspect the capabilities required by `inst` (according to its decorations),
    /// and ensure that those capabilities have been detected and stored in the
    /// target-specific extension tracker.
    void handleRequiredCapabilities(IRInst* inst);
    virtual void handleRequiredCapabilitiesImpl(IRInst* inst) { SLANG_UNUSED(inst); }

    static IRVarLayout* getVarLayout(IRInst* var);

    static BaseType extractBaseType(IRType* inType);
};

} // namespace Slang
#endif
