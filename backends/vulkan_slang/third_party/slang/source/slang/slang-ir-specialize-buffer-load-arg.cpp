// slang-ir-specialize-buffer-load-arg.cpp
#include "slang-ir-specialize-buffer-load-arg.h"

#include "slang-ir-insts.h"
#include "slang-ir-specialize-function-call.h"
#include "slang-ir.h"

namespace Slang
{

// This file implements a pass that translates function call sites where
// the result of a buffer load from a global shader parameter (e.g., a
// global constant buffer) is being passed through to the callee. It
// replaces those with calls to specialized callee functions that directly
// reference the chosen global.
//
// As swith most of our IR passes, we encapsulate the logic here in a context
// type so that the data that needs to be shared throughout the pass can
// be conveniently scoped.

struct FuncBufferLoadSpecializationCondition : FunctionCallSpecializeCondition
{
    typedef FunctionCallSpecializeCondition Super;

    virtual bool doesParamWantSpecialization(IRParam* param, IRInst* arg)
    {
        // We only want to specialize for `struct` types and not base types.
        //
        // TODO: We might want to consider some criteria here for the "large-ness"
        // of a structure (in terms of bytes and/or fields), so that we don't
        // eliminate loads of sufficiently small types (which are cheap to pass
        // by value).
        //
        auto paramType = param->getDataType();
        if (!as<IRStructType>(paramType))
            return false;

        // We also only want to specialize for arguments that are a load
        // from some kind of global shader parameter.
        //
        IRInst* a = arg;
        if (auto argLoad = as<IRLoad>(arg))
        {
            a = argLoad->getPtr();
        }
        else
        {
            return false;
        }

        // We want to handle loads from a shader parameter that is an array
        // of buffers, and not just a single global buffer.
        //
        while (auto argGetElement = as<IRGetElement>(a))
        {
            a = argGetElement->getBase();
        }

        // The "root" of the parameter must be a reference to a global-scope
        // shader parameter, so that we know we can substitute it into the callee.
        //
        if (const auto argGlobalParam = as<IRGlobalParam>(a))
        {
            return true;
        }
        else
        {
            return false;
        }

        // TODO: There are other patterns that we could attempt to optimize here.
        // For example, this logic only handles loads of the *entire* contents of
        // a buffer, so it would miss:
        //
        // * A load of a large structure from field in a constant buffer, so that
        //   the value loaded is not the entire buffer contents.
        //
        // * A load of a large structure from a structured buffer, or any other kind
        //   of buffer that requires an index.
        //
        // * Any resource load that is not expressed at the IR level with a `load`
        //   instruction (e.g., those that might use an intrinsic function).
        //
    }
};

void specializeFuncsForBufferLoadArgs(CodeGenContext* codegenContext, IRModule* module)
{
    FuncBufferLoadSpecializationCondition condition;
    specializeFunctionCalls(codegenContext, module, &condition);
}

} // namespace Slang
