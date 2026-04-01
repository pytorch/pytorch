// slang-ir-entry-point-raw-ptr-params.cpp
#include "slang-ir-entry-point-raw-ptr-params.h"

#include "slang-ir-insts.h"

namespace Slang
{

// This pass transforms the entry points in a module
// so that any entry-point parameters of pointer
// type (or a pointer-like type like `ConstantBuffer<T>`)
// are replaced with parameters of raw pointer (`void*`)
// type, with a cast in teh function body used to
// produce a value of the expected type.

struct ConvertEntryPointPtrParamsToRawPtrsPass
{
    IRModule* m_module;

    void processModule()
    {
        IRBuilder builder(m_module);

        // We start by getting and caching the raw pointer type.
        //
        auto rawPtrType = builder.getRawPointerType();

        // Now we loop over global-scope instructions searching
        // for any entry points.
        //
        for (auto inst : m_module->getGlobalInsts())
        {
            auto func = as<IRFunc>(inst);
            if (!func)
                continue;

            if (!func->findDecoration<IREntryPointDecoration>())
                continue;

            // We can only modify entry points with definitions here.
            //
            auto firstBlock = func->getFirstBlock();
            if (!firstBlock)
                continue;

            // Any code we introduce for casts will need to be inserted
            // before the first ordinary instruction in the first block
            // of the function (right after the parameters).
            //
            builder.setInsertBefore(firstBlock->getFirstOrdinaryInst());

            // Note: because we are inserting code right after the parameters
            // it doesn't work here to use `firstBlock->getParams()`, because
            // that captures a begin/end range where the "end" is the
            // first ordinary instruction at the time of the call, which will
            // chane when we insert code.
            //
            // TODO: We chould probably change the represnetation of ranges
            // of instructions to use first/last instead of begin/end so
            // that ranges are robust against changes to instructions outside
            // of a range.
            //
            for (auto param = firstBlock->getFirstParam(); param; param = param->getNextParam())
            {
                // We only want to transform parameters of pointer or
                // pointer-like type.
                //
                auto paramType = param->getDataType();
                if (!as<IRPtrTypeBase>(paramType) && !as<IRPointerLikeType>(paramType))
                    continue;

                // We will overwrite the type of the parameter to
                // be the raw pointer type instead.
                //
                builder.setDataType(param, rawPtrType);

                // We are going to replace uses of the parameter with
                // uses of a bit-cast operation based on the parameter,
                // but we need to be careful because that bit-cast operation
                // will itself be a use (which we don't want to replace
                // because that would create a circularity).
                //
                // Instead we capture the list of uses *before* we create
                // the bit cast instruction.
                //
                List<IRUse*> uses;
                for (auto use = param->firstUse; use; use = use->nextUse)
                    uses.add(use);

                // Now we emit a bit-cast operation into the first block
                // of the entry-point function to cast the raw-pointer
                // parameter to the type that the body code expects.
                //
                auto cast = builder.emitBitCast(paramType, param);

                // Now we can replace all the (captured) uses of the
                // parameter with the bit-cast operation instead.
                //
                for (auto use : uses)
                    use->set(cast);
            }

            // Because our operation might have changed the parameter
            // types of the function, we need to make sure to fix up
            // the IR type of the function to match its parameter list.
            //
            fixUpFuncType(func);
        }
    }
};

void convertEntryPointPtrParamsToRawPtrs(IRModule* module)
{
    ConvertEntryPointPtrParamsToRawPtrsPass pass;
    pass.m_module = module;
    pass.processModule();
}

} // namespace Slang
