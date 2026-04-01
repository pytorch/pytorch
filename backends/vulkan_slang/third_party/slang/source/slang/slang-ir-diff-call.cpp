// slang-ir-diff-call.cpp
#include "slang-ir-diff-call.h"

#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{

struct DerivativeCallProcessContext
{
    // This type passes over the module and replaces
    // derivative calls with the processed derivative
    // function.
    //
    IRModule* module;

    bool processModule()
    {
        // Run through all the global-level instructions,
        // looking for callable blocks.
        for (auto inst : module->getGlobalInsts())
        {
            // If the instr is a callable, get all the basic blocks
            if (auto callable = as<IRGlobalValueWithCode>(inst))
            {
                // Iterate over each block in the callable
                for (auto block : callable->getBlocks())
                {
                    // Iterate over each child instruction.
                    auto child = block->getFirstInst();
                    if (!child)
                        continue;

                    do
                    {
                        auto nextChild = child->getNextInst();
                        // Look for IRForwardDifferentiate
                        if (auto derivOf = as<IRForwardDifferentiate>(child))
                        {
                            processDifferentiate(derivOf);
                        }
                        child = nextChild;
                    } while (child);
                }
            }
        }
        return true;
    }

    // Perform forward-mode automatic differentiation on
    // the intstructions.
    void processDifferentiate(IRForwardDifferentiate* derivOfInst)
    {
        IRInst* jvpCallable = nullptr;

        // First get base function
        auto origCallable = derivOfInst->getBaseFn();

        // Resolve the derivative function for IRForwardDifferentiate(IRSpecialize(IRFunc))
        // Check the specialize inst for a reference to the derivative fn.
        //
        if (auto origSpecialize = as<IRSpecialize>(origCallable))
        {
            if (auto jvpSpecRefDecorator =
                    origSpecialize->findDecoration<IRForwardDerivativeDecoration>())
            {
                jvpCallable = jvpSpecRefDecorator->getForwardDerivativeFunc();
            }
        }

        // Resolve the derivative function for an IRForwardDifferentiate(IRFunc)
        //
        // Check for the 'JVPDerivativeReference' decorator on the
        // base function.
        //
        if (auto jvpRefDecorator = origCallable->findDecoration<IRForwardDerivativeDecoration>())
        {
            jvpCallable = jvpRefDecorator->getForwardDerivativeFunc();
        }

        SLANG_ASSERT(jvpCallable);

        // Substitute all uses of the 'derivativeOf' operation
        // with the resolved derivative function.
        derivOfInst->replaceUsesWith(jvpCallable);

        // Remove the 'derivativeOf' inst.
        derivOfInst->removeAndDeallocate();
    }
};

// Set up context and call main process method.
//
bool processDerivativeCalls(IRModule* module, IRDerivativeCallProcessOptions const&)
{
    DerivativeCallProcessContext context;
    context.module = module;

    return context.processModule();
}

} // namespace Slang
