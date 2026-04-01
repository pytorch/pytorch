// slang-ir-entry-point-pass.cpp
#include "slang-ir-entry-point-pass.h"

namespace Slang
{

void PerEntryPointPass::processModule(IRModule* module)
{
    m_module = module;

    // Note that we are only looking at true global-scope
    // functions and not functions nested inside of
    // IR generics. When using generic entry points, this
    // pass should be run after the entry point(s) have
    // been specialized to their generic type parameters.

    for (auto inst : module->getGlobalInsts())
    {
        // We are only interested in entry points.
        //
        // Every entry point must be a function.
        //
        auto func = as<IRFunc>(inst);
        if (!func)
            continue;

        // Entry points will always have the `[entryPoint]`
        // decoration to differentiate them from ordinary
        // functions.
        //
        auto entryPointDecoration = func->findDecoration<IREntryPointDecoration>();
        if (!entryPointDecoration)
            continue;

        // If we find a candidate entry point, then we
        // will process it.
        //
        processEntryPoint(func, entryPointDecoration);
    }
}

void PerEntryPointPass::processEntryPoint(
    IRFunc* entryPointFunc,
    IREntryPointDecoration* entryPointDecoration)
{
    m_entryPoint.func = entryPointFunc;
    m_entryPoint.decoration = entryPointDecoration;
    processEntryPointImpl(m_entryPoint);
}

} // namespace Slang
