// ir-entry-point-pass.h
#pragma once

#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{

struct PerEntryPointPass
{
public:
    // We will process a whole module by visiting all
    // its global functions, looking for entry points.
    //
    void processModule(IRModule* module);

    struct EntryPointInfo
    {
        IRFunc* func = nullptr;
        IREntryPointDecoration* decoration = nullptr;
    };

protected:
    void processEntryPoint(IRFunc* entryPointFunc, IREntryPointDecoration* entryPointDecoration);

    virtual void processEntryPointImpl(EntryPointInfo const& info) = 0;

    // We'll hang on to the module we are processing,
    // so that we can refer to it when setting up `IRBuilder`s.
    //
    IRModule* m_module = nullptr;

    EntryPointInfo m_entryPoint;
};

} // namespace Slang
