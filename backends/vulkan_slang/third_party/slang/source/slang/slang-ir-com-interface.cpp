// slang-ir-com-interface.cpp
#include "slang-ir-com-interface.h"

#include "slang-ir-insts.h"
#include "slang-ir-lower-com-methods.h"
#include "slang-ir.h"

namespace Slang
{

static bool _canReplace(IRUse* use)
{
    switch (use->getUser()->getOp())
    {
    case kIROp_WitnessTableIDType:
    case kIROp_WitnessTableType:
    case kIROp_RTTIPointerType:
    case kIROp_RTTIHandleType:
    case kIROp_ComPtrType:
    case kIROp_NativePtrType:
        {
            // Don't replace
            return false;
        }
    case kIROp_ThisType:
        {
            // Appears replacable.
            break;
        }
    case kIROp_PtrType:
        {
            // We can have ** and ComPtr<T>*.
            // If it's a pointer type it could be because it is a global.
            break;
        }
    default:
        break;
    }
    return true;
}

void lowerComInterfaces(IRModule* module, ArtifactStyle artifactStyle, DiagnosticSink* sink)
{
    // First, lower all COM methods and their call sites out of `Result` and other managed types.
    lowerComMethods(module, sink);

    // Find all of the COM interfaces
    List<IRInterfaceType*> comInterfaces;
    for (auto child : module->getGlobalInsts())
    {
        auto intf = as<IRInterfaceType>(child);
        if (intf && intf->findDecoration<IRComInterfaceDecoration>())
        {
            comInterfaces.add(intf);
        }
    }

    // For all interfaces found replace uses
    {
        IRBuilder builder(module);
        builder.setInsertInto(module->getModuleInst());

        List<IRUse*> uses;

        for (auto comIntf : comInterfaces)
        {
            uses.clear();

            // Find all of the uses *before* doing any replacement
            // Otherwise we end up replacing the replacement leading
            // to it pointing to itself.
            for (auto use = comIntf->firstUse; use; use = use->nextUse)
            {
                // Only store off uses where replacement can be made
                if (_canReplace(use))
                {
                    uses.add(use);
                }
            }

            // If there are no uses that can be replaced, then we don't need
            // to create a replacement result
            if (uses.getCount() <= 0)
            {
                continue;
            }

            // NOTE! The following code relies on the fact that the builder
            // *doesn't* dedup in general, and in particular doesn't ptr types.
            // This allows the creation a 'new'  pointer type, and subsequent replacment all old
            // uses, leading to a `IInterface*` becoming `IInterface**`.
            //

            // TODO(JS): This is a temporary fix, in that whether kernel or not
            // shouldn't control the ptr type in general
            // It's necessary here though because Kernel doesn't have ComPtr<>
            // so has to be a raw pointer
            IRType* result = (artifactStyle == ArtifactStyle::Host)
                                 ? static_cast<IRType*>(builder.getComPtrType(comIntf))
                                 : static_cast<IRType*>(builder.getNativePtrType(comIntf));

            // Go through replacing all of the replacable uses
            for (auto use : uses)
            {
                // Do the replacement
                builder.replaceOperand(use, result);
            }
        }
    }
}

} // namespace Slang
