#include "slang-ir-user-type-hint.h"

#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{

void addUserTypeHintDecorations(IRModule* module)
{
    for (auto globalInst : module->getGlobalInsts())
    {
        auto inst = as<IRGlobalParam>(globalInst);
        if (!inst)
            continue;
        if (inst->getDataType())
        {
            // Preserve the original type name as a decoration before we do any type lowering.
            // This is needed to implement -fspv-reflect, which allows the compiler to output the
            // original user-friendly type name of each shader parameter as a SPIRV decoration.
            //
            StringBuilder sb;
            getTypeNameHint(sb, inst->getDataType());
            if (sb.getLength())
            {
                IRBuilder builder(inst);
                builder.addDecoration(
                    inst,
                    kIROp_UserTypeNameDecoration,
                    builder.getStringValue(sb.produceString().getUnownedSlice()));
            }
        }
    }
}

} // namespace Slang
