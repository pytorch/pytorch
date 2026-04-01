#include "slang-ir-legalize-mesh-outputs.h"

#include "slang-ir-clone.h"
#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{

void legalizeMeshOutputTypes(IRModule* module)
{
    IRBuilder builder(module);

    for (auto inst : module->getGlobalInsts())
    {
        if (auto meshOutput = as<IRMeshOutputType>(inst))
        {
            auto elemType = meshOutput->getElementType();
            auto maxCount = meshOutput->getMaxElementCount();
            auto arrayType = builder.getArrayType(elemType, maxCount);
            IROp decorationOp =
                as<IRVerticesType>(meshOutput)  ? kIROp_VerticesDecoration
                : as<IRIndicesType>(meshOutput) ? kIROp_IndicesDecoration
                : as<IRPrimitivesType>(meshOutput)
                    ? kIROp_PrimitivesDecoration
                    : (SLANG_UNREACHABLE("Missing case for IRMeshOutputType"), IROp(0));
            // Ensure that all params are marked up as vertices/indices/primitives
            traverseUsers<IRParam>(
                meshOutput,
                [&](IRParam* i) { builder.addMeshOutputDecoration(decorationOp, i, maxCount); });
            meshOutput->replaceUsesWith(arrayType);
        }
    }
}

} // namespace Slang
