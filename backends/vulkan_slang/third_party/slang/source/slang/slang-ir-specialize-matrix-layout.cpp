#include "slang-ir-specialize-matrix-layout.h"

#include "slang-compiler.h"
#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{

void visitParent(List<IRMatrixType*>& typeWorkList, IRInst* parent)
{
    for (auto child : parent->getChildren())
    {
        if (auto matrixType = as<IRMatrixType>(child))
        {
            if (auto constLayout = as<IRIntLit>(matrixType->getLayout()))
            {
                if (constLayout->getValue() == SLANG_MATRIX_LAYOUT_MODE_UNKNOWN)
                {
                    typeWorkList.add(matrixType);
                }
            }
        }
        visitParent(typeWorkList, child);
    }
}

void specializeMatrixLayout(TargetProgram* target, IRModule* module)
{
    List<IRMatrixType*> typeWorkList;
    visitParent(typeWorkList, module->getModuleInst());

    IRIntegerValue defaultLayout = target->getOptionSet().getMatrixLayoutMode();
    if (defaultLayout == SLANG_MATRIX_LAYOUT_MODE_UNKNOWN)
        defaultLayout = SLANG_MATRIX_LAYOUT_ROW_MAJOR;

    IRBuilder builder(module);
    for (auto matrixType : typeWorkList)
    {
        builder.setInsertBefore(matrixType);
        auto replacementMatrixType = builder.getMatrixType(
            matrixType->getElementType(),
            matrixType->getRowCount(),
            matrixType->getColumnCount(),
            builder.getIntValue(builder.getIntType(), defaultLayout));
        matrixType->replaceUsesWith(replacementMatrixType);
    }
}

} // namespace Slang
