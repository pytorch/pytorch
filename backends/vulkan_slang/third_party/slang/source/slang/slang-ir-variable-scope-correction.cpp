#include "slang-ir-variable-scope-correction.h"

#include "slang-ir-clone.h"
#include "slang-ir-dominators.h"
#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{

bool isCPUTarget(TargetRequest* targetReq);
bool isCUDATarget(TargetRequest* targetReq);

namespace
{ // anonymous
struct VariableScopeCorrectionContext
{
    VariableScopeCorrectionContext(IRModule* module, TargetRequest* targetReq)
        : m_module(module), m_builder(module), m_targetReq(targetReq)
    {
    }

    void processModule();

    /// Process a function in the module
    void _processFunction(IRFunc* funcInst);
    void _processInstruction(
        IRDominatorTree* dominatorTree,
        IRInst* instAfterParam,
        IRInst* originInst,
        const List<IRLoop*>& loopHeaderList,
        List<IRInst*>& workList);
    void _processStorableInst(IRInst* insertLoc, IRInst* inst, const List<IRUse*>& outOfScopeUses);
    void _processUnstorableInst(IRInst* inst, const List<IRUse*>& outOfScopeUser);

    bool _isStorableType(IRType* inst);
    bool _isOutOfScopeUse(
        IRInst* inst,
        IRDominatorTree* domTree,
        const List<IRLoop*>& loopHeaderList);

    IRModule* m_module;
    IRBuilder m_builder;
    TargetRequest* m_targetReq;
};

void VariableScopeCorrectionContext::processModule()
{
    IRModuleInst* moduleInst = m_module->getModuleInst();
    for (IRInst* child : moduleInst->getChildren())
    {
        // We want to find all of the functions, and process them
        if (auto funcInst = as<IRFunc>(child))
        {
            if (funcInst->getFirstBlock())
            {
                _processFunction(funcInst);
            }
        }
    }
}

void VariableScopeCorrectionContext::_processFunction(IRFunc* funcInst)
{
    IRDominatorTree* dominatorTree = m_module->findOrCreateDominatorTree(funcInst);
    List<IRInst*> workList;
    Dictionary<IRBlock*, List<IRLoop*>> loopHeaderMap;

    // traverse all blocks in the function
    for (auto block : funcInst->getBlocks())
    {
        // Traverse all the dominators of a given block to check whether this given block is in a
        // loop region. Loop region blocks are the blocks that are dominated by the loop header
        // block but not dominated by the loop break block.
        auto dominatorBlock = dominatorTree->getImmediateDominator(block);
        List<IRLoop*> loopHeaderList;
        for (; dominatorBlock;
             dominatorBlock = dominatorTree->getImmediateDominator(dominatorBlock))
        {
            // Find if the block is loop header block
            if (auto loopHeader = as<IRLoop>(dominatorBlock->getTerminator()))
            {
                // Get the break block of the loop and check if such block
                auto breakBlock = loopHeader->getBreakBlock();

                // Check if the current block is dominated by the break block. If so, it means that
                // the block is in the loop region.
                if (!dominatorTree->dominates(breakBlock, block))
                {
                    loopHeaderList.add(loopHeader);
                }
            }
        }
        loopHeaderMap.add(block, loopHeaderList);
    }

    if (loopHeaderMap.getCount() == 0)
    {
        return;
    }

    // Traverse all the instructions in function.
    for (auto block : funcInst->getBlocks())
    {
        if (loopHeaderMap.containsKey(block))
        {
            for (auto inst : block->getChildren())
            {
                List<IRInst*> instList;
                // Don't process the variable declaration instruction because the code is not
                // emitted for them unless there is a use.
                if (inst->getOp() == kIROp_Var)
                {
                    continue;
                }
                workList.add(inst);
            }
        }
    }

    auto instAfterParam = funcInst->getFirstBlock()->getFirstOrdinaryInst();

    for (Index i = 0; i < workList.getCount(); i++)
    {
        auto inst = workList[i];
        if (auto loopHeaderList = loopHeaderMap.tryGetValue(getBlock(inst)))
        {
            _processInstruction(dominatorTree, instAfterParam, inst, *loopHeaderList, workList);
        }
    }
}

// Check if the instruction is used outside of the loop.
// The loopHeaderList contains all the loop headers where the original instruction is defined.
// So we if the block of the user instruction is dominated by the break block of the loop header,
// it means that it was out of the loop, so it's out of the scope of the loop.
// Note the reason we use the loopHeaderList is because there could be nested loops, so we need to
// check all the loop headers from inner to outer.
bool VariableScopeCorrectionContext::_isOutOfScopeUse(
    IRInst* userInst,
    IRDominatorTree* domTree,
    const List<IRLoop*>& loopHeaderList)
{
    if (auto block = getBlock(userInst))
    {
        // If the use site of this instruction is dominated by the break block, it means that the
        // instruction is used after the break block, so we need to make that instruction available
        // globally. By doing so, we record all the users of this instructions.
        for (auto loopHeader : loopHeaderList)
        {
            auto breakBlock = loopHeader->getBreakBlock();
            if (domTree->dominates(breakBlock, block))
            {
                return true;
            }
        }
    }
    return false;
}

void VariableScopeCorrectionContext::_processInstruction(
    IRDominatorTree* dominatorTree,
    IRInst* instAfterParam,
    IRInst* originInst,
    const List<IRLoop*>& loopHeaderList,
    List<IRInst*>& workList)
{
    List<IRUse*> outOfScopeUses;
    for (auto use = originInst->firstUse; use; use = use->nextUse)
    {
        if (_isOutOfScopeUse(use->getUser(), dominatorTree, loopHeaderList))
        {
            outOfScopeUses.add(use);
        }
    }

    if (outOfScopeUses.getCount() == 0)
        return;

    if (_isStorableType(originInst->getDataType()))
    {
        _processStorableInst(instAfterParam, originInst, outOfScopeUses);
    }
    else
    {
        _processUnstorableInst(originInst, outOfScopeUses);
        // After processing the user, we need to add operands of the instruction to the worklist
        // for later processing.
        for (UInt idx = 0; idx < originInst->getOperandCount(); idx++)
        {
            workList.add(originInst->getOperand(idx));
        }
    }
}

void VariableScopeCorrectionContext::_processStorableInst(
    IRInst* insertLoc,
    IRInst* inst,
    const List<IRUse*>& outOfScopeUses)
{
    auto type = inst->getDataType();
    // store instruction must have a result type
    SLANG_ASSERT(type);

    // declare a new variable at the beginning of the function used to store the result of the
    // instruction
    m_builder.setInsertBefore(insertLoc);
    auto dstPtr = m_builder.emitVar(type);

    // insert a store instruction after the instruction
    m_builder.setInsertAfter(inst);
    m_builder.emitStore(dstPtr, inst);

    // last, replace operands in the use site instruction with the new variable
    // Note, because "dstPtr" is a pointer type, we have to insert a load(dstPtr) instruction before
    // use it. Simply replace any operand with pointer could generate error code.
    for (auto use : outOfScopeUses)
    {
        m_builder.setInsertBefore(use->getUser());
        auto loadInst = m_builder.emitLoad(type, dstPtr);
        m_builder.replaceOperand(use, loadInst);
    }
}

void VariableScopeCorrectionContext::_processUnstorableInst(
    IRInst* inst,
    const List<IRUse*>& outOfScopeUsers)
{
    IRCloneEnv cloneEnv;
    auto clonedInst = cloneInst(&cloneEnv, &m_builder, inst);

    for (auto user : outOfScopeUsers)
    {
        // duplicate the invisible instruction and insert it right before the use site,
        // then replace the operand with the duplicated instruction
        clonedInst->insertBefore(user->getUser());
        m_builder.replaceOperand(user, clonedInst);
    }
}

bool VariableScopeCorrectionContext::_isStorableType(IRType* type)
{
    if (!type)
        return false;

    // C/CPP/CUDA can store any type.
    if (isCPUTarget(m_targetReq) || isCUDATarget(m_targetReq))
        return true;

    if (as<IRBasicType>(type))
        return true;

    switch (type->getOp())
    {
    case kIROp_VectorType:
    case kIROp_MatrixType:
    case kIROp_StructType:
        return true;
    case kIROp_ArrayType:
        {
            if (auto arrayType = as<IRArrayTypeBase>(type))
                return _isStorableType(arrayType->getElementType());
            else
                return false;
        }
    case kIROp_UnsizedArrayType:
        return false;
    default:
        return false;
    }
}

} // namespace

void applyVariableScopeCorrection(IRModule* module, TargetRequest* targetReq)
{
    VariableScopeCorrectionContext context(module, targetReq);

    context.processModule();
}

} // namespace Slang
