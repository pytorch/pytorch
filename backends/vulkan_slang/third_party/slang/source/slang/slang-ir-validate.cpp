// slang-ir-validate.cpp
#include "slang-ir-validate.h"

#include "slang-ir-dominators.h"
#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{
struct IRValidateContext
{
    // The IR module we are validating.
    IRModule* module;

    RefPtr<IRDominatorTree> domTree;

    // A diagnostic sink to send errors to if anything is invalid.
    DiagnosticSink* sink;

    DiagnosticSink* getSink() { return sink; }

    // A set of instructions we've seen, to help confirm that
    // values are defined before they are used in a given block.
    HashSet<IRInst*> seenInsts;
};

void validateIRInst(IRValidateContext* context, IRInst* inst);

void validate(IRValidateContext* context, bool condition, IRInst* inst, char const* message)
{
    if (!condition)
    {
        if (context)
        {
            context->getSink()->diagnose(inst, Diagnostics::irValidationFailed, message);
        }
        else
        {
            SLANG_ASSERT_FAILURE("IR validation failed");
        }
    }
}

void validateIRInstChildren(IRValidateContext* context, IRInst* parent)
{
    // We want to check that child instructions are correctly
    // ordered so that decorations come first, then any parameters,
    // and then any ordinary instructions.
    //
    // We will track what we have seen so far with a simple state
    // machine, which in valid IR should proceed monitonically
    // up through the following states:
    //
    enum State
    {
        kState_Initial = 0,
        kState_AfterDecoration,
        kState_AfterParam,
        kState_AfterOrdinary,
    };
    State state = kState_Initial;

    IRInst* prevChild = nullptr;
    bool hasSeenTerminatorInst = false;
    for (auto child : parent->getDecorationsAndChildren())
    {
        // We need to check the integrity of the parent/next/prev links of
        // all of our instructions
        validate(context, child->parent == parent, child, "parent link");
        validate(context, child->prev == prevChild, child, "next/prev link");

        // Recursively validate the instruction itself.
        validateIRInst(context, child);

        if (as<IRDecoration>(child))
        {
            validate(
                context,
                state <= kState_AfterDecoration,
                child,
                "decorations must come before other child instructions");
            state = kState_AfterDecoration;
        }
        else if (as<IRParam, IRDynamicCastBehavior::NoUnwrap>(child))
        {
            validate(
                context,
                state <= kState_AfterParam,
                child,
                "parameters must come before ordinary instructions");
            state = kState_AfterParam;
        }
        else
        {
            state = kState_AfterOrdinary;
        }

        // Do some extra validation around terminator instructions:
        //
        // * The last instruction of a block should always be a terminator
        // * No other instruction should be a terminator
        //
        if (as<IRBlock>(parent) && (child == parent->getLastDecorationOrChild()))
        {
            validate(
                context,
                as<IRTerminatorInst>(child) != nullptr,
                child,
                "last instruction in block must be terminator");
        }
        else
        {
            validate(
                context,
                !as<IRTerminatorInst>(child),
                child,
                "terminator must be last instruction in a block");
        }

        if (as<IRTerminatorInst>(child))
        {
            validate(
                context,
                !hasSeenTerminatorInst,
                child,
                "block must not contain more than one terminator");
            hasSeenTerminatorInst = true;
        }
        prevChild = child;
    }
}

void validateIRInstOperand(IRValidateContext* context, IRInst* inst, IRUse* operandUse)
{
    // The `IRUse` for the operand had better have `inst` as its user.
    validate(context, operandUse->getUser() == inst, inst, "operand user");

    // The value we are using needs to fit into one of a few cases.
    //
    // * If the parent of `inst` and of `operand` is the same block, then
    //   we require that `operand` is defined before `inst`
    //
    // * If the parents of `inst` and `operand` are both blocks in the
    //   same functin, then the block defining `operand` must dominate
    //   the block defining `inst`.
    //
    // * Otherwise, we simply require that the parent of `operand` be
    //   an ancestor (transitive parent) of `inst`.

    auto instParent = inst->getParent();

    auto operandValue = operandUse->get();

    if (!operandValue)
    {
        // A null operand should almost always be an error, but
        // we currently have a few cases where this arises.
        //
        // TODO: plug the leaks.
        return;
    }

    auto operandParent = operandValue->getParent();

    auto instParentBlock = getBlock(inst);
    if (instParentBlock)
    {
        if (auto operandParentBlock = as<IRBlock>(operandParent))
        {
            if (instParentBlock == operandParentBlock)
            {
                // If `operandValue` precedes `inst`, then we should
                // have already seen it, because we scan parent instructions
                // in order.
                if (context)
                {
                    validate(
                        context,
                        context->seenInsts.contains(operandValue),
                        inst,
                        "def must come before use in same block");
                }
                return;
            }

            auto instFunc = instParentBlock->getParent();
            auto operandFunc = operandParentBlock->getParent();
            if (instFunc == operandFunc)
            {
                // The two instructions are defined in different blocks of
                // the same function (or another value with code). We need
                // to validate that `operandParentBlock` dominates `instParentBlock`.
                //
                if (context && context->domTree)
                {
                    validate(
                        context,
                        context->domTree->dominates(operandParentBlock, instParentBlock),
                        inst,
                        "def must dominate use");
                }
                return;
            }
        }
    }

    // If the special cases above did not trigger, then either the two values
    // are nested in the same parent, but that parent isn't a block, or they
    // are nested in distinct parents, and those parents aren't both children
    // of a function.
    //
    // In either case, we need to enforce that the parent of `operand` needs
    // to be an ancestor of `inst`.
    //
    for (auto pp = instParent; pp; pp = pp->getParent())
    {
        if (pp == operandParent)
            return;
    }

    // We allow out-of-order def-use in global scope.
    bool allInGlobalScope = inst->getParent() && inst->getParent()->getOp() == kIROp_Module;
    if (allInGlobalScope)
    {
        for (UInt i = 0; i < inst->getOperandCount(); i++)
        {
            auto op = inst->getOperand(i);
            if (!op)
                continue;
            if (!op->getParent())
                continue;
            if (op->getParent()->getOp() != kIROp_Module)
            {
                allInGlobalScope = false;
                break;
            }
        }
    }
    if (allInGlobalScope)
        return;

    // Allow exceptions.
    switch (inst->getOp())
    {
    case kIROp_DifferentiableTypeDictionaryItem:
        return;
    }
    //
    // We failed to find `operandParent` while walking the ancestors of `inst`,
    // so something had gone wrong.
    validate(context, false, inst, "def must be ancestor of use");
}

void validateIRInstOperands(IRValidateContext* context, IRInst* inst)
{
    if (inst->getFullType())
        validateIRInstOperand(context, inst, &inst->typeUse);

    // Avoid validating decoration operands
    // since they don't have to conform to inst visibility
    // constraints.
    //
    if (as<IRDecoration>(inst))
        return;

    UInt operandCount = inst->getOperandCount();
    for (UInt ii = 0; ii < operandCount; ++ii)
    {
        validateIRInstOperand(context, inst, inst->getOperands() + ii);
    }
}

static thread_local bool _enableIRValidationAtInsert = false;
void disableIRValidationAtInsert()
{
    _enableIRValidationAtInsert = false;
}
void enableIRValidationAtInsert()
{
    _enableIRValidationAtInsert = true;
}
void validateIRInstOperands(IRInst* inst)
{
    if (!_enableIRValidationAtInsert)
        return;
    switch (inst->getOp())
    {
    case kIROp_loop:
    case kIROp_ifElse:
    case kIROp_unconditionalBranch:
    case kIROp_conditionalBranch:
    case kIROp_Switch:
        return;
    default:
        break;
    }

    validateIRInstOperands(nullptr, inst);
}

void validateCodeBody(IRValidateContext* context, IRGlobalValueWithCode* code)
{
    HashSet<IRBlock*> blocks;
    for (auto block : code->getBlocks())
        blocks.add(block);
    auto validateBranchTarget = [&](IRInst* inst, IRBlock* target)
    {
        validate(
            context,
            blocks.contains(target),
            inst,
            "branch inst must have a valid target block that is defined within the same "
            "scope.");
    };
    for (auto block : code->getBlocks())
    {
        auto terminator = block->getTerminator();
        validate(context, terminator, block, "block must have valid terminator inst.");
        switch (terminator->getOp())
        {
        case kIROp_conditionalBranch:
            validateBranchTarget(terminator, as<IRConditionalBranch>(terminator)->getTrueBlock());
            validateBranchTarget(terminator, as<IRConditionalBranch>(terminator)->getFalseBlock());
            break;
        case kIROp_loop:
        case kIROp_unconditionalBranch:
            validateBranchTarget(
                terminator,
                as<IRUnconditionalBranch>(terminator)->getTargetBlock());
            break;
        case kIROp_Switch:
            {
                auto switchInst = as<IRSwitch>(terminator);
                for (UInt i = 0; i < switchInst->getCaseCount(); i++)
                {
                    validateBranchTarget(switchInst, switchInst->getCaseLabel(i));
                }
                validateBranchTarget(switchInst, switchInst->getDefaultLabel());
                validateBranchTarget(switchInst, switchInst->getBreakLabel());
            }
        }
    }
}

void validateIRInst(IRValidateContext* context, IRInst* inst)
{
    // Validate that any operands of the instruction are used appropriately
    validateIRInstOperands(context, inst);
    context->seenInsts.add(inst);

    if (auto code = as<IRGlobalValueWithCode>(inst))
    {
        context->domTree = computeDominatorTree(code);
        validateCodeBody(context, code);
    }

    // If `inst` is itself a parent instruction, then we need to recursively
    // validate its children.
    validateIRInstChildren(context, inst);

    if (as<IRGlobalValueWithCode>(inst))
        context->domTree = nullptr;
}

void validateIRInst(IRInst* inst)
{
    IRValidateContext contextStorage;
    IRValidateContext* context = &contextStorage;
    DiagnosticSink sink;
    context->module = inst->getModule();
    context->sink = &sink;
    if (auto func = as<IRFunc>(inst))
        context->domTree = computeDominatorTree(func);
    validateIRInst(context, inst);
}

void validateIRModule(IRModule* module, DiagnosticSink* sink)
{
    IRValidateContext contextStorage;
    IRValidateContext* context = &contextStorage;
    context->module = module;
    context->sink = sink;

    auto moduleInst = module->getModuleInst();

    validate(context, moduleInst != nullptr, moduleInst, "module instruction");
    validate(context, moduleInst->parent == nullptr, moduleInst, "module instruction parent");
    validate(context, moduleInst->prev == nullptr, moduleInst, "module instruction prev");
    validate(context, moduleInst->next == nullptr, moduleInst, "module instruction next");

    validateIRInst(context, moduleInst);
}

void validateIRModuleIfEnabled(CompileRequestBase* compileRequest, IRModule* module)
{
    if (!compileRequest->getLinkage()->m_optionSet.getBoolOption(CompilerOptionName::ValidateIr))
        return;

    auto sink = compileRequest->getSink();
    validateIRModule(module, sink);
}

void validateIRModuleIfEnabled(CodeGenContext* codeGenContext, IRModule* module)
{
    if (!codeGenContext->shouldValidateIR())
        return;

    auto sink = codeGenContext->getSink();
    validateIRModule(module, sink);
}

// Returns whether 'dst' is a valid destination for atomic operations, meaning
// it leads either to 'groupshared' or 'device buffer' memory.
static bool isValidAtomicDest(bool skipFuncParamValidation, IRInst* dst)
{
    bool isGroupShared = as<IRGroupSharedRate>(dst->getRate());
    if (isGroupShared)
        return true;

    if (as<IRRWStructuredBufferGetElementPtr>(dst))
        return true;
    if (as<IRImageSubscript>(dst))
        return true;

    if (auto ptrType = as<IRPtrType>(dst->getDataType()))
    {
        switch (ptrType->getAddressSpace())
        {
        case AddressSpace::Global:
        case AddressSpace::GroupShared:
        case AddressSpace::StorageBuffer:
        case AddressSpace::UserPointer:
            return true;
        default:
            break;
        }
    }

    if (as<IRGlobalParam>(dst))
    {
        switch (dst->getDataType()->getOp())
        {
        case kIROp_GLSLShaderStorageBufferType:
        case kIROp_TextureType:
            return true;
        default:
            return false;
        }
    }

    if (auto param = as<IRParam>(dst))
    {
        auto paramType = param->getDataType();
        if (auto outType = as<IROutTypeBase>(paramType))
        {
            if (outType->getAddressSpace() == AddressSpace::GroupShared)
            {
                return true;
            }
            else if (skipFuncParamValidation)
            {
                // We haven't actually verified that this is a valid atomic operation destination,
                // but the callee wants to skip this specific validation.
                return true;
            }
        }
    }
    if (auto getElementPtr = as<IRGetElementPtr>(dst))
        return isValidAtomicDest(skipFuncParamValidation, getElementPtr->getBase());
    if (auto getOffsetPtr = as<IRGetOffsetPtr>(dst))
        return isValidAtomicDest(skipFuncParamValidation, getOffsetPtr->getBase());
    if (auto fieldAddress = as<IRFieldAddress>(dst))
        return isValidAtomicDest(skipFuncParamValidation, fieldAddress->getBase());

    return false;
}

void validateAtomicOperations(bool skipFuncParamValidation, DiagnosticSink* sink, IRInst* inst)
{
    switch (inst->getOp())
    {
    case kIROp_AtomicLoad:
    case kIROp_AtomicStore:
    case kIROp_AtomicExchange:
    case kIROp_AtomicCompareExchange:
    case kIROp_AtomicAdd:
    case kIROp_AtomicSub:
    case kIROp_AtomicAnd:
    case kIROp_AtomicOr:
    case kIROp_AtomicXor:
    case kIROp_AtomicMin:
    case kIROp_AtomicMax:
    case kIROp_AtomicInc:
    case kIROp_AtomicDec:
        {
            IRInst* destinationPtr = inst->getOperand(0);
            if (!isValidAtomicDest(skipFuncParamValidation, destinationPtr))
                sink->diagnose(inst->sourceLoc, Diagnostics::invalidAtomicDestinationPointer);
        }
        break;

    default:
        break;
    }

    for (auto child : inst->getModifiableChildren())
    {
        validateAtomicOperations(skipFuncParamValidation, sink, child);
    }
}

} // namespace Slang
