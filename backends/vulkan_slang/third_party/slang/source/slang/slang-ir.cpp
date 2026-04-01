// slang-ir.cpp
#include "slang-ir.h"

#include "../core/slang-basic.h"
#include "../core/slang-writer.h"
#include "slang-ir-dominators.h"
#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-mangle.h"

namespace Slang
{
struct IRSpecContext;

// !!!!!!!!!!!!!!!!!!!!!!!!!!!! DiagnosticSink Impls !!!!!!!!!!!!!!!!!!!!!

SourceLoc const& getDiagnosticPos(IRInst* inst)
{
    while (inst)
    {
        if (inst->sourceLoc.isValid())
            return inst->sourceLoc;
        inst = inst->parent;
    }
    static SourceLoc invalid = SourceLoc();
    return invalid;
}

void printDiagnosticArg(StringBuilder& sb, IRInst* irObject)
{
    if (!irObject)
        return;
    if (as<IRType>(irObject))
    {
        getTypeNameHint(sb, irObject);
        return;
    }
    if (auto nameHint = irObject->findDecoration<IRNameHintDecoration>())
    {
        sb << nameHint->getName();
        return;
    }
    if (auto linkage = irObject->findDecoration<IRLinkageDecoration>())
    {
        sb << linkage->getMangledName();
        return;
    }
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


bool isSimpleDecoration(IROp op)
{
    switch (op)
    {
    case kIROp_EarlyDepthStencilDecoration:
    case kIROp_KeepAliveDecoration:
    case kIROp_LineAdjInputPrimitiveTypeDecoration:
    case kIROp_LineInputPrimitiveTypeDecoration:
    case kIROp_NoInlineDecoration:
    case kIROp_DerivativeGroupQuadDecoration:
    case kIROp_DerivativeGroupLinearDecoration:
    case kIROp_PointInputPrimitiveTypeDecoration:
    case kIROp_PreciseDecoration:
    case kIROp_PublicDecoration:
    case kIROp_HLSLExportDecoration:
    case kIROp_ReadNoneDecoration:
    case kIROp_NoSideEffectDecoration:
    case kIROp_ForwardDifferentiableDecoration:
    case kIROp_BackwardDifferentiableDecoration:
    case kIROp_RequiresNVAPIDecoration:
    case kIROp_TriangleAdjInputPrimitiveTypeDecoration:
    case kIROp_TriangleInputPrimitiveTypeDecoration:
    case kIROp_UnsafeForceInlineEarlyDecoration:
    case kIROp_VulkanCallablePayloadDecoration:
    case kIROp_VulkanCallablePayloadInDecoration:
    case kIROp_VulkanHitAttributesDecoration:
    case kIROp_VulkanRayPayloadDecoration:
    case kIROp_VulkanRayPayloadInDecoration:
    case kIROp_VulkanHitObjectAttributesDecoration:
        {
            return true;
        }
    default:
        break;
    }
    return false;
}


IRInst* cloneGlobalValueWithLinkage(
    IRSpecContext* context,
    IRInst* originalVal,
    IRLinkageDecoration* originalLinkage);

struct IROpMapEntry
{
    IROp op;
    IROpInfo info;
};

// TODO: We should ideally be speeding up the name->inst
// mapping by using a dictionary, or even by pre-computing
// a hash table to be stored as a `static const` array.
//
// NOTE! That this array is now constructed in such a way that looking up
// an entry from an op is fast, by keeping blocks of main, and pseudo ops in same order
// as the ops themselves. Care must be taken to keep this constraint.
static const IROpMapEntry kIROps[] = {

// Main ops in order
#define INST(ID, MNEMONIC, ARG_COUNT, FLAGS) \
    {kIROp_##ID,                             \
     {                                       \
         #MNEMONIC,                          \
         ARG_COUNT,                          \
         FLAGS,                              \
     }},
#include "slang-ir-inst-defs.h"

    // Invalid op sentinel value comes after all the valid ones
    {kIROp_Invalid, {"invalid", 0, 0}},
};

IROpInfo getIROpInfo(IROp opIn)
{
    const int op = opIn & kIROpMask_OpMask;
    if (op < kIROpCount)
    {
        // It's a main op
        const auto& entry = kIROps[op];
        SLANG_ASSERT(entry.op == op);
        return entry.info;
    }

    // Don't know what this is
    SLANG_ASSERT(!"Invalid op");
    SLANG_ASSERT(kIROps[kIROpCount].op == kIROp_Invalid);
    return kIROps[kIROpCount].info;
}

IROp findIROp(const UnownedStringSlice& name)
{
    for (auto ee : kIROps)
    {
        if (name == ee.info.name)
            return ee.op;
    }

    return IROp(kIROp_Invalid);
}


//

void IRUse::debugValidate()
{
#ifdef _DEBUG
    auto uv = this->usedValue;
    if (!uv)
    {
        assert(!nextUse);
        assert(!prevLink);
        return;
    }

    auto pp = &uv->firstUse;
    for (auto u = uv->firstUse; u;)
    {
        assert(u->prevLink == pp);

        pp = &u->nextUse;
        u = u->nextUse;
    }
#endif
}

void IRUse::init(IRInst* u, IRInst* v)
{
    clear();
    user = u;
    usedValue = v;
    if (v)
    {
        nextUse = v->firstUse;
        prevLink = &v->firstUse;

        if (nextUse)
        {
            nextUse->prevLink = &this->nextUse;
        }

        v->firstUse = this;
    }
#ifdef SLANG_ENABLE_FULL_IR_VALIDATION
    debugValidate();
#endif
}

void IRUse::set(IRInst* uv)
{
    // Normally we should never be modifying the operand of an hoistable inst.
    // They can be modified by `replaceUsesWith`, or to be replaced by a new inst.
    SLANG_ASSERT(!getIROpInfo(user->getOp()).isHoistable() || uv == usedValue);
    init(user, uv);
}

void IRUse::clear()
{
    // This `IRUse` is part of the linked list
    // of uses for  `usedValue`.
#ifdef SLANG_ENABLE_FULL_IR_VALIDATION
    debugValidate();
#endif

    if (usedValue)
    {
#ifdef SLANG_ENABLE_FULL_IR_VALIDATION
        auto uv = usedValue;
#endif
        *prevLink = nextUse;
        if (nextUse)
        {
            nextUse->prevLink = prevLink;
        }

        user = nullptr;
        usedValue = nullptr;
        nextUse = nullptr;
        prevLink = nullptr;

#ifdef SLANG_ENABLE_FULL_IR_VALIDATION
        if (uv->firstUse)
            uv->firstUse->debugValidate();
#endif
    }
}

// IRInstListBase

void IRInstListBase::Iterator::operator++()
{
    if (inst)
    {
        inst = inst->next;
    }
}

IRInstListBase::Iterator IRInstListBase::begin()
{
    return Iterator(first);
}
IRInstListBase::Iterator IRInstListBase::end()
{
    return Iterator(last ? last->next : nullptr);
}

//

IRUse* IRInst::getOperands()
{
    // We assume that *all* instructions are laid out
    // in memory such that their arguments come right
    // after the first `sizeof(IRInst)` bytes.
    //
    // TODO: we probably need to be careful and make
    // this more robust.

    return (IRUse*)(this + 1);
}

IRDecoration* IRInst::findDecorationImpl(IROp decorationOp)
{
    for (auto dd : getDecorations())
    {
        if (dd->getOp() == decorationOp)
            return dd;
    }
    return nullptr;
}

IROperandList<IRAttr> IRInst::getAllAttrs()
{
    // We assume as an invariant that all attributes appear at the end of the operand
    // list, after all the non-attribute operands.
    //
    // We will therefore define a range that ends at the end of the operand list ...
    //
    IRUse* end = getOperands() + getOperandCount();
    //
    // ... and begins after the last non-attribute operand.
    //
    IRUse* cursor = getOperands();
    while (cursor != end && !as<IRAttr>(cursor->get()))
        cursor++;

    return IROperandList<IRAttr>(cursor, end);
}

// IRConstant

IRIntegerValue getIntVal(IRInst* inst)
{
    switch (inst->getOp())
    {
    default:
        SLANG_UNEXPECTED("needed a known integer value");
        UNREACHABLE_RETURN(0);

    case kIROp_IntLit:
        return static_cast<IRConstant*>(inst)->value.intVal;
        break;
    }
}

// IRCapabilitySet

CapabilitySet IRCapabilitySet::getCaps()
{
    switch (getOp())
    {
    case kIROp_CapabilityConjunction:
        {
            List<CapabilityName> atoms;

            Index count = (Index)getOperandCount();
            for (Index i = 0; i < count; ++i)
            {
                auto operand = cast<IRIntLit>(getOperand(i));
                atoms.add(CapabilityName(operand->getValue()));
            }

            return CapabilitySet(atoms.getCount(), atoms.getBuffer());
        }
        break;
    case kIROp_CapabilityDisjunction:
        {
            CapabilitySet result;
            Index count = (Index)getOperandCount();
            for (Index i = 0; i < count; ++i)
            {
                auto operand = cast<IRCapabilitySet>(getOperand(i));
                result.unionWith(operand->getCaps());
            }
            return result;
        }
        break;
    }
    return CapabilitySet();
}


// IRParam

IRParam* IRParam::getNextParam()
{
    return as<IRParam, IRDynamicCastBehavior::NoUnwrap>(getNextInst());
}

IRParam* IRParam::getPrevParam()
{
    return as<IRParam, IRDynamicCastBehavior::NoUnwrap>(getPrevInst());
}

// IRArrayTypeBase

IRInst* IRArrayTypeBase::getElementCount()
{
    if (auto arrayType = as<IRArrayType>(this))
        return arrayType->getOperand(1);

    return nullptr;
}

// IRPtrTypeBase

IRType* tryGetPointedToType(IRBuilder* builder, IRType* type)
{
    if (auto rateQualType = as<IRRateQualifiedType>(type))
    {
        type = rateQualType->getValueType();
    }

    // The "true" pointers and the pointer-like core module types are the easy cases.
    if (auto ptrType = as<IRPtrTypeBase>(type))
    {
        return ptrType->getValueType();
    }
    else if (auto ptrLikeType = as<IRPointerLikeType>(type))
    {
        return ptrLikeType->getElementType();
    }
    //
    // A more interesting case arises when we have a `BindExistentials<P<T>, ...>`
    // where `P<T>` is a pointer(-like) type.
    //
    else if (auto bindExistentials = as<IRBindExistentialsType>(type))
    {
        // We know that `BindExistentials` won't introduce its own
        // existential type parameters, nor will any of the pointer(-like)
        // type constructors `P`.
        //
        // Thus we know that the type that is pointed to should be
        // the same as `BindExistentials<T, ...>`.
        //
        auto baseType = bindExistentials->getBaseType();
        if (auto baseElementType = tryGetPointedToType(builder, baseType))
        {
            UInt existentialArgCount = bindExistentials->getExistentialArgCount();
            List<IRInst*> existentialArgs;
            for (UInt ii = 0; ii < existentialArgCount; ++ii)
            {
                existentialArgs.add(bindExistentials->getExistentialArg(ii));
            }
            return builder->getBindExistentialsType(
                baseElementType,
                existentialArgCount,
                existentialArgs.getBuffer());
        }
    }

    // TODO: We may need to handle other cases here.

    return nullptr;
}


// IRBlock

IRParam* IRBlock::getLastParam()
{
    IRParam* param = getFirstParam();
    if (!param)
        return nullptr;

    while (auto nextParam = param->getNextParam())
        param = nextParam;

    return param;
}

void IRBlock::addParam(IRParam* param)
{
    // If there are any existing parameters,
    // then insert after the last of them.
    //
    if (auto lastParam = getLastParam())
    {
        if (lastParam->next)
            param->insertAfter(lastParam);
        else
            param->insertAtEnd(this);
    }
    //
    // Otherwise, if there are any existing
    // "ordinary" instructions, insert before
    // the first of them.
    //
    else if (auto firstOrdinary = getFirstOrdinaryInst())
    {
        param->insertBefore(firstOrdinary);
    }
    //
    // Otherwise the block currently has neither
    // parameters nor orindary instructions,
    // so we can safely insert at the end of
    // the list of (raw) children.
    //
    else
    {
        param->insertAtEnd(this);
    }
}

// Similar to addParam, but instead of appending `param` to the end
// of the parameter list, this function inserts `param` before the
// head of the list.
void IRBlock::insertParamAtHead(IRParam* param)
{
    if (auto firstParam = getFirstParam())
    {
        param->insertBefore(firstParam);
    }
    else if (auto firstOrdinary = getFirstOrdinaryInst())
    {
        param->insertBefore(firstOrdinary);
    }
    else
    {
        param->insertAtEnd(this);
    }
}

IRInst* IRBlock::getFirstOrdinaryInst()
{
    // Find the last parameter (if any) of the block
    auto lastParam = getLastParam();
    if (lastParam)
    {
        // If there is a last parameter, then the
        // instructions after it are the ordinary
        // instructions.
        return lastParam->getNextInst();
    }
    else
    {
        // If there isn't a last parameter, then
        // there must not have been *any* parameters,
        // and so the first instruction in the block
        // is also the first ordinary one.
        return getFirstInst();
    }
}

IRInst* IRBlock::getLastOrdinaryInst()
{
    // Under normal circumstances, the last instruction
    // in the block is also the last ordinary instruction.
    // However, there is the special case of a block with
    // only parameters (which might happen as a temporary
    // state while we are building IR).
    auto inst = getLastInst();

    // If the last instruction is a parameter, then
    // there are no ordinary instructions, so the last
    // one is a null pointer.
    if (as<IRParam, IRDynamicCastBehavior::NoUnwrap>(inst))
        return nullptr;

    // Otherwise the last instruction is the last "ordinary"
    // instruction as well.
    return inst;
}


// The predecessors of a block should all show up as users
// of its value, so rather than explicitly store the CFG,
// we will recover it on demand from the use-def information.
//
// Note: we are really iterating over incoming/outgoing *edges*
// for a block, because there might be multiple uses of a block,
// if more than one way of an N-way branch targets the same block.

// Get the list of successor blocks for an instruction,
// which we expect to be the last instruction in a block.
static IRBlock::SuccessorList getSuccessors(IRInst* terminator)
{
    // If the block somehow isn't terminated, then
    // there is no way to read its successors, so
    // we return an empty list.
    if (!terminator || !as<IRTerminatorInst>(terminator))
        return IRBlock::SuccessorList(nullptr, nullptr);

    // Otherwise, based on the opcode of the terminator
    // instruction, we will build up our list of uses.
    IRUse* begin = nullptr;
    IRUse* end = nullptr;
    UInt stride = 1;

    auto operands = terminator->getOperands();
    switch (terminator->getOp())
    {
    case kIROp_Return:
    case kIROp_Unreachable:
    case kIROp_MissingReturn:
    case kIROp_GenericAsm:
        break;

    case kIROp_unconditionalBranch:
    case kIROp_loop:
        // unconditonalBranch <block>
        begin = operands + 0;
        end = begin + 1;
        break;

    case kIROp_conditionalBranch:
    case kIROp_ifElse:
        // conditionalBranch <condition> <trueBlock> <falseBlock>
        begin = operands + 1;
        end = begin + 2;
        break;

    case kIROp_Switch:
        // switch <val> <break> <default> <caseVal1> <caseBlock1> ...
        begin = operands + 2;

        // TODO: this ends up point one *after* the "one after the end"
        // location, so we should really change the representation
        // so that we don't need to form this pointer...
        end = operands + terminator->getOperandCount() + 1;
        stride = 2;
        break;
    case kIROp_TargetSwitch:
        begin = operands + 2;
        end = operands + terminator->getOperandCount() + 1;
        stride = 2;
        break;

    case kIROp_Defer:
        // defer <deferBlock> <mergeBlock> <scopeEndBlock>
        begin = operands + 0;
        end = begin + 1;
        break;

    default:
        SLANG_UNEXPECTED("unhandled terminator instruction");
        UNREACHABLE_RETURN(IRBlock::SuccessorList(nullptr, nullptr));
    }

    return IRBlock::SuccessorList(begin, end, stride);
}

static IRUse* adjustPredecessorUse(IRUse* use)
{
    // We will search until we either find a
    // suitable use, or run out of uses.
    for (; use; use = use->nextUse)
    {
        // We only want to deal with uses that represent
        // a "sucessor" operand to some terminator instruction.
        // We will re-use the logic for getting the successor
        // list from such an instruction.

        auto successorList = getSuccessors((IRInst*)use->getUser());

        if (use >= successorList.begin_ && use < successorList.end_)
        {
            UInt index = (use - successorList.begin_);
            if ((index % successorList.stride) == 0)
            {
                // This use is in the range of the sucessor list,
                // and so it represents a real edge between
                // blocks.
                return use;
            }
        }
    }

    // If we ran out of uses, then we are at the end
    // of the list of incoming edges.
    return nullptr;
}

IRBlock::PredecessorList IRBlock::getPredecessors()
{
    // We want to iterate over the predecessors of this block.
    // First, we resign ourselves to iterating over the
    // incoming edges, rather than the blocks themselves.
    // This might sound like a trival distinction, but it is
    // possible for there to be multiple edges between two
    // blocks (as for a `switch` with multiple cases that
    // map to the same code). Any client that wants just
    // the unique predecessor blocks needs to deal with
    // the deduplication themselves.
    //
    // Next, we note that for any predecessor edge, there will
    // be a use of this block in the terminator instruction of
    // the predecessor. We basically just want to iterate over
    // the users of this block, then, but we need to be careful
    // to rule out anything that doesn't actually represent
    // an edge. The `adjustPredecessorUse` function will be
    // used to search for a use that actually represents an edge.

    return PredecessorList(adjustPredecessorUse(firstUse));
}

UInt IRBlock::PredecessorList::getCount()
{
    UInt count = 0;
    for (auto ii : *this)
    {
        (void)ii;
        count++;
    }
    return count;
}

bool IRBlock::PredecessorList::isEmpty()
{
    return !(begin() != end());
}


void IRBlock::PredecessorList::Iterator::operator++()
{
    if (!use)
        return;
    use = adjustPredecessorUse(use->nextUse);
}

IRBlock* IRBlock::PredecessorList::Iterator::operator*()
{
    if (!use)
        return nullptr;
    return (IRBlock*)use->getUser()->parent;
}

IRBlock::SuccessorList IRBlock::getSuccessors()
{
    // The successors of a block will all be listed
    // as operands of its terminator instruction.
    // Depending on the terminator, we might have
    // different numbers of operands to deal with.
    //
    // (We might also have to deal with a "stride"
    // in the case where the basic-block operands
    // are mixed up with non-block operands)

    auto terminator = getLastInst();
    return Slang::getSuccessors(terminator);
}

UInt IRBlock::SuccessorList::getCount()
{
    UInt count = 0;
    for (auto ii : *this)
    {
        (void)ii;
        count++;
    }
    return count;
}

void IRBlock::SuccessorList::Iterator::operator++()
{
    use += stride;
}

IRBlock* IRBlock::SuccessorList::Iterator::operator*()
{
    return (IRBlock*)use->get();
}

UInt IRUnconditionalBranch::getArgCount()
{
    switch (getOp())
    {
    case kIROp_unconditionalBranch:
        return getOperandCount() - 1;

    case kIROp_loop:
        return getOperandCount() - 3;

    default:
        SLANG_UNEXPECTED("unhandled unconditional branch opcode");
        UNREACHABLE_RETURN(0);
    }
}

IRUse* IRUnconditionalBranch::getArgs()
{
    switch (getOp())
    {
    case kIROp_unconditionalBranch:
        return getOperands() + 1;

    case kIROp_loop:
        return getOperands() + 3;

    default:
        SLANG_UNEXPECTED("unhandled unconditional branch opcode");
        UNREACHABLE_RETURN(0);
    }
}

void IRUnconditionalBranch::removeArgument(UInt index)
{
    switch (getOp())
    {
    case kIROp_unconditionalBranch:
        removeOperand(1 + index);
        break;
    case kIROp_loop:
        removeOperand(3 + index);
        break;
    default:
        SLANG_UNEXPECTED("unhandled unconditional branch opcode");
    }
}

IRInst* IRUnconditionalBranch::getArg(UInt index)
{
    return getArgs()[index].usedValue;
}

IRParam* IRGlobalValueWithParams::getFirstParam()
{
    auto entryBlock = getFirstBlock();
    if (!entryBlock)
        return nullptr;

    return entryBlock->getFirstParam();
}

IRParam* IRGlobalValueWithParams::getLastParam()
{
    auto entryBlock = getFirstBlock();
    if (!entryBlock)
        return nullptr;

    return entryBlock->getLastParam();
}

IRInstList<IRParam> IRGlobalValueWithParams::getParams()
{
    auto entryBlock = getFirstBlock();
    if (!entryBlock)
        return IRInstList<IRParam>();

    return entryBlock->getParams();
}

IRInst* IRGlobalValueWithParams::getFirstOrdinaryInst()
{
    auto firstBlock = getFirstBlock();
    if (!firstBlock)
        return nullptr;
    return firstBlock->getFirstOrdinaryInst();
}

// IRFunc

IRType* IRFunc::getResultType()
{
    return getDataType()->getResultType();
}
UInt IRFunc::getParamCount()
{
    return getDataType()->getParamCount();
}
IRType* IRFunc::getParamType(UInt index)
{
    return getDataType()->getParamType(index);
}

void fixUpFuncType(IRFunc* func, IRType* resultType)
{
    SLANG_ASSERT(func);

    auto irModule = func->getModule();
    SLANG_ASSERT(irModule);

    IRBuilder builder(irModule);
    builder.setInsertBefore(func);

    List<IRType*> paramTypes;
    for (auto param : func->getParams())
    {
        paramTypes.add(param->getFullType());
    }

    auto funcType = builder.getFuncType(paramTypes, resultType);
    builder.setDataType(func, funcType);
}

void fixUpFuncType(IRFunc* func)
{
    fixUpFuncType(func, func->getResultType());
}

//

bool isTerminatorInst(IROp op)
{
    switch (op)
    {
    default:
        return false;

    case kIROp_Return:
    case kIROp_unconditionalBranch:
    case kIROp_conditionalBranch:
    case kIROp_loop:
    case kIROp_ifElse:
    case kIROp_Switch:
    case kIROp_Unreachable:
    case kIROp_MissingReturn:
    case kIROp_Defer:
        return true;
    }
}

bool isTerminatorInst(IRInst* inst)
{
    if (!inst)
        return false;
    return isTerminatorInst(inst->getOp());
}

//
// IRTypeLayout
//

IRTypeSizeAttr* IRTypeLayout::findSizeAttr(LayoutResourceKind kind)
{
    // TODO: If we could assume the attributes were sorted
    // by `kind`, then we could use a binary search here
    // instead of linear.
    //
    // In practice, the number of entries will be very small,
    // so the cost of the linear search should not be too bad.

    for (auto sizeAttr : getSizeAttrs())
    {
        if (sizeAttr->getResourceKind() == kind)
            return sizeAttr;
    }
    return nullptr;
}

IRTypeLayout* IRTypeLayout::unwrapArray()
{
    auto typeLayout = this;
    while (auto arrayTypeLayout = as<IRArrayTypeLayout>(typeLayout))
        typeLayout = arrayTypeLayout->getElementTypeLayout();
    return typeLayout;
}

IRTypeLayout* IRTypeLayout::getPendingDataTypeLayout()
{
    if (auto attr = findAttr<IRPendingLayoutAttr>())
        return cast<IRTypeLayout>(attr->getLayout());
    return nullptr;
}

IROperandList<IRTypeSizeAttr> IRTypeLayout::getSizeAttrs()
{
    return findAttrs<IRTypeSizeAttr>();
}

IRTypeLayout::Builder::Builder(IRBuilder* irBuilder)
    : m_irBuilder(irBuilder)
{
}

void IRTypeLayout::Builder::addResourceUsage(LayoutResourceKind kind, LayoutSize size)
{
    auto& resInfo = m_resInfos[Int(kind)];
    resInfo.kind = kind;
    resInfo.size += size;
}

void IRTypeLayout::Builder::addResourceUsage(IRTypeSizeAttr* sizeAttr)
{
    addResourceUsage(sizeAttr->getResourceKind(), sizeAttr->getSize());
}

void IRTypeLayout::Builder::addResourceUsageFrom(IRTypeLayout* typeLayout)
{
    for (auto sizeAttr : typeLayout->getSizeAttrs())
    {
        addResourceUsage(sizeAttr);
    }
}

IRTypeLayout* IRTypeLayout::Builder::build()
{
    IRBuilder* irBuilder = getIRBuilder();

    List<IRInst*> operands;

    addOperands(operands);
    addAttrs(operands);

    return irBuilder->getTypeLayout(getOp(), operands);
}

void IRTypeLayout::Builder::addOperands(List<IRInst*>& operands)
{
    addOperandsImpl(operands);
}

void IRTypeLayout::Builder::addAttrs(List<IRInst*>& operands)
{
    auto irBuilder = getIRBuilder();

    for (auto resInfo : m_resInfos)
    {
        if (resInfo.kind == LayoutResourceKind::None)
            continue;

        IRInst* sizeAttr = irBuilder->getTypeSizeAttr(resInfo.kind, resInfo.size);
        operands.add(sizeAttr);
    }

    if (auto pendingTypeLayout = m_pendingTypeLayout)
    {
        operands.add(irBuilder->getPendingLayoutAttr(pendingTypeLayout));
    }

    addAttrsImpl(operands);
}

//
// IRParameterGroupTypeLayout
//

void IRParameterGroupTypeLayout::Builder::addOperandsImpl(List<IRInst*>& ioOperands)
{
    ioOperands.add(m_containerVarLayout);
    ioOperands.add(m_elementVarLayout);
    ioOperands.add(m_offsetElementTypeLayout);
}

IRParameterGroupTypeLayout* IRParameterGroupTypeLayout::Builder::build()
{
    return cast<IRParameterGroupTypeLayout>(Super::Builder::build());
}

//
// IRStructTypeLayout
//

void IRStructTypeLayout::Builder::addAttrsImpl(List<IRInst*>& ioOperands)
{
    auto irBuilder = getIRBuilder();
    for (auto field : m_fields)
    {
        ioOperands.add(irBuilder->getFieldLayoutAttr(field.key, field.layout));
    }
}

//
// IRTupleTypeLayout
//

void IRTupleTypeLayout::Builder::addAttrsImpl(List<IRInst*>& ioOperands)
{
    auto irBuilder = getIRBuilder();
    for (auto field : m_fields)
    {
        ioOperands.add(irBuilder->getTupleFieldLayoutAttr(field.layout));
    }
}

//
// IRArrayTypeLayout
//

void IRArrayTypeLayout::Builder::addOperandsImpl(List<IRInst*>& ioOperands)
{
    ioOperands.add(m_elementTypeLayout);
}

//
// IRStructuredBufferTypeLayout
//

void IRStructuredBufferTypeLayout::Builder::addOperandsImpl(List<IRInst*>& ioOperands)
{
    ioOperands.add(m_elementTypeLayout);
}

//
// IRPointerTypeLayout
//

void IRPointerTypeLayout::Builder::addOperandsImpl(List<IRInst*>& ioOperands)
{
    SLANG_UNUSED(ioOperands);
    // TODO(JS): For now we don't store the value types layout to avoid
    // infinite recursion.
    // ioOperands.add(m_valueTypeLayout);
}

//
// IRStreamOutputTypeLayout
//

void IRStreamOutputTypeLayout::Builder::addOperandsImpl(List<IRInst*>& ioOperands)
{
    ioOperands.add(m_elementTypeLayout);
}

//
// IRMatrixTypeLayout
//

IRMatrixTypeLayout::Builder::Builder(IRBuilder* irBuilder, MatrixLayoutMode mode)
    : Super::Builder(irBuilder)
{
    m_modeInst = irBuilder->getIntValue(irBuilder->getIntType(), IRIntegerValue(mode));
}

void IRMatrixTypeLayout::Builder::addOperandsImpl(List<IRInst*>& ioOperands)
{
    ioOperands.add(m_modeInst);
}

//
// IRVarLayout
//

bool IRVarLayout::usesResourceKind(LayoutResourceKind kind)
{
    // TODO: basing this check on whether or not the
    // var layout has an entry for `kind` means that
    // we can't just optimize away any entry where
    // the offset is zero (which might be a small
    // but nice optimization). We could consider shifting
    // this test to use the entries on the type layout
    // instead (since non-zero resource consumption
    // should be an equivalent test).

    return findOffsetAttr(kind) != nullptr;
}

bool IRVarLayout::usesResourceFromKinds(LayoutResourceKindFlags kindFlags)
{
    // Like usesResourceKind this works because there is an offset stored even if it's 0.
    if (kindFlags)
    {
        for (auto offsetAttr : getOffsetAttrs())
        {
            if (LayoutResourceKindFlag::make(offsetAttr->getResourceKind()) & kindFlags)
                return true;
        }
    }
    return false;
}

IRSystemValueSemanticAttr* IRVarLayout::findSystemValueSemanticAttr()
{
    return findAttr<IRSystemValueSemanticAttr>();
}

IRVarOffsetAttr* IRVarLayout::findOffsetAttr(LayoutResourceKind kind)
{
    for (auto offsetAttr : getOffsetAttrs())
    {
        if (offsetAttr->getResourceKind() == kind)
            return offsetAttr;
    }
    return nullptr;
}

IROperandList<IRVarOffsetAttr> IRVarLayout::getOffsetAttrs()
{
    return findAttrs<IRVarOffsetAttr>();
}

Stage IRVarLayout::getStage()
{
    if (auto stageAttr = findAttr<IRStageAttr>())
        return stageAttr->getStage();
    return Stage::Unknown;
}

IRVarLayout* IRVarLayout::getPendingVarLayout()
{
    if (auto pendingLayoutAttr = findAttr<IRPendingLayoutAttr>())
    {
        return cast<IRVarLayout>(pendingLayoutAttr->getLayout());
    }
    return nullptr;
}

IRVarLayout::Builder::Builder(IRBuilder* irBuilder, IRTypeLayout* typeLayout)
    : m_irBuilder(irBuilder), m_typeLayout(typeLayout)
{
}

bool IRVarLayout::Builder::usesResourceKind(LayoutResourceKind kind)
{
    return m_resInfos[Int(kind)].kind != LayoutResourceKind::None;
}

IRVarLayout::Builder::ResInfo* IRVarLayout::Builder::findOrAddResourceInfo(LayoutResourceKind kind)
{
    auto& resInfo = m_resInfos[Int(kind)];
    resInfo.kind = kind;
    return &resInfo;
}

void IRVarLayout::Builder::setSystemValueSemantic(String const& name, UInt index)
{
    m_systemValueSemantic = getIRBuilder()->getSystemValueSemanticAttr(name, index);
}

void IRVarLayout::Builder::setUserSemantic(String const& name, UInt index)
{
    m_userSemantic = getIRBuilder()->getUserSemanticAttr(name, index);
}

void IRVarLayout::Builder::setStage(Stage stage)
{
    m_stageAttr = getIRBuilder()->getStageAttr(stage);
}

void IRVarLayout::Builder::cloneEverythingButOffsetsFrom(IRVarLayout* that)
{
    if (auto systemValueSemantic = that->findAttr<IRSystemValueSemanticAttr>())
        m_systemValueSemantic = systemValueSemantic;

    if (auto userSemantic = that->findAttr<IRUserSemanticAttr>())
        m_userSemantic = userSemantic;

    if (auto stageAttr = that->findAttr<IRStageAttr>())
        m_stageAttr = stageAttr;
}

IRVarLayout* IRVarLayout::Builder::build()
{
    SLANG_ASSERT(m_typeLayout);

    IRBuilder* irBuilder = getIRBuilder();

    List<IRInst*> operands;

    operands.add(m_typeLayout);

    for (auto resInfo : m_resInfos)
    {
        if (resInfo.kind == LayoutResourceKind::None)
            continue;

        IRInst* varOffsetAttr =
            irBuilder->getVarOffsetAttr(resInfo.kind, resInfo.offset, resInfo.space);
        operands.add(varOffsetAttr);
    }

    if (auto semanticAttr = m_userSemantic)
        operands.add(semanticAttr);

    if (auto semanticAttr = m_systemValueSemantic)
        operands.add(semanticAttr);

    if (auto stageAttr = m_stageAttr)
        operands.add(stageAttr);

    if (auto pendingVarLayout = m_pendingVarLayout)
    {
        IRInst* pendingLayoutAttr = irBuilder->getPendingLayoutAttr(pendingVarLayout);
        operands.add(pendingLayoutAttr);
    }

    return irBuilder->getVarLayout(operands);
}

//
// IREntryPointLayout
//

IRStructTypeLayout* getScopeStructLayout(IREntryPointLayout* scopeLayout)
{
    auto scopeTypeLayout = scopeLayout->getParamsLayout()->getTypeLayout();

    if (auto constantBufferTypeLayout = as<IRParameterGroupTypeLayout>(scopeTypeLayout))
    {
        scopeTypeLayout = constantBufferTypeLayout->getOffsetElementTypeLayout();
    }

    if (auto structTypeLayout = as<IRStructTypeLayout>(scopeTypeLayout))
    {
        return structTypeLayout;
    }

    SLANG_UNEXPECTED("uhandled global-scope binding layout");
    UNREACHABLE_RETURN(nullptr);
}

//

IRInst* IRInsertLoc::getParent() const
{
    auto inst = getInst();
    switch (getMode())
    {
    default:
    case Mode::None:
        return nullptr;
    case Mode::Before:
    case Mode::After:
        return inst->getParent();
    case Mode::AtStart:
    case Mode::AtEnd:
        return inst;
    }
}

IRBlock* IRInsertLoc::getBlock() const
{
    return as<IRBlock>(getParent());
}

// Get the current function (or other value with code)
// that we are inserting into (if any).
IRInst* IRInsertLoc::getFunc() const
{
    auto pp = getParent();
    if (const auto block = as<IRBlock>(pp))
    {
        pp = pp->getParent();
    }
    if (as<IRGlobalValueWithCode>(pp) || as<IRExpand>(pp))
        return pp;
    return nullptr;
}

void addHoistableInst(IRBuilder* builder, IRInst* inst);

// Add an instruction into the current scope
void IRBuilder::addInst(IRInst* inst)
{
    if (getIROpInfo(inst->getOp()).isGlobal())
    {
        addHoistableInst(this, inst);
        return;
    }

    if (!inst->parent)
        inst->insertAt(m_insertLoc);
}

IRInst* IRBuilder::replaceOperand(IRUse* use, IRInst* newValue)
{
    auto user = use->getUser();
    if (user->getModule())
    {
        user->getModule()->getDeduplicationContext()->getInstReplacementMap().tryGetValue(
            newValue,
            newValue);
    }

    if (!getIROpInfo(user->getOp()).isHoistable())
    {
        use->set(newValue);
        return user;
    }

    // If user is hoistable, we need to remove it from the global number map first,
    // perform the update, then try to reinsert it back to the global number map.
    // If we find an equivalent entry already exists in the global number map,
    // we return the existing entry.
    auto builder = user->getModule()->getDeduplicationContext();
    builder->_removeGlobalNumberingEntry(user);
    use->init(user, newValue);

    IRInst* existingVal = nullptr;
    if (builder->getGlobalValueNumberingMap().tryGetValue(IRInstKey{user}, existingVal))
    {
        user->replaceUsesWith(existingVal);
        return existingVal;
    }
    else
    {
        builder->_addGlobalNumberingEntry(user);
        return user;
    }
}

// Given two parent instructions, pick the better one to use as as
// insertion location for a "hoistable" instruction.
//
IRInst* mergeCandidateParentsForHoistableInst(IRInst* left, IRInst* right)
{
    // If the candidates are both the same, then who cares?
    if (left == right)
        return left;

    // If either `left` or `right` is a block, then we need to be
    // a bit careful, because blocks can see other values just using
    // the dominance relationship, without a direct parent-child relationship.
    //
    // First, check if each of `left` and `right` is a block.
    //
    auto leftBlock = as<IRBlock>(left);
    auto rightBlock = as<IRBlock>(right);
    //
    // As a special case, if both of these are blocks in the same parent,
    // then we need to pick between them based on dominance.
    //
    if (leftBlock && rightBlock && (leftBlock->getParent() == rightBlock->getParent()))
    {
        // We assume that the order of basic blocks in a function is compatible
        // with the dominance relationship (that is, if A dominates B, then
        // A comes before B in the list of blocks), so it suffices to pick
        // the *later* of the two blocks.
        //
        // There are ways we could try to speed up this search, but no matter
        // what it will be O(n) in the number of blocks, unless we build
        // an explicit dominator tree, which is infeasible during IR building.
        // Thus we just do a simple linear walk here.
        //
        // We will start at `leftBlock` and walk forward, until either...
        //
        for (auto ll = leftBlock; ll; ll = ll->getNextBlock())
        {
            // ... we see `rightBlock` (in which case `rightBlock` came later), or ...
            //
            if (ll == rightBlock)
                return rightBlock;
        }
        //
        // ... we run out of blocks (in which case `leftBlock` came later).
        //
        return leftBlock;
    }

    //
    // If the special case above doesn't apply, then `left` or `right` might
    // still be a block, but they aren't blocks nested in the same function.
    // We will find the first non-block ancestor of `left` and/or `right`.
    // This will either be the inst itself (it is isn't a block), or
    // its immediate parent (if it *is* a block).
    //
    auto leftNonBlock = leftBlock ? leftBlock->getParent() : left;
    auto rightNonBlock = rightBlock ? rightBlock->getParent() : right;

    // If either side is null, then take the non-null one.
    //
    if (!leftNonBlock)
        return right;
    if (!rightNonBlock)
        return left;

    // If the non-block on the left or right is a descendent of
    // the other, then that is what we should use.
    //
    IRInst* parentNonBlock = nullptr;
    for (auto ll = leftNonBlock; ll; ll = ll->getParent())
    {
        if (ll == rightNonBlock)
        {
            parentNonBlock = leftNonBlock;
            break;
        }
    }
    for (auto rr = rightNonBlock; rr; rr = rr->getParent())
    {
        if (rr == leftNonBlock)
        {
            SLANG_ASSERT(!parentNonBlock || parentNonBlock == leftNonBlock);
            parentNonBlock = rightNonBlock;
            break;
        }
    }

    // As a matter of validity in the IR, we expect one
    // of the two to be an ancestor (in the non-block case),
    // because otherwise we'd be violating the basic dominance
    // assumptions.
    //
    SLANG_ASSERT(parentNonBlock);

    // As a fallback, try to use the left parent as a default
    // in case things go badly.
    //
    if (!parentNonBlock)
    {
        parentNonBlock = leftNonBlock;
    }

    IRInst* parent = parentNonBlock;

    // At this point we've found a non-block parent where we
    // could stick things, but we have to fix things up in
    // case we should be inserting into a block beneath
    // that non-block parent.
    if (leftBlock && (parentNonBlock == leftNonBlock))
    {
        // We have a left block, and have picked its parent.

        // It cannot be the case that there is a right block
        // with the same parent, or else our special case
        // would have triggered at the start.
        SLANG_ASSERT(!rightBlock || (parentNonBlock != rightNonBlock));

        parent = leftBlock;
    }
    else if (rightBlock && (parentNonBlock == rightNonBlock))
    {
        // We have a right block, and have picked its parent.

        // We already tested above, so we know there isn't a
        // matching situation on the left side.

        parent = rightBlock;
    }

    // Okay, we've picked the parent we want to insert into,
    // *but* one last special case arises, because an `IRGlobalValueWithCode`
    // is not actually a suitable place to insert instructions.
    // Furthermore, there is no actual need to insert instructions at
    // that scope, because any parameters, etc. are actually attached
    // to the block(s) within the function.
    if (auto parentFunc = as<IRGlobalValueWithCode>(parent))
    {
        // Insert in the parent of the function (or other value with code).
        // We know that the parent must be able to hold ordinary instructions,
        // because it was able to hold this `IRGlobalValueWithCode`
        parent = parentFunc->getParent();
    }

    return parent;
}

IRInst* IRModule::_allocateInst(IROp op, Int operandCount, size_t minSizeInBytes)
{
    // There are two basic cases for instructions that affect how we compute size:
    //
    // * The default case is that an instruction's state is fully defined by the fields
    //   in the `IRInst` base type, along with the trailing operand list (a tail-allocated
    //   array of `IRUse`s. Almost all instructions need space allocated this way.
    //
    // * A small number of cases (currently `IRConstant`s and the `IRModule` type) have
    //   *zero* operands but include additional state beyond the fields in `IRInst`.
    //   For these cases we want to ensure that at least `sizeof(T)` bytes are allocated,
    //   based on the specific leaf type `T`.
    //
    // We handle the combination of the two cases by just taking the maximum of the two
    // different sizes.
    //
    size_t defaultSize = sizeof(IRInst) + (operandCount) * sizeof(IRUse);
    size_t totalSize = minSizeInBytes > defaultSize ? minSizeInBytes : defaultSize;

    IRInst* inst = (IRInst*)m_memoryArena.allocateAndZero(totalSize);

    // TODO: Is it actually important to run a constructor here?
    new (inst) IRInst();

    inst->operandCount = uint32_t(operandCount);
    inst->m_op = op;

    return inst;
}

/// Return whichever of `left` or `right` represents the later point in a common parent
static IRInst* pickLaterInstInSameParent(IRInst* left, IRInst* right)
{
    // When using instructions to represent insertion locations,
    // a null instruction represents the end of the parent block,
    // so if either of the two instructions is null, it indicates
    // the end of the parent, and thus comes later.
    //
    if (!left)
        return nullptr;
    if (!right)
        return nullptr;

    // In the non-null case, we must have the precondition that
    // the two candidates have the same parent.
    //
    SLANG_ASSERT(left->getParent() == right->getParent());

    // No matter what, figuring out which instruction comes first
    // is a linear-time operation in the number of instructions
    // in the same parent, but we can optimize based on the
    // assumption that in common cases one of the following will
    // hold:
    //
    // * `left` and `right` are close to one another in the IR
    // * `left` and/or `right` is close to the start of its parent
    //
    // To optimize for those conditions, we create two cursors that
    // start at `left` and `right` respectively, and scan backward.
    //
    auto ll = left;
    auto rr = right;
    for (;;)
    {
        // If one of the cursors runs into the other while scanning
        // backwards, then it implies it must have been the later
        // of the two.
        //
        // This is our early-exit condition for `left` and `right`
        // being close together.
        //
        // Note: this condition will trigger on the first iteration
        // in the case where `left == right`.
        //
        if (ll == right)
            return left;
        if (rr == left)
            return right;

        // If one of the cursors reaches the start of the block,
        // then that implies it started at the earlier position.
        // In that case, the other candidate must be the later
        // one.
        //
        // This is the early-exit condition for `left` and/or `right`
        // being close to the start of the parent.
        //
        if (!ll)
            return right;
        if (!rr)
            return left;

        // Otherwise, we move both cursors backward and continue
        // the search.
        //
        ll = ll->getPrevInst();
        rr = rr->getPrevInst();

        // Note: in the worst case, one of the cursors is
        // at the end of the parent, and the other is halfway
        // through, so that each cursor needs to visit half
        // of the instructions in the parent before we reach
        // one of our termination conditions.
        //
        // As a result the worst-case running time is still O(N),
        // and there is nothing we can do to improve that
        // with our linked-list representation.
        //
        // If the assumptions given turn out to be wrong, and
        // we find that a common case is instructions close
        // to the *end* of a block, we can either flip the
        // direction that the cursors traverse, or even add
        // two more cursors that scan forward instead of
        // backward.
    }
}

// Given an instruction that represents a constant, a type, etc.
// Try to "hoist" it as far toward the global scope as possible
// to insert it at a location where it will be maximally visible.
//
void addHoistableInst(IRBuilder* builder, IRInst* inst)
{
    // Start with the assumption that we would insert this instruction
    // into the global scope (the instruction that represents the module)
    IRInst* parent = builder->getModule()->getModuleInst();

    // The above decision might be invalid, because there might be
    // one or more operands of the instruction that are defined in
    // more deeply nested parents than the global scope.
    //
    // Therefore, we will scan the operands of the instruction, and
    // look at the parents that define them.
    //
    UInt operandCount = inst->getOperandCount();
    for (UInt ii = 0; ii < operandCount; ++ii)
    {
        auto operand = inst->getOperand(ii);
        if (!operand)
            continue;

        auto operandParent = operand->getParent();

        parent = mergeCandidateParentsForHoistableInst(parent, operandParent);
    }
    if (inst->getFullType())
    {
        parent = mergeCandidateParentsForHoistableInst(parent, inst->getFullType()->getParent());
    }

    // We better have ended up with a parent to insert into,
    // or else the invariants of our IR have been violated.
    //
    SLANG_ASSERT(parent);

    // Once we determine the parent instruction that the
    // new instruction should be inserted into, we need
    // to find an appropriate place to insert it.
    //
    // There are two concerns at play here, both of which
    // stem from the property that within a block we
    // require definitions to precede their uses.
    //
    // The first concern is that we want to emit a
    // "hoistable" instruction like a type as early as possible,
    // so that if a subsequent optimization pass requests
    // the same type/value again, it doesn't get a cached/deduplicated
    // pointer to an instruction that comes after the code being
    // processed.
    //
    // The second concern is that we must emit any hoistable
    // instruction after any of its operands (or its type)
    // if they come from the same block/parent.
    //
    // These two conditions together indicate that we want
    // to insert the instruction right after whichever of
    // its operands come last in the parent block and if
    // none of the operands come from the same block, we
    // should try to insert it as early as possible in
    // that block.
    //
    // We want to insert a hoistable instruction at the
    // earliest possible point in its parent, which
    // should be right after whichever of its operands
    // is defined in that same block (if any)
    //
    // We will solve this problem by computing the
    // earliest instruction that it would be valid for
    // us to insert before.
    //
    // We start by considering insertion before the
    // first instruction in the parent (if any) and
    // then move the insertion point later as needed.
    //
    // Note: a null `insertBeforeInst` is used
    // here to mean to insert at the end of the parent.
    //
    IRInst* insertBeforeInst = parent->getFirstChild();

    // Hoistable instructions are always "ordinary"
    // instructions, so they need to come after
    // any parameters of the parent.
    //
    while (insertBeforeInst && insertBeforeInst->getOp() == kIROp_Param)
    {
        insertBeforeInst = insertBeforeInst->getNextInst();
    }

    // For instructions that will be placed at module scope,
    // we don't care about relative ordering, but for everything
    // else, we want to ensure that an instruction comes after
    // its type and operands.
    //
    if (!as<IRModuleInst>(parent))
    {
        // We need to make sure that if any of
        // the operands of `inst` come from the same
        // block that we insert after them.
        //
        for (UInt ii = 0; ii < operandCount; ++ii)
        {
            auto operand = inst->getOperand(ii);
            if (!operand)
                continue;

            if (operand->getParent() != parent)
                continue;

            insertBeforeInst = pickLaterInstInSameParent(insertBeforeInst, operand->getNextInst());
        }
        //
        // Similarly, if the type of `inst` comes from
        // the same parent, then we need to make sure
        // we insert after the type.
        //
        if (auto type = inst->getFullType())
        {
            if (type->getParent() == parent)
            {
                insertBeforeInst = pickLaterInstInSameParent(insertBeforeInst, type->getNextInst());
            }
        }
    }

    if (insertBeforeInst)
    {
        inst->insertBefore(insertBeforeInst);
    }
    else
    {
        inst->insertAtEnd(parent);
    }
}

void IRBuilder::_maybeSetSourceLoc(IRInst* inst)
{
    auto sourceLocInfo = getSourceLocInfo();
    if (!sourceLocInfo)
        return;

    // Try to find something with usable location info
    for (;;)
    {
        if (sourceLocInfo->sourceLoc.getRaw())
            break;

        if (!sourceLocInfo->next)
            break;

        sourceLocInfo = sourceLocInfo->next;
    }

    inst->sourceLoc = sourceLocInfo->sourceLoc;
}

#if SLANG_ENABLE_IR_BREAK_ALLOC
SLANG_API uint32_t _slangIRAllocBreak = 0xFFFFFFFF;
uint32_t& _debugGetIRAllocCounter()
{
    static uint32_t counter = 0;
    return counter;
}
uint32_t _debugGetAndIncreaseInstCounter()
{
    if (_slangIRAllocBreak != 0xFFFFFFFF && _debugGetIRAllocCounter() == _slangIRAllocBreak)
    {
#if _WIN32 && defined(_MSC_VER)
        __debugbreak();
#endif
    }
    return _debugGetIRAllocCounter()++;
}
#endif

IRInst* IRBuilder::_createInst(
    size_t minSizeInBytes,
    IRType* type,
    IROp op,
    Int fixedArgCount,
    IRInst* const* fixedArgs,
    Int varArgListCount,
    Int const* listArgCounts,
    IRInst* const* const* listArgs)
{
    IRInst* instReplacement = type;
    m_dedupContext->getInstReplacementMap().tryGetValue(type, instReplacement);
    type = (IRType*)instReplacement;

    if (getIROpInfo(op).flags & kIROpFlag_Hoistable)
    {
        return _findOrEmitHoistableInst(
            type,
            op,
            fixedArgCount,
            fixedArgs,
            varArgListCount,
            listArgCounts,
            listArgs);
    }

    Int varArgCount = 0;
    for (Int ii = 0; ii < varArgListCount; ++ii)
    {
        varArgCount += listArgCounts[ii];
    }

    Int totalOperandCount = fixedArgCount + varArgCount;

    auto module = getModule();
    SLANG_ASSERT(module);
    IRInst* inst = module->_allocateInst(op, totalOperandCount, minSizeInBytes);

#if SLANG_ENABLE_IR_BREAK_ALLOC
    inst->_debugUID = _debugGetAndIncreaseInstCounter();
#endif

    inst->typeUse.init(inst, type);

    _maybeSetSourceLoc(inst);

    auto operand = inst->getOperands();

    for (Int aa = 0; aa < fixedArgCount; ++aa)
    {
        if (fixedArgs)
        {
            auto arg = fixedArgs[aa];
            m_dedupContext->getInstReplacementMap().tryGetValue(arg, arg);
            operand->init(inst, arg);
        }
        else
        {
            operand->init(inst, nullptr);
        }
        operand++;
    }

    for (Int ii = 0; ii < varArgListCount; ++ii)
    {
        Int listArgCount = listArgCounts[ii];
        for (Int jj = 0; jj < listArgCount; ++jj)
        {
            if (listArgs[ii])
            {
                auto arg = listArgs[ii][jj];
                m_dedupContext->getInstReplacementMap().tryGetValue(arg, arg);
                operand->init(inst, arg);
            }
            else
            {
                operand->init(inst, nullptr);
            }
            operand++;
        }
    }
    return inst;
}

// Create an IR instruction/value and initialize it.
//
// In this case `argCount` and `args` represent the
// arguments *after* the type (which is a mandatory
// argument for all instructions).
template<typename T>
static T* createInstImpl(
    IRBuilder* builder,
    IROp op,
    IRType* type,
    Int fixedArgCount,
    IRInst* const* fixedArgs,
    Int varArgListCount,
    Int const* listArgCounts,
    IRInst* const* const* listArgs)
{
    return (T*)builder->_createInst(
        sizeof(T),
        type,
        op,
        fixedArgCount,
        fixedArgs,
        varArgListCount,
        listArgCounts,
        listArgs);
}

template<typename T>
static T* createInstImpl(
    IRBuilder* builder,
    IROp op,
    IRType* type,
    Int fixedArgCount,
    IRInst* const* fixedArgs,
    Int varArgCount = 0,
    IRInst* const* varArgs = nullptr)
{
    return createInstImpl<T>(
        builder,
        op,
        type,
        fixedArgCount,
        fixedArgs,
        1,
        &varArgCount,
        &varArgs);
}

template<typename T>
static T* createInst(IRBuilder* builder, IROp op, IRType* type, Int argCount, IRInst* const* args)
{
    return createInstImpl<T>(builder, op, type, argCount, args);
}

template<typename T>
static T* createInst(IRBuilder* builder, IROp op, IRType* type)
{
    return createInstImpl<T>(builder, op, type, 0, nullptr);
}

template<typename T>
static T* createInst(IRBuilder* builder, IROp op, IRType* type, IRInst* arg)
{
    return createInstImpl<T>(builder, op, type, 1, &arg);
}

template<typename T>
static T* createInst(IRBuilder* builder, IROp op, IRType* type, IRInst* arg1, IRInst* arg2)
{
    IRInst* args[] = {arg1, arg2};
    return createInstImpl<T>(builder, op, type, 2, &args[0]);
}

template<typename T>
static T* createInst(
    IRBuilder* builder,
    IROp op,
    IRType* type,
    IRInst* arg1,
    IRInst* arg2,
    IRInst* arg3)
{
    IRInst* args[] = {arg1, arg2, arg3};
    return createInstImpl<T>(builder, op, type, 3, &args[0]);
}

template<typename T>
static T* createInst(
    IRBuilder* builder,
    IROp op,
    IRType* type,
    IRInst* arg1,
    IRInst* arg2,
    IRInst* arg3,
    IRInst* arg4)
{
    IRInst* args[] = {arg1, arg2, arg3, arg4};
    return createInstImpl<T>(builder, op, type, 4, &args[0]);
}

template<typename T>
static T* createInstWithTrailingArgs(
    IRBuilder* builder,
    IROp op,
    IRType* type,
    Int argCount,
    IRInst* const* args)
{
    return createInstImpl<T>(builder, op, type, argCount, args);
}

template<typename T>
static T* createInstWithTrailingArgs(
    IRBuilder* builder,
    IROp op,
    IRType* type,
    Int fixedArgCount,
    IRInst* const* fixedArgs,
    Int varArgCount,
    IRInst* const* varArgs)
{
    return createInstImpl<T>(builder, op, type, fixedArgCount, fixedArgs, varArgCount, varArgs);
}

template<typename T>
static T* createInstWithTrailingArgs(
    IRBuilder* builder,
    IROp op,
    IRType* type,
    IRInst* arg1,
    Int varArgCount,
    IRInst* const* varArgs)
{
    IRInst* fixedArgs[] = {arg1};
    UInt fixedArgCount = sizeof(fixedArgs) / sizeof(fixedArgs[0]);

    return createInstImpl<T>(builder, op, type, fixedArgCount, fixedArgs, varArgCount, varArgs);
}
//

HashCode IRInstKey::_getHashCode()
{
    auto code = Slang::getHashCode(inst->getOp());
    code = combineHash(code, Slang::getHashCode(inst->getFullType()));
    code = combineHash(code, Slang::getHashCode(inst->getOperandCount()));

    auto argCount = inst->getOperandCount();
    auto args = inst->getOperands();
    for (UInt aa = 0; aa < argCount; ++aa)
    {
        code = combineHash(code, Slang::getHashCode(args[aa].get()));
    }
    return code;
}

UnownedStringSlice IRConstant::getStringSlice()
{
    assert(getOp() == kIROp_StringLit || getOp() == kIROp_BlobLit);
    // If the transitory decoration is set, then this is uses the transitoryStringVal for the text
    // storage. This is typically used when we are using a transitory IRInst held on the stack (such
    // that it can be looked up in cached), that just points to a string elsewhere, and NOT the
    // typical normal style, where the string is held after the instruction in memory.
    //
    if (findDecorationImpl(kIROp_TransitoryDecoration))
    {
        return UnownedStringSlice(
            value.transitoryStringVal.chars,
            value.transitoryStringVal.numChars);
    }
    else
    {
        return UnownedStringSlice(value.stringVal.chars, value.stringVal.numChars);
    }
}

bool IRConstant::isFinite() const
{
    SLANG_ASSERT(getOp() == kIROp_FloatLit);

    // Lets check we can analyze as double, at least in principal
    SLANG_COMPILE_TIME_ASSERT(sizeof(IRFloatingPointValue) == sizeof(double));
    // We are in effect going to type pun (yay!), lets make sure they are the same size
    SLANG_COMPILE_TIME_ASSERT(sizeof(IRIntegerValue) == sizeof(IRFloatingPointValue));

    const uint64_t i = uint64_t(value.intVal);
    int e = int(i >> 52) & 0x7ff;
    return (e != 0x7ff);
}

IRConstant::FloatKind IRConstant::getFloatKind() const
{
    SLANG_ASSERT(getOp() == kIROp_FloatLit);

    const uint64_t i = uint64_t(value.intVal);
    int e = int(i >> 52) & 0x7ff;
    if (e == 0x7ff)
    {
        if (i << 12)
        {
            return FloatKind::Nan;
        }
        // Sign bit (top bit) will indicate positive or negative nan
        return value.intVal < 0 ? FloatKind::NegativeInfinity : FloatKind::PositiveInfinity;
    }
    return FloatKind::Finite;
}

bool IRConstant::isValueEqual(IRConstant* rhs)
{
    // If they are literally the same thing..
    if (this == rhs)
    {
        return true;
    }
    // Check the type and they are the same op & same type
    if (getOp() != rhs->getOp())
    {
        return false;
    }

    switch (getOp())
    {
    case kIROp_BoolLit:
    case kIROp_FloatLit:
    case kIROp_IntLit:
        {
            SLANG_COMPILE_TIME_ASSERT(sizeof(IRFloatingPointValue) == sizeof(IRIntegerValue));
            // ... we can just compare as bits
            return value.intVal == rhs->value.intVal;
        }
    case kIROp_PtrLit:
        {
            return value.ptrVal == rhs->value.ptrVal;
        }
    case kIROp_BlobLit:
    case kIROp_StringLit:
        {
            return getStringSlice() == rhs->getStringSlice();
        }
    case kIROp_VoidLit:
        {
            return true;
        }
    default:
        break;
    }

    SLANG_ASSERT(!"Unhandled type");
    return false;
}

/// True if constants are equal
bool IRConstant::equal(IRConstant* rhs)
{
    // TODO(JS): Only equal if pointer types are identical (to match how getHashCode works below)
    return isValueEqual(rhs) && getFullType() == rhs->getFullType();
}

HashCode IRConstant::getHashCode()
{
    auto code = Slang::getHashCode(getOp());
    code = combineHash(code, Slang::getHashCode(getFullType()));

    switch (getOp())
    {
    case kIROp_BoolLit:
    case kIROp_FloatLit:
    case kIROp_IntLit:
        {
            SLANG_COMPILE_TIME_ASSERT(sizeof(IRFloatingPointValue) == sizeof(IRIntegerValue));
            // ... we can just compare as bits
            return combineHash(code, Slang::getHashCode(value.intVal));
        }
    case kIROp_PtrLit:
        {
            return combineHash(code, Slang::getHashCode(value.ptrVal));
        }
    case kIROp_BlobLit:
    case kIROp_StringLit:
        {
            const UnownedStringSlice slice = getStringSlice();
            return combineHash(code, Slang::getHashCode(slice.begin(), slice.getLength()));
        }
    case kIROp_VoidLit:
        {
            return code;
        }
    default:
        {
            SLANG_ASSERT(!"Invalid type");
            return 0;
        }
    }
}

void IRBuilder::setInsertAfter(IRInst* insertAfter)
{
    auto next = insertAfter->getNextInst();
    if (next)
    {
        setInsertBefore(next);
    }
    else
    {
        setInsertInto(insertAfter->parent);
    }
}

IRConstant* IRBuilder::_findOrEmitConstant(IRConstant& keyInst)
{
    // We now know where we want to insert, but there might
    // already be an equivalent instruction in that block.
    //
    // We will check for such an instruction in a slightly hacky
    // way: we will construct a temporary instruction and
    // then use it to look up in a cache of instructions.
    // The 'fake' instruction is passed in as keyInst.

    IRConstantKey key;
    key.inst = &keyInst;

    IRConstant* irValue = nullptr;
    if (m_dedupContext->getConstantMap().tryGetValue(key, irValue))
    {
        // We found a match, so just use that.
        return irValue;
    }

    // Calculate the minimum object size (ie not including the payload of value)
    const size_t prefixSize = SLANG_OFFSET_OF(IRConstant, value);

    switch (keyInst.getOp())
    {
    default:
        SLANG_UNEXPECTED("missing case for IR constant");
        break;

    case kIROp_BoolLit:
    case kIROp_IntLit:
        {
            const size_t instSize = prefixSize + sizeof(IRIntegerValue);
            irValue = static_cast<IRConstant*>(
                _createInst(instSize, keyInst.getFullType(), keyInst.getOp()));
            irValue->value.intVal = keyInst.value.intVal;
            break;
        }
    case kIROp_FloatLit:
        {
            const size_t instSize = prefixSize + sizeof(IRFloatingPointValue);
            irValue = static_cast<IRConstant*>(
                _createInst(instSize, keyInst.getFullType(), keyInst.getOp()));
            irValue->value.floatVal = keyInst.value.floatVal;
            break;
        }
    case kIROp_PtrLit:
        {
            const size_t instSize = prefixSize + sizeof(void*);
            irValue = static_cast<IRConstant*>(
                _createInst(instSize, keyInst.getFullType(), keyInst.getOp()));
            irValue->value.ptrVal = keyInst.value.ptrVal;
            break;
        }
    case kIROp_VoidLit:
        {
            const size_t instSize = prefixSize + sizeof(void*);
            irValue = static_cast<IRConstant*>(
                _createInst(instSize, keyInst.getFullType(), keyInst.getOp()));
            irValue->value.ptrVal = keyInst.value.ptrVal;
            break;
        }
    case kIROp_BlobLit:
    case kIROp_StringLit:
        {
            const UnownedStringSlice slice = keyInst.getStringSlice();

            const size_t sliceSize = slice.getLength();
            const size_t instSize =
                prefixSize + offsetof(IRConstant::StringValue, chars) + sliceSize;

            irValue = static_cast<IRConstant*>(
                _createInst(instSize, keyInst.getFullType(), keyInst.getOp()));

            IRConstant::StringValue& dstString = irValue->value.stringVal;

            dstString.numChars = uint32_t(sliceSize);
            // Turn into pointer to avoid warning of array overrun
            char* dstChars = dstString.chars;
            // Copy the chars
            memcpy(dstChars, slice.begin(), sliceSize);

            break;
        }
    }

    key.inst = irValue;
    m_dedupContext->getConstantMap().add(key, irValue);

    addHoistableInst(this, irValue);

    return irValue;
}

//

IRInst* IRBuilder::getBoolValue(bool inValue)
{
    IRConstant keyInst;
    memset(&keyInst, 0, sizeof(keyInst));
    keyInst.m_op = kIROp_BoolLit;
    keyInst.typeUse.usedValue = getBoolType();
    keyInst.value.intVal = IRIntegerValue(inValue);
    return _findOrEmitConstant(keyInst);
}

IRInst* IRBuilder::getIntValue(IRIntegerValue value)
{
    return getIntValue(getIntType(), value);
}

IRInst* IRBuilder::getIntValue(IRType* type, IRIntegerValue inValue)
{
    IRConstant keyInst;
    memset(&keyInst, 0, sizeof(keyInst));
    keyInst.m_op = kIROp_IntLit;
    keyInst.typeUse.usedValue = type;
    // Truncate the input value based on `type`.
    switch (type->getOp())
    {
    case kIROp_Int8Type:
        keyInst.value.intVal = static_cast<int8_t>(inValue);
        break;
    case kIROp_Int16Type:
        keyInst.value.intVal = static_cast<int16_t>(inValue);
        break;
    case kIROp_IntType:
        keyInst.value.intVal = static_cast<int32_t>(inValue);
        break;
    case kIROp_UInt8Type:
        keyInst.value.intVal = static_cast<uint8_t>(inValue);
        break;
    case kIROp_UInt16Type:
        keyInst.value.intVal = static_cast<uint16_t>(inValue);
        break;
    case kIROp_BoolType:
        keyInst.m_op = kIROp_BoolLit;
        keyInst.value.intVal = ((inValue != 0) ? 1 : 0);
        break;
    case kIROp_UIntType:
        keyInst.value.intVal = static_cast<uint32_t>(inValue);
        break;
    default:
        keyInst.value.intVal = inValue;
        break;
    }
    return _findOrEmitConstant(keyInst);
}

IRInst* IRBuilder::getFloatValue(IRType* type, IRFloatingPointValue inValue)
{
    IRConstant keyInst;
    memset(&keyInst, 0, sizeof(keyInst));
    keyInst.m_op = kIROp_FloatLit;
    keyInst.typeUse.usedValue = type;
    // Truncate the input value based on `type`.
    switch (type->getOp())
    {
    case kIROp_FloatType:
        keyInst.value.floatVal = static_cast<float>(inValue);
        break;
    case kIROp_HalfType:
        keyInst.value.floatVal = HalfToFloat(FloatToHalf((float)inValue));
        break;
    default:
        keyInst.value.floatVal = inValue;
        break;
    }

    return _findOrEmitConstant(keyInst);
}

IRStringLit* IRBuilder::getStringValue(const UnownedStringSlice& inSlice)
{
    IRConstant keyInst;
    memset(&keyInst, 0, sizeof(keyInst));

    // Mark that this is on the stack...
    IRDecoration stackDecoration;
    memset(&stackDecoration, 0, sizeof(stackDecoration));
    stackDecoration.m_op = kIROp_TransitoryDecoration;
    stackDecoration.insertAtEnd(&keyInst);

    keyInst.m_op = kIROp_StringLit;
    keyInst.typeUse.usedValue = getStringType();

    IRConstant::StringSliceValue& dstSlice = keyInst.value.transitoryStringVal;
    dstSlice.chars = const_cast<char*>(inSlice.begin());
    dstSlice.numChars = uint32_t(inSlice.getLength());

    return static_cast<IRStringLit*>(_findOrEmitConstant(keyInst));
}

IRBlobLit* IRBuilder::getBlobValue(ISlangBlob* blob)
{
    IRConstant keyInst;
    memset(&keyInst, 0, sizeof(keyInst));

    char* buffer = (char*)(getModule()->getMemoryArena().allocate(blob->getBufferSize()));
    if (!buffer)
    {
        return nullptr;
    }
    memcpy(buffer, blob->getBufferPointer(), blob->getBufferSize());

    UnownedStringSlice inSlice(buffer, blob->getBufferSize());

    // Mark that this is on the stack...
    IRDecoration stackDecoration;
    memset(&stackDecoration, 0, sizeof(stackDecoration));
    stackDecoration.m_op = kIROp_TransitoryDecoration;
    stackDecoration.insertAtEnd(&keyInst);

    keyInst.m_op = kIROp_BlobLit;
    keyInst.typeUse.usedValue = nullptr; // not used

    IRConstant::StringSliceValue& dstSlice = keyInst.value.transitoryStringVal;
    dstSlice.chars = const_cast<char*>(inSlice.begin());
    dstSlice.numChars = uint32_t(inSlice.getLength());

    return static_cast<IRBlobLit*>(_findOrEmitConstant(keyInst));
}

IRPtrLit* IRBuilder::getPtrValue(IRType* type, void* data)
{
    IRConstant keyInst;
    memset(&keyInst, 0, sizeof(keyInst));
    keyInst.m_op = kIROp_PtrLit;
    keyInst.typeUse.usedValue = type;
    keyInst.value.ptrVal = data;
    return (IRPtrLit*)_findOrEmitConstant(keyInst);
}

IRPtrLit* IRBuilder::getNullPtrValue(IRType* type)
{
    IRConstant keyInst;
    memset(&keyInst, 0, sizeof(keyInst));
    keyInst.m_op = kIROp_PtrLit;
    keyInst.typeUse.usedValue = type;
    keyInst.value.ptrVal = nullptr;
    return (IRPtrLit*)_findOrEmitConstant(keyInst);
}

IRVoidLit* IRBuilder::getVoidValue()
{
    IRType* type = getVoidType();

    IRConstant keyInst;
    memset(&keyInst, 0, sizeof(keyInst));
    keyInst.m_op = kIROp_VoidLit;
    keyInst.typeUse.usedValue = type;
    keyInst.value.intVal = 0;
    return (IRVoidLit*)_findOrEmitConstant(keyInst);
}

IRInst* IRBuilder::getCapabilityValue(CapabilitySet const& caps)
{
    IRType* capabilityAtomType = getIntType();
    IRType* capabilitySetType = getCapabilitySetType();

    // Not: Our `CapabilitySet` representation consists of a list
    // of `CapabilityAtom`s, and by default the list is stored
    // "expanded" so that it includes atoms that are transitively
    // implied by one another.
    //
    // For representation in the IR, it is preferable to include
    // as few atoms as possible, so that we don't store anything
    // redundant in, e.g., serialized modules.
    //
    // We thus requqest a list of "compacted" atoms which should
    // be a minimal list of atoms such that they will produce
    // the same `CapabilitySet` when expanded.

    auto compactedAtoms = caps.getAtomSets();
    List<IRInst*> conjunctions;
    for (auto& atomConjunctionSet : compactedAtoms)
    {
        List<IRInst*> args;
        for (auto atom : atomConjunctionSet)
            args.add(getIntValue(capabilityAtomType, Int(atom)));
        auto conjunctionInst = createIntrinsicInst(
            capabilitySetType,
            kIROp_CapabilityConjunction,
            args.getCount(),
            args.getBuffer());
        conjunctions.add(conjunctionInst);
    }
    if (conjunctions.getCount() == 1)
        return conjunctions[0];
    return createIntrinsicInst(
        capabilitySetType,
        kIROp_CapabilityDisjunction,
        conjunctions.getCount(),
        conjunctions.getBuffer());
}

static void canonicalizeInstOperands(IRBuilder& builder, IROp op, ArrayView<IRInst*> operands)
{
    // For Array types, we always want to make sure its element count
    // has an int32_t type. We will convert all other int types to int32_t
    // to avoid things like float[8] and float[8U] being distinct types.
    if (op == kIROp_ArrayType)
    {
        if (operands.getCount() < 2)
            return;
        IRInst* elementCount = operands[1];
        if (auto intLit = as<IRIntLit>(elementCount))
        {
            if (intLit->getDataType()->getOp() != kIROp_IntType)
            {
                IRInst* newElementCount =
                    builder.getIntValue(builder.getIntType(), intLit->getValue());
                operands[1] = newElementCount;
            }
        }
    }
}

static void addGlobalValue(IRBuilder* builder, IRInst* value)
{
    // If the value is already in the parent, keep it as-is.
    // Because when the inst is Hoistable, the parent can have
    // only one instance of the inst. The order among
    // siblings should remain because the later siblings may
    // have dependency to the earlier siblings.
    //
    if (value->parent)
    {
        SLANG_ASSERT(getIROpInfo(value->getOp()).isHoistable());
        return;
    }

    // Try to find a suitable parent for the
    // global value we are emitting.
    //
    // We will start out search at the current
    // parent instruction for the builder, and
    // possibly work our way up.
    //
    auto defaultInsertLoc = builder->getInsertLoc();
    auto defaultParent = defaultInsertLoc.getParent();
    auto parent = defaultParent;
    while (parent)
    {
        // Inserting into the top level of a module?
        // That is fine, and we can stop searching.
        if (as<IRModuleInst>(parent))
            break;

        // Inserting into a basic block inside of
        // a generic? That is okay too.
        if (auto block = as<IRBlock>(parent))
        {
            if (as<IRGeneric>(block->parent))
                break;
        }

        // Otherwise, move up the chain.
        parent = parent->parent;
    }

    // If we somehow ran out of parents (possibly
    // because an instruction wasn't linked into
    // the full hierarchy yet), then we will
    // fall back to inserting into the overall module.
    if (!parent)
    {
        parent = builder->getModule()->getModuleInst();
    }

    // If it turns out that we are inserting into the
    // current "insert into" parent for the builder, then
    // we need to respect its "insert before" setting
    // as well.
    if (parent == defaultParent)
    {
        value->insertAt(defaultInsertLoc);
    }
    else
    {
        value->insertAtEnd(parent);
    }
}

IRInst* IRBuilder::_findOrEmitHoistableInst(
    IRType* type,
    IROp op,
    Int fixedArgCount,
    IRInst* const* fixedArgs,
    Int varArgListCount,
    Int const* listArgCounts,
    IRInst* const* const* listArgs)
{
    UInt operandCount = fixedArgCount;
    for (Int ii = 0; ii < varArgListCount; ++ii)
    {
        operandCount += listArgCounts[ii];
    }

    ShortList<IRInst*, 8> canonicalizedOperands;
    canonicalizedOperands.setCount(fixedArgCount);
    for (Index i = 0; i < fixedArgCount; i++)
        canonicalizedOperands[i] = fixedArgs[i];

    canonicalizeInstOperands(*this, op, canonicalizedOperands.getArrayView().arrayView);

    auto& memoryArena = getModule()->getMemoryArena();
    void* cursor = memoryArena.getCursor();

    // We are going to create a 'dummy' instruction on the memoryArena
    // which can be used as a key for lookup, so see if we
    // already have an equivalent instruction available to use.
    size_t keySize = sizeof(IRInst) + operandCount * sizeof(IRUse);
    IRInst* inst = (IRInst*)memoryArena.allocateAndZero(keySize);

    void* endCursor = memoryArena.getCursor();
    // Mark as 'unused' cos it is unused on release builds.
    SLANG_UNUSED(endCursor);

    new (inst) IRInst();
#if SLANG_ENABLE_IR_BREAK_ALLOC
    inst->_debugUID = _debugGetAndIncreaseInstCounter();
#endif
    inst->m_op = op;
    inst->typeUse.usedValue = type;
    inst->operandCount = (uint32_t)operandCount;

    // Don't link up as we may free (if we already have this key)
    {
        IRUse* operand = inst->getOperands();
        for (Int ii = 0; ii < fixedArgCount; ++ii)
        {
            auto arg = canonicalizedOperands[ii];
            m_dedupContext->getInstReplacementMap().tryGetValue(arg, arg);
            operand->usedValue = arg;
            operand++;
        }
        for (Int ii = 0; ii < varArgListCount; ++ii)
        {
            UInt listOperandCount = listArgCounts[ii];
            for (UInt jj = 0; jj < listOperandCount; ++jj)
            {
                auto arg = listArgs[ii][jj];
                m_dedupContext->getInstReplacementMap().tryGetValue(arg, arg);
                operand->usedValue = arg;
                operand++;
            }
        }
    }

    // Find or add the key/inst
    {
        IRInstKey key = {inst};

        IRInst** found = m_dedupContext->getGlobalValueNumberingMap().tryGetValueOrAdd(key, inst);
        SLANG_ASSERT(endCursor == memoryArena.getCursor());
        // If it's found, just return, and throw away the instruction
        if (found)
        {
            memoryArena.rewindToCursor(cursor);

            // If the found inst is defined in the same parent as current insert location but
            // is located after the insert location, we need to move it to the insert location.
            auto foundInst = *found;
            if (foundInst->getParent() && foundInst->getParent() == getInsertLoc().getParent() &&
                getInsertLoc().getMode() == IRInsertLoc::Mode::Before)
            {
                auto insertLoc = getInsertLoc().getInst();
                bool isAfter = false;
                for (auto cur = insertLoc->next; cur; cur = cur->next)
                {
                    if (cur == foundInst)
                    {
                        isAfter = true;
                        break;
                    }
                }
                if (isAfter)
                    foundInst->insertBefore(insertLoc);
            }
            return *found;
        }
    }

    // Make the lookup 'inst' instruction into 'proper' instruction. Equivalent to
    // IRInst* inst = createInstImpl<IRInst>(builder, op, type, 0, nullptr, operandListCount,
    // listOperandCounts, listOperands);
    {
        if (type)
        {
            inst->typeUse.usedValue = nullptr;
            inst->typeUse.init(inst, type);
        }

        _maybeSetSourceLoc(inst);

        IRUse* const operands = inst->getOperands();
        for (UInt i = 0; i < operandCount; ++i)
        {
            IRUse& operand = operands[i];
            auto value = operand.usedValue;

            operand.usedValue = nullptr;
            operand.init(inst, value);
        }
    }

    // When an hoistable inst is already a child, skip adding it.
    if (inst->parent == nullptr)
    {
        // In order to de-duplicate them, Witness-table is marked as Hoistable.
        // But it is not exactly a hoistable type and it should be added as a global value.
        if (inst->getOp() == kIROp_WitnessTable)
            addGlobalValue(this, inst);
        else
            addHoistableInst(this, inst);
    }

    return inst;
}

IRType* IRBuilder::getType(IROp op, UInt operandCount, IRInst* const* operands)
{
    return (IRType*)createIntrinsicInst(nullptr, op, operandCount, operands);
}

IRType* IRBuilder::getType(IROp op)
{
    return getType(op, 0, nullptr);
}

IRType* IRBuilder::getType(IROp op, IRInst* operand0)
{
    return getType(op, 1, &operand0);
}

IRBasicType* IRBuilder::getBasicType(BaseType baseType)
{
    return (IRBasicType*)getType(IROp((UInt)kIROp_FirstBasicType + (UInt)baseType));
}

IRBasicType* IRBuilder::getVoidType()
{
    return (IRVoidType*)getType(kIROp_VoidType);
}

IRBasicType* IRBuilder::getBoolType()
{
    return (IRBoolType*)getType(kIROp_BoolType);
}

IRBasicType* IRBuilder::getIntType()
{
    return (IRBasicType*)getType(kIROp_IntType);
}

IRBasicType* IRBuilder::getInt64Type()
{
    return (IRBasicType*)getType(kIROp_Int64Type);
}

IRBasicType* IRBuilder::getUIntType()
{
    return (IRBasicType*)getType(kIROp_UIntType);
}

IRBasicType* IRBuilder::getUInt64Type()
{
    return (IRBasicType*)getType(kIROp_UInt64Type);
}

IRBasicType* IRBuilder::getUInt16Type()
{
    return (IRBasicType*)getType(kIROp_UInt16Type);
}

IRBasicType* IRBuilder::getUInt8Type()
{
    return (IRBasicType*)getType(kIROp_UInt8Type);
}

IRBasicType* IRBuilder::getFloatType()
{
    return (IRBasicType*)getType(kIROp_FloatType);
}

IRBasicType* IRBuilder::getCharType()
{
    return (IRBasicType*)getType(kIROp_CharType);
}

IRStringType* IRBuilder::getStringType()
{
    return (IRStringType*)getType(kIROp_StringType);
}

IRNativeStringType* IRBuilder::getNativeStringType()
{
    return (IRNativeStringType*)getType(kIROp_NativeStringType);
}

IRNativePtrType* IRBuilder::getNativePtrType(IRType* valueType)
{
    return (IRNativePtrType*)getType(kIROp_NativePtrType, (IRInst*)valueType);
}


IRType* IRBuilder::getCapabilitySetType()
{
    return getType(kIROp_CapabilitySetType);
}

IRDynamicType* IRBuilder::getDynamicType()
{
    return (IRDynamicType*)getType(kIROp_DynamicType);
}

IRTargetTupleType* IRBuilder::getTargetTupleType(UInt count, IRType* const* types)
{
    return (IRTargetTupleType*)getType(kIROp_TargetTupleType, count, (IRInst* const*)types);
}

IRAssociatedType* IRBuilder::getAssociatedType(ArrayView<IRInterfaceType*> constraintTypes)
{
    return (IRAssociatedType*)getType(
        kIROp_AssociatedType,
        constraintTypes.getCount(),
        (IRInst**)constraintTypes.getBuffer());
}

IRThisType* IRBuilder::getThisType(IRType* interfaceType)
{
    return (IRThisType*)getType(kIROp_ThisType, interfaceType);
}

IRRawPointerType* IRBuilder::getRawPointerType()
{
    return (IRRawPointerType*)getType(kIROp_RawPointerType);
}

IRRTTIPointerType* IRBuilder::getRTTIPointerType(IRInst* rttiPtr)
{
    return (IRRTTIPointerType*)getType(kIROp_RTTIPointerType, rttiPtr);
}

IRRTTIType* IRBuilder::getRTTIType()
{
    return (IRRTTIType*)getType(kIROp_RTTIType);
}

IRRTTIHandleType* IRBuilder::getRTTIHandleType()
{
    return (IRRTTIHandleType*)getType(kIROp_RTTIHandleType);
}

IRAnyValueType* IRBuilder::getAnyValueType(IRIntegerValue size)
{
    return (IRAnyValueType*)getType(kIROp_AnyValueType, getIntValue(getUIntType(), size));
}

IRAnyValueType* IRBuilder::getAnyValueType(IRInst* size)
{
    return (IRAnyValueType*)getType(kIROp_AnyValueType, size);
}

IRTupleType* IRBuilder::getTupleType(UInt count, IRType* const* types)
{
    return (IRTupleType*)getType(kIROp_TupleType, count, (IRInst* const*)types);
}

IRTupleType* IRBuilder::getTupleType(IRType* type0, IRType* type1)
{
    IRType* operands[] = {type0, type1};
    return getTupleType(2, operands);
}

IRTupleType* IRBuilder::getTupleType(IRType* type0, IRType* type1, IRType* type2)
{
    IRType* operands[] = {type0, type1, type2};
    return getTupleType(3, operands);
}

IRTupleType* IRBuilder::getTupleType(IRType* type0, IRType* type1, IRType* type2, IRType* type3)
{
    IRType* operands[] = {type0, type1, type2, type3};
    return getTupleType(SLANG_COUNT_OF(operands), operands);
}

IRTypePack* IRBuilder::getTypePack(UInt count, IRType* const* types)
{
    return (IRTypePack*)getType(kIROp_TypePack, count, (IRInst* const*)types);
}

IRExpandType* IRBuilder::getExpandTypeOrVal(
    IRType* type,
    IRInst* pattern,
    ArrayView<IRInst*> capture)
{
    ShortList<IRInst*> args;
    args.add(pattern);
    args.addRange(capture);
    return (IRExpandType*)emitIntrinsicInst(
        type,
        kIROp_ExpandTypeOrVal,
        args.getCount(),
        args.getArrayView().getBuffer());
}

IRResultType* IRBuilder::getResultType(IRType* valueType, IRType* errorType)
{
    IRInst* operands[] = {valueType, errorType};
    return (IRResultType*)getType(kIROp_ResultType, 2, operands);
}

IROptionalType* IRBuilder::getOptionalType(IRType* valueType)
{
    return (IROptionalType*)getType(kIROp_OptionalType, valueType);
}

IRBasicBlockType* IRBuilder::getBasicBlockType()
{
    return (IRBasicBlockType*)getType(kIROp_BasicBlockType);
}

IRTypeKind* IRBuilder::getTypeKind()
{
    return (IRTypeKind*)getType(kIROp_TypeKind);
}

IRGenericKind* IRBuilder::getGenericKind()
{
    return (IRGenericKind*)getType(kIROp_GenericKind);
}

IRPtrType* IRBuilder::getPtrType(IRType* valueType)
{
    return (IRPtrType*)getPtrType(kIROp_PtrType, valueType);
}

IROutType* IRBuilder::getOutType(IRType* valueType)
{
    return (IROutType*)getPtrType(kIROp_OutType, valueType);
}

IRInOutType* IRBuilder::getInOutType(IRType* valueType)
{
    return (IRInOutType*)getPtrType(kIROp_InOutType, valueType);
}

IRRefType* IRBuilder::getRefType(IRType* valueType, AddressSpace addrSpace)
{
    return (IRRefType*)getPtrType(kIROp_RefType, valueType, addrSpace);
}

IRConstRefType* IRBuilder::getConstRefType(IRType* valueType)
{
    return (IRConstRefType*)getPtrType(kIROp_ConstRefType, valueType);
}

IRSPIRVLiteralType* IRBuilder::getSPIRVLiteralType(IRType* type)
{
    IRInst* operands[] = {type};
    return (IRSPIRVLiteralType*)getType(kIROp_SPIRVLiteralType, 1, operands);
}

IRPtrTypeBase* IRBuilder::getPtrType(IROp op, IRType* valueType)
{
    IRInst* operands[] = {valueType};
    return (IRPtrTypeBase*)getType(op, 1, operands);
}

IRPtrTypeBase* IRBuilder::getPtrTypeWithAddressSpace(
    IRType* valueType,
    IRPtrTypeBase* ptrWithAddrSpace)
{
    if (ptrWithAddrSpace->hasAddressSpace())
        return (IRPtrTypeBase*)
            getPtrType(ptrWithAddrSpace->getOp(), valueType, ptrWithAddrSpace->getAddressSpace());
    return (IRPtrTypeBase*)getPtrType(ptrWithAddrSpace->getOp(), valueType);
}

IRPtrType* IRBuilder::getPtrType(IROp op, IRType* valueType, AddressSpace addressSpace)
{
    return (IRPtrType*)getPtrType(
        op,
        valueType,
        getIntValue(getUInt64Type(), static_cast<IRIntegerValue>(addressSpace)));
}

IRPtrType* IRBuilder::getPtrType(IROp op, IRType* valueType, IRInst* addressSpace)
{
    IRInst* operands[] = {valueType, addressSpace};
    return (IRPtrType*)getType(op, addressSpace ? 2 : 1, operands);
}

IRTextureTypeBase* IRBuilder::getTextureType(
    IRType* elementType,
    IRInst* shape,
    IRInst* isArray,
    IRInst* isMS,
    IRInst* sampleCount,
    IRInst* access,
    IRInst* isShadow,
    IRInst* isCombined,
    IRInst* format)
{
    IRInst* args[] = {
        (IRInst*)elementType,
        shape,
        isArray,
        isMS,
        sampleCount,
        access,
        isShadow,
        isCombined,
        format};
    return as<IRTextureTypeBase>(emitIntrinsicInst(
        getTypeKind(),
        kIROp_TextureType,
        (UInt)(sizeof(args) / sizeof(IRInst*)),
        args));
}

IRComPtrType* IRBuilder::getComPtrType(IRType* valueType)
{
    return (IRComPtrType*)getType(kIROp_ComPtrType, valueType);
}

IRArrayTypeBase* IRBuilder::getArrayTypeBase(
    IROp op,
    IRType* elementType,
    IRInst* elementCount,
    IRInst* stride)
{
    if (op == kIROp_ArrayType)
    {
        IRInst* operands[] = {elementType, elementCount, stride};
        return (IRArrayTypeBase*)getType(op, stride ? 3 : 2, operands);
    }
    else
    {
        IRInst* operands[] = {elementType, stride};
        return (IRArrayTypeBase*)getType(op, stride ? 2 : 1, operands);
    }
}

IRArrayType* IRBuilder::getArrayType(IRType* elementType, IRInst* elementCount)
{
    IRInst* operands[] = {elementType, elementCount};
    return (IRArrayType*)getType(kIROp_ArrayType, sizeof(operands) / sizeof(operands[0]), operands);
}

IRUnsizedArrayType* IRBuilder::getUnsizedArrayType(IRType* elementType)
{
    IRInst* operands[] = {elementType};
    return (IRUnsizedArrayType*)
        getType(kIROp_UnsizedArrayType, sizeof(operands) / sizeof(operands[0]), operands);
}

IRArrayType* IRBuilder::getArrayType(IRType* elementType, IRInst* elementCount, IRInst* stride)
{
    IRInst* operands[] = {elementType, elementCount, stride};
    return (IRArrayType*)getType(kIROp_ArrayType, sizeof(operands) / sizeof(operands[0]), operands);
}

IRUnsizedArrayType* IRBuilder::getUnsizedArrayType(IRType* elementType, IRInst* stride)
{
    IRInst* operands[] = {elementType, stride};
    return (IRUnsizedArrayType*)
        getType(kIROp_UnsizedArrayType, sizeof(operands) / sizeof(operands[0]), operands);
}

IRVectorType* IRBuilder::getVectorType(IRType* elementType, IRInst* elementCount)
{
    IRInst* operands[] = {elementType, elementCount};
    return (
        IRVectorType*)getType(kIROp_VectorType, sizeof(operands) / sizeof(operands[0]), operands);
}

IRVectorType* IRBuilder::getVectorType(IRType* elementType, IRIntegerValue elementCount)
{
    return getVectorType(elementType, getIntValue(getIntType(), elementCount));
}

IRCoopVectorType* IRBuilder::getCoopVectorType(IRType* elementType, IRInst* elementCount)
{
    IRInst* operands[] = {elementType, elementCount};
    return (IRCoopVectorType*)
        getType(kIROp_CoopVectorType, sizeof(operands) / sizeof(operands[0]), operands);
}

IRMatrixType* IRBuilder::getMatrixType(
    IRType* elementType,
    IRInst* rowCount,
    IRInst* columnCount,
    IRInst* layout)
{
    IRInst* operands[] = {elementType, rowCount, columnCount, layout};
    return (
        IRMatrixType*)getType(kIROp_MatrixType, sizeof(operands) / sizeof(operands[0]), operands);
}

IRArrayListType* IRBuilder::getArrayListType(IRType* elementType)
{
    return (IRArrayListType*)getType(kIROp_ArrayListType, 1, (IRInst**)&elementType);
}

IRTensorViewType* IRBuilder::getTensorViewType(IRType* elementType)
{
    return (IRTensorViewType*)getType(kIROp_TensorViewType, 1, (IRInst**)&elementType);
}

IRTorchTensorType* IRBuilder::getTorchTensorType(IRType* elementType)
{
    return (IRTorchTensorType*)getType(kIROp_TorchTensorType, 1, (IRInst**)&elementType);
}

IRDifferentialPairType* IRBuilder::getDifferentialPairType(IRType* valueType, IRInst* witnessTable)
{
    IRInst* operands[] = {valueType, witnessTable};
    return (IRDifferentialPairType*)
        getType(kIROp_DifferentialPairType, sizeof(operands) / sizeof(operands[0]), operands);
}

IRDifferentialPtrPairType* IRBuilder::getDifferentialPtrPairType(
    IRType* valueType,
    IRInst* witnessTable)
{
    IRInst* operands[] = {valueType, witnessTable};
    return (IRDifferentialPtrPairType*)
        getType(kIROp_DifferentialPtrPairType, sizeof(operands) / sizeof(operands[0]), operands);
}

IRDifferentialPairUserCodeType* IRBuilder::getDifferentialPairUserCodeType(
    IRType* valueType,
    IRInst* witnessTable)
{
    IRInst* operands[] = {valueType, witnessTable};
    return (IRDifferentialPairUserCodeType*)getType(
        kIROp_DifferentialPairUserCodeType,
        sizeof(operands) / sizeof(operands[0]),
        operands);
}

IRBackwardDiffIntermediateContextType* IRBuilder::getBackwardDiffIntermediateContextType(
    IRInst* func)
{
    if (!func)
        func = getVoidValue();
    return (IRBackwardDiffIntermediateContextType*)
        getType(kIROp_BackwardDiffIntermediateContextType, 1, &func);
}

IRFuncType* IRBuilder::getFuncType(UInt paramCount, IRType* const* paramTypes, IRType* resultType)
{
    return (IRFuncType*)createIntrinsicInst(
        nullptr,
        kIROp_FuncType,
        resultType,
        paramCount,
        (IRInst* const*)paramTypes);
}

IRFuncType* IRBuilder::getFuncType(
    UInt paramCount,
    IRType* const* paramTypes,
    IRType* resultType,
    IRAttr* attribute)
{
    UInt counts[3] = {1, paramCount, 1};
    IRInst** lists[3] = {(IRInst**)&resultType, (IRInst**)paramTypes, (IRInst**)&attribute};
    return (IRFuncType*)createIntrinsicInst(nullptr, kIROp_FuncType, 3, counts, lists);
}

IRWitnessTableType* IRBuilder::getWitnessTableType(IRType* baseType)
{
    return (IRWitnessTableType*)
        createIntrinsicInst(nullptr, kIROp_WitnessTableType, 1, (IRInst* const*)&baseType);
}

IRWitnessTableIDType* IRBuilder::getWitnessTableIDType(IRType* baseType)
{
    return (IRWitnessTableIDType*)
        createIntrinsicInst(nullptr, kIROp_WitnessTableIDType, 1, (IRInst* const*)&baseType);
}

IRConstantBufferType* IRBuilder::getConstantBufferType(IRType* elementType, IRType* layoutType)
{
    IRInst* operands[] = {elementType, layoutType};
    return (IRConstantBufferType*)getType(kIROp_ConstantBufferType, 2, operands);
}

IRGLSLOutputParameterGroupType* IRBuilder::getGLSLOutputParameterGroupType(IRType* elementType)
{
    IRInst* operands[] = {elementType};
    return (
        IRGLSLOutputParameterGroupType*)getType(kIROp_GLSLOutputParameterGroupType, 1, operands);
}

IRConstExprRate* IRBuilder::getConstExprRate()
{
    return (IRConstExprRate*)getType(kIROp_ConstExprRate);
}

IRGroupSharedRate* IRBuilder::getGroupSharedRate()
{
    return (IRGroupSharedRate*)getType(kIROp_GroupSharedRate);
}
IRActualGlobalRate* IRBuilder::getActualGlobalRate()
{
    return (IRActualGlobalRate*)getType(kIROp_ActualGlobalRate);
}

IRRateQualifiedType* IRBuilder::getRateQualifiedType(IRRate* rate, IRType* dataType)
{
    IRInst* operands[] = {rate, dataType};
    return (IRRateQualifiedType*)
        getType(kIROp_RateQualifiedType, sizeof(operands) / sizeof(operands[0]), operands);
}

IRType* IRBuilder::getBindExistentialsType(
    IRInst* baseType,
    UInt slotArgCount,
    IRInst* const* slotArgs)
{
    if (slotArgCount == 0)
        return (IRType*)baseType;

    // If we are trying to bind an interface type, then
    // we will go ahead and simplify the instruction
    // away impmediately.
    //
    if (as<IRInterfaceType>(baseType))
    {
        if (slotArgCount >= 2)
        {
            // We are being asked to emit `BindExistentials(someInterface, someConcreteType, ...)`
            // so we just want to return `ExistentialBox<someConcreteType>`.
            //
            auto concreteType = (IRType*)slotArgs[0];
            auto witnessTable = slotArgs[1];
            auto ptrType = getBoundInterfaceType((IRType*)baseType, concreteType, witnessTable);
            return ptrType;
        }
    }

    return (IRType*)createIntrinsicInst(
        getTypeKind(),
        kIROp_BindExistentialsType,
        baseType,
        slotArgCount,
        (IRInst* const*)slotArgs);
}

IRType* IRBuilder::getBindExistentialsType(
    IRInst* baseType,
    UInt slotArgCount,
    IRUse const* slotArgUses)
{
    if (slotArgCount == 0)
        return (IRType*)baseType;

    List<IRInst*> slotArgs;
    for (UInt ii = 0; ii < slotArgCount; ++ii)
    {
        slotArgs.add(slotArgUses[ii].get());
    }
    return getBindExistentialsType(baseType, slotArgCount, slotArgs.getBuffer());
}

IRType* IRBuilder::getBoundInterfaceType(
    IRType* interfaceType,
    IRType* concreteType,
    IRInst* witnessTable)
{
    // Don't wrap an existential box if concreteType is __Dynamic.
    if (as<IRDynamicType>(concreteType))
        return interfaceType;

    IRInst* operands[] = {interfaceType, concreteType, witnessTable};
    return getType(kIROp_BoundInterfaceType, SLANG_COUNT_OF(operands), operands);
}

IRType* IRBuilder::getPseudoPtrType(IRType* concreteType)
{
    IRInst* operands[] = {concreteType};
    return getType(kIROp_PseudoPtrType, SLANG_COUNT_OF(operands), operands);
}

IRType* IRBuilder::getConjunctionType(UInt typeCount, IRType* const* types)
{
    return getType(kIROp_ConjunctionType, typeCount, (IRInst* const*)types);
}

IRType* IRBuilder::getAttributedType(
    IRType* baseType,
    UInt attributeCount,
    IRAttr* const* attributes)
{
    List<IRInst*> operands;
    operands.add(baseType);
    for (UInt i = 0; i < attributeCount; ++i)
        operands.add(attributes[i]);
    return getType(kIROp_AttributedType, operands.getCount(), operands.getBuffer());
}


void IRBuilder::setDataType(IRInst* inst, IRType* dataType)
{
    if (auto oldRateQualifiedType = as<IRRateQualifiedType>(inst->getFullType()))
    {
        // Construct a new rate-qualified type using the same rate.

        auto newRateQualifiedType = getRateQualifiedType(oldRateQualifiedType->getRate(), dataType);

        inst->setFullType(newRateQualifiedType);
    }
    else
    {
        // No rate? Just clobber the data type.
        inst->setFullType(dataType);
    }
}

IRInst* IRBuilder::emitGetCurrentStage()
{
    return emitIntrinsicInst(getIntType(), kIROp_GetCurrentStage, 0, nullptr);
}

IRInst* IRBuilder::emitGetValueFromBoundInterface(IRType* type, IRInst* boundInterfaceValue)
{
    auto inst =
        createInst<IRInst>(this, kIROp_GetValueFromBoundInterface, type, 1, &boundInterfaceValue);
    addInst(inst);
    return inst;
}


IRUndefined* IRBuilder::emitUndefined(IRType* type)
{
    auto inst = createInst<IRUndefined>(this, kIROp_undefined, type);

    addInst(inst);

    return inst;
}

IRInst* IRBuilder::emitByteAddressBufferStore(
    IRInst* byteAddressBuffer,
    IRInst* offset,
    IRInst* value)
{
    IRInst* args[] = {byteAddressBuffer, offset, getIntValue(getUIntType(), 0), value};
    return emitIntrinsicInst(getVoidType(), kIROp_ByteAddressBufferStore, 4, args);
}

IRInst* IRBuilder::emitByteAddressBufferStore(
    IRInst* byteAddressBuffer,
    IRInst* offset,
    IRInst* alignment,
    IRInst* value)
{
    IRInst* args[] = {byteAddressBuffer, offset, alignment, value};
    return emitIntrinsicInst(getVoidType(), kIROp_ByteAddressBufferStore, 4, args);
}

IRInst* IRBuilder::emitReinterpret(IRInst* type, IRInst* value)
{
    return emitIntrinsicInst((IRType*)type, kIROp_Reinterpret, 1, &value);
}
IRInst* IRBuilder::emitInOutImplicitCast(IRInst* type, IRInst* value)
{
    return emitIntrinsicInst((IRType*)type, kIROp_InOutImplicitCast, 1, &value);
}
IRInst* IRBuilder::emitOutImplicitCast(IRInst* type, IRInst* value)
{
    return emitIntrinsicInst((IRType*)type, kIROp_OutImplicitCast, 1, &value);
}
IRInst* IRBuilder::emitDebugSource(UnownedStringSlice fileName, UnownedStringSlice source)
{
    IRInst* args[] = {getStringValue(fileName), getStringValue(source)};
    return emitIntrinsicInst(getVoidType(), kIROp_DebugSource, 2, args);
}
IRInst* IRBuilder::emitDebugLine(
    IRInst* source,
    IRIntegerValue lineStart,
    IRIntegerValue lineEnd,
    IRIntegerValue colStart,
    IRIntegerValue colEnd)
{
    IRInst* args[] = {
        source,
        getIntValue(getUIntType(), lineStart),
        getIntValue(getUIntType(), lineEnd),
        getIntValue(getUIntType(), colStart),
        getIntValue(getUIntType(), colEnd)};
    return emitIntrinsicInst(getVoidType(), kIROp_DebugLine, 5, args);
}
IRInst* IRBuilder::emitDebugVar(
    IRType* type,
    IRInst* source,
    IRInst* line,
    IRInst* col,
    IRInst* argIndex)
{
    if (argIndex)
    {
        IRInst* args[] = {source, line, col, argIndex};
        return emitIntrinsicInst(getPtrType(type), kIROp_DebugVar, 4, args);
    }
    else
    {
        IRInst* args[] = {source, line, col};
        return emitIntrinsicInst(getPtrType(type), kIROp_DebugVar, 3, args);
    }
}

IRInst* IRBuilder::emitDebugValue(IRInst* debugVar, IRInst* debugValue)
{
    List<IRInst*> args;
    args.add(debugVar);
    args.add(debugValue);
    return emitIntrinsicInst(
        getVoidType(),
        kIROp_DebugValue,
        (UInt)args.getCount(),
        args.getBuffer());
}

IRLiveRangeStart* IRBuilder::emitLiveRangeStart(IRInst* referenced)
{
    // This instruction doesn't produce any result,
    // so we make it's type void.
    auto inst = createInst<IRLiveRangeStart>(this, kIROp_LiveRangeStart, getVoidType(), referenced);

    addInst(inst);

    return inst;
}

IRLiveRangeEnd* IRBuilder::emitLiveRangeEnd(IRInst* referenced)
{
    // This instruction doesn't produce any result,
    // so we make it's type void.
    auto inst = createInst<IRLiveRangeEnd>(this, kIROp_LiveRangeEnd, getVoidType(), referenced);

    addInst(inst);

    return inst;
}

IRInst* IRBuilder::emitExtractExistentialValue(IRType* type, IRInst* existentialValue)
{
    auto inst = createInst<IRInst>(this, kIROp_ExtractExistentialValue, type, 1, &existentialValue);
    addInst(inst);
    return inst;
}

IRType* IRBuilder::emitExtractExistentialType(IRInst* existentialValue)
{
    auto type = getTypeKind();
    auto inst = createInst<IRInst>(this, kIROp_ExtractExistentialType, type, 1, &existentialValue);
    addInst(inst);
    return (IRType*)inst;
}

IRInst* IRBuilder::emitExtractExistentialWitnessTable(IRInst* existentialValue)
{
    auto type = getWitnessTableType(existentialValue->getDataType());
    auto inst =
        createInst<IRInst>(this, kIROp_ExtractExistentialWitnessTable, type, 1, &existentialValue);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitForwardDifferentiateInst(IRType* type, IRInst* baseFn)
{
    auto inst = createInst<IRForwardDifferentiate>(this, kIROp_ForwardDifferentiate, type, baseFn);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitPrimalSubstituteInst(IRType* type, IRInst* baseFn)
{
    auto inst = createInst<IRPrimalSubstitute>(this, kIROp_PrimalSubstitute, type, baseFn);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitDetachDerivative(IRType* type, IRInst* value)
{
    auto inst = createInst<IRDetachDerivative>(this, kIROp_DetachDerivative, type, value);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitIsDifferentialNull(IRInst* value)
{
    auto inst =
        createInst<IRIsDifferentialNull>(this, kIROp_IsDifferentialNull, getBoolType(), value);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitBackwardDifferentiateInst(IRType* type, IRInst* baseFn)
{
    auto inst =
        createInst<IRBackwardDifferentiate>(this, kIROp_BackwardDifferentiate, type, baseFn);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitDispatchKernelInst(
    IRType* type,
    IRInst* baseFn,
    IRInst* threadGroupSize,
    IRInst* dispatchSize,
    Int argCount,
    IRInst* const* inArgs)
{
    List<IRInst*> args = {baseFn, threadGroupSize, dispatchSize};
    args.addRange(inArgs, (Index)argCount);
    auto inst = createInst<IRDispatchKernel>(
        this,
        kIROp_DispatchKernel,
        type,
        (Int)args.getCount(),
        args.getBuffer());
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitCudaKernelLaunch(
    IRInst* baseFn,
    IRInst* gridDim,
    IRInst* blockDim,
    IRInst* argsArray,
    IRInst* cudaStream)
{
    IRInst* args[5] = {baseFn, gridDim, blockDim, argsArray, cudaStream};
    return emitIntrinsicInst(getVoidType(), kIROp_CudaKernelLaunch, 5, args);
}

IRInst* IRBuilder::emitGetTorchCudaStream()
{
    return emitIntrinsicInst(getPtrType(getVoidType()), kIROp_TorchGetCudaStream, 0, nullptr);
}

IRInst* IRBuilder::emitBackwardDifferentiatePrimalInst(IRType* type, IRInst* baseFn)
{
    auto inst = createInst<IRBackwardDifferentiatePrimal>(
        this,
        kIROp_BackwardDifferentiatePrimal,
        type,
        baseFn);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitBackwardDifferentiatePropagateInst(IRType* type, IRInst* baseFn)
{
    auto inst = createInst<IRBackwardDifferentiatePropagate>(
        this,
        kIROp_BackwardDifferentiatePropagate,
        type,
        baseFn);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitMakeDifferentialValuePair(IRType* type, IRInst* primal, IRInst* differential)
{
    SLANG_RELEASE_ASSERT(as<IRDifferentialPairType>(type));
    SLANG_RELEASE_ASSERT(as<IRDifferentialPairType>(type)->getValueType() != nullptr);

    IRInst* args[] = {primal, differential};
    auto inst = createInstWithTrailingArgs<IRMakeDifferentialPair>(
        this,
        kIROp_MakeDifferentialPair,
        type,
        2,
        args);
    addInst(inst);
    inst->sourceLoc = primal->sourceLoc;
    return inst;
}

IRInst* IRBuilder::emitMakeDifferentialPtrPair(IRType* type, IRInst* primal, IRInst* differential)
{
    SLANG_RELEASE_ASSERT(as<IRDifferentialPtrPairType>(type));
    SLANG_RELEASE_ASSERT(as<IRDifferentialPtrPairType>(type)->getValueType() != nullptr);

    IRInst* args[] = {primal, differential};
    auto inst = createInstWithTrailingArgs<IRMakeDifferentialPtrPair>(
        this,
        kIROp_MakeDifferentialPtrPair,
        type,
        2,
        args);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitMakeDifferentialPair(IRType* pairType, IRInst* primalVal, IRInst* diffVal)
{
    if (as<IRDifferentialPairType>(pairType))
    {
        return emitMakeDifferentialValuePair(pairType, primalVal, diffVal);
    }
    else if (as<IRDifferentialPtrPairType>(pairType))
    {
        // Quick optimization:
        // If primalVal and diffVal are extracted from the same pointer-pair,
        // we can just use the pointer-pair directly.
        //
        if (auto primalPtrVal = as<IRDifferentialPtrPairGetPrimal>(primalVal))
        {
            if (auto diffPtrVal = as<IRDifferentialPtrPairGetDifferential>(diffVal))
            {
                if (primalPtrVal->getBase() == diffPtrVal->getBase())
                    return primalPtrVal->getBase();
            }
        }
        return emitMakeDifferentialPtrPair(pairType, primalVal, diffVal);
    }
    else
    {
        SLANG_ASSERT(!"unreachable");
        return nullptr;
    }
}

IRInst* IRBuilder::emitDifferentialPairGetDifferential(IRType* diffType, IRInst* pairVal)
{
    if (as<IRDifferentialPairType>(pairVal->getDataType()))
    {
        return emitDifferentialValuePairGetDifferential(diffType, pairVal);
    }
    else if (as<IRDifferentialPtrPairType>(pairVal->getDataType()))
    {
        return emitDifferentialPtrPairGetDifferential(diffType, pairVal);
    }
    else
    {
        SLANG_ASSERT(!"unreachable");
        return nullptr;
    }
}

IRInst* IRBuilder::emitDifferentialPairGetPrimal(IRInst* pairVal)
{
    if (as<IRDifferentialPairType>(pairVal->getDataType()))
    {
        return emitDifferentialValuePairGetPrimal(pairVal);
    }
    else if (as<IRDifferentialPtrPairType>(pairVal->getDataType()))
    {
        return emitDifferentialPtrPairGetPrimal(pairVal);
    }
    else
    {
        SLANG_ASSERT(!"unreachable");
        return nullptr;
    }
}

IRInst* IRBuilder::emitDifferentialPairGetPrimal(IRType* primalType, IRInst* pairVal)
{
    if (as<IRDifferentialPairType>(pairVal->getDataType()))
    {
        return emitDifferentialValuePairGetPrimal(primalType, pairVal);
    }
    else if (as<IRDifferentialPtrPairType>(pairVal->getDataType()))
    {
        return emitDifferentialPtrPairGetPrimal(primalType, pairVal);
    }
    else
    {
        SLANG_ASSERT(!"unreachable");
        return nullptr;
    }
}

IRInst* IRBuilder::emitMakeDifferentialPairUserCode(
    IRType* type,
    IRInst* primal,
    IRInst* differential)
{
    SLANG_RELEASE_ASSERT(as<IRDifferentialPairTypeBase>(type));
    SLANG_RELEASE_ASSERT(as<IRDifferentialPairTypeBase>(type)->getValueType() != nullptr);

    IRInst* args[] = {primal, differential};
    auto inst = createInstWithTrailingArgs<IRMakeDifferentialPair>(
        this,
        kIROp_MakeDifferentialPairUserCode,
        type,
        2,
        args);
    addInst(inst);
    inst->sourceLoc = primal->sourceLoc;
    return inst;
}

IRInst* IRBuilder::emitSpecializeInst(
    IRType* type,
    IRInst* genericVal,
    UInt argCount,
    IRInst* const* args)
{
    auto innerReturnVal = findInnerMostGenericReturnVal(as<IRGeneric>(genericVal));
    if (as<IRWitnessTable>(innerReturnVal))
    {
        return createIntrinsicInst(type, kIROp_Specialize, genericVal, argCount, args);
    }

    auto inst = createInstWithTrailingArgs<IRSpecialize>(
        this,
        kIROp_Specialize,
        type,
        1,
        &genericVal,
        argCount,
        args);

    if (!inst->parent)
        addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitExpandInst(IRType* type, UInt capturedArgCount, IRInst* const* capturedArgs)
{
    auto inst = createInstWithTrailingArgs<IRSpecialize>(
        this,
        kIROp_Expand,
        type,
        capturedArgCount,
        capturedArgs,
        0,
        nullptr);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitEachInst(IRType* type, IRInst* base, IRInst* indexArg)
{
    IRInst* args[] = {base, indexArg};
    return emitIntrinsicInst(type, kIROp_Each, indexArg ? 2 : 1, args);
}

IRInst* IRBuilder::emitLookupInterfaceMethodInst(
    IRType* type,
    IRInst* witnessTableVal,
    IRInst* interfaceMethodVal)
{
    // TODO: if somebody tries to declare a struct that inherits
    // an interface conformance from a base type, then we hit
    // this assert. The problem should be fixed higher up in
    // the emit logic, but this is a reasonably early place
    // to catch it.
    //
    SLANG_ASSERT(witnessTableVal && witnessTableVal->getOp() != kIROp_StructKey);

    IRInst* args[] = {witnessTableVal, interfaceMethodVal};

    return createIntrinsicInst(type, kIROp_LookupWitness, 2, args);
}

IRInst* IRBuilder::emitGetSequentialIDInst(IRInst* rttiObj)
{
    auto inst = createInst<IRAlloca>(this, kIROp_GetSequentialID, getUIntType(), rttiObj);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitBitfieldExtract(IRType* type, IRInst* value, IRInst* offset, IRInst* bits)
{
    auto inst = createInst<IRInst>(this, kIROp_BitfieldExtract, type, value, offset, bits);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitBitfieldInsert(
    IRType* type,
    IRInst* base,
    IRInst* insert,
    IRInst* offset,
    IRInst* bits)
{
    auto inst = createInst<IRInst>(this, kIROp_BitfieldInsert, type, base, insert, offset, bits);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitAlloca(IRInst* type, IRInst* rttiObjPtr)
{
    auto inst = createInst<IRAlloca>(this, kIROp_Alloca, (IRType*)type, rttiObjPtr);

    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitGlobalValueRef(IRInst* globalInst)
{
    auto inst = createInst<IRGlobalValueRef>(
        this,
        kIROp_GlobalValueRef,
        (IRType*)globalInst->getFullType(),
        globalInst);

    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitPackAnyValue(IRType* type, IRInst* value)
{
    auto inst = createInst<IRPackAnyValue>(this, kIROp_PackAnyValue, type, value);

    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitUnpackAnyValue(IRType* type, IRInst* value)
{
    auto inst = createInst<IRPackAnyValue>(this, kIROp_UnpackAnyValue, type, value);

    addInst(inst);
    return inst;
}

IRCall* IRBuilder::emitCallInst(IRType* type, IRInst* pFunc, UInt argCount, IRInst* const* args)
{
    auto inst =
        createInstWithTrailingArgs<IRCall>(this, kIROp_Call, type, 1, &pFunc, argCount, args);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitTryCallInst(
    IRType* type,
    IRBlock* successBlock,
    IRBlock* failureBlock,
    IRInst* func,
    UInt argCount,
    IRInst* const* args)
{
    IRInst* fixedArgs[] = {successBlock, failureBlock, func};
    auto inst = createInstWithTrailingArgs<IRTryCall>(
        this,
        kIROp_TryCall,
        type,
        3,
        fixedArgs,
        argCount,
        args);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::createIntrinsicInst(IRType* type, IROp op, UInt argCount, IRInst* const* args)
{
    return createInstWithTrailingArgs<IRInst>(this, op, type, argCount, args);
}

IRInst* IRBuilder::createIntrinsicInst(
    IRType* type,
    IROp op,
    IRInst* operand,
    UInt operandCount,
    IRInst* const* operands)
{
    return createInstWithTrailingArgs<IRInst>(this, op, type, operand, operandCount, operands);
}

IRInst* IRBuilder::createIntrinsicInst(
    IRType* type,
    IROp op,
    UInt operandListCount,
    UInt const* listOperandCounts,
    IRInst* const* const* listOperands)
{
    return createInstImpl<IRInst>(
        this,
        op,
        type,
        0,
        nullptr,
        (Int)operandListCount,
        (Int const*)listOperandCounts,
        listOperands);
}


IRInst* IRBuilder::emitIntrinsicInst(IRType* type, IROp op, UInt argCount, IRInst* const* args)
{
    auto inst = createIntrinsicInst(type, op, argCount, args);
    if (!inst->parent)
        addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitDefaultConstructRaw(IRType* type)
{
    return emitIntrinsicInst(type, kIROp_DefaultConstruct, 0, nullptr);
}

IRInst* IRBuilder::emitDefaultConstruct(IRType* type, bool fallback)
{
    IRType* actualType = type;
    for (;;)
    {
        if (auto attr = as<IRAttributedType>(actualType))
            actualType = attr->getBaseType();
        else if (auto rateQualified = as<IRRateQualifiedType>(actualType))
            actualType = rateQualified->getValueType();
        else
            break;
    }
    switch (actualType->getOp())
    {
    case kIROp_Int8Type:
    case kIROp_Int16Type:
    case kIROp_IntType:
    case kIROp_IntPtrType:
    case kIROp_Int64Type:
    case kIROp_UInt8Type:
    case kIROp_UInt16Type:
    case kIROp_UIntType:
    case kIROp_UIntPtrType:
    case kIROp_UInt64Type:
    case kIROp_CharType:
        return getIntValue(type, 0);
    case kIROp_BoolType:
        return getBoolValue(false);
    case kIROp_FloatType:
    case kIROp_HalfType:
    case kIROp_DoubleType:
        return getFloatValue(type, 0.0);
    case kIROp_VoidType:
        return getVoidValue();
    case kIROp_StringType:
        return getStringValue(UnownedStringSlice());
    case kIROp_PtrType:
    case kIROp_InOutType:
    case kIROp_OutType:
    case kIROp_RawPointerType:
    case kIROp_RefType:
    case kIROp_ConstRefType:
    case kIROp_ComPtrType:
    case kIROp_NativePtrType:
    case kIROp_NativeStringType:
        return getNullPtrValue(type);
    case kIROp_OptionalType:
        {
            auto inner =
                emitDefaultConstruct(as<IROptionalType>(actualType)->getValueType(), fallback);
            if (!inner)
                return nullptr;
            return emitMakeOptionalNone(type, inner);
        }
    case kIROp_TupleType:
        {
            List<IRInst*> elements;
            auto tupleType = as<IRTupleType>(actualType);
            for (UInt i = 0; i < tupleType->getOperandCount(); i++)
            {
                auto operand = tupleType->getOperand(i);
                if (as<IRAttr>(operand))
                    break;
                auto inner = emitDefaultConstruct((IRType*)operand, fallback);
                if (!inner)
                    return nullptr;
                elements.add(inner);
            }
            return emitMakeTuple(type, elements);
        }
    case kIROp_StructType:
        {
            List<IRInst*> elements;
            auto structType = as<IRStructType>(actualType);
            for (auto field : structType->getFields())
            {
                auto fieldType = field->getFieldType();
                auto inner = emitDefaultConstruct(fieldType, fallback);
                if (!inner)
                    return nullptr;
                elements.add(inner);
            }
            return emitMakeStruct(type, elements);
        }
    case kIROp_ArrayType:
        {
            auto arrayType = as<IRArrayType>(actualType);
            if (auto count = as<IRIntLit>(arrayType->getElementCount()))
            {
                auto element = emitDefaultConstruct(arrayType->getElementType(), fallback);
                if (!element)
                    return nullptr;
                List<IRInst*> elements;
                constexpr int maxCount = 4096;
                if (count->getValue() > maxCount)
                    break;
                for (IRIntegerValue i = 0; i < count->getValue(); i++)
                {
                    elements.add(element);
                }
                return emitMakeArray(type, elements.getCount(), elements.getBuffer());
            }
            break;
        }
    case kIROp_VectorType:
        {
            auto inner =
                emitDefaultConstruct(as<IRVectorType>(actualType)->getElementType(), fallback);
            if (!inner)
                return nullptr;
            return emitIntrinsicInst(type, kIROp_MakeVectorFromScalar, 1, &inner);
        }
    case kIROp_CoopVectorType:
        {
            auto coopVecType = as<IRCoopVectorType>(actualType);
            if (auto count = as<IRIntLit>(coopVecType->getElementCount()))
            {
                auto element = emitDefaultConstruct(coopVecType->getElementType(), fallback);
                if (!element)
                    return nullptr;
                List<IRInst*> elements;
                constexpr int maxCount = 4096;
                if (count->getValue() > maxCount)
                    break;
                for (IRIntegerValue i = 0; i < count->getValue(); i++)
                {
                    elements.add(element);
                }
                return emitMakeCoopVector(type, elements.getCount(), elements.getBuffer());
            }
            break;
        }
    case kIROp_MatrixType:
        {
            auto inner =
                emitDefaultConstruct(as<IRMatrixType>(actualType)->getElementType(), fallback);
            if (!inner)
                return nullptr;
            return emitIntrinsicInst(type, kIROp_MakeMatrixFromScalar, 1, &inner);
        }
    default:
        break;
    }
    if (fallback)
    {
        return emitIntrinsicInst(type, kIROp_DefaultConstruct, 0, nullptr);
    }
    return nullptr;
}

IRInst* IRBuilder::emitEmbeddedDownstreamIR(CodeGenTarget target, ISlangBlob* blob)
{
    IRInst* args[] = {getIntValue(getIntType(), (int)target), getBlobValue(blob)};

    return emitIntrinsicInst(getVoidType(), kIROp_EmbeddedDownstreamIR, 2, args);
}

enum class TypeCastStyle
{
    Unknown = -1,
    Int,
    Float,
    Bool,
    Ptr,
    Enum,
    Void
};
static TypeCastStyle _getTypeStyleId(IRType* type)
{
    type = (IRType*)unwrapAttributedType(type);

    if (auto vectorType = as<IRVectorType>(type))
    {
        return _getTypeStyleId(vectorType->getElementType());
    }
    if (auto matrixType = as<IRMatrixType>(type))
    {
        return _getTypeStyleId(matrixType->getElementType());
    }
    auto style = getTypeStyle(type->getOp());
    switch (style)
    {
    case kIROp_IntType:
        return TypeCastStyle::Int;
    case kIROp_FloatType:
    case kIROp_HalfType:
    case kIROp_DoubleType:
        return TypeCastStyle::Float;
    case kIROp_BoolType:
        return TypeCastStyle::Bool;
    case kIROp_PtrType:
    case kIROp_InOutType:
    case kIROp_OutType:
    case kIROp_RawPointerType:
    case kIROp_RefType:
    case kIROp_ConstRefType:
        return TypeCastStyle::Ptr;
    case kIROp_EnumType:
        return TypeCastStyle::Enum;
    case kIROp_VoidType:
        return TypeCastStyle::Void;
    default:
        return TypeCastStyle::Unknown;
    }
}

IRInst* IRBuilder::emitCast(IRType* type, IRInst* value, bool fallbackToBuiltinCast)
{
    if (isTypeEqual(type, value->getDataType()))
        return value;

    auto toStyle = _getTypeStyleId(type);
    auto fromStyle = _getTypeStyleId(value->getDataType());

    if (fromStyle == TypeCastStyle::Void)
    {
        // We shouldn't be casting from void to other types.
        SLANG_UNREACHABLE("cast from void type");
    }

    if (toStyle == TypeCastStyle::Unknown || fromStyle == TypeCastStyle::Unknown)
    {
        if (fallbackToBuiltinCast)
        {
            return emitIntrinsicInst(type, kIROp_BuiltinCast, 1, &value);
        }
        else
        {
            return nullptr;
        }
    }

    struct OpSeq
    {
        IROp op0, op1;
        OpSeq(IROp op)
        {
            op0 = op;
            op1 = kIROp_Nop;
        }
        OpSeq(IROp op, IROp inOp1)
        {
            op0 = op;
            op1 = inOp1;
        }
    };

    static const OpSeq opMap[5][6] = {
        /*      To:      Int, Float, Bool, Ptr, Enum, Void */
        /* From Int   */ {
            kIROp_IntCast,
            kIROp_CastIntToFloat,
            kIROp_IntCast,
            kIROp_CastIntToPtr,
            kIROp_CastIntToEnum,
            kIROp_CastToVoid},
        /* From Float */
        {kIROp_CastFloatToInt,
         kIROp_FloatCast,
         {kIROp_Neq},
         {kIROp_CastFloatToInt, kIROp_CastIntToPtr},
         {kIROp_CastFloatToInt, kIROp_CastIntToEnum},
         kIROp_CastToVoid},
        /* From Bool  */
        {kIROp_IntCast,
         kIROp_CastIntToFloat,
         kIROp_Nop,
         kIROp_CastIntToPtr,
         kIROp_CastIntToEnum,
         kIROp_CastToVoid},
        /* From Ptr   */
        {kIROp_CastPtrToInt,
         {kIROp_CastPtrToInt, kIROp_CastIntToFloat},
         kIROp_CastPtrToBool,
         kIROp_BitCast,
         {kIROp_CastPtrToInt, kIROp_CastIntToEnum},
         kIROp_CastToVoid},
        /* From Enum   */
        {kIROp_CastEnumToInt,
         {kIROp_CastEnumToInt, kIROp_CastIntToFloat},
         {kIROp_CastEnumToInt, kIROp_IntCast},
         {kIROp_CastEnumToInt, kIROp_CastIntToPtr},
         kIROp_EnumCast,
         kIROp_CastToVoid},
    };

    auto op = opMap[(int)fromStyle][(int)toStyle];
    if (op.op0 == kIROp_Nop)
        return value;
    auto t = type;
    if (op.op1 != kIROp_Nop)
    {
        if (toStyle == TypeCastStyle::Bool)
            t = getIntType();
        else
            t = getUInt64Type();
        if (auto vecType = as<IRVectorType>(type))
            t = getVectorType(t, vecType->getElementCount());
        else if (auto matType = as<IRMatrixType>(type))
            t = getMatrixType(
                t,
                matType->getRowCount(),
                matType->getColumnCount(),
                matType->getLayout());
    }

    if (op.op0 == kIROp_Neq)
    {
        IRInst* args[2] = {value, emitDefaultConstruct(value->getDataType())};
        return emitIntrinsicInst(type, op.op0, 2, args);
    }

    auto result = emitIntrinsicInst(t, op.op0, 1, &value);
    if (op.op1 != kIROp_Nop)
    {
        result = emitIntrinsicInst(type, op.op1, 1, &result);
    }
    return result;
}

IRInst* IRBuilder::emitVectorReshape(IRType* type, IRInst* value)
{
    auto targetVectorType = as<IRVectorType>(type);
    auto sourceVectorType = as<IRVectorType>(value->getDataType());
    if (targetVectorType && !sourceVectorType)
    {
        auto elementType = targetVectorType->getElementType();
        Index elemCount = 1;
        if (auto intLit = as<IRIntLit>(targetVectorType->getElementCount()))
        {
            elemCount = (Index)intLit->getValue();
        }
        IRInst* zeroVal = emitDefaultConstruct(elementType);
        List<IRInst*> defaultVals;
        defaultVals.reserve(elemCount);
        defaultVals.add(value);
        for (auto i = 1; i < elemCount; i++)
            defaultVals.add(zeroVal);
        return emitMakeVector(targetVectorType, defaultVals);
    }
    else if (!targetVectorType)
    {
        if (!sourceVectorType)
            return emitCast(targetVectorType, value);
        else
        {
            UInt index = 0;
            return emitCast(
                type,
                emitSwizzle(sourceVectorType->getElementType(), value, 1, &index));
        }
    }
    if (targetVectorType->getElementCount() != sourceVectorType->getElementCount())
    {
        auto fromCount = as<IRIntLit>(sourceVectorType->getElementCount());
        auto toCount = as<IRIntLit>(targetVectorType->getElementCount());
        if (fromCount && toCount)
        {
            if (toCount->getValue() < fromCount->getValue())
            {
                List<UInt> indices;
                for (UInt i = 0; i < (UInt)toCount->getValue(); i++)
                    indices.add(i);
                return emitSwizzle(
                    targetVectorType,
                    value,
                    (UInt)indices.getCount(),
                    indices.getBuffer());
            }
            else if (toCount->getValue() > fromCount->getValue())
            {
                List<IRInst*> args;
                for (UInt i = 0; i < (UInt)fromCount->getValue(); i++)
                {
                    auto element = emitSwizzle(sourceVectorType->getElementType(), value, 1, &i);
                    args.add(element);
                }
                for (IRIntegerValue i = fromCount->getValue(); i < toCount->getValue(); i++)
                {
                    args.add(emitDefaultConstruct(targetVectorType->getElementType()));
                }
                return emitMakeVector(targetVectorType, args);
            }
            else
            {
                // Sizes match, no need to reshape.
                return value;
            }
        }
        auto reshape = emitIntrinsicInst(
            getVectorType(sourceVectorType->getElementType(), targetVectorType->getElementCount()),
            kIROp_VectorReshape,
            1,
            &value);
        return emitCast(type, reshape);
    }
    return value;
}

IRInst* IRBuilder::emitMakeUInt64(IRInst* low, IRInst* high)
{
    IRInst* args[2] = {low, high};
    return emitIntrinsicInst(getUInt64Type(), kIROp_MakeUInt64, 2, args);
}

IRInst* IRBuilder::emitMakeRTTIObject(IRInst* typeInst)
{
    auto inst = createInst<IRRTTIObject>(this, kIROp_RTTIObject, getRTTIType(), typeInst);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitMakeValuePack(IRType* type, UInt count, IRInst* const* args)
{
    return emitIntrinsicInst(type, kIROp_MakeValuePack, count, args);
}

IRInst* IRBuilder::emitMakeValuePack(UInt count, IRInst* const* args)
{
    ShortList<IRType*> types;
    for (UInt i = 0; i < count; ++i)
        types.add(args[i]->getFullType());

    auto type = getTypePack((UInt)types.getCount(), types.getArrayView().getBuffer());
    return emitIntrinsicInst(type, kIROp_MakeValuePack, count, args);
}

IRInst* IRBuilder::emitMakeTuple(IRType* type, UInt count, IRInst* const* args)
{
    return emitIntrinsicInst(type, kIROp_MakeTuple, count, args);
}

IRInst* IRBuilder::emitMakeTargetTuple(IRType* type, UInt count, IRInst* const* args)
{
    return emitIntrinsicInst(type, kIROp_MakeTargetTuple, count, args);
}

IRInst* IRBuilder::emitTargetTupleGetElement(
    IRType* elementType,
    IRInst* targetTupleVal,
    IRInst* indexVal)
{
    IRInst* args[] = {targetTupleVal, indexVal};
    return emitIntrinsicInst(elementType, kIROp_GetTargetTupleElement, 2, args);
}

IRInst* IRBuilder::emitMakeTuple(UInt count, IRInst* const* args)
{
    List<IRType*> types;
    for (UInt i = 0; i < count; ++i)
        types.add(args[i]->getFullType());

    auto type = getTupleType(types);
    return emitMakeTuple(type, count, args);
}

IRInst* IRBuilder::emitMakeString(IRInst* nativeStr)
{
    return emitIntrinsicInst(getStringType(), kIROp_MakeString, 1, &nativeStr);
}

IRInst* IRBuilder::emitGetNativeString(IRInst* str)
{
    return emitIntrinsicInst(getNativeStringType(), kIROp_getNativeStr, 1, &str);
}

IRInst* IRBuilder::emitGetElement(IRType* type, IRInst* arrayLikeType, IRIntegerValue element)
{
    IRInst* args[] = {arrayLikeType, getIntValue(getIntType(), element)};
    return emitIntrinsicInst(type, kIROp_GetElement, 2, args);
}

IRInst* IRBuilder::emitGetElementPtr(IRType* type, IRInst* arrayLikeType, IRIntegerValue element)
{
    IRInst* args[] = {arrayLikeType, getIntValue(getIntType(), element)};
    return emitIntrinsicInst(type, kIROp_GetElementPtr, 2, args);
}

IRInst* IRBuilder::emitGetTupleElement(IRType* type, IRInst* tuple, IRInst* element)
{
    IRInst* args[] = {tuple, element};
    return emitIntrinsicInst(type, kIROp_GetTupleElement, 2, args);
}

IRInst* IRBuilder::emitGetTupleElement(IRType* type, IRInst* tuple, UInt element)
{
    // As a quick simplification/optimization, if the user requests
    // `getTupleElement(makeTuple(a_0, a_1, ... a_N), i)` then we should
    // just return `a_i`, provided that the index is properly in range.
    //
    switch (tuple->getOp())
    {
    case kIROp_MakeTuple:
    case kIROp_MakeValuePack:
    case kIROp_MakeWitnessPack:
    case kIROp_TypePack:
        if (element < tuple->getOperandCount())
        {
            return tuple->getOperand(element);
        }
        break;
    }
    return emitGetTupleElement(type, tuple, getIntValue(getIntType(), element));
}

IRInst* IRBuilder::emitMakeResultError(IRType* resultType, IRInst* errorVal)
{
    return emitIntrinsicInst(resultType, kIROp_MakeResultError, 1, &errorVal);
}

IRInst* IRBuilder::emitMakeResultValue(IRType* resultType, IRInst* value)
{
    return emitIntrinsicInst(resultType, kIROp_MakeResultValue, 1, &value);
}

IRInst* IRBuilder::emitIsResultError(IRInst* result)
{
    return emitIntrinsicInst(getBoolType(), kIROp_IsResultError, 1, &result);
}

IRInst* IRBuilder::emitGetResultError(IRInst* result)
{
    SLANG_ASSERT(result->getDataType());
    return emitIntrinsicInst(
        cast<IRResultType>(result->getDataType())->getErrorType(),
        kIROp_GetResultError,
        1,
        &result);
}

IRInst* IRBuilder::emitGetResultValue(IRInst* result)
{
    SLANG_ASSERT(result->getDataType());
    return emitIntrinsicInst(
        cast<IRResultType>(result->getDataType())->getValueType(),
        kIROp_GetResultValue,
        1,
        &result);
}

IRInst* IRBuilder::emitOptionalHasValue(IRInst* optValue)
{
    return emitIntrinsicInst(getBoolType(), kIROp_OptionalHasValue, 1, &optValue);
}

IRInst* IRBuilder::emitGetOptionalValue(IRInst* optValue)
{
    return emitIntrinsicInst(
        cast<IROptionalType>(optValue->getDataType())->getValueType(),
        kIROp_GetOptionalValue,
        1,
        &optValue);
}

IRInst* IRBuilder::emitMakeOptionalValue(IRInst* optType, IRInst* value)
{
    return emitIntrinsicInst((IRType*)optType, kIROp_MakeOptionalValue, 1, &value);
}

IRInst* IRBuilder::emitMakeOptionalNone(IRInst* optType, IRInst* defaultValue)
{
    return emitIntrinsicInst((IRType*)optType, kIROp_MakeOptionalNone, 1, &defaultValue);
}

IRInst* IRBuilder::emitMakeVectorFromScalar(IRType* type, IRInst* scalarValue)
{
    return emitIntrinsicInst(type, kIROp_MakeVectorFromScalar, 1, &scalarValue);
}

IRInst* IRBuilder::emitMakeCompositeFromScalar(IRType* type, IRInst* scalarValue)
{
    switch (type->getOp())
    {
    case kIROp_VectorType:
        return emitMakeVectorFromScalar(type, scalarValue);
    case kIROp_MatrixType:
        return emitMakeMatrixFromScalar(type, scalarValue);
    case kIROp_ArrayType:
        return emitMakeArrayFromElement(type, scalarValue);
    default:
        SLANG_UNEXPECTED("unhandled composite type");
        UNREACHABLE_RETURN(nullptr);
    }
}

IRInst* IRBuilder::emitMatrixReshape(IRType* type, IRInst* inst)
{
    return emitIntrinsicInst(type, kIROp_MatrixReshape, 1, &inst);
}

IRInst* IRBuilder::emitMakeVector(IRType* type, UInt argCount, IRInst* const* args)
{
    return emitIntrinsicInst(type, kIROp_MakeVector, argCount, args);
}

IRInst* IRBuilder::emitDifferentialValuePairGetDifferential(IRType* diffType, IRInst* diffPair)
{
    SLANG_ASSERT(as<IRDifferentialPairTypeBase>(diffPair->getDataType()));
    return emitIntrinsicInst(diffType, kIROp_DifferentialPairGetDifferential, 1, &diffPair);
}


IRInst* IRBuilder::emitDifferentialPtrPairGetDifferential(IRType* diffType, IRInst* diffPair)
{
    SLANG_ASSERT(as<IRDifferentialPtrPairType>(diffPair->getDataType()));
    return emitIntrinsicInst(diffType, kIROp_DifferentialPtrPairGetDifferential, 1, &diffPair);
}

IRInst* IRBuilder::emitDifferentialValuePairGetPrimal(IRInst* diffPair)
{
    auto valueType = cast<IRDifferentialPairTypeBase>(diffPair->getDataType())->getValueType();
    return emitIntrinsicInst(valueType, kIROp_DifferentialPairGetPrimal, 1, &diffPair);
}

IRInst* IRBuilder::emitDifferentialValuePairGetPrimal(IRType* primalType, IRInst* diffPair)
{
    return emitIntrinsicInst(primalType, kIROp_DifferentialPairGetPrimal, 1, &diffPair);
}

IRInst* IRBuilder::emitDifferentialPtrPairGetPrimal(IRInst* diffPair)
{
    auto valueType = cast<IRDifferentialPairTypeBase>(diffPair->getDataType())->getValueType();
    return emitIntrinsicInst(valueType, kIROp_DifferentialPtrPairGetPrimal, 1, &diffPair);
}

IRInst* IRBuilder::emitDifferentialPtrPairGetPrimal(IRType* primalType, IRInst* diffPair)
{
    return emitIntrinsicInst(primalType, kIROp_DifferentialPtrPairGetPrimal, 1, &diffPair);
}

IRInst* IRBuilder::emitDifferentialPairGetDifferentialUserCode(IRType* diffType, IRInst* diffPair)
{
    SLANG_ASSERT(as<IRDifferentialPairTypeBase>(diffPair->getDataType()));
    return emitIntrinsicInst(diffType, kIROp_DifferentialPairGetDifferentialUserCode, 1, &diffPair);
}

IRInst* IRBuilder::emitDifferentialPairGetPrimalUserCode(IRInst* diffPair)
{
    auto valueType = cast<IRDifferentialPairTypeBase>(diffPair->getDataType())->getValueType();
    return emitIntrinsicInst(valueType, kIROp_DifferentialPairGetPrimalUserCode, 1, &diffPair);
}

IRInst* IRBuilder::emitMakeMatrix(IRType* type, UInt argCount, IRInst* const* args)
{
    return emitIntrinsicInst(type, kIROp_MakeMatrix, argCount, args);
}

IRInst* IRBuilder::emitMakeMatrixFromScalar(IRType* type, IRInst* scalarValue)
{
    return emitIntrinsicInst(type, kIROp_MakeMatrixFromScalar, 1, &scalarValue);
}

IRInst* IRBuilder::emitMakeCoopVector(IRType* type, UInt argCount, IRInst* const* args)
{
    return emitIntrinsicInst(type, kIROp_MakeCoopVector, argCount, args);
}

IRInst* IRBuilder::emitMakeArray(IRType* type, UInt argCount, IRInst* const* args)
{
    return emitIntrinsicInst(type, kIROp_MakeArray, argCount, args);
}

IRInst* IRBuilder::emitMakeArrayList(IRType* type, UInt argCount, IRInst* const* args)
{
    return emitIntrinsicInst(type, kIROp_MakeArrayList, argCount, args);
}

IRInst* IRBuilder::emitMakeArrayFromElement(IRType* type, IRInst* element)
{
    return emitIntrinsicInst(type, kIROp_MakeArrayFromElement, 1, &element);
}

IRInst* IRBuilder::emitMakeStruct(IRType* type, UInt argCount, IRInst* const* args)
{
    return emitIntrinsicInst(type, kIROp_MakeStruct, argCount, args);
}

IRInst* IRBuilder::emitMakeTensorView(IRType* type, IRInst* val)
{
    return emitIntrinsicInst(type, kIROp_MakeTensorView, 1, &val);
}

IRInst* IRBuilder::emitMakeExistential(IRType* type, IRInst* value, IRInst* witnessTable)
{
    IRInst* args[] = {value, witnessTable};
    return emitIntrinsicInst(type, kIROp_MakeExistential, SLANG_COUNT_OF(args), args);
}

IRInst* IRBuilder::emitMakeExistentialWithRTTI(
    IRType* type,
    IRInst* value,
    IRInst* witnessTable,
    IRInst* rtti)
{
    IRInst* args[] = {value, witnessTable, rtti};
    return emitIntrinsicInst(type, kIROp_MakeExistentialWithRTTI, SLANG_COUNT_OF(args), args);
}

IRInst* IRBuilder::emitWrapExistential(
    IRType* type,
    IRInst* value,
    UInt slotArgCount,
    IRInst* const* slotArgs)
{
    if (slotArgCount == 0)
        return value;

    // If we are wrapping a single concrete value into
    // an interface type, then this is really a `makeExistential`
    //
    // TODO: We may want to check for a `specialize` of a generic interface as well.
    //
    if (as<IRInterfaceType>(type))
    {
        if (slotArgCount >= 2)
        {
            // We are being asked to emit `wrapExistential(value, concreteType, witnessTable, ...) :
            // someInterface`
            //
            // We also know that a concrete value being wrapped will always be an existential box,
            // so we expect that `value : BindInterface<I, C>` for some concrete `C`.
            //
            // We want to emit `makeExistential(getValueFromBoundInterface(value) : C,
            // witnessTable)`.
            //
            auto concreteType = (IRType*)(slotArgs[0]);
            auto witnessTable = slotArgs[1];
            if (slotArgs[0]->getOp() == kIROp_DynamicType)
                return value;
            auto deref = emitGetValueFromBoundInterface(concreteType, value);
            return emitMakeExistential(type, deref, witnessTable);
        }
    }

    IRInst* fixedArgs[] = {value};
    auto inst = createInstImpl<IRInst>(
        this,
        kIROp_WrapExistential,
        type,
        SLANG_COUNT_OF(fixedArgs),
        fixedArgs,
        slotArgCount,
        slotArgs);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::addPrimalValueStructKeyDecoration(IRInst* target, IRStructKey* key)
{
    return addDecoration(target, kIROp_PrimalValueStructKeyDecoration, key);
}

IRInst* IRBuilder::addPrimalElementTypeDecoration(IRInst* target, IRInst* type)
{
    return addDecoration(target, kIROp_PrimalElementTypeDecoration, type);
}

IRInst* IRBuilder::addIntermediateContextFieldDifferentialTypeDecoration(
    IRInst* target,
    IRInst* witness)
{
    return addDecoration(target, kIROp_IntermediateContextFieldDifferentialTypeDecoration, witness);
}

RefPtr<IRModule> IRModule::create(Session* session)
{
    RefPtr<IRModule> module = new IRModule(session);

    auto moduleInst = module->_allocateInst<IRModuleInst>(kIROp_Module, 0);

    module->m_moduleInst = moduleInst;
    moduleInst->module = module;

    return module;
}

void IRModule::buildMangledNameToGlobalInstMap()
{
    m_mapMangledNameToGlobalInst.clear();
    for (auto inst : getGlobalInsts())
    {
        if (auto linkageDecor = inst->findDecoration<IRLinkageDecoration>())
        {
            m_mapMangledNameToGlobalInst[linkageDecor->getMangledName()].add(inst);
        }
    }
}

IRDominatorTree* IRModule::findOrCreateDominatorTree(IRGlobalValueWithCode* func)
{
    IRAnalysis* analysis = m_mapInstToAnalysis.tryGetValue(func);
    if (analysis)
        return analysis->getDominatorTree();
    else
    {
        m_mapInstToAnalysis[func] = IRAnalysis();
        analysis = m_mapInstToAnalysis.tryGetValue(func);
    }
    analysis->domTree = computeDominatorTree(func);
    return analysis->getDominatorTree();
}

IRInst* IRBuilder::addDifferentiableTypeDictionaryDecoration(IRInst* target)
{
    return addDecoration(target, kIROp_DifferentiableTypeDictionaryDecoration);
}

IRInst* IRBuilder::addDifferentiableTypeEntry(
    IRInst* dictDecoration,
    IRInst* irType,
    IRInst* conformanceWitness)
{
    auto oldLoc = this->getInsertLoc();

    IRDifferentiableTypeDictionaryItem* item = nullptr;

    this->setInsertInto(dictDecoration);

    IRInst* args[2] = {irType, conformanceWitness};
    item = createInstWithTrailingArgs<IRDifferentiableTypeDictionaryItem>(
        this,
        kIROp_DifferentiableTypeDictionaryItem,
        nullptr,
        2,
        args);

    addInst(item);

    this->setInsertLoc(oldLoc);

    return item;
}


IRFunc* IRBuilder::createFunc()
{
    IRFunc* rsFunc = createInst<IRFunc>(this, kIROp_Func, nullptr);
    _maybeSetSourceLoc(rsFunc);
    addGlobalValue(this, rsFunc);
    return rsFunc;
}

IRGlobalVar* IRBuilder::createGlobalVar(IRType* valueType)
{
    auto ptrType = getPtrType(valueType);
    IRGlobalVar* globalVar = createInst<IRGlobalVar>(this, kIROp_GlobalVar, ptrType);
    _maybeSetSourceLoc(globalVar);
    addGlobalValue(this, globalVar);
    return globalVar;
}

IRGlobalVar* IRBuilder::createGlobalVar(IRType* valueType, AddressSpace addressSpace)
{
    auto ptrType = getPtrType(kIROp_PtrType, valueType, addressSpace);
    IRGlobalVar* globalVar = createInst<IRGlobalVar>(this, kIROp_GlobalVar, ptrType);
    _maybeSetSourceLoc(globalVar);
    addGlobalValue(this, globalVar);
    return globalVar;
}

IRGlobalParam* IRBuilder::createGlobalParam(IRType* valueType)
{
    IRGlobalParam* inst = createInst<IRGlobalParam>(this, kIROp_GlobalParam, valueType);
    _maybeSetSourceLoc(inst);
    addGlobalValue(this, inst);
    return inst;
}

IRWitnessTable* IRBuilder::createWitnessTable(IRType* baseType, IRType* subType)
{
    IRWitnessTable* witnessTable = createInst<IRWitnessTable>(
        this,
        kIROp_WitnessTable,
        getWitnessTableType(baseType),
        subType);
    addGlobalValue(this, witnessTable);
    return witnessTable;
}

IRWitnessTableEntry* IRBuilder::createWitnessTableEntry(
    IRWitnessTable* witnessTable,
    IRInst* requirementKey,
    IRInst* satisfyingVal)
{
    IRWitnessTableEntry* entry = createInst<IRWitnessTableEntry>(
        this,
        kIROp_WitnessTableEntry,
        nullptr,
        requirementKey,
        satisfyingVal);

    if (witnessTable)
    {
        entry->insertAtEnd(witnessTable);
    }

    return entry;
}

IRInterfaceRequirementEntry* IRBuilder::createInterfaceRequirementEntry(
    IRInst* requirementKey,
    IRInst* requirementVal)
{
    IRInterfaceRequirementEntry* entry = createInst<IRInterfaceRequirementEntry>(
        this,
        kIROp_InterfaceRequirementEntry,
        nullptr,
        requirementKey,
        requirementVal);
    addGlobalValue(this, entry);
    return entry;
}

IRInst* IRBuilder::createThisTypeWitness(IRType* interfaceType)
{
    IRInst* witness = createInst<IRThisTypeWitness>(
        this,
        kIROp_ThisTypeWitness,
        getWitnessTableType(interfaceType));
    addGlobalValue(this, witness);
    return witness;
}

IRInst* IRBuilder::getTypeEqualityWitness(IRType* witnessType, IRType* type1, IRType* type2)
{
    IRInst* operands[2] = {type1, type2};
    return (IRType*)createIntrinsicInst(witnessType, kIROp_TypeEqualityWitness, 2, operands);
}

IRStructType* IRBuilder::createStructType()
{
    IRStructType* structType = createInst<IRStructType>(this, kIROp_StructType, getTypeKind());
    addGlobalValue(this, structType);
    return structType;
}

IRClassType* IRBuilder::createClassType()
{
    IRClassType* classType = createInst<IRClassType>(this, kIROp_ClassType, getTypeKind());
    addGlobalValue(this, classType);
    return classType;
}

IREnumType* IRBuilder::createEnumType(IRType* tagType)
{
    IREnumType* enumType = createInst<IREnumType>(this, kIROp_EnumType, getTypeKind(), tagType);
    addGlobalValue(this, enumType);
    return enumType;
}

IRGLSLShaderStorageBufferType* IRBuilder::createGLSLShaderStorableBufferType()
{
    IRGLSLShaderStorageBufferType* ssboType = createInst<IRGLSLShaderStorageBufferType>(
        this,
        kIROp_GLSLShaderStorageBufferType,
        getTypeKind());
    addGlobalValue(this, ssboType);
    return ssboType;
}

IRGLSLShaderStorageBufferType* IRBuilder::createGLSLShaderStorableBufferType(
    UInt operandCount,
    IRInst* const* operands)
{
    IRGLSLShaderStorageBufferType* ssboType = createInst<IRGLSLShaderStorageBufferType>(
        this,
        kIROp_GLSLShaderStorageBufferType,
        getTypeKind(),
        operandCount,
        operands);
    addGlobalValue(this, ssboType);
    return ssboType;
}

IRInterfaceType* IRBuilder::createInterfaceType(UInt operandCount, IRInst* const* operands)
{
    IRInterfaceType* interfaceType = createInst<IRInterfaceType>(
        this,
        kIROp_InterfaceType,
        getTypeKind(),
        operandCount,
        operands);
    addGlobalValue(this, interfaceType);
    return interfaceType;
}

IRStructKey* IRBuilder::createStructKey()
{
    IRStructKey* structKey = createInst<IRStructKey>(this, kIROp_StructKey, nullptr);
    addGlobalValue(this, structKey);
    return structKey;
}

// Create a field nested in a struct type, declaring that
// the specified field key maps to a field with the specified type.
IRStructField* IRBuilder::createStructField(
    IRType* aggType,
    IRStructKey* fieldKey,
    IRType* fieldType)
{
    IRInst* operands[] = {fieldKey, fieldType};
    IRStructField* field = (IRStructField*)createInstWithTrailingArgs<IRInst>(
        this,
        kIROp_StructField,
        nullptr,
        0,
        nullptr,
        2,
        operands);

    if (aggType)
    {
        field->insertAtEnd(aggType);
    }

    return field;
}

IRGeneric* IRBuilder::createGeneric()
{
    IRGeneric* irGeneric = createInst<IRGeneric>(this, kIROp_Generic, nullptr);
    return irGeneric;
}

IRGeneric* IRBuilder::emitGeneric()
{
    auto irGeneric = createGeneric();
    addGlobalValue(this, irGeneric);
    return irGeneric;
}

IRBlock* IRBuilder::createBlock()
{
    return createInst<IRBlock>(this, kIROp_Block, getBasicBlockType());
}

void IRBuilder::insertBlock(IRBlock* block)
{
    // If we are emitting into a function
    // (or another value with code), then
    // append the block to the function and
    // set this block as the new parent for
    // subsequent instructions we insert.
    //
    // TODO: This should probably insert the block
    // after the current "insert into" block if
    // there is one. Right now we are always
    // adding the block to the end of the list,
    // which is technically valid (the ordering
    // of blocks doesn't affect the CFG topology),
    // but some later passes might assume the ordering
    // is significant in representing the intent
    // of the original code.
    //
    auto f = getFunc();
    if (f)
    {
        f->addBlock(block);
        setInsertInto(block);
    }
}

IRBlock* IRBuilder::emitBlock()
{
    auto block = createBlock();
    insertBlock(block);
    return block;
}

IRParam* IRBuilder::createParam(IRType* type)
{
    auto param = createInst<IRParam>(this, kIROp_Param, type);
    return param;
}

IRParam* IRBuilder::emitParam(IRType* type)
{
    auto param = createParam(type);
    if (auto bb = getBlock())
    {
        bb->addParam(param);
    }
    return param;
}

IRParam* IRBuilder::emitParamAtHead(IRType* type)
{
    auto param = createParam(type);
    if (auto bb = getBlock())
    {
        bb->insertParamAtHead(param);
    }
    return param;
}

IRInst* IRBuilder::emitAllocObj(IRType* type)
{
    return emitIntrinsicInst(type, kIROp_AllocObj, 0, nullptr);
}

IRVar* IRBuilder::emitVar(IRType* type)
{
    auto allocatedType = getPtrType(type);
    auto inst = createInst<IRVar>(this, kIROp_Var, allocatedType);
    addInst(inst);
    return inst;
}

IRVar* IRBuilder::emitVar(IRType* type, AddressSpace addressSpace)
{
    auto allocatedType = getPtrType(kIROp_PtrType, type, addressSpace);
    auto inst = createInst<IRVar>(this, kIROp_Var, allocatedType);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitLoadReverseGradient(IRType* type, IRInst* diffValue)
{
    auto inst = createInst<IRLoadReverseGradient>(this, kIROp_LoadReverseGradient, type, diffValue);

    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitReverseGradientDiffPairRef(IRType* type, IRInst* primalVar, IRInst* diffVar)
{
    auto inst = createInst<IRReverseGradientDiffPairRef>(
        this,
        kIROp_ReverseGradientDiffPairRef,
        type,
        primalVar,
        diffVar);

    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitPrimalParamRef(IRInst* param)
{
    auto type = param->getFullType();
    auto ptrType = as<IRPtrTypeBase>(type);
    auto valueType = type;
    if (ptrType)
        valueType = ptrType->getValueType();
    auto pairType = as<IRDifferentialPairType>(valueType);
    IRType* finalType = pairType->getValueType();
    if (ptrType)
        finalType = getPtrType(ptrType->getOp(), finalType);
    auto inst = createInst<IRPrimalParamRef>(this, kIROp_PrimalParamRef, finalType, param);

    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitDiffParamRef(IRType* type, IRInst* param)
{
    auto inst = createInst<IRDiffParamRef>(this, kIROp_DiffParamRef, type, param);

    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitLoad(IRType* type, IRInst* ptr)
{
    auto inst = createInst<IRLoad>(this, kIROp_Load, type, ptr);

    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitLoad(IRType* type, IRInst* ptr, IRInst* align)
{
    auto inst = createInst<IRLoad>(this, kIROp_Load, type, ptr, getAttr(kIROp_AlignedAttr, align));

    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitLoad(IRInst* ptr)
{
    // Note: a `load` operation does not consider the rate
    // (if any) attached to its operand (see the use of `getDataType`
    // below). This means that a load from a rate-qualified
    // variable will still conceptually execute (and return
    // results) at the "default" rate of the parent function,
    // unless a subsequent analysis pass constraints it.

    IRType* valueType = tryGetPointedToType(this, ptr->getFullType());
    SLANG_ASSERT(valueType);

    // Ugly special case: if the front-end created a variable with
    // type `Ptr<@R T>` instead of `@R Ptr<T>`, then the above
    // logic will yield `@R T` instead of `T`, and we need to
    // try and fix that up here.
    //
    // TODO: Lowering to the IR should be fixed to never create
    // that case: rate-qualified types should only be allowed
    // to appear as the type of an instruction, and should not
    // be allowed as operands to type constructors (except
    // in special cases we decide to allow).
    //
    if (auto rateType = as<IRRateQualifiedType>(valueType))
    {
        valueType = rateType->getValueType();
    }

    return emitLoad(valueType, ptr);
}

IRInst* IRBuilder::emitStore(IRInst* dstPtr, IRInst* srcVal)
{
    auto inst = createInst<IRStore>(this, kIROp_Store, nullptr, dstPtr, srcVal);

    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitStore(IRInst* dstPtr, IRInst* srcVal, IRInst* align)
{
    auto inst = createInst<IRStore>(
        this,
        kIROp_Store,
        nullptr,
        dstPtr,
        srcVal,
        getAttr(kIROp_AlignedAttr, align));

    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitAtomicStore(IRInst* dstPtr, IRInst* srcVal, IRInst* memoryOrder)
{
    auto inst = createInst<IRAtomicStore>(
        this,
        kIROp_AtomicStore,
        getVoidType(),
        dstPtr,
        srcVal,
        memoryOrder);

    addInst(inst);
    return inst;
}

/// @param params An ordered list of imageLoad parameters { image, coord, [optional]
/// seperateArrayCoord, [optional] seperateSampleCoord }
IRInst* IRBuilder::emitImageLoad(IRType* type, ShortList<IRInst*> params)
{
    auto inst = createInst<IRImageLoad>(
        this,
        kIROp_ImageLoad,
        type,
        params.getCount(),
        params.getArrayView().getBuffer());
    addInst(inst);
    return inst;
}

/// @param params An ordered list of imageStore parameters { image, coord, value, [optional]
/// seperateArrayCoord, [optional] seperateSampleCoord }
IRInst* IRBuilder::emitImageStore(IRType* type, ShortList<IRInst*> params)
{
    auto inst = createInst<IRImageStore>(
        this,
        kIROp_ImageStore,
        type,
        params.getCount(),
        params.getArrayView().getBuffer());
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitIsType(
    IRInst* value,
    IRInst* witness,
    IRInst* typeOperand,
    IRInst* targetWitness)
{
    IRInst* args[] = {value, witness, typeOperand, targetWitness};
    auto inst = createInst<IRIsType>(this, kIROp_IsType, getBoolType(), SLANG_COUNT_OF(args), args);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitFieldExtract(IRInst* base, IRInst* fieldKey)
{
    IRType* resultType = nullptr;
    auto valueType = base->getDataType();
    auto structType = as<IRStructType>(valueType);
    SLANG_RELEASE_ASSERT(structType);
    for (auto child : valueType->getChildren())
    {
        auto field = as<IRStructField>(child);
        if (!field)
            continue;
        if (field->getKey() == fieldKey)
        {
            resultType = field->getFieldType();
            break;
        }
    }
    SLANG_RELEASE_ASSERT(resultType);
    return emitFieldExtract(resultType, base, fieldKey);
}

IRInst* IRBuilder::emitFieldExtract(IRType* type, IRInst* base, IRInst* field)
{
    auto inst = createInst<IRFieldExtract>(this, kIROp_FieldExtract, type, base, field);

    addInst(inst);
    return inst;
}

IRType* maybePropagateAddressSpace(IRBuilder* builder, IRInst* basePtr, IRType* type)
{
    if (auto basePtrType = as<IRPtrTypeBase>(basePtr->getDataType()))
    {
        if (auto resultPtrType = as<IRPtrTypeBase>(type))
        {
            if (basePtrType->getAddressSpace() != resultPtrType->getAddressSpace())
            {
                type = builder->getPtrType(
                    resultPtrType->getOp(),
                    resultPtrType->getValueType(),
                    basePtrType->getAddressSpace());
            }
        }
    }
    return type;
}

IRInst* IRBuilder::emitFieldAddress(IRInst* basePtr, IRInst* fieldKey)
{
    AddressSpace addrSpace = AddressSpace::Generic;
    IRInst* valueType = nullptr;
    auto basePtrType = unwrapAttributedType(basePtr->getDataType());
    if (auto ptrType = as<IRPtrTypeBase>(basePtrType))
    {
        addrSpace = ptrType->getAddressSpace();
        valueType = ptrType->getValueType();
    }
    else if (auto ptrLikeType = as<IRPointerLikeType>(basePtrType))
    {
        valueType = ptrLikeType->getElementType();
    }
    IRType* resultType = nullptr;
    auto structType = as<IRStructType>(valueType);
    SLANG_RELEASE_ASSERT(structType);
    for (auto child : valueType->getChildren())
    {
        auto field = as<IRStructField>(child);
        if (!field)
            continue;
        if (field->getKey() == fieldKey)
        {
            resultType = field->getFieldType();
            break;
        }
    }
    SLANG_RELEASE_ASSERT(resultType);
    return emitFieldAddress(getPtrType(kIROp_PtrType, resultType, addrSpace), basePtr, fieldKey);
}

IRInst* IRBuilder::emitFieldAddress(IRType* type, IRInst* base, IRInst* field)
{
    // Propagate pointer address space if it is available on base.
    type = maybePropagateAddressSpace(this, base, type);

    auto inst = createInst<IRFieldAddress>(this, kIROp_FieldAddress, type, base, field);

    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitElementExtract(IRType* type, IRInst* base, IRInst* index)
{
    if (auto vectorFromScalar = as<IRMakeVectorFromScalar>(base))
        return vectorFromScalar->getOperand(0);
    if (base->getOp() == kIROp_MakeArrayFromElement)
        return base->getOperand(0);

    auto inst = createInst<IRGetElement>(this, kIROp_GetElement, type, base, index);

    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitElementExtract(IRInst* base, IRInst* index)
{
    IRType* type = nullptr;
    if (auto arrayType = as<IRArrayType>(base->getDataType()))
    {
        type = arrayType->getElementType();
    }
    else if (auto vectorType = as<IRVectorType>(base->getDataType()))
    {
        type = vectorType->getElementType();
    }
    else if (auto matrixType = as<IRMatrixType>(base->getDataType()))
    {
        type = getVectorType(matrixType->getElementType(), matrixType->getColumnCount());
    }
    else if (auto tupleType = as<IRTupleType>(base->getDataType()))
    {
        type = (IRType*)tupleType->getOperand(getIntVal(index));
        return emitGetTupleElement(type, base, index);
    }
    SLANG_RELEASE_ASSERT(type);

    return emitElementExtract(type, base, index);
}

IRInst* IRBuilder::emitElementExtract(IRInst* base, IRIntegerValue index)
{
    return emitElementExtract(base, getIntValue(getIntType(), index));
}

IRInst* IRBuilder::emitElementExtract(IRInst* base, const ArrayView<IRInst*>& accessChain)
{
    for (auto access : accessChain)
    {
        IRType* resultType = nullptr;
        if (auto structKey = as<IRStructKey>(access))
        {
            auto structType = as<IRStructType>(base->getDataType());
            SLANG_RELEASE_ASSERT(structType);
            for (auto field : structType->getFields())
            {
                if (field->getKey() == structKey)
                {
                    resultType = field->getFieldType();
                    break;
                }
            }
            SLANG_RELEASE_ASSERT(resultType);
            base = emitFieldExtract(resultType, base, structKey);
        }
        else
        {
            base = emitElementExtract(base, access);
        }
    }
    return base;
}

IRInst* IRBuilder::emitElementAddress(IRType* type, IRInst* basePtr, IRInst* index)
{
    // Propagate pointer address space if it is available on base.
    type = maybePropagateAddressSpace(this, basePtr, type);

    auto inst = createInst<IRFieldAddress>(this, kIROp_GetElementPtr, type, basePtr, index);

    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitElementAddress(IRInst* basePtr, IRIntegerValue index)
{
    return emitElementAddress(basePtr, getIntValue(getIntType(), index));
}

IRInst* IRBuilder::emitElementAddress(IRInst* basePtr, IRInst* index)
{
    AddressSpace addrSpace = AddressSpace::Generic;
    IRInst* valueType = nullptr;
    auto basePtrType = unwrapAttributedType(basePtr->getDataType());
    if (auto ptrType = as<IRPtrTypeBase>(basePtrType))
    {
        addrSpace = ptrType->getAddressSpace();
        valueType = ptrType->getValueType();
    }
    else if (auto ptrLikeType = as<IRPointerLikeType>(basePtrType))
    {
        valueType = ptrLikeType->getElementType();
    }
    IRType* type = nullptr;
    valueType = unwrapAttributedType(valueType);
    if (auto arrayType = as<IRArrayTypeBase>(valueType))
    {
        type = arrayType->getElementType();
    }
    else if (auto vectorType = as<IRVectorType>(valueType))
    {
        type = vectorType->getElementType();
    }
    else if (auto coopVecType = as<IRCoopVectorType>(valueType))
    {
        type = coopVecType->getElementType();
    }
    else if (auto matrixType = as<IRMatrixType>(valueType))
    {
        type = getVectorType(matrixType->getElementType(), matrixType->getColumnCount());
    }
    else if (auto coopMatType = as<IRCoopMatrixType>(valueType))
    {
        type = coopMatType->getElementType();
    }
    else if (const auto basicType = as<IRBasicType>(valueType))
    {
        // HLSL support things like float.x, in which case we just return the base pointer.
        return basePtr;
    }
    else if (const auto tupleType = as<IRTupleType>(valueType))
    {
        SLANG_ASSERT(as<IRIntLit>(index));
        type = (IRType*)tupleType->getOperand(getIntVal(index));
    }
    else if (auto hlslInputPatchType = as<IRHLSLInputPatchType>(valueType))
    {
        type = hlslInputPatchType->getElementType();
    }

    SLANG_RELEASE_ASSERT(type);
    auto inst = createInst<IRGetElementPtr>(
        this,
        kIROp_GetElementPtr,
        getPtrType(kIROp_PtrType, type, addrSpace),
        basePtr,
        index);

    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitElementAddress(IRInst* basePtr, const ArrayView<IRInst*>& accessChain)
{
    for (auto access : accessChain)
    {
        if (auto structKey = as<IRStructKey>(access))
        {
            basePtr = emitFieldAddress(basePtr, structKey);
        }
        else
        {
            basePtr = emitElementAddress(basePtr, access);
        }
    }
    return basePtr;
}

IRInst* IRBuilder::emitElementAddress(
    IRInst* basePtr,
    const ArrayView<IRInst*>& accessChain,
    const ArrayView<IRInst*>& types)
{
    for (Index i = 0; i < accessChain.getCount(); i++)
    {
        auto access = accessChain[i];
        auto type = (IRType*)types[i];
        if (auto structKey = as<IRStructKey>(access))
        {
            basePtr = emitFieldAddress(type, basePtr, structKey);
        }
        else
        {
            basePtr = emitElementAddress(type, basePtr, access);
        }
    }
    return basePtr;
}

IRInst* IRBuilder::emitUpdateElement(IRInst* base, IRInst* index, IRInst* newElement)
{
    auto inst = createInst<IRUpdateElement>(
        this,
        kIROp_UpdateElement,
        base->getFullType(),
        base,
        newElement,
        index);

    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitUpdateElement(IRInst* base, IRIntegerValue index, IRInst* newElement)
{
    return emitUpdateElement(base, getIntValue(getIntType(), index), newElement);
}

IRInst* IRBuilder::emitUpdateElement(
    IRInst* base,
    ArrayView<IRInst*> accessChain,
    IRInst* newElement)
{
    List<IRInst*> args;
    args.add(base);
    args.add(newElement);
    args.addRange(accessChain);
    auto inst = createInst<IRUpdateElement>(
        this,
        kIROp_UpdateElement,
        base->getFullType(),
        (Int)args.getCount(),
        args.getBuffer());
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitGetOffsetPtr(IRInst* base, IRInst* offset)
{
    IRInst* args[] = {base, offset};
    return emitIntrinsicInst(base->getDataType(), kIROp_GetOffsetPtr, 2, args);
}

IRInst* IRBuilder::emitGetAddress(IRType* type, IRInst* value)
{
    auto inst = createInst<IRGetAddress>(this, kIROp_GetAddr, type, value);

    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitSwizzle(
    IRType* type,
    IRInst* base,
    UInt elementCount,
    IRInst* const* elementIndices)
{
    auto inst = createInstWithTrailingArgs<IRSwizzle>(
        this,
        kIROp_swizzle,
        type,
        base,
        elementCount,
        elementIndices);

    addInst(inst);
    return inst;
}

IRInst* IRBuilder::addFloatingModeOverrideDecoration(IRInst* dest, FloatingPointMode mode)
{
    return addDecoration(
        dest,
        kIROp_FloatingPointModeOverrideDecoration,
        getIntValue(getIntType(), (IRIntegerValue)mode));
}

IRInst* IRBuilder::addNumThreadsDecoration(IRInst* inst, IRInst* x, IRInst* y, IRInst* z)
{
    IRInst* operands[3] = {x, y, z};

    return addDecoration(inst, kIROp_NumThreadsDecoration, operands, 3);
}

IRInst* IRBuilder::addWaveSizeDecoration(IRInst* inst, IRInst* numLanes)
{
    IRInst* operands[1] = {numLanes};

    return addDecoration(inst, kIROp_WaveSizeDecoration, operands, 1);
}

IRInst* IRBuilder::emitSwizzle(
    IRType* type,
    IRInst* base,
    UInt elementCount,
    uint64_t const* elementIndices)
{
    auto intType = getBasicType(BaseType::Int);

    IRInst* irElementIndices[4];
    for (UInt ii = 0; ii < elementCount; ++ii)
    {
        irElementIndices[ii] = getIntValue(intType, elementIndices[ii]);
    }

    return emitSwizzle(type, base, elementCount, irElementIndices);
}

IRInst* IRBuilder::emitSwizzle(
    IRType* type,
    IRInst* base,
    UInt elementCount,
    uint32_t const* elementIndices)
{
    auto intType = getBasicType(BaseType::Int);

    IRInst* irElementIndices[4];
    for (UInt ii = 0; ii < elementCount; ++ii)
    {
        irElementIndices[ii] = getIntValue(intType, elementIndices[ii]);
    }

    return emitSwizzle(type, base, elementCount, irElementIndices);
}

IRMetalSetVertex* IRBuilder::emitMetalSetVertex(IRInst* index, IRInst* vertex)
{
    auto inst =
        createInst<IRMetalSetVertex>(this, kIROp_MetalSetVertex, getVoidType(), index, vertex);
    addInst(inst);
    return inst;
}

IRMetalSetPrimitive* IRBuilder::emitMetalSetPrimitive(IRInst* index, IRInst* primitive)
{
    auto inst = createInst<IRMetalSetPrimitive>(
        this,
        kIROp_MetalSetVertex,
        getVoidType(),
        index,
        primitive);
    addInst(inst);
    return inst;
}

IRMetalSetIndices* IRBuilder::emitMetalSetIndices(IRInst* index, IRInst* indices)
{
    auto inst =
        createInst<IRMetalSetIndices>(this, kIROp_MetalSetVertex, getVoidType(), index, indices);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitSwizzleSet(
    IRType* type,
    IRInst* base,
    IRInst* source,
    UInt elementCount,
    IRInst* const* elementIndices)
{
    IRInst* fixedArgs[] = {base, source};
    UInt fixedArgCount = sizeof(fixedArgs) / sizeof(fixedArgs[0]);

    auto inst = createInstWithTrailingArgs<IRSwizzleSet>(
        this,
        kIROp_swizzleSet,
        type,
        fixedArgCount,
        fixedArgs,
        elementCount,
        elementIndices);

    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitSwizzleSet(
    IRType* type,
    IRInst* base,
    IRInst* source,
    UInt elementCount,
    uint32_t const* elementIndices)
{
    auto intType = getBasicType(BaseType::Int);

    IRInst* irElementIndices[4];
    for (UInt ii = 0; ii < elementCount; ++ii)
    {
        irElementIndices[ii] = getIntValue(intType, elementIndices[ii]);
    }

    return emitSwizzleSet(type, base, source, elementCount, irElementIndices);
}

IRInst* IRBuilder::emitSwizzleSet(
    IRType* type,
    IRInst* base,
    IRInst* source,
    UInt elementCount,
    uint64_t const* elementIndices)
{
    auto intType = getBasicType(BaseType::Int);

    IRInst* irElementIndices[4];
    for (UInt ii = 0; ii < elementCount; ++ii)
    {
        irElementIndices[ii] = getIntValue(intType, elementIndices[ii]);
    }

    return emitSwizzleSet(type, base, source, elementCount, irElementIndices);
}

IRInst* IRBuilder::emitSwizzledStore(
    IRInst* dest,
    IRInst* source,
    UInt elementCount,
    IRInst* const* elementIndices)
{
    IRInst* fixedArgs[] = {dest, source};
    UInt fixedArgCount = sizeof(fixedArgs) / sizeof(fixedArgs[0]);

    auto inst = createInstImpl<IRSwizzledStore>(
        this,
        kIROp_SwizzledStore,
        nullptr,
        fixedArgCount,
        fixedArgs,
        elementCount,
        elementIndices);

    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitSwizzledStore(
    IRInst* dest,
    IRInst* source,
    UInt elementCount,
    uint32_t const* elementIndices)
{
    auto intType = getBasicType(BaseType::Int);

    IRInst* irElementIndices[4];
    for (UInt ii = 0; ii < elementCount; ++ii)
    {
        irElementIndices[ii] = getIntValue(intType, elementIndices[ii]);
    }

    return emitSwizzledStore(dest, source, elementCount, irElementIndices);
}

IRInst* IRBuilder::emitSwizzledStore(
    IRInst* dest,
    IRInst* source,
    UInt elementCount,
    uint64_t const* elementIndices)
{
    auto intType = getBasicType(BaseType::Int);

    IRInst* irElementIndices[4];
    for (UInt ii = 0; ii < elementCount; ++ii)
    {
        irElementIndices[ii] = getIntValue(intType, elementIndices[ii]);
    }

    return emitSwizzledStore(dest, source, elementCount, irElementIndices);
}

IRInst* IRBuilder::emitReturn(IRInst* val)
{
    auto inst = createInst<IRReturn>(this, kIROp_Return, nullptr, val);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitYield(IRInst* val)
{
    auto inst = createInst<IRYield>(this, kIROp_Yield, nullptr, val);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitReturn()
{
    auto voidVal = getVoidValue();
    auto inst = createInst<IRReturn>(this, kIROp_Return, nullptr, voidVal);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitDefer(IRBlock* deferBlock, IRBlock* mergeBlock, IRBlock* scopeEndBlock)
{
    auto inst =
        createInst<IRDefer>(this, kIROp_Defer, nullptr, deferBlock, mergeBlock, scopeEndBlock);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitThrow(IRInst* val)
{
    auto inst = createInst<IRThrow>(this, kIROp_Throw, nullptr, val);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitUnreachable()
{
    auto inst = createInst<IRUnreachable>(this, kIROp_Unreachable, nullptr);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitMissingReturn()
{
    auto inst = createInst<IRMissingReturn>(this, kIROp_MissingReturn, nullptr);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitDiscard()
{
    auto inst = createInst<IRDiscard>(this, kIROp_discard, nullptr);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitCheckpointObject(IRInst* value)
{
    auto inst =
        createInst<IRCheckpointObject>(this, kIROp_CheckpointObject, value->getFullType(), value);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitLoopExitValue(IRInst* value)
{
    auto inst = createInst<IRLoopExitValue>(this, kIROp_LoopExitValue, value->getFullType(), value);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitBranch(IRBlock* pBlock)
{
    auto inst = createInst<IRUnconditionalBranch>(this, kIROp_unconditionalBranch, nullptr, pBlock);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitBranch(IRBlock* block, Int argCount, IRInst* const* args)
{
    List<IRInst*> argList;
    argList.add(block);
    for (Int i = 0; i < argCount; ++i)
        argList.add(args[i]);
    auto inst = createInst<IRUnconditionalBranch>(
        this,
        kIROp_unconditionalBranch,
        nullptr,
        argList.getCount(),
        argList.getBuffer());
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitBreak(IRBlock* target)
{
    return emitBranch(target);
}

IRInst* IRBuilder::emitContinue(IRBlock* target)
{
    return emitBranch(target);
}

IRInst* IRBuilder::emitLoop(IRBlock* target, IRBlock* breakBlock, IRBlock* continueBlock)
{
    IRInst* args[] = {target, breakBlock, continueBlock};
    UInt argCount = sizeof(args) / sizeof(args[0]);

    auto inst = createInst<IRLoop>(this, kIROp_loop, nullptr, argCount, args);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitLoop(
    IRBlock* target,
    IRBlock* breakBlock,
    IRBlock* continueBlock,
    Int argCount,
    IRInst* const* args)
{
    List<IRInst*> argList;

    argList.add(target);
    argList.add(breakBlock);
    argList.add(continueBlock);

    for (Count ii = 0; ii < argCount; ii++)
        argList.add(args[ii]);

    auto inst =
        createInst<IRLoop>(this, kIROp_loop, nullptr, argList.getCount(), argList.getBuffer());
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitBranch(IRInst* val, IRBlock* trueBlock, IRBlock* falseBlock)
{
    IRInst* args[] = {val, trueBlock, falseBlock};
    UInt argCount = sizeof(args) / sizeof(args[0]);

    auto inst =
        createInst<IRConditionalBranch>(this, kIROp_conditionalBranch, nullptr, argCount, args);
    addInst(inst);
    return inst;
}

IRIfElse* IRBuilder::emitIfElse(
    IRInst* val,
    IRBlock* trueBlock,
    IRBlock* falseBlock,
    IRBlock* afterBlock)
{
    IRInst* args[] = {val, trueBlock, falseBlock, afterBlock};
    UInt argCount = sizeof(args) / sizeof(args[0]);

    auto inst = createInst<IRIfElse>(this, kIROp_ifElse, nullptr, argCount, args);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitIfElseWithBlocks(
    IRInst* val,
    IRBlock*& outTrueBlock,
    IRBlock*& outFalseBlock,
    IRBlock*& outAfterBlock)
{
    outTrueBlock = createBlock();
    outAfterBlock = createBlock();
    outFalseBlock = createBlock();
    auto f = getFunc();
    SLANG_ASSERT(f);
    if (f)
    {
        f->addBlock(outTrueBlock);
        f->addBlock(outAfterBlock);
        f->addBlock(outFalseBlock);
    }
    auto result = emitIfElse(val, outTrueBlock, outFalseBlock, outAfterBlock);
    setInsertInto(outTrueBlock);
    return result;
}

IRInst* IRBuilder::emitIf(IRInst* val, IRBlock* trueBlock, IRBlock* afterBlock)
{
    return emitIfElse(val, trueBlock, afterBlock, afterBlock);
}

IRInst* IRBuilder::emitIfWithBlocks(IRInst* val, IRBlock*& outTrueBlock, IRBlock*& outAfterBlock)
{
    outTrueBlock = createBlock();
    outAfterBlock = createBlock();
    auto result = emitIf(val, outTrueBlock, outAfterBlock);
    insertBlock(outTrueBlock);
    insertBlock(outAfterBlock);
    setInsertInto(outTrueBlock);
    return result;
}

IRInst* IRBuilder::emitLoopTest(IRInst* val, IRBlock* bodyBlock, IRBlock* breakBlock)
{
    return emitIfElse(val, bodyBlock, breakBlock, bodyBlock);
}

IRInst* IRBuilder::emitSwitch(
    IRInst* val,
    IRBlock* breakLabel,
    IRBlock* defaultLabel,
    UInt caseArgCount,
    IRInst* const* caseArgs)
{
    IRInst* fixedArgs[] = {val, breakLabel, defaultLabel};
    UInt fixedArgCount = sizeof(fixedArgs) / sizeof(fixedArgs[0]);

    auto inst = createInstWithTrailingArgs<IRSwitch>(
        this,
        kIROp_Switch,
        nullptr,
        fixedArgCount,
        fixedArgs,
        caseArgCount,
        caseArgs);
    addInst(inst);
    return inst;
}

IRGlobalGenericParam* IRBuilder::emitGlobalGenericParam(IRType* type)
{
    IRGlobalGenericParam* irGenericParam =
        createInst<IRGlobalGenericParam>(this, kIROp_GlobalGenericParam, type);
    addGlobalValue(this, irGenericParam);
    return irGenericParam;
}

IRBindGlobalGenericParam* IRBuilder::emitBindGlobalGenericParam(IRInst* param, IRInst* val)
{
    auto inst = createInst<IRBindGlobalGenericParam>(
        this,
        kIROp_BindGlobalGenericParam,
        nullptr,
        param,
        val);
    addInst(inst);
    return inst;
}

IRDecoration* IRBuilder::addBindExistentialSlotsDecoration(
    IRInst* value,
    UInt argCount,
    IRInst* const* args)
{
    auto decoration = createInstWithTrailingArgs<IRDecoration>(
        this,
        kIROp_BindExistentialSlotsDecoration,
        getVoidType(),
        0,
        nullptr,
        argCount,
        args);

    decoration->insertAtStart(value);

    return decoration;
}

IRInst* IRBuilder::emitExtractTaggedUnionTag(IRInst* val)
{
    auto inst =
        createInst<IRInst>(this, kIROp_ExtractTaggedUnionTag, getBasicType(BaseType::UInt), val);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitExtractTaggedUnionPayload(IRType* type, IRInst* val, IRInst* tag)
{
    auto inst = createInst<IRInst>(this, kIROp_ExtractTaggedUnionPayload, type, val, tag);
    addInst(inst);
    return inst;
}


IRInst* IRBuilder::emitSizeOf(IRInst* sizedType)
{
    auto inst = createInst<IRInst>(this, kIROp_SizeOf, getIntType(), sizedType);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitAlignOf(IRInst* sizedType)
{
    auto inst = createInst<IRInst>(this, kIROp_AlignOf, getIntType(), sizedType);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitCountOf(IRType* type, IRInst* sizedType)
{
    auto inst = createInst<IRInst>(this, kIROp_CountOf, type, sizedType);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitBitCast(IRType* type, IRInst* val)
{
    auto inst = createInst<IRInst>(this, kIROp_BitCast, type, val);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitCastPtrToBool(IRInst* val)
{
    auto inst = createInst<IRInst>(this, kIROp_CastPtrToBool, getBoolType(), val);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitCastPtrToInt(IRInst* val)
{
    auto inst = createInst<IRInst>(this, kIROp_CastPtrToInt, getUInt64Type(), val);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitCastIntToPtr(IRType* ptrType, IRInst* val)
{
    auto inst = createInst<IRInst>(this, kIROp_CastIntToPtr, ptrType, val);
    addInst(inst);
    return inst;
}

IRGlobalConstant* IRBuilder::emitGlobalConstant(IRType* type)
{
    auto inst = createInst<IRGlobalConstant>(this, kIROp_GlobalConstant, type);
    addGlobalValue(this, inst);
    return inst;
}

IRGlobalConstant* IRBuilder::emitGlobalConstant(IRType* type, IRInst* val)
{
    auto inst = createInst<IRGlobalConstant>(this, kIROp_GlobalConstant, type, val);
    addGlobalValue(this, inst);
    return inst;
}

IRInst* IRBuilder::emitWaveMaskBallot(IRType* type, IRInst* mask, IRInst* condition)
{
    auto inst = createInst<IRInst>(this, kIROp_WaveMaskBallot, type, mask, condition);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitWaveMaskMatch(IRType* type, IRInst* mask, IRInst* value)
{
    auto inst = createInst<IRInst>(this, kIROp_WaveMaskMatch, type, mask, value);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitBitAnd(IRType* type, IRInst* left, IRInst* right)
{
    auto inst = createInst<IRInst>(this, kIROp_BitAnd, type, left, right);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitBitOr(IRType* type, IRInst* left, IRInst* right)
{
    auto inst = createInst<IRInst>(this, kIROp_BitOr, type, left, right);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitBitNot(IRType* type, IRInst* value)
{
    auto inst = createInst<IRInst>(this, kIROp_BitNot, type, value);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitNeg(IRType* type, IRInst* value)
{
    auto inst = createInst<IRInst>(this, kIROp_Neg, type, value);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitNot(IRType* type, IRInst* value)
{
    auto inst = createInst<IRInst>(this, kIROp_Not, type, value);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitAdd(IRType* type, IRInst* left, IRInst* right)
{
    auto inst = createInst<IRInst>(this, kIROp_Add, type, left, right);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitSub(IRType* type, IRInst* left, IRInst* right)
{
    auto inst = createInst<IRInst>(this, kIROp_Sub, type, left, right);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitEql(IRInst* left, IRInst* right)
{
    auto inst = createInst<IRInst>(this, kIROp_Eql, getBoolType(), left, right);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitNeq(IRInst* left, IRInst* right)
{
    auto inst = createInst<IRInst>(this, kIROp_Neq, getBoolType(), left, right);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitLess(IRInst* left, IRInst* right)
{
    auto inst = createInst<IRInst>(this, kIROp_Less, getBoolType(), left, right);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitGeq(IRInst* left, IRInst* right)
{
    auto inst = createInst<IRInst>(this, kIROp_Geq, getBoolType(), left, right);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitMul(IRType* type, IRInst* left, IRInst* right)
{
    auto inst = createInst<IRInst>(this, kIROp_Mul, type, left, right);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitDiv(IRType* type, IRInst* numerator, IRInst* denominator)
{
    auto inst = createInst<IRInst>(this, kIROp_Div, type, numerator, denominator);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitShr(IRType* type, IRInst* left, IRInst* right)
{
    auto inst = createInst<IRInst>(this, kIROp_Rsh, type, left, right);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitShl(IRType* type, IRInst* left, IRInst* right)
{
    auto inst = createInst<IRInst>(this, kIROp_Lsh, type, left, right);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitAnd(IRType* type, IRInst* left, IRInst* right)
{
    auto inst = createInst<IRInst>(this, kIROp_And, type, left, right);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitOr(IRType* type, IRInst* left, IRInst* right)
{
    auto inst = createInst<IRInst>(this, kIROp_Or, type, left, right);
    addInst(inst);
    return inst;
}

IRInst* IRBuilder::emitGetNativePtr(IRInst* value)
{
    auto valueType = value->getDataType();
    SLANG_RELEASE_ASSERT(valueType);
    switch (valueType->getOp())
    {
    case kIROp_InterfaceType:
        return emitIntrinsicInst(
            getNativePtrType((IRType*)valueType),
            kIROp_GetNativePtr,
            1,
            &value);
        break;
    case kIROp_ComPtrType:
        return emitIntrinsicInst(
            getNativePtrType((IRType*)valueType->getOperand(0)),
            kIROp_GetNativePtr,
            1,
            &value);
        break;
    case kIROp_ExtractExistentialType:
        return emitGetNativePtr(value->getOperand(0));
    default:
        SLANG_UNEXPECTED("invalid operand type for `getNativePtr`.");
        UNREACHABLE_RETURN(nullptr);
    }
}

IRInst* IRBuilder::emitManagedPtrAttach(IRInst* managedPtrVar, IRInst* value)
{
    IRInst* args[] = {managedPtrVar, value};
    return emitIntrinsicInst(getVoidType(), kIROp_ManagedPtrAttach, 2, args);
}

IRInst* IRBuilder::emitManagedPtrDetach(IRType* type, IRInst* managedPtrVal)
{
    return emitIntrinsicInst(type, kIROp_ManagedPtrDetach, 1, &managedPtrVal);
}

IRInst* IRBuilder::emitGetManagedPtrWriteRef(IRInst* ptrToManagedPtr)
{
    auto type = ptrToManagedPtr->getDataType();
    auto ptrType = as<IRPtrTypeBase>(type);
    SLANG_RELEASE_ASSERT(ptrType);
    auto managedPtrType = ptrType->getValueType();
    switch (managedPtrType->getOp())
    {
    case kIROp_InterfaceType:
        return emitIntrinsicInst(
            getPtrType(getNativePtrType((IRType*)managedPtrType)),
            kIROp_GetManagedPtrWriteRef,
            1,
            &ptrToManagedPtr);
        break;
    case kIROp_ComPtrType:
        return emitIntrinsicInst(
            getPtrType(getNativePtrType((IRType*)managedPtrType->getOperand(0))),
            kIROp_GetManagedPtrWriteRef,
            1,
            &ptrToManagedPtr);
        break;
    default:
        SLANG_UNEXPECTED("invalid operand type for `getNativePtr`.");
        UNREACHABLE_RETURN(nullptr);
    }
}

IRInst* IRBuilder::emitGpuForeach(List<IRInst*> args)
{
    auto inst = createInst<IRInst>(
        this,
        kIROp_GpuForeach,
        getVoidType(),
        args.getCount(),
        args.getBuffer());
    addInst(inst);
    return inst;
}

IRSPIRVAsmOperand* IRBuilder::emitSPIRVAsmOperandLiteral(IRInst* literal)
{
    SLANG_ASSERT(as<IRSPIRVAsm>(m_insertLoc.getParent()));
    const auto i = createInst<IRSPIRVAsmOperand>(
        this,
        kIROp_SPIRVAsmOperandLiteral,
        literal->getFullType(),
        literal);
    addInst(i);
    return i;
}

IRSPIRVAsmOperand* IRBuilder::emitSPIRVAsmOperandInst(IRInst* inst)
{
    SLANG_ASSERT(as<IRSPIRVAsm>(m_insertLoc.getParent()));
    const auto i =
        createInst<IRSPIRVAsmOperand>(this, kIROp_SPIRVAsmOperandInst, inst->getFullType(), inst);
    addInst(i);
    return i;
}

IRSPIRVAsmOperand* IRBuilder::createSPIRVAsmOperandInst(IRInst* inst)
{
    SLANG_ASSERT(as<IRSPIRVAsm>(m_insertLoc.getParent()));
    auto i =
        createInst<IRSPIRVAsmOperand>(this, kIROp_SPIRVAsmOperandInst, inst->getFullType(), inst);
    return i;
}
IRSPIRVAsmOperand* IRBuilder::emitSPIRVAsmOperandConvertTexel(IRInst* inst)
{
    SLANG_ASSERT(as<IRSPIRVAsm>(m_insertLoc.getParent()));
    auto i = createInst<IRSPIRVAsmOperand>(
        this,
        kIROp_SPIRVAsmOperandConvertTexel,
        inst->getFullType(),
        inst);
    addInst(i);
    return i;
}
IRSPIRVAsmOperand* IRBuilder::emitSPIRVAsmOperandRayPayloadFromLocation(IRInst* inst)
{
    SLANG_ASSERT(as<IRSPIRVAsm>(m_insertLoc.getParent()));
    auto i = createInst<IRSPIRVAsmOperand>(
        this,
        kIROp_SPIRVAsmOperandRayPayloadFromLocation,
        inst->getFullType(),
        inst);
    addInst(i);
    return i;
}
IRSPIRVAsmOperand* IRBuilder::emitSPIRVAsmOperandRayAttributeFromLocation(IRInst* inst)
{
    SLANG_ASSERT(as<IRSPIRVAsm>(m_insertLoc.getParent()));
    auto i = createInst<IRSPIRVAsmOperand>(
        this,
        kIROp_SPIRVAsmOperandRayAttributeFromLocation,
        inst->getFullType(),
        inst);
    addInst(i);
    return i;
}
IRSPIRVAsmOperand* IRBuilder::emitSPIRVAsmOperandRayCallableFromLocation(IRInst* inst)
{
    SLANG_ASSERT(as<IRSPIRVAsm>(m_insertLoc.getParent()));
    auto i = createInst<IRSPIRVAsmOperand>(
        this,
        kIROp_SPIRVAsmOperandRayCallableFromLocation,
        inst->getFullType(),
        inst);
    addInst(i);
    return i;
}
IRSPIRVAsmOperand* IRBuilder::emitSPIRVAsmOperandId(IRInst* inst)
{
    SLANG_ASSERT(as<IRSPIRVAsm>(m_insertLoc.getParent()));
    const auto i =
        createInst<IRSPIRVAsmOperand>(this, kIROp_SPIRVAsmOperandId, inst->getFullType(), inst);
    addInst(i);
    return i;
}

IRSPIRVAsmOperand* IRBuilder::emitSPIRVAsmOperandResult()
{
    SLANG_ASSERT(as<IRSPIRVAsm>(m_insertLoc.getParent()));
    const auto i = createInst<IRSPIRVAsmOperand>(this, kIROp_SPIRVAsmOperandResult, getVoidType());
    addInst(i);
    return i;
}

IRSPIRVAsmOperand* IRBuilder::emitSPIRVAsmOperandEnum(IRInst* inst)
{
    SLANG_ASSERT(as<IRSPIRVAsm>(m_insertLoc.getParent()));
    const auto i =
        createInst<IRSPIRVAsmOperand>(this, kIROp_SPIRVAsmOperandEnum, inst->getFullType(), inst);
    addInst(i);
    return i;
}

IRSPIRVAsmOperand* IRBuilder::emitSPIRVAsmOperandEnum(IRInst* inst, IRType* constantType)
{
    SLANG_ASSERT(as<IRSPIRVAsm>(m_insertLoc.getParent()));
    const auto i = createInst<IRSPIRVAsmOperand>(
        this,
        kIROp_SPIRVAsmOperandEnum,
        inst->getFullType(),
        inst,
        constantType);
    addInst(i);
    return i;
}

IRSPIRVAsmOperand* IRBuilder::emitSPIRVAsmOperandBuiltinVar(IRInst* type, IRInst* builtinKind)
{
    SLANG_ASSERT(as<IRSPIRVAsm>(m_insertLoc.getParent()));
    const auto i = createInst<IRSPIRVAsmOperand>(
        this,
        kIROp_SPIRVAsmOperandBuiltinVar,
        (IRType*)type,
        builtinKind);
    addInst(i);
    return i;
}

IRSPIRVAsmOperand* IRBuilder::emitSPIRVAsmOperandGLSL450Set()
{
    SLANG_ASSERT(as<IRSPIRVAsm>(m_insertLoc.getParent()));
    const auto i =
        createInst<IRSPIRVAsmOperand>(this, kIROp_SPIRVAsmOperandGLSL450Set, getVoidType());
    addInst(i);
    return i;
}

IRSPIRVAsmOperand* IRBuilder::emitSPIRVAsmOperandDebugPrintfSet()
{
    SLANG_ASSERT(as<IRSPIRVAsm>(m_insertLoc.getParent()));
    const auto i =
        createInst<IRSPIRVAsmOperand>(this, kIROp_SPIRVAsmOperandDebugPrintfSet, getVoidType());
    addInst(i);
    return i;
}

IRSPIRVAsmOperand* IRBuilder::emitSPIRVAsmOperandSampledType(IRType* elementType)
{
    SLANG_ASSERT(as<IRSPIRVAsm>(m_insertLoc.getParent()));
    const auto i = createInst<IRSPIRVAsmOperand>(
        this,
        kIROp_SPIRVAsmOperandSampledType,
        getTypeType(),
        elementType);
    addInst(i);
    return i;
}

IRSPIRVAsmOperand* IRBuilder::emitSPIRVAsmOperandImageType(IRInst* element)
{
    SLANG_ASSERT(as<IRSPIRVAsm>(m_insertLoc.getParent()));
    const auto i =
        createInst<IRSPIRVAsmOperand>(this, kIROp_SPIRVAsmOperandImageType, getTypeType(), element);
    addInst(i);
    return i;
}

IRSPIRVAsmOperand* IRBuilder::emitSPIRVAsmOperandSampledImageType(IRInst* element)
{
    SLANG_ASSERT(as<IRSPIRVAsm>(m_insertLoc.getParent()));
    const auto i = createInst<IRSPIRVAsmOperand>(
        this,
        kIROp_SPIRVAsmOperandSampledImageType,
        getTypeType(),
        element);
    addInst(i);
    return i;
}

IRSPIRVAsmOperand* IRBuilder::emitSPIRVAsmOperandTruncate()
{
    SLANG_ASSERT(as<IRSPIRVAsm>(m_insertLoc.getParent()));
    const auto i =
        createInst<IRSPIRVAsmOperand>(this, kIROp_SPIRVAsmOperandTruncate, getVoidType());
    addInst(i);
    return i;
}

IRSPIRVAsmOperand* IRBuilder::emitSPIRVAsmOperandEntryPoint()
{
    SLANG_ASSERT(as<IRSPIRVAsm>(m_insertLoc.getParent()));
    const auto i =
        createInst<IRSPIRVAsmOperand>(this, kIROp_SPIRVAsmOperandEntryPoint, getVoidType());
    addInst(i);
    return i;
}

IRSPIRVAsmInst* IRBuilder::emitSPIRVAsmInst(IRInst* opcode, List<IRInst*> operands)
{
    SLANG_ASSERT(as<IRSPIRVAsm>(m_insertLoc.getParent()));
    operands.insert(0, opcode);
    const auto i = createInst<IRSPIRVAsmInst>(
        this,
        kIROp_SPIRVAsmInst,
        getVoidType(),
        operands.getCount(),
        operands.getBuffer());
    addInst(i);
    return i;
}

IRSPIRVAsm* IRBuilder::emitSPIRVAsm(IRType* type)
{
    const auto asmInst = createInst<IRSPIRVAsm>(this, kIROp_SPIRVAsm, type);
    addInst(asmInst);
    return asmInst;
}

IRInst* IRBuilder::emitGenericAsm(UnownedStringSlice asmText)
{
    IRInst* arg = getStringValue(asmText);
    return emitIntrinsicInst(nullptr, kIROp_GenericAsm, 1, &arg);
}

IRInst* IRBuilder::emitRWStructuredBufferGetElementPtr(IRInst* structuredBuffer, IRInst* index)
{
    const auto sbt = cast<IRHLSLStructuredBufferTypeBase>(structuredBuffer->getDataType());
    const auto t = getPtrType(sbt->getElementType());
    IRInst* const operands[2] = {structuredBuffer, index};
    const auto i = createInst<IRRWStructuredBufferGetElementPtr>(
        this,
        kIROp_RWStructuredBufferGetElementPtr,
        t,
        2,
        operands);
    addInst(i);
    return i;
}

// IR emitter for a dedicated instruction to represent NonUniformResourceIndex qualifier.
IRInst* IRBuilder::emitNonUniformResourceIndexInst(IRInst* val)
{
    const auto i = createInst<IRInst>(this, kIROp_NonUniformResourceIndex, getTypeType(), val);
    addInst(i);
    return i;
}

//
// Decorations
//

IRDecoration* IRBuilder::addDecoration(
    IRInst* value,
    IROp op,
    IRInst* const* operands,
    Int operandCount)
{
    // If it's a simple (ie stateless) decoration, don't add it again.
    if (operandCount == 0 && isSimpleDecoration(op))
    {
        auto decoration = value->findDecorationImpl(op);
        if (decoration)
        {
            return decoration;
        }
    }

    auto decoration =
        createInstWithTrailingArgs<IRDecoration>(this, op, getVoidType(), operandCount, operands);

    // Decoration order should not, in general, be semantically
    // meaningful, so we will elect to insert a new decoration
    // at the start of an instruction (constant time) rather
    // than at the end of any existing list of deocrations
    // (which would take time linear in the number of decorations).
    //
    // TODO: revisit this if maintaining decoration ordering
    // from input source code is desirable.
    //
    decoration->insertAtStart(value);

    return decoration;
}


void IRBuilder::addHighLevelDeclDecoration(IRInst* inst, Decl* decl)
{
    auto ptrConst = getPtrValue(getPtrType(getVoidType()), decl);
    addDecoration(inst, kIROp_HighLevelDeclDecoration, ptrConst);
}

IRLayoutDecoration* IRBuilder::addLayoutDecoration(IRInst* value, IRLayout* layout)
{
    return as<IRLayoutDecoration>(addDecoration(value, kIROp_LayoutDecoration, layout));
}

IRTypeSizeAttr* IRBuilder::getTypeSizeAttr(LayoutResourceKind kind, LayoutSize size)
{
    auto kindInst = getIntValue(getIntType(), IRIntegerValue(kind));
    auto sizeInst = getIntValue(getIntType(), IRIntegerValue(size.raw));

    IRInst* operands[] = {kindInst, sizeInst};

    return cast<IRTypeSizeAttr>(
        createIntrinsicInst(getVoidType(), kIROp_TypeSizeAttr, SLANG_COUNT_OF(operands), operands));
}

IRVarOffsetAttr* IRBuilder::getVarOffsetAttr(LayoutResourceKind kind, UInt offset, UInt space)
{
    IRInst* operands[3];
    UInt operandCount = 0;

    auto kindInst = getIntValue(getIntType(), IRIntegerValue(kind));
    operands[operandCount++] = kindInst;

    auto offsetInst = getIntValue(getIntType(), IRIntegerValue(offset));
    operands[operandCount++] = offsetInst;

    if (space)
    {
        auto spaceInst = getIntValue(getIntType(), IRIntegerValue(space));
        operands[operandCount++] = spaceInst;
    }

    return cast<IRVarOffsetAttr>(
        createIntrinsicInst(getVoidType(), kIROp_VarOffsetAttr, operandCount, operands));
}

IRPendingLayoutAttr* IRBuilder::getPendingLayoutAttr(IRLayout* pendingLayout)
{
    IRInst* operands[] = {pendingLayout};

    return cast<IRPendingLayoutAttr>(createIntrinsicInst(
        getVoidType(),
        kIROp_PendingLayoutAttr,
        SLANG_COUNT_OF(operands),
        operands));
}

IRStructFieldLayoutAttr* IRBuilder::getFieldLayoutAttr(IRInst* key, IRVarLayout* layout)
{
    IRInst* operands[] = {key, layout};

    return cast<IRStructFieldLayoutAttr>(createIntrinsicInst(
        getVoidType(),
        kIROp_StructFieldLayoutAttr,
        SLANG_COUNT_OF(operands),
        operands));
}

IRTupleFieldLayoutAttr* IRBuilder::getTupleFieldLayoutAttr(IRTypeLayout* layout)
{
    IRInst* operands[] = {layout};

    return cast<IRTupleFieldLayoutAttr>(createIntrinsicInst(
        getVoidType(),
        kIROp_TupleFieldLayoutAttr,
        SLANG_COUNT_OF(operands),
        operands));
}

IRCaseTypeLayoutAttr* IRBuilder::getCaseTypeLayoutAttr(IRTypeLayout* layout)
{
    IRInst* operands[] = {layout};

    return cast<IRCaseTypeLayoutAttr>(createIntrinsicInst(
        getVoidType(),
        kIROp_CaseTypeLayoutAttr,
        SLANG_COUNT_OF(operands),
        operands));
}

IRSemanticAttr* IRBuilder::getSemanticAttr(IROp op, String const& name, UInt index)
{
    auto nameInst = getStringValue(name.getUnownedSlice());
    auto indexInst = getIntValue(getIntType(), index);

    IRInst* operands[] = {nameInst, indexInst};

    return cast<IRSemanticAttr>(
        createIntrinsicInst(getVoidType(), op, SLANG_COUNT_OF(operands), operands));
}

IRStageAttr* IRBuilder::getStageAttr(Stage stage)
{
    auto stageInst = getIntValue(getIntType(), IRIntegerValue(stage));
    IRInst* operands[] = {stageInst};
    return cast<IRStageAttr>(
        createIntrinsicInst(getVoidType(), kIROp_StageAttr, SLANG_COUNT_OF(operands), operands));
}

IRAttr* IRBuilder::getAttr(IROp op, UInt operandCount, IRInst* const* operands)
{
    return cast<IRAttr>(createIntrinsicInst(getVoidType(), op, operandCount, operands));
}


IRTypeLayout* IRBuilder::getTypeLayout(IROp op, List<IRInst*> const& operands)
{
    return cast<IRTypeLayout>(
        createIntrinsicInst(getVoidType(), op, operands.getCount(), operands.getBuffer()));
}

IRVarLayout* IRBuilder::getVarLayout(List<IRInst*> const& operands)
{
    return cast<IRVarLayout>(createIntrinsicInst(
        getVoidType(),
        kIROp_VarLayout,
        operands.getCount(),
        operands.getBuffer()));
}

IREntryPointLayout* IRBuilder::getEntryPointLayout(
    IRVarLayout* paramsLayout,
    IRVarLayout* resultLayout)
{
    IRInst* operands[] = {paramsLayout, resultLayout};

    return cast<IREntryPointLayout>(createIntrinsicInst(
        getVoidType(),
        kIROp_EntryPointLayout,
        SLANG_COUNT_OF(operands),
        operands));
}

//

struct IRDumpContext
{
    StringBuilder* builder = nullptr;
    int indent = 0;

    IRDumpOptions options;
    SourceManager* sourceManager;
    PathInfo lastPathInfo = PathInfo::makeUnknown();

    Dictionary<IRInst*, String> mapValueToName;
    Dictionary<String, UInt> uniqueNameCounters;
    UInt uniqueIDCounter = 1;
};

static void dump(IRDumpContext* context, char const* text)
{
    context->builder->append(text);
}

static void dump(IRDumpContext* context, String const& text)
{
    context->builder->append(text);
}

/*
static void dump(
    IRDumpContext*  context,
    UInt            val)
{
    context->builder->append(val);
}
*/

static void dump(IRDumpContext* context, IntegerLiteralValue val)
{
    context->builder->append(val);
}

static void dump(IRDumpContext* context, FloatingPointLiteralValue val)
{
    context->builder->append(val);
}

static void dumpIndent(IRDumpContext* context)
{
    for (int ii = 0; ii < context->indent; ++ii)
    {
        dump(context, "\t");
    }
}

bool opHasResult(IRInst* inst)
{
    auto type = inst->getDataType();
    if (!type)
        return false;

    // As a bit of a hack right now, we need to check whether
    // the function returns the distinguished `Void` type,
    // since that is conceptually the same as "not returning
    // a value."
    if (type->getOp() == kIROp_VoidType)
        return false;

    return true;
}

bool instHasUses(IRInst* inst)
{
    return inst->firstUse != nullptr;
}

static void scrubName(String const& name, StringBuilder& sb)
{
    // Note: this function duplicates a lot of the logic
    // in `EmitVisitor::scrubName`, so we should consider
    // whether they can share code at some point.
    //
    // There is no requirement that assembly dumps and output
    // code follow the same model, though, so this is just
    // a nice-to-have rather than a maintenance problem
    // waiting to happen.

    // Allow an empty name
    // Special case a name that is the empty string, just in case.
    if (name.getLength() == 0)
    {
        sb.append('_');
    }

    int prevChar = -1;
    for (auto c : name)
    {
        if (c == '.')
        {
            c = '_';
        }

        if (((c >= 'a') && (c <= 'z')) || ((c >= 'A') && (c <= 'Z')))
        {
            // Ordinary ASCII alphabetic characters are assumed
            // to always be okay.
        }
        else if ((c >= '0') && (c <= '9'))
        {
            // We don't want to allow a digit as the first
            // byte in a name.
            if (prevChar == -1)
            {
                sb.append('_');
            }
        }
        else
        {
            // If we run into a character that wouldn't normally
            // be allowed in an identifier, we need to translate
            // it into something that *is* valid.
            //
            // Our solution for now will be very clumsy: we will
            // emit `x` and then the hexadecimal version of
            // the byte we were given.
            sb.append("x");
            sb.append(uint32_t((unsigned char)c), 16);

            // We don't want to apply the default handling below,
            // so skip to the top of the loop now.
            prevChar = c;
            continue;
        }

        sb.append(c);
        prevChar = c;
    }

    // If the whole thing ended with a digit, then add
    // a final `_` just to make sure that we can append
    // a unique ID suffix without risk of collisions.
    if (('0' <= prevChar) && (prevChar <= '9'))
    {
        sb.append('_');
    }
}

static String createName(IRDumpContext* context, IRInst* value)
{
    if (auto nameHintDecoration = value->findDecoration<IRNameHintDecoration>())
    {
        String nameHint = nameHintDecoration->getName();

        StringBuilder sb;
        scrubName(nameHint, sb);

        String key = sb.produceString();
        UInt count = 0;
        context->uniqueNameCounters.tryGetValue(key, count);

        context->uniqueNameCounters[key] = count + 1;

        if (count)
        {
            sb.append(count);
        }
        return sb.produceString();
    }
    else
    {
        StringBuilder sb;
        auto id = context->uniqueIDCounter++;
        sb.append(id);
        return sb.produceString();
    }
}

static String getName(IRDumpContext* context, IRInst* value)
{
    String name;
    if (context->mapValueToName.tryGetValue(value, name))
        return name;

    name = createName(context, value);
    context->mapValueToName.add(value, name);
    return name;
}

static void dumpDebugID(IRDumpContext* context, IRInst* inst)
{
#if SLANG_ENABLE_IR_BREAK_ALLOC
    if (context->options.flags & IRDumpOptions::Flag::DumpDebugIds)
    {
        dump(context, "{");
        dump(context, String(inst->_debugUID));
        dump(context, "}\t");
    }
#else
    SLANG_UNUSED(context);
    SLANG_UNUSED(inst);
#endif
}

static void dumpID(IRDumpContext* context, IRInst* inst)
{
    if (!inst)
    {
        dump(context, "<null>");
        return;
    }

    if (opHasResult(inst) || instHasUses(inst))
    {
        dump(context, "%");
        dump(context, getName(context, inst));
    }
    else
    {
        dump(context, "_");
    }
}

static void dumpEncodeString(IRDumpContext* context, const UnownedStringSlice& slice)
{
    auto handler = StringEscapeUtil::getHandler(StringEscapeUtil::Style::Slang);
    StringBuilder& builder = *context->builder;
    StringEscapeUtil::appendQuoted(handler, slice, builder);
}

static void dumpType(IRDumpContext* context, IRType* type);

static bool shouldFoldInstIntoUses(IRDumpContext* context, IRInst* inst)
{
    // Never fold an instruction into its use site
    // in the "detailed" mode, so that we always
    // accurately reflect the structure of the IR.
    //
    if (context->options.mode == IRDumpOptions::Mode::Detailed)
        return false;

    if (as<IRConstant>(inst))
        return true;

    // We are going to have a general rule that
    // a type should be folded into its use site,
    // which improves output in most cases, but
    // we would like to not apply that rule to
    // "nominal" types like `struct`s.
    //
    switch (inst->getOp())
    {
    case kIROp_StructType:
    case kIROp_ClassType:
    case kIROp_GLSLShaderStorageBufferType:
    case kIROp_InterfaceType:
        return false;

    default:
        break;
    }

    if (as<IRType>(inst))
        return true;

    if (as<IRSPIRVAsmOperand>(inst))
        return true;

    return false;
}

static void dumpInst(IRDumpContext* context, IRInst* inst);

static void dumpInstBody(IRDumpContext* context, IRInst* inst);

static void dumpInstExpr(IRDumpContext* context, IRInst* inst);

static void dumpOperand(IRDumpContext* context, IRInst* inst)
{
    // TODO: we should have a dedicated value for the `undef` case
    if (!inst)
    {
        dumpID(context, inst);
        return;
    }

    if (shouldFoldInstIntoUses(context, inst))
    {
        dumpInstExpr(context, inst);
        return;
    }

    dumpID(context, inst);
}

static void dumpType(IRDumpContext* context, IRType* type)
{
    if (!type)
    {
        dump(context, "_");
        return;
    }

    // TODO: we should consider some special-case printing
    // for types, so that the IR doesn't get too hard to read
    // (always having to back-reference for what a type expands to)
    dumpOperand(context, type);
}

static void dumpInstTypeClause(IRDumpContext* context, IRType* type)
{
    dump(context, "\t: ");
    dumpType(context, type);
}

void dumpIRDecorations(IRDumpContext* context, IRInst* inst)
{
    for (auto dd : inst->getDecorations())
    {
        dump(context, "[");
        dumpInstBody(context, dd);
        dump(context, "]\n");

        dumpIndent(context);
    }
}

static void dumpBlock(IRDumpContext* context, IRBlock* block)
{
    context->indent--;
    dumpDebugID(context, block);
    dump(context, "block ");
    dumpID(context, block);

    IRInst* inst = block->getFirstInst();

    // First walk through any `param` instructions,
    // so that we can format them nicely
    if (auto firstParam = as<IRParam, IRDynamicCastBehavior::NoUnwrap>(inst))
    {
        dump(context, "(\n");
        context->indent += 2;

        for (;;)
        {
            auto param = as<IRParam, IRDynamicCastBehavior::NoUnwrap>(inst);
            if (!param)
                break;

            if (param != firstParam)
                dump(context, ",\n");

            inst = inst->getNextInst();

            dumpIndent(context);
            dumpIRDecorations(context, param);
            dump(context, "param ");
            dumpID(context, param);
            dumpInstTypeClause(context, param->getFullType());
        }
        context->indent -= 2;
        dump(context, ")");
    }
    dump(context, ":\n");
    context->indent++;

    for (; inst; inst = inst->getNextInst())
    {
        dumpInst(context, inst);
    }
}

void dumpIRGlobalValueWithCode(IRDumpContext* context, IRGlobalValueWithCode* code)
{
    auto opInfo = getIROpInfo(code->getOp());

    dumpIndent(context);
    dump(context, opInfo.name);
    dump(context, " ");
    dumpID(context, code);

    dumpInstTypeClause(context, code->getFullType());

    if (!code->getFirstBlock())
    {
        // Just a declaration.
        dump(context, ";\n");
        return;
    }

    dump(context, "\n");

    dumpIndent(context);
    dump(context, "{\n");
    context->indent++;

    for (auto bb = code->getFirstBlock(); bb; bb = bb->getNextBlock())
    {
        if (bb != code->getFirstBlock())
            dump(context, "\n");
        dumpBlock(context, bb);
    }

    context->indent--;
    dump(context, "}");
}

static void dumpInstOperandList(IRDumpContext* context, IRInst* inst)
{
    UInt argCount = inst->getOperandCount();

    if (argCount == 0)
        return;

    UInt ii = 0;

    // Special case: make printing of `call` a bit
    // nicer to look at
    if (inst->getOp() == kIROp_Call && argCount > 0)
    {
        dump(context, " ");
        auto argVal = inst->getOperand(ii++);
        dumpOperand(context, argVal);
    }

    bool first = true;
    dump(context, "(");
    for (; ii < argCount; ++ii)
    {
        if (!first)
            dump(context, ", ");

        auto argVal = inst->getOperand(ii);

        dumpOperand(context, argVal);

        first = false;
    }

    dump(context, ")");
}

void dumpIRWitnessTableEntry(IRDumpContext* context, IRWitnessTableEntry* entry)
{
    dump(context, "witness_table_entry(");
    dumpOperand(context, entry->requirementKey.get());
    dump(context, ",");
    dumpOperand(context, entry->satisfyingVal.get());
    dump(context, ")\n");
}

void dumpIRParentInst(IRDumpContext* context, IRInst* inst)
{
    auto opInfo = getIROpInfo(inst->getOp());

    dump(context, opInfo.name);
    dump(context, " ");
    dumpID(context, inst);

    dumpInstTypeClause(context, inst->getFullType());

    dumpInstOperandList(context, inst);

    if (!inst->getFirstChild())
    {
        // Empty.
        dump(context, ";\n");
        return;
    }

    dump(context, "\n");

    dumpIndent(context);
    dump(context, "{\n");
    context->indent++;

    for (auto child : inst->getChildren())
    {
        dumpInst(context, child);
    }

    context->indent--;
    dumpIndent(context);
    dump(context, "}\n");
}

void dumpIRGeneric(IRDumpContext* context, IRGeneric* witnessTable)
{
    dump(context, "\n");
    dumpIndent(context);
    dump(context, "ir_witness_table ");
    dumpID(context, witnessTable);
    dump(context, "\n{\n");
    context->indent++;

    for (auto ii : witnessTable->getChildren())
    {
        dumpInst(context, ii);
    }

    context->indent--;
    dump(context, "}\n");
}

static void dumpEmbeddedDownstream(IRDumpContext* context, IRInst* inst)
{
    auto targetInst = inst->getOperand(0);
    auto blobInst = inst->getOperand(1);

    // Get the target value
    auto targetLit = as<IRIntLit>(targetInst);
    if (!targetLit)
    {
        dump(context, "EmbeddedDownstreamIR(invalid target)");
        return;
    }

    // Get the blob
    auto blobLitInst = as<IRBlobLit>(blobInst);
    if (!blobLitInst)
    {
        dump(context, "EmbeddedDownstreamIR(invalid blob)");
        return;
    }

    dump(context, "EmbeddedDownstreamIR(");
    dump(context, targetLit->getValue());
    dump(context, " : Int, ");

    // If target is SPIR-V (6), disassemble the blob
    if (targetLit->getValue() == (IRIntegerValue)CodeGenTarget::SPIRV)
    {
        auto blob = blobLitInst->getStringSlice();
        const uint32_t* spirvCode = (const uint32_t*)blob.begin();
        const size_t spirvWordCount = blob.getLength() / sizeof(uint32_t);

        // Get the compiler from the session through the module
        auto module = inst->getModule();
        auto session = module->getSession();
        IDownstreamCompiler* compiler =
            session->getOrLoadDownstreamCompiler(PassThroughMode::SpirvDis, nullptr);

        if (compiler)
        {
            // Use glslang interface to disassemble with string output
            String disassemblyOutput;
            if (SLANG_SUCCEEDED(compiler->disassembleWithResult(
                    spirvCode,
                    int(spirvWordCount),
                    disassemblyOutput)))
            {
                // Dump the captured disassembly
                dump(context, "\n");
                dumpIndent(context);
                dump(context, disassemblyOutput);
            }
            else
            {
                dump(context, "<disassembly failed>");
            }
        }
        else
        {
            dump(context, "<unavailable disassembler>");
        }
    }
    else
    {
        // TODO: Add DXIL disassembly call here.
        dump(context, "<binary blob>");
    }
    dump(context, ")");
}

static void dumpInstExpr(IRDumpContext* context, IRInst* inst)
{
    if (!inst)
    {
        dump(context, "<null>");
        return;
    }

    auto op = inst->getOp();
    auto opInfo = getIROpInfo(op);

    // Special-case the literal instructions.
    if (auto irConst = as<IRConstant>(inst))
    {
        switch (op)
        {
        case kIROp_IntLit:
            dump(context, irConst->value.intVal);
            dump(context, " : ");
            dumpType(context, irConst->getFullType());
            return;

        case kIROp_FloatLit:
            dump(context, irConst->value.floatVal);
            dump(context, " : ");
            dumpType(context, irConst->getFullType());
            return;

        case kIROp_BoolLit:
            dump(context, irConst->value.intVal ? "true" : "false");
            return;

        case kIROp_BlobLit:
            dump(context, "<binary blob>");
            return;

        case kIROp_StringLit:
            dumpEncodeString(context, irConst->getStringSlice());
            return;

        case kIROp_PtrLit:
            dump(context, "<ptr>");
            return;

        default:
            break;
        }
    }

    // Special case EmbeddedDownstreamIR to show SPIR-V disassembly
    if (op == kIROp_EmbeddedDownstreamIR)
    {
        dumpEmbeddedDownstream(context, inst);
        return;
    }

    // Special case the SPIR-V asm operands as the distinction here is
    // clear anyway to the user
    switch (op)
    {
    case kIROp_SPIRVAsmOperandEnum:
        dumpInstExpr(context, inst->getOperand(0));
        return;
    case kIROp_SPIRVAsmOperandLiteral:
        dumpInstExpr(context, inst->getOperand(0));
        return;
    case kIROp_SPIRVAsmOperandInst:
        dumpInstExpr(context, inst->getOperand(0));
        return;
    case kIROp_SPIRVAsmOperandRayPayloadFromLocation:
        dump(context, "__rayPayloadFromLocation(");
        dumpInstExpr(context, inst->getOperand(0));
        dump(context, ")");
        return;
    case kIROp_SPIRVAsmOperandRayAttributeFromLocation:
        dump(context, "__rayAttributeFromLocation(");
        dumpInstExpr(context, inst->getOperand(0));
        dump(context, ")");
        return;
    case kIROp_SPIRVAsmOperandRayCallableFromLocation:
        dump(context, "__rayCallableFromLocation(");
        dumpInstExpr(context, inst->getOperand(0));
        dump(context, ")");
        return;
    case kIROp_SPIRVAsmOperandId:
        dump(context, "%");
        dumpInstExpr(context, inst->getOperand(0));
        return;
    case kIROp_SPIRVAsmOperandResult:
        dump(context, "result");
        return;
    case kIROp_SPIRVAsmOperandTruncate:
        dump(context, "__truncate");
        return;
    case kIROp_SPIRVAsmOperandSampledType:
        dump(context, "__sampledType(");
        dumpInstExpr(context, inst->getOperand(0));
        dump(context, ")");
        return;
    case kIROp_SPIRVAsmOperandImageType:
        dump(context, "__imageType(");
        dumpInstExpr(context, inst->getOperand(0));
        dump(context, ")");
        return;
    case kIROp_SPIRVAsmOperandSampledImageType:
        dump(context, "__sampledImageType(");
        dumpInstExpr(context, inst->getOperand(0));
        dump(context, ")");
        return;
    }

    dump(context, opInfo.name);
    dumpInstOperandList(context, inst);
}

static void dumpInstBody(IRDumpContext* context, IRInst* inst)
{
    if (!inst)
    {
        dump(context, "<null>");
        return;
    }

    auto op = inst->getOp();

    dumpIRDecorations(context, inst);

    dumpDebugID(context, inst);

    // There are several ops we want to special-case here,
    // so that they will be more pleasant to look at.
    //
    switch (op)
    {
    case kIROp_Func:
    case kIROp_GlobalVar:
    case kIROp_Generic:
    case kIROp_Expand:
        dumpIRGlobalValueWithCode(context, (IRGlobalValueWithCode*)inst);
        return;

    case kIROp_WitnessTable:
    case kIROp_StructType:
    case kIROp_ClassType:
    case kIROp_GLSLShaderStorageBufferType:
    case kIROp_SPIRVAsm:
        dumpIRParentInst(context, inst);
        return;

    case kIROp_WitnessTableEntry:
        dumpIRWitnessTableEntry(context, (IRWitnessTableEntry*)inst);
        return;

    default:
        break;
    }

    // Okay, we have a seemingly "ordinary" op now
    auto dataType = inst->getDataType();
    auto rate = inst->getRate();

    if (rate)
    {
        dump(context, "@");
        dumpOperand(context, rate);
        dump(context, " ");
    }

    if (opHasResult(inst) || instHasUses(inst))
    {
        dump(context, "let  ");
        dumpID(context, inst);
        dumpInstTypeClause(context, dataType);
        dump(context, "\t= ");
    }
    else
    {
        // No result, okay...
    }

    dumpInstExpr(context, inst);
}

static void dumpInst(IRDumpContext* context, IRInst* inst)
{
    if (shouldFoldInstIntoUses(context, inst))
        return;

    dumpIndent(context);
    dumpInstBody(context, inst);

    // Output the originating source location
    {
        SourceManager* sourceManager = context->sourceManager;
        if (sourceManager && context->options.flags & IRDumpOptions::Flag::SourceLocations)
        {
            StringBuilder buf;
            buf << " loc: ";

            // Output the line number information
            if (inst->sourceLoc.isValid())
            {
                // Might want to output actual, but nominal is okay for default
                const SourceLocType sourceLocType = SourceLocType::Nominal;

                // Get the source location
                const HumaneSourceLoc humaneLoc =
                    sourceManager->getHumaneLoc(inst->sourceLoc, sourceLocType);
                if (humaneLoc.line >= 0)
                {
                    buf << humaneLoc.line << "," << humaneLoc.column;

                    if (humaneLoc.pathInfo != context->lastPathInfo)
                    {
                        buf << " ";
                        // Output the the location
                        humaneLoc.pathInfo.appendDisplayName(buf);
                        context->lastPathInfo = humaneLoc.pathInfo;
                    }
                }
                else
                {
                    buf << "not found";
                }
            }
            else
            {
                buf << "na";
            }

            dump(context, buf.getUnownedSlice());
        }
    }

    dump(context, "\n");
}

void dumpIRModule(IRDumpContext* context, IRModule* module)
{
    for (auto ii : module->getGlobalInsts())
    {
        dumpInst(context, ii);
    }
}

void printSlangIRAssembly(
    StringBuilder& builder,
    IRModule* module,
    const IRDumpOptions& options,
    SourceManager* sourceManager)
{
    IRDumpContext context;
    context.builder = &builder;
    context.indent = 0;
    context.options = options;
    context.sourceManager = sourceManager;

    dumpIRModule(&context, module);
}

void dumpIR(
    IRInst* globalVal,
    const IRDumpOptions& options,
    SourceManager* sourceManager,
    ISlangWriter* writer)
{
    StringBuilder sb;

    IRDumpContext context;
    context.builder = &sb;
    context.indent = 0;
    context.options = options;
    context.sourceManager = sourceManager;

    if (globalVal->getOp() == kIROp_Module)
        dumpIRModule(&context, globalVal->getModule());
    else
        dumpInst(&context, globalVal);

    writer->write(sb.getBuffer(), sb.getLength());
    writer->flush();
}

void dumpIR(
    IRModule* module,
    const IRDumpOptions& options,
    char const* label,
    SourceManager* sourceManager,
    ISlangWriter* inWriter)
{
    WriterHelper writer(inWriter);

    if (label)
    {
        writer.put("### ");
        writer.put(label);
        writer.put(":\n");
    }

    dumpIR(module, options, sourceManager, inWriter);

    if (label)
    {
        writer.put("###\n");
    }
}

String getSlangIRAssembly(
    IRModule* module,
    const IRDumpOptions& options,
    SourceManager* sourceManager)
{
    StringBuilder sb;
    printSlangIRAssembly(sb, module, options, sourceManager);
    return sb;
}

void dumpIR(
    IRModule* module,
    const IRDumpOptions& options,
    SourceManager* sourceManager,
    ISlangWriter* writer)
{
    String ir = getSlangIRAssembly(module, options, sourceManager);
    writer->write(ir.getBuffer(), ir.getLength());
    writer->flush();
}

// Pre-declare
static bool _isTypeOperandEqual(IRInst* a, IRInst* b);

static bool _areTypeOperandsEqual(IRInst* a, IRInst* b)
{
    // Must have same number of operands
    const auto operandCountA = Index(a->getOperandCount());
    if (operandCountA != Index(b->getOperandCount()))
    {
        return false;
    }

    // All the operands must be equal
    for (Index i = 0; i < operandCountA; ++i)
    {
        IRInst* operandA = a->getOperand(i);
        IRInst* operandB = b->getOperand(i);

        if (!_isTypeOperandEqual(operandA, operandB))
        {
            return false;
        }
    }

    return true;
}

bool isNominalOp(IROp op)
{
    // True if the op identity is 'nominal'
    switch (op)
    {
    case kIROp_StructType:
    case kIROp_ClassType:
    case kIROp_GLSLShaderStorageBufferType:
    case kIROp_InterfaceType:
    case kIROp_Generic:
    case kIROp_Param:
        {
            return true;
        }
    }
    return false;
}

// True if a type operand is equal. Operands are 'IRInst' - but it's only a restricted set that
// can be operands of IRType instructions
static bool _isTypeOperandEqual(IRInst* a, IRInst* b)
{
    if (a == b)
    {
        return true;
    }

    if (a == nullptr || b == nullptr)
    {
        return false;
    }

    const IROp opA = IROp(a->getOp() & kIROpMask_OpMask);
    const IROp opB = IROp(b->getOp() & kIROpMask_OpMask);

    if (opA != opB)
    {
        return false;
    }

    // If the type is nominal - it can only be the same if the pointer is the same.
    if (isNominalOp(opA))
    {
        // The pointer isn't the same (as that was already tested), so cannot be equal
        return false;
    }

    // Both are types
    if (IRType::isaImpl(opA))
    {
        if (IRBasicType::isaImpl(opA))
        {
            // If it's a basic type, then their op being the same means we are done
            return true;
        }

        // We don't care about the parent or positioning
        // We also don't care about 'type' - because these instructions are defining the type.
        //
        // We may want to care about decorations.

        // TODO(JS): There is a question here about what to do about decorations.
        // For now we ignore decorations. Are two types potentially different if there decorations
        // different? If decorations play a part in difference in types - the order of decorations
        // presumably is not important.

        // All the operands of the types must be equal
        return _areTypeOperandsEqual(a, b);
    }

    // If it's a constant...
    if (IRConstant::isaImpl(opA))
    {
        // TODO: This is contrived in that we want two types that are the same, but are different
        // pointers to match here.
        // If we make getHashCode for IRType* compatible with isTypeEqual, then we should probably
        // use that.
        return static_cast<IRConstant*>(a)->isValueEqual(static_cast<IRConstant*>(b)) &&
               isTypeEqual(a->getFullType(), b->getFullType());
    }
    if (IRSpecialize::isaImpl(opA) || opA == kIROp_LookupWitness)
    {
        return _areTypeOperandsEqual(a, b);
    }
    // We can't equate any other type..
    return false;
}

bool isTypeEqual(IRType* a, IRType* b)
{
    // _isTypeOperandEqual handles comparison of types so can defer to it
    return _isTypeOperandEqual(a, b);
}

bool isIntegralType(IRType* t)
{
    if (auto basic = as<IRBasicType>(t))
    {
        switch (basic->getBaseType())
        {
        case BaseType::Int8:
        case BaseType::Int16:
        case BaseType::Int:
        case BaseType::Int64:
        case BaseType::UInt8:
        case BaseType::UInt16:
        case BaseType::UInt:
        case BaseType::UInt64:
        case BaseType::IntPtr:
        case BaseType::UIntPtr:
            return true;
        default:
            return false;
        }
    }
    return false;
}

bool isFloatingType(IRType* t)
{
    if (auto basic = as<IRBasicType>(t))
    {
        switch (basic->getBaseType())
        {
        case BaseType::Float:
        case BaseType::Half:
        case BaseType::Double:
            return true;
        default:
            return false;
        }
    }
    return false;
}

IntInfo getIntTypeInfo(const IRType* intType)
{
    switch (intType->getOp())
    {
    case kIROp_UInt8Type:
        return {8, false};
    case kIROp_UInt16Type:
        return {16, false};
    case kIROp_UIntType:
        return {32, false};
    case kIROp_UInt64Type:
        return {64, false};
    case kIROp_Int8Type:
        return {8, true};
    case kIROp_Int16Type:
        return {16, true};
    case kIROp_IntType:
        return {32, true};
    case kIROp_Int64Type:
        return {64, true};

    case kIROp_IntPtrType:  // target platform dependent
    case kIROp_UIntPtrType: // target platform dependent
    default:
        SLANG_UNEXPECTED("Unhandled type passed to getIntTypeInfo");
    }
}

IROp getIntTypeOpFromInfo(const IntInfo info)
{
    switch (info.width)
    {
    case 8:
        return info.isSigned ? kIROp_Int8Type : kIROp_UInt8Type;
    case 16:
        return info.isSigned ? kIROp_Int16Type : kIROp_UInt16Type;
    case 32:
        return info.isSigned ? kIROp_IntType : kIROp_UIntType;
    case 64:
        return info.isSigned ? kIROp_Int64Type : kIROp_UInt64Type;
    default:
        SLANG_UNEXPECTED("Unhandled info passed to getIntTypeOpFromInfo");
    }
}

FloatInfo getFloatingTypeInfo(const IRType* floatType)
{
    switch (floatType->getOp())
    {
    case kIROp_HalfType:
        return {16};
    case kIROp_FloatType:
        return {32};
    case kIROp_DoubleType:
        return {64};
    default:
        SLANG_UNEXPECTED("Unhandled type passed to getFloatTypeInfo");
    }
}

bool isIntegralScalarOrCompositeType(IRType* t)
{
    if (!t)
        return false;
    switch (t->getOp())
    {
    case kIROp_VectorType:
    case kIROp_MatrixType:
        return isIntegralType((IRType*)t->getOperand(0));
    default:
        return isIntegralType(t);
    }
}

IRStructField* findStructField(IRInst* type, IRStructKey* key)
{
    if (auto irStructType = as<IRStructType>(type))
    {
        for (auto field : irStructType->getFields())
        {
            if (field->getKey() == key)
            {
                return field;
            }
        }
    }
    else if (auto irSpecialize = as<IRSpecialize>(type))
    {
        if (auto irGeneric = as<IRGeneric>(irSpecialize->getBase()))
        {
            if (auto irGenericStructType =
                    as<IRStructType>(findInnerMostGenericReturnVal(irGeneric)))
            {
                return findStructField(irGenericStructType, key);
            }
        }
    }

    return nullptr;
}

void findAllInstsBreadthFirst(IRInst* inst, List<IRInst*>& outInsts)
{
    Index index = outInsts.getCount();

    outInsts.add(inst);

    while (index < outInsts.getCount())
    {
        IRInst* cur = outInsts[index++];

        IRInstListBase childrenList = cur->getDecorationsAndChildren();
        for (IRInst* child : childrenList)
        {
            outInsts.add(child);
        }
    }
}

IRDecoration* IRInst::getFirstDecoration()
{
    return as<IRDecoration>(getFirstDecorationOrChild());
}

IRDecoration* IRInst::getLastDecoration()
{
    IRDecoration* decoration = getFirstDecoration();
    if (!decoration)
        return nullptr;

    while (auto nextDecoration = decoration->getNextDecoration())
        decoration = nextDecoration;

    return decoration;
}

IRInstList<IRDecoration> IRInst::getDecorations()
{
    return IRInstList<IRDecoration>(getFirstDecoration(), getLastDecoration());
}

IRInst* IRInst::getFirstChild()
{
    // The children come after any decorations,
    // so if there are any decorations, then the
    // first child is right after the last decoration.
    //
    if (auto lastDecoration = getLastDecoration())
        return lastDecoration->getNextInst();
    //
    // Otherwise, there must be no decorations, so
    // that the first "child or decoration" is a child.
    //
    return getFirstDecorationOrChild();
}

IRInst* IRInst::getLastChild()
{
    // The children come after any decorations, so
    // that the last item in the list of children
    // and decorations is the last child *unless*
    // it is a decoration, in which case there are
    // no children.
    //
    auto lastChild = getLastDecorationOrChild();
    return as<IRDecoration>(lastChild) ? nullptr : lastChild;
}


IRRate* IRInst::getRate()
{
    if (auto rateQualifiedType = as<IRRateQualifiedType>(getFullType()))
        return rateQualifiedType->getRate();

    return nullptr;
}

IRType* IRInst::getDataType()
{
    auto type = getFullType();
    if (auto rateQualifiedType = as<IRRateQualifiedType>(type))
        return rateQualifiedType->getValueType();

    return type;
}

void validateIRInstOperands(IRInst*);


// Returns true if `instToCheck` is defined after `otherInst`.
static bool _isInstDefinedAfter(IRInst* instToCheck, IRInst* otherInst)
{
    for (auto inst = otherInst->getNextInst(); inst; inst = inst->getNextInst())
    {
        if (inst == instToCheck)
            return true;
    }
    return false;
}

static void _maybeHoistOperand(IRUse* use)
{
    ShortList<IRUse*, 16> workList1, workList2;
    workList1.add(use);
    while (workList1.getCount())
    {
        for (auto item : workList1)
        {
            auto user = item->getUser();
            auto operand = item->get();
            if (!operand)
                continue;

            if (!getIROpInfo(operand->getOp()).isHoistable())
                continue;

            // We can't handle the case where operand and user are in different blocks.
            if (operand->getParent() != user->getParent())
                continue;

            // We allow out-of-order uses in global scope.
            if (operand->getParent() && operand->getParent()->getOp() == kIROp_Module)
                continue;

            // If the operand is defined after user, move it to before user.
            if (_isInstDefinedAfter(operand, user))
            {
                operand->insertBefore(user);
                for (UInt i = 0; i < operand->getOperandCount(); i++)
                {
                    workList2.add(operand->getOperands() + i);
                }
                workList2.add(&operand->typeUse);
            }
        }
        workList1 = _Move(workList2);
    }
}

static void _replaceInstUsesWith(IRInst* thisInst, IRInst* other)
{
    IRDeduplicationContext* dedupContext = nullptr;

    struct WorkItem
    {
        IRInst* thisInst;
        IRInst* otherInst;
    };

    // A work list of hoistable users for which we need
    // to deduplicate/update their entry in the global numbering map.
    List<WorkItem> workList;
    HashSet<IRInst*> workListSet;

    auto addToWorkList = [&](IRInst* src, IRInst* target)
    {
        if (workListSet.add(src))
        {
            WorkItem item;
            item.thisInst = src;
            item.otherInst = target;
            workList.add(item);
        }
    };

    addToWorkList(thisInst, other);

    for (Index i = 0; i < workList.getCount(); i++)
    {
        auto workItem = workList[i];
        thisInst = workItem.thisInst;
        other = workItem.otherInst;

        SLANG_ASSERT(other);

        // Safety check: don't try to replace something with itself.
        if (other == thisInst)
            continue;

        if (getIROpInfo(thisInst->getOp()).isHoistable())
        {
            if (!dedupContext)
            {
                SLANG_ASSERT(thisInst->getModule());
                dedupContext = thisInst->getModule()->getDeduplicationContext();
            }
            dedupContext->getInstReplacementMap()[thisInst] = other;
        }

        // We will walk through the list of uses for the current
        // instruction, and make them point to the other inst.
        IRUse* ff = thisInst->firstUse;

        // No uses? Nothing to do.
        if (!ff)
            continue;

        // ff->debugValidate();

        IRUse* uu = ff;
        for (;;)
        {
            // The uses had better all be uses of this
            // instruction, or invariants are broken.
            SLANG_ASSERT(uu->get() == thisInst);

            auto user = uu->getUser();
            bool userIsHoistable = getIROpInfo(user->getOp()).isHoistable();

            // We want to de-duplicate WitnessTable but we don't really want to hoist them.
            bool userNeedToBeHoisted = userIsHoistable && (user->getOp() != kIROp_WitnessTable);

            if (userNeedToBeHoisted)
            {
                if (!dedupContext)
                {
                    SLANG_ASSERT(user->getModule());
                    dedupContext = user->getModule()->getDeduplicationContext();
                }
                dedupContext->_removeGlobalNumberingEntry(user);
            }

            // Swap this use over to use the other value.
            uu->usedValue = other;

            // If `other` is hoistable, then we need to make sure `other` is hoisted
            // to a point before `user`, if it is not already so.
            _maybeHoistOperand(uu);

            if (userNeedToBeHoisted)
            {
                // Is the updated inst already exists in the global numbering map?
                // If so, we need to continue work on replacing the updated inst with the existing
                // value.
                IRInst* existingVal = nullptr;
                if (dedupContext->getGlobalValueNumberingMap().tryGetValue(
                        IRInstKey{user},
                        existingVal))
                {
                    // If existingVal has been replaced by something else, use that.
                    dedupContext->getInstReplacementMap().tryGetValue(existingVal, existingVal);
                    addToWorkList(user, existingVal);
                }
                else
                {
                    dedupContext->_addGlobalNumberingEntry(user);
                }
            }

            // Try to move to the next use, but bail
            // out if we are at the last one.
            IRUse* nn = uu->nextUse;
            if (!nn)
                break;

            uu = nn;
        }

        // We are at the last use (and there must
        // be at least one, because we handled
        // the case of an empty list earlier).
        SLANG_ASSERT(uu);

        // Our job at this point is to splice
        // our list of uses onto the other
        // value's uses.
        //
        // If the value already had uses, then
        // we need to patch our new list onto
        // the front.
        if (auto nn = other->firstUse)
        {
            uu->nextUse = nn;
            nn->prevLink = &uu->nextUse;
        }

        // No matter what, our list of
        // uses will become the start
        // of the list of uses for
        // `other`
        other->firstUse = ff;
        ff->prevLink = &other->firstUse;

        // And `this` will have no uses any more.
        thisInst->firstUse = nullptr;

        ff->debugValidate();
    }
}

void IRInst::replaceUsesWith(IRInst* other)
{
    _replaceInstUsesWith(this, other);
}

// Insert this instruction into the same basic block
// as `other`, right before it.
void IRInst::insertBefore(IRInst* other)
{
    SLANG_ASSERT(other);
    if (other->getPrevInst() == this)
        return;
    if (other == this)
        return;
    _insertAt(other->getPrevInst(), other, other->getParent());
}

void IRInst::insertAtStart(IRInst* newParent)
{
    SLANG_ASSERT(newParent);
    _insertAt(nullptr, newParent->getFirstDecorationOrChild(), newParent);
}

void IRInst::moveToStart()
{
    auto p = parent;
    removeFromParent();
    insertAtStart(p);
}

void IRInst::_insertAt(IRInst* inPrev, IRInst* inNext, IRInst* inParent)
{
    // Make sure this instruction has been removed from any previous parent
    this->removeFromParent();

    SLANG_ASSERT(inParent);
    SLANG_ASSERT(!inPrev || (inPrev->getNextInst() == inNext) && (inPrev->getParent() == inParent));
    SLANG_ASSERT(!inNext || (inNext->getPrevInst() == inPrev) && (inNext->getParent() == inParent));

    if (inPrev)
    {
        inPrev->next = this;
    }
    else
    {
        inParent->m_decorationsAndChildren.first = this;
    }

    if (inNext)
    {
        inNext->prev = this;
    }
    else
    {
        inParent->m_decorationsAndChildren.last = this;
    }

    this->prev = inPrev;
    this->next = inNext;
    this->parent = inParent;

#if _DEBUG
    validateIRInstOperands(this);
#endif
}

void IRInst::insertAfter(IRInst* other)
{
    SLANG_ASSERT(other);
    removeFromParent();
    _insertAt(other, other->getNextInst(), other->getParent());
}

void IRInst::insertAtEnd(IRInst* newParent)
{
    SLANG_ASSERT(newParent);
    removeFromParent();
    _insertAt(newParent->getLastDecorationOrChild(), nullptr, newParent);
}

void IRInst::moveToEnd()
{
    auto p = parent;
    removeFromParent();
    insertAtEnd(p);
}

void IRInst::insertAt(IRInsertLoc const& loc)
{
    removeFromParent();
    IRInst* other = loc.getInst();
    switch (loc.getMode())
    {
    case IRInsertLoc::Mode::None:
        break;
    case IRInsertLoc::Mode::Before:
        insertBefore(other);
        break;
    case IRInsertLoc::Mode::After:
        insertAfter(other);
        break;
    case IRInsertLoc::Mode::AtStart:
        insertAtStart(other);
        break;
    case IRInsertLoc::Mode::AtEnd:
        insertAtEnd(other);
        break;
    }
}

// Remove this instruction from its parent block,
// and then destroy it (it had better have no uses!)
void IRInst::removeFromParent()
{
    auto oldParent = getParent();

    // If we don't currently have a parent, then
    // we are doing fine.
    if (!oldParent)
        return;

    auto pp = getPrevInst();
    auto nn = getNextInst();

    if (pp)
    {
        SLANG_ASSERT(pp->getParent() == oldParent);
        pp->next = nn;
    }
    else
    {
        oldParent->m_decorationsAndChildren.first = nn;
    }

    if (nn)
    {
        SLANG_ASSERT(nn->getParent() == oldParent);
        nn->prev = pp;
    }
    else
    {
        oldParent->m_decorationsAndChildren.last = pp;
    }

    prev = nullptr;
    next = nullptr;
    parent = nullptr;
}

void IRInst::removeArguments()
{
    typeUse.clear();
    for (UInt aa = 0; aa < operandCount; ++aa)
    {
        IRUse& use = getOperands()[aa];
        use.clear();
    }
}

void IRInst::removeOperand(Index index)
{
    for (Index i = index; i < (Index)operandCount - 1; i++)
    {
        getOperands()[i].set(getOperand(i + 1));
    }
    getOperands()[operandCount - 1].clear();
    operandCount--;
    return;
}

// Remove this instruction from its parent block,
// and then destroy it (it had better have no uses, or descendants with uses!)
void IRInst::removeAndDeallocate()
{
    removeAndDeallocateAllDecorationsAndChildren();

    if (auto module = getModule())
    {
        if (getIROpInfo(getOp()).isHoistable())
        {
            module->getDeduplicationContext()->removeHoistableInstFromGlobalNumberingMap(this);
        }
        else if (auto constInst = as<IRConstant>(this))
        {
            module->getDeduplicationContext()->getConstantMap().remove(IRConstantKey{constInst});
        }
        module->getDeduplicationContext()->getInstReplacementMap().remove(this);
        if (auto func = as<IRGlobalValueWithCode>(this))
            module->invalidateAnalysisForInst(func);
    }
    removeArguments();
    removeFromParent();

    // Run destructor to be sure...
    this->~IRInst();
}

void IRInst::removeAndDeallocateAllDecorationsAndChildren()
{
    IRInst* nextChild = nullptr;
    for (IRInst* child = getFirstDecorationOrChild(); child; child = nextChild)
    {
        nextChild = child->getNextInst();
        child->removeAndDeallocate();
    }
}

void IRInst::transferDecorationsTo(IRInst* target)
{
    while (auto decoration = getFirstDecoration())
    {
        decoration->removeFromParent();
        decoration->insertAtStart(target);
    }
}

bool IRInst::mightHaveSideEffects(SideEffectAnalysisOptions options)
{
    // TODO: We should drive this based on flags specified
    // in `ir-inst-defs.h` isntead of hard-coding things here,
    // but this is good enough for now if we are conservative:

    if (as<IRType>(this))
        return false;

    if (as<IRConstant>(this))
        return false;

    if (as<IRLayout>(this))
        return false;

    if (as<IRAttr>(this))
        return false;

    if (as<IRSPIRVAsmOperand>(this))
        return false;

    switch (getOp())
    {
    // By default, assume that we might have side effects,
    // to safely cover all the instructions we haven't had time to think about.
    default:
        break;

    case kIROp_Call:
        {
            // In the general case, a function call must be assumed to
            // have almost arbitrary side effects.
            //
            // However, it is possible that the callee can be identified,
            // and it may be a function with an attribute that explicitly
            // limits the side effects it is allowed to have.
            //
            // For now, we will explicitly check for the `[__readNone]`
            // attribute, which was used to mark functions that compute
            // their result strictly as a function of the arguments (and
            // not anything they point to, or other non-argument state).
            // Calls to such functions cannot have side effects (except
            // for things like stack overflow that abstract language models
            // tend to ignore), and can be subject to dead code elimination,
            // common subexpression elimination, etc.
            //
            auto call = cast<IRCall>(this);
            return !(isSideEffectFreeFunctionalCall(call, options));
        }
        break;

        // All of the cases for "global values" are side-effect-free.
    case kIROp_StructType:
    case kIROp_StructField:
    case kIROp_GLSLShaderStorageBufferType:
    case kIROp_RTTIPointerType:
    case kIROp_RTTIObject:
    case kIROp_RTTIType:
    case kIROp_Func:
    case kIROp_Generic:
    case kIROp_Var:
    case kIROp_Param:
    case kIROp_GlobalVar: // Note: the IRGlobalVar represents the *address*, so only a
                          // load/store would have side effects
    case kIROp_GlobalConstant:
    case kIROp_GlobalParam:
    case kIROp_StructKey:
    case kIROp_GlobalGenericParam:
    case kIROp_ThisTypeWitness:
    case kIROp_WitnessTable:
    case kIROp_WitnessTableEntry:
    case kIROp_InterfaceRequirementEntry:
    case kIROp_Block:
    case kIROp_Each:
    case kIROp_TypeEqualityWitness:
        return false;

        /// Liveness markers have no side effects
    case kIROp_LiveRangeStart:
    case kIROp_LiveRangeEnd:

    case kIROp_Nop:
    case kIROp_undefined:
    case kIROp_DefaultConstruct:
    case kIROp_Specialize:
    case kIROp_LookupWitness:
    case kIROp_GetSequentialID:
    case kIROp_GetAddr:
    case kIROp_GetValueFromBoundInterface:
    case kIROp_MakeUInt64:
    case kIROp_MakeCoopVector:
    case kIROp_MakeVector:
    case kIROp_MakeMatrix:
    case kIROp_MakeMatrixFromScalar:
    case kIROp_MatrixReshape:
    case kIROp_VectorReshape:
    case kIROp_MakeWitnessPack:
    case kIROp_MakeArray:
    case kIROp_MakeArrayFromElement:
    case kIROp_MakeStruct:
    case kIROp_MakeString:
    case kIROp_getNativeStr:
    case kIROp_MakeResultError:
    case kIROp_MakeResultValue:
    case kIROp_GetResultError:
    case kIROp_GetResultValue:
    case kIROp_IsResultError:
    case kIROp_MakeOptionalValue:
    case kIROp_MakeOptionalNone:
    case kIROp_OptionalHasValue:
    case kIROp_GetOptionalValue:
    case kIROp_DifferentialPairGetPrimal:
    case kIROp_DifferentialPairGetDifferential:
    case kIROp_MakeDifferentialPair:
    case kIROp_MakeTuple:
    case kIROp_MakeValuePack:
    case kIROp_GetTupleElement:
    case kIROp_StructuredBufferLoad:
    case kIROp_RWStructuredBufferLoad:
    case kIROp_RWStructuredBufferGetElementPtr:
    case kIROp_CombinedTextureSamplerGetSampler:
    case kIROp_CombinedTextureSamplerGetTexture:
    case kIROp_Load: // We are ignoring the possibility of loads from bad addresses, or
                     // `volatile` loads
    case kIROp_LoadReverseGradient:
    case kIROp_ReverseGradientDiffPairRef:
    case kIROp_ImageSubscript:
    case kIROp_FieldExtract:
    case kIROp_FieldAddress:
    case kIROp_GetElement:
    case kIROp_GetElementPtr:
    case kIROp_GetOffsetPtr:
    case kIROp_UpdateElement:
    case kIROp_MeshOutputRef:
    case kIROp_MakeVectorFromScalar:
    case kIROp_swizzle:
    case kIROp_swizzleSet: // Doesn't actually "set" anything - just returns the resulting
                           // vector
    case kIROp_Add:
    case kIROp_Sub:
    case kIROp_Mul:
    case kIROp_Lsh:
    case kIROp_Rsh:
    case kIROp_Eql:
    case kIROp_Neq:
    case kIROp_Greater:
    case kIROp_Less:
    case kIROp_Geq:
    case kIROp_Leq:
    case kIROp_BitAnd:
    case kIROp_BitXor:
    case kIROp_BitOr:
    case kIROp_And:
    case kIROp_Or:
    case kIROp_Neg:
    case kIROp_Not:
    case kIROp_BitNot:
    case kIROp_Select:
    case kIROp_MakeExistential:
    case kIROp_ExtractExistentialType:
    case kIROp_ExtractExistentialValue:
    case kIROp_ExtractExistentialWitnessTable:
    case kIROp_WrapExistential:
    case kIROp_BuiltinCast:
    case kIROp_BitCast:
    case kIROp_CastFloatToInt:
    case kIROp_CastIntToFloat:
    case kIROp_IntCast:
    case kIROp_FloatCast:
    case kIROp_CastPtrToInt:
    case kIROp_CastIntToPtr:
    case kIROp_PtrCast:
    case kIROp_CastEnumToInt:
    case kIROp_CastIntToEnum:
    case kIROp_EnumCast:
    case kIROp_CastUInt2ToDescriptorHandle:
    case kIROp_CastDescriptorHandleToUInt2:
    case kIROp_CastDescriptorHandleToResource:
    case kIROp_GetDynamicResourceHeap:
    case kIROp_CastDynamicResource:
    case kIROp_AllocObj:
    case kIROp_BitfieldExtract:
    case kIROp_BitfieldInsert:
    case kIROp_PackAnyValue:
    case kIROp_UnpackAnyValue:
    case kIROp_Reinterpret:
    case kIROp_GetNativePtr:
    case kIROp_BackwardDiffIntermediateContextType:
    case kIROp_MakeTargetTuple:
    case kIROp_GetTargetTupleElement:
    case kIROp_TorchGetCudaStream:
    case kIROp_MakeTensorView:
    case kIROp_TorchTensorGetView:
    case kIROp_GetStringHash:
    case kIROp_AllocateOpaqueHandle:
    case kIROp_GetArrayLength:
    case kIROp_ResolveVaryingInputRef:
    case kIROp_GetPerVertexInputArray:
    case kIROp_MetalCastToDepthTexture:
    case kIROp_GetCurrentStage:
        return false;

    case kIROp_ForwardDifferentiate:
    case kIROp_BackwardDifferentiate:
    case kIROp_BackwardDifferentiatePrimal:
    case kIROp_BackwardDifferentiatePropagate:
    case kIROp_DetachDerivative:
        return false;

    case kIROp_Div:
    case kIROp_IRem:
        if (isIntegralScalarOrCompositeType(getFullType()))
        {
            if (auto intLit = as<IRIntLit>(getOperand(1)))
            {
                if (intLit->getValue() != 0)
                    return false;
            }
            return true;
        }
        return false;

    case kIROp_FRem:
        return false;
    }
    return true;
}

IRModule* IRInst::getModule()
{
    IRInst* ii = this;
    while (ii)
    {
        if (auto moduleInst = as<IRModuleInst>(ii))
            return moduleInst->module;

        ii = ii->getParent();
    }
    return nullptr;
}

//
// IRType
//

IRType* unwrapArray(IRType* type)
{
    IRType* t = type;
    while (auto arrayType = as<IRArrayTypeBase>(t))
    {
        t = arrayType->getElementType();
    }
    return t;
}

//
// IRTargetIntrinsicDecoration
//

IRTargetIntrinsicDecoration* findAnyTargetIntrinsicDecoration(IRInst* val)
{
    IRInst* inst = getResolvedInstForDecorations(val);
    return inst->findDecoration<IRTargetIntrinsicDecoration>();
}

template<typename T>
IRTargetSpecificDecoration* findBestTargetDecoration(
    IRInst* inInst,
    CapabilitySet const& targetCaps)
{
    IRInst* inst = getResolvedInstForDecorations(inInst);

    // We will search through all the `IRTargetIntrinsicDecoration`s on
    // the instruction, looking for those that are applicable to the
    // current code generation target. Among the application decorations
    // we will try to find one that is "best" in the sense that it is
    // more (or at least as) specialized for the target than the
    // others.
    //
    IRTargetSpecificDecoration* bestDecoration = nullptr;
    CapabilitySet bestCaps;
    for (auto dd : inst->getDecorations())
    {
        auto decoration = as<IRTargetSpecificDecoration>(dd);
        if (!decoration)
            continue;
        if (!T::isaImpl(decoration->getOp()))
            continue;

        auto decorationCaps = decoration->getTargetCaps();
        if (decorationCaps.isIncompatibleWith(targetCaps))
            continue;

        if (decoration->hasPredicate())
        {
            const auto scrutinee = decoration->getTypeScrutinee();
            const auto predicate = decoration->getTypePredicate();
            const auto predicateFun = predicate == "boolean" ? [](auto t)
            { return t->getOp() == kIROp_BoolType; }
                                      : predicate == "integral" ? isIntegralType
                                      : predicate == "floating" ? isFloatingType
                                                                : nullptr;

            SLANG_ASSERT(predicateFun);
            if (!predicateFun(scrutinee))
                continue;
        }

        bool isEqual;
        if (!bestDecoration || decorationCaps.isBetterForTarget(bestCaps, targetCaps, isEqual))
        {
            bestDecoration = decoration;
            bestCaps = decorationCaps;
        }
    }

    return bestDecoration;
}

template<typename T>
IRTargetSpecificDecoration* findBestTargetDecoration(
    IRInst* val,
    CapabilityName targetCapabilityAtom)
{
    return findBestTargetDecoration<T>(val, CapabilitySet(targetCapabilityAtom));
}

template IRTargetSpecificDecoration* findBestTargetDecoration<IRRequirePreludeDecoration>(
    IRInst* val,
    CapabilityName targetCapabilityAtom);

bool findTargetIntrinsicDefinition(
    IRInst* callee,
    CapabilitySet const& targetCaps,
    UnownedStringSlice& outDefinition,
    IRInst*& outInst)
{
    if (auto decor = findBestTargetIntrinsicDecoration(callee, targetCaps))
    {
        outDefinition = decor->getDefinition();
        outInst = decor;
        return true;
    }
    auto func = as<IRGlobalValueWithCode>(callee);
    if (!func)
        return false;
    for (auto block : func->getBlocks())
    {
        if (auto genAsm = as<IRGenericAsm>(block->getTerminator()))
        {
            outDefinition = genAsm->getAsm();
            outInst = genAsm;
            return true;
        }
    }
    return false;
}

#if 0
    IRFunc* cloneSimpleFuncWithoutRegistering(IRSpecContextBase* context, IRFunc* originalFunc)
    {
        auto clonedFunc = context->builder->createFunc();
        cloneFunctionCommon(context, clonedFunc, originalFunc, false);
        return clonedFunc;
    }
#endif

IRInst* findGenericReturnVal(IRGeneric* generic)
{
    auto lastBlock = generic->getLastBlock();
    if (!lastBlock)
        return nullptr;

    auto returnInst = as<IRReturn>(lastBlock->getTerminator());
    if (!returnInst)
        return nullptr;

    auto val = returnInst->getVal();
    return val;
}

IRInst* findInnerMostGenericReturnVal(IRGeneric* generic)
{
    IRInst* inst = generic;
    while (auto genericInst = as<IRGeneric>(inst))
        inst = findGenericReturnVal(genericInst);
    return inst;
}

IRInst* findOuterGeneric(IRInst* inst)
{
    if (inst)
    {
        inst = inst->getParent();
    }
    else
    {
        return nullptr;
    }

    while (inst)
    {
        if (as<IRGeneric>(inst))
            return inst;

        inst = inst->getParent();
    }
    return nullptr;
}

IRInst* maybeFindOuterGeneric(IRInst* inst)
{
    auto outerGeneric = findOuterGeneric(inst);
    if (!outerGeneric)
        return inst;
    return outerGeneric;
}

IRInst* findOuterMostGeneric(IRInst* inst)
{
    IRInst* currInst = inst;
    while (auto outerGeneric = findOuterGeneric(currInst))
    {
        currInst = outerGeneric;
    }
    return currInst;
}

IRGeneric* findSpecializedGeneric(IRSpecialize* specialize)
{
    return as<IRGeneric>(specialize->getBase());
}


IRInst* findSpecializeReturnVal(IRSpecialize* specialize)
{
    auto base = specialize->getBase();

    while (auto baseSpec = as<IRSpecialize>(base))
    {
        auto returnVal = findSpecializeReturnVal(baseSpec);
        if (!returnVal)
            break;

        base = returnVal;
    }

    if (auto generic = as<IRGeneric>(base))
    {
        return findGenericReturnVal(generic);
    }

    return nullptr;
}

IRInst* getResolvedInstForDecorations(IRInst* inst, bool resolveThroughDifferentiation)
{
    IRInst* candidate = inst;
    for (;;)
    {
        if (auto specInst = as<IRSpecialize>(candidate))
        {
            candidate = specInst->getBase();
            continue;
        }
        if (resolveThroughDifferentiation)
        {
            switch (candidate->getOp())
            {
            case kIROp_ForwardDifferentiate:
            case kIROp_BackwardDifferentiate:
            case kIROp_BackwardDifferentiatePrimal:
            case kIROp_BackwardDifferentiatePropagate:
                candidate = candidate->getOperand(0);
                continue;
            default:
                break;
            }
        }
        if (auto genericInst = as<IRGeneric>(candidate))
        {
            if (auto returnVal = findGenericReturnVal(genericInst))
            {
                candidate = returnVal;
                continue;
            }
        }

        return candidate;
    }
}

bool isDefinition(IRInst* inVal)
{
    IRInst* val = getResolvedInstForDecorations(inVal);

    // Some cases of instructions have structural
    // rules about when they are considered to have
    // a definition (e.g., a function must have a body).
    //
    switch (val->getOp())
    {
    case kIROp_Func:
        return val->getFirstChild() != nullptr;

    case kIROp_GlobalConstant:
        return cast<IRGlobalConstant>(val)->getValue() != nullptr;

    default:
        break;
    }

    // In all other cases, if we have an instruciton
    // that has *not* been marked for import, then
    // we consider it to be a definition.
    return true;
}

void markConstExpr(IRBuilder* builder, IRInst* irValue)
{
    // We will take an IR value with type `T`,
    // and turn it into one with type `@ConstExpr T`.

    // TODO: need to be careful if the value already has a rate
    // qualifier set.

    irValue->setFullType(
        builder->getRateQualifiedType(builder->getConstExprRate(), irValue->getDataType()));
}

bool isBuiltin(IRInst* inst)
{
    return inst->findDecoration<IRBuiltinDecoration>() != nullptr;
}
IRFunc* getParentFunc(IRInst* inst)
{
    auto parent = inst->getParent();
    while (parent)
    {
        if (auto func = as<IRFunc>(parent))
            return func;
        parent = parent->getParent();
    }
    return nullptr;
}

bool hasDescendent(IRInst* inst, IRInst* child)
{
    auto parent = child->getParent();
    while (parent)
    {
        if (inst == parent)
            return true;
        parent = parent->getParent();
    }
    return false;
}

IRInst* getGenericReturnVal(IRInst* inst)
{
    if (auto gen = as<IRGeneric>(inst))
    {
        return findGenericReturnVal(gen);
    }
    return inst;
}

IRDominatorTree* IRAnalysis::getDominatorTree()
{
    return static_cast<IRDominatorTree*>(domTree.get());
}

bool isMovableInst(IRInst* inst)
{
    // Don't try to modify hoistable insts, they are already globally deduplicated.
    if (getIROpInfo(inst->getOp()).isHoistable())
        return false;

    switch (inst->getOp())
    {
    case kIROp_MakeCoopVector:
    case kIROp_Add:
    case kIROp_Sub:
    case kIROp_Mul:
    case kIROp_FRem:
    case kIROp_IRem:
    case kIROp_Lsh:
    case kIROp_Rsh:
    case kIROp_And:
    case kIROp_Or:
    case kIROp_Not:
    case kIROp_Neg:
    case kIROp_FieldExtract:
    case kIROp_FieldAddress:
    case kIROp_GetElement:
    case kIROp_GetElementPtr:
    case kIROp_GetOffsetPtr:
    case kIROp_UpdateElement:
    case kIROp_Specialize:
    case kIROp_LookupWitness:
    case kIROp_OptionalHasValue:
    case kIROp_GetOptionalValue:
    case kIROp_MakeOptionalValue:
    case kIROp_MakeTuple:
    case kIROp_GetTupleElement:
    case kIROp_MakeStruct:
    case kIROp_MakeArray:
    case kIROp_MakeArrayFromElement:
    case kIROp_MakeVector:
    case kIROp_MakeMatrix:
    case kIROp_MakeMatrixFromScalar:
    case kIROp_MakeVectorFromScalar:
    case kIROp_swizzle:
    case kIROp_swizzleSet:
    case kIROp_MatrixReshape:
    case kIROp_MakeString:
    case kIROp_MakeResultError:
    case kIROp_MakeResultValue:
    case kIROp_GetResultError:
    case kIROp_GetResultValue:
    case kIROp_CastFloatToInt:
    case kIROp_CastIntToFloat:
    case kIROp_CastIntToPtr:
    case kIROp_CastPtrToBool:
    case kIROp_CastPtrToInt:
    case kIROp_PtrCast:
    case kIROp_CastEnumToInt:
    case kIROp_CastIntToEnum:
    case kIROp_EnumCast:
    case kIROp_CastDynamicResource:
    case kIROp_BitAnd:
    case kIROp_BitNot:
    case kIROp_BitOr:
    case kIROp_BitXor:
    case kIROp_BitCast:
    case kIROp_IntCast:
    case kIROp_FloatCast:
    case kIROp_Reinterpret:
    case kIROp_Greater:
    case kIROp_Less:
    case kIROp_Geq:
    case kIROp_Leq:
    case kIROp_Neq:
    case kIROp_Eql:
    case kIROp_ExtractExistentialType:
    case kIROp_ExtractExistentialValue:
    case kIROp_ExtractExistentialWitnessTable:
        return true;
    case kIROp_Call:
        // Similar to the case in IRInst::mightHaveSideEffects, pure
        // calls are ok
        return isPureFunctionalCall(cast<IRCall>(inst));
    case kIROp_Load:
        // Load is generally not movable, an exception is loading a global constant buffer.
        if (auto load = as<IRLoad>(inst))
        {
            auto addrType = load->getPtr()->getDataType();
            switch (addrType->getOp())
            {
            case kIROp_ConstantBufferType:
            case kIROp_ParameterBlockType:
                return true;
            default:
                break;
            }
        }
        return false;
    default:
        return false;
    }
}

void IRInst::addBlock(IRBlock* block)
{
    block->insertAtEnd(this);
}

void IRInst::dump(String& outStr)
{
    StringBuilder sb;

    if (auto intLit = as<IRIntLit>(this))
    {
        sb << intLit->getValue();
    }
    else if (auto stringLit = as<IRStringLit>(this))
    {
        sb << stringLit->getStringSlice();
    }
    else
    {
        IRDumpOptions options;
        StringWriter writer(&sb, Slang::WriterFlag::AutoFlush);
        dumpIR(this, options, nullptr, &writer);
    }

    outStr = sb.toString();
}

void IRInst::dump()
{
    String s;
    dump(s);
    std::cout << s.begin() << std::endl;
}
} // namespace Slang

#if SLANG_VC
#ifdef _DEBUG
// Natvis sometimes cannot find enum values.
// Export symbols for them to make sure natvis works correctly when debugging external projects.
SLANG_API const int SlangDebug__IROpNameHint = Slang::kIROp_NameHintDecoration;
SLANG_API const int SlangDebug__IROpExport = Slang::kIROp_ExportDecoration;
SLANG_API const int SlangDebug__IROpImport = Slang::kIROp_ImportDecoration;
SLANG_API const int SlangDebug__IROpStringLit = Slang::kIROp_StringLit;
SLANG_API const int SlangDebug__IROpIntLit = Slang::kIROp_IntLit;
#endif
#endif
