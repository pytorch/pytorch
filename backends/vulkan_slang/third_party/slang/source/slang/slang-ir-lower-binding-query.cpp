// slang-ir-lower-tuple-types.cpp

#include "slang-ir-insts.h"
#include "slang-ir-lower-tuple-types.h"
#include "slang-ir.h"

// The pass in this file lowers the `getRegisterIndex()` and
// `getSpaceIndex()` intrinsics, by replacing them with literal
// values derived from the binding information on shader parameters.
//
// If these operations are applied to a global shader parameter,
// then we can simply read the binding information from that parameter
// and use it directly.
//
// Otherwise, we expect that the opaque object (resource/sampler/etc.)
// being referenced was passed down into the current function from
// a caller. We thus introduce new function parameters after the
// resource in question, transforming, e.g., this:
//
//      void doThings(
//          float       a,
//          Texture2D   t,
//          float       b )
//      {
//          ... __getRegisterIndex(t) ...
//          ... __getSpaceIndex(t) ...
//      }
//      ...
//      doThings(myTexture);
//
// into this:
//
//      void doThings(
//          float       a,
//          Texture2D   t,
//          uint        t_index,
//          uint        t_space,
//          float       b )
//      {
//          ... t_index ...
//          ... t_space ...
//      }
//      ...
//      doThings(myTexture, __getRegisterIndex(myTexture), __getRegisterSpace(myTexture));
//
// At that point we have removed the invocations of `getRegisterIndex`
// and `getRegisterSpace` in the callee function, but introduced new
// invocations in the caller function, so we need to iterate until
// we eventually either bottom out at a global shader parameter, or
// run into a context that we cannot simplify.

namespace Slang
{
// There are a ton of passes we've implemented now that use
// some basic work-list structures, and it seems a bit silly
// to be writing that code intermixed with the actual algorithm.
//
// For this file, we break the common work-list functionality
// out into a base type that we can re-use in specific passes.
//
struct WorkListPass
{
public:
    IRModule* module;
    DiagnosticSink* sink;

protected:
    // The base type needs to abstract over how the
    // concrete pass will process each instruction
    // that gets placed into the work list.

    virtual void processInst(IRInst* inst) = 0;

    // Otherwise, the implementation of the work list
    // itself is straightforward, and not anything
    // that hasn't been seen in other files.

    InstWorkList workList;
    InstHashSet workListSet;

    WorkListPass(IRModule* inModule)
        : module(inModule)
        , workList(inModule)
        , workListSet(inModule)
        , toBeDeleted(inModule)
        , toBeDeletedSet(inModule)
    {
    }

    void addToWorkList(IRInst* inst)
    {
        if (workListSet.contains(inst))
            return;

        workList.add(inst);
        workListSet.add(inst);
    }

    void processWorkList()
    {
        while (workList.getCount() != 0)
        {
            IRInst* inst = workList.getLast();

            workList.removeLast();
            workListSet.remove(inst);

            processInst(inst);

            for (auto child = inst->getLastChild(); child; child = child->getPrevInst())
            {
                addToWorkList(child);
            }
        }
    }

    // As long as we are factoring out repeated cruft,
    // it seems reasonable to *also* deal with the
    // frequent need to buffer up instructions to
    // be deleted when a pass is complete.

    InstWorkList toBeDeleted;
    InstHashSet toBeDeletedSet;

    void addToBeDeleted(IRInst* inst)
    {
        if (toBeDeletedSet.contains(inst))
            return;

        toBeDeleted.add(inst);
        toBeDeletedSet.add(inst);
    }

    void processDeletions()
    {
        for (auto inst : toBeDeleted)
        {
            inst->removeAndDeallocate();
        }
        toBeDeleted.clear();
        toBeDeletedSet.clear();
    }
};

// The concrete pass will then be a specialization of
// the base work-list abstraction.
//
struct BindingQueryLoweringContext : public WorkListPass
{
    BindingQueryLoweringContext(IRModule* inModule)
        : WorkListPass(inModule)
    {
    }

    // All of the intrinsics we will be processing use
    // the same result type (`uint`), so it is helpful
    // to cache a pointer to the IR type at the start
    // of the pass and re-use it.
    //
    IRType* indexType = nullptr;

    void processModule()
    {
        IRBuilder builder(module);
        indexType = builder.getUIntType();

        // Processing the module consists of recursively
        // processing all the instructions in one pass,
        // and then potentially revisiting instructions
        // that had new intrinsics added to their bodies.
        //
        addToWorkList(module->getModuleInst());
        processWorkList();
    }

    void processInst(IRInst* inst)
    {
        // For this pass, we really only care about
        // our binding query instructions.
        //
        if (auto query = as<IRBindingQuery>(inst))
        {
            processQueryInst(query);
        }
    }

    void processQueryInst(IRBindingQuery* inst)
    {
        // Processing one of the query instructions is conceptually
        // simple: we find a compute a value to replace it with,
        // and then simply *replace* the instruction.
        //
        auto replacementValue = findOrComputeReplacementValueFor(inst);
        if (!replacementValue)
        {
            // If we cannot find or compute a replacement value,
            // then we need to treat it as an error, since the
            // binding query intrinsics don't admit any reasonable
            // runtime implementation.
            //
            sink->diagnose(inst, Diagnostics::opaqueReferenceMustResolveToGlobal);
            return;
        }

        inst->replaceUsesWith(replacementValue);
        inst->removeAndDeallocate();
    }

    // We want to cache the results of computing the binding
    // information for an opaque-type value, in case doing
    // so required adding or modifying code.
    //
    // For that purpose, we introduce a simple data structure
    // to hold the two pieces of binding information we
    // care about.
    //
    struct OpaqueValueInfo
    {
        IRInst* registerIndex = nullptr;
        IRInst* registerSpace = nullptr;
    };

    IRInst* findOrComputeReplacementValueFor(IRBindingQuery* query)
    {
        // Finding the replacement for a given query instruction
        // then amounts to computing (or caching) the binding
        // information for the opaque-type value it queries,
        // and then projecting out the appropriate field.

        auto opaqueValue = query->getOpaqueValue();
        auto opaqueValueInfo = findOrComputeOpaqueValueInfo(opaqueValue);

        switch (query->getOp())
        {
        default:
            SLANG_UNEXPECTED("unhandled binding query instruction type");
            UNREACHABLE_RETURN(query);

        case kIROp_GetRegisterIndex:
            return opaqueValueInfo.registerIndex;

        case kIROp_GetRegisterSpace:
            return opaqueValueInfo.registerSpace;
        }
    }

    // The information will be cached in a dictionary,
    // keyed on the opaque-type value that the information
    // was computed for.
    //
    Dictionary<IRInst*, OpaqueValueInfo> mapOpaqueValueToInfo;

    // Looking up the cached information (if any) is a simple
    // matter of using the dictionary.
    //
    // (We have a distinct operation for lookup vs. the
    // memo-cached lookup below, because we may want to
    // query this information while computing an entry,
    // and we don't want to introduce potential recursion.
    //
    OpaqueValueInfo* findOpaqueValueInfo(IRInst* opaqueValue)
    {
        return mapOpaqueValueToInfo.tryGetValue(opaqueValue);
    }

    OpaqueValueInfo findOrComputeOpaqueValueInfo(IRInst* opaqueValue)
    {
        if (auto foundInfo = findOpaqueValueInfo(opaqueValue))
            return *foundInfo;

        // If there is no information registered in the cache, we
        // compute it on-demand.
        //
        // Note that there is no potential for circularity, so
        // long as the implementation of `computeOpaqueValueInfo`
        // does not itself call `findOrComputeValueInfo`.
        //
        auto computedInfo = computeOpaqueValueInfo(opaqueValue);
        mapOpaqueValueToInfo.add(opaqueValue, computedInfo);
        return computedInfo;
    }

    // We are now (finally) getting into the meat of what this
    // pass needs to do. Given an instruction with an opaque
    // type, we need to try to compute the register and space
    // it is bound to, or conspire to have that information
    // passed along.
    //
    OpaqueValueInfo computeOpaqueValueInfo(IRInst* opaqueValue)
    {
        if (auto getElement = as<IRGetElement>(opaqueValue))
        {
            IRInst* baseInst = getElement->getBase();
            IRInst* indexInst = getElement->getIndex();

            IRInst* elementType = getElement->getDataType();

            // TODO(JS): This a hack to make this work for arrays of resource type.
            // It won't work in the general case as it stands because we would need
            // to propogate layout kind types needed at usage sites.
            // Without knowing the resource kind that is being processed it's not possible
            // to accumulate the calculation.
            //
            // So presumably we need to request a binding query for a specific resource kind.
            // We could do this by making the type of the binding query hold the type.

            // We need to add instructions which will work out the binding for the base
            OpaqueValueInfo baseInfo = findOrComputeOpaqueValueInfo(baseInst);

            // If we couldn't find it we are done
            if (baseInfo.registerIndex == nullptr || baseInfo.registerSpace == nullptr)
            {
                return baseInfo;
            }


            LayoutResourceKind kind = LayoutResourceKind::None;
            Index stride = 1;

            if (auto resourceType = as<IRResourceType>(elementType))
            {
                const auto shape = resourceType->getShape();

                switch (shape)
                {
                case SLANG_TEXTURE_1D:
                case SLANG_TEXTURE_2D:
                case SLANG_TEXTURE_3D:
                case SLANG_TEXTURE_CUBE:
                case SLANG_STRUCTURED_BUFFER:
                case SLANG_BYTE_ADDRESS_BUFFER:
                case SLANG_TEXTURE_BUFFER:
                    {
                        const auto access = resourceType->getAccess();
                        bool isReadOnly = (access == SLANG_RESOURCE_ACCESS_READ);

                        kind = isReadOnly ? LayoutResourceKind::ShaderResource
                                          : LayoutResourceKind::UnorderedAccess;
                        break;
                    }
                default:
                    break;
                }
            }
            else if (as<IRSamplerStateTypeBase>(elementType))
            {
                kind = LayoutResourceKind::SamplerState;
            }
            else if (as<IRConstantBufferType>(elementType))
            {
                kind = LayoutResourceKind::ConstantBuffer;
            }

            if (kind == LayoutResourceKind::None)
            {
                // Can't determine the kind
                return OpaqueValueInfo();
            }

            // If the element type has type layout we can try and use that
            if (auto layoutDecoration = elementType->findDecoration<IRLayoutDecoration>())
            {
                // We have to calculate
                if (auto elementTypeLayout = as<IRTypeLayout>(layoutDecoration->getLayout()))
                {
                    IRTypeSizeAttr* sizeAttr = elementTypeLayout->findSizeAttr(kind);
                    sizeAttr = sizeAttr ? sizeAttr
                                        : elementTypeLayout->findSizeAttr(
                                              LayoutResourceKind::DescriptorTableSlot);

                    if (!sizeAttr)
                    {
                        // Couldn't work it out
                        return OpaqueValueInfo();
                    }

                    // TODO(JS): Perhaps we have to do something else if not finite?
                    stride = sizeAttr->getFiniteSize();
                }
            }

            SLANG_UNUSED(indexInst);

            // Okay we need to create an instruction which is
            // base + stride * index

            IRBuilder builder(module);

            builder.setInsertBefore(opaqueValue);

            auto calcRegisterInst = builder.emitAdd(
                indexType,
                builder.emitMul(indexType, builder.getIntValue(indexType, stride), indexInst),
                baseInfo.registerIndex);

            OpaqueValueInfo finalInfo;
            finalInfo.registerIndex = calcRegisterInst;
            finalInfo.registerSpace = baseInfo.registerSpace;

            return finalInfo;
        }
        else if (auto globalParam = as<IRGlobalParam>(opaqueValue))
        {
            // The simple/base case is when we have a global shader
            // parameter that has layout information attached.
            //
            // Note that this pass needs to run late enough that
            // shader parameters declared at other scopes will have
            // been massaged into the appropriate form.
            //
            if (auto layoutDecoration = globalParam->findDecoration<IRLayoutDecoration>())
            {
                if (auto layout = as<IRVarLayout>(layoutDecoration->getLayout()))
                {
                    // We expect any shader parameter of an opaque type
                    // to have a relevant resource kind, but it isn't
                    // too hard to code defensively. We will iterate
                    // over the resource kinds that are present and
                    // take the first one that represents an opaque type.
                    //
                    for (auto offsetAttr : layout->getOffsetAttrs())
                    {
                        switch (offsetAttr->getResourceKind())
                        {
                        default:
                            break;

                        case LayoutResourceKind::ShaderResource:
                        case LayoutResourceKind::UnorderedAccess:
                        case LayoutResourceKind::ConstantBuffer:
                        case LayoutResourceKind::SamplerState:
                        case LayoutResourceKind::DescriptorTableSlot:
                            {
                                IRBuilder builder(module);

                                OpaqueValueInfo info;
                                info.registerIndex =
                                    builder.getIntValue(indexType, offsetAttr->getOffset());
                                info.registerSpace =
                                    builder.getIntValue(indexType, offsetAttr->getSpace());
                                return info;
                            }
                            break;
                        }
                    }
                }
            }
        }
        else if (auto param = as<IRParam>(opaqueValue))
        {
            // The other very interesting case is when the opaque-type
            // value is an `IRParam`, which indicates that it is either
            // a function parameter or a phi node of a basic block.
            //
            // Either way, we always expect a parameter to appear as
            // a child of a block.
            //
            auto block = as<IRBlock>(param->getParent());
            SLANG_ASSERT(block);

            // When rewriting call sites, we will need to know the
            // index of `param` within the parameter list.
            //
            Index paramIndex = -1;
            {
                Count paramCounter = 0;
                for (auto p : block->getParams())
                {
                    Index i = paramCounter++;
                    if (p == param)
                    {
                        paramIndex = i;
                        break;
                    }
                }
                SLANG_ASSERT(paramIndex >= 0);
            }

            // In either case (function parameter or block parameter),
            // we will insert additional parameters after the original
            // parameter, so that the register index and space can
            // be passed along explicitly.
            //
            IRBuilder builder(module);

            // We create new parameters to pass along the register index/space,
            // and manually insert them where we want them in the parameter list.
            //
            auto registerIndexParam = builder.createParam(builder.getUIntType());
            auto registerSpaceParam = builder.createParam(builder.getUIntType());
            //
            registerSpaceParam->insertAfter(param);
            registerIndexParam->insertAfter(param);

            // We would like for the newly-introduced parameters to have
            // nice human-readable names, if the original parameter did.
            //
            if (auto nameHintDecoration = param->findDecoration<IRNameHintDecoration>())
            {
                String hint;
                hint.append(nameHintDecoration->getName());
                hint.append(".");
                builder.addNameHintDecoration(
                    registerIndexParam,
                    (hint + "index").getUnownedSlice());
                builder.addNameHintDecoration(
                    registerSpaceParam,
                    (hint + "space").getUnownedSlice());
            }

            // Similarly, the new parameters should get debugging-related
            // source location information from the original parameter,
            // if it had any.
            //
            registerIndexParam->sourceLoc = param->sourceLoc;
            registerSpaceParam->sourceLoc = param->sourceLoc;

            // Now we need to scan for the places that the function or block
            // that the parameter belongs to gets referenced. At each such
            // location, we will pass along arguments to match the additional
            // parameters.
            //
            if (!block->getPrevBlock())
            {
                // If this is the first block in the parent function,
                // then this is a function parameter, and we will
                // iterate over call sites of the function and rewrite
                // them to pass along arguments for the new parameters.
                //
                auto func = block->getParent();

                for (auto use = func->firstUse; use; use = use->nextUse)
                {
                    auto user = use->getUser();
                    if (auto call = as<IRCall>(user))
                    {
                        if (call->getCallee() == func)
                        {
                            rewriteCall(call, paramIndex);
                        }
                    }
                }
            }
            else
            {
                // If this is a block parameter, we will iterate over
                // the instructions that branch to the block, and rewrite
                // their argument lists, similar to what we do for function calls.
                //
                for (auto use = block->firstUse; use; use = use->nextUse)
                {
                    auto user = use->getUser();
                    if (auto branch = as<IRUnconditionalBranch>(user))
                    {
                        if (branch->getTargetBlock() == block)
                        {
                            rewriteBranch(branch, paramIndex);
                        }
                    }
                }
            }

            // The new parameters that we introduced will be used to
            // replace any binding query intrinsics applied to
            // this opaque value.
            //
            OpaqueValueInfo info;
            info.registerIndex = registerIndexParam;
            info.registerSpace = registerSpaceParam;
            return info;
        }

        // By default we find that we cannot query binding information
        // for the given instruction.
        OpaqueValueInfo info;
        return info;
    }

    // In our IR, there isn't a lot of difference between a `call`
    // and an `unconditionalBranch`; indeed, this is part of what
    // motivates the use of `IRParam`s for both function parameters
    // and phi nodes.
    //
    // However, while both blocks and functions use the same `IRParam`
    // representation, we (currently) do not have a common base
    // between the `call` and `unconditionalBranch` instructions.
    //
    // Rather than have duplicate logic between the two cases, we
    // simply observe that for our purposes rewriting either a
    // `call` or an `unconditionalBranch` amounts to doing
    // special-case work on *one* operand of the original, while
    // copying over all the other operands as-is.
    //
    // Given this observation, we can bottleneck both calls and
    // branches into a common worker routine by passing down
    // the instruction to be rewritten and a pointer to the
    // `IRUse` for the one "interesting" operand.

    void rewriteCall(IRCall* oldCall, Index paramIndex)
    {
        rewriteCallOrBranch(oldCall, oldCall->getArgs() + paramIndex);
    }

    void rewriteBranch(IRUnconditionalBranch* oldBranch, Index paramIndex)
    {
        rewriteCallOrBranch(oldBranch, oldBranch->getArgs() + paramIndex);
    }

    void rewriteCallOrBranch(IRInst* oldCallOrBranch, IRUse* oldOperandToRewrite)
    {
        // Our goal here is to generate a new version of
        // `oldCallOrBranch` that copies over most of the
        // operands as-is, but introduces our rewrites
        // around the chosen operand.

        IRBuilder builder(module);
        builder.setInsertBefore(oldCallOrBranch);

        // We capture the old operand list as a range of
        // `IRUse`s, and set up a fresh list to hold the
        // new operands.
        //
        auto oldOperandsBegin = oldCallOrBranch->getOperands();
        auto oldOperandsEnd = oldOperandsBegin + oldCallOrBranch->getOperandCount();
        //
        List<IRInst*> newOperands;

        // All of the operands that precede the interesting
        // one can be copied over from the old list to the
        // new one as-is.
        //
        for (auto u = oldOperandsBegin; u < oldOperandToRewrite; ++u)
        {
            auto operand = u->get();
            newOperands.add(operand);
        }

        // Next we look at the value of the "intersting"
        // operand, knowing that we need to pass along
        // not only the original value but also the
        // binding information.
        //
        IRInst* arg = oldOperandToRewrite->get();
        IRInst* registerIndex = nullptr;
        IRInst* registerSpace = nullptr;

        // As a simple optimization, if we have *already*
        // computed and cached binding information for
        // the argument, we can re-use that information
        // here and now.
        //
        if (auto info = findOpaqueValueInfo(arg))
        {
            registerIndex = info->registerIndex;
            registerSpace = info->registerSpace;
        }
        else
        {
            // If there is no cached information for
            // the argument, we choose *not* to make
            // a recursive call into `findOrComputeOpaqueValueInfo`.
            //
            // Instead we will simply emit additional
            // binding query intrinsics into the body
            // of the caller (right before the call site),
            // and add those instructions to our work
            // list, to be eliminated later.
            //
            registerIndex = builder.emitIntrinsicInst(indexType, kIROp_GetRegisterIndex, 1, &arg);
            registerSpace = builder.emitIntrinsicInst(indexType, kIROp_GetRegisterSpace, 1, &arg);
            //
            addToWorkList(registerIndex);
            addToWorkList(registerSpace);
        }

        // Whether we have found existing binding information,
        // or emitted new intrinsics, we are now ready
        // to append the argument and its binding information
        // to the new operand list.
        //
        newOperands.add(arg);
        newOperands.add(registerIndex);
        newOperands.add(registerSpace);

        // Any operands of the original instruction that come
        // after the one we rewrite can be copied over as-is.
        //
        // Note: we don't currently have any operands that would
        // appear after the arguments of a `call` or `branch`,
        // but the fact that we encode `IRAttr`s on an instruction
        // as additional (trailing) operands means that this could
        // conceivably happen at some point.
        //
        for (auto u = oldOperandToRewrite + 1; u < oldOperandsEnd; ++u)
        {
            auto operand = u->get();
            newOperands.add(operand);
        }

        // Once we've built up the new operand list, we can emit
        // a new instruction that has the same opcode and type,
        // with the new operands, and then use it to replace
        // the existing instruction.
        //
        auto newCallOrBranch = builder.emitIntrinsicInst(
            oldCallOrBranch->getFullType(),
            oldCallOrBranch->getOp(),
            newOperands.getCount(),
            newOperands.getBuffer());

        oldCallOrBranch->transferDecorationsTo(newCallOrBranch);
        oldCallOrBranch->replaceUsesWith(newCallOrBranch);
        oldCallOrBranch->removeAndDeallocate();
    }
};

void lowerBindingQueries(IRModule* module, DiagnosticSink* sink)
{
    BindingQueryLoweringContext context(module);
    context.sink = sink;
    context.processModule();
}
} // namespace Slang
