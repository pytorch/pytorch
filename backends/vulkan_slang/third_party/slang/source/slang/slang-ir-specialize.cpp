// slang-ir-specialize.cpp
#include "slang-ir-specialize.h"

#include "../core/slang-performance-profiler.h"
#include "slang-ir-clone.h"
#include "slang-ir-dce.h"
#include "slang-ir-insts.h"
#include "slang-ir-lower-witness-lookup.h"
#include "slang-ir-peephole.h"
#include "slang-ir-sccp.h"
#include "slang-ir-ssa-simplification.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{

// This file implements the primary specialization pass, that takes
// generic/polymorphic Slang code and specializes/monomorphises it.
//
// At present this primarily means generating specialized copies
// of generic functions/types based on the concrete types used
// at specialization sites, and also specializing instances
// of witness-table lookup to directly refer to the concrete
// values for witnesses when witness tables are known.
//
// This pass also performs some amount of simplification and
// specialization for code using existential (interface) types
// for local variables and function parameters/results.
//
// Eventually, this pass will also need to perform specialization
// of functions to argument values for parameters that must
// be compile-time constants,
//
// All of these passes are inter-related in that applying
// simplifications/specializations of one category can open
// up opportunities for transformations in the other categories.

struct SpecializationContext;

IRInst* specializeGenericImpl(
    IRGeneric* genericVal,
    IRSpecialize* specializeInst,
    IRModule* module,
    SpecializationContext* context);

struct SpecializationContext
{
    // For convenience, we will keep a pointer to the module
    // we are specializing.
    IRModule* module;
    DiagnosticSink* sink;
    TargetProgram* targetProgram;
    SpecializationOptions options;
    bool changed = false;


    SpecializationContext(IRModule* inModule, TargetProgram* target, SpecializationOptions options)
        : workList(*inModule->getContainerPool().getList<IRInst>())
        , workListSet(*inModule->getContainerPool().getHashSet<IRInst>())
        , cleanInsts(*inModule->getContainerPool().getHashSet<IRInst>())
        , module(inModule)
        , targetProgram(target)
        , options(options)
    {
    }
    ~SpecializationContext()
    {
        module->getContainerPool().free(&workList);
        module->getContainerPool().free(&workListSet);
        module->getContainerPool().free(&cleanInsts);
    }

    bool isUnsimplifiedArithmeticInst(IRInst* inst)
    {
        switch (inst->getOp())
        {
        case kIROp_Add:
        case kIROp_Sub:
        case kIROp_Mul:
        case kIROp_Div:
        case kIROp_Neg:
        case kIROp_Not:
        case kIROp_Eql:
        case kIROp_Neq:
        case kIROp_Leq:
        case kIROp_Geq:
        case kIROp_Less:
        case kIROp_IRem:
        case kIROp_FRem:
        case kIROp_Greater:
        case kIROp_Lsh:
        case kIROp_Rsh:
        case kIROp_BitAnd:
        case kIROp_BitOr:
        case kIROp_BitXor:
        case kIROp_BitNot:
        case kIROp_BitCast:
        case kIROp_CastIntToFloat:
        case kIROp_CastFloatToInt:
        case kIROp_IntCast:
        case kIROp_FloatCast:
        case kIROp_Select:
            return true;
        default:
            return false;
        }
    }

    // An instruction is then fully specialized if and only
    // if it is in our set.
    //
    bool isInstFullySpecialized(IRInst* inst)
    {
        // A small wrinkle is that a null instruction pointer
        // sometimes appears a a type, and so should be treated
        // as fully specialized too.
        //
        // TODO: It would be nice to remove this wrinkle.
        //
        if (!inst)
            return true;

        switch (inst->getOp())
        {
        case kIROp_GlobalGenericParam:
        case kIROp_LookupWitness:
        case kIROp_GetTupleElement:
            return false;
        case kIROp_Specialize:
            // The `specialize` instruction is a bit sepcial,
            // because it is possible to have a `specialize`
            // of a built-in type so that it never gets
            // substituted for another type. (e.g., the specific
            // case where this code path first showed up
            // as necessary was `RayQuery<>`)
            //
            {
                auto specialize = cast<IRSpecialize>(inst);
                auto base = specialize->getBase();
                if (auto generic = as<IRGeneric>(base))
                {
                    // If the thing being specialized can be resolved,
                    // *and* it is a target intrinsic, ...
                    //
                    if (auto result = findGenericReturnVal(generic))
                    {
                        if (result->findDecoration<IRTargetIntrinsicDecoration>())
                        {
                            // ... then we should consider the instruction as
                            // "fully specialized" in the same cases as for
                            // any ordinary instruciton.
                            //

                            if (areAllOperandsFullySpecialized(inst))
                            {
                                return true;
                            }
                        }
                    }
                }
                return false;
            }
        }

        if (isWrapperType(inst))
        {
            // For all the wrapper type, we need to make sure the operands are fully specialized.
            return areAllOperandsFullySpecialized(inst);
        }

        if (isUnsimplifiedArithmeticInst(inst))
        {
            // For arithmetic insts, we want to wait for simplification before specialization,
            // since different insts can simplify to the same value.
            //
            return false;
        }

        // The default case is that a global value is always specialized.
        if (inst->getParent() == module->getModuleInst())
        {
            return true;
        }

        return false;
    }

    // Check if an inst is a dynamic dispatch witness table.
    // These insts may not have any uses yet, and do not have side effects,
    // but should be specialized if necessary.
    //
    bool isWitnessTableType(IRInst* inst)
    {
        return inst->findDecoration<IRDynamicDispatchWitnessDecoration>();
    }

    // When an instruction isn't fully specialized, but its operands *are*
    // then it is a candidate for specialization itself, so we will have
    // a query to check for the "all operands fully specialized" case.
    //
    bool areAllOperandsFullySpecialized(IRInst* inst)
    {
        if (!isInstFullySpecialized(inst->getFullType()))
            return false;

        UInt operandCount = inst->getOperandCount();
        for (UInt ii = 0; ii < operandCount; ++ii)
        {
            IRInst* operand = inst->getOperand(ii);
            if (!isInstFullySpecialized(operand))
                return false;
        }

        return true;
    }

    // We will use a single work list of instructions that need
    // to be considered for specialization or simplification,
    // whether generic, existential, etc.
    //
    List<IRInst*>& workList;
    HashSet<IRInst*>& workListSet;
    HashSet<IRInst*>& cleanInsts;

    void addToWorkList(IRInst* inst)
    {
        if (workListSet.add(inst))
        {
            workList.add(inst);


            addUsersToWorkList(inst);
        }
    }

    // When a transformation makes a change to an instruction,
    // we may need to re-consider transformations for instructions
    // that use its value. In those cases we will call `addUsersToWorkList`
    // on the instruction that is being modified or replaced.
    //
    void addUsersToWorkList(IRInst* inst)
    {
        for (auto use = inst->firstUse; use; use = use->nextUse)
        {
            auto user = use->getUser();

            addToWorkList(user);
        }
    }

    // Of course, somewhere along the way we expect
    // to run into uses of `specialize(...)` instructions
    // to bind a generic to arguments that we want to
    // specialize into concrete code.
    //
    // We also know that if we encouter `specialize(g, a, b, c)`
    // and then later `specialize(g, a, b, c)` again, we
    // only want to generate the specialized code for `g<a,b,c>`
    // *once*, and re-use it for both versions.
    //
    // We will cache existing specializations of generic function/types
    // using the simple key type defined as part of the IR cloning infrastructure.
    //
    typedef IRSimpleSpecializationKey Key;
    Dictionary<Key, IRInst*> genericSpecializations;


    // Now let's look at the task of finding or generation a
    // specialization of some generic `g`, given a specialization
    // instruction like `specialize(g, a, b, c)`.
    //
    // The `specializeGeneric` function will return a value
    // suitable for use as a replacement for the `specialize(...)`
    // instruction.
    //
    IRInst* specializeGeneric(IRGeneric* genericVal, IRSpecialize* specializeInst)
    {
        // We need to fold the generic arguments here in order to uniquely identify
        // which specializations need to be generated.
        // The folding of the generic-arguments are deferred until the specialization
        // step here, because the exact value of the generic-arguments can be unknown
        // until the specialization. The exact values of the generic-arguments may come
        // from the other modules, and they will be unknown until linking the modules.
        //
        UInt argCount = specializeInst->getArgCount();
        {
            IRBuilder builder(module);
            for (UInt ii = 0; ii < argCount; ++ii)
            {
                IRUse* argUse = specializeInst->getArgOperand(ii);
                auto originalArg = argUse->get();
                IRInst* foldedArg = tryConstantFoldInst(module, originalArg);
                if (foldedArg == originalArg)
                    continue;

                specializeInst = as<IRSpecialize>(builder.replaceOperand(argUse, foldedArg));
            }
        }

        // We want to see if an existing specialization
        // has already been made. To do that we will construct a key
        // for lookup in the generic specialization context.
        //
        // Our key will consist of the identity of the generic
        // being specialized, and each of the argument values
        // being pased to it. In our hypothetical example of
        // `specialize(g, a, b, c)` the key will then be
        // the array `[g, a, b, c]`.
        //
        Key key;
        key.vals.add(specializeInst->getBase());
        for (UInt ii = 0; ii < argCount; ++ii)
        {
            key.vals.add(specializeInst->getArg(ii));
        }

        {
            // We use our generated key to look for an
            // existing specialization that has been registered.
            // If one is found, our work is done.
            //
            IRInst* specializedVal = nullptr;
            if (genericSpecializations.tryGetValue(key, specializedVal))
                return specializedVal;
        }

        // If no existing specialization is found, we need
        // to create the specialization instead.
        // This mostly amounts to evaluating the generic as
        // if it were a function being called.
        //
        // We will use a free function to do the actual work
        // of evaluating the generic, so that the logic
        // can be re-used in other cases that need to
        // do one-off specialization.
        //
        IRInst* specializedVal = specializeGenericImpl(genericVal, specializeInst, module, this);

        // The body of the specialized generic may expose more specialization opportunities, so
        // we add the children to workList.
        for (auto child : specializedVal->getDecorationsAndChildren())
            addToWorkList(child);

        // The value that was returned from evaluating
        // the generic is the specialized value, and we
        // need to remember it in our dictionary of
        // specializations so that we don't instantiate
        // this generic again for the same arguments.
        //
        genericSpecializations.add(key, specializedVal);

        return specializedVal;
    }

    // The logic for generating a specialization of an IR generic
    // relies on the ability to "evaluate" the code in the body of
    // the generic, but that obviously doesn't work if we don't
    // actually have the full definition for the body.
    //
    // This can arise in particular for builtin operations/types.
    //
    // Before calling `specializeGeneric()` we need to make sure
    // that the generic is actually amenable to specialization,
    // by looking at whether it is a definition or a declaration.
    //
    bool canSpecializeGeneric(IRGeneric* generic)
    {
        // It is possible to have multiple "layers" of generics
        // (e.g., when a generic function is nested in a generic
        // type). Therefore we need to drill down through all
        // of the layers present to see if at the leaf we have
        // something that looks like a definition.
        //
        IRGeneric* g = generic;
        for (;;)
        {
            // Given the generic `g`, we will find the value
            // it appears to return in its body.
            //
            const auto val = findGenericReturnVal(g);
            if (!val)
                return false;

            // If `g` returns an inner generic, then we need
            // to drill down further.
            //
            if (auto nestedGeneric = as<IRGeneric>(val))
            {
                g = nestedGeneric;
                continue;
            }

            // HACK: there are conflicting features in our target intrinsic decorations
            // Some reference generic parameter with the `$G0` syntax (the
            // first generic parameter) while other have a predicate
            // `boolean(T)` where the T is resolved properly in IR lowering. We
            // need to specialize the latter and not the former.
            //
            // The solution is to remove the `$G` syntax and replace that with
            // resolved types too.
            bool intrinsicNeedsSpecialization = false;
            for (const auto dec : val->getDecorations())
            {
                // TODO: We should probably take into account our target to see if the intrinsic
                // applies
                if (const auto intrinsicDec = as<IRTargetIntrinsicDecoration>(dec))
                    intrinsicNeedsSpecialization =
                        intrinsicNeedsSpecialization || intrinsicDec->hasPredicate();
            }

            // We can't specialize a generic if it is marked as
            // being imported from an external module (in which
            // case its definition is not available to us).
            //
            if (!isDefinition(g) && !intrinsicNeedsSpecialization)
                return false;

            // We should never specialize intrinsic types.
            //
            // TODO: This logic assumes that having *any* target
            // intrinsic decoration makes a type skip specialization,
            // even if the decoration isn't applicable to the
            // current target. This should be made true in practice
            // by having the linking step strip/skip decorations
            // that aren't applicable to the chosen target at link time.
            //
            if (as<IRStructType>(val) && val->findDecoration<IRTargetIntrinsicDecoration>())
                return false;

            // Once we've found the leaf value that will be produced
            // after all specialization is complete, we can check
            // whether it looks like a definition or not.
            //
            return isDefinition(val);
        }
    }

    // Now that we know when we can specialize a generic, and how
    // to do it, we can write a subroutine that takes a
    // `specialize(g, a, b, c, ...)` instruction and performs
    // specialization if it is possible.
    //
    bool maybeSpecializeGeneric(IRSpecialize* specInst)
    {
        // We will only attempt to specialize when all of the
        // operands to the `speicalize(...)` instruction are
        // themselves fully specialized.
        //
        if (!areAllOperandsFullySpecialized(specInst))
            return false;

        // The invariant that the arguments are fully specialized
        // should mean that `a, b, c, ...` are in a form that
        // we can work with, but it does *not* guarantee
        // that the `g` operand is something we can work with.
        //
        // We can only perform specialization in the case where
        // the base `g` is a known `generic` instruction.
        //
        auto baseVal = specInst->getBase();
        auto genericVal = as<IRGeneric>(baseVal);
        if (!genericVal)
            return false;

        // We can also only specialize a generic if it
        // represents a definition rather than a declaration.
        //
        if (!canSpecializeGeneric(genericVal))
        {
            // We have to consider a special case here if baseVal is
            // an intrinsic, and contains a custom differential.
            // This is a case where the base cannot be specialized since it has
            // no body, but the custom should be specialized.
            // A better way to handle this would be to grab a reference to the
            // appropriate custom differential, if one exists, at checking time
            // during CheckInvoke() and construct it's specialization args appropriately.
            //
            // For now, we will overwrite the specialization args for the differential
            // using the args for the base.
            //
            auto genericReturnVal = findInnerMostGenericReturnVal(genericVal);
            if (genericReturnVal->findDecoration<IRTargetIntrinsicDecoration>())
            {
                for (auto decor : genericReturnVal->getDecorations())
                {
                    bool specialized = false;
                    if (decor->getOp() == kIROp_ForwardDerivativeDecoration ||
                        decor->getOp() == kIROp_UserDefinedBackwardDerivativeDecoration)
                    {
                        // If we already have a diff func on this specialize, skip.
                        if (const auto specDiffRef = specInst->findDecorationImpl(decor->getOp()))
                        {
                            continue;
                        }

                        auto specDiffFunc = as<IRSpecialize>(decor->getOperand(0));

                        // If the base is specialized, the JVP version must be also be a specialized
                        // generic.
                        //
                        SLANG_RELEASE_ASSERT(specDiffFunc);

                        // Build specialization arguments from specInst.
                        // Note that if we've reached this point, we can safely assume
                        // that our args are fully specialized/concrete.
                        //
                        UCount argCount = specInst->getArgCount();
                        ShortList<IRInst*> args;
                        for (UIndex ii = 0; ii < argCount; ii++)
                            args.add(specInst->getArg(ii));

                        IRBuilder builder(module);

                        // Specialize the custom derivative function type with the original
                        // arguments.
                        builder.setInsertInto(module);
                        auto newDiffFuncType = builder.emitSpecializeInst(
                            builder.getTypeKind(),
                            specDiffFunc->getBase()->getDataType(),
                            argCount,
                            args.getArrayView().getBuffer());

                        // Specialize the custom derivative function with the original arguments.
                        builder.setInsertBefore(specInst);
                        auto newDiffFunc = builder.emitSpecializeInst(
                            (IRType*)newDiffFuncType,
                            specDiffFunc->getBase(),
                            argCount,
                            args.getArrayView().getBuffer());

                        // Add the new spec insts to the list so they get specialized with
                        // the usual logic.
                        //
                        addToWorkList(newDiffFuncType);
                        addToWorkList(newDiffFunc);

                        builder.addDecoration(specInst, decor->getOp(), newDiffFunc);

                        specialized = true;
                    }
                    if (specialized)
                        return true;
                }
            }
            return false;
        }

        // Once we know that specialization is possible,
        // the actual work is fairly simple.
        //
        // First, we find or generate a specialized
        // version of the result of the generic (a specialized
        // type, function, or whatever).
        //
        auto specializedVal = specializeGeneric(genericVal, specInst);

        // Any uses of this `specialize(...)` instruction will
        // become uses of `specializeVal`, so we want to re-consider
        // them for subsequent transformations.
        //
        addUsersToWorkList(specInst);

        // Then we simply replace any uses of the `specialize(...)`
        // instruction with the specialized value and delete
        // the `specialize(...)` instruction from existence.
        //
        specInst->replaceUsesWith(specializedVal);
        specInst->removeAndDeallocate();

        return true;
    }

    // The core of this pass is to look at one instruction
    // at a time, and try to perform whatever specialization
    // is appropriate based on its opcode.
    //
    bool maybeSpecializeInst(IRInst* inst)
    {
        switch (inst->getOp())
        {
        default:
            // By default we assume that specialization is
            // not possible for a given opcode.
            //
            return false;

        case kIROp_Specialize:
            // The logic for specializing a `specialize(...)`
            // instruction has already been elaborated above.
            //
            return maybeSpecializeGeneric(cast<IRSpecialize>(inst));

        case kIROp_LookupWitness:
            // The remaining case we need to consider here for generics
            // is when we have a `lookup_witness_method` instruction
            // that is being applied to a concrete witness table,
            // because we can specialize it to just be a direct
            // reference to the actual witness value from the table.
            //
            return maybeSpecializeWitnessLookup(cast<IRLookupWitnessMethod>(inst));

        case kIROp_Call:
            // When writing functions with existential-type parameters,
            // we need additional support to specialize a callee
            // function based on the concrete type encapsulated in
            // an argument of existential type.
            //
            return maybeSpecializeExistentialsForCall(cast<IRCall>(inst));

            // The specialization of functions with existential-type
            // parameters can create further opportunities for specialization,
            // but in order to realize these we often need to propagate
            // through local simplification on values of existential type.
            //
        case kIROp_ExtractExistentialType:
            return maybeSpecializeExtractExistentialType(inst);
        case kIROp_ExtractExistentialValue:
            return maybeSpecializeExtractExistentialValue(inst);
        case kIROp_ExtractExistentialWitnessTable:
            return maybeSpecializeExtractExistentialWitnessTable(inst);

        case kIROp_Load:
            return maybeSpecializeLoad(as<IRLoad>(inst));

        case kIROp_FieldExtract:
            return maybeSpecializeFieldExtract(as<IRFieldExtract>(inst));
        case kIROp_FieldAddress:
            return maybeSpecializeFieldAddress(as<IRFieldAddress>(inst));

        case kIROp_GetElement:
            return maybeSpecializeGetElement(as<IRGetElement>(inst));
        case kIROp_GetElementPtr:
            return maybeSpecializeGetElementAddress(as<IRGetElementPtr>(inst));

        case kIROp_BindExistentialsType:
            return maybeSpecializeBindExistentialsType(as<IRBindExistentialsType>(inst));

        case kIROp_Expand:
            return maybeSpecializeExpand(as<IRExpand>(inst));

        case kIROp_GetTupleElement:
            return maybeSpecializeFoldableInst(inst);

        case kIROp_TypePack:
        case kIROp_TupleType:
            return maybeSpecializeTypePackOrTupleType(inst);

        case kIROp_MakeValuePack:
        case kIROp_MakeTuple:
            return maybeSpecializeMakeValuePackOrTuple(inst);

        case kIROp_CountOf:
            return maybeSpecializeCountOf(inst);

        case kIROp_Func:

            if (tryExpandParameterPack(as<IRFunc>(inst)))
            {
                addUsersToWorkList(inst);
                return true;
            }
            return false;
        }
    }


    void flattenPackOperand(ShortList<IRInst*>& flattenedList, IRInst* inst)
    {
        if (auto makeValuePack = as<IRMakeValuePack>(inst))
        {
            for (UInt i = 0; i < makeValuePack->getOperandCount(); i++)
            {
                flattenPackOperand(flattenedList, makeValuePack->getOperand(i));
            }
        }
        else if (auto typePack = as<IRTypePack>(inst))
        {
            for (UInt i = 0; i < typePack->getOperandCount(); i++)
            {
                flattenPackOperand(flattenedList, typePack->getOperand(i));
            }
        }
        else
        {
            SLANG_ASSERT(inst);
            flattenedList.add(inst);
        }
    }

    bool maybeSpecializeTypePackOrTupleType(IRInst* inst)
    {
        // If any element of the type pack or tuple is a TypePack, we want to
        // flatten that type pack into the current type pack or tuple.

        bool needProcess = false;
        for (UInt i = 0; i < inst->getOperandCount(); i++)
        {
            if (as<IRTypePack>(inst->getOperand(i)))
            {
                needProcess = true;
                break;
            }
        }
        // If none of the operands are MakeValuePack, there is no need to flatten anything.
        if (!needProcess)
            return false;

        // We will recursively flatten all MakeValuePack operands.
        ShortList<IRInst*> flattendOperands;
        for (UInt i = 0; i < inst->getOperandCount(); i++)
        {
            auto operand = inst->getOperand(i);
            flattenPackOperand(flattendOperands, operand);
        }

        IRBuilder builder(module);
        builder.setInsertBefore(inst);
        IRInst* newInst;
        if (inst->getOp() == kIROp_TypePack)
            newInst = builder.getTypePack(
                flattendOperands.getCount(),
                (IRType* const*)flattendOperands.getArrayView().getBuffer());
        else
            newInst = builder.getTupleType(
                flattendOperands.getCount(),
                (IRType* const*)flattendOperands.getArrayView().getBuffer());
        inst->replaceUsesWith(newInst);
        inst->removeAndDeallocate();
        addUsersToWorkList(newInst);
        return true;
    }

    bool maybeSpecializeMakeValuePackOrTuple(IRInst* inst)
    {
        // If any element of the value pack or tuple is a ValuePack, we want to
        // flatten that value pack into the current value pack or tuple.

        bool needProcess = false;
        for (UInt i = 0; i < inst->getOperandCount(); i++)
        {
            if (as<IRMakeValuePack>(inst->getOperand(i)))
            {
                needProcess = true;
                break;
            }
        }
        // If none of the operands are MakeValuePack, there is no need to flatten anything.
        if (!needProcess)
            return false;

        // We will recursively flatten all MakeValuePack operands.
        ShortList<IRInst*> flattendOperands;
        for (UInt i = 0; i < inst->getOperandCount(); i++)
        {
            auto operand = inst->getOperand(i);
            flattenPackOperand(flattendOperands, operand);
        }

        IRBuilder builder(module);
        builder.setInsertBefore(inst);
        IRInst* newInst = nullptr;
        if (inst->getOp() == kIROp_MakeValuePack)
            newInst = builder.emitMakeValuePack(
                inst->getFullType(),
                flattendOperands.getCount(),
                flattendOperands.getArrayView().getBuffer());
        else
            newInst = builder.emitMakeTuple(
                inst->getFullType(),
                flattendOperands.getCount(),
                flattendOperands.getArrayView().getBuffer());

        inst->replaceUsesWith(newInst);
        inst->removeAndDeallocate();
        addUsersToWorkList(newInst);
        return true;
    }

    bool maybeSpecializeCountOf(IRInst* inst)
    {
        auto operand = inst->getOperand(0);

        // If operand is a value, make sure we are working on its type.

        switch (operand->getOp())
        {
        case kIROp_MakeValuePack:
        case kIROp_MakeTuple:
            operand = operand->getDataType();
            break;
        }

        // We can only figure out the count of a type pack or tuple type.
        switch (operand->getOp())
        {
        case kIROp_TypePack:
        case kIROp_TupleType:
            break;
        default:
            return false;
        }

        // If none of the element type is a TypePack, we can just return the count.
        for (UInt i = 0; i < operand->getOperandCount(); i++)
        {
            switch (operand->getOperand(i)->getOp())
            {
            case kIROp_Param:
            case kIROp_TypePack:
            case kIROp_ExpandTypeOrVal:
                return false;
            }
        }
        IRBuilder builder(module);
        builder.setInsertBefore(inst);
        auto newInst = builder.getIntValue(inst->getDataType(), operand->getOperandCount());
        addUsersToWorkList(inst);
        inst->replaceUsesWith(newInst);
        inst->removeAndDeallocate();
        return true;
    }

    // Specializing lookup on witness tables is a general
    // transformation that helps with both generic and
    // existential-based code.
    //
    bool maybeSpecializeWitnessLookup(IRLookupWitnessMethod* lookupInst)
    {
        // Note: While we currently have named the instruction
        // `lookup_witness_method`, the `method` part is a misnomer
        // and the same instruction can look up *any* interface
        // requirement based on the witness table that provides
        // a conformance, and the "key" that indicates the interface
        // requirement.

        // We can only specialize in the case where the lookup
        // is being done on a concrete witness table, and not
        // the result of a `specialize` instruction or other
        // operation that will yield such a table.
        //
        auto witnessTable = as<IRWitnessTable>(lookupInst->getWitnessTable());
        if (!witnessTable)
        {
            return false;
        }

        // Because we have a concrete witness table, we can
        // use it to look up the IR value that satisfies
        // the given interface requirement.
        //
        auto requirementKey = lookupInst->getRequirementKey();
        auto satisfyingVal = findWitnessVal(witnessTable, requirementKey);

        // We expect to always find a satisfying value, but
        // we will go ahead and code defensively so that
        // we leave "correct" but unspecialized code if
        // we cannot find a concrete value to use.
        //
        if (!satisfyingVal)
            return false;

        // At this point, we know that `satisfyingVal` is what
        // would result from executing this `lookup_witness_method`
        // instruction dynamically, so we can go ahead and
        // replace the original instruction with that value.
        //
        // We also make sure to add any uses of the lookup
        // instruction to our work list, because subsequent
        // simplifications might be possible now.
        //
        addUsersToWorkList(lookupInst);
        lookupInst->replaceUsesWith(satisfyingVal);
        lookupInst->removeAndDeallocate();

        return true;
    }

    bool maybeSpecializeFoldableInst(IRInst* inst)
    {
        auto firstUse = inst->firstUse;
        bool instChanged = peepholeOptimizeInst(targetProgram, module, inst);

        for (auto use = firstUse; use; use = use->nextUse)
        {
            auto user = use->getUser();
            addToWorkList(user);
        }
        return instChanged;
    }

    // The above subroutine needed a way to look up
    // the satisfying value for a given requirement
    // key in a concrete witness table, so let's
    // define that now.
    //
    IRInst* findWitnessVal(IRWitnessTable* witnessTable, IRInst* requirementKey)
    {
        // A witness table is basically just a container
        // for key-value pairs, and so the best we can
        // do for now is a naive linear search.
        //
        for (auto entry : witnessTable->getEntries())
        {
            if (requirementKey == entry->getRequirementKey())
            {
                return entry->getSatisfyingVal();
            }
        }
        return nullptr;
    }
    template<typename TDict>
    void _readSpecializationDictionaryImpl(TDict& dict, IRInst* dictInst)
    {
        int childrenCount = 0;
        for (auto child = dictInst->getFirstChild(); child; child = child->next)
            childrenCount++;
        dict.reserve(Index{1} << Math::Log2Ceil(childrenCount * 2));
        for (auto child : dictInst->getChildren())
        {
            auto item = as<IRSpecializationDictionaryItem>(child);
            if (!item)
                continue;
            IRSimpleSpecializationKey key;
            bool shouldSkip = false;
            for (UInt i = 0; i < item->getOperandCount(); i++)
            {
                if (item->getOperand(i) == nullptr)
                {
                    shouldSkip = true;
                    break;
                }
                if (item->getOperand(i)->getParent() == nullptr)
                {
                    shouldSkip = true;
                    break;
                }
                if (item->getOperand(i)->getOp() == kIROp_undefined)
                {
                    shouldSkip = true;
                    break;
                }
                if (i > 0)
                {
                    key.vals.add(item->getOperand(i));
                }
            }
            if (shouldSkip)
                continue;
            auto value = as<typename std::remove_pointer<typename TDict::ValueType>::type>(
                item->getOperand(0));
            SLANG_ASSERT(value);
            dict[key] = value;
        }
        dictInst->removeAndDeallocate();
    }
    void readSpecializationDictionaries()
    {
        auto moduleInst = module->getModuleInst();
        for (auto child : moduleInst->getChildren())
        {
            switch (child->getOp())
            {
            case kIROp_GenericSpecializationDictionary:
                _readSpecializationDictionaryImpl(genericSpecializations, child);
                break;
            case kIROp_ExistentialFuncSpecializationDictionary:
                _readSpecializationDictionaryImpl(existentialSpecializedFuncs, child);
                break;
            case kIROp_ExistentialTypeSpecializationDictionary:
                _readSpecializationDictionaryImpl(existentialSpecializedStructs, child);
                break;
            default:
                continue;
            }
        }
    }

    template<typename TDict>
    void _writeSpecializationDictionaryImpl(TDict& dict, IROp dictOp, IRInst* moduleInst)
    {
        IRBuilder builder(moduleInst);
        builder.setInsertInto(moduleInst);
        auto dictInst = builder.emitIntrinsicInst(nullptr, dictOp, 0, nullptr);
        builder.setInsertInto(dictInst);
        List<IRInst*> args;
        for (const auto& [key, value] : dict)
        {
            if (!value->parent)
                continue;
            for (auto keyVal : key.vals)
            {
                if (!keyVal->parent)
                    goto next;
            }
            {
                args.clear();
                args.add(value);
                args.addRange(key.vals);
                builder.emitIntrinsicInst(
                    nullptr,
                    kIROp_SpecializationDictionaryItem,
                    (UInt)args.getCount(),
                    args.getBuffer());
            }
        next:;
        }
    }
    void writeSpecializationDictionaries()
    {
        auto moduleInst = module->getModuleInst();
        _writeSpecializationDictionaryImpl(
            genericSpecializations,
            kIROp_GenericSpecializationDictionary,
            moduleInst);
        _writeSpecializationDictionaryImpl(
            existentialSpecializedFuncs,
            kIROp_ExistentialFuncSpecializationDictionary,
            moduleInst);
        _writeSpecializationDictionaryImpl(
            existentialSpecializedStructs,
            kIROp_ExistentialTypeSpecializationDictionary,
            moduleInst);
    }

    // All of the machinery for generic specialization
    // has been defined above, so we will now walk
    // through the flow of the overall specialization pass.
    //
    void processModule()
    {
        // Read specialization dictionary from module if it is defined.
        // This prevents us from generating duplicated specializations
        // when this pass is invoked iteratively.
        readSpecializationDictionaries();

        // The unspecialized IR we receive as input will have
        // `IRBindGlobalGenericParam` instructions that associate
        // each global-scope generic parameter (a type, witness
        // table, or what-have-you) with the value that it should
        // be bound to for the purposes of this code-generation
        // pass.
        //
        // Before doing any other specialization work, we will
        // iterate over these instructions (which may only
        // appear at the global scope) and use them to drive
        // replacement of the given generic type parameter with
        // the desired concrete value.
        //
        // TODO: When we start to support global shader parameters
        // that include existential/interface types, we will need
        // to support a similar specialization step for them.
        //
        specializeGlobalGenericParameters();

        // Now that we've eliminated all cases of global generic parameters,
        // we should now have the properties that:
        //
        // 1. Execution starts in non-generic code, with no unbound
        //    generic parameters in scope.
        //
        // 2. Any case where non-generic code makes use of a generic
        //    type/function, there will be a `specialize` instruction
        //    that specifies both the generic and the (concrete) type
        //    arguments that should be provided to it.
        //
        // The basic approach now is to look for opportunities to apply
        // our specialization rules (e.g., a `specialize` instruction
        // where all the type arguments are concrete types) and then
        // processing any additional opportunities created along the way.
        //
        // We start out simple by putting the root instruction for the
        // module onto our work list.
        //
        for (;;)
        {
            bool iterChanged = false;
            for (;;)
            {
                bool hasSpecialization = false;
                addToWorkList(module->getModuleInst());

                // We will then iterate until our work list goes dry.
                //
                while (workList.getCount() != 0)
                {
                    IRInst* inst = workList.getLast();

                    workList.removeLast();
                    workListSet.remove(inst);

                    if (!inst->getParent() && inst->getOp() != kIROp_Module)
                        continue;

                    // For each instruction we process, we want to perform
                    // a few steps.
                    //
                    // First we will look for all the general-purpose
                    // specialization opportunities (generic specialization,
                    // existential specialization, simplifications, etc.)
                    //
                    if (inst->hasUses() || inst->mightHaveSideEffects() || isWitnessTableType(inst))
                    {
                        hasSpecialization |= maybeSpecializeInst(inst);
                    }

                    // Finally, we need to make our logic recurse through
                    // the whole IR module, so we want to add the children
                    // of any parent instructions to our work list so that
                    // we process them too.
                    //
                    // Note that we are adding the children of an instruction
                    // in reverse order. This is because the way we are
                    // using the work list treats it like a stack (LIFO) and
                    // we know that fully-specialized-ness will tend to flow
                    // top-down through the program, so that we want to process
                    // the children of an instruction in their original order.
                    //
                    for (auto child = inst->getLastChild(); child; child = child->getPrevInst())
                    {
                        // Also note that `addToWorkList` has been written
                        // to avoid adding any instruction that is a descendent
                        // of an IR generic, because we don't actually want
                        // to perform specialization inside of generics.
                        //
                        addToWorkList(child);
                    }
                }
                if (hasSpecialization)
                    iterChanged = true;
                else
                    break;
            }

            if (iterChanged)
            {
                this->changed = true;
                eliminateDeadCode(module->getModuleInst());
                applySparseConditionalConstantPropagationForGlobalScope(this->module, this->sink);
            }

            // Once the work list has gone dry, we should have the invariant
            // that there are no `specialize` instructions inside of non-generic
            // functions that in turn reference a generic type/function unless the generic is for a
            // builtin type/function, or some of the type arguments are unknown at compile time, in
            // which case we will rely on a follow up pass the translate it into a dynamic dispatch
            // function.
            //
            // Now we consider lower lookupWitnessMethod insts into dynamic dispatch calls,
            // which may open up more specialization opportunities.
            //
            if (options.lowerWitnessLookups)
            {
                iterChanged = lowerWitnessLookup(module, sink);
            }

            if (!iterChanged || sink->getErrorCount())
                break;
        }


        // For functions that still have `specialize` uses left, we need to preserve the
        // its specializations in resulting IR so they can be reconstructed when this
        // specialization pass gets invoked again.
        writeSpecializationDictionaries();
    }

    void addInstsToWorkListRec(IRInst* inst)
    {
        addToWorkList(inst);

        for (auto child = inst->getLastChild(); child; child = child->getPrevInst())
        {
            addInstsToWorkListRec(child);
        }
    }

    // Returns true if the call inst represents a call to
    // StructuredBuffer::operator[]/Load/Consume methods.
    bool isBufferLoadCall(IRCall* inst)
    {
        // TODO: We should have something like a `[knownSemantics(...)]` decoration
        // that can identify that an `IRFunc` has semantics that are consistent
        // with a list of compiler-known behaviors. The operand to `knownSemantics`
        // could come from a `KnownSemantics` enumeration or something similar,
        // so that we don't have to make string-based checks here.
        //
        // Note that `[knownSemantics(...)]` would be independent of any targets,
        // and could even apply to functions that are implemented entirely in Slang.

        if (auto targetIntrinsic = findAnyTargetIntrinsicDecoration(inst->getCallee()))
        {
            auto name = targetIntrinsic->getDefinition();
            if (name == ".operator[]" || name == ".Load" || name == ".Consume")
            {
                return true;
            }
        }
        return false;
    }

    /// Used by `maybeSpecailizeBufferLoadCall`, this function returns a new specialized callee that
    /// replaces a `specialize(.operator[], oldType)` to `specialize(.operator[], newElementType)`.
    IRInst* getNewSpecializedBufferLoadCallee(
        IRInst* oldSpecializedCallee,
        IRType* newContainerType,
        IRType* newElementType)
    {
        auto oldSpecialize = cast<IRSpecialize>(oldSpecializedCallee);
        SLANG_ASSERT(oldSpecialize->getArgCount() == 1);
        IRBuilder builder(module);
        builder.setInsertBefore(oldSpecializedCallee);
        auto calleeType = builder.getFuncType(1, &newContainerType, newElementType);
        auto newSpecialize = builder.emitSpecializeInst(
            calleeType,
            oldSpecialize->getBase(),
            1,
            (IRInst**)&newElementType);
        return newSpecialize;
    }

    /// Transform a buffer load intrinsic call.
    /// `bufferLoad(wrapExistential(bufferObj, wrapArgs), loadArgs)` should be transformed into
    /// `wrapExistential(bufferLoad(bufferObj, loadArgs), wragArgs)`.
    /// Returns true if `inst` matches the pattern and the load is transformed, otherwise,
    /// returns false.
    bool maybeSpecializeBufferLoadCall(IRCall* inst)
    {
        if (isBufferLoadCall(inst))
        {
            SLANG_ASSERT(inst->getArgCount() > 0);
            if (auto wrapExistential = as<IRWrapExistential>(inst->getArg(0)))
            {
                if (auto sbType = as<IRHLSLStructuredBufferTypeBase>(
                        wrapExistential->getWrappedValue()->getDataType()))
                {
                    // We are seeing the instruction sequence in the form of
                    // .operator[](wrapExistential(structuredBuffer), idx).
                    // Similar to handling load(wrapExistential(..)) insts,
                    // we need to replace it into wrapExistential(.operator[](sb, idx))
                    auto resultType = inst->getFullType();
                    auto elementType = sbType->getElementType();

                    IRBuilder builder(module);
                    builder.setInsertBefore(inst);

                    ShortList<IRInst*> args;
                    args.add(wrapExistential->getWrappedValue());
                    for (UInt i = 1; i < inst->getArgCount(); i++)
                        args.add(inst->getArg(i));
                    ShortList<IRInst*> slotOperands;
                    UInt slotOperandCount = wrapExistential->getSlotOperandCount();
                    for (UInt ii = 0; ii < slotOperandCount; ++ii)
                    {
                        slotOperands.add(wrapExistential->getSlotOperand(ii));
                    }
                    // The old callee should be in the form of `specialize(.operator[],
                    // IInterfaceType)`, we should update it to be `specialize(.operator[],
                    // elementType)`, so the return type of the load call is `elementType`.

                    // A subscript operation on mutable buffers returns a ptr type instead of a
                    // value type. We need to make sure the pointer-ness is preserved correctly.
                    auto innerResultType = elementType;
                    if (const auto ptrResultType = as<IRPtrType>(inst->getDataType()))
                    {
                        innerResultType = builder.getPtrType(elementType);
                    }
                    auto newCallee = getNewSpecializedBufferLoadCallee(
                        inst->getCallee(),
                        sbType,
                        innerResultType);
                    auto newCall = builder.emitCallInst(
                        innerResultType,
                        newCallee,
                        (UInt)args.getCount(),
                        args.getArrayView().getBuffer());
                    auto newWrapExistential = builder.emitWrapExistential(
                        resultType,
                        newCall,
                        slotOperandCount,
                        slotOperands.getArrayView().getBuffer());
                    inst->replaceUsesWith(newWrapExistential);
                    inst->removeAndDeallocate();
                    addUsersToWorkList(newWrapExistential);
                    SLANG_ASSERT(!wrapExistential->hasUses());
                    wrapExistential->removeAndDeallocate();
                    return true;
                }
            }
        }
        return false;
    }

    // Is the function's actual return type statically known?
    // If so can we specialize the function even if it has no existential parameters.
    //
    bool isExistentialReturnTypeSpecializable(IRFunc* callee)
    {
        if (!as<IRInterfaceType>(callee->getResultType()))
            return false;

        IRInst* witness = nullptr;

        for (auto block : callee->getBlocks())
        {
            if (auto returnInst = as<IRReturn>(block->getTerminator()))
            {
                if (auto makeExistential = as<IRMakeExistential>(returnInst->getVal()))
                {
                    if (witness == nullptr)
                        witness = makeExistential->getWitnessTable();
                    else if (witness != makeExistential->getWitnessTable())
                        return false;
                    if (isChildInstOf(witness, callee))
                        return false;
                }
                else
                {
                    return false;
                }
            }
        }
        return true;
    }

    // Given a `call` instruction in the IR, we need to detect the case
    // where the callee has some interface-type parameter(s) and at the
    // call site it is statically clear what concrete type(s) the arguments
    // will have.
    //
    bool maybeSpecializeExistentialsForCall(IRCall* inst)
    {
        // Handle a special case of `StructuredBuffer.operator[]/Load/Consume`
        // calls first. These calls on builtin generic types should be handled
        // the same way as a `load` inst.
        if (maybeSpecializeBufferLoadCall(inst))
            return false;

        // If any arguments are value packs, we need to flatten them.
        bool isCalleeFullyExpanded = false;
        tryExpandParameterPack(as<IRFunc>(inst->getCallee()), &isCalleeFullyExpanded);
        if (isCalleeFullyExpanded)
        {
            inst = tryExpandArgPack((IRCall*)inst);
        }

        // We can only specialize a call when the callee function is known.
        //
        auto calleeFunc = as<IRFunc>(inst->getCallee());
        if (!calleeFunc)
            return false;

        // Update result type since the callee may have been changed.
        if (inst->getDataType() != calleeFunc->getResultType())
        {
            inst->setFullType(calleeFunc->getResultType());
        }

        // We can only specialize if we have access to a body for the callee.
        //
        if (!calleeFunc->isDefinition())
            return false;

        // We shouldn't bother specializing unless the callee has at least
        // one parameter/return type that has an existential/interface type.
        //
        bool returnTypeNeedSpecialization = isExistentialReturnTypeSpecializable(calleeFunc);
        bool argumentNeedSpecialization = false;
        UInt argCounter = 0;
        for (auto param : calleeFunc->getParams())
        {
            auto arg = inst->getArg(argCounter++);
            if (!isExistentialType(param->getDataType()))
                continue;

            // Is arg in the most simplified form for specialization? If not we are
            // not ready to consider specialization yet.
            if (!isSimplifiedExistentialArg(arg))
                return false;

            // We *cannot* specialize unless the argument value corresponding
            // to such a parameter is one we can specialize.
            //
            if (!canSpecializeExistentialArg(arg))
                continue;

            argumentNeedSpecialization = true;
        }

        // If we never found a parameter or return type worth specializing, we should bail out.
        //
        if (!returnTypeNeedSpecialization && !argumentNeedSpecialization)
            return false;

        // At this point, we believe we *should* and *can* specialize.
        //
        // We need a specialized variant of the callee (with the concrete
        // types substituted in for existential-type parameters), and then
        // we can replace the call site to call the new function instead.
        //
        // Any two call sites where the argument types are the same can
        // re-use the same callee, so we will  cache and re-use the
        // specialized functions that we generate (similar to how generic
        // specialization works). Therefore we will construct a key
        // for use when caching the specialized functions.
        //
        IRSimpleSpecializationKey key;

        // The specialized callee will always depend on the unspecialized
        // function from which it is generated, so we add that to our key.
        //
        key.vals.add(calleeFunc);

        // Also, for any parameter that has an existential type, the
        // specialized function will depend on the concrete type of the
        // argument.
        //
        argCounter = 0;
        for (auto param : calleeFunc->getParams())
        {
            auto arg = inst->getArg(argCounter++);
            if (!isExistentialType(param->getDataType()))
                continue;
            if (auto makeExistential = as<IRMakeExistential>(arg))
            {
                // Note that we use the *type* stored in the
                // existential-type argument, but not anything to
                // do with the particular value (otherwise we'd only
                // be able to re-use the specialized callee for
                // call sites that pass in the exact same argument).
                //
                auto val = makeExistential->getWrappedValue();
                auto valType = val->getDataType();
                if (isCompileTimeConstantType(valType))
                {
                    key.vals.add(valType);

                    // We are also including the witness table in the key.
                    // This isn't required with our current language model,
                    // since a given type can only conform to a given interface
                    // in one way (so there can be only one witness table).
                    // That means that the `valType` and the existential
                    // type of `param` above should uniquely determine
                    // the witness table we see.
                    //
                    // There are forward-looking cases where supporting
                    // "overlapping conformances" could be required, and
                    // there is low incremental cost to future-proofing
                    // this code, so we go ahead and add the witness
                    // table even if it is redundant.
                    //
                    auto witnessTable = makeExistential->getWitnessTable();
                    key.vals.add(witnessTable);
                }
                else
                {
                    key.vals.add(param->getDataType());
                }
            }
            else if (auto wrapExistential = as<IRWrapExistential>(arg))
            {
                auto val = wrapExistential->getWrappedValue();
                auto valType = val->getFullType();
                key.vals.add(valType);

                UInt slotOperandCount = wrapExistential->getSlotOperandCount();
                for (UInt ii = 0; ii < slotOperandCount; ++ii)
                {
                    auto slotOperand = wrapExistential->getSlotOperand(ii);
                    key.vals.add(slotOperand);
                }
            }
            else
            {
                SLANG_UNEXPECTED("unhandled existential argument");
            }
        }

        // Once we've constructed our key, we can try to look for an
        // existing specialization of the callee that we can use.
        //
        IRFunc* specializedCallee = nullptr;
        if (!existentialSpecializedFuncs.tryGetValue(key, specializedCallee))
        {
            // If we didn't find a specialized callee already made, then we
            // will go ahead and create one, and then register it in our cache.
            //
            specializedCallee = createExistentialSpecializedFunc(inst, calleeFunc);
            existentialSpecializedFuncs.add(key, specializedCallee);
        }

        // At this point we have found or generated a specialized version
        // of the callee, and we need to emit a call to it.
        //
        // We will start by constructing the argument list for the new call.
        //
        argCounter = 0;
        ShortList<IRInst*> newArgs;
        for (auto param : calleeFunc->getParams())
        {
            auto arg = inst->getArg(argCounter++);

            // How we handle each argument depends on whether the corresponding
            // parameter has an existential type or not.
            //
            if (!isExistentialType(param->getDataType()))
            {
                // If the parameter doesn't have an existential type, then we
                // don't want to change up the argument we pass at all.
                //
                newArgs.add(arg);
            }
            else
            {
                // Any place where the original function had a parameter of
                // existential type, we will now be passing in the concrete
                // argument value instead of an existential wrapper.
                //
                if (auto makeExistential = as<IRMakeExistential>(arg))
                {
                    auto val = makeExistential->getWrappedValue();
                    auto valType = val->getDataType();
                    if (isCompileTimeConstantType(valType))
                        newArgs.add(val);
                    else
                        newArgs.add(arg);
                }
                else if (auto wrapExistential = as<IRWrapExistential>(arg))
                {
                    auto val = wrapExistential->getWrappedValue();
                    newArgs.add(val);
                }
                else
                {
                    SLANG_UNEXPECTED("missing case for existential argument");
                }
            }
        }

        // Now that we've built up our argument list, it is simple enough
        // to construct a new `call` instruction.
        //
        IRBuilder builderStorage(module);
        auto builder = &builderStorage;

        builder->setInsertBefore(inst);
        auto callResultType = specializedCallee->getResultType();
        IRInst* newCall = builder->emitCallInst(
            callResultType,
            specializedCallee,
            (UInt)newArgs.getCount(),
            newArgs.getArrayView().getBuffer());

        if (as<IRInterfaceType>(inst->getDataType()))
        {
            // If the result of the original call is specialized to a concrete type,
            // we need to wrap it back into an existential type.
            //
            if (auto resultWitnessDecor =
                    specializedCallee->findDecoration<IRResultWitnessDecoration>())
            {
                newCall = builder->emitMakeExistential(
                    inst->getDataType(),
                    newCall,
                    resultWitnessDecor->getWitness());
            }
        }

        // We will completely replace the old `call` instruction with the
        // new one, and will go so far as to transfer any decorations
        // that were attached to the old call over to the new one.
        //
        inst->transferDecorationsTo(newCall);
        inst->replaceUsesWith(newCall);
        inst->removeAndDeallocate();

        // Just in case, we will add any instructions that used the
        // result of this call to our work list for re-consideration.
        // At this moment this shouldn't open up new opportunities
        // for specialization, but we can always play it safe.
        //
        addUsersToWorkList(newCall);

        return true;
    }

    // The above `maybeSpecializeExistentialsForCall` routine needed
    // a few utilities, which we will now define.

    // First, we want to be able to test whether a type (used by
    // a parameter) is an existential type so that we should specialize it.
    //
    bool isExistentialType(IRType* type)
    {
        // An IR-level interface type is always an existential.
        //
        if (as<IRInterfaceType>(type))
            return true;
        if (calcExistentialTypeParamSlotCount(type) != 0)
            return true;
        // Eventually we will also want to handle arrays over
        // existential types, but that will require careful
        // handling in many places.

        return false;
    }

    // Test if a type is compile time constant.
    HashSet<IRInst*> seenTypeSet;
    bool isCompileTimeConstantType(IRInst* inst)
    {
        // TODO: We probably need/want a more robust test here.
        // For now we are just look into the dependency graph of the inst and
        // see if there are any opcodes that are causing problems.
        if (!isInstFullySpecialized(inst))
            return false;

        ShortList<IRInst*> localWorkList;
        seenTypeSet.clear();

        localWorkList.add(inst);
        seenTypeSet.add(inst);

        while (localWorkList.getCount() != 0)
        {
            IRInst* curInst = localWorkList.getLast();

            localWorkList.removeLast();

            switch (curInst->getOp())
            {
            case kIROp_Load:
            case kIROp_Call:
            case kIROp_ExtractExistentialType:
            case kIROp_CreateExistentialObject:
            case kIROp_Param:
                return false;
            default:
                break;
            }

            for (UInt i = 0; i < curInst->getOperandCount(); ++i)
            {
                auto operand = curInst->getOperand(i);
                if (seenTypeSet.add(operand))
                    localWorkList.add(operand);
            }
        }
        return true;
    }


    // Returns true if `inst` is a simplified existential argument ready for specialization.
    bool isSimplifiedExistentialArg(IRInst* inst)
    {
        if (as<IRMakeExistential>(inst))
            return true;
        if (as<IRWrapExistential>(inst))
            return true;
        return false;
    }


    // Similarly, we want to be able to test whether an instruction
    // used as an argument for an existential-type parameter is
    // suitable for use in specialization.
    //
    bool canSpecializeExistentialArg(IRInst* inst)
    {
        // A `makeExistential(v, w)` instruction can be used
        // for specialization, since we have the concrete value `v`
        // (which implicitly determines the concrete type), and
        // the witness table `w.
        //
        if (auto makeExistential = as<IRMakeExistential>(inst))
        {
            // We need to be careful about the type that we'd be specializing
            // to, since it needs to be visible to both the caller and calee.
            //
            // In particular, if the type is the result of a function-local
            // operation like `extractExistentialType`, then we can't possibly
            // specialize the callee, since it wouldn't be able to access
            // the same type (since the type is the result of an instruction in
            // the body of the caller)
            //
            auto concreteVal = makeExistential->getWrappedValue();
            auto concreteType = concreteVal->getDataType();

            return isCompileTimeConstantType(concreteType);
        }

        // A `wrapExistential(v, T0,w0, T1, w1, ...)` instruction
        // is just a generalization of `makeExistential`, so it
        // can apply in the same cases.
        //
        if (as<IRWrapExistential>(inst))
            return true;

        // If we start to specialize functions that take arrays
        // of existentials as input, we will need a strategy to
        // determine arguments suitable for use in specializing
        // them (these would need to be arrays that nominally
        // have an existential element type, but somehow have
        // annotations to indicate that the concrete type
        // underlying the elements in homogeneous).

        return false;
    }

    // In order to cache and re-use functions that have had existential-type
    // parameters specialized, we need storage for the cache.
    //
    Dictionary<IRSimpleSpecializationKey, IRFunc*> existentialSpecializedFuncs;

    // The logic for creating a specialized callee function by plugging
    // in concrete types for existentials is similar to other cases of
    // specialization in the compiler.
    //
    IRFunc* createExistentialSpecializedFunc(IRCall* oldCall, IRFunc* oldFunc)
    {
        // We will make use of the infrastructure for cloning
        // IR code, that is defined in `ir-clone.{h,cpp}`.
        //
        // In order to do the cloning work we need an
        // "environment" that will map old values to
        // their replacements.
        //
        IRCloneEnv cloneEnv;
        cloneEnv.squashChildrenMapping = true;

        // We also need some IR building state, for any
        // new instructions we will emit.
        //
        IRBuilder builderStorage(module);
        auto builder = &builderStorage;

        // To get started, we will create the skeleton of the new
        // specialized function, so newly created insts
        // will be placed in a proper parent.
        //

        IRFunc* newFunc = builder->createFunc();

        builder->setInsertInto(newFunc);
        IRBlock* tempHeaderBlock = builder->emitBlock();

        // We will start out by determining what the parameters
        // of the specialized function should be, based on
        // the parameters of the original, and the concrete
        // type of selected arguments at the call site.
        //
        // Along the way we will build up explicit lists of
        // the parameters, as well as any new instructions
        // that need to be added to the body of the function
        // we generate (as a kind of "prologue"). We build
        // the lists here because we don't yet have a basic
        // block, or even a function, to insert them into.
        //
        ShortList<IRParam*, 16> newParams;
        UInt argCounter = 0;
        for (auto oldParam : oldFunc->getParams())
        {
            auto arg = oldCall->getArg(argCounter++);

            // Given an old parameter, and the argument value at
            // the (old) call site, we need to determine what
            // value should stand in for that parameter in
            // the specialized callee.
            //
            IRInst* replacementVal = nullptr;

            // The trickier case is when we have an existential-type
            // parameter, because we need to extract out the concrete
            // type that is coming from the call site.
            //
            if (auto oldMakeExistential = as<IRMakeExistential>(arg))
            {
                // In this case, the `arg` is `makeExistential(val, witnessTable)`
                // and we know that the specialized call site will just be
                // passing in `val`.
                //
                auto val = oldMakeExistential->getWrappedValue();
                auto witnessTable = oldMakeExistential->getWitnessTable();

                // Our specialized function needs to take a parameter with the
                // same type as `val`, to match the call site(s) that will be
                // created.
                //
                auto valType = val->getFullType();
                if (isCompileTimeConstantType(valType) &&
                    isCompileTimeConstantType(oldParam->getFullType()))
                {
                    if (auto extractExistentialType = as<IRExtractExistentialType>(valType))
                    {
                        valType = extractExistentialType->getOperand(0)->getDataType();
                        auto newParam = builder->createParam(valType);
                        newParams.add(newParam);
                        replacementVal = newParam;
                    }
                    else
                    {
                        auto newParam = builder->createParam(valType);
                        newParams.add(newParam);

                        // Within the body of the function we cannot just use `val`
                        // directly, because the existing code expects an existential
                        // value, including its witness table.
                        //
                        // Therefore we will create a `makeExistential(newParam, witnessTable)`
                        // in the body of the new function and use *that* as the replacement
                        // value for the original parameter (since it will have the
                        // correct existential type, and stores the right witness table).
                        //
                        auto newMakeExistential = builder->emitMakeExistential(
                            oldParam->getFullType(),
                            newParam,
                            witnessTable);
                        replacementVal = newMakeExistential;
                    }
                    cloneEnv.mapOldValToNew.add(oldParam, replacementVal);
                    continue;
                }
            }
            else if (auto oldWrapExistential = as<IRWrapExistential>(arg))
            {
                auto val = oldWrapExistential->getWrappedValue();
                auto valType = val->getFullType();

                auto newParam = builder->createParam(valType);
                newParams.add(newParam);

                // Within the body of the function we cannot just use `val`
                // directly, because the existing code expects an existential
                // value, including its witness table.
                //
                // Therefore we will create a `makeExistential(newParam, witnessTable)`
                // in the body of the new function and use *that* as the replacement
                // value for the original parameter (since it will have the
                // correct existential type, and stores the right witness table).
                //
                auto newWrapExistential = builder->emitWrapExistential(
                    oldParam->getFullType(),
                    newParam,
                    oldWrapExistential->getSlotOperandCount(),
                    oldWrapExistential->getSlotOperands());
                cloneEnv.mapOldValToNew.add(oldParam, newWrapExistential);
                continue;
            }

            // If we go here, then the parameter is either not an existential type,
            // or the argument/parameter is not specialized yet.
            //
            // For first case there is nothing interesting to do. The new function
            // will also have a parameter of the exact same type, and we'll use that
            // instead of the original parameter.
            //
            //
            // For the second case if the argument/parameter is not specialized yet, don't
            // aggressively specialize the parameter.
            //
            // If we specialize the parameter type too early, we will lose the opportunity
            // to specialize the callee later. The principal is to always let the
            // specialization happen at the same time for both on argument and parameter.
            //
            // Note we should not use `createParam` here, because this call won't assign the
            // parent to the new parameter, therefore during the cloning process, some
            // existential related IR inst could be hoisted to the global scope, which is
            // unexpected. Instead, we should use cloneInst here, such that the new
            // parameter will be inserted into the function scope.
            auto newParam = (IRParam*)cloneInst(&cloneEnv, builder, oldParam);
            newParams.add(newParam);
        }

        // The above steps have accomplished the "first phase"
        // of cloning the function (since `IRFunc`s have no
        // operands).
        //
        // We can now use the shared IR cloning infrastructure
        // to perform the second phase of cloning, which will recursively
        // clone any nested decorations, blocks, and instructions.
        //
        cloneInstDecorationsAndChildren(&cloneEnv, builder->getModule(), oldFunc, newFunc);

        //
        // In order to construct the type of the new function, we
        // need to extract the types of all its parameters.
        //
        ShortList<IRType*> newParamTypes;
        for (auto newParam : newParams)
        {
            newParamTypes.add(newParam->getFullType());
        }
        IRType* newFuncType = builder->getFuncType(
            newParamTypes.getCount(),
            newParamTypes.getArrayView().getBuffer(),
            oldFunc->getResultType());
        newFunc->setFullType(newFuncType);

        // Now that the main body of existing instructions have
        // been cloned into the new function, we can go ahead
        // and insert all the parameters and body instructions
        // we built up into the function at the right place.
        //
        // We expect the function to always have at least one
        // block (this was an invariant established before
        // we decided to specialize).
        //
        auto newEntryBlock = as<IRBlock>(cloneEnv.mapOldValToNew[oldFunc->getFirstBlock()]);
        SLANG_ASSERT(newEntryBlock);

        // We expect every valid block to have at least one
        // "ordinary" instruction (it will at least have
        // a terminator like a `return`).
        //
        auto newFirstOrdinary = newEntryBlock->getFirstOrdinaryInst();
        SLANG_ASSERT(newFirstOrdinary);

        // All of our parameters will get inserted before
        // the first ordinary instruction (since the function parameters
        // should come at the start of the first block).
        //
        for (auto newParam : newParams)
        {
            newParam->insertBefore(newFirstOrdinary);
        }

        // All of our new body instructions will *also* be inserted
        // before the first ordinary instruction (but will come
        // *after* the parameters by the order of these two loops).
        //
        for (auto newBodyInst = tempHeaderBlock->getFirstChild(); newBodyInst;)
        {
            auto next = newBodyInst->next;
            newBodyInst->insertBefore(newFirstOrdinary);
            newBodyInst = next;
        }

        // After moving all param and existential insts in tempHeaderBlock
        // it should be empty now and we can remove it.
        tempHeaderBlock->removeAndDeallocate();

        // After all this work we have a valid `newFunc` that has been
        // specialized to match the types at the call site.
        //
        // There might be further opportunities for simplification and
        // specialization in the function body now that we've plugged
        // in some more concrete type information, so we will
        // add the whole function to our work list for subsequent
        // consideration.
        //
        addToWorkList(newFunc);

        simplifyFunc(targetProgram, newFunc, IRSimplificationOptions::getFast(targetProgram));

        if (as<IRInterfaceType>(newFunc->getResultType()))
        {
            // If th result type is an interface type, and all return values are of the same
            // concrete type, we can simplify the function to return the concrete type.
            // We also need to mark the simplfiied function with a result witness decoration
            // so we can rewrite all the callsites into IRMakeExistential using the witness.
            // This is effectively pushing the MakeExistential to the call sites, so optimizations
            // can happen across the function call boundaries.
            IRInst* witnessTable = nullptr;
            IRInst* concreteType = nullptr;
            for (auto block : newFunc->getBlocks())
            {
                if (auto returnInst = as<IRReturn>(block->getTerminator()))
                {
                    if (auto makeExistential = as<IRMakeExistential>(returnInst->getVal()))
                    {
                        if (!concreteType)
                        {
                            concreteType = makeExistential->getWrappedValue()->getDataType();
                            witnessTable = makeExistential->getWitnessTable();
                        }
                        else if (concreteType != makeExistential->getWrappedValue()->getDataType())
                        {
                            concreteType = nullptr;
                            break;
                        }
                        if (isChildInstOf(witnessTable, newFunc))
                        {
                            concreteType = nullptr;
                            break;
                        }
                    }
                    else
                    {
                        concreteType = nullptr;
                        break;
                    }
                }
            }
            if (concreteType)
            {
                for (auto block : newFunc->getBlocks())
                {
                    if (auto returnInst = as<IRReturn>(block->getTerminator()))
                    {
                        if (auto makeExistential = as<IRMakeExistential>(returnInst->getVal()))
                        {
                            returnInst->setOperand(0, makeExistential->getWrappedValue());
                        }
                    }
                }
                builder->addResultWitnessDecoration(newFunc, witnessTable);
                fixUpFuncType(newFunc, (IRType*)concreteType);
            }
        }

        return newFunc;
    }

    // When we've specialized a function with an interface-type parameter
    // we will still end up with a `makeExistential` operation in its
    // body, which could impede subequent specializations.
    //
    // For example, if we have the following after specialization:
    //
    //      e = makeExistential(v, w1);
    //      w2 = extractExistentialWitnessTable(e);
    //      f = lookup_witness_method(w2, k);
    //      call(f, ...);
    //
    // We cannot then specialize the lookup for `f` in this code as written,
    // but it seems obvious that we could replace `w2` with `w1` and maybe
    // get further along.
    //
    // In order to set up further specialization opportunities we need
    // to implement a few simplification rules around operations that
    // extract from an existential, when their operand is a `makeExistential`.
    //
    // Let's start with the routine for the case above of extracting
    // a witness table.
    //
    bool maybeSpecializeExtractExistentialWitnessTable(IRInst* inst)
    {
        // We know `inst` is `extractExistentialWitnessTable(existentialArg)`.
        //
        auto existentialArg = inst->getOperand(0);

        if (auto makeExistential = as<IRMakeExistential>(existentialArg))
        {
            // In this case we know `inst` is:
            //
            //      extractExistentialWitnessTable(makeExistential(..., witnessTable))
            //
            // and we can just simplify that to `witnessTable`.
            //
            auto witnessTable = makeExistential->getWitnessTable();

            // Anything that used this instruction is now a candidate for
            // further simplification or specialization (e.g., one of
            // the users of this instruction could be a `lookup_witness_method`
            // that we can now specialize).
            //
            addUsersToWorkList(inst);

            inst->replaceUsesWith(witnessTable);
            inst->removeAndDeallocate();
            return true;
        }
        return false;
    }

    // The cases for simplifying `extractExistentialValue` is more or less the same
    // as for witness tables.
    //
    bool maybeSpecializeExtractExistentialValue(IRInst* inst)
    {
        // We know `inst` is `extractExistentialValue(existentialArg)`.
        //
        auto existentialArg = inst->getOperand(0);
        if (auto makeExistential = as<IRMakeExistential>(existentialArg))
        {
            // Now we know `inst` is:
            //
            //      extractExistentialValue(makeExistential(val, ...))
            //
            // and we can just simplify that to `val`.
            //
            auto val = makeExistential->getWrappedValue();

            addUsersToWorkList(inst);

            inst->replaceUsesWith(val);
            inst->removeAndDeallocate();
            return true;
        }
        return false;
    }

    // The cases for simplifying `extractExistentialType` is more or less the same
    // as for witness tables.
    //
    bool maybeSpecializeExtractExistentialType(IRInst* inst)
    {
        // We know `inst` is `extractExistentialValue(existentialArg)`.
        //
        auto existentialArg = inst->getOperand(0);
        if (auto makeExistential = as<IRMakeExistential>(existentialArg))
        {
            // Now we know `inst` is:
            //
            //      extractExistentialType(makeExistential(val, ...))
            //
            // and we can just simplify that to type type of `val`.
            //
            auto val = makeExistential->getWrappedValue();
            auto valType = val->getFullType();

            addUsersToWorkList(inst);

            inst->replaceUsesWith(valType);
            inst->removeAndDeallocate();
            return true;
        }
        return false;
    }

    bool maybeSpecializeLoad(IRLoad* inst)
    {
        auto ptrArg = inst->ptr.get();

        if (auto wrapInst = as<IRWrapExistential>(ptrArg))
        {
            // We have an instruction of the form `load(wrapExistential(val, ...))`
            //
            auto val = wrapInst->getWrappedValue();

            // We know what type we are expected to
            // produce (which should be the pointed-to
            // type for whatever the type of the
            // `wrapExistential` is).
            //
            auto resultType = inst->getFullType();

            IRBuilder builder(module);
            builder.setInsertBefore(inst);

            // We'd *like* to replace this instruction with
            // `wrapExistential(load(val))` instead, since that
            // will enable subsequent specializations.
            //
            // To do that, we need to be able to determine
            // the type that `load(val)` should return.
            //
            auto elementType = tryGetPointedToType(&builder, val->getDataType());
            if (!elementType)
                return false;


            ShortList<IRInst*> slotOperands;
            UInt slotOperandCount = wrapInst->getSlotOperandCount();
            for (UInt ii = 0; ii < slotOperandCount; ++ii)
            {
                slotOperands.add(wrapInst->getSlotOperand(ii));
            }

            auto newLoadInst = builder.emitLoad(elementType, val);
            auto newWrapExistentialInst = builder.emitWrapExistential(
                resultType,
                newLoadInst,
                slotOperandCount,
                slotOperands.getArrayView().getBuffer());

            addUsersToWorkList(inst);

            inst->replaceUsesWith(newWrapExistentialInst);
            inst->removeAndDeallocate();
            return true;
        }
        return false;
    }

    UInt calcExistentialBoxSlotCount(IRType* type)
    {
    top:
        if (as<IRBoundInterfaceType>(type))
        {
            return 2;
        }
        else if (as<IRInterfaceType>(type))
        {
            return 2;
        }
        else if (auto ptrType = as<IRPtrTypeBase>(type))
        {
            type = ptrType->getValueType();
            goto top;
        }
        else if (auto ptrLikeType = as<IRPointerLikeType>(type))
        {
            type = ptrLikeType->getElementType();
            goto top;
        }
        else if (auto arrayType = as<IRArrayTypeBase>(type))
        {
            type = arrayType->getElementType();
            goto top;
        }
        else if (auto sbType = as<IRHLSLStructuredBufferTypeBase>(type))
        {
            type = sbType->getElementType();
            goto top;
        }
        else if (auto attributedType = as<IRAttributedType>(type))
        {
            type = attributedType->getBaseType();
            goto top;
        }
        else if (auto structType = as<IRStructType>(type))
        {
            UInt count = 0;
            for (auto field : structType->getFields())
            {
                count += calcExistentialBoxSlotCount(field->getFieldType());
            }
            return count;
        }
        else
        {
            return 0;
        }
    }

    bool maybeSpecializeFieldExtract(IRFieldExtract* inst)
    {
        auto baseArg = inst->getBase();
        auto fieldKey = inst->getField();

        if (auto wrapInst = as<IRWrapExistential>(baseArg))
        {
            // We have `getField(wrapExistential(val, ...), fieldKey)`
            //
            auto val = wrapInst->getWrappedValue();

            // We know what type we are expected to produce.
            //
            auto resultType = inst->getFullType();

            IRBuilder builder(module);
            builder.setInsertBefore(inst);

            // We'd *like* to replace this instruction with
            // `wrapExistential(getField(val, fieldKey), ...)` instead, since that
            // will enable subsequent specializations.
            //
            // To do that, we need to figure out:
            //
            // 1. What type that inner `getField` would return (what
            // is the type of the `fieldKey` field in `val`?)
            //
            // 2. Which of the existential slot operands in `...` there
            // actually apply to the given field.
            //

            // To determine these things, we need the type of
            // `val` to be a structure type so that we can look
            // up the field corresponding to `fieldKey`.
            //
            auto valType = val->getDataType();
            auto valStructType = as<IRStructType>(valType);
            if (!valStructType)
                return false;

            UInt slotOperandOffset = 0;

            IRStructField* foundField = nullptr;
            for (auto valField : valStructType->getFields())
            {
                if (valField->getKey() == fieldKey)
                {
                    foundField = valField;
                    break;
                }

                slotOperandOffset += calcExistentialBoxSlotCount(valField->getFieldType());
            }

            if (!foundField)
                return false;

            auto foundFieldType = foundField->getFieldType();

            ShortList<IRInst*> slotOperands;
            UInt slotOperandCount = calcExistentialBoxSlotCount(foundFieldType);

            for (UInt ii = 0; ii < slotOperandCount; ++ii)
            {
                slotOperands.add(wrapInst->getSlotOperand(slotOperandOffset + ii));
            }

            auto newGetField = builder.emitFieldExtract(foundFieldType, val, fieldKey);

            auto newWrapExistentialInst = builder.emitWrapExistential(
                resultType,
                newGetField,
                slotOperandCount,
                slotOperands.getArrayView().getBuffer());

            addUsersToWorkList(inst);
            inst->replaceUsesWith(newWrapExistentialInst);
            inst->removeAndDeallocate();
            return true;
        }
        return false;
    }


    bool maybeSpecializeFieldAddress(IRFieldAddress* inst)
    {
        auto baseArg = inst->getBase();
        auto fieldKey = inst->getField();

        if (auto wrapInst = as<IRWrapExistential>(baseArg))
        {
            // We have `getFieldAddr(wrapExistential(val, ...), fieldKey)`
            //
            auto val = wrapInst->getWrappedValue();

            // We know what type we are expected to produce.
            //
            auto resultType = inst->getFullType();

            IRBuilder builder(module);
            builder.setInsertBefore(inst);

            // We'd *like* to replace this instruction with
            // `wrapExistential(getFieldAddr(val, fieldKey), ...)` instead, since that
            // will enable subsequent specializations.
            //
            // To do that, we need to figure out:
            //
            // 1. What type that inner `getFieldAddr` would return (what
            // is the type of the `fieldKey` field in `val`?)
            //
            // 2. Which of the existential slot operands in `...` there
            // actually apply to the given field.
            //

            // To determine these things, we need the type of
            // `val` to be a (pointer to a) structure type so that we can look
            // up the field corresponding to `fieldKey`.
            //
            auto valType = tryGetPointedToType(&builder, val->getDataType());
            if (!valType)
                return false;

            auto valStructType = as<IRStructType>(valType);
            if (!valStructType)
                return false;

            UInt slotOperandOffset = 0;

            IRStructField* foundField = nullptr;
            for (auto valField : valStructType->getFields())
            {
                if (valField->getKey() == fieldKey)
                {
                    foundField = valField;
                    break;
                }

                slotOperandOffset += calcExistentialBoxSlotCount(valField->getFieldType());
            }

            if (!foundField)
                return false;

            auto foundFieldType = foundField->getFieldType();

            ShortList<IRInst*> slotOperands;
            UInt slotOperandCount = calcExistentialBoxSlotCount(foundFieldType);

            for (UInt ii = 0; ii < slotOperandCount; ++ii)
            {
                slotOperands.add(wrapInst->getSlotOperand(slotOperandOffset + ii));
            }

            auto newGetFieldAddr =
                builder.emitFieldAddress(builder.getPtrType(foundFieldType), val, fieldKey);

            auto newWrapExistentialInst = builder.emitWrapExistential(
                resultType,
                newGetFieldAddr,
                slotOperandCount,
                slotOperands.getArrayView().getBuffer());

            addUsersToWorkList(inst);
            inst->replaceUsesWith(newWrapExistentialInst);
            inst->removeAndDeallocate();
            return true;
        }
        return false;
    }

    bool maybeSpecializeGetElement(IRGetElement* inst)
    {
        auto baseArg = inst->getBase();
        if (auto wrapInst = as<IRWrapExistential>(baseArg))
        {
            // We have `getElement(wrapExistential(val, ...), index)`
            // We need to replace this instruction with
            // `wrapExistential(getElement(val, index), ...)`
            auto index = inst->getIndex();

            auto val = wrapInst->getWrappedValue();
            auto resultType = inst->getFullType();

            IRBuilder builder(module);
            builder.setInsertBefore(inst);

            auto elementType = cast<IRArrayTypeBase>(val->getDataType())->getElementType();

            ShortList<IRInst*> slotOperands;
            UInt slotOperandCount = wrapInst->getSlotOperandCount();

            for (UInt ii = 0; ii < slotOperandCount; ++ii)
            {
                slotOperands.add(wrapInst->getSlotOperand(ii));
            }

            auto newGetElement = builder.emitElementExtract(elementType, val, index);

            auto newWrapExistentialInst = builder.emitWrapExistential(
                resultType,
                newGetElement,
                slotOperandCount,
                slotOperands.getArrayView().getBuffer());

            addUsersToWorkList(inst);
            inst->replaceUsesWith(newWrapExistentialInst);
            inst->removeAndDeallocate();
            return true;
        }
        return false;
    }

    bool maybeSpecializeGetElementAddress(IRGetElementPtr* inst)
    {
        auto baseArg = inst->getBase();
        if (auto wrapInst = as<IRWrapExistential>(baseArg))
        {
            // We have `getElementPtr(wrapExistential(val, ...), index)`
            // We need to replace this instruction with
            // `wrapExistential(getElementPtr(val, index), ...)`
            auto index = inst->getIndex();

            auto val = wrapInst->getWrappedValue();

            auto resultType = inst->getFullType();

            IRBuilder builder(module);
            builder.setInsertBefore(inst);

            ShortList<IRInst*> slotOperands;
            UInt slotOperandCount = wrapInst->getSlotOperandCount();

            for (UInt ii = 0; ii < slotOperandCount; ++ii)
            {
                slotOperands.add(wrapInst->getSlotOperand(ii));
            }

            auto newElementAddr = builder.emitElementAddress(val, index);

            auto newWrapExistentialInst = builder.emitWrapExistential(
                resultType,
                newElementAddr,
                slotOperandCount,
                slotOperands.getArrayView().getBuffer());

            addUsersToWorkList(inst);
            inst->replaceUsesWith(newWrapExistentialInst);
            inst->removeAndDeallocate();
            return true;
        }
        return false;
    }

    UInt calcExistentialTypeParamSlotCount(IRType* type)
    {
    top:
        if (as<IRInterfaceType>(type))
        {
            return 2;
        }
        else if (auto ptrLikeType = as<IRPointerLikeType>(type))
        {
            type = ptrLikeType->getElementType();
            goto top;
        }
        else if (auto sbType = as<IRHLSLStructuredBufferTypeBase>(type))
        {
            type = sbType->getElementType();
            goto top;
        }
        else if (auto attributedType = as<IRAttributedType>(type))
        {
            type = attributedType->getBaseType();
            goto top;
        }
        else if (auto structType = as<IRStructType>(type))
        {
            UInt count = 0;
            for (auto field : structType->getFields())
            {
                count += calcExistentialTypeParamSlotCount(field->getFieldType());
            }
            return count;
        }
        else
        {
            return 0;
        }
    }

    Dictionary<IRSimpleSpecializationKey, IRStructType*> existentialSpecializedStructs;

    bool maybeSpecializeBindExistentialsType(IRBindExistentialsType* type)
    {
        auto baseType = type->getBaseType();
        UInt slotOperandCount = type->getExistentialArgCount();

        IRBuilder builder(module);
        builder.setInsertBefore(type);

        if (auto baseInterfaceType = as<IRInterfaceType>(baseType))
        {
            // We always expect two slot operands, one for the concrete type
            // and one for the witness table.
            //
            SLANG_ASSERT(slotOperandCount == 2);
            if (slotOperandCount < 2)
                return false;

            auto concreteType = (IRType*)type->getExistentialArg(0);
            auto witnessTable = type->getExistentialArg(1);
            auto newVal =
                builder.getBoundInterfaceType(baseInterfaceType, concreteType, witnessTable);

            addUsersToWorkList(type);
            type->replaceUsesWith(newVal);
            type->removeAndDeallocate();
            return true;
        }
        else if (
            as<IRPointerLikeType>(baseType) || as<IRHLSLStructuredBufferTypeBase>(baseType) ||
            as<IRArrayTypeBase>(baseType) || as<IRAttributedType>(baseType))
        {
            // A `BindExistentials<P<T>, ...>` can be simplified to
            // `P<BindExistentials<T, ...>>` when `P` is a pointer-like
            // type constructor.
            //
            IRType* baseElementType = nullptr;
            if (auto basePtrLikeType = as<IRPointerLikeType>(baseType))
                baseElementType = basePtrLikeType->getElementType();
            else if (auto arrayType = as<IRArrayTypeBase>(baseType))
                baseElementType = arrayType->getElementType();
            else if (auto baseSBType = as<IRHLSLStructuredBufferTypeBase>(baseType))
                baseElementType = baseSBType->getElementType();
            else if (auto baseAttrType = as<IRAttributedType>(baseType))
                baseElementType = baseAttrType->getBaseType();

            IRInst* wrappedElementType = builder.getBindExistentialsType(
                baseElementType,
                slotOperandCount,
                type->getExistentialArgs());
            // Create a new type inst where the first operand is replaced
            // with wrappedElementType.
            ShortList<IRInst*> operands;
            operands.add(wrappedElementType);
            for (UInt i = 1; i < baseType->getOperandCount(); i++)
                operands.add(baseType->getOperand(i));
            IRInst* newPtrLikeType = builder.getType(
                baseType->getOp(),
                operands.getCount(),
                operands.getArrayView().getBuffer());
            addUsersToWorkList(type);
            addToWorkList(newPtrLikeType);
            addToWorkList(wrappedElementType);

            type->replaceUsesWith(newPtrLikeType);
            type->removeAndDeallocate();
            return true;
        }
        else if (auto baseStructType = as<IRStructType>(baseType))
        {
            // In order to bind a `struct` type we will generate
            // a new specialized `struct` type on demand and then
            // cache and re-use it.
            //
            // We don't want to start specializing here unless
            // all the operand types (and witness tables) we
            // will be specializing to are themselves fully
            // specialized, so that we can be sure that we
            // have a unique type.
            //
            if (!areAllOperandsFullySpecialized(type))
                return false;

            // Now we we check to see if we've already created
            // a specialized struct type or not.
            //
            IRSimpleSpecializationKey key;
            key.vals.add(baseStructType);
            for (UInt ii = 0; ii < slotOperandCount; ++ii)
            {
                key.vals.add(type->getExistentialArg(ii));
            }

            IRStructType* newStructType = nullptr;
            addUsersToWorkList(type);

            if (!existentialSpecializedStructs.tryGetValue(key, newStructType))
            {
                builder.setInsertBefore(baseStructType);
                newStructType = builder.createStructType();
                addToWorkList(newStructType);

                auto fieldSlotArgs = type->getExistentialArgs();

                for (auto oldField : baseStructType->getFields())
                {
                    // TODO: we need to figure out which of the specialization arguments
                    // apply to this field...

                    auto oldFieldType = oldField->getFieldType();
                    auto fieldSlotArgCount = calcExistentialTypeParamSlotCount(oldFieldType);

                    auto newFieldType = builder.getBindExistentialsType(
                        oldFieldType,
                        fieldSlotArgCount,
                        fieldSlotArgs);

                    addToWorkList(newFieldType);

                    fieldSlotArgs += fieldSlotArgCount;

                    builder.createStructField(newStructType, oldField->getKey(), newFieldType);
                }

                existentialSpecializedStructs.add(key, newStructType);
            }

            type->replaceUsesWith(newStructType);
            type->removeAndDeallocate();
            return true;
        }
        return false;
    }

    IRInst* specializeExpandChildInst(
        IRCloneEnv& cloneEnv,
        IRBuilder* builder,
        IRInst* childInst,
        UInt index)
    {
        IRCloneEnv freshEnv;
        IRCloneEnv* subEnv = &cloneEnv;
        switch (childInst->getOp())
        {
        case kIROp_Expand:
            {
                subEnv = &freshEnv;
                break;
            }
        }
        auto newInst = cloneInst(subEnv, builder, childInst);
        if (newInst != childInst)
            addToWorkList(newInst);
        subEnv->mapOldValToNew[childInst] = newInst;
        IRBuilder subBuilder(*builder);
        subBuilder.setInsertInto(newInst);
        for (auto child : childInst->getChildren())
        {
            specializeExpandChildInst(*subEnv, &subBuilder, child, index);
        }
        return newInst;
    }

    // A helper function to emit a MakeWitnessPack, MakeTypePack or MakeValuePack inst from
    // a collection of elements, dependending on `type`.
    //
    IRInst* makeSpecializedPack(IRBuilder& builder, IRType* type, ArrayView<IRInst*> elements)
    {
        IRInst* resultPack = nullptr;
        if (as<IRWitnessTableType>(type))
        {
            List<IRType*> types;
            for (auto element : elements)
                types.add(element->getDataType());
            auto newTypePack = builder.getTypePack(elements.getCount(), types.getBuffer());
            resultPack = builder.emitMakeWitnessPack(newTypePack, elements);
        }
        else if (as<IRTypeKind>(type) || as<IRTypeType>(type))
        {
            auto newTypePack =
                builder.getTypePack(elements.getCount(), (IRType* const*)elements.getBuffer());
            resultPack = newTypePack;
        }
        else
        {
            resultPack = builder.emitMakeValuePack((UInt)elements.getCount(), elements.getBuffer());
        }
        return resultPack;
    }

    bool maybeSpecializeExpand(IRExpand* expandInst)
    {
        if (expandInst->getCaptureCount() == 0)
            return false;

        for (UInt i = 0; i < expandInst->getCaptureCount(); i++)
        {
            if (!as<IRTypePack>(expandInst->getCapture(i)))
                return false;
        }

        IRBuilder builder(expandInst);
        builder.setInsertBefore(expandInst);
        List<IRInst*> elements;
        UInt elementCount = 0;
        if (auto firstTypePack = as<IRTypePack>(expandInst->getCapture(0)))
        {
            elementCount = firstTypePack->getOperandCount();
        }
        if (elementCount == 0)
        {
            auto resultPack =
                makeSpecializedPack(builder, expandInst->getDataType(), elements.getArrayView());
            expandInst->replaceUsesWith(resultPack);
            expandInst->removeAndDeallocate();
            addUsersToWorkList(resultPack);
            return true;
        }

        bool isMultiBlock = as<IRYield>(expandInst->getFirstBlock()->getTerminator()) == nullptr;

        for (UInt i = 0; i < elementCount; i++)
        {
            IRCloneEnv cloneEnv;
            IRBuilder subBuilder = builder;
            IRBlock* mergeBlock = nullptr;
            if (isMultiBlock)
            {
                IRBlock* firstBlock = nullptr;
                for (auto childBlock : expandInst->getBlocks())
                {
                    auto newBlock = subBuilder.emitBlock();
                    if (!firstBlock)
                        firstBlock = newBlock;
                    cloneEnv.mapOldValToNew[childBlock] = newBlock;
                }

                builder.emitBranch(firstBlock);

                mergeBlock = subBuilder.emitBlock();
                builder.setInsertInto(mergeBlock);
            }

            auto indexParam = expandInst->getFirstBlock()->getFirstParam();
            SLANG_ASSERT(indexParam);
            cloneEnv.mapOldValToNew[indexParam] =
                subBuilder.getIntValue(subBuilder.getIntType(), i);

            for (auto childBlock : expandInst->getBlocks())
            {
                if (isMultiBlock)
                {
                    auto newBlock = cloneEnv.mapOldValToNew[childBlock];
                    subBuilder.setInsertInto(newBlock);
                }
                for (auto child : childBlock->getChildren())
                {
                    if (as<IRYield>(child))
                    {
                        auto currentResult = child->getOperand(0);
                        currentResult = findCloneForOperand(&cloneEnv, currentResult);
                        elements.add(currentResult);
                        if (isMultiBlock)
                            subBuilder.emitBranch(mergeBlock);
                        continue;
                    }
                    specializeExpandChildInst(cloneEnv, &subBuilder, child, i);
                    addToWorkList(childBlock);
                }
            }
        }

        IRInst* resultPack =
            makeSpecializedPack(builder, expandInst->getDataType(), elements.getArrayView());
        if (isMultiBlock)
        {
            auto currentBlock = builder.getBlock();
            for (auto nextInst = expandInst->next; nextInst;)
            {
                auto next = nextInst->next;
                nextInst->insertAtEnd(currentBlock);
                nextInst = next;
            }
        }
        addUsersToWorkList(expandInst);
        expandInst->replaceUsesWith(resultPack);
        expandInst->removeAndDeallocate();
        return true;
    }

    // The handling of specialization for global generic type
    // parameters involves searching for all `bind_global_generic_param`
    // instructions in the input module.
    //
    void specializeGlobalGenericParameters()
    {
        auto moduleInst = module->getModuleInst();
        for (auto inst : moduleInst->getChildren())
        {
            // We only want to consider the `bind_global_generic_param`
            // instructions, and ignore everything else.
            //
            auto bindInst = as<IRBindGlobalGenericParam>(inst);
            if (!bindInst)
                continue;

            // HACK: Our current front-end emit logic can end up emitting multiple
            // `bind_global_generic_param` instructions for the same parameter. This is
            // a buggy behavior, but a real fix would require refactoring the way
            // global generic arguments are specified today.
            //
            // For now we will do a sanity check to detect parameters that
            // have already been specialized.
            //
            if (!as<IRGlobalGenericParam>(bindInst->getOperand(0)))
            {
                // The "parameter" operand is no longer a parameter, so it
                // seems things must have been specialized already.
                //
                continue;
            }

            // The actual logic for applying the substitution is
            // almost trivial: we will replace any uses of the
            // global generic parameter with its desired value.
            //
            auto param = bindInst->getParam();
            auto val = bindInst->getVal();
            param->replaceUsesWith(val);
        }
        {
            // Now that we've replaced any uses of global generic
            // parameters, we will do a second pass to remove
            // the parameters and any `bind_global_generic_param`
            // instructions, since both should be dead/unused.
            //
            IRInst* next = nullptr;
            for (auto inst = moduleInst->getFirstChild(); inst; inst = next)
            {
                next = inst->getNextInst();

                switch (inst->getOp())
                {
                default:
                    break;

                case kIROp_GlobalGenericParam:
                case kIROp_BindGlobalGenericParam:
                    // A `bind_global_generic_param` instruction should
                    // have no uses in the first place, and all the global
                    // generic parameters should have had their uses replaced.
                    //
                    SLANG_ASSERT(!inst->firstUse);
                    inst->removeAndDeallocate();
                    break;
                }
            }
        }
    }


    // If `func` has any parameters whose types are `IRTypePack`, then we will expand them
    // into multiple parameters, so that the function has no parameters of type `IRTypePack`.
    // returns true if changes are made.
    // For example, this function turns `int f(TypePack<int, float> v)` into
    // ```
    // int f(int v0, float v1)
    // {
    //     v = MakeValuePack(v0,. v1);
    //     ...
    // }
    // ```
    //
    bool tryExpandParameterPack(IRFunc* func, bool* outIsFullyExpanded = nullptr)
    {
        if (!func)
            return false;
        if (outIsFullyExpanded)
            *outIsFullyExpanded = true;
        ShortList<IRInst*> params;
        for (auto param : func->getParams())
        {
            if (as<IRTypePack>(param->getDataType()))
                params.add(param);
            if (as<IRExpand>(param->getDataType()))
            {
                if (outIsFullyExpanded)
                    *outIsFullyExpanded = false;
                return false;
            }
        }
        if (params.getCount() == 0)
            return false;

        IRBuilder builder(func);
        for (auto param : params)
        {
            builder.setInsertBefore(param);
            auto typePack = as<IRTypePack>(param->getDataType());
            ShortList<IRInst*> newParams;
            for (UInt i = 0; i < typePack->getOperandCount(); i++)
            {
                auto newParam = builder.createParam((IRType*)typePack->getOperand(i));
                newParam->insertBefore(param);
                newParams.add(newParam);
            }
            setInsertBeforeOrdinaryInst(&builder, param);
            auto val = builder.emitMakeValuePack(
                typePack,
                (UInt)newParams.getCount(),
                newParams.getArrayView().getBuffer());
            param->replaceUsesWith(val);
            param->removeAndDeallocate();
            addUsersToWorkList(val);
        }

        fixUpFuncType(func);
        return true;
    }

    // If any arguments in a call is a value pack, we will expand them into the argument list,
    // so that the call has no arguments of type `IRTypePack`.
    // For example, we will turn `f(MakeValuePack(a, b))` into `f(a, b)`.
    //
    IRCall* tryExpandArgPack(IRCall* call)
    {
        bool anyArgPack = false;
        for (UInt i = 0; i < call->getArgCount(); i++)
        {
            auto arg = call->getArg(i);
            if (as<IRTypePack>(arg->getDataType()))
            {
                anyArgPack = true;
                break;
            }
        }
        if (!anyArgPack)
            return call;
        IRBuilder builder(call);
        builder.setInsertBefore(call);
        List<IRInst*> newArgs;
        for (UInt i = 0; i < call->getArgCount(); i++)
        {
            auto arg = call->getArg(i);
            if (auto typePack = as<IRTypePack>(arg->getDataType()))
            {
                for (UInt elementIndex = 0; elementIndex < typePack->getOperandCount();
                     elementIndex++)
                {
                    auto newArg = builder.emitGetTupleElement(
                        (IRType*)typePack->getOperand(elementIndex),
                        arg,
                        elementIndex);
                    newArgs.add(newArg);
                }
            }
            else
            {
                newArgs.add(arg);
            }
        }
        auto newCall =
            builder.emitCallInst(call->getFullType(), call->getCallee(), newArgs.getArrayView());
        call->replaceUsesWith(newCall);
        call->transferDecorationsTo(newCall);
        call->removeAndDeallocate();
        return newCall;
    }
};

bool specializeModule(
    TargetProgram* target,
    IRModule* module,
    DiagnosticSink* sink,
    SpecializationOptions options)
{
    SLANG_PROFILE;
    SpecializationContext context(module, target, options);
    context.sink = sink;
    context.processModule();
    return context.changed;
}

void finalizeSpecialization(IRModule* module)
{
    // Go through all the top-level children of module's module inst,
    // and remove any specialization dictionary insts.
    //

    auto moduleInst = module->getModuleInst();
    IRInst* next = nullptr;
    for (auto inst = moduleInst->getFirstChild(); inst; inst = next)
    {
        next = inst->getNextInst();

        switch (inst->getOp())
        {
        default:
            break;

        case kIROp_StructKey:
        case kIROp_Func:
            for (auto decor = inst->getFirstDecoration(); decor;)
            {
                auto nextDecor = decor->getNextDecoration();
                switch (decor->getOp())
                {
                case kIROp_DispatchFuncDecoration:
                case kIROp_ResultWitnessDecoration:
                    decor->removeAndDeallocate();
                    break;
                default:
                    break;
                }
                decor = nextDecor;
            }
            break;

        case kIROp_ExistentialFuncSpecializationDictionary:
        case kIROp_ExistentialTypeSpecializationDictionary:
        case kIROp_GenericSpecializationDictionary:
            inst->removeAndDeallocate();
            break;
        }
    }
}

IRInst* specializeGenericImpl(
    IRGeneric* genericVal,
    IRSpecialize* specializeInst,
    IRModule* module,
    SpecializationContext* context)
{
    // Effectively, specializing a generic amounts to "calling" the generic
    // on its concrete argument values and computing the
    // result it returns.
    //
    // For now, all of our generics consist of a single
    // basic block, so we can "call" them just by
    // cloning the instructions in their single block
    // into the global scope, using an environment for
    // cloning that maps the generic parameters to
    // the concrete arguments that were provided
    // by the `specialize(...)` instruction.
    //
    IRCloneEnv env;

    // We will walk through the parameters of the generic and
    // register the corresponding argument of the `specialize`
    // instruction to be used as the "cloned" value for each
    // parameter.
    //
    // Suppose we are looking at `specialize(g, a, b, c)` and `g` has
    // three generic parameters: `T`, `U`, and `V`. Then we will
    // be initializing our environment to map `T -> a`, `U -> b`,
    // and `V -> c`.
    //
    UInt argCounter = 0;
    for (auto param : genericVal->getParams())
    {
        UInt argIndex = argCounter++;
        SLANG_ASSERT(argIndex < specializeInst->getArgCount());

        IRInst* arg = specializeInst->getArg(argIndex);

        env.mapOldValToNew.add(param, arg);
    }

    // We will set up an IR builder for insertion
    // into the global scope, at the same location
    // as the original generic.
    //
    IRBuilder builderStorage(module);
    IRBuilder* builder = &builderStorage;
    builder->setInsertBefore(genericVal);

    List<IRInst*> pendingWorkList;
    SLANG_DEFER(for (Index ii = pendingWorkList.getCount() - 1; ii >= 0; ii--) if (context)
                    context->addToWorkList(pendingWorkList[ii]););

    // Now we will run through the body of the generic and
    // clone each of its instructions into the global scope,
    // until we reach a `return` instruction.
    //
    for (auto bb : genericVal->getBlocks())
    {
        // We expect a generic to only ever contain a single block.
        //
        SLANG_ASSERT(bb == genericVal->getFirstBlock());

        // We will iterate over the non-parameter ("ordinary")
        // instructions only, because parameters were dealt
        // with explictly at an earlier point.
        //
        for (auto ii : bb->getOrdinaryInsts())
        {
            // The last block of the generic is expected to end with
            // a `return` instruction for the specialized value that
            // comes out of the abstraction.
            //
            // We thus use that cloned value as the result of the
            // specialization step.
            //
            if (auto returnValInst = as<IRReturn>(ii))
            {
                auto specializedVal = findCloneForOperand(&env, returnValInst->getVal());

                // Clone decorations on the orignal `specialize` inst over to the newly specialized
                // value.
                cloneInstDecorationsAndChildren(&env, module, specializeInst, specializedVal);

                // Perform IR simplifications to fold constants in this specialized value if it is a
                // function, so further specializations from the specialized function will have as
                // simple specialization arguments as possible to avoid creating specializations
                // that eventually simplified into the same thing.
                if (context)
                {
                    if (auto func = as<IRFunc>(specializedVal))
                    {
                        context->tryExpandParameterPack(func);
                        simplifyFunc(
                            context->targetProgram,
                            func,
                            IRSimplificationOptions::getFast(context->targetProgram));
                    }
                }
                pendingWorkList.add(specializedVal);
                return specializedVal;
            }

            // For any instruction other than a `return`, we will
            // simply clone it completely into the global scope.
            //
            IRInst* clonedInst = cloneInst(&env, builder, ii);

            // Any new instructions we create during cloning were
            // not present when we initially built our work list,
            // so we need to make sure to consider them now.
            //
            // This is important for the cases where one generic
            // invokes another, because there will be `specialize`
            // operations nested inside the first generic that refer
            // to the second.
            //
            if (context)
            {
                pendingWorkList.add(clonedInst);
            }
        }
    }

    // If we reach this point, something went wrong, because we
    // never encountered a `return` inside the body of the generic.
    //
    SLANG_UNEXPECTED("no return from generic");
    UNREACHABLE_RETURN(nullptr);
}

IRInst* specializeGeneric(IRSpecialize* specializeInst)
{
    auto baseGeneric = as<IRGeneric>(specializeInst->getBase());
    SLANG_ASSERT(baseGeneric);
    if (!baseGeneric)
        return specializeInst;

    auto module = specializeInst->getModule();
    SLANG_ASSERT(module);
    if (!module)
        return specializeInst;

    return specializeGenericImpl(baseGeneric, specializeInst, module, nullptr);
}

} // namespace Slang
