// slang-ir-link.cpp
#include "slang-ir-link.h"

#include "../compiler-core/slang-artifact.h"
#include "../core/slang-performance-profiler.h"
#include "slang-capability.h"
#include "slang-ir-autodiff.h"
#include "slang-ir-insts.h"
#include "slang-ir-layout.h"
#include "slang-ir-specialize-target-switch.h"
#include "slang-ir-string-hash.h"
#include "slang-ir.h"
#include "slang-legalize-types.h"
#include "slang-mangle.h"
#include "slang-module-library.h"

namespace Slang
{

/// Find a suitable layout for `entryPoint` in `programLayout`.
///
/// TODO: This function should be eliminated. See its body
/// for an explanation of the problems.
EntryPointLayout* findEntryPointLayout(ProgramLayout* programLayout, EntryPoint* entryPoint);

struct IRSpecSymbol : RefObject
{
    IRInst* irGlobalValue;
    RefPtr<IRSpecSymbol> nextWithSameName;
};

struct IRSpecEnv
{
    IRSpecEnv* parent = nullptr;

    // A map from original values to their cloned equivalents.
    typedef Dictionary<IRInst*, IRInst*> ClonedValueDictionary;
    ClonedValueDictionary clonedValues;
};

struct IRSharedSpecContext
{
    // The code-generation target in use
    CodeGenTarget target;

    // The API-level target request
    TargetRequest* targetReq = nullptr;

    // The specialized module we are building
    RefPtr<IRModule> module;

    // A map from mangled symbol names to zero or
    // more global IR values that have that name,
    // in the *original* module.
    typedef Dictionary<ImmutableHashedString, RefPtr<IRSpecSymbol>> SymbolDictionary;
    SymbolDictionary symbols;

    Dictionary<ImmutableHashedString, bool> isImportedSymbol;

    bool useAutodiff = false;

    IRBuilder builderStorage;

    // The "global" specialization environment.
    IRSpecEnv globalEnv;
};

void insertGlobalValueSymbol(IRSharedSpecContext* sharedContext, IRInst* gv);

struct WitnessTableCloneInfo : RefObject
{
    IRWitnessTable* clonedTable;
    IRWitnessTable* originalTable;
    Dictionary<UnownedStringSlice, IRWitnessTableEntry*> deferredEntries;
};

struct IRSpecContextBase
{
    IRSharedSpecContext* shared;

    IRSharedSpecContext* getShared() { return shared; }

    IRModule* getModule() { return getShared()->module; }

    List<IRModule*> irModules;

    HashSet<UnownedStringSlice> deferredWitnessTableEntryKeys;
    List<RefPtr<WitnessTableCloneInfo>> witnessTables;

    IRSpecSymbol* findSymbols(UnownedStringSlice mangledName)
    {
        ImmutableHashedString hashedName(mangledName);
        RefPtr<IRSpecSymbol> symbol;
        if (shared->symbols.tryGetValue(hashedName, symbol))
            return symbol;
        for (auto m : irModules)
        {
            for (auto inst : m->findSymbolByMangledName(hashedName))
                insertGlobalValueSymbol(shared, inst);
        }
        if (shared->symbols.tryGetValue(hashedName, symbol))
            return symbol;
        shared->symbols[hashedName] = nullptr;
        return nullptr;
    }

    // The current specialization environment to use.
    IRSpecEnv* env = nullptr;
    IRSpecEnv* getEnv()
    {
        // TODO: need to actually establish environments on contexts we create.
        //
        // Or more realistically we need to change the whole approach
        // to specialization and cloning so that we don't try to share
        // logic between two very different cases.


        return env;
    }

    // The IR builder to use for creating nodes
    IRBuilder* builder;

    // A callback to be used when a value that is not registerd in `clonedValues`
    // is needed during cloning. This gives the subtype a chance to intercept
    // the operation and clone (or not) as needed.
    virtual IRInst* maybeCloneValue(IRInst* originalVal) { return originalVal; }
};

void registerClonedValue(IRSpecContextBase* context, IRInst* clonedValue, IRInst* originalValue)
{
    if (!originalValue)
        return;

    // TODO: now that things are scoped using environments, we
    // shouldn't be running into the cases where a value with
    // the same key already exists. This should be changed to
    // an `Add()` call.
    //
    context->getEnv()->clonedValues[originalValue] = clonedValue;

    switch (clonedValue->getOp())
    {
    case kIROp_LookupWitness:

        // If `originalVal` represents a witness table entry key, add the key
        // to witnessTableEntryWorkList.
        context->deferredWitnessTableEntryKeys.add(
            getMangledName(as<IRLookupWitnessMethod>(clonedValue)->getRequirementKey()));
        break;
    case kIROp_ForwardDerivativeDecoration:
    case kIROp_BackwardDerivativeDecoration:
    case kIROp_UserDefinedBackwardDerivativeDecoration:
        if (context->getShared()->useAutodiff)
        {
            if (auto key = as<IRStructKey>(clonedValue->getOperand(0)))
                context->deferredWitnessTableEntryKeys.add(getMangledName(key));
        }
        break;
    }
}

// Information on values to use when registering a cloned value
struct IROriginalValuesForClone
{
    IRInst* originalVal = nullptr;
    IRSpecSymbol* sym = nullptr;

    IROriginalValuesForClone() {}

    IROriginalValuesForClone(IRInst* originalValue)
        : originalVal(originalValue)
    {
    }

    IROriginalValuesForClone(IRSpecSymbol* symbol)
        : sym(symbol)
    {
    }
};

void registerClonedValue(
    IRSpecContextBase* context,
    IRInst* clonedValue,
    IROriginalValuesForClone const& originalValues)
{
    registerClonedValue(context, clonedValue, originalValues.originalVal);
    for (auto s = originalValues.sym; s; s = s->nextWithSameName)
    {
        registerClonedValue(context, clonedValue, s->irGlobalValue);
    }
}

IRInst* cloneInst(
    IRSpecContextBase* context,
    IRBuilder* builder,
    IRInst* originalInst,
    IROriginalValuesForClone const& originalValues);

IRInst* cloneInst(IRSpecContextBase* context, IRBuilder* builder, IRInst* originalInst)
{
    return cloneInst(context, builder, originalInst, originalInst);
}

bool isAutoDiffDecoration(IRInst* decor)
{
    switch (decor->getOp())
    {
    case kIROp_ForwardDerivativeDecoration:
    case kIROp_BackwardDerivativeIntermediateTypeDecoration:
    case kIROp_BackwardDerivativePrimalDecoration:
    case kIROp_BackwardDerivativePropagateDecoration:
    case kIROp_BackwardDerivativePrimalContextDecoration:
    case kIROp_BackwardDerivativePrimalReturnDecoration:
    case kIROp_PrimalSubstituteDecoration:
    case kIROp_BackwardDerivativeDecoration:
    case kIROp_UserDefinedBackwardDerivativeDecoration:
    case kIROp_DifferentiableTypeDictionaryDecoration:
    case kIROp_ForwardDifferentiableDecoration:
    case kIROp_BackwardDifferentiableDecoration:
        return true;
    default:
        return false;
    }
}

/// Clone any decorations from `originalValue` onto `clonedValue`
void cloneDecorations(IRSpecContextBase* context, IRInst* clonedValue, IRInst* originalValue)
{
    // TODO: In many cases we might be able to use this as a general-purpose
    // place to do cloning of *all* the children of an instruction, and
    // not just its decorations. We should look to refactor this code
    // later.

    IRBuilder builderStorage = *context->builder;
    IRBuilder* builder = &builderStorage;
    builder->setInsertInto(clonedValue);


    SLANG_UNUSED(context);
    for (auto originalDecoration : originalValue->getDecorations())
    {
        if (!context->shared->useAutodiff && isAutoDiffDecoration(originalDecoration))
            continue;
        cloneInst(context, builder, originalDecoration);
    }

    // We will also clone the location here, just because this is a convenient bottleneck
    clonedValue->sourceLoc = originalValue->sourceLoc;
}

/// Clone any decorations and children from `originalValue` onto `clonedValue`
void cloneDecorationsAndChildren(
    IRSpecContextBase* context,
    IRInst* clonedValue,
    IRInst* originalValue)
{
    IRBuilder builderStorage = *context->builder;
    IRBuilder* builder = &builderStorage;
    builder->setInsertInto(clonedValue);

    SLANG_UNUSED(context);
    for (auto originalItem : originalValue->getDecorationsAndChildren())
    {
        if (!context->shared->useAutodiff && isAutoDiffDecoration(originalItem))
            continue;
        cloneInst(context, builder, originalItem);
    }

    // We will also clone the location here, just because this is a convenient bottleneck
    clonedValue->sourceLoc = originalValue->sourceLoc;
}

// We use an `IRSpecContext` for the case where we are cloning
// code from one or more input modules to create a "linked" output
// module. Along the way, we will resolve profile-specific functions
// to the best definition for a given target.
//
struct IRSpecContext : IRSpecContextBase
{
    // Override the "maybe clone" logic so that we always clone
    virtual IRInst* maybeCloneValue(IRInst* originalVal) override;
};


IRInst* cloneGlobalValue(IRSpecContext* context, IRInst* originalVal);

IRInst* cloneValue(IRSpecContextBase* context, IRInst* originalValue);

IRType* cloneType(IRSpecContextBase* context, IRType* originalType);

IRInst* IRSpecContext::maybeCloneValue(IRInst* originalValue)
{
    switch (originalValue->getOp())
    {
    case kIROp_StructType:
    case kIROp_ClassType:
    case kIROp_Func:
    case kIROp_Generic:
    case kIROp_GlobalVar:
    case kIROp_GlobalParam:
    case kIROp_GlobalConstant:
    case kIROp_StructKey:
    case kIROp_InterfaceRequirementEntry:
    case kIROp_GlobalGenericParam:
    case kIROp_WitnessTable:
    case kIROp_InterfaceType:
        return cloneGlobalValue(this, originalValue);

    case kIROp_BoolLit:
        {
            IRConstant* c = (IRConstant*)originalValue;
            return builder->getBoolValue(c->value.intVal != 0);
        }
        break;


    case kIROp_IntLit:
        {
            IRConstant* c = (IRConstant*)originalValue;
            return builder->getIntValue(cloneType(this, c->getDataType()), c->value.intVal);
        }
        break;

    case kIROp_FloatLit:
        {
            IRConstant* c = (IRConstant*)originalValue;
            return builder->getFloatValue(cloneType(this, c->getDataType()), c->value.floatVal);
        }
        break;

    case kIROp_StringLit:
        {
            IRConstant* c = (IRConstant*)originalValue;
            return builder->getStringValue(c->getStringSlice());
        }
        break;

    case kIROp_PtrLit:
        {
            IRConstant* c = (IRConstant*)originalValue;
            SLANG_RELEASE_ASSERT(c->value.ptrVal == nullptr);
            return builder->getNullPtrValue(cloneType(this, c->getFullType()));
        }
        break;

    case kIROp_VoidLit:
        {
            return builder->getVoidValue();
        }
        break;

    default:
        {
            // In the default case, assume that we have some sort of "hoistable"
            // instruction that requires us to create a clone of it.

            UInt argCount = originalValue->getOperandCount();
            ShortList<IRInst*> newArgs;
            newArgs.setCount(argCount);
            for (UInt aa = 0; aa < argCount; ++aa)
            {
                IRInst* originalArg = originalValue->getOperand(aa);
                IRInst* clonedArg = cloneValue(this, originalArg);
                newArgs[aa] = clonedArg;
            }
            IRInst* clonedValue = builder->createIntrinsicInst(
                cloneType(this, originalValue->getFullType()),
                originalValue->getOp(),
                argCount,
                newArgs.getArrayView().getBuffer());
            registerClonedValue(this, clonedValue, originalValue);

            cloneDecorationsAndChildren(this, clonedValue, originalValue);
            addHoistableInst(builder, clonedValue);
            return clonedValue;
        }
        break;
    }
}

IRInst* cloneValue(IRSpecContextBase* context, IRInst* originalValue);

// Find a pre-existing cloned value, or return null if none is available.
IRInst* findClonedValue(IRSpecContextBase* context, IRInst* originalValue)
{
    IRInst* clonedValue = nullptr;
    for (auto env = context->getEnv(); env; env = env->parent)
    {
        if (env->clonedValues.tryGetValue(originalValue, clonedValue))
        {
            return clonedValue;
        }
    }

    return nullptr;
}

IRInst* cloneValue(IRSpecContextBase* context, IRInst* originalValue)
{
    if (!originalValue)
        return nullptr;

    if (IRInst* clonedValue = findClonedValue(context, originalValue))
        return clonedValue;

    return context->maybeCloneValue(originalValue);
}

IRType* cloneType(IRSpecContextBase* context, IRType* originalType)
{
    return (IRType*)cloneValue(context, originalType);
}

void cloneGlobalValueWithCodeCommon(
    IRSpecContextBase* context,
    IRGlobalValueWithCode* clonedValue,
    IRGlobalValueWithCode* originalValue,
    IROriginalValuesForClone const& originalValues);

IRRate* cloneRate(IRSpecContextBase* context, IRRate* rate)
{
    return (IRRate*)cloneType(context, rate);
}

void maybeSetClonedRate(
    IRSpecContextBase* context,
    IRBuilder* builder,
    IRInst* clonedValue,
    IRInst* originalValue)
{
    if (auto rate = originalValue->getRate())
    {
        clonedValue->setFullType(
            builder->getRateQualifiedType(cloneRate(context, rate), clonedValue->getFullType()));
    }
}

IRGlobalVar* cloneGlobalVarImpl(
    IRSpecContextBase* context,
    IRBuilder* builder,
    IRGlobalVar* originalVar,
    IROriginalValuesForClone const& originalValues)
{
    auto clonedVar =
        builder->createGlobalVar(cloneType(context, originalVar->getDataType()->getValueType()));

    maybeSetClonedRate(context, builder, clonedVar, originalVar);

    registerClonedValue(context, clonedVar, originalValues);

    // Clone any code in the body of the variable, since this
    // represents the initializer.
    cloneGlobalValueWithCodeCommon(context, clonedVar, originalVar, originalValues);

    return clonedVar;
}

/// Clone certain special decorations for `clonedInst` from its (potentially multiple) definitions.
///
/// In most cases, once we've decided on the "best" definition to use for an IR instruction,
/// we only want the linking process to use the decorations from the single best definition.
/// In some casses, though, the canonical best definition might not have all the information.
///
/// A concrete example is the `[bindExistentialsSlots(...)]` decorations for global shader
/// parameters and entry points. These decorations are only generated as part of the IR
/// associated with a specialization of a program, and not the original IR for the modules
/// of the program.
///
/// This function scans through all the `originalValues` that were considered for `clonedInst`,
/// and copies over any decorations that are allowed to come from a non-"best" definition.
/// For a given decoration opcode, only one such decoration will ever be copied, and nothing
/// will be copied if the instruction already has a matching decoration (that was cloned
/// from the "best" definition).
///
static void cloneExtraDecorationsFromInst(
    IRSpecContextBase* context,
    IRBuilder* builder,
    IRInst* clonedInst,
    IRInst* originalInst)
{
    for (auto decoration : originalInst->getDecorations())
    {
        switch (decoration->getOp())
        {
        default:
            break;
        case kIROp_ForwardDerivativeDecoration:
        case kIROp_UserDefinedBackwardDerivativeDecoration:
        case kIROp_PrimalSubstituteDecoration:
            if (!context->getShared()->useAutodiff)
                break;
            [[fallthrough]];
        case kIROp_HLSLExportDecoration:
        case kIROp_BindExistentialSlotsDecoration:
        case kIROp_LayoutDecoration:
        case kIROp_PublicDecoration:
        case kIROp_SequentialIDDecoration:
        case kIROp_IntrinsicOpDecoration:
        case kIROp_NonCopyableTypeDecoration:
        case kIROp_DynamicDispatchWitnessDecoration:
            if (!clonedInst->findDecorationImpl(decoration->getOp()))
            {
                cloneInst(context, builder, decoration);
            }
            break;
        }
    }

    // We will also copy over source location information from the alternative
    // values, in case any of them has it available.
    //
    if (originalInst->sourceLoc.isValid() && !clonedInst->sourceLoc.isValid())
    {
        clonedInst->sourceLoc = originalInst->sourceLoc;
    }
}

static void cloneExtraDecorations(
    IRSpecContextBase* context,
    IRInst* clonedInst,
    IROriginalValuesForClone const& originalValues)
{
    IRBuilder builderStorage = *context->builder;
    IRBuilder* builder = &builderStorage;
    builder->setInsertInto(clonedInst);

    // If the `clonedInst` already has any non-decoration
    // children, then we want to insert before those,
    // to maintain the invariant that decorations always
    // precede non-decoration instructions in the list of
    // decorations and children.
    //
    if (auto firstChild = clonedInst->getFirstChild())
    {
        builder->setInsertBefore(firstChild);
    }

    for (auto sym = originalValues.sym; sym; sym = sym->nextWithSameName)
    {
        cloneExtraDecorationsFromInst(context, builder, clonedInst, sym->irGlobalValue);
    }
}

void cloneSimpleGlobalValueImpl(
    IRSpecContextBase* context,
    IRInst* originalInst,
    IROriginalValuesForClone const& originalValues,
    IRInst* clonedInst,
    bool registerValue = true)
{
    if (registerValue)
        registerClonedValue(context, clonedInst, originalValues);

    // Set up an IR builder for inserting into the inst
    IRBuilder builderStorage = *context->builder;
    IRBuilder* builder = &builderStorage;
    builder->setInsertInto(clonedInst);

    // Clone any children of the instruction
    for (auto child : originalInst->getDecorationsAndChildren())
    {
        cloneInst(context, builder, child);
    }

    // Also clone certain decorations if they appear on *any*
    // definition of the symbol (not necessarily the one
    // we picked as the primary/best).
    //
    cloneExtraDecorations(context, clonedInst, originalValues);
}

IRGlobalParam* cloneGlobalParamImpl(
    IRSpecContextBase* context,
    IRBuilder* builder,
    IRGlobalParam* originalVal,
    IROriginalValuesForClone const& originalValues)
{
    auto clonedVal = builder->createGlobalParam(cloneType(context, originalVal->getFullType()));
    cloneSimpleGlobalValueImpl(context, originalVal, originalValues, clonedVal);

    return clonedVal;
}

IRGlobalConstant* cloneGlobalConstantImpl(
    IRSpecContextBase* context,
    IRBuilder* builder,
    IRGlobalConstant* originalVal,
    IROriginalValuesForClone const& originalValues)
{
    auto oldBuilder = context->builder;
    context->builder = builder;
    auto clonedType = cloneType(context, originalVal->getFullType());
    IRGlobalConstant* clonedVal = nullptr;
    if (auto originalInitVal = originalVal->getValue())
    {
        auto clonedInitVal = cloneValue(context, originalInitVal);
        clonedVal = builder->emitGlobalConstant(clonedType, clonedInitVal);
    }
    else
    {
        clonedVal = builder->emitGlobalConstant(clonedType);
    }

    cloneSimpleGlobalValueImpl(context, originalVal, originalValues, clonedVal);
    context->builder = oldBuilder;
    return clonedVal;
}

IRGeneric* cloneGenericImpl(
    IRSpecContextBase* context,
    IRBuilder* builder,
    IRGeneric* originalVal,
    IROriginalValuesForClone const& originalValues)
{
    auto clonedVal = builder->emitGeneric();
    registerClonedValue(context, clonedVal, originalValues);

    // Clone any code in the body of the generic, since this
    // computes its result value.
    cloneGlobalValueWithCodeCommon(context, clonedVal, originalVal, originalValues);

    // We want to clone extra decorations on the
    // return value from other symbols as well.
    auto clonedInnerVal = findGenericReturnVal(clonedVal);
    for (auto originalSym = originalValues.sym; originalSym;
         originalSym = originalSym->nextWithSameName.get())
    {
        auto originalGeneric = as<IRGeneric>(originalSym->irGlobalValue);
        if (!originalGeneric)
            continue;
        auto originalInnerVal = findGenericReturnVal(originalGeneric);

        // Register all generic parameters before cloning the decorations.
        auto clonedParam = clonedVal->getFirstParam();
        auto originalParam = originalGeneric->getFirstParam();

        ShortList<KeyValuePair<IRInst*, IRInst*>> paramMapping;
        for (; clonedParam && originalParam;
             (clonedParam = as<IRParam, IRDynamicCastBehavior::NoUnwrap>(clonedParam->next)),
             (originalParam = as<IRParam, IRDynamicCastBehavior::NoUnwrap>(originalParam->next)))
        {
            paramMapping.add(KeyValuePair<IRInst*, IRInst*>(clonedParam, originalParam));
        }
        // Generic parameter list does not match, bail.
        if (clonedParam || originalParam)
            continue;
        for (const auto& [key, value] : paramMapping)
            registerClonedValue(context, key, value);

        IRBuilder builderStorage = *builder;
        IRBuilder* decorBuilder = &builderStorage;
        decorBuilder->setInsertInto(clonedInnerVal);
        if (auto firstChild = clonedInnerVal->getFirstChild())
        {
            decorBuilder->setInsertBefore(firstChild);
        }
        cloneExtraDecorationsFromInst(context, decorBuilder, clonedInnerVal, originalInnerVal);
    }
    return clonedVal;
}

IRStructKey* cloneStructKeyImpl(
    IRSpecContextBase* context,
    IRBuilder* builder,
    IRStructKey* originalVal,
    IROriginalValuesForClone const& originalValues)
{
    auto clonedVal = builder->createStructKey();
    cloneSimpleGlobalValueImpl(context, originalVal, originalValues, clonedVal);
    return clonedVal;
}

IRGlobalGenericParam* cloneGlobalGenericParamImpl(
    IRSpecContextBase* context,
    IRBuilder* builder,
    IRGlobalGenericParam* originalVal,
    IROriginalValuesForClone const& originalValues)
{
    auto clonedVal = builder->emitGlobalGenericParam(originalVal->getFullType());
    cloneSimpleGlobalValueImpl(context, originalVal, originalValues, clonedVal);
    return clonedVal;
}

bool shouldDeepCloneWitnessTable(IRSpecContextBase* context, IRWitnessTable* table)
{
    for (auto decor : table->getDecorations())
    {
        switch (decor->getOp())
        {
        case kIROp_HLSLExportDecoration:
        case kIROp_KeepAliveDecoration:
            return true;
        }
    }

    auto conformanceType = getResolvedInstForDecorations(table->getConformanceType());

    for (auto decor : conformanceType->getDecorations())
    {
        switch (decor->getOp())
        {
        case kIROp_ComInterfaceDecoration:
            return true;
        case kIROp_KnownBuiltinDecoration:
            {
                auto name = as<IRKnownBuiltinDecoration>(decor)->getName();
                if (name == toSlice("IDifferentiable") || name == toSlice("IDifferentiablePtr"))
                    return context->getShared()->useAutodiff;
                break;
            }
        default:
            break;
        }
    }

    return false;
}

IRWitnessTable* cloneWitnessTableImpl(
    IRSpecContextBase* context,
    IRBuilder* builder,
    IRWitnessTable* originalTable,
    IROriginalValuesForClone const& originalValues,
    IRWitnessTable* dstTable = nullptr,
    bool registerValue = true)
{
    IRWitnessTable* clonedTable = dstTable;
    IRType* clonedBaseType = nullptr;
    if (!clonedTable)
    {
        clonedBaseType = cloneType(context, (IRType*)(originalTable->getConformanceType()));
        auto clonedSubType = cloneType(context, (IRType*)(originalTable->getConcreteType()));
        clonedTable = builder->createWitnessTable(clonedBaseType, clonedSubType);
        if (clonedTable->hasDecorationOrChild())
            return clonedTable;
    }
    else
    {
        clonedBaseType = (IRType*)clonedTable->getConformanceType();
    }
    if (registerValue)
        registerClonedValue(context, clonedTable, originalValues);

    // Set up an IR builder for inserting into the witness table
    IRBuilder builderStorage = *context->builder;
    IRBuilder* entryBuilder = &builderStorage;
    entryBuilder->setInsertInto(clonedTable);

    // Clone decorations first
    for (auto decoration : originalTable->getDecorations())
    {
        cloneInst(context, entryBuilder, decoration);
    }
    cloneExtraDecorations(context, clonedTable, originalValues);

    RefPtr<WitnessTableCloneInfo> witnessInfo = new WitnessTableCloneInfo();
    witnessInfo->clonedTable = clonedTable;
    witnessInfo->originalTable = originalTable;

    bool shouldDeepClone = shouldDeepCloneWitnessTable(context, originalTable);

    // Clone only the witness table entries that are actually used
    for (auto child : originalTable->getDecorationsAndChildren())
    {
        if (auto entry = as<IRWitnessTableEntry>(child))
        {
            if (!shouldDeepClone)
            {
                // Skip witness table entries during the first pass,
                // and just add them to the deferred work list.
                witnessInfo->deferredEntries.add(getMangledName(entry->getRequirementKey()), entry);
                continue;
            }
        }
        // Clone any non-entry children as is
        cloneInst(context, entryBuilder, child);
    }
    context->witnessTables.add(witnessInfo);

    return clonedTable;
}

IRWitnessTable* cloneWitnessTableWithoutRegistering(
    IRSpecContextBase* context,
    IRBuilder* builder,
    IRWitnessTable* originalTable,
    IRWitnessTable* dstTable = nullptr)
{
    return cloneWitnessTableImpl(
        context,
        builder,
        originalTable,
        IROriginalValuesForClone(),
        dstTable,
        false);
}

IRStructType* cloneStructTypeImpl(
    IRSpecContextBase* context,
    IRBuilder* builder,
    IRStructType* originalStruct,
    IROriginalValuesForClone const& originalValues)
{
    auto clonedStruct = builder->createStructType();
    cloneSimpleGlobalValueImpl(context, originalStruct, originalValues, clonedStruct);
    return clonedStruct;
}


IRInterfaceType* cloneInterfaceTypeImpl(
    IRSpecContextBase* context,
    IRBuilder* builder,
    IRInterfaceType* originalInterface,
    IROriginalValuesForClone const& originalValues)
{
    auto clonedInterface =
        builder->createInterfaceType(originalInterface->getOperandCount(), nullptr);
    registerClonedValue(context, clonedInterface, originalValues);

    for (UInt i = 0; i < originalInterface->getOperandCount(); i++)
    {
        auto clonedKey = cloneValue(context, originalInterface->getOperand(i));
        clonedInterface->setOperand(i, clonedKey);
    }
    cloneSimpleGlobalValueImpl(context, originalInterface, originalValues, clonedInterface, false);
    return clonedInterface;
}

void cloneGlobalValueWithCodeCommon(
    IRSpecContextBase* context,
    IRGlobalValueWithCode* clonedValue,
    IRGlobalValueWithCode* originalValue,
    IROriginalValuesForClone const& originalValues)
{
    // Next we are going to clone the actual code.
    IRBuilder builderStorage = *context->builder;
    IRBuilder* builder = &builderStorage;
    builder->setInsertInto(clonedValue);

    cloneDecorations(context, clonedValue, originalValue);
    cloneExtraDecorations(context, clonedValue, originalValues);
    clonedValue->setFullType((IRType*)cloneValue(context, originalValue->getFullType()));

    // We will walk through the blocks of the function, and clone each of them.
    //
    // We need to create the cloned blocks first, and then walk through them,
    // because blocks might be forward referenced (this is not possible
    // for other cases of instructions).
    for (auto originalBlock = originalValue->getFirstBlock(); originalBlock;
         originalBlock = originalBlock->getNextBlock())
    {
        IRBlock* clonedBlock = builder->createBlock();
        clonedValue->addBlock(clonedBlock);
        registerClonedValue(context, clonedBlock, originalBlock);

#if 0
        // We can go ahead and clone parameters here, while we are at it.
        builder->curBlock = clonedBlock;
        for (auto originalParam = originalBlock->getFirstParam();
            originalParam;
            originalParam = originalParam->getNextParam())
        {
            IRParam* clonedParam = builder->emitParam(
                context->maybeCloneType(
                    originalParam->getFullType()));
            cloneDecorations(context, clonedParam, originalParam);
            registerClonedValue(context, clonedParam, originalParam);
        }
#endif
    }

    // Okay, now we are in a good position to start cloning
    // the instructions inside the blocks.
    {
        IRBlock* ob = originalValue->getFirstBlock();
        IRBlock* cb = clonedValue->getFirstBlock();
        struct ParamCloneInfo
        {
            IRParam* originalParam;
            IRParam* clonedParam;
        };
        while (ob)
        {
            ShortList<ParamCloneInfo> paramCloneInfos;
            SLANG_ASSERT(cb);

            builder->setInsertInto(cb);
            for (auto oi = ob->getFirstInst(); oi; oi = oi->getNextInst())
            {
                if (oi->getOp() == kIROp_Param)
                {
                    // Params may have forward references in its type and
                    // decorations, so we just create a placeholder for it
                    // in this first pass.
                    IRParam* clonedParam = builder->emitParam(nullptr);
                    registerClonedValue(context, clonedParam, oi);
                    paramCloneInfos.add({(IRParam*)oi, clonedParam});
                }
                else
                {
                    if (oi->getOp() == kIROp_DifferentiableTypeAnnotation &&
                        !context->getShared()->useAutodiff)
                        continue;
                    cloneInst(context, builder, oi);
                }
            }
            // Clone the type and decorations of parameters after all instructs in the block
            // have been cloned.
            for (auto param : paramCloneInfos)
            {
                builder->setInsertInto(param.clonedParam);
                param.clonedParam->setFullType(
                    (IRType*)cloneValue(context, param.originalParam->getFullType()));
                cloneDecorations(context, param.clonedParam, param.originalParam);
            }
            ob = ob->getNextBlock();
            cb = cb->getNextBlock();
        }
    }
}

void checkIRDuplicate(IRInst* inst, IRInst* moduleInst, UnownedStringSlice const& mangledName)
{
#ifdef _DEBUG
    for (auto child : moduleInst->getDecorationsAndChildren())
    {
        if (child == inst)
            continue;

        if (auto childLinkage = child->findDecoration<IRLinkageDecoration>())
        {
            if (mangledName == childLinkage->getMangledName())
            {
                SLANG_UNEXPECTED("duplicate global instruction");
            }
        }
    }
#else
    SLANG_UNREFERENCED_PARAMETER(inst);
    SLANG_UNREFERENCED_PARAMETER(moduleInst);
    SLANG_UNREFERENCED_PARAMETER(mangledName);
#endif
}

void cloneFunctionCommon(
    IRSpecContextBase* context,
    IRFunc* clonedFunc,
    IRFunc* originalFunc,
    IROriginalValuesForClone const& originalValues,
    bool checkDuplicate = true)
{
    // First clone all the simple properties.
    clonedFunc->setFullType(cloneType(context, originalFunc->getFullType()));

    cloneGlobalValueWithCodeCommon(context, clonedFunc, originalFunc, originalValues);

    // Shuffle the function to the end of the list, because
    // it needs to follow its dependencies.
    //
    // TODO: This isn't really a good requirement to place on the IR...
    clonedFunc->moveToEnd();

    if (checkDuplicate)
    {
        if (auto linkage = clonedFunc->findDecoration<IRLinkageDecoration>())
        {
            checkIRDuplicate(
                clonedFunc,
                context->getModule()->getModuleInst(),
                linkage->getMangledName());
        }
    }
}

// We will forward-declare the subroutine for eagerly specializing
// an IR-level generic to argument values, because `specializeIRForEntryPoint`
// needs to perform this operation even though it is logically part of
// the later generic specialization pass.
//
IRInst* specializeGeneric(IRSpecialize* specializeInst);

/// Copy layout information for an entry-point function to its parameters.
///
/// When layout information is initially attached to an IR entry point,
/// it may be attached to a declaration that would have no `IRParam`s
/// to represent the entry-point parameters.
///
/// After linking, we expect an entry point to have a full definition,
/// so it becomes possible to copy per-parameter layout information
/// from the entry-point layout down to the individual parameters,
/// which simplifies subsequent IR steps taht want to look for
/// per-parameter layout information.
///
/// TODO: This step should probably be hoisted out to be an independent
/// IR pass that runs after linking, rather than being folded into
/// the linking step.
///
static void maybeCopyLayoutInformationToParameters(IRFunc* func, IRBuilder* builder)
{
    auto layoutDecor = func->findDecoration<IRLayoutDecoration>();
    if (!layoutDecor)
        return;

    auto entryPointLayout = as<IREntryPointLayout>(layoutDecor->getLayout());
    if (!entryPointLayout)
        return;

    if (auto firstBlock = func->getFirstBlock())
    {
        auto paramsStructLayout = getScopeStructLayout(entryPointLayout);
        Index paramLayoutCount = paramsStructLayout->getFieldCount();
        Index paramCounter = 0;
        for (auto pp = firstBlock->getFirstParam(); pp; pp = pp->getNextParam())
        {
            Index paramIndex = paramCounter++;
            if (paramIndex < paramLayoutCount)
            {
                auto paramLayout = paramsStructLayout->getFieldLayout(paramIndex);

                auto offsetParamLayout = applyOffsetToVarLayout(
                    builder,
                    paramLayout,
                    entryPointLayout->getParamsLayout());

                builder->addLayoutDecoration(pp, offsetParamLayout);
            }
            else
            {
                SLANG_UNEXPECTED("too many parameters");
            }
        }
    }
}

IRFunc* specializeIRForEntryPoint(
    IRSpecContext* context,
    String const& mangledName,
    EntryPoint* entryPoint,
    UnownedStringSlice nameOverride)
{
    // We start by looking up the IR symbol that
    // matches the mangled name given to the
    // function we want to emit.
    //
    // Note: the function decl-ref may refer to
    // a specialization of a generic function,
    // so that the mangled name of the decl-ref is
    // not the same as the mangled name of the decl.
    //
    IRSpecSymbol* sym = context->findSymbols(mangledName.getUnownedSlice());
    if (!sym)
    {
        String hashedName = getHashedName(mangledName.getUnownedSlice());
        sym = context->findSymbols(hashedName.getUnownedSlice());
        if (!sym)
        {
            SLANG_UNEXPECTED("no matching IR symbol");
            return nullptr;
        }
    }

    // Note: it is possible that `sym` shows multiple
    // definitions, coming from different IR modules that
    // were input to the linking process. We don't have
    // to do anything special about that here, because
    // we can use *any* of the IR values as the starting
    // point for cloning, and `cloneGlobalValue` will
    // follow the linkage decoration and discover the
    // other values on its own.
    //
    auto originalVal = sym->irGlobalValue;

    // We will start by cloning the entry point reference
    // like any other global value.
    //
    auto clonedVal = cloneGlobalValue(context, originalVal);

    // In the case where the user is requesting a specialization
    // of a generic entry point, we have a bit of a problem.
    //
    // This function is expected to return an `IRFunc` and
    // subsequent passes expect to find, e.g., layout information
    // attached to the parameters of such a func.
    //
    // In the generic case, the `clonedValue` won't be an
    // `IRFunc`, but instead an `IRSpecialize`.
    //
    if (auto clonedSpec = as<IRSpecialize>(clonedVal))
    {
        // The Right Thing to do here is to perform some
        // amount of generic specialization, at least
        // until we get back an `IRFunc`.
        //
        // The dangerous thing is that the generic specialization
        // pass can, in principle, change the signature of
        // functions, so that attaching parameter layout
        // information *after* specialization might not work.
        //
        // The compromise we make here is to directly
        // invoke the logic for specializing a generic.
        //
        // In theory this isn't valid, because there is no
        // way we can register the specialized function we
        // create so that it would be re-used by other instantiations
        // with the same arguments (because we cannot be
        // sure the generic arguments are themselves fully specialized)
        //
        // In practice this isn't really a problem, because
        // we don't want to share the definition between
        // an entry point and an ordinary function anyway.
        //
        auto specializedClone = specializeGeneric(clonedSpec);

        // One special case we need to be aware of is that there
        // might be decorations attached to the `specialize`
        // instruction, which we then want to copy over to the
        // result of specialization.
        //
        cloneExtraDecorations(context, specializedClone, IROriginalValuesForClone(sym));

        // Now we will move to considering the specialized instruction
        // instead of the unspecialized one for all further steps.
        //
        clonedVal = specializedClone;
    }

    auto clonedFunc = as<IRFunc>(clonedVal);
    if (!clonedFunc)
    {
        SLANG_UNEXPECTED("expected entry point to be a function");
        UNREACHABLE_RETURN(nullptr);
    }

    if (!clonedFunc->findDecorationImpl(kIROp_KeepAliveDecoration))
    {
        context->builder->addKeepAliveDecoration(clonedFunc);
    }

    // If an IREntryPointDecoration already exist in the function,
    // check if we need to update its name with nameOverride.
    // If the decoration doesn't exist, create it with the desired name.
    if (auto entryPointDecor = clonedFunc->findDecoration<IREntryPointDecoration>())
    {
        if (nameOverride.getLength())
        {
            IRInst* operands[] = {
                entryPointDecor->getProfileInst(),
                context->builder->getStringValue(nameOverride),
                entryPointDecor->getModuleName()};
            context->builder
                ->addDecoration(clonedFunc, IROp::kIROp_EntryPointDecoration, operands, 3);
            entryPointDecor->removeAndDeallocate();
        }
    }
    else
    {
        if (!nameOverride.getLength())
            nameOverride = getUnownedStringSliceText(entryPoint->getName());
        IRInst* operands[] = {
            context->builder->getIntValue(
                context->builder->getIntType(),
                entryPoint->getProfile().raw),
            context->builder->getStringValue(nameOverride),
            context->builder->getStringValue(
                UnownedStringSlice(entryPoint->getModule()->getName()))};
        context->builder->addDecoration(clonedFunc, IROp::kIROp_EntryPointDecoration, operands, 3);
    }

    // We will also go on and attach layout information
    // to the function parameters, so that we have it
    // available directly on the parameters, rather
    // than having to look it up on the original entry-point layout.
    maybeCopyLayoutInformationToParameters(clonedFunc, context->builder);

    return clonedFunc;
}

// Get a string form of the target so that we can
// use it to match against target-specialization modifiers
//
CapabilitySet getTargetCapabilities(IRSpecContext* context)
{
    return context->getShared()->targetReq->getTargetCaps();
}

/// Get the most appropriate ("best") capability requirements for `inVal` based on the
/// `targetCaps`.
static CapabilitySet _getBestSpecializationCaps(IRInst* inVal, CapabilitySet const& targetCaps)
{
    IRInst* val = getResolvedInstForDecorations(inVal);

    // If the instruction has no target-related decorations,
    // then it is implied to be an unspecialized, target-independent
    // declaration.
    //
    // Such a declaration amounts to an empty set of capabilities.
    //
    if (!val->findDecoration<IRTargetDecoration>())
        return CapabilitySet::makeEmpty();

    if (auto targetSpecificDecoration =
            findBestTargetDecoration<IRTargetSpecificDefinitionDecoration>(inVal, targetCaps))
    {
        return targetSpecificDecoration->getTargetCaps();
    }
    else
    {
        return CapabilitySet::makeInvalid();
    }
}

// Is `newVal` marked as being a better match for our
// chosen code-generation target?
//
// TODO: there is a missing step here where we need
// to check if things are even available in the first place...
bool isBetterForTarget(IRSpecContext* context, IRInst* newVal, IRInst* oldVal)
{
    // Anything is better than nothing..
    if (oldVal == nullptr)
    {
        return true;
    }

    // For right now every declaration might have zero or more
    // decorations, representing the capabilities for which it is specialized.
    // Each decorations has a `CapabilitySet` to represent what it requires of a target.
    //
    // We need to look at all the candidate declarations for a symbol
    // and pick the one that has the "most specialized" set of capabilities
    // for our chosen target.
    //
    // In principle, this should be as simple as:
    //
    // * Ignore all decorations with capability sets that aren't subsets
    //   of the capabilities of our target.
    //
    // * From the remaining decorations, pick the one that is a superset
    //   of all the others (and give an ambiguity error if there is
    //   no unique "best" option).
    //
    // In practice, the choice is complicated by the way that we currently
    // have the compiler automatically deduce dependencies on extensions
    // or other features that were not included as part of the target
    // description by the user.
    //
    // In order to preserve the ability to infer more specialized requirements
    // than what the target includes, we change the two steps slightly:
    //
    // * Ignore all decorations with capability sets that are *incompatible*
    //   with the capabilities of our target, such that they could never be
    //   used together.
    //
    // * From all the remaining decorations, pick the one that is "better"
    //   than all the others in that it is either a supserset of each other
    //   candidate, or for each feature that another candidate requires,
    //   it requires a "better" feature that covers the same ground.
    //
    // Note: This approach isn't really sound, so we are likely to have
    // to tweak it over time. Most notably, we probably need/want to
    // push back on the automatic inference of extensions/versions in
    // the compiler as much as possible.
    //
    CapabilitySet targetCaps = getTargetCapabilities(context);
    CapabilitySet newCaps = _getBestSpecializationCaps(newVal, targetCaps);
    CapabilitySet oldCaps = _getBestSpecializationCaps(oldVal, targetCaps);

    // If either value returned an invalid capability set, it implies
    // that it cannot be used on this target at all, and the other
    // value should be considered better by default.
    //
    // Note: if both of the candidate values we have are incompatible
    // with our target, then it doesn't matter which we favor.
    //
    if (newCaps.isInvalid())
        return false;
    if (oldCaps.isInvalid())
        return true;

    bool isEqual = false;
    bool isNewBetter = newCaps.isBetterForTarget(oldCaps, targetCaps, isEqual);
    if (!isEqual)
        return isNewBetter;

    // All preceding factors being equal, an `[export]` is better
    // than an `[import]`.
    //
    bool newIsExport = newVal->findDecoration<IRExportDecoration>() != nullptr;
    bool oldIsExport = oldVal->findDecoration<IRExportDecoration>() != nullptr;
    if (newIsExport != oldIsExport)
        return newIsExport;

    // All preceding factors being equal, a definition is
    // better than a declaration.
    auto newIsDef = isDefinition(newVal);
    auto oldIsDef = isDefinition(oldVal);
    if (newIsDef != oldIsDef)
        return newIsDef;

    return false;
}

IRFunc* cloneFuncImpl(
    IRSpecContextBase* context,
    IRBuilder* builder,
    IRFunc* originalFunc,
    IROriginalValuesForClone const& originalValues)
{
    auto clonedFunc = builder->createFunc();
    registerClonedValue(context, clonedFunc, originalValues);
    cloneFunctionCommon(context, clonedFunc, originalFunc, originalValues);
    return clonedFunc;
}

// Can an inst with `opcode` contain basic blocks as children?
bool canInstContainBasicBlocks(IROp opcode)
{
    switch (opcode)
    {
    case kIROp_Expand:
        return true;
    default:
        return false;
    }
}

IRInst* cloneInst(
    IRSpecContextBase* context,
    IRBuilder* builder,
    IRInst* originalInst,
    IROriginalValuesForClone const& originalValues)
{
    switch (originalInst->getOp())
    {
        // We need to special-case any instruction that is not
        // allocated like an ordinary `IRInst` with trailing args.
    case kIROp_IntLit:
    case kIROp_FloatLit:
    case kIROp_BoolLit:
    case kIROp_StringLit:
    case kIROp_BlobLit:
    case kIROp_PtrLit:
    case kIROp_VoidLit:
        return cloneValue(context, originalInst);

    case kIROp_Func:
        return cloneFuncImpl(context, builder, cast<IRFunc>(originalInst), originalValues);

    case kIROp_GlobalVar:
        return cloneGlobalVarImpl(
            context,
            builder,
            cast<IRGlobalVar>(originalInst),
            originalValues);

    case kIROp_GlobalParam:
        return cloneGlobalParamImpl(
            context,
            builder,
            cast<IRGlobalParam>(originalInst),
            originalValues);

    case kIROp_GlobalConstant:
        return cloneGlobalConstantImpl(
            context,
            builder,
            cast<IRGlobalConstant>(originalInst),
            originalValues);

    case kIROp_WitnessTable:
        return cloneWitnessTableImpl(
            context,
            builder,
            cast<IRWitnessTable>(originalInst),
            originalValues);

    case kIROp_StructType:
        return cloneStructTypeImpl(
            context,
            builder,
            cast<IRStructType>(originalInst),
            originalValues);

    case kIROp_InterfaceType:
        return cloneInterfaceTypeImpl(
            context,
            builder,
            cast<IRInterfaceType>(originalInst),
            originalValues);

    case kIROp_Generic:
        return cloneGenericImpl(context, builder, cast<IRGeneric>(originalInst), originalValues);

    case kIROp_StructKey:
        return cloneStructKeyImpl(
            context,
            builder,
            cast<IRStructKey>(originalInst),
            originalValues);

    case kIROp_GlobalGenericParam:
        return cloneGlobalGenericParamImpl(
            context,
            builder,
            cast<IRGlobalGenericParam>(originalInst),
            originalValues);

    default:
        break;
    }

    // The common case is that we just need to construct a cloned
    // instruction with the right number of operands, intialize
    // it, and then add it to the sequence.
    UInt argCount = originalInst->getOperandCount();
    ShortList<IRInst*> newArgs;
    newArgs.setCount(argCount);
    auto oldBuilder = context->builder;
    context->builder = builder;
    for (UInt aa = 0; aa < argCount; ++aa)
    {
        IRInst* originalArg = originalInst->getOperand(aa);
        IRInst* clonedArg = cloneValue(context, originalArg);
        newArgs[aa] = clonedArg;
    }
    auto funcType = cloneType(context, originalInst->getFullType());
    context->builder = oldBuilder;

    IRInst* clonedInst = builder->createIntrinsicInst(
        funcType,
        originalInst->getOp(),
        argCount,
        newArgs.getArrayView().getBuffer());
    builder->addInst(clonedInst);
    registerClonedValue(context, clonedInst, originalValues);
    if (canInstContainBasicBlocks(clonedInst->getOp()))
        cloneGlobalValueWithCodeCommon(
            context,
            (IRGlobalValueWithCode*)clonedInst,
            (IRGlobalValueWithCode*)originalInst,
            originalValues);
    else
        cloneDecorationsAndChildren(context, clonedInst, originalInst);
    cloneExtraDecorations(context, clonedInst, originalValues);
    return clonedInst;
}

IRInst* cloneGlobalValueImpl(
    IRSpecContext* context,
    IRInst* originalInst,
    IROriginalValuesForClone const& originalValues)
{
    auto clonedValue =
        cloneInst(context, &context->shared->builderStorage, originalInst, originalValues);
    clonedValue->moveToEnd();
    return clonedValue;
}

/// Clone a global value, which has the given `originalLinkage`.
///
/// The `originalVal` is a known global IR value with that linkage, if one is available.
/// (It is okay for this parameter to be null).
///
IRInst* cloneGlobalValueWithLinkage(
    IRSpecContext* context,
    IRInst* originalVal,
    IRLinkageDecoration* originalLinkage)
{
    // If the global value being cloned is already in target module, don't clone
    // Why checking this?
    //   When specializing a generic function G (which is already in target module),
    //   where G calls a normal function F (which is already in target module),
    //   then when we are making a copy of G via cloneFuncCommom(), it will recursively clone F,
    //   however we don't want to make a duplicate of F in the target module.
    if (originalVal->getParent() == context->getModule()->getModuleInst())
        return originalVal;

    // Check if we've already cloned this value, for the case where
    // an original value has already been established.
    if (originalVal)
    {
        if (IRInst* clonedVal = findClonedValue(context, originalVal))
        {
            return clonedVal;
        }
    }

    if (!originalLinkage)
    {
        // If there is no mangled name, then we assume this is a local symbol,
        // and it can't possibly have multiple declarations.
        return cloneGlobalValueImpl(context, originalVal, IROriginalValuesForClone(originalVal));
    }

    //
    // We will scan through all of the available declarations
    // with the same mangled name as `originalVal` and try
    // to pick the "best" one for our target.

    auto mangledName = originalLinkage->getMangledName();
    IRSpecSymbol* sym = context->findSymbols(mangledName);
    if (!sym)
    {
        if (!originalVal)
            return nullptr;

        // This shouldn't happen!
        SLANG_UNEXPECTED("no matching values registered");
        UNREACHABLE_RETURN(cloneGlobalValueImpl(context, originalVal, IROriginalValuesForClone()));
    }

    // We will try to track the "best" declaration we can find.
    //
    // Generally, one declaration will be better than another if it is
    // more specialized for the chosen target. Otherwise, we simply favor
    // definitions over declarations.
    //
    IRInst* bestVal = nullptr;
    for (IRSpecSymbol* ss = sym; ss; ss = ss->nextWithSameName)
    {
        IRInst* newVal = ss->irGlobalValue;
        if (isBetterForTarget(context, newVal, bestVal))
            bestVal = newVal;
    }

    if (!bestVal)
    {
        return nullptr;
    }

    // Check if we've already cloned this value, for the case where
    // we didn't have an original value (just a name), but we've
    // now found a representative value.
    if (!originalVal)
    {
        if (IRInst* clonedVal = findClonedValue(context, bestVal))
        {
            return clonedVal;
        }
    }

    return cloneGlobalValueImpl(context, bestVal, IROriginalValuesForClone(sym));
}

// Clone a global value, where `originalVal` is one declaration/definition, but we might
// have to consider others, in order to find the "best" version of the symbol.
IRInst* cloneGlobalValue(IRSpecContext* context, IRInst* originalVal)
{
    // We are being asked to clone a particular global value, but in
    // the IR that comes out of the front-end there could still
    // be multiple, target-specific, declarations of any given
    // global value, all of which share the same mangled name.
    return cloneGlobalValueWithLinkage(
        context,
        originalVal,
        originalVal->findDecoration<IRLinkageDecoration>());
}

void insertGlobalValueSymbol(IRSharedSpecContext* sharedContext, IRInst* gv)
{
    auto linkage = gv->findDecoration<IRLinkageDecoration>();

    // Don't try to register a symbol for global values
    // that don't have linkage.
    //
    if (!linkage)
        return;

    auto mangledName = String(linkage->getMangledName());

    RefPtr<IRSpecSymbol> sym = new IRSpecSymbol();
    sym->irGlobalValue = gv;

    RefPtr<IRSpecSymbol> prev;
    if (sharedContext->symbols.tryGetValue(mangledName, prev))
    {
        sym->nextWithSameName = prev->nextWithSameName;
        prev->nextWithSameName = sym;
    }
    else
    {
        sharedContext->symbols.add(mangledName, sym);
    }

    if (as<IRImportDecoration>(linkage))
        sharedContext->isImportedSymbol.tryGetValueOrAdd(mangledName, true);
    else
        sharedContext->isImportedSymbol.set(mangledName, false);
}

void insertGlobalValueSymbols(IRSharedSpecContext* sharedContext, IRModule* originalModule)
{
    if (!originalModule)
        return;

    for (auto ii : originalModule->getGlobalInsts())
    {
        insertGlobalValueSymbol(sharedContext, ii);
    }
}

void initializeSharedSpecContext(
    IRSharedSpecContext* sharedContext,
    Session* session,
    IRModule* inModule,
    CodeGenTarget target,
    TargetRequest* targetReq)
{
    RefPtr<IRModule> module = inModule;
    if (!module)
    {
        module = IRModule::create(session);
    }

    sharedContext->builderStorage = IRBuilder(module);

    sharedContext->module = module;
    sharedContext->target = target;
    sharedContext->targetReq = targetReq;
}

struct IRSpecializationState
{
    CodeGenTarget target;
    TargetRequest* targetReq;

    IRModule* irModule = nullptr;

    IRSharedSpecContext sharedContextStorage;
    IRSpecContext contextStorage;

    IRSpecEnv globalEnv;

    IRSharedSpecContext* getSharedContext() { return &sharedContextStorage; }
    IRSpecContext* getContext() { return &contextStorage; }

    IRSpecializationState() { contextStorage.env = &globalEnv; }

    ~IRSpecializationState()
    {
        contextStorage = IRSpecContext();
        sharedContextStorage = IRSharedSpecContext();
    }
};

static bool _isHLSLExported(IRInst* inst)
{
    for (auto decoration : inst->getDecorations())
    {
        const auto op = decoration->getOp();
        if (op == kIROp_HLSLExportDecoration || op == kIROp_DownstreamModuleExportDecoration)
        {
            return true;
        }
    }
    return false;
}

static bool doesFuncHaveDefinition(IRFunc* func)
{
    if (func->getFirstBlock() != nullptr)
        return true;
    for (auto decor : func->getDecorations())
    {
        switch (decor->getOp())
        {
        case kIROp_IntrinsicOpDecoration:
        case kIROp_TargetIntrinsicDecoration:
            return true;
        default:
            continue;
        }
    }
    return false;
}

static bool doesWitnessTableHaveDefinition(IRWitnessTable* wt)
{
    auto interfaceType = as<IRInterfaceType>(wt->getConformanceType());
    if (!interfaceType)
        return true;
    auto interfaceRequirementCount = interfaceType->getRequirementCount();
    if (interfaceRequirementCount == 0)
        return true;
    for (auto entry : wt->getChildren())
    {
        if (as<IRWitnessTableEntry>(entry))
            return true;
    }
    return false;
}

static bool doesTargetAllowUnresolvedFuncSymbol(TargetRequest* req)
{
    switch (req->getTarget())
    {
    case CodeGenTarget::HLSL:
    case CodeGenTarget::Metal:
    case CodeGenTarget::MetalLib:
    case CodeGenTarget::MetalLibAssembly:
    case CodeGenTarget::WGSL:
    case CodeGenTarget::DXIL:
    case CodeGenTarget::DXILAssembly:
    case CodeGenTarget::HostCPPSource:
    case CodeGenTarget::PyTorchCppBinding:
    case CodeGenTarget::ShaderHostCallable:
    case CodeGenTarget::ShaderSharedLibrary:
    case CodeGenTarget::HostHostCallable:
    case CodeGenTarget::CPPSource:
    case CodeGenTarget::CUDASource:
    case CodeGenTarget::SPIRV:
        if (req->getOptionSet().getBoolOption(CompilerOptionName::IncompleteLibrary))
            return true;
        return false;
    default:
        return false;
    }
}

static void diagnoseUnresolvedSymbols(TargetRequest* req, DiagnosticSink* sink, IRModule* module)
{
    for (auto globalSym : module->getGlobalInsts())
    {
        if (globalSym->findDecoration<IRImportDecoration>())
        {
            for (;;)
            {
                if (auto constant = as<IRGlobalConstant>(globalSym))
                {
                    if (constant->getOperandCount() == 0)
                        sink->diagnose(
                            globalSym->sourceLoc,
                            Diagnostics::unresolvedSymbol,
                            globalSym);
                }
                else if (auto genericSym = as<IRGeneric>(globalSym))
                {
                    globalSym = findGenericReturnVal(genericSym);
                    continue;
                }
                else if (auto funcSym = as<IRFunc>(globalSym))
                {
                    if (!doesFuncHaveDefinition(funcSym) &&
                        !doesTargetAllowUnresolvedFuncSymbol(req))
                        sink->diagnose(
                            globalSym->sourceLoc,
                            Diagnostics::unresolvedSymbol,
                            globalSym);
                }
                else if (auto witnessSym = as<IRWitnessTable>(globalSym))
                {
                    if (!doesWitnessTableHaveDefinition(witnessSym))
                    {
                        sink->diagnose(
                            globalSym->sourceLoc,
                            Diagnostics::unresolvedSymbol,
                            witnessSym);
                        if (auto concreteType = witnessSym->getConcreteType())
                            sink->diagnose(
                                concreteType->sourceLoc,
                                Diagnostics::seeDeclarationOf,
                                concreteType);
                    }
                }
                break;
            }
        }
    }
}

void convertAtomicToStorageBuffer(
    IRSpecContext* context,
    Dictionary<int, List<IRInst*>>& bindingToInstMapUnsorted)
{
    // Atomic_uint definitions needs to become a storage buffer to follow
    // GL_EXT_vulkan_glsl_relaxed and to allow translation of atomic_uint into SPIRV

    IRBuilder builder = *context->builder;

    for (auto& bindingToInstList : bindingToInstMapUnsorted)
    {
        int64_t maxOffset = 0;
        for (auto& i : bindingToInstList.second)
        {
            int64_t currOffset =
                int64_t(i->findDecoration<IRGLSLOffsetDecoration>()->getOffset()->getValue());
            maxOffset = (maxOffset < currOffset) ? currOffset : maxOffset;
        }
        auto instToSwitch = *bindingToInstList.second.begin();
        builder.setInsertBefore(instToSwitch);

        auto elementType = builder.getArrayType(
            builder.getUIntType(),
            builder.getIntValue(builder.getUIntType(), (maxOffset / sizeof(uint32_t)) + 1));

        StringBuilder nameStruct;
        nameStruct << "atomic_uints";
        nameStruct << bindingToInstList.first;
        nameStruct << "_t";
        nameStruct << "_paramGroup";
        auto structType = builder.createStructType();
        builder.addNameHintDecoration(structType, nameStruct.produceString().getUnownedSlice());

        auto elementBufferKey = builder.createStructKey();
        builder.addNameHintDecoration(elementBufferKey, UnownedStringSlice("_data"));
        auto elementBufferType = elementType;
        auto _dataField =
            builder.createStructField(structType, elementBufferKey, elementBufferType);

        auto std430 = builder._createInst(
            sizeof(IRTypeLayoutRules),
            builder.getType(kIROp_Std430BufferLayoutType),
            kIROp_Std430BufferLayoutType);
        IRGLSLShaderStorageBufferType* storageBuffer;
        {
            IRInst* ops[] = {structType, std430};
            storageBuffer = builder.createGLSLShaderStorableBufferType(2, ops);
        }

        instToSwitch->setFullType(storageBuffer);

        // All references to a atomic_uint need to be an element ref. to emulate storage buffer
        // usage All function calls must be inlined since storage buffers cannot pass as
        // parameters to atomic methods
        for (auto& i : bindingToInstList.second)
        {
            int64_t currOffset =
                int64_t(i->findDecoration<IRGLSLOffsetDecoration>()->getOffset()->getValue());

            // we need a next node to be stored since the following code
            // changes IRUse* of the use->user node, meaning we will lose
            // our IRUse list of the atomic_uint being swapped out
            IRUse* next = nullptr;
            for (auto use = i->firstUse; use; use = next)
            {
                next = use->nextUse;
                auto user = use->user;

                switch (user->getOp())
                {
                case kIROp_StructFieldLayoutAttr:
                    {
                        // Definitions do nothing if unused
                        break;
                    }
                case kIROp_Call:
                    {
                        builder.setInsertBefore(user);
                        auto fieldAddress = builder.emitFieldAddress(
                            builder.getPtrType(_dataField->getFieldType()),
                            instToSwitch,
                            _dataField->getKey());
                        auto elementAddr = builder.emitElementAddress(
                            builder.getPtrType(builder.getUIntType()),
                            fieldAddress,
                            builder.getIntValue(builder.getIntType(), currOffset / 4));

                        user->setOperand(1, elementAddr);
                        auto funcTypeInst = (user->getOperand(0));
                        auto funcType = funcTypeInst->getFullType();

                        auto paramReplacment = builder.getInOutType(builder.getUIntType());
                        funcType->getOperand(1)->replaceUsesWith(paramReplacment);
                        builder.addForceInlineDecoration(funcTypeInst);

                        break;
                    }
                }
            }
            if (i->typeUse.usedValue->getOp() == kIROp_GLSLAtomicUintType)
            {
                i->removeAndDeallocate();
            }
        }
    }
}

void GLSLReplaceAtomicUint(IRSpecContext* context, TargetProgram* targetProgram, IRModule* irModule)
{
    if (!targetProgram->getOptionSet().getBoolOption(CompilerOptionName::AllowGLSL))
        return;

    Dictionary<int, List<IRInst*>> bindingToInstMapUnsorted;
    for (auto inst : irModule->getGlobalInsts())
    {
        if (inst->typeUse.usedValue)
        {
            switch (inst->typeUse.usedValue->getOp())
            {
            case kIROp_GLSLAtomicUintType:
                {
                    // atomic_uint are supported by GLSL->VK through converting to a different
                    // type (GL_EXT_vulkan_glsl_relaxed). atomic_uint are not supported by
                    // SPIR-V->VK; this means that to get SPIR-V to work we must convert the
                    // type ourselves to an equivlent representation (storage buffer); the added
                    // benifit is that then HLSL is possible to emit as a target as well since
                    // atomic_uint is not an HLSL concept, but storageBuffer->RWBuffer is and
                    // HLSL concept
                    auto layout = inst->findDecoration<IRLayoutDecoration>()->getLayout();
                    auto layoutVal = as<IRVarOffsetAttr>(layout->getOperand(1));
                    assert(layoutVal != nullptr);
                    bindingToInstMapUnsorted
                        .getOrAddValue(uint32_t(layoutVal->getOffset()), List<IRInst*>())
                        .add(inst);
                    break;
                }
            };
        }
    }

    convertAtomicToStorageBuffer(context, bindingToInstMapUnsorted);
}

bool isDiffPairType(IRInst* type)
{
    for (;;)
    {
        auto type1 = (IRType*)unwrapAttributedType(type);
        auto type2 = unwrapArray(type1);
        if (type2 == type)
            break;
        type = type2;
    }
    return as<IRDifferentialPairTypeBase>(type) != nullptr;
}

bool doesModuleUseAutodiff(IRInst* inst)
{
    switch (inst->getOp())
    {
    case kIROp_Call:
        if (auto callee = getResolvedInstForDecorations(inst->getOperand(0)))
        {
            switch (callee->getOp())
            {
            case kIROp_ForwardDifferentiate:
            case kIROp_BackwardDifferentiate:
            case kIROp_BackwardDifferentiatePrimal:
            case kIROp_BackwardDifferentiatePropagate:
                return true;
            }
        }
        return false;
    case kIROp_DifferentialPairGetDifferentialUserCode:
    case kIROp_DifferentialPairGetPrimalUserCode:
    case kIROp_DifferentialPtrPairGetPrimal:
    case kIROp_DifferentialPtrPairGetDifferential:
        return true;
    case kIROp_StructField:
        return isDiffPairType(as<IRStructField>(inst)->getFieldType());
    case kIROp_Param:
        return isDiffPairType(inst->getDataType());
    default:
        for (auto child : inst->getChildren())
        {
            bool isImported = false;
            for (auto decor : child->getDecorations())
            {
                if (as<IRImportDecoration>(decor))
                {
                    isImported = true;
                    break;
                }
                else if (as<IRAutoPyBindCudaDecoration>(decor))
                {
                    return true;
                }
                else if (as<IRAutoPyBindExportInfoDecoration>(decor))
                {
                    return true;
                }
            }
            if (isImported)
                continue;
            for (auto decor : child->getDecorations())
            {
                if (isAutoDiffDecoration(decor))
                    return true;
            }
            if (doesModuleUseAutodiff(child))
                return true;
        }
        return false;
    }
}

void cloneUsedWitnessTableEntries(IRSpecContext* context)
{
    bool changed = true;
    while (changed)
    {
        changed = false;
        for (Index i = 0; i < context->witnessTables.getCount(); i++)
        {
            auto table = context->witnessTables[i].get();
            ShortList<UnownedStringSlice> entriesToRemove;
            for (auto entry : table->deferredEntries)
            {
                if (context->deferredWitnessTableEntryKeys.contains(entry.first))
                {
                    IRBuilder builder(table->clonedTable);
                    builder.setInsertInto(table->clonedTable);
                    auto deferredKeyCount = context->deferredWitnessTableEntryKeys.getCount();
                    cloneInst(context, &builder, entry.second);
                    entriesToRemove.add(entry.first);
                    if (deferredKeyCount != context->deferredWitnessTableEntryKeys.getCount())
                        changed = true;
                }
            }
            for (auto entry : entriesToRemove)
            {
                table->deferredEntries.remove(entry);
            }
        }
    }
}

LinkedIR linkIR(CodeGenContext* codeGenContext)
{
    SLANG_PROFILE;

    auto linkage = codeGenContext->getLinkage();
    auto program = codeGenContext->getProgram();
    auto session = codeGenContext->getSession();
    auto target = codeGenContext->getTargetFormat();
    auto targetProgram = codeGenContext->getTargetProgram();
    auto targetReq = codeGenContext->getTargetReq();

    // TODO: We need to make sure that the program we are being asked
    // to compile has been "resolved" so that it has no outstanding
    // unsatisfied requirements.

    IRSpecializationState stateStorage;
    auto state = &stateStorage;

    state->target = target;
    state->targetReq = targetReq;
    auto& irModules = stateStorage.contextStorage.irModules;

    auto sharedContext = state->getSharedContext();
    initializeSharedSpecContext(sharedContext, session, nullptr, target, targetReq);

    state->irModule = sharedContext->module;

    // We need to be able to look up IR definitions for any symbols in
    // modules that the program depends on (transitively). To
    // accelerate lookup, we will create a symbol table for looking
    // up IR definitions by their mangled name.
    //
    auto globalSession = static_cast<Session*>(linkage->getGlobalSession());
    List<IRModule*> builtinModules;
    for (auto& m : globalSession->coreModules)
        builtinModules.add(m->getIRModule());

    // Link modules in the program.
    program->enumerateIRModules(
        [&](IRModule* module)
        {
            if (module->getName() == globalSession->glslModuleName)
                builtinModules.add(module);
            else
                irModules.add(module);
        });

    // We will also consider the IR global symbols from the IR module
    // attached to the `TargetProgram`, since this module is
    // responsible for associating layout information to those
    // global symbols via decorations.
    //
    auto irModuleForLayout = targetProgram->getExistingIRModuleForLayout();
    if (irModuleForLayout)
        irModules.add(irModuleForLayout);

    Index userModuleCount = irModules.getCount();
    irModules.addRange(builtinModules);
    ArrayView<IRModule*> userModules = irModules.getArrayView(0, userModuleCount);

    // Check if any user module uses auto-diff, if so we will need to link
    // additional witnesses and decorations.
    for (IRModule* irModule : userModules)
    {
        if (sharedContext->useAutodiff)
            break;
        sharedContext->useAutodiff = doesModuleUseAutodiff(irModule->getModuleInst());
    }

    auto context = state->getContext();

    // Combine all of the contents of IRGlobalHashedStringLiterals
    {
        StringSlicePool pool(StringSlicePool::Style::Empty);
        for (IRModule* irModule : userModules)
        {
            findGlobalHashedStringLiterals(irModule, pool);
        }
        addGlobalHashedStringLiterals(pool, state->irModule);
    }

    // Set up shared and builder insert point

    context->shared = sharedContext;
    context->builder = &sharedContext->builderStorage;

    context->builder->setInsertInto(context->getModule()->getModuleInst());

    // Next, we make sure to clone the global value for
    // the entry point function itself, and rely on
    // this step to recursively copy over anything else
    // it might reference.
    //
    // Note: We query the mangled name of the entry point from
    // the `program` instead of the `entryPoint` directly,
    // because the `program` will include any specialization
    // arguments which might end up affecting the mangled
    // entry point name.
    //

    List<IRFunc*> irEntryPoints;
    for (auto entryPointIndex : codeGenContext->getEntryPointIndices())
    {
        auto entryPointMangledName = program->getEntryPointMangledName(entryPointIndex);
        auto nameOverride = program->getEntryPointNameOverride(entryPointIndex);
        auto entryPoint = program->getEntryPoint(entryPointIndex).get();
        irEntryPoints.add(specializeIRForEntryPoint(
            context,
            entryPointMangledName,
            entryPoint,
            nameOverride.getUnownedSlice()));
    }

    // Layout information for global shader parameters is also required,
    // and in particular every global parameter that is part of the layout
    // should be present in the initial IR module so that steps that
    // need to operate on all the global parameters can do so.
    //
    IRVarLayout* irGlobalScopeVarLayout = nullptr;
    if (irModuleForLayout)
    {
        if (auto irGlobalScopeLayoutDecoration =
                irModuleForLayout->getModuleInst()->findDecoration<IRLayoutDecoration>())
        {
            auto irOriginalGlobalScopeVarLayout = irGlobalScopeLayoutDecoration->getLayout();
            irGlobalScopeVarLayout =
                cast<IRVarLayout>(cloneValue(context, irOriginalGlobalScopeVarLayout));
        }
    }

    // Bindings for global generic parameters are currently represented
    // as stand-alone global-scope instructions in the IR module for
    // `SpecializedComponentType`s. These instructions are required for
    // correct codegen, and so we must make sure to copy them all over,
    // even though they are not directly referenced.
    //
    // TODO: We should change these to decorations, akin to how
    // `[bindExistentialSlots(...)]` works, so that they can be attached
    // to the relevant parameters and cloned via `cloneExtraDecorations`.
    // In the long run we do not want to *ever* iterate over all the
    // instructions in all the input modules.
    //

    for (IRModule* irModule : userModules)
    {
        for (auto inst : irModule->getGlobalInsts())
        {
            if (auto bindInst = as<IRBindGlobalGenericParam>(inst))
            {
                cloneValue(context, bindInst);
            }
        }
    }

    bool shouldCopyGlobalParams =
        linkage->m_optionSet.getBoolOption(CompilerOptionName::PreserveParameters);

    for (IRModule* irModule : irModules)
    {
        for (auto inst : irModule->getGlobalInsts())
        {
            // We need to copy over exported symbols,
            // and any global parameters if preserve-params option is set.
            if (_isHLSLExported(inst) || shouldCopyGlobalParams && as<IRGlobalParam>(inst) ||
                sharedContext->useAutodiff &&
                    (as<IRDifferentiableTypeAnnotation>(inst) ||
                     inst->findDecorationImpl(kIROp_AutoDiffBuiltinDecoration) != nullptr))
            {
                auto cloned = cloneValue(context, inst);
                if (!cloned->findDecorationImpl(kIROp_KeepAliveDecoration))
                {
                    context->builder->addKeepAliveDecoration(cloned);
                }
            }
        }
    }

    // In previous steps, we have skipped cloning the witness table entries, and
    // registered any used witness table entry keys to context->deferredWitnessTableEntryKeys
    // for on-demand cloning. Now we will use the deferred keys to clone the witness table
    // entries that are referenced.
    cloneUsedWitnessTableEntries(context);

    // It is possible that metadata has been attached to the input modules
    // themselves, which should be copied over to the output module.
    //
    // In cases where multiple input modules specify the same metadata
    // decoration, we will need rules to merge such decorations according
    // to opcode-specific policy (e.g., a `[maxStackSizeRequired(...)]`
    // decoration might use a maximum over all specified values, while a
    // `[assumedWaveSize(...)]` decoration might require that all specified
    // values match exactly).
    //
    for (IRModule* irModule : userModules)
    {
        for (auto decoration : irModule->getModuleInst()->getDecorations())
        {
            switch (decoration->getOp())
            {
            case kIROp_NVAPISlotDecoration:
                {
                    // For now we just clone every decoration we see,
                    // which means that an arbitrary one will end up
                    // "winning" and being the one found by searches
                    // in later code.
                    //
                    // TODO: need validation to check if decorations are
                    // consistent with one another, in the case where
                    // multiple input modules have matching decorations.
                    //
                    auto cloned = cloneInst(context, context->builder, decoration);
                    cloned->insertAtStart(state->irModule->getModuleInst());
                }
                break;

            default:
                break;
            }
        }
    }

    // Specialize target_switch branches to use the best branch for the target.
    specializeTargetSwitch(targetReq, state->irModule, codeGenContext->getSink());

    // Diagnose on unresolved symbols if we are compiling into a target that does
    // not allow incomplete symbols.
    // At this point, we should not see any [import] symbols that does not have a
    // definition.
    diagnoseUnresolvedSymbols(targetReq, codeGenContext->getSink(), state->irModule);

    // type-use reformatter of GLSL types (only if compiler is set to AllowGLSL mode)
    // which are not supported by SPIRV->Vulkan but is supported by GLSL->Vulkan through
    // compiler magic tricks
    GLSLReplaceAtomicUint(context, targetProgram, state->irModule);

    // TODO: *technically* we should consider the case where
    // we have global variables with initializers, since
    // these should get run whether or not the entry point
    // references them.
    //
    // Or alternatively we can define by fiat that the initializers
    // on global variables get run at an unspecified time between
    // program startup and the first access to a given global.
    // Such a definition gives us the freedom to eliminate globals
    // that are never accessed, while still doing "eager"
    // initialization for globals that are referenced (instead of
    // having to add the overhead of lazy initialization a la
    // function-`static` variables).

    // Now that we've cloned the entry point and everything
    // it refers to, we can package up the data we return
    // to the caller.
    //
    LinkedIR linkedIR;
    linkedIR.module = state->irModule;
    linkedIR.globalScopeVarLayout = irGlobalScopeVarLayout;
    linkedIR.entryPoints = irEntryPoints;
    return linkedIR;
}

struct ReplaceGlobalConstantsPass
{
    void process(IRModule* module)
    {
        _processInstRec(module->getModuleInst());

        for (auto inst : instsToRemove)
            inst->removeAndDeallocate();
    }

    List<IRInst*> instsToRemove;

    void _processInstRec(IRInst* inst)
    {
        if (auto globalConstant = as<IRGlobalConstant>(inst))
        {
            if (auto val = globalConstant->getValue())
            {
                // Attempt to transfer the name hint from the global
                // constant to its value. If multiple constants
                // have the same value, then the first one will "win"
                // and transfer its name over.
                //
                if (auto nameHint = globalConstant->findDecoration<IRNameHintDecoration>())
                {
                    if (!val->findDecoration<IRNameHintDecoration>())
                    {
                        nameHint->removeFromParent();
                        nameHint->insertAtStart(val);
                    }
                }

                // Replace the global constant
                globalConstant->replaceUsesWith(val);
                instsToRemove.add(globalConstant);
            }
        }

        for (auto child : inst->getDecorationsAndChildren())
        {
            _processInstRec(child);
        }
    }
};

void replaceGlobalConstants(IRModule* module)
{
    ReplaceGlobalConstantsPass pass;
    pass.process(module);
}


} // namespace Slang
