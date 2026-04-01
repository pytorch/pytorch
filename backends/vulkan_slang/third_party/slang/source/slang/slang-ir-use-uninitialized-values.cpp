#include "slang-ir-use-uninitialized-values.h"

#include "slang-ir-insts.h"
#include "slang-ir-reachability.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{
static bool isMetaOp(IRInst* inst)
{
    switch (inst->getOp())
    {
    // These instructions only look at the parameter's type,
    // so passing an undefined value to them is permissible
    case kIROp_IsBool:
    case kIROp_IsInt:
    case kIROp_IsUnsignedInt:
    case kIROp_IsSignedInt:
    case kIROp_IsHalf:
    case kIROp_IsFloat:
    case kIROp_IsVector:
    case kIROp_GetNaturalStride:
    case kIROp_TypeEquals:
        return true;
    default:
        break;
    }

    return false;
}

static bool isUninitializedValue(IRInst* inst)
{
    // Also consider var since it does not
    // automatically mean it will be initialized
    // (at least not as the user may have intended)
    return (inst->m_op == kIROp_undefined) || (inst->m_op == kIROp_Var);
}

static bool isUnmodifying(IRFunc* func)
{
    auto intr = func->findDecoration<IRIntrinsicOpDecoration>();
    return (intr && intr->getIntrinsicOp() == kIROp_Unmodified);
}

enum ParameterCheckType
{
    Never,  // Parameter does NOT to be checked for uninitialization (e.g. is `in` or special type)
    AsOut,  // Parameter DOES need to be checked for usage before initializations
    AsInOut // Parameter DOES need to be checked to see if it is ever written to
};

static ParameterCheckType isPotentiallyUnintended(IRParam* param, Stage stage, int index)
{
    IRType* type = param->getFullType();
    if (auto out = as<IROutType>(param->getFullType()))
    {
        // Don't check `out Vertices<T>` or `out Indices<T>` parameters
        // in mesh shaders.
        // TODO: we should find a better way to represent these mesh shader
        // parameters so they conform to the initialize before use convention.
        // For example, we can use a `OutputVetices` and `OutputIndices` type
        // to represent an output, like `OutputPatch` in domain shader.
        // For now, we just skip the check for these parameters.
        switch (out->getValueType()->getOp())
        {
        case kIROp_VerticesType:
        case kIROp_IndicesType:
        case kIROp_PrimitivesType:
            return Never;
        default:
            break;
        }

        return AsOut;
    }
    else if (auto inout = as<IRInOutType>(type))
    {
        // TODO: some way to check if the method
        // is actually used for autodiff
        if (as<IRDifferentialPairUserCodeType>(inout->getValueType()))
            return Never;

        switch (stage)
        {
        case Stage::AnyHit:
        case Stage::ClosestHit:
            // In HLSL the payload is required to be `inout`
            return (index == 0) ? Never : AsInOut;
        case Stage::Geometry:
            // Second parameter is the triangle stream
            return (index == 1) ? Never : AsInOut;
        default:
            break;
        }

        return AsInOut;
    }

    return Never;
}

static bool isAliasable(IRInst* inst)
{
    switch (inst->getOp())
    {
    // These instructions generate (implicit) references to inst
    case kIROp_FieldExtract:
    case kIROp_FieldAddress:
    case kIROp_GetElement:
    case kIROp_GetElementPtr:
    case kIROp_InOutImplicitCast:
        return true;
    default:
        break;
    }

    return false;
}

static bool isDifferentiableFunc(IRInst* func)
{
    for (auto decor = func->getFirstDecoration(); decor; decor = decor->getNextDecoration())
    {
        switch (decor->getOp())
        {
        case kIROp_ForwardDerivativeDecoration:
        case kIROp_ForwardDifferentiableDecoration:
        case kIROp_BackwardDerivativeDecoration:
        case kIROp_BackwardDifferentiableDecoration:
        case kIROp_UserDefinedBackwardDerivativeDecoration:
            return true;
        default:
            break;
        }
    }

    return false;
}

// The `upper` field contains the struct that the type is
// is contained in. It is used to check for empty structs.
static bool canIgnoreType(IRType* type, IRType* upper)
{
    // In case specialization returns a function instead
    if (!type)
        return true;

    if (as<IRVoidType>(type))
        return true;

    // For structs, ignore if its empty
    if (auto str = as<IRStructType>(type))
    {
        int count = 0;
        for (auto field : str->getFields())
        {
            IRType* ftype = field->getFieldType();
            count += !canIgnoreType(ftype, type);
        }

        return (count == 0);
    }

    // Nothing to initialize for a pure interface
    if (as<IRInterfaceType>(type))
        return true;

    // We don't know what type it will be yet.
    if (as<IRParam>(type))
        return true;

    // For pointers, check the value type (primarily for globals)
    if (auto ptr = as<IRPtrType>(type))
    {
        // Avoid the recursive step if its a
        // recursive structure like a linked list
        IRType* ptype = ptr->getValueType();
        if (auto resolvedType = as<IRType>(getResolvedInstForDecorations(ptype)))
            ptype = resolvedType;
        return (ptype != upper) && canIgnoreType(ptype, upper);
    }

    // In the case of specializations, check returned type
    if (auto spec = as<IRSpecialize>(type))
    {
        IRInst* inner = getResolvedInstForDecorations(spec);
        IRType* innerType = (IRType*)(inner);
        return canIgnoreType(innerType, upper);
    }

    return false;
}

static List<IRInst*> getAliasableInstructions(IRInst* inst)
{
    List<IRInst*> addresses;

    addresses.add(inst);
    for (auto use = inst->firstUse; use; use = use->nextUse)
    {
        IRInst* user = use->getUser();

        // Meta instructions only use the argument type
        if (isMetaOp(user) || !isAliasable(user))
            continue;

        addresses.addRange(getAliasableInstructions(user));
    }

    return addresses;
}

enum InstructionUsageType
{
    None,        // Instruction neither stores nor loads from the soruce (e.g. meta operations)
    Store,       // Instruction acts as a write to the source
    StoreParent, // Instruction's parent acts as a write to the source
    Load         // Instruciton acts as a load from the source
};

static InstructionUsageType getCallUsageType(IRCall* call, IRInst* inst)
{
    IRInst* callee = call->getCallee();

    // Resolve the actual function
    IRFunc* ftn = nullptr;
    IRFuncType* ftype = nullptr;
    if (auto spec = as<IRSpecialize>(callee))
        ftn = as<IRFunc>(getResolvedInstForDecorations(spec));

    // Differentiable functions are mostly ignored, treated as having inout parameters
    else if (as<IRForwardDifferentiate>(callee))
        return Store;
    else if (as<IRBackwardDifferentiate>(callee))
        return Store;

    else if (auto wit = as<IRLookupWitnessMethod>(callee))
        ftype = as<IRFuncType>(wit->getFullType());
    else
        ftn = as<IRFunc>(callee);

    // Find the argument index so we can fetch the type
    int index = 0;

    auto args = call->getArgsList();
    for (int i = 0; i < args.getCount(); i++)
    {
        if (args[i] == inst)
        {
            index = i;
            break;
        }
    }

    if (ftn)
        ftype = as<IRFuncType>(ftn->getFullType());

    if (!ftype)
        return None;

    // Consider it as a store if its passed
    // as an out/inout/ref parameter
    auto type = unwrapAttributedType(ftype->getParamType(index));
    return (as<IROutType>(type) || as<IRInOutType>(type) || as<IRRefType>(type)) ? Store : Load;
}

static InstructionUsageType getInstructionUsageType(IRInst* user, IRInst* inst)
{
    // Meta intrinsics (which evaluate on type) do nothing
    if (isMetaOp(user))
        return None;

    // Ignore instructions generating more aliases
    if (isAliasable(user))
        return None;

    switch (user->getOp())
    {
    case kIROp_loop:
    case kIROp_unconditionalBranch:
        // TODO: Ignore branches for now
        return None;

    case kIROp_Call:
        // Function calls can be either
        // stores or loads depending on
        // whether the callee takes it
        // in as a out parameter or not
        return getCallUsageType(as<IRCall>(user), inst);

    // These instructions will store data...
    case kIROp_Store:
    case kIROp_SwizzledStore:
    case kIROp_SPIRVAsm:
    case kIROp_AtomicStore:
        return Store;

    case kIROp_SPIRVAsmOperandInst:
        // For SPIRV asm instructions, need to check out the entire
        // block when doing reachability checks
        return StoreParent;

    case kIROp_MakeExistential:
    case kIROp_MakeExistentialWithRTTI:
        // For specializing generic structs
        return Store;

    // Miscellaenous cases
    case kIROp_ManagedPtrAttach:
    case kIROp_Unmodified:
        return Store;

    default:
        // Default case is that if the instruction is a pointer, it
        // is considered a store, otherwise a load.
        if (as<IRPtrTypeBase>(user->getDataType()))
            return Store;
        return Load;
    }
}

static void collectSpecialCaseInstructions(List<IRInst*>& stores, IRBlock* block)
{
    for (auto inst = block->getFirstInst(); inst; inst = inst->next)
    {
        if (as<IRGenericAsm>(inst))
            stores.add(inst);
    }
}

static void collectInstructionByUsage(
    List<IRInst*>& stores,
    List<IRInst*>& loads,
    IRInst* user,
    IRInst* inst)
{
    InstructionUsageType usage = getInstructionUsageType(user, inst);
    switch (usage)
    {
    case Load:
        return loads.add(user);
    case Store:
        return stores.add(user);
    case StoreParent:
        return stores.add(user->getParent());
    }
}

static void cancelLoads(
    ReachabilityContext& reachability,
    const List<IRInst*>& stores,
    List<IRInst*>& loads)
{
    // Remove all loads which are reachable from stores
    for (auto store : stores)
    {
        for (Index i = 0; i < loads.getCount();)
        {
            if (reachability.isInstReachable(store, loads[i]))
                loads.fastRemoveAt(i);
            else
                i++;
        }
    }
}

static void collectAliasableLoadStores(IRInst* inst, List<IRInst*>& stores, List<IRInst*>& loads)
{
    auto addresses = getAliasableInstructions(inst);

    for (auto alias : addresses)
    {
        // TODO: Mark specific parts assigned to for partial initialization checks
        for (auto use = alias->firstUse; use; use = use->nextUse)
            collectInstructionByUsage(stores, loads, use->getUser(), alias);
    }
}

static List<IRInst*> getUnresolvedParamLoads(
    ReachabilityContext& reachability,
    IRFunc* func,
    IRInst* inst)
{
    // Partition instructions
    List<IRInst*> stores;
    List<IRInst*> loads;

    collectAliasableLoadStores(inst, stores, loads);

    // Special cases for parameters
    for (const auto& b : func->getBlocks())
    {
        collectSpecialCaseInstructions(stores, b);

        auto t = b->getTerminator();
        if (as<IRReturn>(t))
            loads.add(t);
    }

    cancelLoads(reachability, stores, loads);

    return loads;
}

static List<IRInst*> getUnresolvedVariableLoads(ReachabilityContext& reachability, IRInst* inst)
{
    // Partition instructions
    List<IRInst*> stores;
    List<IRInst*> loads;

    collectAliasableLoadStores(inst, stores, loads);

    cancelLoads(reachability, stores, loads);

    return loads;
}

static bool isInstStoredInto(ReachabilityContext& reachability, IRInst* reference, IRInst* inst)
{
    List<IRInst*> stores;
    List<IRInst*> loads;

    for (auto alias : getAliasableInstructions(inst))
    {
        for (auto use = alias->firstUse; use; use = use->nextUse)
            collectInstructionByUsage(stores, loads, use->getUser(), alias);
    }

    for (auto store : stores)
    {
        if (reachability.isInstReachable(store, reference))
            return true;
    }

    return false;
}

static IRInst* traceInstOrigin(IRInst* inst)
{
    if (auto load = as<IRLoad>(inst))
        return traceInstOrigin(load->getPtr());

    return inst;
}

static bool isReturnedValue(IRInst* inst)
{
    for (auto use = inst->firstUse; use; use = use->nextUse)
    {
        IRInst* user = use->getUser();
        if (as<IRReturn>(user))
            return true;

        // Loading from a Ptr type should be
        // treated as an aliased path to any return
        IRLoad* load = as<IRLoad>(user);
        if (load && isReturnedValue(load))
            return true;
    }
    return false;
}

static bool isDirectlyWrittenTo(IRInst* inst)
{
    for (auto use = inst->firstUse; use; use = use->nextUse)
    {
        InstructionUsageType usage = getInstructionUsageType(use->getUser(), inst);
        if (usage == Store || usage == StoreParent)
            return true;
    }

    return false;
}

static List<IRStructField*> checkFieldsFromExit(
    ReachabilityContext& reachability,
    IRReturn* ret,
    IRStructType* type)
{
    IRInst* origin = traceInstOrigin(ret->getVal());

    // We don't want to warn on delegated construction
    if (!isUninitializedValue(origin))
        return {};

    // Check if the origin instruction is ever written to
    if (isDirectlyWrittenTo(origin))
        return {};

    // Now we can look for all references to fields
    HashSet<IRStructKey*> usedKeys;
    for (auto use = origin->firstUse; use; use = use->nextUse)
    {
        IRInst* user = use->getUser();

        auto fieldAddress = as<IRFieldAddress>(user);
        if (!fieldAddress || !isInstStoredInto(reachability, ret, user))
            continue;

        IRInst* field = fieldAddress->getField();
        usedKeys.add(as<IRStructKey>(field));
    }

    List<IRStructField*> uninitializedFields;

    auto fields = type->getFields();
    for (auto field : fields)
    {
        if (canIgnoreType(field->getFieldType(), nullptr))
            continue;

        if (!usedKeys.contains(field->getKey()))
            uninitializedFields.add(field);
    }

    return uninitializedFields;
}

static void checkConstructor(IRFunc* func, ReachabilityContext& reachability, DiagnosticSink* sink)
{
    auto constructor = func->findDecoration<IRConstructorDecorartion>();
    if (!constructor)
        return;

    IRStructType* stype = as<IRStructType>(func->getResultType());
    if (!stype)
        return;

    // Don't bother giving warnings if its not being used
    bool synthesized = constructor->getSynthesizedStatus();
    if (synthesized && !func->firstUse)
        return;

    auto printWarnings = [&](const List<IRStructField*>& fields, IRReturn* ret)
    {
        for (auto field : fields)
        {
            if (synthesized)
            {
                sink->diagnose(
                    field->getKey(),
                    Diagnostics::fieldNotDefaultInitialized,
                    stype,
                    field->getKey());
            }
            else
            {
                sink->diagnose(ret, Diagnostics::constructorUninitializedField, field->getKey());
            }
        }
    };

    // Work backwards, get exit points and find sources
    for (auto block : func->getBlocks())
    {
        for (auto inst = block->getFirstInst(); inst; inst = inst->next)
        {
            auto ret = as<IRReturn>(inst);
            if (!ret)
                continue;

            auto fields = checkFieldsFromExit(reachability, ret, stype);
            printWarnings(fields, ret);
        }
    }
}

static void checkParameterAsOut(
    ReachabilityContext& reachability,
    IRFunc* func,
    IRParam* param,
    DiagnosticSink* sink)
{
    auto loads = getUnresolvedParamLoads(reachability, func, param);
    for (auto load : loads)
    {
        sink->diagnose(
            load,
            as<IRTerminatorInst>(load) ? Diagnostics::returningWithUninitializedOut
                                       : Diagnostics::usingUninitializedOut,
            param);
    }
}

static void checkUninitializedValues(IRFunc* func, DiagnosticSink* sink)
{
    // Differentiable functions will generate undefined values
    // strictly so that they can be set in a differentiable way
    if (isDifferentiableFunc(func))
        return;

    auto firstBlock = func->getFirstBlock();
    if (!firstBlock)
        return;

    ReachabilityContext reachability(func);

    // Used for a further analysis and to skip usual return checks
    auto constructor = func->findDecoration<IRConstructorDecorartion>();

    // Special checks for stages e.g. raytracing shader
    Stage stage = Stage::Unknown;
    if (auto entry = func->findDecoration<IREntryPointDecoration>())
        stage = entry->getProfile().getStage();

    // Check out parameters
    if (!isUnmodifying(func))
    {
        int index = 0;
        for (auto param : firstBlock->getParams())
        {
            ParameterCheckType checkType = isPotentiallyUnintended(param, stage, index);
            if (checkType == AsOut)
                checkParameterAsOut(reachability, func, param, sink);
            index++;
        }
    }

    // Check ordinary instructions
    for (auto block : func->getBlocks())
    {
        for (auto inst = block->getFirstInst(); inst; inst = inst->getNextInst())
        {
            if (!isUninitializedValue(inst))
                continue;

            // This will be looked into later
            if (constructor && isReturnedValue(inst))
                continue;

            IRType* type = inst->getFullType();
            if (canIgnoreType(type, nullptr))
                continue;

            auto loads = getUnresolvedVariableLoads(reachability, inst);
            for (auto load : loads)
            {
                sink->diagnose(load, Diagnostics::usingUninitializedVariable, inst);
            }
        }
    }

    // Separate analysis for constructors
    checkConstructor(func, reachability, sink);
}

static void checkUninitializedGlobals(IRGlobalVar* variable, DiagnosticSink* sink)
{
    IRType* type = variable->getFullType();
    if (canIgnoreType(type, nullptr))
        return;

    // Check for semantic decorations
    // (e.g. globals like gl_GlobalInvocationID)
    if (variable->findDecoration<IRSemanticDecoration>())
        return;

    if (variable->findDecoration<IRGlobalInputDecoration>())
        return;

    if (variable->findDecoration<IRVulkanHitAttributesDecoration>())
        return;

    // Check for initialization blocks
    for (auto inst : variable->getChildren())
    {
        if (as<IRBlock>(inst))
            return;
    }

    auto addresses = getAliasableInstructions(variable);

    List<IRInst*> loads;
    for (auto alias : addresses)
    {
        for (auto use = alias->firstUse; use; use = use->nextUse)
        {
            InstructionUsageType usage = getInstructionUsageType(use->getUser(), alias);
            if (usage == Store || usage == StoreParent)
                return;

            if (usage == Load)
                loads.add(use->getUser());
        }
    }

    for (auto load : loads)
    {
        sink->diagnose(load, Diagnostics::usingUninitializedGlobalVariable, variable);
    }
}

void checkForUsingUninitializedValues(IRModule* module, DiagnosticSink* sink)
{
    for (auto inst : module->getGlobalInsts())
    {
        if (auto func = as<IRFunc>(inst))
        {
            checkUninitializedValues(func, sink);
        }
        else if (auto generic = as<IRGeneric>(inst))
        {
            auto retVal = findGenericReturnVal(generic);
            if (auto funcVal = as<IRFunc>(retVal))
                checkUninitializedValues(funcVal, sink);
        }
        else if (auto global = as<IRGlobalVar>(inst))
        {
            checkUninitializedGlobals(global, sink);
        }
    }
}
} // namespace Slang
