// slang-ir-util.h
#ifndef SLANG_IR_UTIL_H_INCLUDED
#define SLANG_IR_UTIL_H_INCLUDED

// This file contains utility functions for operating with Slang IR.
//
#include "slang-ir-insts.h"
#include "slang-ir.h"
namespace Slang
{
struct GenericChildrenMigrationContextImpl;
struct IRCloneEnv;

// A helper class to clone children insts to a different generic parent that has equivalent set of
// generic parameters. The clone will take care of substitution of equivalent generic parameters and
// intermediate values between the two generic parents.
struct GenericChildrenMigrationContext : public RefObject
{
private:
    GenericChildrenMigrationContextImpl* impl;

public:
    IRCloneEnv* getCloneEnv();

    GenericChildrenMigrationContext();
    ~GenericChildrenMigrationContext();

    void init(IRGeneric* genericSrc, IRGeneric* genericDst, IRInst* insertBefore);

    IRInst* deduplicate(IRInst* value);

    IRInst* cloneInst(IRBuilder* builder, IRInst* src);
};


struct DeduplicateContext
{
    Dictionary<IRInstKey, IRInst*> deduplicateMap;

    template<typename TFunc>
    IRInst* deduplicate(IRInst* value, const TFunc& shouldDeduplicate)
    {
        if (!value)
            return nullptr;
        if (!shouldDeduplicate(value))
            return value;
        IRInstKey key = {value};
        if (auto newValue = deduplicateMap.tryGetValue(key))
            return *newValue;
        for (UInt i = 0; i < value->getOperandCount(); i++)
        {
            auto deduplicatedOperand = deduplicate(value->getOperand(i), shouldDeduplicate);
            if (deduplicatedOperand != value->getOperand(i))
                value->unsafeSetOperand(i, deduplicatedOperand);
        }
        if (auto newValue = deduplicateMap.tryGetValue(key))
            return *newValue;
        deduplicateMap[key] = value;
        return value;
    }
};

bool isPtrToClassType(IRInst* type);

bool isPtrToArrayType(IRInst* type);

// True if ptrType is a pointer type to elementType
bool isPointerOfType(IRInst* ptrType, IRInst* elementType);

// True if ptrType is a pointer type to a type of opCode
bool isPointerOfType(IRInst* ptrType, IROp opCode);

// Builds a dictionary that maps from requirement key to requirement value for `interfaceType`.
Dictionary<IRInst*, IRInst*> buildInterfaceRequirementDict(IRInterfaceType* interfaceType);

bool isComInterfaceType(IRType* type);

// If `type` is a vector, returns its element type. Otherwise, return `type`.
IRType* getVectorElementType(IRType* type);

// If `type` is a vector or a coop matrix, returns its element type. Otherwise, return `type`.
IRType* getVectorOrCoopMatrixElementType(IRType* type);

// If `type` is a matrix, returns its element type. Otherwise, return `type`.
IRType* getMatrixElementType(IRType* type);

// True if type is a resource backing memory
bool isResourceType(IRType* type);
bool isOpaqueType(IRType* type, IRType** outLeafOpaqueHandleType);

// True if type is a pointer to a resource
bool isPointerToResourceType(IRType* type);

IROp getTypeStyle(IROp op);
IROp getTypeStyle(BaseType op);

inline bool isScalarIntegerType(IRType* type)
{
    return getTypeStyle(type->getOp()) == kIROp_IntType;
}

// No side effect can take place through a value of a "Value" type.
bool isValueType(IRInst* type);

bool isScalarOrVectorType(IRInst* type);

bool isSimpleDataType(IRType* type);

bool isSimpleHLSLDataType(IRInst* inst);

bool isWrapperType(IRInst* inst);

SourceLoc findFirstUseLoc(IRInst* inst);

inline bool isChildInstOf(IRInst* inst, IRInst* parent)
{
    while (inst)
    {
        if (inst == parent)
            return true;
        inst = inst->getParent();
    }
    return false;
}

// Specialize `genericToSpecialize` with the generic parameters defined in `userGeneric`.
// For example:
// ```
// int f<T>(T a);
// ```
// will be extended into
// ```
// struct IntermediateFor_f<T> { T t0; }
// int f_primal<T>(T a, IntermediateFor_f<T> imm);
// ```
// Given a user generic `f_primal<T>` and a used value parameterized on the same set of generic
// parameters `IntermediateFor_f`, `genericToSpecialize` constructs `IntermediateFor_f<T>` (using
// the parameter list from user generic).
//
IRInst* specializeWithGeneric(
    IRBuilder& builder,
    IRInst* genericToSpecialize,
    IRGeneric* userGeneric);

IRInst* maybeSpecializeWithGeneric(
    IRBuilder& builder,
    IRInst* genericToSpecailize,
    IRInst* userGeneric);

// For a value inside a generic, create a standalone generic wrapping just the value, and replace
// the use of the original value with a specialization of the new generic using the current generic
// arguments if `replaceExistingValue` is true. For example, if we have
// ```
//     generic G { param T; v = x(T); f = y(v); return f; }
// ```
// hoistValueFromGeneric(G, v) turns the code into:
// ```
//     generic G1 { param T1; v1 = x(T); return v1; }
//     generic G { param T; v = specialize(G1, T); f = y(v); return f; }
// ```
// This function returns newly created generic inst.
// if `value` is not inside any generic, this function makes no change to IR, and returns `value`.
IRInst* hoistValueFromGeneric(
    IRBuilder& builder,
    IRInst* value,
    IRInst*& outSpecializedVal,
    bool replaceExistingValue = false);

// Clear dest and move all chidlren from src to dest.
void moveInstChildren(IRInst* dest, IRInst* src);

inline bool isGenericParam(IRInst* param)
{
    auto parent = param->getParent();
    if (auto block = as<IRBlock>(parent))
        parent = block->getParent();
    if (as<IRGeneric>(parent))
        return true;
    return false;
}

inline IRInst* unwrapAttributedType(IRInst* type)
{
    for (;;)
    {
        if (auto attrType = as<IRAttributedType>(type))
            type = attrType->getBaseType();
        else if (auto rateType = as<IRRateQualifiedType>(type))
            type = rateType->getValueType();
        else
            return type;
    }
}

// Remove hlsl's 'unorm' and 'snorm' modifiers
IRType* dropNormAttributes(IRType* const t);

void getTypeNameHint(StringBuilder& sb, IRInst* type);
void copyNameHintAndDebugDecorations(IRInst* dest, IRInst* src);
IRInst* getRootAddr(IRInst* addrInst);
IRInst* getRootAddr(
    IRInst* addrInst,
    List<IRInst*>& outAccessChain,
    List<IRInst*>* outTypes = nullptr);

bool canAddressesPotentiallyAlias(IRGlobalValueWithCode* func, IRInst* addr1, IRInst* addr2);

String dumpIRToString(
    IRInst* root,
    IRDumpOptions options = {IRDumpOptions::Mode::Simplified, IRDumpOptions::Flag::DumpDebugIds});

// Returns whether a call insts can be treated as a pure functional inst, and thus can be
// DCE'd and deduplicated.
// (no writes to memory, no reads from unknown memory, no side effects).
bool isPureFunctionalCall(
    IRCall* callInst,
    SideEffectAnalysisOptions options = SideEffectAnalysisOptions::None);

// Returns whether a call insts can be treated as a pure functional inst, and thus can be
// DCE'd (but not necessarily deduplicated).
// (no side effects).
bool isSideEffectFreeFunctionalCall(
    IRCall* call,
    SideEffectAnalysisOptions options = SideEffectAnalysisOptions::None);

bool doesCalleeHaveSideEffect(IRInst* callee);

bool isPtrLikeOrHandleType(IRInst* type);

bool canInstHaveSideEffectAtAddress(IRGlobalValueWithCode* func, IRInst* inst, IRInst* addr);

IRInst* getUndefInst(IRBuilder builder, IRModule* module);

// The the equivalent op of (a op b) in (b op' a). For example, a > b is equivalent to b < a. So (<)
// ==> (>).
IROp getSwapSideComparisonOp(IROp op);

// Set IRBuilder to insert before `inst`. If `inst` is a param, it will insert after the last param.
void setInsertBeforeOrdinaryInst(IRBuilder* builder, IRInst* inst);

// Set IRBuilder to insert after `inst`. If `inst` is a param, it will insert after the last param.
void setInsertAfterOrdinaryInst(IRBuilder* builder, IRInst* inst);

// Emit a loop structure with a simple incrementing counter.
// Returns the loop counter `IRParam`.
IRInst* emitLoopBlocks(
    IRBuilder* builder,
    IRInst* initVal,
    IRInst* finalVal,
    IRBlock*& loopBodyBlock,
    IRBlock*& loopBreakBlock);

void sortBlocksInFunc(IRGlobalValueWithCode* func);

// Remove all linkage decorations from func.
void removeLinkageDecorations(IRGlobalValueWithCode* func);

IRInst* findInterfaceRequirement(IRInterfaceType* type, IRInst* key);

IRInst* findWitnessTableEntry(IRWitnessTable* table, IRInst* key);

IRInst* getVulkanPayloadLocation(IRInst* payloadGlobalVar);

IRInst* getInstInBlock(IRInst* inst);

void removePhiArgs(IRInst* phiParam);

ShortList<IRInst*> getPhiArgs(IRInst* phiParam);

int getParamIndexInBlock(IRParam* paramInst);

bool isGlobalOrUnknownMutableAddress(IRGlobalValueWithCode* parentFunc, IRInst* inst);

bool isZero(IRInst* inst);

bool isOne(IRInst* inst);

// Casts inst to IRPtrTypeBase, excluding UserPointer address space.
IRPtrTypeBase* asRelevantPtrType(IRInst* inst);

// Returns the pointer type if it is pointer type that is not a const ref or a user pointer.
IRPtrTypeBase* isMutablePointerType(IRInst* inst);

void initializeScratchData(IRInst* inst);
void resetScratchDataBit(IRInst* inst, int bitIndex);
///
/// IRBlock related common helper methods
///
void moveParams(IRBlock* dest, IRBlock* src);

List<IRBlock*> collectBlocksInRegion(IRDominatorTree* dom, IRLoop* loop);

List<IRBlock*> collectBlocksInRegion(IRDominatorTree* dom, IRSwitch* switchInst);

List<IRBlock*> collectBlocksInRegion(
    IRDominatorTree* dom,
    IRSwitch* switchInst,
    bool* outHasMultilevelBreaks);

List<IRBlock*> collectBlocksInRegion(
    IRDominatorTree* dom,
    IRLoop* loop,
    bool* outHasMultilevelBreaks);

List<IRBlock*> collectBlocksInRegion(
    IRDominatorTree* dom,
    IRBlock* breakBlock,
    IRBlock* firstBlock,
    bool includeFirstBlock,
    bool* outHasMultilevelBreaks);

List<IRBlock*> collectBlocksInRegion(
    IRGlobalValueWithCode* func,
    IRLoop* loopInst,
    bool* outHasMultilevelBreaks);

List<IRBlock*> collectBlocksInRegion(IRGlobalValueWithCode* func, IRLoop* loopInst);

HashSet<IRBlock*> getParentBreakBlockSet(IRDominatorTree* dom, IRBlock* block);

IRBlock* getBlock(IRInst* inst);

///
/// End of IRBlock utility methods
///

IRVarLayout* findVarLayout(IRInst* value);

UnownedStringSlice getBuiltinFuncName(IRInst* callee);

// Run an operation over every block in a module
template<typename F>
static void overAllBlocks(IRModule* module, F f)
{
    for (auto globalInst : module->getGlobalInsts())
    {
        if (auto func = as<IRGlobalValueWithCode>(globalInst))
        {
            for (auto block : func->getBlocks())
            {
                f(block);
            }
        }
    }
}

void hoistInstOutOfASMBlocks(IRBlock* block);

inline bool isCompositeType(IRType* type)
{
    switch (type->getOp())
    {
    case kIROp_StructType:
    case kIROp_ArrayType:
    case kIROp_UnsizedArrayType:
        return true;
    default:
        return false;
    }
}

IRType* getSPIRVSampledElementType(IRInst* sampledType);

IRType* replaceVectorElementType(IRType* originalVectorType, IRType* t);

IRParam* getParamAt(IRBlock* block, UIndex ii);

void verifyComputeDerivativeGroupModifiers(
    DiagnosticSink* sink,
    SourceLoc errorLoc,
    bool quadAttr,
    bool linearAttr,
    IRNumThreadsDecoration* numThreadsDecor);


inline bool isSPIRV(CodeGenTarget codeGenTarget)
{
    return codeGenTarget == CodeGenTarget::SPIRV || codeGenTarget == CodeGenTarget::SPIRVAssembly;
}

int getIRVectorElementSize(IRType* type);
IRType* getIRVectorBaseType(IRType* type);

Int getSpecializationConstantId(IRGlobalParam* param);

void legalizeDefUse(IRGlobalValueWithCode* func);

UnownedStringSlice getMangledName(IRInst* inst);

bool isFirstBlock(IRInst* inst);
} // namespace Slang

#endif
