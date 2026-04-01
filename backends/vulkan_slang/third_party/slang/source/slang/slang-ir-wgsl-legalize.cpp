#include "slang-ir-wgsl-legalize.h"

#include "slang-ir-insts.h"
#include "slang-ir-legalize-binary-operator.h"
#include "slang-ir-legalize-global-values.h"
#include "slang-ir-legalize-varying-params.h"
#include "slang-ir.h"

namespace Slang
{

static void legalizeCall(IRCall* call)
{
    // WGSL does not allow forming a pointer to a sub part of a composite value.
    // For example, if we have
    // ```
    // struct S { float x; float y; };
    // void foo(inout float v) { v = 1.0f; }
    // void main() { S s; foo(s.x); }
    // ```
    // The call to `foo(s.x)` is illegal in WGSL because `s.x` is a sub part of `s`.
    // And trying to form `&s.x` in WGSL is illegal.
    // To work around this, we will create a local variable to hold the sub part of
    // the composite value.
    // And then pass the local variable to the function.
    // After the call, we will write back the local variable to the sub part of the
    // composite value.
    //
    IRBuilder builder(call);
    builder.setInsertBefore(call);
    struct WritebackPair
    {
        IRInst* dest;
        IRInst* value;
    };
    ShortList<WritebackPair> pendingWritebacks;

    for (UInt i = 0; i < call->getArgCount(); i++)
    {
        auto arg = call->getArg(i);
        auto ptrType = as<IRPtrTypeBase>(arg->getDataType());
        if (!ptrType)
            continue;
        switch (arg->getOp())
        {
        case kIROp_Var:
        case kIROp_Param:
        case kIROp_GlobalParam:
        case kIROp_GlobalVar:
            continue;
        default:
            break;
        }

        // Create a local variable to hold the input argument.
        auto var = builder.emitVar(ptrType->getValueType(), AddressSpace::Function);

        // Store the input argument into the local variable.
        builder.emitStore(var, builder.emitLoad(arg));
        builder.replaceOperand(call->getArgs() + i, var);
        pendingWritebacks.add({arg, var});
    }

    // Perform writebacks after the call.
    builder.setInsertAfter(call);
    for (auto& pair : pendingWritebacks)
    {
        builder.emitStore(pair.dest, builder.emitLoad(pair.value));
    }
}

static void legalizeFunc(IRFunc* func)
{
    // Insert casts to convert integer return types
    auto funcReturnType = func->getResultType();
    if (isIntegralType(funcReturnType))
    {
        for (auto block : func->getBlocks())
        {
            if (auto returnInst = as<IRReturn>(block->getTerminator()))
            {
                auto returnedValue = returnInst->getOperand(0);
                auto returnedValueType = returnedValue->getDataType();
                if (isIntegralType(returnedValueType))
                {
                    IRBuilder builder(returnInst);
                    builder.setInsertBefore(returnInst);
                    auto newOp = builder.emitCast(funcReturnType, returnedValue);
                    builder.replaceOperand(returnInst->getOperands(), newOp);
                }
            }
        }
    }
}

static void legalizeSwitch(IRSwitch* switchInst)
{
    // WGSL Requires all switch statements to contain a default case.
    // If the switch statement does not contain a default case, we will add one.
    if (switchInst->getDefaultLabel() != switchInst->getBreakLabel())
        return;
    IRBuilder builder(switchInst);
    auto defaultBlock = builder.createBlock();
    builder.setInsertInto(defaultBlock);
    builder.emitBranch(switchInst->getBreakLabel());
    defaultBlock->insertBefore(switchInst->getBreakLabel());
    List<IRInst*> cases;
    for (UInt i = 0; i < switchInst->getCaseCount(); i++)
    {
        cases.add(switchInst->getCaseValue(i));
        cases.add(switchInst->getCaseLabel(i));
    }
    builder.setInsertBefore(switchInst);
    auto newSwitch = builder.emitSwitch(
        switchInst->getCondition(),
        switchInst->getBreakLabel(),
        defaultBlock,
        (UInt)cases.getCount(),
        cases.getBuffer());
    switchInst->transferDecorationsTo(newSwitch);
    switchInst->removeAndDeallocate();
}

static void processInst(IRInst* inst, DiagnosticSink* sink)
{
    switch (inst->getOp())
    {
    case kIROp_Call:
        legalizeCall(static_cast<IRCall*>(inst));
        break;

    case kIROp_Switch:
        legalizeSwitch(as<IRSwitch>(inst));
        break;

    // For all binary operators, make sure both side of the operator have the same type
    // (vector-ness and matrix-ness).
    case kIROp_Add:
    case kIROp_Sub:
    case kIROp_Mul:
    case kIROp_Div:
    case kIROp_FRem:
    case kIROp_IRem:
    case kIROp_And:
    case kIROp_Or:
    case kIROp_BitAnd:
    case kIROp_BitOr:
    case kIROp_BitXor:
    case kIROp_Lsh:
    case kIROp_Rsh:
    case kIROp_Eql:
    case kIROp_Neq:
    case kIROp_Greater:
    case kIROp_Less:
    case kIROp_Geq:
    case kIROp_Leq:
        legalizeBinaryOp(inst, sink, CodeGenTarget::WGSL);
        break;

    case kIROp_Func:
        legalizeFunc(static_cast<IRFunc*>(inst));
        [[fallthrough]];
    default:
        for (auto child : inst->getModifiableChildren())
        {
            processInst(child, sink);
        }
    }
}

struct GlobalInstInliningContext : public GlobalInstInliningContextGeneric
{
    bool isLegalGlobalInstForTarget(IRInst* /* inst */) override
    {
        // The global instructions that are generically considered legal are fine for
        // WGSL.
        return false;
    }

    bool isInlinableGlobalInstForTarget(IRInst* /* inst */) override
    {
        // The global instructions that are generically considered inlineable are fine
        // for WGSL.
        return false;
    }

    bool shouldBeInlinedForTarget(IRInst* /* user */) override
    {
        // WGSL doesn't do any extra inlining beyond what is generically done by default.
        return false;
    }

    IRInst* getOutsideASM(IRInst* beforeInst) override
    {
        // Not needed for WGSL, check e.g. the SPIR-V case to see why this is used.
        return beforeInst;
    }
};

void legalizeIRForWGSL(IRModule* module, DiagnosticSink* sink)
{
    List<EntryPointInfo> entryPoints;
    for (auto inst : module->getGlobalInsts())
    {
        IRFunc* const func{as<IRFunc>(inst)};
        if (!func)
            continue;
        IREntryPointDecoration* const entryPointDecor =
            func->findDecoration<IREntryPointDecoration>();
        if (!entryPointDecor)
            continue;
        EntryPointInfo info;
        info.entryPointDecor = entryPointDecor;
        info.entryPointFunc = func;
        entryPoints.add(info);
    }

    legalizeEntryPointVaryingParamsForWGSL(module, sink, entryPoints);

    // Go through every instruction in the module and legalize them as needed.
    processInst(module->getModuleInst(), sink);

    // Some global insts are illegal, e.g. function calls.
    // We need to inline and remove those.
    GlobalInstInliningContext().inlineGlobalValuesAndRemoveIfUnused(module);
}

} // namespace Slang
