// slang-ir-bind-existentials.cpp
#include "slang-ir-bind-existentials.h"

#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{

// The code that comes out of the linking step will have instructions added
// that indicate how parameters with existential (interface) types are supposed
// to be specialized to concrete types.
//
// If there are any global existential-type parameters there should be a
// `bindGlobalExistentialSlots(...)` instruction at module scope.
//
// For each entry point with entry-point existential parameters, there should
// be a `[bindExistentialSlots(...)]` decoration attached to the entry
// point itself.
//
// In each case, the operands of the instruction should be a sequence of
// pairs. The number of pairs should match the number of existential "slots"
// at global or entry-point scope. Each pair should comprise a type `T`
// to plug into the slot, and a witness table `w` for the conformance of
// `T` to the interface type in that slot.
//
// In the simplest case, if we have a global shader parameter of interface
// type:
//
//      IFoo p;
//
// Then this will lower to the IR as:
//
//      global_param p : IFoo;
//
// And if the user tries to specialize `p` to type `Bar`, and a witness
// table `bar_is_ifoo`, we've have:
//
//      bindGlobalExistentialSlots(Bar, bar_is_ifoo);
//
// The goal of this pass is to replace the parameter of interface type
// with one of concrete type:
//
//      global_param p_new : Bar;
//
// and replace any reference to the old `p` parameter with
// a `makeExistential(p_new, bar_is_ifoo)`. That preserves the
// fact that a reference to `p` is conceptually of type `IFoo`,
// but allows downstream optimization passes to start specializing
// code based on the concrete knowledge that the value "backing"
// the parameter is actaully of type `Bar`.

// As is typically for IR passes, we will encapsulate all the
// logic in a `struct` type.
//
struct BindExistentialSlots
{
    IRModule* module = nullptr;
    DiagnosticSink* sink = nullptr;

    void processModule()
    {
        // We will start by dealing with the global existential slots.
        processGlobalExistentialSlots();

        // Then we will process the per-entry-point existential slots.
        processEntryPointExistentialSlots();
    }

    void processGlobalExistentialSlots()
    {
        // We will search for global shader parameters that make
        // use of existential specialization parameters.
        //
        for (auto inst : module->getGlobalInsts())
        {
            // We only care about global shader parameters.
            //
            auto globalParam = as<IRGlobalParam>(inst);
            if (!globalParam)
                continue;

            // We only care about global shader parameters
            // that have existential specialization parameters,
            // and we expect all such parameters to have a
            // `[bindExistentialSlots(...)]` decoration that
            // was added during IR linking.
            //
            auto bindSlotsInst =
                globalParam->findDecorationImpl(kIROp_BindExistentialSlotsDecoration);
            if (!bindSlotsInst)
                continue;

            replaceTypeUsingExistentialSlots(
                globalParam,
                bindSlotsInst->getOperandCount(),
                bindSlotsInst->getOperands());

            // Once we have propagated the information from
            // the `[bindExistentialSlots(...)]` decoration
            // down into the parameter's type, we no longer
            // need the decoration.
            //
            bindSlotsInst->removeAndDeallocate();
        }
    }

    void processEntryPointExistentialSlots()
    {
        // The overall flow for the entry-point case is similar
        // to the global case.
        //
        // We start by iterating over all the functions at
        // global scope and look for entry points.
        //
        for (auto inst : module->getGlobalInsts())
        {
            auto func = as<IRFunc>(inst);
            if (!func)
                continue;

            if (!func->findDecorationImpl(kIROp_EntryPointDecoration))
                continue;

            // We then process each entry point we find.
            //
            processEntryPointExistentialSlots(func);
        }
    }

    void processEntryPointExistentialSlots(IRFunc* func)
    {
        // When looking at a single `func`, we need
        // to find the `[bindExistentialSlots(...)]` decoration,
        // if it has one.
        //
        auto bindEntryPointExistentialSlotsInst =
            func->findDecorationImpl(kIROp_BindExistentialSlotsDecoration);

        // We then need to process each of the entry-point
        // parameters just like we did for global parameters.
        //
        // Because the existential slot arguments for *all* of the parameters
        // are attached in a single `[bindExistentialSlots(...)]` decoration,
        // we need to carve them up appropriately across the parameters.
        // The way we do this is a bit of a kludge, in that we track a
        // single `slotOffset` and increment it for each parameter by the
        // number of arguments it consumed.
        //
        // Note: a better approach here might rely on the layout information
        // for the parameters, which should directly encode an offset for
        // the existential specialization parameters it uses. The challenge
        // with this is that we'd need to correctly interpret the offset
        // relative to any global-scope specialization parameters or
        // generic specialization parameters of the entry point.
        // Ultimately the simplistic counter approach is less complicated.
        //
        Index slotOffset = 0;
        for (auto param : func->getParams())
        {
            processEntryPointParameter(param, bindEntryPointExistentialSlotsInst, slotOffset);
        }

        // TODO: We would need to consider what to do if
        // we had an existential return type for `func`.
        //
        // In general, it probably doesn't make sense to
        // have existential types in varying input/output
        // at all, so the front-end should probably be
        // validating that.

        // Once we've processed all the parameters, the information
        // in the `[bindExistentialSlots(...)]` decoration is
        // no longer needed, and we can remove it.
        //
        if (bindEntryPointExistentialSlotsInst)
        {
            bindEntryPointExistentialSlotsInst->removeAndDeallocate();
        }
    }

    // When processing a single parameter we need to have access
    // to the corresponding instruction that will bind its slots.
    //
    // We don't care whether we have a `global_param` and a
    // `bindGlobalExistentialSlots` instruction, or an entry-point
    // function `param` and a `[bindExistentialSlots(...)]`
    // decoration; both use the same subroutine.
    //
    void processEntryPointParameter(
        IRInst* param,
        IRInst* bindSlotsInst,
        Index& ioSlotOperandOffset)
    {
        // We expect all shader parameters to have layout information,
        // but to be defensive we will skip any that don't.
        //
        auto layoutDecoration = param->findDecoration<IRLayoutDecoration>();
        if (!layoutDecoration)
            return;
        auto varLayout = as<IRVarLayout>(layoutDecoration->getLayout());
        if (!varLayout)
            return;

        // We only care about parameters that are associated
        // with one or more existential slots.
        //
        auto resInfo = varLayout->findOffsetAttr(LayoutResourceKind::ExistentialTypeParam);
        if (!resInfo)
            return;

        // We will use the layout information on the variable to
        // find out the stating slot, and the information on
        // the type to find out the number of slots.
        //
        UInt slotCount = 0;
        if (auto typeResInfo =
                varLayout->getTypeLayout()->findSizeAttr(LayoutResourceKind::ExistentialTypeParam))
            slotCount = UInt(typeResInfo->getFiniteSize());

        // At this point we know that the parameter consumes
        // some number of slots, so it would be an error
        // if we don't have an instruction to bind the slots.
        //
        if (!bindSlotsInst)
        {
            // Note: This error is considered an internal error because
            // we should be detecting and diagnosing this problem before
            // we make it to back-end code generation.
            //
            sink->diagnose(param->sourceLoc, Diagnostics::missingExistentialBindingsForParameter);
            return;
        }

        // Each existential slot corresponds to *two* arguments
        // on the binding instruction: one for the type, and
        // another for the witness table.
        //
        // We will check to make sure we have enough operands to cover
        // this parameter.
        //
        UInt bindOperandCount = bindSlotsInst->getOperandCount();
        UInt slotOperandCount = 2 * slotCount;
        if ((ioSlotOperandOffset + slotOperandCount) > bindOperandCount)
        {
            sink->diagnose(param->sourceLoc, Diagnostics::missingExistentialBindingsForParameter);
            return;
        }
        //
        // If there are enough operands, then we will offset to
        // get to the starting point for the current parameter,
        // keeping in mind that each slot accounts for two
        // operands.
        //
        auto operandsForInst = bindSlotsInst->getOperands() + ioSlotOperandOffset;

        // Once we've found the operands that are relevent to
        // the slots used by `param`, we will defer to a routine
        // that replaces the type of `param` based on the
        // information in the slots.
        //
        replaceTypeUsingExistentialSlots(param, slotOperandCount, operandsForInst);

        ioSlotOperandOffset += slotOperandCount;
    }

    void replaceTypeUsingExistentialSlots(
        IRInst* inst,
        UInt slotOperandCount,
        IRUse const* slotArgs)
    {
        // We are going to alter the type of the
        // given `inst` based on information in
        // the `slotArgs`.

        auto fullType = inst->getFullType();

        IRBuilder builder(module);

        // Every argument that is filling an existential
        // type param/slot comprises both a type and
        // a witness table, so the total number of operands
        // is twice the number of slots we are filling.
        //
        List<IRInst*> slotOperands;
        for (UInt ii = 0; ii < slotOperandCount; ++ii)
            slotOperands.add(slotArgs[ii].get());

        // We are going to create a proxy type that represents
        // the results of plugging all the information
        // from the existential slots into the original type.
        //
        auto newType =
            builder.getBindExistentialsType(fullType, slotOperandCount, slotOperands.getBuffer());

        // We will replace the type of the original parameter
        // with the new proxy type.
        //
        builder.setDataType(inst, newType);

        // Next we want to replace all uses of `inst` (which
        // expect a value of its old type) with a fresh
        // `wrapExistential(...)` instruction that refers to
        // `inst` with its new type.
        //
        // Note: we make a copy of the list of uses for `inst`
        // before going through and replacing them, because
        // during the replacement we make *more* uses of `inst`,
        // as an operand to the `makeExistential` instructions.
        // We only want to replace the old uses, and not the
        // new ones we'll be making.
        //
        List<IRUse*> usesToReplace;
        for (auto use = inst->firstUse; use; use = use->nextUse)
        {
            auto user = use->getUser();

            // Note: We don't want to replace uses that are
            // just referring to an instruction to identify
            // it (e.g., a global shader parameter). We enumerate
            // the relevant cases here and skip them.
            //
            if (as<IRDecoration>(user))
                continue;
            if (as<IRAttr>(user))
                continue;
            if (as<IRLayout>(user))
                continue;

            usesToReplace.add(use);
        }

        // Now we can loop over our list of uses and replace each.
        //
        for (auto use : usesToReplace)
        {
            // We are going to emit a `wrapExistential` (or `makeExistential`)
            // right before each use site of the value.
            //
            builder.setInsertBefore(use->getUser());

            // The `inst` used to have an existential/interface type,
            // but will now have a concrete/bound type, and we need
            // to wrap it up again to get a value of the original
            // expected type.
            //
            auto newVal = builder.emitWrapExistential(
                fullType,
                inst,
                slotOperandCount,
                slotOperands.getBuffer());
            builder.replaceOperand(use, newVal);
        }
    }
};

void bindExistentialSlots(IRModule* module, DiagnosticSink* sink)
{
    BindExistentialSlots context;
    context.module = module;
    context.sink = sink;
    context.processModule();
}

} // namespace Slang
