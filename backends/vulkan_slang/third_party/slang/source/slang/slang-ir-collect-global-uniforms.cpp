// slang-ir-collect-global-uniforms.cpp
#include "slang-ir-collect-global-uniforms.h"

#include "slang-ir-insts.h"

namespace Slang
{

// This file implements a pass that takes input code like:
//
//      uniform int gA;
//      uniform float gB;
//
//      void main() { ... gA + gB ... }
//
// and transforms it into code like:
//
//      struct GlobalParams
//      {
//          int gA;
//          float gB;
//      }
//
//      ConstantBuffer<GlobalParams> globalParams;
//
//      void main() { ... globalParams.gA + globalParams.gB ... }
//
// The main consequence of this transformation is that we can support
// global `uniform` shader parameters of "ordinary" data types when
// compiling for targets that don't directly support that feature
// (e.g., GLSL/SPIR-V).
//
// In addition, on targets that already support a similar transformation
// (e.g., when compiling to DXBC/DXIL via fxc/dxc), making the `globalParams`
// constant buffer explicit allows us to control the binding that is
// assigned to it using our existing logic for automatic layout, rather than
// being left at the whims of the undocumented defaults of those compilers.
//
// A final consequence of this pass is that for targets where *all*
// shader parameters use "ordinary" data types (because there are no
// non-first-class types), we end up with a conveniently packaged up
// single parameter and type that encapsulates all of the shader inputs.
//
struct CollectGlobalUniformParametersContext
{
    // In orderto perform our transformation, we need access to the module
    // to be transformed, as well as the layout information representing
    // the global-scope shader parameters.
    //
    IRModule* module;
    IRVarLayout* globalScopeVarLayout;

    IRGlobalParam* _getGlobalParamFromLayoutFieldKey(IRInst* key)
    {
        switch (key->getOp())
        {
        case kIROp_GlobalParam:
            return cast<IRGlobalParam>(key);
        case kIROp_MakeExistential:
        case kIROp_WrapExistential:
            return as<IRGlobalParam>(key->getOperand(0));
        default:
            return nullptr;
        }
    }

    // This is a relatively simple pass, and it is all driven
    // by a single subroutine.
    //
    void processModule()
    {
        if (!globalScopeVarLayout)
        {
            return;
        }

        // We start by looking at the layout that was computed for the global-scope
        // parameters to determine how the parameters are supposed to be pacakged.
        //
        // This step relies on the earlier layout computation logic to have implemented
        // any target-specific policies around how the global-scope parametesr are
        // to be passed, and therefore we avoid trying to make any target-specific
        // decisions in this pass.
        //
        auto globalScopeTypeLayout = globalScopeVarLayout->getTypeLayout();
        auto globalParamsTypeLayout = globalScopeTypeLayout;

        // One example of a difference that might appear in the global-scope layout
        // depending on the target is that the global-scope parameters might end
        // up just pacakged as a `struct`, *or* they might be packaged up in a
        // `ConstantBuffer<...>` or other parameter group that wraps that `struct`.
        //
        IRParameterGroupTypeLayout* globalParameterGroupTypeLayout =
            as<IRParameterGroupTypeLayout>(globalParamsTypeLayout);
        if (globalParameterGroupTypeLayout)
        {
            // In the case where there is a wrapping `ConstantBuffer<...>`, we want to
            // get at the element type of that constant buffer, because it represents
            // the packaged-up `struct` that we want to produce.
            //
            globalParamsTypeLayout =
                globalParameterGroupTypeLayout->getElementVarLayout()->getTypeLayout();
        }

        // As a special case (in order to avoid disruption to any downstream passes),
        // if the layout for the global-scope parameters doesn't include any "ordinary"
        // data (represented as `LayoutResourceKind::Uniform`), then we will not do
        // the "packaging up" step at all.
        //
        // This means that the current pass will not change the results for a majority
        // of targets (notably, all the current graphics APIs) *unless* global shader
        // parameters are declared that use "ordinary' data.
        //
        // TODO: eventually we should remove this special case, and confirm that the resulting
        // logic works across all shaders (it should). Doing so will be a necessary
        // step if want to start applying the packaging-up of global-scope parameters on
        // a per-module basis.
        //
        if (!globalParameterGroupTypeLayout &&
            !globalParamsTypeLayout->findSizeAttr(LayoutResourceKind::Uniform))
            return;

        // We expect that the layout for the global-scope parameters is always
        // computed for a `struct` type.
        //
        auto globalParamsStructTypeLayout = as<IRStructTypeLayout>(globalParamsTypeLayout);
        SLANG_ASSERT(globalParamsStructTypeLayout);

        // We need to construct a single IR parameter that will represent
        // the collected global-scope parameters. The `IRBuilder` we construct
        // for this will also be used when replacing the individual parameters.
        //

        IRBuilder builderStorage(module);
        IRBuilder* builder = &builderStorage;
        builder->setInsertInto(module->getModuleInst());

        // The packaged-up global parameters will be turned into fields of
        // a new global IR `struct` type, which we give a name of `GlobalParams`
        // so that it is identifiable in the output.
        //
        // Note: the equivalent in fxc/dxc is the `$Globals` constant buffer.
        //
        auto wrapperStructType = builder->createStructType();
        builder->addNameHintDecoration(
            wrapperStructType,
            UnownedTerminatedStringSlice("GlobalParams"));
        builder->addBinaryInterfaceTypeDecoration(wrapperStructType);

        // If the computed layout used a bare `struct` type, then we will use
        // our `GlobalParams` struct as-is, but if the layout involved an
        // implicitly defined `ConstantBuffer<...>`, this is where we construct
        // the type `ConstantBuffer<GlobalParams>`.
        //
        IRType* wrapperParamType = wrapperStructType;
        if (globalParameterGroupTypeLayout)
        {
            auto wrapperParamGroupType = builder->getConstantBufferType(
                wrapperStructType,
                builder->getType(kIROp_DefaultBufferLayoutType));
            wrapperParamType = wrapperParamGroupType;
        }

        // Now that we've determined what the type of the new single global parameter
        // should be, we can go ahead and emit it into the IR module.
        //
        // We will call the implicit parameter for all the globals `globalParams`.
        //
        IRGlobalParam* wrapperParam = builder->createGlobalParam(wrapperParamType);
        builder->addLayoutDecoration(wrapperParam, globalScopeVarLayout);
        builder->addNameHintDecoration(wrapperParam, UnownedTerminatedStringSlice("globalParams"));

        // With the setup work out of the way, we can iterate over the global
        // parameters that were present in the layout information (they are
        // represented as the fields of the global-scope `struct` layout).
        //
        for (auto fieldLayoutAttr : globalParamsStructTypeLayout->getFieldLayoutAttrs())
        {
            // We expect the IR layout pass to have encoded field per-field
            // layout so that the "key" for the field is the corresponding
            // global shader parameter.

            // Save the original global param before replacement.
            auto globalParam = _getGlobalParamFromLayoutFieldKey(fieldLayoutAttr->getFieldKey());

            auto globalParamLayout = fieldLayoutAttr->getLayout();

            // Set insert position to a valid instruction under the global parent scope so we can
            // create struct keys.
            builder->setInsertAfter(fieldLayoutAttr->getFieldKey());

            // This global parameter needs to be turned into a field of the global
            // parameter structure type, and that field will need a key.
            //
            auto fieldKey = builder->createStructKey();

            // In order to make sure that the existing IR layout information for
            // the global scope remains valid, we will swap out the key in the
            // per-field layout information to reference the key we created
            // instead of the existing parameter (which we will be removing).
            //
            fieldLayoutAttr = as<IRStructFieldLayoutAttr>(
                builder->replaceOperand(fieldLayoutAttr->getOperands(), fieldKey));

            // If the given parameter doesn't contribute to uniform/ordinary usage, then
            // we can safely leave it at the global scope and potentially avoid a lot
            // of complications that might otherwise arise (that is, we don't need to worry
            // about downstream passes that might have worked for a simple global parameter,
            // but that would not work for one nested inside a structure.
            //
            // TODO: It would be more consistent and robust to *always* wrap up
            // these global parameters appropriately, and ensure that all the downstream
            // passes can handle that case, since they would need to do so in general.
            //
            if (!globalParamLayout->getTypeLayout()->findSizeAttr(LayoutResourceKind::Uniform))
                continue;

            SLANG_ASSERT(globalParam);

            // The new structure field will need to have whatever decorations
            // had been put on the global parameter (notably including any name hint)
            //
            globalParam->transferDecorationsTo(fieldKey);

            // Now we can add a field to the `GlobalParams` type that
            // will stand in for the parameter: it will have the key we
            // just generated, and the type of the original parameter.
            //
            auto globalParamType = globalParam->getFullType();
            builder->createStructField(wrapperStructType, fieldKey, globalParamType);

            // Next we need to replace the uses of the parameter will
            // logic to extract the appropriate field from the aggregated
            // parameter.
            //
            // Unlike trivial cases that can work with `replaceAllUsesWith`,
            // we are going to need to different code for each use, and that
            // potentially puts us in the bad case of modifying the use-def
            // information while also iterating it.
            //
            // To worka around the problem, we will make a copy of the list of
            // uses and work with that instead.
            //
            List<IRUse*> uses;
            for (auto use = globalParam->firstUse; use; use = use->nextUse)
            {
                uses.add(use);
            }
            for (auto use : uses)
            {
                auto user = use->user;

                // There is an annoying gotcha here, in that we are using
                // global shader parameters themselves (the `IRGlobalParam`s)
                // to represent their "keys" in the layout objects that
                // represent the layout of the global scope.
                //
                // We don't want to replace the reference to the global
                // parameter in one of these layouts with a reference
                // to a field of our new collected parameter, and instead
                // want to replace such a reference with the key for that
                // field.
                //
                // TODO: We should probably be assigning keys to global
                // parameters, and using those keys in the layout instructions
                // instead of directly using the parameters. The parameters
                // could then have a decoration to assocaite them with their
                // key.
                //
                // TODO: Alternatively, we could considering doing this
                // kind of collection work earlier, on a per-module basis,
                // so that we don't need to perform collection as a back-end step.
                // (Note that the main sticking point there is explicit layout
                // markers on global parameters, that stop the entire parameter
                // range for a module from being contiguous).
                //
                if (auto layoutAttr = as<IRStructFieldLayoutAttr>(user))
                {
                    builder->replaceOperand(layoutAttr->getOperands(), fieldKey);
                    continue;
                }

                // NumThreadsDecoration may sometimes be the user for a global
                // parameter. This occurs when the parameter was supposed to be
                // a specialization constant, but isn't due to that not being
                // supported for the target. These can be skipped here and
                // diagnosed later.
                if (as<IRNumThreadsDecoration>(user))
                {
                    continue;
                }

                // For each use site for the global parameter, we will
                // insert new code right before the instruction that uses
                // the parameter.
                //
                // TODO: In some cases we might want to emit a single load of
                // a global parameter at the start of a function, rather
                // than individual loads at multiple points in the body
                // of a function. Ideally we can/should annotate the
                // `globalParams` parameter with the equivalent of `__restrict__`
                // so that these loads can be merged/moved without concern
                // for aliasing.
                //
                builder->setInsertBefore(user);

                IRInst* value = nullptr;
                if (globalParameterGroupTypeLayout)
                {
                    // If the global parameters are being placed in a
                    // `ConstantBuffer<GlobalParams>`, then we need to
                    // emit an instruction to compute a pointer to the
                    // desired field, and then load from it.
                    //
                    auto ptrType = builder->getPtrType(globalParamType);
                    auto fieldAddr = builder->emitFieldAddress(ptrType, wrapperParam, fieldKey);
                    value = builder->emitLoad(globalParamType, fieldAddr);
                }
                else
                {
                    // If the global parameters are being bundled in a
                    // plain old `struct`, then we simple want to emit
                    // an instruction to extract the desired field.
                    //
                    value = builder->emitFieldExtract(globalParamType, wrapperParam, fieldKey);
                }

                // Whatever replacement value we computed, we need
                // to install it as the value to be used at the use site.
                //
                use->set(value);
            }

            // Once we've replaced all uses of the global parameter,
            // we can remove it from the IR module completely.
            //
            globalParam->removeAndDeallocate();
        }
    }
};

void collectGlobalUniformParameters(IRModule* module, IRVarLayout* globalScopeVarLayout)
{
    CollectGlobalUniformParametersContext context;
    context.module = module;
    context.globalScopeVarLayout = globalScopeVarLayout;

    context.processModule();
}

} // namespace Slang
