// slang-ir-optix-entry-point-uniforms.cpp

// Note: A significant portion of this code is taken and modified from
// slang-ir-entry-point-uniforms.cpp

#include "slang-ir-optix-entry-point-uniforms.h"

#include "slang-ir-entry-point-pass.h"
#include "slang-ir-insts.h"
#include "slang-ir-restructure.h"
#include "slang-ir.h"

namespace Slang
{

struct CollectOptixEntryPointUniformParams : PerEntryPointPass
{

    // *If* the entry point has any uniform parameter then we want to create a
    // structure type to house them, and then replace the shader parameter
    // references with an SBT record access.

    // We only want to create these if actually needed, so we will declare
    // them here and then initialize them on-demand.
    IRStructType* paramStructType = nullptr;
    IRParam* collectedParam = nullptr;
    IRVarLayout* entryPointParamsLayout = nullptr;

    void processEntryPointImpl(EntryPointInfo const& info) SLANG_OVERRIDE
    {
        auto entryPointFunc = info.func;
        auto entryPointDecoration = info.decoration;

        // This pass object may be used across multiple entry points,
        // so we need to make sure to reset state that could have been
        // left over from a previous entry point.
        //
        paramStructType = nullptr;
        collectedParam = nullptr;

        // We only want to process entry points that are used in OptiX/ray-tracing
        // stages, and not ordinary compute entry points (the entry-point `uniform`
        // parameters of an ordinary compute entry point will translate to CUDA
        // launch parameters).
        //
        switch (entryPointDecoration->getProfile().getStage())
        {
        default:
            break;

        case Stage::Compute:
            return;
        }

        // We expect all entry points to have explicit layout information attached.
        //
        // We will assert that we have the information we need, but try to be
        // defensive and bail out in the failure case in release builds.
        //
        auto funcLayoutDecoration = entryPointFunc->findDecoration<IRLayoutDecoration>();
        SLANG_ASSERT(funcLayoutDecoration);
        if (!funcLayoutDecoration)
            return;

        auto entryPointLayout = as<IREntryPointLayout>(funcLayoutDecoration->getLayout());
        SLANG_ASSERT(entryPointLayout);
        if (!entryPointLayout)
            return;

        // The parameter layout for an entry point will either be a structure
        // type layout, or a constant buffer (a case of parameter group)
        // wrapped around such a structure.
        //
        // If we are in the latter case we will need to make sure to allocate
        // an explicit IR constant buffer for that wrapper,
        //
        // TODO: Reconcile the above with CUDA / OptiX...
        entryPointParamsLayout = entryPointLayout->getParamsLayout();
        auto entryPointParamsStructLayout = getScopeStructLayout(entryPointLayout);

        // We will set up an IR builder so that we are ready to generate code.
        //
        IRBuilder builderStorage(m_module);
        auto builder = &builderStorage;

        // We will be removing any uniform parameters we run into, so we
        // need to iterate the parameter list carefully to deal with
        // us modifying it along the way.
        //
        IRParam* nextParam = nullptr;
        UInt paramCounter = 0;
        for (IRParam* param = entryPointFunc->getFirstParam(); param; param = nextParam)
        {
            nextParam = param->getNextParam();
            UInt paramIndex = paramCounter++;

            // We expect all entry-point parameters to have layout information,
            // but we will be defensive and skip parameters without the required
            // information when we are in a release build.
            //
            auto layoutDecoration = param->findDecoration<IRLayoutDecoration>();
            SLANG_ASSERT(layoutDecoration);
            if (!layoutDecoration)
                continue;
            auto paramLayout = as<IRVarLayout>(layoutDecoration->getLayout());
            SLANG_ASSERT(paramLayout);
            if (!paramLayout)
                continue;

            // A parameter that has varying input/output behavior should be left alone,
            // since this pass is only supposed to apply to uniform (non-varying)
            // parameters.
            //
            // In the case of optix, these varyings come in the form of ray payload
            // and hit attributes
            //
            if (isVaryingParameter(paramLayout))
                continue;

            // At this point we know that `param` is not a varying shader parameter,
            // so we'll treat it as part of the SBT record.
            //
            // If this is the first parameter we are running into, then we need
            // to deal with creating the structure type and global shader
            // parameter that our transformed entry point will use.
            //
            ensureCollectedParamAndTypeHaveBeenCreated();

            // Now that we've ensured the global `struct` type and collected shader parameter
            // exist, we need to add a field to the `struct` to represent the
            // current parameter.
            //

            auto paramType = param->getFullType();

            builder->setInsertBefore(paramStructType);
            // We need to know the "key" that should be used for the parameter,
            // so we will read it off of the entry-point layout information.
            //
            // TODO: Maybe we should associate the key to the parameter via
            // a decoration to avoid this indirection?
            //
            // TODO: Alternatively, we should make this pass responsible for
            // dealing with the transfer of layout information from the entry
            // point to its parameters, rather than baking that behavior into
            // the linker. After all, this pass is traversing the same information
            // anyway, so it could do the work while it is here...
            //
            auto paramFieldKey = cast<IRStructKey>(
                entryPointParamsStructLayout->getFieldLayoutAttrs()[paramIndex]->getFieldKey());

            auto paramField = builder->createStructField(paramStructType, paramFieldKey, paramType);
            SLANG_UNUSED(paramField);

            // We will transfer all decorations on the parameter over to the key
            // so that they can affect downstream emit logic.
            //
            // TODO: We should double-check whether any of the decorations should
            // be moved to the *field* instead.
            //
            param->transferDecorationsTo(paramFieldKey);

            // At this point we want to eliminate the original entry point
            // parameter, in favor of the `struct` field we declared.
            // That requires replacing any uses of the parameter with
            // appropriate code to pull out the field.
            //
            // We *could* extract the field at the start of the shader
            // and then do a `replaceAllUsesWith` to propragate it
            // down, but in practice we expect that it is better for
            // performance to "rematerialize" the value of a shader
            // parameter as close to where it is used as possible.
            //
            // We are therefore going to replace the uses one at a time.
            //
            while (auto use = param->firstUse)
            {
                // Given a `use` of the paramter, we will insert
                // the replacement code right before the instruction
                // that is doing the using.
                //
                builder->setInsertBefore(use->getUser());

                // The way to extract the field that corresponds
                // to the parameter depends on whether or not
                // we generated a constant buffer.
                //
                IRInst* fieldVal = nullptr;

                // A constant buffer behaves like a pointer
                // at the IR level, so we first do a pointer
                // offset operation to compute what amounts
                // to `&cb->field`, and then load from that address.
                //
                auto fieldAddress = builder->emitFieldAddress(
                    builder->getPtrType(paramType),
                    collectedParam,
                    paramFieldKey);
                fieldVal = builder->emitLoad(fieldAddress);

                // We replace the value used at this use site, which
                // will have a side effect of making `use` no longer
                // be on the list of uses for `param`, so that when
                // we get back to the top of the loop the list of
                // uses will be shorter.
                //
                use->set(fieldVal);
            }

            // Once we've replaced all the uses of `param`, we
            // can go ahead and remove it completely.
            //
            param->removeAndDeallocate();
        }

        if (collectedParam)
        {
            collectedParam->insertBefore(entryPointFunc->getFirstBlock()->getFirstChild());
        }
        else
        {
            // If we didn't find a uniform parameter, we can safely return now.
            return;
        }

        // Now, replace the collected parameter with OptiX SBT accesses.
        auto paramType = collectedParam->getFullType();
        builder->setInsertBefore(entryPointFunc->getFirstBlock()->getFirstOrdinaryInst());
        IRInst* getAttr =
            builder->emitIntrinsicInst(paramType, kIROp_GetOptiXSbtDataPtr, 0, nullptr);
        collectedParam->replaceUsesWith(getAttr);
        collectedParam->removeAndDeallocate();
        fixUpFuncType(entryPointFunc);
    }

    void ensureCollectedParamAndTypeHaveBeenCreated()
    {
        if (paramStructType)
            return;

        IRBuilder builder(m_module);

        // First we create the structure to hold the parameters.
        //
        builder.setInsertBefore(m_entryPoint.func);
        paramStructType = builder.createStructType();
        builder.addNameHintDecoration(
            paramStructType,
            UnownedTerminatedStringSlice("ShaderRecordParams"));

        // If we need a constant buffer, then the global
        // shader parameter will be a `ConstantBuffer<paramStructType>`
        // TODO: reconcile this with OptiX, as the current logic works, but is still focused on
        // VK/DXR..
        //
        auto constantBufferType = builder.getConstantBufferType(
            paramStructType,
            builder.getType(kIROp_DefaultBufferLayoutType));
        collectedParam = builder.createParam(constantBufferType);

        // The global shader parameter should have the layout
        // information from the entry point attached to it, so that the
        // contained parameters will end up in the right place(s).
        //
        builder.addLayoutDecoration(collectedParam, entryPointParamsLayout);

        // We add a name hint to the global parameter so that it will
        // emit to more readable code when referenced.
        //
        builder.addNameHintDecoration(
            collectedParam,
            UnownedTerminatedStringSlice("shaderRecordParams"));
    }
};

void collectOptiXEntryPointUniformParams(IRModule* module)
{
    // look into all entry point functions by checking the IREntryPointDecoration on the children
    // Insts of the module. For any ray tracing entry points, collect all uniform parameters into
    // one common struct, and replace parameter usage with SBT record accesses.
    CollectOptixEntryPointUniformParams context;
    context.processModule(module);
}

} // namespace Slang
