// slang-ir-specialize-resources.cpp
#include "slang-ir-specialize-resources.h"

#include "slang-ir-clone.h"
#include "slang-ir-inline.h"
#include "slang-ir-insts.h"
#include "slang-ir-specialize-function-call.h"
#include "slang-ir-ssa-simplification.h"
#include "slang-ir.h"

namespace Slang
{

struct ResourceParameterSpecializationCondition : FunctionCallSpecializeCondition
{
    // This pass is intended to specialize functions
    // with resource parameters to ensure that they are
    // legal for a given target.

    TargetRequest* targetRequest = nullptr;
    TargetProgram* targetProgram = nullptr;

    bool doesParamWantSpecialization(IRParam* param, IRInst* arg)
    {
        SLANG_UNUSED(arg);

        // Whether or not a parameter needs specialization is really
        // a function of its type:
        //
        IRType* type = param->getDataType();

        // What's more, if a parameter of type `T` would need
        // specialization, then it seems clear that a parameter
        // of type "array of `T`" would also need specialization.
        // We will "unwrap" any outer arrays from the parameter
        // type before moving on, since they won't affect
        // our decision.
        //
        type = unwrapArray(type);
        bool isArray = type != param->getDataType();

        // On all of our (current) targets, a function that
        // takes a `ConstantBuffer<T>` parameter requires
        // specialization. Surprisingly this includes DXIL
        // because dxc apparently does not treat `ConstantBuffer<T>`
        // as a first-class type.
        //
        // TODO: This should not apply to CPU or CUDA, where
        // `ConstantBuffer<T>` is just `T*`. Right now this
        // optimization is not applying to those targets by
        // coincidence (because the shader parameters are not
        // globals, there is no way for the specialization to
        // succeed), but eventually we should turn it off more
        // carefully.
        //
        if (as<IRUniformParameterGroupType>(type))
            return true;

        // For GL/Vulkan targets, we also need to specialize
        // any parameters that use structured or byte-addressed
        // buffers or images with format qualifiers.
        //
        if (isKhronosTarget(targetRequest))
        {
            if (targetProgram->getOptionSet().shouldEmitSPIRVDirectly())
                return isIllegalSPIRVParameterType(type, isArray);
            else
                return isIllegalGLSLParameterType(type);
        }
        else if (isWGPUTarget(targetRequest))
        {
            return isIllegalWGSLParameterType(type);
        }

        // For now, we will not treat any other parameters as
        // needing specialization, even if they use resource
        // types like `Texure2D`, because these are allowed
        // as function parameters in both HLSL and GLSL.
        //
        // TODO: We may want to perform more aggressive
        // specialization in general, especially insofar
        // as it could simplify the task of supporting
        // functions with resource-type outputs.

        return false;
    }
};

bool specializeResourceParameters(CodeGenContext* codeGenContext, IRModule* module)
{
    bool result = false;
    ResourceParameterSpecializationCondition condition;
    condition.targetProgram = codeGenContext->getTargetProgram();
    condition.targetRequest = codeGenContext->getTargetReq();
    bool changed = true;
    while (changed)
    {
        changed = specializeFunctionCalls(codeGenContext, module, &condition);
        result |= changed;
    }
    return result;
}

void inlineAllCallsOfFunction(IRFunc* func)
{
    traverseUses(
        func,
        [&](IRUse* use)
        {
            auto user = use->getUser();
            auto call = as<IRCall>(user);
            if (!call)
                return;
            if (call->getCallee() != func)
                return;
            inlineCall(call);
        });
}

/// A pass to specialize resource-typed function outputs
struct ResourceOutputSpecializationPass
{
    // This pass is kind of a dual to `specializeResourceParameters()`.
    // Whereas that pass identifies call sites that pass suitable argument
    // values and specializes the callee functionfor each such call site,
    // this pass identifies *functions* that *output* suitable values (either
    // via `return` or `out`/`inout` parmeters), and then specializes the
    // *call sites* for those functions based on the values that are output.

    CodeGenContext* codeGenContext;
    TargetRequest* targetRequest;
    IRModule* module;

    /// Functions that requires specialization but are currently unspecializable.
    HashSet<IRFunc*>* unspecializableFuncs;

    /// Functions that required specialization and were specialized.
    HashSet<IRFunc*> specializedFuncs;

    enum class SpecializeFuncResult
    {
        OtherFuncFailed = -2,
        ThisFuncFailed = -1,
        Ok = 1,
    };

    bool failedResult(SpecializeFuncResult val) { return val < SpecializeFuncResult::Ok; }

    bool processModule()
    {
        specializedFuncs.clear();
        bool changed = false;

        // The main logic consists of iterating over all functions
        // (which must appear at the global level) and specializing
        // them if needed.
        //
        for (auto inst : module->getGlobalInsts())
        {
            auto func = as<IRFunc>(inst);
            if (!func)
                continue;

            changed |= processFunc(func);
        }
        return changed;
    }

    bool processFunc(IRFunc* oldFunc)
    {
        // Avoid re-computing by checking our 'processFunc' cache.
        if (specializedFuncs.contains(oldFunc))
            return true;
        if (unspecializableFuncs->contains(oldFunc))
            return false;

        // We don't want to waste any effort on functions that don't merit
        // specialization, so the first step is to identify if the function
        // has any outputs that use resource types.
        //
        // If there are no suitable outputs, then we bail out and skip
        // the given function.
        //
        if (!shouldSpecializeFunc(oldFunc))
            return false;

        // It is possible that we have a function that we *should* specialize
        // (based on its signature), but we *cannot* yet specialize it.
        //
        // Rather than try to detect that situation as a pre-process, we
        // will instead take the simpler approach of trying to produce
        // a specialized version of `oldFunc`, and bail out if we run
        // into any problems.
        //
        // TODO: It is possible that the allocation we perform here could
        // lead to performance issues if this pass gets iterated. Eventually
        // we should probably merge the resource-based specialization logic
        // into a combined pass that specializes in both directions and
        // also folds in SSA formation to clean up temporaries.

        // We start the specialization process by making a clone of the
        // original function.
        //
        IRBuilder builder(module);
        builder.setInsertBefore(oldFunc);
        IRFunc* newFunc = builder.createFunc();
        newFunc->setFullType(oldFunc->getFullType());

        IRCloneEnv cloneEnv;
        cloneInstDecorationsAndChildren(&cloneEnv, module, oldFunc, newFunc);

        // At first `newFunc` is a direct clone of `oldFunc`, and thus doesn't
        // solve any of our problems. We will traverse `oldFunc` and specialize
        // it as needed, while also collecting information that will allow
        // us to rewrite call sites.
        //
        FuncInfo funcInfo;
        SpecializeFuncResult result = specializeFunc(newFunc, funcInfo);
        if (failedResult(result))
        {
            // Even though we deterined that we *should* specialize
            // this function, we were not able to because of some
            // failure inside the body of the function.
            //
            // For now, we don't treat this as an error condition,
            // because subsequent optimization could make it so
            // that another attempt at this pass succeeds.
            //
            // TODO: We should iterate on this pass and the relevant
            // simplifications, and keep attempting until we hit
            // a steady state, before running this pass one
            // last time with a flag that causes it to emit an
            // error message on this falure path.
            //
            // TODO: Of course, we should *also* have front-end
            // validation that ensures that functions that include
            // targets with limited resource capabilities only use
            // potentially-resource-bearing types in ways that we
            // are sure we can optimize/simplify, so that the error
            // messages can be front-end rather than back-end errors.
            //
            newFunc->removeAndDeallocate();
            // Check if `oldFunc` is the reason for failing,
            // Otherwise don't add to 'unspecializableFuncs'
            //
            // Ensure oldFunc has uses, else, there is nothing to specialize here.
            // If oldFunc has IRKeepAlive, this code should be assumed to have a
            // "dynamic" resource value.
            if (result == SpecializeFuncResult::ThisFuncFailed && oldFunc->hasUses())
                unspecializableFuncs->add(oldFunc);
            return false;
        }

        // Specialization might have changed the signature of `newFunc`,
        // by adding/removing parameters, or by changing the result
        // type it returns.
        //
        // There is a utility function called `fixUpFuncType` that can
        // change the type of an IR function to match its parameter list,
        // but we need to compute the desired result type manually.
        //
        // The result type defaults to the result type of the original
        // function, but should be changed to `void` if specialization
        // was applied to the function result.
        //
        IRType* newResultType = oldFunc->getResultType();
        if (funcInfo.result.flavor != OutputInfo::Flavor::None)
            newResultType = builder.getVoidType();
        fixUpFuncType(newFunc, newResultType);

        // At this point, we generated a `newFunc` that specialized `oldFunc`,
        // and can be used instead of it at any direct call sites.
        //
        // We are going to replace those call sites, which will modify the
        // use-def information for `oldFunc`, so we start by collecting the
        // call sites into an array.
        //
        // Note: We are ignoring any uses that are not direct calls of `oldFunc`;
        // alternative use sites might include references to the function from
        // witness tables, etc. The expectation when using this pass is that
        // any other uses of `oldFunc` will eventually be eliminated, so that
        // only the specialized version remains. If uses of the unspecialized
        // function remain, they could cause problems for downstream code generation.
        //
        // Targets that want to support true dynamic dispatch through witness
        // tables or higher-order functions will need to either disallow resource-type
        // returns from such functions, or support resource-type returns without
        // the aid of this pass.
        //
        List<IRCall*> calls;
        traverseUses(
            oldFunc,
            [&](IRUse* use)
            {
                auto user = use->getUser();
                auto call = as<IRCall>(user);
                if (!call)
                    return;
                if (call->getCallee() != oldFunc)
                    return;
                calls.add(call);
            });

        // Once we have identified the calls to `oldFunc`, we will set about replacing
        // them with calls to `newFunc`.
        //
        // Note: from this point on specialization is not allowed to fail; if the callee
        // function could be specialized then all call sites to it must be specialized.
        // There should be no conditions at call sites that can cause specialization to
        // fail, because specialization does not depend on what is passed *in* to each
        // call, but only on what gets passed *out*.
        //
        for (auto oldCall : calls)
        {
            specializeCallSite(oldCall, newFunc, funcInfo);
        }
        specializedFuncs.add(oldFunc);

        // Since we can no longer fail and we are replacing all `Func` uses, 'KeepAlive'
        // can be removed from the oldFunc so DCE can it clean-up.
        if (auto keepAliveDecoration = oldFunc->findDecoration<IRKeepAliveDecoration>())
            keepAliveDecoration->removeAndDeallocate();
        return true;
    }

    // With the overall flow of the pass described, we can now drill down
    // to the subroutines and data structures that make the whole task possible.
    //
    // We start with the simple problm of deciding whether or not we should
    // (attempt to) specialize a given function.
    //
    bool shouldSpecializeFunc(IRFunc* func)
    {
        // We cannot specialize a function if we do not have
        // access to its definition.
        //
        if (!func->isDefinition())
            return false;
        UnownedStringSlice def;
        IRInst* intrinsicInst;
        if (findTargetIntrinsicDefinition(func, targetRequest->getTargetCaps(), def, intrinsicInst))
            return false;

        // If any of the parameters of the function are `out`
        // or `inout` parameters of a resource type, then we
        // should specialize the function.
        //
        for (auto param : func->getParams())
        {
            auto paramType = param->getDataType();
            auto outType = as<IROutTypeBase>(paramType);
            if (!outType)
                continue;
            auto valueType = outType->getValueType();
            if (isResourceType(valueType))
                return true;
        }

        // If the result type of the function is a resource type,
        // then we should specialize the function.
        //
        if (isResourceType(func->getResultType()))
        {
            return true;
        }

        // If the above checks do not trigger, then we don't
        // need/want to specialize the function after all.
        //
        return false;
    }

    // For the above function to work, we need to be able to identify
    // the resource types (and arrays thereof) that require specialization.
    //
    // TODO: It seems like we should be able to share a central definition
    // of resource-ness.
    //
    // Note: we do not worry about parameters/results that are structures
    // with resource-type fields, under the assumption that resource
    // legalization has already been applied, exposing all resource-type
    // parameters as their own top-level parameters.
    //
    // TODO: resource legalization may not apply correctly to function
    // results and `out`/`inout` parameters, in which case we need to
    // fix that pass.
    //
    bool isResourceType(IRType* type)
    {
        type = unwrapArray(type);

        if (as<IRResourceTypeBase>(type))
            return true;

        if (as<IRUniformParameterGroupType>(type))
            return true;

        if (as<IRHLSLStructuredBufferTypeBase>(type))
            return true;

        if (as<IRByteAddressBufferTypeBase>(type))
            return true;

        if (as<IRSamplerStateTypeBase>(type))
            return true;

        if (as<IRRayQueryType>(type))
            return true;

        if (as<IRHitObjectType>(type))
            return true;

        // TODO: more cases here?

        return false;
    }

    // Once we've decided that a function is worth specializing,
    // we will both transform the function and collect information
    // about its outputs.
    //
    // The central piece of the data structure we will use is
    // `OutputInfo`, which will track information about one
    // (possible) function output that might need specialization.

    /// Information about a possible output of a function (return value or output parameter)
    struct OutputInfo
    {
        enum class Flavor
        {
            None, ///< Not actually an output, or does not need specialization

            Undefined, ///< Needs specialization, but no suitable replacement value is known

            Replace, ///< A replacement value should be computed based on `representative`
        };

        /// What sort of output value is this?
        Flavor flavor = Flavor::None;

        /// For an output value with the `Replace` flavor, the representative value to clone.
        IRInst* representative = nullptr;

        /// The index of the first new output parameter introduced for this output
        Index firstNewOutputParamIndex = 0;

        /// The number of new output parameters introduced for this output
        Index newOutputParamCount = 0;
    };

    // The function result will be tracked as an `OutputInfo`, and
    // we will define a subtype specific to that case, even though
    // it does not currently need to track any additional data.

    /// A representation of the return-value output of a function
    struct ReturnValueInfo : OutputInfo
    {
    };

    // Parameters can be outputs, so they will also collect information
    // into `OutputInfo`s, but they also need additional information
    // related to the fact that parameters have corresponding arguments
    // at call sites, and how we specialize the parameter affects
    // what we need to do with those arguments.

    /// A representation of a parameter (possibly an output) of a function
    struct ParamInfo : OutputInfo
    {
        /// Represents what to do with an existing argument at a call site.
        enum class OldArgMode
        {
            Keep,   ///< Keep the argument as-is.
            Ignore, ///< Ignore the argument (eliminate it from the call)
            Deref,  ///< Dereference the argument; it used to be `inout` and is now just `in`
        };

        /// What do do with existing arguments at call sites
        OldArgMode oldArgMode = OldArgMode::Keep;
    };

    // It is possible that specializing a function output may require
    // us to add new output parameters to the function, to enable
    // the caller to compute the correct output resource.
    //
    // For example, consider this input:
    //
    //      Texture2D getRandomTexture()
    //      {
    //          int index = /* complicated logic */;
    //          return gTextures[index];
    //      }
    //      ...
    //      Texture2D t = getRandomTexture();
    //
    // The desired output is:
    //
    //      void _getRandomTexture(out int i)
    //      {
    //          int index = /* complicated logic */;
    //          i = index;
    //      }
    //      ...
    //      int i;
    //      getRandomTexture(i);
    //      Texture2D t = gTextures[i];
    //
    // In this case we have made the computation of `t` be
    // valid for targets with limited resource support, but
    // we have kept the complicated computation of `index`
    // in a subroutine, so that we have not bloated the
    // code more than necessary.
    //
    // In order to track new parameters like `i` above,
    // we introduce the `NewOutputParamInfo` type.

    /// Represents a new output parameter introduced during speicalization
    struct NewOutputParamInfo
    {
        /// The type of the new parameter's *value* (not the pointer type for an `out` parameter)
        IRType* type;
    };

    // Finally, we can aggregate the types above to represent the
    // collected information about a function to be specialized.

    /// Information about a function to be specialized
    struct FuncInfo
    {
        ReturnValueInfo result;
        List<ParamInfo> oldParams;
        List<NewOutputParamInfo> newOutputParams;
    };

    // We now turn to the code that fills in the `FuncInfo` structure.

    SpecializeFuncResult specializeFunc(IRFunc* func, FuncInfo& outFuncInfo)
    {
        // To specialize a function, we attempt to specialize
        // all the applicable parameters and the function result.
        //
        // Any failures along the way cause the whole process to fail.

        // Note: We are introducing new parameters at the same time as we
        // iterate over the parameter list, so we cannot just use the
        // `func->getParams()` convenience accessor. Instead, we manually
        // iterate over the parameters in a way that avoids invalidation
        // if we remove the `param` we are working on.
        //
        // Note: it might seem odd that we are modifying `func` but will
        // still bail out on any errors. You might ask: isn't there a chance
        // that we will end up with the function in a partially-modified state?
        //
        // The important thing to remember is that `func` is  *copy* of the
        // original function, so any modifications we make to it do not
        // affect the original, so that if we *do* have to bail out we can
        // leave any call sites intact as calls to the original. The result
        // is that bailing out here may leave the new/copied function in
        // a state where it isn't useful, but it also won't have any uses,
        // and can be eliminated later.
        //
        IRParam* nextParam = nullptr;
        for (IRParam* param = func->getFirstParam(); param; param = nextParam)
        {
            nextParam = param->getNextParam();

            ParamInfo paramInfo;
            auto result = maybeSpecializeParam(param, paramInfo, outFuncInfo);
            if (failedResult(result))
                return result;
            outFuncInfo.oldParams.add(paramInfo);
        }

        auto result = maybeSpecializeResult(func, outFuncInfo.result, outFuncInfo);
        if (failedResult(result))
            return result;

        return SpecializeFuncResult::Ok;
    }

    // The logic for specializing a function result (the return value) is
    // simpler than that for parameters, so we will look at it first.

    SpecializeFuncResult maybeSpecializeResult(
        IRFunc* func,
        ReturnValueInfo& outResultInfo,
        FuncInfo& ioFuncInfo)
    {
        // If the result type of the function isn't a resource type,
        // then we don't need to specialize the result, and we
        // can succeed without doing anything.
        //
        if (!isResourceType(func->getResultType()))
            return SpecializeFuncResult::Ok;

        // Otherwise, we know that we will need to produce specialization
        // information in `outResultInfo` or fail in the attempt.
        //
        // We start with the `prepareOutputValue` subroutine which will
        // handle some common logic shared with the parameter case.
        //
        prepareOutputValue(outResultInfo, ioFuncInfo);

        // Next, we want to identify all the places where the function
        // `return`s a value, since those establish all the possible
        // values for the function result.
        //
        // Specialization will only be possible if all of those results
        // return the "same" value, or values that are in some way
        // similar enough for us to collapse into a single pattern.
        //
        // Identifying the return sites is as simple as looking at
        // the terminator instructions of all blocks in the function.
        //
        for (auto block : func->getBlocks())
        {
            auto returnInst = as<IRReturn>(block->getTerminator());
            if (!returnInst)
                continue;

            auto value = returnInst->getVal();

            IRBuilder builder(module);
            builder.setInsertBefore(returnInst);

            // Given the `value` being returned, we need to determine
            // if it is usable for specializing call sites to this
            // function.
            //
            // If there is a single `return` site, then we can use
            // the value returned there as a representative of the
            // value returned.
            //
            // If there are multiple `return` sites, then any sites
            // after the first will check if they are similar enough
            // in structure to the first one to allow specialization
            // to proceed.
            //
            // If we either fail to identify a specializable result
            // or to match a new `return` value against previous
            // ones, then the specialization process will fail.
            //
            auto result = specializeOutputValue(value, outResultInfo, ioFuncInfo);
            if (failedResult(result))
                return result;

            // We will replace the `return <value>;` operation with
            // a simple `return;`, because the new specialized function
            // will have no return value.
            //
            builder.emitReturn();
            returnInst->removeAndDeallocate();
        }

        // If we have succeeded in gathering information from all
        // the `return` sites, then we can finish up computing
        // `outResultInfo` and return successfully.
        //
        completeOutputValue(outResultInfo, ioFuncInfo);
        return SpecializeFuncResult::Ok;
    }

    void prepareOutputValue(OutputInfo& ioValueInfo, FuncInfo& ioFuncInfo)
    {
        // This function is called when we have identified that a particular
        // value *does* represent an output, but before we have determined
        // what value(s) are used for that output.
        //
        // As such, we set the output into a mode where its value is undefined,
        // since that is the approrpiate default to use in the case where
        // the function doesn't actually write anything to an output.
        //
        ioValueInfo.flavor = OutputInfo::Flavor::Undefined;

        // We also know that the output might require zero or more new output
        // parameters, and we can set the starting index of those parameters
        // based on what (if any) has been generated so far.
        //
        ioValueInfo.firstNewOutputParamIndex = ioFuncInfo.newOutputParams.getCount();
    }

    void completeOutputValue(OutputInfo& ioValueInfo, FuncInfo& ioFuncInfo)
    {
        // This function is called when we are done computing the information
        // required to specialize a particular output value.
        //
        // We can now determine how many new output parameters, if any,
        // were introduced for the sake of this output.
        //
        ioValueInfo.newOutputParamCount =
            ioFuncInfo.newOutputParams.getCount() - ioValueInfo.firstNewOutputParamIndex;
    }

    SpecializeFuncResult specializeOutputValue(
        IRInst* value,
        OutputInfo& ioOutputInfo,
        FuncInfo& ioFuncInfo)
    {
        // This function is called or each `value` that might be written
        // to the output identified by `ioOutputInfo`.

        // If this is the first call to for the given output, then
        // the `representative` value will not have been set.
        //
        IRInst* representative = ioOutputInfo.representative;
        if (!representative)
        {
            // In that case, we will use the given `value` as the
            // representative value of this output.
            //
            representative = value;
            ioOutputInfo.flavor = OutputInfo::Flavor::Replace;
            ioOutputInfo.representative = representative;
        }

        // If this is *not* the first call for the given output,
        // then we need to confirm that `value` and `representative`
        // are suitably matched so that specialization based on `representative`
        // will also suffice for `value`.
        //
        // At the very least, we expect them to be operations with
        // the same opcode.
        //
        if (value->getOp() != representative->getOp())
            return SpecializeFuncResult::ThisFuncFailed;

        // Furthermore, only certain instructions are amenable to
        // specialization, because in general we cannot reproduce
        // an instruction outside of its containing function and
        // have it mean the same thing.
        //
        // We will specifically enumerate the case that we support,
        // and expand them over time.
        //
        // Each supported instruction opcode might introduce new
        // constraints on how `value` and `representative` must match.
        //
        switch (value->getOp())
        {
        default:
            // Any opcode we do not specifically enable should cause
            // specialization to fail.
            //
            return SpecializeFuncResult::ThisFuncFailed;

        case kIROp_GlobalParam:
            // A direct reference to a global shader parameter is
            // the easiest case to handle.
            //
            // We do need to require that all values used for the
            // same output refer to the *same* global parameter.
            //
            if (value != representative)
                return SpecializeFuncResult::ThisFuncFailed;
            return SpecializeFuncResult::Ok;

            // TODO: There are a number of additional cases that we should
            // enable here.
            //
            // The most obvious new cases to support are:
            //
            // * Function parameters: if the output value is one of the
            //   parameter of the function, then callers can just use the
            //   same value they passed for the corresponding argument.
            //
            // * Array indexing: if the array itself is suitable to specialize,
            //   then it should be possible to return the array index via
            //   a new `out` parameter, and have the caller do the indexing.
        }

        // Note: the `FuncInfo` is currently being passed in in aid of the
        // array-indexing case, but because that case is not implemented
        // the parameter is not being used.
        //
        SLANG_UNUSED(ioFuncInfo);

        // TODO: One of the hardest cases here would be `inout` parameters
        // of texture type, where the result value depends on the input value(s):
        //
        //      void swap(inout Texture2D a, inout Texture2D b)
        //      {
        //          Texture2D tmp = a;
        //          a = b;
        //          b = tmp;
        //      }
        //
        // In such a case, the value written to `a` is a `load` from parameter `b`,
        // but it would be difficult to *prove* that such a load represents the
        // value of the parameter on input to the function, rather than on output.
        //
        // It might be best if resource type legalization replaced `inout`
        // parameters of resource type with distinct `in` and `out` parameters,
        // to make the relationships more clear.
    }

    // As discussed earlier, the case for `out`/`inout` function parameters
    // is more involved than that for the function `return` value, so we
    // put it off until we'd discussed the shared subroutines.

    SpecializeFuncResult maybeSpecializeParam(
        IRParam* param,
        ParamInfo& outParamInfo,
        FuncInfo& ioFuncInfo)
    {
        // We only want to specialize in the case where the parameter
        // is an `out` or `inout` (both inherit from `IROutTypeBase`),
        // and the pointed-to type is a resource.
        //
        auto paramType = param->getDataType();
        auto outType = as<IROutTypeBase>(paramType);
        if (!outType)
            return SpecializeFuncResult::Ok;
        auto valueType = outType->getValueType();
        if (!isResourceType(valueType))
            return SpecializeFuncResult::Ok;

        prepareOutputValue(outParamInfo, ioFuncInfo);

        // We are going to remove the parameter and add zero or more
        // replacements, and we want any replacements to end up
        // at the same point in the function signature.
        //
        IRBuilder paramsBuilder(module);
        paramsBuilder.setInsertBefore(param);

        // We also need to introduce new instructions into the function
        // body, as part of the entry block.
        //
        IRBlock* block = as<IRBlock>(param->getParent());
        IRBuilder bodyBuilder(module);
        bodyBuilder.setInsertBefore(block->getFirstOrdinaryInst());

        // No matter what, we create a local variable that will be
        // used to replace the parameter.
        //
        IRVar* newVar = bodyBuilder.emitVar(valueType);

        if (as<IRInOutType>(outType))
        {
            // If the parameter is an `inout` rather than just
            // an `out`, then we still need a parameter to
            // be passed in, but it can be an `in` parameter
            // instead, which means a `T` instead of an
            // `InOut<T>`.
            //
            IRInst* newParam = paramsBuilder.createParam(valueType);
            newParam->insertBefore(param);
            param->transferDecorationsTo(newParam);

            // The start of the function body should assign
            // from the `in` parameter to the local variable.
            //
            bodyBuilder.emitStore(newVar, newParam);

            // We also need call sites to pass in an argument
            // for the new `in` parameter, which will have to
            // be dereferenced by one level from the original
            // argument they were passing.
            //
            outParamInfo.oldArgMode = ParamInfo::OldArgMode::Deref;
        }
        else
        {
            // The case for a pure `out` parameter is easier:
            // we don't need to initialize the local variable,
            // and we don't need callers to pass in anything.
            //
            outParamInfo.oldArgMode = ParamInfo::OldArgMode::Ignore;
        }

        // Before we change something (and likely break this
        // function if something fails after a change) we want
        // to identify all the places in the function
        // that `store` to the given output parameter.
        //
        // Note: this logic is subtly depending on the structure
        // of how the front-end generates code for `out` and `inout`
        // parameters:
        //
        // * The only `load` of an `inout` parameter is emitted at
        //   the very start of a function body, to copy it over to
        //   a temporary variable.
        //
        // * The only `store`s of an `out` or `inout` parameter are
        //   right before `return` instructions, to establish the
        //   final value of that parameter, and every `out`/`inout`
        //   parameter is stored along every control-flow path
        //   that reaches a `return`.
        //
        // Those invariants could easily be eliminated in a few different
        // ways. Notably, if we added some more clever memory optimizations,
        // then a pass could notice that we have:
        //
        //      let val = load(inoutParam);
        //      ...
        //      store(inoutParam, val);
        //
        // and optimize away the `store` (at least).
        //
        // For now we can get away with this because we don't do very many
        // interesting memory/pointer optimizations in Slang, but it is
        // still worrying to have this kind of assumption baked in.
        //
        // TODO: We should decide on an encoding for the behavior of
        // `out`/`inout` parameters that doesn't have as many "gotcha" cases.
        //
        // We will also now recursively specialize all `IRCall` inside a 'parent function'
        // when trying to specialize a 'parent function'. This is to ensure we do not remove
        // a parameter SSA needs for SSA'ing a localVar into a globalVar (and DCE requires
        // to not DCE an important 'IRCall').
        //
        SpecializeFuncResult recursiveSpecializationResult = SpecializeFuncResult::Ok;
        List<IRStore*> stores;

        // We'll first specialize any relevant calls that may affect the value stored into the
        // param. This may create more stores into the param.
        //
        traverseUses(
            param,
            [&](IRUse* use)
            {
                auto user = use->getUser();
                switch (user->getOp())
                {
                case kIROp_Call:
                    {
                        // This call may require an inline if it fails to specialize
                        IRFunc* func = as<IRFunc>(as<IRCall>(user)->getCallee());
                        if (!func)
                            return;

                        if (!processFunc(func))
                        {
                            recursiveSpecializationResult = SpecializeFuncResult::OtherFuncFailed;
                        }
                        return;
                    }
                default:
                    return;
                };
            });

        // If any call specialization fails, we may need to revisit this function at a later
        // iteration.
        if (failedResult(recursiveSpecializationResult))
            return recursiveSpecializationResult;

        // Then, traverse all stores into this param.
        traverseUses(
            param,
            [&](IRUse* use)
            {
                auto user = use->getUser();
                switch (user->getOp())
                {
                case kIROp_Store:
                    {
                        auto store = as<IRStore>(user);
                        if (store->ptr.get() != param)
                            return;
                        stores.add(store);
                        return;
                    }
                default:
                    return;
                };
            });

        // Having identified the places where a value is stored to
        // the output parameter, we iterate over those values to
        // ensure that they are all specializable and consistent
        // with one another.
        //
        for (auto store : stores)
        {
            auto value = store->val.get();
            auto result = specializeOutputValue(value, outParamInfo, ioFuncInfo);
            if (failedResult(result))
                return result;

            // Given our assumptions about how `store`s to output
            // parameters are used, we can eliminate all these `store`s
            // since the values they write won't ever be used.
            //
            store->removeAndDeallocate();
        }

        // It is possible that there will still be used of the parameter
        // even after we eliminate all the `store`s from it (e.g., the initial
        // `load` that pulls the value from an `inout` parameter), so we
        // replace any remaining uses of the parameter with the local
        // variable we introduced, before removing the parameter.
        //
        param->replaceUsesWith(newVar);
        param->removeAndDeallocate();

        completeOutputValue(outParamInfo, ioFuncInfo);
        return SpecializeFuncResult::Ok;
    }

    void specializeCallSite(IRCall* oldCall, IRFunc* newFunc, FuncInfo const& funcInfo)
    {
        // Given an existing call, we will insert a new call right before
        // it and then remove the old one.
        //
        IRBuilder builder(module);
        builder.setInsertBefore(oldCall);

        // The new callee may have additional `out` parameters that
        // represent things like array indices required by the
        // new lookup operations. The new call site will need
        // to introduce temporaries to capture the values of
        // these outputs.
        //
        List<IRVar*> newOutputVars;
        for (auto const& newOutputParamInfo : funcInfo.newOutputParams)
        {
            auto newOutputVar = builder.emitVar(newOutputParamInfo.type);
            newOutputVars.add(newOutputVar);
        }

        // Next, we need to build up the argument list for
        // the call, by looking at the information that
        // was recorded for each parameter of the original.
        //
        List<IRInst*> newArgs;
        Index oldParamCounter = 0;
        for (auto const& oldParamInfo : funcInfo.oldParams)
        {
            // We can grab the argument from the old call
            // that was being used for this parameter, but
            // we need to check whether or not the new call
            // will use it.
            //
            auto oldParamIndex = oldParamCounter++;
            auto oldArg = oldCall->getArg(oldParamIndex);

            // Depending on how the callee specialized this parameter,
            // we will pass the argument, or data derived from it,
            // or nothing.
            //
            switch (oldParamInfo.oldArgMode)
            {
            default:
                SLANG_UNEXPECTED("unhandled case");
                break;

            case ParamInfo::OldArgMode::Keep:
                // If the parameter was not specialized away, then
                // the argument should be passed as-is.
                //
                newArgs.add(oldArg);
                break;

            case ParamInfo::OldArgMode::Ignore:
                // If the parameter was specialized out of existence,
                // then we don't pass the argument in at all.
                //
                break;

            case ParamInfo::OldArgMode::Deref:
                // If an `inout` argument has been specialized into an
                // `in` argument, then we need to dereference the pointer
                // that was being passed in before, and pass in the value
                // it points to instead.
                //
                // Note: the expectation is that once the call site has
                // been specialized, subsequent optimization will eliminate
                // this `load`, and replace it with whatever value was
                // being stored for the `inout` argument before the call.
                //
                newArgs.add(builder.emitLoad(oldArg));
                break;
            }

            // A resource parameter that got specialized might also introduce new
            // `out` parameters that help the caller compute the right result
            // value (e.g., array indices). Those parameters will come right
            // in the parameter list right after the location of the original
            // parameter.
            //
            for (Index i = 0; i < oldParamInfo.newOutputParamCount; ++i)
            {
                newArgs.add(newOutputVars[oldParamInfo.firstNewOutputParamIndex + i]);
            }
        }

        // The function return value can also require new `out` parameters as
        // part of specialization; any parameters it introduces will go
        // over all the others.
        //
        for (Index i = 0; i < funcInfo.result.newOutputParamCount; ++i)
        {
            newArgs.add(newOutputVars[funcInfo.result.firstNewOutputParamIndex + i]);
        }

        // Once we've built up the argument list for the new call we can emit
        // it, and also transfer any helpful decorations from the old call
        // over to the new one.
        //
        auto newResultType = newFunc->getResultType();
        auto newCall = builder.emitCallInst(newResultType, newFunc, newArgs);
        oldCall->transferDecorationsTo(newCall);

        // Just calling the specialized function is not enough, of course,
        // because the whole point of this pass was to move the logic
        // that computes a result resource from the callee up to the caller.
        //
        // After the call is completed, any additional `out` arguments will
        // have had their values filled in (e.g., the callee will have
        // computed the array index to be used, etc.).
        //
        // We can now iterate over the parameters again, and identify
        // the output parameters that have been specialized.
        //
        oldParamCounter = 0;
        for (auto const& oldParamInfo : funcInfo.oldParams)
        {
            auto oldParamIndex = oldParamCounter++;
            auto oldArg = oldCall->getArg(oldParamIndex);

            // We skip over parameters that were not specialized.
            //
            if (oldParamInfo.flavor == OutputInfo::Flavor::None)
                continue;

            if (oldParamInfo.flavor == OutputInfo::Flavor::Undefined)
                continue;

            // For any paraemter that was specialized, we will use
            // the computed information on the parameter to materialize
            // a value for the output in the context of the caller.
            //
            auto value = materialize(builder, oldParamInfo);

            // For an `out` or `inout` parameter, the `oldArg` represents
            // a pointer where the output value should be stored, so
            // we can emulate the behavior of the original function
            // by storing the value as expected.
            //
            builder.emitStore(oldArg, value);
        }

        // If the function result is an output that needs to be
        // specialized, then we need to handle it much like the
        // parameter case above.
        //
        if (funcInfo.result.flavor != OutputInfo::Flavor::None)
        {
            // We materialize the expected function result into
            // an IR value in the context of the caller, and then
            // use that value to replace any uses of the return
            // value of the original call.
            //
            auto value = materialize(builder, funcInfo.result);
            oldCall->replaceUsesWith(value);
        }
        else
        {
            // If the call was specialized, but the return value
            // was not something that needed specialization, then
            // we still need to replace any uses of the original
            // call to use the value of the new call.
            //
            oldCall->replaceUsesWith(newCall);
        }

        // After we've fully wired up the new call, we eliminate
        // the old call site, which will have no more uses.
        //
        // Note: At this point, the body of the caller function
        // is likely to have opportunities for further optimization.
        // Simple dataflow optimizations should now be able to
        // resolve the identities of resources that had previously
        // only been visible as the value of local variables or
        // the results of `call` instructions.
        //
        oldCall->removeAndDeallocate();
    }

    // In order to specialize call sites to functions that output
    // resources, we need a way to materialize the value for an
    // output in the context of the caller, based on the information
    // that was gathered in the callee.

    IRInst* materialize(IRBuilder& builder, OutputInfo const& info)
    {
        // For now, we are only handling a small fraction of the
        // possible cases.

        SLANG_UNUSED(builder);

        // The basic idea is to look at the `representative` instruction
        // that stands in for the output value (which is an instruction
        // from the body of the callee), and to produce an equivalent
        // value in the context of the caller.
        //
        auto representative = info.representative;
        switch (representative->getOp())
        {
        default:
            // Because we only allow certain instructions when specializing
            // the callee, any instruction outside of the allowed ones
            // would represent an internal error.
            //
            SLANG_UNEXPECTED("unhandled case");
            UNREACHABLE_RETURN(nullptr);

        case kIROp_GlobalParam:
            // If the value in the callee was a reference to a global parameter,
            // then we can simply refer to the same parameter here in the caller.
            //
            return representative;

            // TODO: As other cases are added to `specializeOutputValue()`, we will
            // need to add corresponding cases here.
        }
    }

    // TODO: A really important mising step here is that we need AST-level rules
    // that express the constraints on how resource-bearing types can and
    // cannot be used for local variables, `out` parameters, etc.

    // TODO: We should add another pass that takes any global variables
    // of resource type and transforms them into `in`/`out`/`inout` parameters
    // in any function that accesses them (and proceeds transitively up
    // the call stack), with a special rule that the globals translate into
    // local variables in each entry point function that needs them.
    //
    // Such a pass would reduce the problem of supporting global variables
    // with resource types to that of supporting locals and return values of
    // resource type.
    //
    // Note: that same pass could just apply to *all* globals for targets where
    // HLSL-style thread-local globals aren't supported. The main challenge that
    // would need to be worked out there is interaction with separate compilation,
    // but transforming them so that the function signatures are changed makes
    // the challenge more explicit and thus perhaps easier to tackle.
};

bool specializeResourceOutputs(
    CodeGenContext* codeGenContext,
    IRModule* module,
    HashSet<IRFunc*>& unspecializableFuncs)
{
    auto targetRequest = codeGenContext->getTargetReq();
    if (isD3DTarget(targetRequest) || isKhronosTarget(targetRequest) || isWGPUTarget(targetRequest))
    {
    }
    else
    {
        // Don't bother applying this pass on targets that won't
        // benefit from it.
        //
        // TODO: it would be good if we could express this kind
        // of conditional in a way that doesn't involve explicitly
        // enumerating matching targets.
        //
        return false;
    }

    ResourceOutputSpecializationPass pass;
    pass.codeGenContext = codeGenContext;
    pass.targetRequest = targetRequest;
    pass.module = module;
    pass.unspecializableFuncs = &unspecializableFuncs;
    return pass.processModule();
}

bool specializeResourceUsage(CodeGenContext* codeGenContext, IRModule* irModule)
{
    bool result = false;
    // We apply two kinds of specialization to clean up resource value usage:
    //
    // * Specalize call sites based on the actual resources
    //   that a called function will return/output.
    //
    // * Specialize called functions based on the actual resources
    //   passed as input at specific call sites.
    //
    // We need to run the two passes in an iterative fashion (combined with IR
    // simplification passes), because each optimization may open up opportunties
    // for the other to apply.
    //
    for (;;)
    {
        bool changed = true;
        HashSet<IRFunc*> unspecializableFuncs;
        while (changed)
        {
            changed = false;
            unspecializableFuncs.clear();
            // Because the legalization may depend on what target
            // we are compiling for (certain things might be okay
            // for D3D targets that are not okay for Vulkan), we
            // pass down the target request along with the IR.
            //
            changed |= specializeResourceOutputs(codeGenContext, irModule, unspecializableFuncs);
            changed |= specializeResourceParameters(codeGenContext, irModule);

            // After specialization of function outputs, we may find that there
            // are cases where opaque-typed local variables can now be eliminated
            // and turned into SSA temporaries. Such optimization may enable
            // the following passes to "see" and specialize more cases.
            //
            if (changed)
            {
                simplifyIR(
                    codeGenContext->getTargetProgram(),
                    irModule,
                    IRSimplificationOptions::getFast(codeGenContext->getTargetProgram()));
            }
            result |= changed;
        }
        if (unspecializableFuncs.getCount() == 0)
            break;

        // Inline unspecializable resource output functions and then continue trying.
        for (auto func : unspecializableFuncs)
            inlineAllCallsOfFunction(func);

        simplifyIR(
            codeGenContext->getTargetProgram(),
            irModule,
            IRSimplificationOptions::getFast(codeGenContext->getTargetProgram()));
    }
    return result;
}

bool isIllegalGLSLParameterType(IRType* type)
{
    if (auto arrayType = as<IRArrayTypeBase>(type))
        return isIllegalGLSLParameterType(arrayType->getElementType());

    if (as<IRParameterGroupType>(type))
        return true;
    if (as<IRHLSLStructuredBufferTypeBase>(type))
        return true;
    if (as<IRByteAddressBufferTypeBase>(type))
        return true;
    if (as<IRGLSLImageType>(type))
        return true;
    if (auto texType = as<IRTextureType>(type))
    {
        switch (texType->getAccess())
        {
        case SLANG_RESOURCE_ACCESS_READ_WRITE:
        case SLANG_RESOURCE_ACCESS_WRITE:
        case SLANG_RESOURCE_ACCESS_RASTER_ORDERED:
            return true;
        default:
            break;
        }
    }
    if (as<IRSubpassInputType>(type))
        return true;
    if (as<IRMeshOutputType>(type))
        return true;
    if (as<IRHLSLStreamOutputType>(type))
        return true;
    if (as<IRDynamicResourceType>(type))
        return true;
    if (as<IRHLSLInputPatchType>(type))
        return true;
    if (as<IRHLSLOutputPatchType>(type))
        return true;
    return false;
}

bool isIllegalSPIRVParameterType(IRType* type, bool isArray)
{
    if (isIllegalGLSLParameterType(type))
        return true;

    // If we are emitting SPIRV direclty, we need to specialize
    // all Texture types.
    if (as<IRTextureType>(type))
        return true;
    if (isArray)
    {
        if (as<IRSamplerStateTypeBase>(type))
        {
            return true;
        }
    }
    return false;
}

bool isIllegalWGSLParameterType(IRType* type)
{
    return isIllegalGLSLParameterType(type);
}

} // namespace Slang
