// slang-ir-legalize-varying-params.cpp
#include "slang-ir-legalize-varying-params.h"

#include "slang-ir-clone.h"
#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-parameter-binding.h"

#include <set>

namespace Slang
{
// Convert semantic name (ignores case) into equivlent `SystemValueSemanticName`
SystemValueSemanticName convertSystemValueSemanticNameToEnum(String rawSemanticName)
{
    auto semanticName = rawSemanticName.toLower();

    SystemValueSemanticName systemValueSemanticName = SystemValueSemanticName::None;

#define CASE(ID, NAME)                                         \
    if (semanticName == String(#NAME).toLower())               \
    {                                                          \
        systemValueSemanticName = SystemValueSemanticName::ID; \
    }                                                          \
    else

    SYSTEM_VALUE_SEMANTIC_NAMES(CASE)
#undef CASE
    {
        systemValueSemanticName = SystemValueSemanticName::Unknown;
        // no match
    }
    return systemValueSemanticName;
}

// This pass implements logic to "legalize" the varying parameter
// signature of an entry point.
//
// The traditional Slang/HLSL model is to have varying input parameters
// be marked with "semantics" that can either mark them as user-defined
// or system-value parameters. In addition the result (return value)
// of the function can be marked, and effectively works like an `out`
// parameter.
//
// Other targets have very different models for how varying parameters
// are passed:
//
// * GLSL/SPIR-V declare user-defined varying input/output as global variables,
//   and system-defined varying parameters are available as magic built-in variables.
//
// * CUDA compute kernels expose varying inputs as magic built-in
//   variables like `threadIdx`.
//
// * Our CPU compilation path requires the caller to pass in a `ComputeThreadVaryingInput`
//   that specifies the values of the critical varying parameters for compute shaders.
//
// While these targets differ in how they prefer to represent varying parameters,
// they share the common theme that they cannot work with the varying parameter
// signature of functions as written in vanilla HLSL.
//
// This pass in this file is responsible for walking the parameters (and result)
// of each entry point in an IR module and transforming them into a form that
// is legal for each target. The shared logic deals with many aspects of the
// HLSL/Slang model for varying parameters that need to be "desugared" for these
// targets:
//
// * Slang allows either an `out` parameter or the result (return value) of the
//   entry point to be used interchangeably, so ensuring both cases are treated
//   the same is handled here.
//
// * Slang allows a varying parameter to use a `struct` or array type, so that
//   we need to recursively process elements and/or fields to find the leaf
//   varying parameters as they will be understood by other targets.
//
// * As an extension of the above, `struct`-type varying parameters in Slang
//   may mix user-defined and system-defined inputs/outputs.
//
// * Slang allows for `inout` varying parameters, which need to desugar into
//   distinct `in` and `out` parameters for targets like GLSL.

/// A placeholder that represents the value of a legalized varying
/// parameter, for the purposes of substituting it into IR code.
///
struct LegalizedVaryingVal
{
public:
    enum class Flavor
    {
        None, ///< No value (conceptually a literal of type `void`)

        Value, ///< A simple value represented as a single `IRInst*`

        Address, ///< A location in memory, identified by an address in an `IRInst*`
    };

    LegalizedVaryingVal() {}

    static LegalizedVaryingVal makeValue(IRInst* irInst)
    {
        return LegalizedVaryingVal(Flavor::Value, irInst);
    }

    static LegalizedVaryingVal makeAddress(IRInst* irInst)
    {
        return LegalizedVaryingVal(Flavor::Address, irInst);
    }

    Flavor getFlavor() const { return m_flavor; }

    IRInst* getValue() const
    {
        SLANG_ASSERT(getFlavor() == Flavor::Value);
        return m_irInst;
    }

    IRInst* getAddress() const
    {
        SLANG_ASSERT(getFlavor() == Flavor::Address);
        return m_irInst;
    }

private:
    LegalizedVaryingVal(Flavor flavor, IRInst* irInst)
        : m_flavor(flavor), m_irInst(irInst)
    {
    }

    Flavor m_flavor = Flavor::None;
    IRInst* m_irInst = nullptr;
};

/// Materialize the value of `val` as a single IR instruction.
///
/// Any IR code that is needed to materialize the value will be emitted to `builder`.
IRInst* materialize(IRBuilder& builder, LegalizedVaryingVal const& val)
{
    switch (val.getFlavor())
    {
    case LegalizedVaryingVal::Flavor::None:
        return nullptr; // TODO: should use a `void` literal

    case LegalizedVaryingVal::Flavor::Value:
        return val.getValue();

    case LegalizedVaryingVal::Flavor::Address:
        return builder.emitLoad(val.getAddress());

    default:
        SLANG_UNEXPECTED("unimplemented");
        break;
    }
}

void assign(IRBuilder& builder, LegalizedVaryingVal const& dest, LegalizedVaryingVal const& src)
{
    switch (dest.getFlavor())
    {
    case LegalizedVaryingVal::Flavor::None:
        break;

    case LegalizedVaryingVal::Flavor::Address:
        builder.emitStore(dest.getAddress(), materialize(builder, src));
        break;

    default:
        SLANG_UNEXPECTED("unimplemented");
        break;
    }
}

void assign(IRBuilder& builder, LegalizedVaryingVal const& dest, IRInst* src)
{
    assign(builder, dest, LegalizedVaryingVal::makeValue(src));
}


// Several of the derived calcluations rely on having
// access to the "group extents" of a compute shader.
// That information is expected to be present on
// the entry point as a `[numthreads(...)]` attribute,
// and we define a convenience routine for accessing
// that information.

IRInst* emitCalcGroupExtents(IRBuilder& builder, IRFunc* entryPoint, IRVectorType* type)
{
    if (auto numThreadsDecor = entryPoint->findDecoration<IRNumThreadsDecoration>())
    {
        static const int kAxisCount = 3;
        IRInst* groupExtentAlongAxis[kAxisCount] = {};

        for (int axis = 0; axis < kAxisCount; axis++)
        {
            auto litValue = as<IRIntLit>(numThreadsDecor->getOperand(axis));
            if (!litValue)
                return nullptr;

            groupExtentAlongAxis[axis] =
                builder.getIntValue(type->getElementType(), litValue->getValue());
        }

        return builder.emitMakeVector(type, kAxisCount, groupExtentAlongAxis);
    }

    // TODO: We may want to implement a backup option here,
    // in case we ever want to support compute shaders with
    // dynamic/flexible group size on targets that allow it.
    //
    SLANG_UNEXPECTED("Expected '[numthreads(...)]' attribute on compute entry point.");
    UNREACHABLE_RETURN(nullptr);
}

// There are some cases of system-value inputs that can be derived
// from other inputs; notably compute shaders support `SV_DispatchThreadID`
// and `SV_GroupIndex` which can both be derived from the more primitive
// `SV_GroupID` and `SV_GroupThreadID`, together with the extents
// of the thread group (which are specified with `[numthreads(...)]`).
//
// As a utilty to target-specific subtypes, we define helpers for
// calculating the value of these derived system values from the
// more primitive ones.

/// Emit code to calculate `SV_DispatchThreadID`
IRInst* emitCalcDispatchThreadID(
    IRBuilder& builder,
    IRType* type,
    IRInst* groupID,
    IRInst* groupThreadID,
    IRInst* groupExtents)
{
    // The dispatch thread ID can be computed as:
    //
    //      dispatchThreadID = groupID*groupExtents + groupThreadID
    //
    // where `groupExtents` is the X,Y,Z extents of
    // each thread group in threads (as given by
    // `[numthreads(X,Y,Z)]`).

    return builder.emitAdd(type, builder.emitMul(type, groupID, groupExtents), groupThreadID);
}

/// Emit code to calculate `SV_GroupIndex`
IRInst* emitCalcGroupIndex(IRBuilder& builder, IRInst* groupThreadID, IRInst* groupExtents)
{
    auto intType = builder.getIntType();
    auto uintType = builder.getBasicType(BaseType::UInt);

    // The group thread index can be computed as:
    //
    //      groupThreadIndex = groupThreadID.x
    //                       + groupThreadID.y*groupExtents.x
    //                       + groupThreadID.z*groupExtents.x*groupExtents.z;
    //
    // or equivalently (with one less multiply):
    //
    //      groupThreadIndex = (groupThreadID.z  * groupExtents.y
    //                        + groupThreadID.y) * groupExtents.x
    //                        + groupThreadID.x;
    //

    // `offset = groupThreadID.z`
    auto zAxis = builder.getIntValue(intType, 2);
    IRInst* offset = builder.emitElementExtract(uintType, groupThreadID, zAxis);

    // `offset *= groupExtents.y`
    // `offset += groupExtents.y`
    auto yAxis = builder.getIntValue(intType, 1);
    offset = builder.emitMul(
        uintType,
        offset,
        builder.emitElementExtract(uintType, groupExtents, yAxis));
    offset = builder.emitAdd(
        uintType,
        offset,
        builder.emitElementExtract(uintType, groupThreadID, yAxis));

    // `offset *= groupExtents.x`
    // `offset += groupExtents.x`
    auto xAxis = builder.getIntValue(intType, 0);
    offset = builder.emitMul(
        uintType,
        offset,
        builder.emitElementExtract(uintType, groupExtents, xAxis));
    offset = builder.emitAdd(
        uintType,
        offset,
        builder.emitElementExtract(uintType, groupThreadID, xAxis));

    return offset;
}

/// Context for the IR pass that legalizing entry-point
/// varying parameters for a target.
///
/// This is an abstract base type that needs to be inherited
/// to implement the appropriate policy for a particular
/// compilation target.
///
struct EntryPointVaryingParamLegalizeContext
{
    // This pass will be invoked on an entire module, and will
    // process all entry points in that module.
    //
public:
    void processModule(IRModule* module, DiagnosticSink* sink)
    {
        m_module = module;
        m_sink = sink;

        // We will use multiple IR builders during the legalization
        // process, to avoid having state changes on one builder
        // affect other builders that might be in use.
        //

        // Once the basic initialization is done, we will allow
        // the subtype to implement its own initialization logic
        // that should occur at the start of processing a module.
        //
        beginModuleImpl();

        // We now search for entry-point definitions in the IR module.
        // All entry points should appear at the global scope.
        //
        for (auto inst : module->getGlobalInsts())
        {
            // Entry points are IR functions.
            //
            auto func = as<IRFunc>(inst);
            if (!func)
                continue;

            // Entry point functions must have the `[entryPoint]` decoration.
            //
            auto entryPointDecor = func->findDecoration<IREntryPointDecoration>();
            if (!entryPointDecor)
                continue;

            // Once we find an entry point we process it immediately.
            //
            processEntryPoint(func, entryPointDecor);
        }
    }

protected:
    // As discussed in `processModule()`, a subtype can overide
    // the `beginModuleImpl()` method to perform work that should
    // only happen once per module that is processed.
    //
    virtual void beginModuleImpl() {}

    // We have both per-module and per-entry-point state that
    // needs to be managed. The former is set up in `processModule()`,
    // while the latter is used during `processEntryPoint`.
    //
    // Note: It would be possible in principle to remove some
    // the statefullness from this pass by factoring the
    // per-module and per-entry-point logic into distinct types,
    // but then every target-specific implementation would
    // need to comprise two types with complicated interdependencies.
    // The current solution of a single type with statefullness
    // seems easier to manage.

    IRModule* m_module = nullptr;
    DiagnosticSink* m_sink = nullptr;

    IRFunc* m_entryPointFunc = nullptr;
    IRBlock* m_firstBlock = nullptr;
    IRInst* m_firstOrdinaryInst = nullptr;
    Stage m_stage = Stage::Unknown;


    void processEntryPoint(IRFunc* entryPointFunc, IREntryPointDecoration* entryPointDecor)
    {
        m_entryPointFunc = entryPointFunc;

        // Before diving into the work of processing an entry point, we start by
        // extracting a bunch of information about the entry point that will
        // be useful to the downstream logic.
        //
        m_stage = entryPointDecor->getProfile().getStage();
        m_firstBlock = entryPointFunc->getFirstBlock();
        m_firstOrdinaryInst = m_firstBlock ? m_firstBlock->getFirstOrdinaryInst() : nullptr;

        auto entryPointLayoutDecoration = entryPointFunc->findDecoration<IRLayoutDecoration>();
        SLANG_ASSERT(entryPointLayoutDecoration);

        auto entryPointLayout = as<IREntryPointLayout>(entryPointLayoutDecoration->getLayout());
        SLANG_ASSERT(entryPointLayout);

        // Note: Of particular importance is that we extract the first/last parameters
        // of the function *before* we allow the subtype to perform per-entry-point
        // setup operations. This ensures that if the subtype adds new parameters to
        // the beginnign or end of the parameter list, those new parameters won't
        // be processed.
        //
        IRParam* firstOriginalParam = m_firstBlock ? m_firstBlock->getFirstParam() : nullptr;
        IRParam* lastOriginalParam = m_firstBlock ? m_firstBlock->getLastParam() : nullptr;

        // We allow the subtype to perform whatever setup or code generation
        // it wants to on a per-entry-point basis. In some cases this might
        // inject code into the start of the function to provide the value
        // of certain system-value parameters.
        //
        beginEntryPointImpl();

        // We now proceed to the meat of the work.
        //
        // We start by considering the result of the entry point function
        // if it is non-`void`.
        //
        auto resultType = entryPointFunc->getResultType();
        if (!as<IRVoidType>(resultType))
        {
            // We need to translate the existing function result type
            // into zero or more varying parameters that are legal for
            // the target. An entry point function result should be
            // processed in a way that semantically matches an `out` parameter.
            //
            auto legalResult = createLegalVaryingVal(
                resultType,
                entryPointLayout->getResultLayout(),
                LayoutResourceKind::VaryingOutput);

            // Now that we have a representation of the value(s) that will
            // be used to hold the entry-point result we need to transform
            // any `returnVal(r)` instructions in the function body to
            // instead assign `r` to `legalResult` and then `returnVoid`.
            //
            IRBuilder builder(m_module);
            for (auto block : entryPointFunc->getBlocks())
            {
                auto returnValInst = as<IRReturn>(block->getTerminator());
                if (!returnValInst)
                    continue;

                // We have a `returnVal` instruction that returns `resultVal`.
                //
                auto resultVal = returnValInst->getVal();

                // To replace the existing `returnVal` instruction we will
                // emit an assignment to the new legalized result (whether
                // a global variable, `out` parameter, etc.) and a `returnVoid`.
                //
                builder.setInsertBefore(returnValInst);
                assign(builder, legalResult, resultVal);
                builder.emitReturn();

                returnValInst->removeAndDeallocate();
            }
        }

        // The parameters of the entry-point function will be processed in
        // order to legalize them. We need to be careful when iterating
        // over the parameters for a few reasons:
        //
        // * The subtype-specific setup logic could have introduce parameters
        //   at the beginning or end of the list. We defend against that by
        //   capturing `firstOriginalParam` and `lastOriginalParam` at the
        //   start of this function, and only iterating over that range.
        //
        // * Somehow we might have an entry point declaration but not a definition
        //   this is unlikely but defended against because `firstOriginalParam`
        //   and `lastOriginalParam` will be null in that case.
        //
        // * We will often be removing the parameters once we have legalized
        //   them, so we will modify the list while traversing it. We defend
        //   against this by capturing `nextParam` at the start of each iteration
        //   so that we move to the same parameter next, even if the current
        //   parameter got removed.
        //
        // * The subtype-specific logic for legalizing a specific parameter
        //   might decide to insert new parameters to replace it. This is another
        //   case of modifying the parameter list while iterating it, and we
        //   defend against it with `nextParam` just like we do for the problem
        //  of deletion.
        //
        IRParam* nextParam = nullptr;
        for (auto param = firstOriginalParam; param; param = nextParam)
        {
            nextParam = param->getNextParam();

            processParam(param);

            if (param == lastOriginalParam)
                break;
        }
    }

    virtual void beginEntryPointImpl() {}

    // The next level down is the per-parameter processing logic, which
    // like the per-module and per-entry-point levels maintains its own
    // state to simplify the code (avoiding lots of long parameters lists).

    IRParam* m_param = nullptr;
    IRVarLayout* m_paramLayout = nullptr;

    void processParam(IRParam* param)
    {
        m_param = param;

        // We expect and require all entry-point parameters to have layout
        // information assocaited with them at this point.
        //
        auto paramLayoutDecoration = param->findDecoration<IRLayoutDecoration>();
        SLANG_ASSERT(paramLayoutDecoration);
        m_paramLayout = as<IRVarLayout>(paramLayoutDecoration->getLayout());
        SLANG_ASSERT(m_paramLayout);

        if (!isVaryingParameter(m_paramLayout))
            return;

        // TODO: The GLSL-specific variant of this pass has several
        // special cases that handle entry-point parameters for things like
        // GS output streams and input primitive topology.

        // TODO: The GLSL-specific variant of this pass has special cases
        // to deal with user-defined varying input to RT shaders, since
        // these don't translate to globals in the same way as all other
        // GLSL varying inputs.

        // We need to start by detecting whether the parameter represents
        // an `in` or an `out`/`inout` parameter, since that will determine
        // the strategy we take.
        //
        auto paramType = param->getDataType();
        if (auto inOutType = as<IRInOutType>(paramType))
        {
            processInOutParam(param, inOutType);
        }
        else if (auto outType = as<IROutType>(paramType))
        {
            processOutParam(param, outType);
        }
        else
        {
            processInParam(param, paramType);
        }
    }

    // We anticipate that some targets may need to customize the handling
    // of `out` and `inout` varying parameters, so we have `virtual` methods
    // to handle those cases, which just delegate to a default implementation
    // that provides baseline behavior that should in theory work for
    // multiple targets.
    //
    virtual void processInOutParam(IRParam* param, IRInOutType* inOutType)
    {
        processMutableParam(param, inOutType);
    }
    virtual void processOutParam(IRParam* param, IROutType* inOutType)
    {
        processMutableParam(param, inOutType);
    }

    void processMutableParam(IRParam* param, IROutTypeBase* paramPtrType)
    {
        // The deafult handling of any mutable (`out` or `inout`) parameter
        // will be to introduce a local variable of the corresponding
        // type and to use that in place of the actual parameter during
        // exeuction of the function.

        // The replacement variable will have the type of the original
        // parameter (the `T` in `Out<T>` or `InOut<T>`).
        //
        auto valueType = paramPtrType->getValueType();

        // The replacement variable will be declared at the top of
        // the function.
        //
        IRBuilder builder(m_module);
        builder.setInsertBefore(m_firstOrdinaryInst);

        auto localVar = builder.emitVar(valueType);
        auto localVal = LegalizedVaryingVal::makeAddress(localVar);

        if (const auto inOutType = as<IRInOutType>(paramPtrType))
        {
            // If the parameter was an `inout` and not just an `out`
            // parameter, we will create one more more legal `in`
            // parameters to represent the incoming value,
            // and then assign from those legalized input(s)
            // into our local variable at the start of the function.
            //
            auto inputVal =
                createLegalVaryingVal(valueType, m_paramLayout, LayoutResourceKind::VaryingInput);
            assign(builder, localVal, inputVal);
        }

        // Because the `out` or `inout` parameter is represented
        // as a pointer, and our local variabel is also a pointer
        // we can directly replace all uses of the original parameter
        // with uses of the variable.
        //
        param->replaceUsesWith(localVar);

        // For both `out` and `inout` parameters, we need to
        // introduce one or more legalized `out` parameters
        // to represent the outgoing value.
        //
        auto outputVal =
            createLegalVaryingVal(valueType, m_paramLayout, LayoutResourceKind::VaryingOutput);

        // In order to have changes to our local variable become
        // visible in the legalized outputs, we need to assign
        // from the local variable to the output as the last
        // operation before any `return` instructions.
        //
        for (auto block : m_entryPointFunc->getBlocks())
        {
            auto returnInst = as<IRReturn>(block->getTerminator());
            if (!returnInst)
                continue;

            builder.setInsertBefore(returnInst);
            assign(builder, outputVal, localVal);
        }

        // Once we are done replacing the original parameter,
        // we can remove it from the function.
        //
        param->removeAndDeallocate();
    }

    void processInParam(IRParam* param, IRType* paramType)
    {
        // Legalizing an `in` parameter is easier than a mutable parameter.

        // We start by creating one or more legalized `in` parameters
        // to represent the incoming value.
        //
        auto legalVal =
            createLegalVaryingVal(paramType, m_paramLayout, LayoutResourceKind::VaryingInput);

        // Next, we "materialize" the legalized value to produce
        // an `IRInst*` that represents it.
        //
        // Note: We materialize each input parameter once, at the top
        // of the entry point. Making a copy in this way could
        // introduce overhead if an input parameter is an array,
        // since all indexing operations will now refer to a copy
        // of the original array.
        //
        // TODO: We could in theory iterate over all uses of
        // `param` and introduce a custom replacement for each.
        // Such a replacement strategy could produce better code
        // for things like indexing into varying arrays, but at the
        // cost of more accesses to the input parameter data.
        //
        IRBuilder builder(m_module);
        builder.setInsertBefore(m_firstOrdinaryInst);
        IRInst* materialized = materialize(builder, legalVal);

        // The materialized value can be used to completely
        // replace the original parameter.
        //
        auto localVar = builder.emitVar(materialized->getDataType());
        builder.emitStore(localVar, materialized);
        param->replaceUsesWith(localVar);
        param->removeAndDeallocate();
    }

    // Depending on the "direction" of the parameter (`in`, `out`, `inout`)
    // we may need to create one or legalized variables to represented it.
    //
    // We now turn our attention to the problem of creating a legalized
    // value (wrapping zero or more variables/parameters) to represent
    // a varying parameter of a given type for a specific direction:
    // either input or output, but not both.
    //
    LegalizedVaryingVal createLegalVaryingVal(
        IRType* type,
        IRVarLayout* varLayout,
        LayoutResourceKind kind)
    {
        // The process we are going to use for creating legalized
        // values is going to involve recursion over the `type`
        // of the parameter, and there is a lot of state that
        // we need to carry along the way.
        //
        // Rather than have our core recursive function have
        // many parameters that need to be followed through
        // all the recursive call sites, we are going to wrap
        // the relevant data up in a `struct` and pass all
        // the information down as a bundle.

        auto typeLayout = varLayout->getTypeLayout();

        VaryingParamInfo info;
        info.type = type;
        info.varLayout = varLayout;
        info.typeLayout = typeLayout;
        info.kind = kind;

        return _createLegalVaryingVal(info);
    }

    // While recursing through the type of a varying parameter,
    // we may need to make a recursive call on the element type
    // of an array, while still tracking the fact that any
    // leaf parameter we encounter needs to have the "outer
    // array brackets" taken into account when giving it a type.
    //
    // For those purposes we have the `VaryingArrayDeclaratorInfo`
    // type that keeps track of outer layers of array-ness
    // for a parameter during our recursive walk.
    //
    // It is stored as a stack-allocated linked list, where the list flows
    // up through the call stack.
    //
    struct VaryingArrayDeclaratorInfo
    {
        IRInst* elementCount = nullptr;
        VaryingArrayDeclaratorInfo* next = nullptr;
    };

    // Here is the declaration of the bundled information we care
    // about when declaring a varying parameter.
    //
    struct VaryingParamInfo
    {
        // We obviously care about the type of the parameter we
        // need to legalize, as well as the layout of that type.
        //
        IRType* type = nullptr;
        IRTypeLayout* typeLayout = nullptr;

        // We also care about the variable layout information for
        // the parameter, because that includes things like the semantic
        // name/index, as well as any binding information that was
        // computed (e.g., for the `location` of GLSL user-defined
        // varying parameters).
        //
        // Note: the `varLayout` member may not represent a layout for
        // a variable of the given `type`, because we might be peeling
        // away layers of array-ness. Consider:
        //
        //      int stuff[3] : STUFF
        //
        // When processing the parameter `stuff`, we start with `type`
        // being `int[3]`, but then we will recurse on `int`. At that
        // point the `varLayout` will still refer to `stuff` with its
        // semantic of `STUFF`, but the `type` and `typeLayout` will
        // refer to the `int` type.
        //
        IRVarLayout* varLayout = nullptr;

        // As discussed above, sometimes `varLayout` will refer to an
        // outer declaration of array type, while `type` and `typeLayout`
        // refer to an element type (perhaps nested).
        //
        // The `arrayDeclarators` field stores a linked list representing
        // outer layers of "array brackets" that surround the variable/field
        // of `type`.
        //
        // If code decides to construct a leaf parameter based on `type`,
        // then it will need to use these `arrayDeclarators` to wrap the
        // type up to make it correct.
        //
        VaryingArrayDeclaratorInfo* arrayDeclarators = nullptr;

        // In some cases the decision-making about how to lower a parameter
        // will depend on the kind of varying parameter (input or output).
        //
        // TODO: We may find that there are cases where a target wants to
        // support true `inout` varying parameters, and `LayoutResourceKind`
        // cannot currently handle those.
        //
        LayoutResourceKind kind = LayoutResourceKind::None;

        // When we arrive at a leaf parameter/field, we can identify whether
        // it is a user-defined or system-value varying based on its semantic name.
        //
        // For convenience, target-specific subtypes only need to understand
        // the enumerated `systemValueSemanticName` rather than needing to
        // implement their own parsing of semantic name strings.
        //
        SystemValueSemanticName systemValueSemanticName = SystemValueSemanticName::None;
    };

    LegalizedVaryingVal _createLegalVaryingVal(VaryingParamInfo const& info)
    {
        // By default, when we seek to creating a legalized value
        // for a varying parameter, we will look at its type to
        // decide what to do.
        //
        // For most basic types, we will immediately delegate to the
        // base case (which will use target-specific logic).
        //
        // Note: The logic here will always fully scalarize the input
        // type, gernerated multiple SOA declarations if the input
        // was AOS. That choice is required for some cases in GLSL,
        // and seems to be a reasonable default policy, but it could
        // lead to some performance issues for shaders that rely
        // on varying arrays.
        //
        // TODO: Consider whether some carefully designed early-out
        // checks could avoid full scalarization when it is possible
        // to avoid. Those early-out cases would probably need to
        // align with the layout logic that is assigning `location`s
        // to varying parameters.
        //
        auto type = info.type;
        if (as<IRVoidType>(type))
        {
            return createSimpleLegalVaryingVal(info);
        }
        else if (as<IRBasicType>(type))
        {
            return createSimpleLegalVaryingVal(info);
        }
        else if (as<IRVectorType>(type))
        {
            return createSimpleLegalVaryingVal(info);
        }
        else if (as<IRMatrixType>(type))
        {
            // Note: For now we are handling matrix types in a varying
            // parameter list as if they were ordinary types like
            // scalars and vectors. This works well enough for simple
            // stuff, and is unlikely to see much use anyway.
            //
            // TODO: A more correct implementation will probably treat
            // a matrix-type varying parameter as if it was syntax
            // sugar for an array of rows.
            //
            return createSimpleLegalVaryingVal(info);
        }
        else if (auto arrayType = as<IRArrayType>(type))
        {
            // A varying parameter of array type is an interesting beast,
            // because depending on the element type of the array we
            // might end up needing to generate multiple parameters in
            // struct-of-arrays (SOA) fashion. This will notably
            // come up in the case where the element type is a `struct`,
            // with fields that mix both user-defined and system-value
            // semantics.
            //
            auto elementType = arrayType->getElementType();
            auto elementCount = arrayType->getElementCount();
            auto arrayLayout = as<IRArrayTypeLayout>(info.typeLayout);
            SLANG_ASSERT(arrayLayout);
            auto elementTypeLayout = arrayLayout->getElementTypeLayout();

            // We are going to recursively apply legalization to the
            // element type of the array, but when doing so we will
            // pass down information about the outer "array brackets"
            // that this type represented.
            //
            VaryingArrayDeclaratorInfo arrayDeclarator;
            arrayDeclarator.elementCount = elementCount;
            arrayDeclarator.next = info.arrayDeclarators;

            VaryingParamInfo elementInfo = info;
            elementInfo.type = elementType;
            elementInfo.typeLayout = elementTypeLayout;
            elementInfo.arrayDeclarators = &arrayDeclarator;

            return _createLegalVaryingVal(elementInfo);
        }
        else if (auto streamType = as<IRHLSLStreamOutputType>(type))
        {
            // Handling a geometry shader stream output type like
            // `TriangleStream<T>` is similar to handling an array,
            // but we do *not* pass down a "declarator" to note
            // the wrapping type.
            //
            // This choice is appropriate for GLSL because geometry
            // shader outputs are just declared as their per-vertex
            // types and not wrapped in array or stream types.
            //
            // TODO: If we ever need to legalize geometry shaders for
            // a target with different rules we might need to revisit
            // this choice.
            //
            auto elementType = streamType->getElementType();
            auto streamLayout = as<IRStreamOutputTypeLayout>(info.typeLayout);
            SLANG_ASSERT(streamLayout);
            auto elementTypeLayout = streamLayout->getElementTypeLayout();

            VaryingParamInfo elementInfo = info;
            elementInfo.type = elementType;
            elementInfo.typeLayout = elementTypeLayout;

            return _createLegalVaryingVal(elementInfo);
        }
        // Note: This file is currently missing the case for handling a varying `struct`.
        // The relevant logic is present in `slang-ir-glsl-legalize`, but it would add
        // a lot of complexity to this file to include it now.
        //
        // The main consequence of this choice is that this pass doesn't support varying
        // parameters wrapped in `struct`s for the targets that require this pass
        // (currently CPU and CUDA).
        //
        // TODO: Copy over the relevant logic from the GLSL-specific pass, as part of
        // readying this file to handle the needs of all targets.
        //
        else
        {
            // When no special case matches, we assume the parameter
            // has a simple type that we can handle directly.
            //
            return createSimpleLegalVaryingVal(info);
        }
    }

    LegalizedVaryingVal createSimpleLegalVaryingVal(VaryingParamInfo const& info)
    {
        // At this point we've bottomed out in the type-based recursion
        // and we have a leaf parameter of some simple type that should
        // also have a single semantic name/index to work with.

        // TODO: This seems like the right place to "wrap" the type back
        // up in layers of array-ness based on the outer array brackets
        // that were accumulated.

        // Our first order of business will be to check whether the
        // parameter represents a system-value parameter.
        //
        auto varLayout = info.varLayout;
        auto semanticInst = varLayout->findSystemValueSemanticAttr();
        if (semanticInst)
        {
            // We will compare the semantic name against our list of
            // system-value semantics using conversion to lower-case
            // to achieve a case-insensitive comparison (this is
            // necessary because semantics in HLSL/Slang do not
            // treat case as significant).
            //
            // TODO: It would be nice to have a case-insensitive
            // comparsion operation on `UnownedStringSlice` to
            // avoid all the `String`s we crete and thren throw
            // away here.
            //
            auto systemValueSemanticName =
                convertSystemValueSemanticNameToEnum(String(semanticInst->getName()));

            if (systemValueSemanticName != SystemValueSemanticName::None)
            {
                // If the leaf parameter has a system-value semantic, then
                // we need to translate the system value in whatever way
                // is appropraite for the target.
                //
                // TODO: The logic here is missing the behavior from the
                // GLSL-specific pass that handles type conversion when
                // a user-declared system-value parameter might not
                // match the type that was expected exactly (e.g., they
                // declare a `uint2` but the parameter is a `uint3`).
                //
                VaryingParamInfo systemValueParamInfo = info;
                systemValueParamInfo.systemValueSemanticName = systemValueSemanticName;
                return createLegalSystemVaryingValImpl(systemValueParamInfo);
            }

            // TODO: We should seemingly do something if the semantic name
            // implies a system-value semantic (starts with `SV_`) but we
            // didn't find a match.
            //
            // In practice, this is probably something that should be handled
            // at the layout level (`slang-parameter-binding.cpp`), and the
            // layout for a parameter should include the `SystemValueSemanticName`
            // as an enumerated value rather than a string (so that downstream
            // code doesn't have to get into the business of parsing it).
        }

        // If there was semantic applied to the parameter *or* the semantic
        // wasn't recognized as a system-value semantic, then we need
        // to do whatever target-specific logic is required to legalize
        // a user-defined varying parameter.
        //
        return createLegalUserVaryingValImpl(info);
    }

    // The base type will provide default implementations of the logic
    // for creating user-defined and system-value varyings, but in
    // each case the default logic will simply diagnose an error.
    //
    // For targets that support either case, it is essential to
    // override these methods with appropriate logic.

    virtual LegalizedVaryingVal createLegalUserVaryingValImpl(VaryingParamInfo const& info)
    {
        return diagnoseUnsupportedUserVal(info);
    }

    virtual LegalizedVaryingVal createLegalSystemVaryingValImpl(VaryingParamInfo const& info)
    {
        return diagnoseUnsupportedSystemVal(info);
    }

    // As a utility for target-specific subtypes, we define a routine
    // to diagnose the case of a system-value semantic that isn't
    // understood by the target.

    LegalizedVaryingVal diagnoseUnsupportedSystemVal(VaryingParamInfo const& info)
    {
        SLANG_UNUSED(info);

        m_sink->diagnose(
            m_param,
            Diagnostics::unimplemented,
            "this target doesn't support this system-defined varying parameter");

        return LegalizedVaryingVal();
    }

    LegalizedVaryingVal diagnoseUnsupportedUserVal(VaryingParamInfo const& info)
    {
        SLANG_UNUSED(info);

        m_sink->diagnose(
            m_param,
            Diagnostics::unimplemented,
            "this target doesn't support this user-defined varying parameter");

        return LegalizedVaryingVal();
    }
};

// With the target-independent core of the pass out of the way, we can
// turn our attention to the target-specific subtypes that handle
// translation of "leaf" varying parameters.

struct CUDAEntryPointVaryingParamLegalizeContext : EntryPointVaryingParamLegalizeContext
{
    // CUDA compute kernels don't support user-defined varying
    // input or output, and there are only a few system-value
    // varying inputs to deal with.
    //
    // CUDA provides built-in global parameters `threadIdx`,
    // `blockIdx`, and `blockDim` that we can make use of.
    //
    IRGlobalParam* threadIdxGlobalParam = nullptr;
    IRGlobalParam* blockIdxGlobalParam = nullptr;
    IRGlobalParam* blockDimGlobalParam = nullptr;

    // All of our system values will be exposed with the
    // `uint3` type, and we'll cache a pointer to that
    // type to void looking it up repeatedly.
    //
    IRType* uint3Type = nullptr;

    // Scans through and returns the first typeLayout attribute of non-zero size.
    static LayoutResourceKind getLayoutResourceKind(IRTypeLayout* typeLayout)
    {
        for (auto attr : typeLayout->getSizeAttrs())
        {
            if (attr->getSize() != 0)
                return attr->getResourceKind();
        }
        return LayoutResourceKind::None;
    }

    IRInst* emitOptiXAttributeFetch(
        int& ioBaseAttributeIndex,
        IRType* typeToFetch,
        IRBuilder* builder)
    {
        if (auto ptrValType = tryGetPointedToType(builder, typeToFetch))
            typeToFetch = ptrValType;
        if (auto structType = as<IRStructType>(typeToFetch))
        {
            List<IRInst*> fieldVals;
            for (auto field : structType->getFields())
            {
                auto fieldType = field->getFieldType();
                auto fieldVal = emitOptiXAttributeFetch(ioBaseAttributeIndex, fieldType, builder);
                if (!fieldVal)
                    return nullptr;

                fieldVals.add(fieldVal);
            }
            return builder->emitMakeStruct(typeToFetch, fieldVals);
        }
        else if (auto arrayType = as<IRArrayTypeBase>(typeToFetch))
        {
            auto elementCountInst = as<IRIntLit>(arrayType->getElementCount());
            IRIntegerValue elementCount = elementCountInst->getValue();
            auto elementType = arrayType->getElementType();
            List<IRInst*> elementVals;
            for (IRIntegerValue ii = 0; ii < elementCount; ++ii)
            {
                auto elementVal =
                    emitOptiXAttributeFetch(ioBaseAttributeIndex, elementType, builder);
                if (!elementVal)
                    return nullptr;
                elementVals.add(elementVal);
            }
            return builder->emitMakeArray(
                typeToFetch,
                elementVals.getCount(),
                elementVals.getBuffer());
        }
        else if (auto matType = as<IRMatrixType>(typeToFetch))
        {
            auto rowCountInst = as<IRIntLit>(matType->getRowCount());
            if (rowCountInst)
            {
                auto rowType =
                    builder->getVectorType(matType->getElementType(), matType->getColumnCount());
                IRType* elementType = rowType;
                IRIntegerValue elementCount = rowCountInst->getValue();
                List<IRInst*> elementVals;
                for (IRIntegerValue ii = 0; ii < elementCount; ++ii)
                {
                    auto elementVal =
                        emitOptiXAttributeFetch(ioBaseAttributeIndex, elementType, builder);
                    if (!elementVal)
                        return nullptr;
                    elementVals.add(elementVal);
                }
                return builder->emitIntrinsicInst(
                    typeToFetch,
                    kIROp_MakeMatrix,
                    elementVals.getCount(),
                    elementVals.getBuffer());
            }
        }
        else if (auto vecType = as<IRVectorType>(typeToFetch))
        {
            auto elementCountInst = as<IRIntLit>(vecType->getElementCount());
            IRIntegerValue elementCount = elementCountInst->getValue();
            IRType* elementType = vecType->getElementType();
            List<IRInst*> elementVals;
            for (IRIntegerValue ii = 0; ii < elementCount; ++ii)
            {
                auto elementVal =
                    emitOptiXAttributeFetch(ioBaseAttributeIndex, elementType, builder);
                if (!elementVal)
                    return nullptr;
                elementVals.add(elementVal);
            }
            return builder->emitMakeVector(
                typeToFetch,
                elementVals.getCount(),
                elementVals.getBuffer());
        }
        else if (const auto basicType = as<IRBasicType>(typeToFetch))
        {
            IRIntegerValue idx = ioBaseAttributeIndex;
            auto idxInst = builder->getIntValue(builder->getIntType(), idx);
            ioBaseAttributeIndex++;
            IRInst* args[] = {typeToFetch, idxInst};
            IRInst* getAttr =
                builder->emitIntrinsicInst(typeToFetch, kIROp_GetOptiXHitAttribute, 2, args);
            return getAttr;
        }

        return nullptr;
    }

    void beginModuleImpl() SLANG_OVERRIDE
    {
        // Because many of the varying parameters are defined
        // as magic globals in CUDA, we can introduce their
        // definitions once per module, instead of once per
        // entry point.
        //
        IRBuilder builder(m_module);
        builder.setInsertInto(m_module->getModuleInst());

        // We begin by looking up the `uint` and `uint3` types.
        //
        auto uintType = builder.getBasicType(BaseType::UInt);
        uint3Type = builder.getVectorType(uintType, builder.getIntValue(builder.getIntType(), 3));

        // Next we create IR type and variable layouts that
        // we can use to mark the global parameters like
        // `threadIdx` as varying parameters instead of
        // uniform.
        //
        IRTypeLayout::Builder typeLayoutBuilder(&builder);
        typeLayoutBuilder.addResourceUsage(LayoutResourceKind::VaryingInput, 1);
        auto typeLayout = typeLayoutBuilder.build();

        IRVarLayout::Builder varLayoutBuilder(&builder, typeLayout);
        varLayoutBuilder.findOrAddResourceInfo(LayoutResourceKind::VaryingInput);
        auto varLayout = varLayoutBuilder.build();

        // Finaly, we construct global parameters to represent
        // `threadIdx`, `blockIdx`, and `blockDim`.
        //
        // Each of these parameters is given a target-intrinsic
        // decoration that ensures that (1) it will not get a declaration
        // emitted in output code, and (2) it will be referenced
        // by exactly the desired name (with no attempt to generate
        // a unique name).

        threadIdxGlobalParam = builder.createGlobalParam(uint3Type);
        builder.addTargetIntrinsicDecoration(
            threadIdxGlobalParam,
            CapabilitySet::makeEmpty(),
            UnownedTerminatedStringSlice("threadIdx"));
        builder.addLayoutDecoration(threadIdxGlobalParam, varLayout);

        blockIdxGlobalParam = builder.createGlobalParam(uint3Type);
        builder.addTargetIntrinsicDecoration(
            blockIdxGlobalParam,
            CapabilitySet::makeEmpty(),
            UnownedTerminatedStringSlice("blockIdx"));
        builder.addLayoutDecoration(blockIdxGlobalParam, varLayout);

        blockDimGlobalParam = builder.createGlobalParam(uint3Type);
        builder.addTargetIntrinsicDecoration(
            blockDimGlobalParam,
            CapabilitySet::makeEmpty(),
            UnownedTerminatedStringSlice("blockDim"));
        builder.addLayoutDecoration(blockDimGlobalParam, varLayout);
    }

    // While CUDA provides many useful system values
    // as built-in globals, it does not provide the
    // equivalent of `SV_DispatchThreadID` or
    // `SV_GroupIndex` as a built-in.
    //
    // We will instead synthesize those values on
    // entry to each kernel.

    IRInst* groupThreadIndex = nullptr;
    IRInst* dispatchThreadID = nullptr;
    void beginEntryPointImpl() SLANG_OVERRIDE
    {
        IRBuilder builder(m_module);
        builder.setInsertBefore(m_firstOrdinaryInst);

        // Note that we can use the built-in `blockDim`
        // variable to determine the group extents,
        // instead of inspecting the `[numthreads(...)]`
        // attribute.
        //
        // This choice makes our output more idomatic
        // as CUDA code, but might also cost a small
        // amount of performance by not folding in
        // the known constant values from `numthreads`.
        //
        // TODO: Add logic to use the values from
        // `numthreads` if it is present, but to fall
        // back to `blockDim` if not?

        dispatchThreadID = emitCalcDispatchThreadID(
            builder,
            uint3Type,
            blockIdxGlobalParam,
            threadIdxGlobalParam,
            blockDimGlobalParam);

        groupThreadIndex = emitCalcGroupIndex(builder, threadIdxGlobalParam, blockDimGlobalParam);

        // Note: we don't pay attention to whether the
        // kernel actually makes use of either of these
        // system values when we synthesize them.
        //
        // We can get away with this because we know
        // that subsequent DCE passes will eliminate
        // the computations if they aren't used.
        //
        // The main alternative would be to compute
        // these values lazily, when they are first
        // referenced. While that is possible, it
        // requires more (and more subtle) code in this pass.
    }

    LegalizedVaryingVal createLegalSystemVaryingValImpl(VaryingParamInfo const& info) SLANG_OVERRIDE
    {
        // Because all of the relevant values are either
        // ambiently available in CUDA, or were computed
        // eagerly in the entry block to the kernel
        // function, we can easily return the right
        // value to use for a system-value parameter.

        switch (info.systemValueSemanticName)
        {
        case SystemValueSemanticName::GroupID:
            return createLegalizedVal(info, blockIdxGlobalParam);
        case SystemValueSemanticName::GroupThreadID:
            return createLegalizedVal(info, threadIdxGlobalParam);
        case SystemValueSemanticName::GroupIndex:
            return createLegalizedVal(info, groupThreadIndex);
        case SystemValueSemanticName::DispatchThreadID:
            return createLegalizedVal(info, dispatchThreadID);
        default:
            return diagnoseUnsupportedSystemVal(info);
        }
    }

    LegalizedVaryingVal createLegalUserVaryingValImpl(VaryingParamInfo const& info) SLANG_OVERRIDE
    {
        auto layoutResourceKind = getLayoutResourceKind(info.typeLayout);
        switch (layoutResourceKind)
        {
        case LayoutResourceKind::RayPayload:
            {
                IRBuilder builder(m_module);
                builder.setInsertBefore(m_firstOrdinaryInst);
                IRPtrType* ptrType = builder.getPtrType(info.type);
                IRInst* getRayPayload =
                    builder.emitIntrinsicInst(ptrType, kIROp_GetOptiXRayPayloadPtr, 0, nullptr);
                return LegalizedVaryingVal::makeAddress(getRayPayload);
                // Todo: compute how many registers are required for the current payload.
                // If more than 32, use the above logic.
                // Otherwise, either use the optix_get_payload or optix_set_payload
                // intrinsics depending on input/output
                /*if (info.kind == LayoutResourceKind::VaryingInput) {
                }
                else if (info.kind == LayoutResourceKind::VaryingOutput) {
                }
                else {
                    return diagnoseUnsupportedUserVal(info);
                }*/
            }
        case LayoutResourceKind::HitAttributes:
            {
                IRBuilder builder(m_module);
                builder.setInsertBefore(m_firstOrdinaryInst);
                int ioBaseAttributeIndex = 0;
                IRInst* getHitAttributes = emitOptiXAttributeFetch(
                    /*ioBaseAttributeIndex*/ ioBaseAttributeIndex,
                    /* type to fetch */ info.type,
                    /*the builder in use*/ &builder);
                if (ioBaseAttributeIndex > 8)
                {
                    m_sink->diagnose(
                        m_param,
                        Diagnostics::unexpected,
                        "the supplied hit attribute exceeds the maximum hit attribute structure "
                        "size (32 bytes)");
                    return LegalizedVaryingVal();
                }
                return LegalizedVaryingVal::makeValue(getHitAttributes);
            }
        default:
            return diagnoseUnsupportedUserVal(info);
        }
    }

    LegalizedVaryingVal createLegalizedVal(VaryingParamInfo const& info, IRInst* id)
    {
        // If the parameter type is not uint3, we need to extract components as needed
        auto paramType = info.type->getOperand(0);
        IRBuilder builder(m_module);
        builder.setInsertBefore(m_firstOrdinaryInst);

        if (as<IRBasicType>(paramType))
        {
            auto uintType = builder.getBasicType(BaseType::UInt);
            UInt swizzleIndex = 0;
            auto xComponent = builder.emitSwizzle(uintType, id, 1, &swizzleIndex);

            if (auto basicType = as<IRBasicType>(paramType))
            {
                if (basicType->getBaseType() != BaseType::UInt)
                {
                    xComponent = builder.emitBitCast(basicType, xComponent);
                }
            }
            return LegalizedVaryingVal::makeValue(xComponent);
        }
        // For vector types, use a swizzle to extract the needed components
        else if (auto vectorType = as<IRVectorType>(paramType))
        {
            auto elementCount = getIntVal(vectorType->getElementCount());

            if (elementCount > 0 && elementCount <= 3)
            {
                // Setup indices for the swizzle (0 for x, 1 for y, 2 for z)
                UInt swizzleIndices[3] = {0, 1, 2};
                auto uintType = builder.getBasicType(BaseType::UInt);

                // Use a swizzle to extract all needed components at once
                auto extractedVector = builder.emitSwizzle(
                    builder.getVectorType(uintType, elementCount),
                    id,
                    elementCount,
                    swizzleIndices);

                // Cast if the element type is not uint
                auto elementType = vectorType->getElementType();
                if (auto basicElementType = as<IRBasicType>(elementType))
                {
                    if (basicElementType->getBaseType() != BaseType::UInt)
                    {
                        extractedVector = builder.emitBitCast(vectorType, extractedVector);
                    }
                }
                return LegalizedVaryingVal::makeValue(extractedVector);
            }
        }
        // Default to the full uint3 if the parameter type doesn't match our expectations
        return LegalizedVaryingVal::makeValue(id);
    }
};


struct CPUEntryPointVaryingParamLegalizeContext : EntryPointVaryingParamLegalizeContext
{
    // Slang translates compute shaders for CPU such that they always have an
    // initial parameter that is a `ComputeThreadVaryingInput*`, and that
    // type provides the essential parameters (`SV_GroupID` and `SV_GroupThreadID`
    // as fields).
    //
    // Our legalization pass for CPU this begins with the per-module logic
    // to synthesize an IR definition of that type and its fields, so that
    // we can use it across entry points.

    IRType* uintType = nullptr;
    IRVectorType* uint3Type = nullptr;
    IRType* uint3PtrType = nullptr;

    IRStructType* varyingInputStructType = nullptr;
    IRPtrType* varyingInputStructPtrType = nullptr;

    IRStructKey* groupIDKey = nullptr;
    IRStructKey* groupThreadIDKey = nullptr;

    void beginModuleImpl() SLANG_OVERRIDE
    {
        IRBuilder builder(m_module);
        builder.setInsertInto(m_module->getModuleInst());

        uintType = builder.getBasicType(BaseType::UInt);
        uint3Type = builder.getVectorType(uintType, builder.getIntValue(builder.getIntType(), 3));
        uint3PtrType = builder.getPtrType(uint3Type);

        // As we construct the `ComputeThreadVaryingInput` type and its fields,
        // we mark them all as target intrinsics, which means that their
        // declarations will *not* be reproduced in the output code, instead
        // coming from the "prelude" file that already defines this type.

        varyingInputStructType = builder.createStructType();
        varyingInputStructPtrType = builder.getPtrType(varyingInputStructType);

        builder.addTargetIntrinsicDecoration(
            varyingInputStructType,
            CapabilitySet::makeEmpty(),
            UnownedTerminatedStringSlice("ComputeThreadVaryingInput"));

        groupIDKey = builder.createStructKey();
        builder.addTargetIntrinsicDecoration(
            groupIDKey,
            CapabilitySet::makeEmpty(),
            UnownedTerminatedStringSlice("groupID"));
        builder.createStructField(varyingInputStructType, groupIDKey, uint3Type);

        groupThreadIDKey = builder.createStructKey();
        builder.addTargetIntrinsicDecoration(
            groupThreadIDKey,
            CapabilitySet::makeEmpty(),
            UnownedTerminatedStringSlice("groupThreadID"));
        builder.createStructField(varyingInputStructType, groupThreadIDKey, uint3Type);
    }

    // While the declaration of the `ComputeVaryingThreadInput` type
    // can be shared across all entry points, each entry point must
    // declare its own parameter to receive the varying parameters.
    //
    // We will extract the relevant fields from the `ComputeVaryingThreadInput`
    // at the start of kernel execution (rather than repeatedly load them
    // at each use site), and will also eagerly compute the derived
    // values for `SV_DispatchThreadID` and `SV_GroupIndex`.

    IRInst* groupID = nullptr;
    IRInst* groupThreadID = nullptr;
    IRInst* groupExtents = nullptr;
    IRInst* dispatchThreadID = nullptr;
    IRInst* groupThreadIndex = nullptr;

    void beginEntryPointImpl() SLANG_OVERRIDE
    {
        groupID = nullptr;
        groupThreadID = nullptr;
        dispatchThreadID = nullptr;

        IRBuilder builder(m_module);

        auto varyingInputParam = builder.createParam(varyingInputStructPtrType);
        varyingInputParam->insertBefore(m_firstBlock->getFirstChild());

        builder.setInsertBefore(m_firstOrdinaryInst);

        groupID =
            builder.emitLoad(builder.emitFieldAddress(uint3PtrType, varyingInputParam, groupIDKey));

        groupThreadID = builder.emitLoad(
            builder.emitFieldAddress(uint3PtrType, varyingInputParam, groupThreadIDKey));

        // Note: we need to rely on the presence of the `[numthreads(...)]` attribute
        // to tell us the size of the compute thread group, which we will then use
        // when computing the dispatch thread ID and group thread index.
        //
        // TODO: If we ever wanted to support flexible thread-group sizes for our
        // CPU target, we'd need to change it so that the thread-group size can
        // be passed in as part of `ComputeVaryingThreadInput`.
        //
        groupExtents = emitCalcGroupExtents(builder, m_entryPointFunc, uint3Type);

        if (!groupExtents)
        {
            m_sink->diagnose(
                m_entryPointFunc,
                Diagnostics::unsupportedSpecializationConstantForNumThreads);

            // Fill in placeholder values.
            static const int kAxisCount = 3;
            IRInst* groupExtentAlongAxis[kAxisCount] = {};
            for (int axis = 0; axis < kAxisCount; axis++)
                groupExtentAlongAxis[axis] = builder.getIntValue(uint3Type->getElementType(), 1);
            groupExtents = builder.emitMakeVector(uint3Type, kAxisCount, groupExtentAlongAxis);
        }

        dispatchThreadID =
            emitCalcDispatchThreadID(builder, uint3Type, groupID, groupThreadID, groupExtents);

        groupThreadIndex = emitCalcGroupIndex(builder, groupThreadID, groupExtents);
    }

    LegalizedVaryingVal createLegalSystemVaryingValImpl(VaryingParamInfo const& info) SLANG_OVERRIDE
    {
        // Because all of the relvant system values were synthesized
        // into the first block of the entry-point function, we can
        // just return them wherever they are referenced.
        //
        // Note that any values that were synthesized but then are
        // not referened will simply be eliminated as dead code
        // in later passes.

        switch (info.systemValueSemanticName)
        {
        case SystemValueSemanticName::GroupID:
            return LegalizedVaryingVal::makeValue(groupID);
        case SystemValueSemanticName::GroupThreadID:
            return LegalizedVaryingVal::makeValue(groupThreadID);
        case SystemValueSemanticName::GroupIndex:
            return LegalizedVaryingVal::makeValue(groupThreadIndex);
        case SystemValueSemanticName::DispatchThreadID:
            return LegalizedVaryingVal::makeValue(dispatchThreadID);

        default:
            return diagnoseUnsupportedSystemVal(info);
        }
    }
};

void legalizeEntryPointVaryingParamsForCPU(IRModule* module, DiagnosticSink* sink)
{
    CPUEntryPointVaryingParamLegalizeContext context;
    context.processModule(module, sink);
}

void legalizeEntryPointVaryingParamsForCUDA(IRModule* module, DiagnosticSink* sink)
{
    CUDAEntryPointVaryingParamLegalizeContext context;
    context.processModule(module, sink);
}

void depointerizeInputParams(IRFunc* entryPointFunc)
{
    List<IRParam*> workList;
    List<Index> modifiedParamIndices;
    Index i = 0;
    for (auto param : entryPointFunc->getParams())
    {
        if (auto constRefType = as<IRConstRefType>(param->getFullType()))
        {
            switch (constRefType->getValueType()->getOp())
            {
            case kIROp_VerticesType:
            case kIROp_IndicesType:
            case kIROp_PrimitivesType:
                continue;
            default:
                break;
            }
            workList.add(param);
            modifiedParamIndices.add(i);
        }
        else if (auto ptrType = as<IRPtrTypeBase>(param->getFullType()))
        {
            switch (ptrType->getAddressSpace())
            {
            case AddressSpace::Input:
            case AddressSpace::BuiltinInput:
                workList.add(param);
                modifiedParamIndices.add(i);
                break;
            }
        }
        i++;
    }
    for (auto param : workList)
    {
        auto valueType = as<IRPtrTypeBase>(param->getDataType())->getValueType();
        IRBuilder builder(param);
        setInsertBeforeOrdinaryInst(&builder, param);
        auto var = builder.emitVar(valueType);
        param->replaceUsesWith(var);
        param->setFullType(valueType);
        builder.emitStore(var, param);
    }

    fixUpFuncType(entryPointFunc);

    // Fix up callsites of the entrypoint func.
    for (auto use = entryPointFunc->firstUse; use; use = use->nextUse)
    {
        auto call = as<IRCall>(use->getUser());
        if (!call)
            continue;
        IRBuilder builder(call);
        builder.setInsertBefore(call);
        for (auto paramIndex : modifiedParamIndices)
        {
            auto arg = call->getArg(paramIndex);
            auto ptrType = as<IRPtrTypeBase>(arg->getDataType());
            if (!ptrType)
                continue;
            auto val = builder.emitLoad(arg);
            call->setArg(paramIndex, val);
        }
    }
}


class LegalizeShaderEntryPointContext
{
public:
    void legalizeEntryPoints(List<EntryPointInfo>& entryPoints)
    {
        for (auto entryPoint : entryPoints)
            legalizeEntryPoint(entryPoint);
        removeSemanticLayoutsFromLegalizedStructs();
    }

protected:
    LegalizeShaderEntryPointContext(IRModule* module, DiagnosticSink* sink)
        : m_module(module), m_sink(sink)
    {
    }

    IRModule* m_module;
    DiagnosticSink* m_sink;

    struct SystemValueInfo
    {
        String systemValueName;
        SystemValueSemanticName systemValueNameEnum;
        ShortList<IRType*> permittedTypes;

        bool isUnsupported = false;
        bool isSpecial = false;
    };

    struct SystemValLegalizationWorkItem
    {
        IRInst* var;
        IRType* varType;

        String attrName;
        UInt attrIndex;
    };

    virtual SystemValueInfo getSystemValueInfo(
        String inSemanticName,
        String* optionalSemanticIndex,
        IRInst* parentVar) const = 0;

    virtual List<SystemValLegalizationWorkItem> collectSystemValFromEntryPoint(
        EntryPointInfo entryPoint) const = 0;

    virtual void flattenNestedStructsTransferKeyDecorations(IRInst* newKey, IRInst* oldKey)
        const = 0;

    virtual UnownedStringSlice getUserSemanticNameSlice(String& loweredName, bool isUserSemantic)
        const = 0;

    virtual void addFragmentShaderReturnValueDecoration(
        IRBuilder& builder,
        IRInst* returnValueStructKey) const = 0;


    virtual IRVarLayout* handleGeometryStageParameterVarLayout(
        IRBuilder& builder,
        IRVarLayout* paramVarLayout) const
    {
        SLANG_UNUSED(builder);
        return paramVarLayout;
    }

    virtual void handleSpecialSystemValue(
        const EntryPointInfo& entryPoint,
        SystemValLegalizationWorkItem& workItem,
        const SystemValueInfo& info,
        IRBuilder& builder)
    {
        SLANG_UNUSED(entryPoint);
        SLANG_UNUSED(workItem);
        SLANG_UNUSED(info);
        SLANG_UNUSED(builder);
    }

    virtual void legalizeAmplificationStageEntryPoint(const EntryPointInfo& entryPoint) const
    {
        SLANG_UNUSED(entryPoint);
    }

    virtual void legalizeMeshStageEntryPoint(const EntryPointInfo& entryPoint) const
    {
        SLANG_UNUSED(entryPoint);
    }


    std::optional<SystemValLegalizationWorkItem> tryToMakeSystemValWorkItem(
        IRInst* var,
        IRType* varType) const
    {
        if (auto semanticDecoration = var->findDecoration<IRSemanticDecoration>())
        {
            if (semanticDecoration->getSemanticName().startsWithCaseInsensitive(toSlice("sv_")))
            {
                return {
                    {var,
                     varType,
                     String(semanticDecoration->getSemanticName()).toLower(),
                     (UInt)semanticDecoration->getSemanticIndex()}};
            }
        }

        auto layoutDecor = var->findDecoration<IRLayoutDecoration>();
        if (!layoutDecor)
            return {};
        auto sysValAttr = layoutDecor->findAttr<IRSystemValueSemanticAttr>();
        if (!sysValAttr)
            return {};
        auto semanticName = String(sysValAttr->getName());
        auto sysAttrIndex = sysValAttr->getIndex();

        return {{var, varType, semanticName, sysAttrIndex}};
    }

    void legalizeSystemValue(EntryPointInfo entryPoint, SystemValLegalizationWorkItem& workItem)
    {
        IRBuilder builder(entryPoint.entryPointFunc);

        auto var = workItem.var;
        auto varType = workItem.varType;
        auto semanticName = workItem.attrName;

        auto indexAsString = String(workItem.attrIndex);
        SystemValueInfo info = getSystemValueInfo(semanticName, &indexAsString, var);
        if (info.isSpecial)
        {
            handleSpecialSystemValue(entryPoint, workItem, info, builder);
        }

        if (info.isUnsupported)
        {
            reportUnsupportedSystemAttribute(var, semanticName);
            return;
        }
        if (!info.permittedTypes.getCount())
            return;

        builder.addTargetSystemValueDecoration(var, info.systemValueName.getUnownedSlice());

        bool varTypeIsPermitted = false;
        for (auto& permittedType : info.permittedTypes)
        {
            varTypeIsPermitted = varTypeIsPermitted || permittedType == varType;
        }

        if (!varTypeIsPermitted)
        {
            // Note: we do not currently prefer any conversion
            // example:
            // * allowed types for semantic: `float4`, `uint4`, `int4`
            // * user used, `float2`
            // * Slang will equally prefer `float4` to `uint4` to `int4`.
            //   This means the type may lose data if slang selects `uint4` or `int4`.
            bool foundAConversion = false;
            for (auto permittedType : info.permittedTypes)
            {
                var->setFullType(permittedType);
                builder.setInsertBefore(
                    entryPoint.entryPointFunc->getFirstBlock()->getFirstOrdinaryInst());

                // get uses before we `tryConvertValue` since this creates a new use
                List<IRUse*> uses;
                for (auto use = var->firstUse; use; use = use->nextUse)
                    uses.add(use);

                auto convertedValue = tryConvertValue(builder, var, varType);
                if (convertedValue == nullptr)
                    continue;

                foundAConversion = true;
                copyNameHintAndDebugDecorations(convertedValue, var);

                for (auto use : uses)
                    builder.replaceOperand(use, convertedValue);
            }
            if (!foundAConversion)
            {
                // If we can't convert the value, report an error.
                for (auto permittedType : info.permittedTypes)
                {
                    StringBuilder typeNameSB;
                    getTypeNameHint(typeNameSB, permittedType);
                    m_sink->diagnose(
                        var->sourceLoc,
                        Diagnostics::systemValueTypeIncompatible,
                        semanticName,
                        typeNameSB.produceString());
                }
            }
        }
    }

private:
    HashSet<IRStructField*> semanticInfoToRemove;

    void removeSemanticLayoutsFromLegalizedStructs()
    {
        // Metal and WGSL does not allow duplicate attributes to appear in the same shader.
        // If we emit our own struct with `[[color(0)]]`, all existing uses of `[[color(0)]]`
        // must be removed.
        for (auto field : semanticInfoToRemove)
        {
            auto key = field->getKey();
            // Some decorations appear twice, destroy all found
            for (;;)
            {
                if (auto semanticDecor = key->findDecoration<IRSemanticDecoration>())
                {
                    semanticDecor->removeAndDeallocate();
                    continue;
                }
                else if (auto layoutDecor = key->findDecoration<IRLayoutDecoration>())
                {
                    layoutDecor->removeAndDeallocate();
                    continue;
                }
                break;
            }
        }
    }

    void hoistEntryPointParameterFromStruct(EntryPointInfo entryPoint)
    {
        // If an entry point has a input parameter with a struct type, we want to hoist out
        // all the fields of the struct type to be individual parameters of the entry point.
        // This will canonicalize the entry point signature, so we can handle all cases uniformly.

        // For example, given an entry point:
        // ```
        // struct VertexInput { float3 pos; float 2 uv; int vertexId : SV_VertexID};
        // void main(VertexInput vin) { ... }
        // ```
        // We will transform it to:
        // ```
        // void main(float3 pos, float2 uv, int vertexId : SV_VertexID) {
        //     VertexInput vin = {pos,uv,vertexId};
        //     ...
        // }
        // ```

        auto func = entryPoint.entryPointFunc;
        List<IRParam*> paramsToProcess;
        for (auto param : func->getParams())
        {
            if (as<IRStructType>(param->getDataType()))
            {
                paramsToProcess.add(param);
            }
        }

        IRBuilder builder(func);
        builder.setInsertBefore(func);
        for (auto param : paramsToProcess)
        {
            auto structType = as<IRStructType>(param->getDataType());
            builder.setInsertBefore(func->getFirstBlock()->getFirstOrdinaryInst());
            auto varLayout = findVarLayout(param);

            // If `param` already has a semantic, we don't want to hoist its fields out.
            if (varLayout->findSystemValueSemanticAttr() != nullptr ||
                param->findDecoration<IRSemanticDecoration>())
                continue;

            IRStructTypeLayout* structTypeLayout = nullptr;
            if (varLayout)
                structTypeLayout = as<IRStructTypeLayout>(varLayout->getTypeLayout());
            Index fieldIndex = 0;
            List<IRInst*> fieldParams;
            for (auto field : structType->getFields())
            {
                auto fieldParam = builder.emitParam(field->getFieldType());
                IRCloneEnv cloneEnv;
                cloneInstDecorationsAndChildren(
                    &cloneEnv,
                    builder.getModule(),
                    field->getKey(),
                    fieldParam);

                IRVarLayout* fieldLayout =
                    structTypeLayout ? structTypeLayout->getFieldLayout(fieldIndex) : nullptr;
                if (varLayout)
                {
                    IRVarLayout::Builder varLayoutBuilder(&builder, fieldLayout->getTypeLayout());
                    varLayoutBuilder.cloneEverythingButOffsetsFrom(fieldLayout);
                    for (auto offsetAttr : fieldLayout->getOffsetAttrs())
                    {
                        auto parentOffsetAttr =
                            varLayout->findOffsetAttr(offsetAttr->getResourceKind());
                        UInt parentOffset = parentOffsetAttr ? parentOffsetAttr->getOffset() : 0;
                        UInt parentSpace = parentOffsetAttr ? parentOffsetAttr->getSpace() : 0;
                        auto resInfo =
                            varLayoutBuilder.findOrAddResourceInfo(offsetAttr->getResourceKind());
                        resInfo->offset = parentOffset + offsetAttr->getOffset();
                        resInfo->space = parentSpace + offsetAttr->getSpace();
                    }
                    builder.addLayoutDecoration(fieldParam, varLayoutBuilder.build());
                }
                param->insertBefore(fieldParam);
                fieldParams.add(fieldParam);
                fieldIndex++;
            }
            builder.setInsertBefore(func->getFirstBlock()->getFirstOrdinaryInst());
            auto reconstructedParam =
                builder.emitMakeStruct(structType, fieldParams.getCount(), fieldParams.getBuffer());
            param->replaceUsesWith(reconstructedParam);
            param->removeFromParent();
        }
        fixUpFuncType(func);
    }

    // Flattens all struct parameters of an entryPoint to ensure parameters are a flat struct
    void flattenInputParameters(EntryPointInfo entryPoint)
    {
        // Goal is to ensure we have a flattened IRParam (0 nested IRStructType members).
        /*
            // Assume the following code
            struct NestedFragment
            {
                float2 p3;
            };
            struct Fragment
            {
                float4 p1;
                float3 p2;
                NestedFragment p3_nested;
            };

            // Fragment flattens into
            struct Fragment
            {
                float4 p1;
                float3 p2;
                float2 p3;
            };
        */

        // This is important since Metal and WGSL does not allow semantic's on a struct
        /*
            // Assume the following code
            struct NestedFragment1
            {
                float2 p3;
            };
            struct Fragment1
            {
                float4 p1 : SV_TARGET0;
                float3 p2 : SV_TARGET1;
                NestedFragment p3_nested : SV_TARGET2; // error, semantic on struct
            };

        */

        // Metal does allow semantics on members of a nested struct but we are avoiding this
        // approach since there are senarios where legalization (and verification) is
        // hard/expensive without creating a flat struct:
        // 1. Entry points may share structs, semantics may be inconsistent across entry points
        // 2. Multiple of the same struct may be used in a param list
        //
        // WGSL does NOT allow semantics on members of a nested struct.
        /*
            // Assume the following code
            struct NestedFragment
            {
                float2 p3;
            };
            struct Fragment
            {
                float4 p1 : SV_TARGET0;
                NestedFragment p2 : SV_TARGET1;
                NestedFragment p3 : SV_TARGET2;
            };

            // Legalized without flattening -- abandoned
            struct NestedFragment1
            {
                float2 p3 : SV_TARGET1;
            };
            struct NestedFragment2
            {
                float2 p3 : SV_TARGET2;
            };
            struct Fragment
            {
                float4 p1 : SV_TARGET0;
                NestedFragment1 p2;
                NestedFragment2 p3;
            };

            // Legalized with flattening -- current approach
            struct Fragment
            {
                float4 p1 : SV_TARGET0;
                float2 p2 : SV_TARGET1;
                float2 p3 : SV_TARGET2;
            };
        */

        auto func = entryPoint.entryPointFunc;
        bool modified = false;
        for (auto param : func->getParams())
        {
            auto layout = findVarLayout(param);
            if (!layout)
                continue;
            if (!layout->findOffsetAttr(LayoutResourceKind::VaryingInput))
                continue;
            if (param->findDecorationImpl(kIROp_HLSLMeshPayloadDecoration))
                continue;
            // If we find a IRParam with a IRStructType member, we need to flatten the entire
            // IRParam
            if (auto structType = as<IRStructType>(param->getDataType()))
            {
                IRBuilder builder(func);
                MapStructToFlatStruct mapOldFieldToNewField;

                // Flatten struct if we have nested IRStructType
                auto flattenedStruct = maybeFlattenNestedStructs(
                    builder,
                    structType,
                    mapOldFieldToNewField,
                    semanticInfoToRemove);

                // Validate/rearange all semantics which overlap in our flat struct.
                fixFieldSemanticsOfFlatStruct(flattenedStruct);
                ensureStructHasUserSemantic<LayoutResourceKind::VaryingInput>(
                    flattenedStruct,
                    layout);
                if (flattenedStruct != structType)
                {
                    // Replace the 'old IRParam type' with a 'new IRParam type'
                    param->setFullType(flattenedStruct);

                    // Emit a new variable at EntryPoint of 'old IRParam type'
                    builder.setInsertBefore(func->getFirstBlock()->getFirstOrdinaryInst());
                    auto dstVal = builder.emitVar(structType);
                    auto dstLoad = builder.emitLoad(dstVal);
                    param->replaceUsesWith(dstLoad);
                    builder.setInsertBefore(dstLoad);
                    // Copy the 'new IRParam type' to our 'old IRParam type'
                    mapOldFieldToNewField
                        .emitCopy<(int)MapStructToFlatStruct::CopyOptions::FlatStructIntoStruct>(
                            builder,
                            dstVal,
                            param);

                    modified = true;
                }
            }
        }
        if (modified)
            fixUpFuncType(func);
    }

    void packStageInParameters(EntryPointInfo entryPoint)
    {
        // If the entry point has any parameters whose layout contains VaryingInput,
        // we need to pack those parameters into a single `struct` type, and decorate
        // the fields with the appropriate `[[attribute]]` decorations.
        // For other parameters that are not `VaryingInput`, we need to leave them as is.
        //
        // For example, given this code after `hoistEntryPointParameterFromStruct`:
        // ```
        // void main(float3 pos, float2 uv, int vertexId : SV_VertexID) {
        //     VertexInput vin = {pos,uv,vertexId};
        //     ...
        // }
        // ```
        // We are going to transform it into:
        // ```
        // struct VertexInput {
        //     float3 pos [[attribute(0)]];
        //     float2 uv [[attribute(1)]];
        // };
        // void main(VertexInput vin, int vertexId : SV_VertexID) {
        //     let pos = vin.pos;
        //     let uv = vin.uv;
        //     ...
        // }

        auto func = entryPoint.entryPointFunc;

        bool isGeometryStage = false;
        switch (entryPoint.entryPointDecor->getProfile().getStage())
        {
        case Stage::Vertex:
        case Stage::Amplification:
        case Stage::Mesh:
        case Stage::Geometry:
        case Stage::Domain:
        case Stage::Hull:
            isGeometryStage = true;
            break;
        }

        List<IRParam*> paramsToPack;
        for (auto param : func->getParams())
        {
            auto layout = findVarLayout(param);
            if (!layout)
                continue;
            if (!layout->findOffsetAttr(LayoutResourceKind::VaryingInput))
                continue;
            if (param->findDecorationImpl(kIROp_HLSLMeshPayloadDecoration))
                continue;
            paramsToPack.add(param);
        }

        if (paramsToPack.getCount() == 0)
            return;

        IRBuilder builder(func);
        builder.setInsertBefore(func);
        IRStructType* structType = builder.createStructType();
        auto stageText = getStageText(entryPoint.entryPointDecor->getProfile().getStage());
        builder.addNameHintDecoration(
            structType,
            (String(stageText) + toSlice("Input")).getUnownedSlice());
        List<IRStructKey*> keys;
        IRStructTypeLayout::Builder layoutBuilder(&builder);
        for (auto param : paramsToPack)
        {
            auto paramVarLayout = findVarLayout(param);
            auto key = builder.createStructKey();
            param->transferDecorationsTo(key);
            builder.createStructField(structType, key, param->getDataType());
            if (auto varyingInOffsetAttr =
                    paramVarLayout->findOffsetAttr(LayoutResourceKind::VaryingInput))
            {
                if (!key->findDecoration<IRSemanticDecoration>() &&
                    !paramVarLayout->findAttr<IRSemanticAttr>())
                {
                    // If the parameter doesn't have a semantic, we need to add one for semantic
                    // matching.
                    builder.addSemanticDecoration(
                        key,
                        toSlice("_slang_attr"),
                        (int)varyingInOffsetAttr->getOffset());
                }
            }

            if (isGeometryStage)
            {
                paramVarLayout = handleGeometryStageParameterVarLayout(builder, paramVarLayout);
            }

            layoutBuilder.addField(key, paramVarLayout);
            builder.addLayoutDecoration(key, paramVarLayout);
            keys.add(key);
        }
        builder.setInsertInto(func->getFirstBlock());
        auto packedParam = builder.emitParamAtHead(structType);
        auto typeLayout = layoutBuilder.build();
        IRVarLayout::Builder varLayoutBuilder(&builder, typeLayout);

        // Add a VaryingInput resource info to the packed parameter layout, so that we can emit
        // the needed `[[stage_in]]` attribute in Metal emitter.
        varLayoutBuilder.findOrAddResourceInfo(LayoutResourceKind::VaryingInput);
        auto paramVarLayout = varLayoutBuilder.build();
        builder.addLayoutDecoration(packedParam, paramVarLayout);

        // Replace the original parameters with the packed parameter
        builder.setInsertBefore(func->getFirstBlock()->getFirstOrdinaryInst());
        for (Index paramIndex = 0; paramIndex < paramsToPack.getCount(); paramIndex++)
        {
            auto param = paramsToPack[paramIndex];
            auto key = keys[paramIndex];
            auto paramField = builder.emitFieldExtract(param->getDataType(), packedParam, key);
            param->replaceUsesWith(paramField);
            param->removeFromParent();
        }
        fixUpFuncType(func);
    }


    void reportUnsupportedSystemAttribute(IRInst* param, String semanticName)
    {
        m_sink->diagnose(
            param->sourceLoc,
            Diagnostics::systemValueAttributeNotSupported,
            semanticName);
    }

    template<LayoutResourceKind K>
    void ensureStructHasUserSemantic(IRStructType* structType, IRVarLayout* varLayout)
    {
        // Ensure each field in an output struct type has either a system semantic or a user
        // semantic, so that signature matching can happen correctly.
        auto typeLayout = as<IRStructTypeLayout>(varLayout->getTypeLayout());
        Index index = 0;
        IRBuilder builder(structType);
        for (auto field : structType->getFields())
        {
            auto key = field->getKey();
            if (auto semanticDecor = key->findDecoration<IRSemanticDecoration>())
            {
                if (semanticDecor->getSemanticName().startsWithCaseInsensitive(toSlice("sv_")))
                {
                    auto indexAsString = String(UInt(semanticDecor->getSemanticIndex()));
                    auto sysValInfo =
                        getSystemValueInfo(semanticDecor->getSemanticName(), &indexAsString, field);
                    if (sysValInfo.isUnsupported)
                    {
                        reportUnsupportedSystemAttribute(field, semanticDecor->getSemanticName());
                    }
                    else
                    {
                        builder.addTargetSystemValueDecoration(
                            key,
                            sysValInfo.systemValueName.getUnownedSlice());
                        semanticDecor->removeAndDeallocate();
                    }
                }
                index++;
                continue;
            }
            typeLayout->getFieldLayout(index);
            auto fieldLayout = typeLayout->getFieldLayout(index);
            if (auto offsetAttr = fieldLayout->findOffsetAttr(K))
            {
                UInt varOffset = 0;
                if (auto varOffsetAttr = varLayout->findOffsetAttr(K))
                    varOffset = varOffsetAttr->getOffset();
                varOffset += offsetAttr->getOffset();
                builder.addSemanticDecoration(key, toSlice("_slang_attr"), (int)varOffset);
            }
            index++;
        }
    }

    // Stores a hicharchy of members and children which map 'oldStruct->member' to
    // 'flatStruct->member' Note: this map assumes we map to FlatStruct since it is easier/faster to
    // process
    struct MapStructToFlatStruct
    {
        /*
        We need a hicharchy map to resolve dependencies for mapping
        oldStruct to newStruct efficently. Example:

        MyStruct
            |
          / | \
         /  |  \
        /   |   \
    M0<A> M1<A> M2<B>
       |    |    |
      A_0  A_0  B_0

      Without storing hicharchy information, there will be no way to tell apart
      `myStruct.M0.A0` from `myStruct.M1.A0` since IRStructKey/IRStructField
      only has 1 instance of `A::A0`
      */

        enum CopyOptions : int
        {
            // Copy a flattened-struct into a struct
            FlatStructIntoStruct = 0,

            // Copy a struct into a flattened-struct
            StructIntoFlatStruct = 1,
        };

    private:
        // Children of member if applicable.
        Dictionary<IRStructField*, MapStructToFlatStruct> members;

        // Field correlating to MapStructToFlatStruct Node.
        IRInst* node;
        IRStructKey* getKey()
        {
            SLANG_ASSERT(as<IRStructField>(node));
            return as<IRStructField>(node)->getKey();
        }
        IRInst* getNode() { return node; }
        IRType* getFieldType()
        {
            SLANG_ASSERT(as<IRStructField>(node));
            return as<IRStructField>(node)->getFieldType();
        }

        // Whom node maps to inside target flatStruct
        IRStructField* targetMapping;

        auto begin() { return members.begin(); }
        auto end() { return members.end(); }

        // Copies members of oldStruct to/from newFlatStruct. Assumes members of val1 maps to
        // members in val2 using `MapStructToFlatStruct`
        template<int copyOptions>
        static void _emitCopy(
            IRBuilder& builder,
            IRInst* val1,
            IRStructType* type1,
            IRInst* val2,
            IRStructType* type2,
            MapStructToFlatStruct& node)
        {
            for (auto& field1Pair : node)
            {
                auto& field1 = field1Pair.second;

                // Get member of val1
                IRInst* fieldAddr1 = nullptr;
                if constexpr (copyOptions == (int)CopyOptions::FlatStructIntoStruct)
                {
                    fieldAddr1 = builder.emitFieldAddress(type1, val1, field1.getKey());
                }
                else
                {
                    if (as<IRPtrTypeBase>(val1))
                        val1 = builder.emitLoad(val1);
                    fieldAddr1 = builder.emitFieldExtract(type1, val1, field1.getKey());
                }

                // If val1 is a struct, recurse
                if (auto fieldAsStruct1 = as<IRStructType>(field1.getFieldType()))
                {
                    _emitCopy<copyOptions>(
                        builder,
                        fieldAddr1,
                        fieldAsStruct1,
                        val2,
                        type2,
                        field1);
                    continue;
                }

                // Get member of val2 which maps to val1.member
                auto field2 = field1.getMapping();
                SLANG_ASSERT(field2);
                IRInst* fieldAddr2 = nullptr;
                if constexpr (copyOptions == (int)CopyOptions::FlatStructIntoStruct)
                {
                    if (as<IRPtrTypeBase>(val2))
                        val2 = builder.emitLoad(val1);
                    fieldAddr2 = builder.emitFieldExtract(type2, val2, field2->getKey());
                }
                else
                {
                    fieldAddr2 = builder.emitFieldAddress(type2, val2, field2->getKey());
                }

                // Copy val2/val1 member into val1/val2 member
                if constexpr (copyOptions == (int)CopyOptions::FlatStructIntoStruct)
                {
                    builder.emitStore(fieldAddr1, fieldAddr2);
                }
                else
                {
                    builder.emitStore(fieldAddr2, fieldAddr1);
                }
            }
        }

    public:
        void setNode(IRInst* newNode) { node = newNode; }
        // Get 'MapStructToFlatStruct' that is a child of 'parent'.
        // Make 'MapStructToFlatStruct' if no 'member' is currently mapped to 'parent'.
        MapStructToFlatStruct& getMember(IRStructField* member) { return members[member]; }
        MapStructToFlatStruct& operator[](IRStructField* member) { return getMember(member); }

        void setMapping(IRStructField* newTargetMapping) { targetMapping = newTargetMapping; }
        // Get 'MapStructToFlatStruct' that is a child of 'parent'.
        // Return nullptr if no member is mapped to 'parent'
        IRStructField* getMapping() { return targetMapping; }

        // Copies srcVal into dstVal using hicharchy map.
        template<int copyOptions>
        void emitCopy(IRBuilder& builder, IRInst* dstVal, IRInst* srcVal)
        {
            auto dstType = dstVal->getDataType();
            if (auto dstPtrType = as<IRPtrTypeBase>(dstType))
                dstType = dstPtrType->getValueType();
            auto dstStructType = as<IRStructType>(dstType);
            SLANG_ASSERT(dstStructType);

            auto srcType = srcVal->getDataType();
            if (auto srcPtrType = as<IRPtrTypeBase>(srcType))
                srcType = srcPtrType->getValueType();
            auto srcStructType = as<IRStructType>(srcType);
            SLANG_ASSERT(srcStructType);

            if constexpr (copyOptions == (int)CopyOptions::FlatStructIntoStruct)
            {
                // CopyOptions::FlatStructIntoStruct copy a flattened-struct (mapped member) into a
                // struct
                SLANG_ASSERT(node == dstStructType);
                _emitCopy<copyOptions>(
                    builder,
                    dstVal,
                    dstStructType,
                    srcVal,
                    srcStructType,
                    *this);
            }
            else
            {
                // CopyOptions::StructIntoFlatStruct copy a struct into a flattened-struct
                SLANG_ASSERT(node == srcStructType);
                _emitCopy<copyOptions>(
                    builder,
                    srcVal,
                    srcStructType,
                    dstVal,
                    dstStructType,
                    *this);
            }
        }
    };

    IRStructType* _flattenNestedStructs(
        IRBuilder& builder,
        IRStructType* dst,
        IRStructType* src,
        IRSemanticDecoration* parentSemanticDecoration,
        IRLayoutDecoration* parentLayout,
        MapStructToFlatStruct& mapFieldToField,
        HashSet<IRStructField*>& varsWithSemanticInfo)
    {
        // For all fields ('oldField') of a struct do the following:
        // 1. Check for 'decorations which carry semantic info' (IRSemanticDecoration,
        // IRLayoutDecoration), store these if found.
        //  * Do not propagate semantic info if the current node has *any* form of semantic
        //  information.
        // Update varsWithSemanticInfo.
        // 2. If IRStructType:
        //  2a. Recurse this function with 'decorations that carry semantic info' from parent.
        // 3. If not IRStructType:
        //  3a Metal. Emit 'newField' equal to 'oldField', add 'decorations which carry semantic
        //  info'.
        //
        //  3a WGSL. Emit 'newField' with 'newKey' equal to 'oldField' and 'oldKey', respectively,
        //      where 'oldKey' is the key corresponding to 'oldField'.
        //      Add 'decorations which carry semantic info' to 'newField', and move all decorations
        //      of 'oldKey' to 'newKey'.
        //  3b. Store a mapping from 'oldField' to 'newField' in 'mapFieldToField'. This info is
        //  needed to copy between types.
        for (auto oldField : src->getFields())
        {
            auto& fieldMappingNode = mapFieldToField[oldField];
            fieldMappingNode.setNode(oldField);

            // step 1
            bool foundSemanticDecor = false;
            auto oldKey = oldField->getKey();
            IRSemanticDecoration* fieldSemanticDecoration = parentSemanticDecoration;
            if (auto oldSemanticDecoration = oldKey->findDecoration<IRSemanticDecoration>())
            {
                foundSemanticDecor = true;
                fieldSemanticDecoration = oldSemanticDecoration;
                parentLayout = nullptr;
            }

            IRLayoutDecoration* fieldLayout = parentLayout;
            if (auto oldLayout = oldKey->findDecoration<IRLayoutDecoration>())
            {
                fieldLayout = oldLayout;
                if (!foundSemanticDecor)
                    fieldSemanticDecoration = nullptr;
            }
            if (fieldSemanticDecoration != parentSemanticDecoration || parentLayout != fieldLayout)
                varsWithSemanticInfo.add(oldField);

            // step 2a
            if (auto structFieldType = as<IRStructType>(oldField->getFieldType()))
            {
                _flattenNestedStructs(
                    builder,
                    dst,
                    structFieldType,
                    fieldSemanticDecoration,
                    fieldLayout,
                    fieldMappingNode,
                    varsWithSemanticInfo);
                continue;
            }

            // step 3a
            auto newKey = builder.createStructKey();
            flattenNestedStructsTransferKeyDecorations(newKey, oldKey);

            auto newField = builder.createStructField(dst, newKey, oldField->getFieldType());
            copyNameHintAndDebugDecorations(newField, oldField);

            if (fieldSemanticDecoration)
                builder.addSemanticDecoration(
                    newKey,
                    fieldSemanticDecoration->getSemanticName(),
                    fieldSemanticDecoration->getSemanticIndex());

            if (fieldLayout)
            {
                IRLayout* oldLayout = fieldLayout->getLayout();
                List<IRInst*> instToCopy;
                // Only copy certain decorations needed for resolving system semantics
                for (UInt i = 0; i < oldLayout->getOperandCount(); i++)
                {
                    auto operand = oldLayout->getOperand(i);
                    if (as<IRVarOffsetAttr>(operand) || as<IRUserSemanticAttr>(operand) ||
                        as<IRSystemValueSemanticAttr>(operand) || as<IRStageAttr>(operand))
                        instToCopy.add(operand);
                }
                IRVarLayout* newLayout = builder.getVarLayout(instToCopy);
                builder.addLayoutDecoration(newKey, newLayout);
            }
            // step 3b
            fieldMappingNode.setMapping(newField);
        }

        return dst;
    }

    // Returns a `IRStructType*` without any `IRStructType*` members. `src` may be returned if there
    // was no struct flattening.
    // @param mapFieldToField Behavior maps all `IRStructField` of `src` to the new struct
    // `IRStructFields`s
    IRStructType* maybeFlattenNestedStructs(
        IRBuilder& builder,
        IRStructType* src,
        MapStructToFlatStruct& mapFieldToField,
        HashSet<IRStructField*>& varsWithSemanticInfo)
    {
        // Find all values inside struct that need flattening and legalization.
        bool hasStructTypeMembers = false;
        for (auto field : src->getFields())
        {
            if (as<IRStructType>(field->getFieldType()))
            {
                hasStructTypeMembers = true;
                break;
            }
        }
        if (!hasStructTypeMembers)
            return src;

        // We need to:
        // 1. Make new struct 1:1 with old struct but without nestested structs (flatten)
        // 2. Ensure semantic attributes propegate. This will create overlapping semantics (can be
        // handled later).
        // 3. Store the mapping from old to new struct fields to allow copying a old-struct to
        // new-struct.
        builder.setInsertAfter(src);
        auto newStruct = builder.createStructType();
        copyNameHintAndDebugDecorations(newStruct, src);
        mapFieldToField.setNode(src);
        return _flattenNestedStructs(
            builder,
            newStruct,
            src,
            nullptr,
            nullptr,
            mapFieldToField,
            varsWithSemanticInfo);
    }

    // Replaces all 'IRReturn' by copying the current 'IRReturn' to a new var of type 'newType'.
    // Copying logic from 'IRReturn' to 'newType' is controlled by 'copyLogicFunc' function.
    template<typename CopyLogicFunc>
    void _replaceAllReturnInst(
        IRBuilder& builder,
        IRFunc* targetFunc,
        IRStructType* newType,
        CopyLogicFunc copyLogicFunc)
    {
        for (auto block : targetFunc->getBlocks())
        {
            if (auto returnInst = as<IRReturn>(block->getTerminator()))
            {
                builder.setInsertBefore(returnInst);
                auto returnVal = returnInst->getVal();
                returnInst->setOperand(0, copyLogicFunc(builder, newType, returnVal));
            }
        }
    }

    UInt _returnNonOverlappingAttributeIndex(std::set<UInt>& usedSemanticIndex)
    {
        // Find first unused semantic index of equal semantic type
        // to fill any gaps in user set semantic bindings
        UInt prev = 0;
        for (auto i : usedSemanticIndex)
        {
            if (i > prev + 1)
            {
                break;
            }
            prev = i;
        }
        usedSemanticIndex.insert(prev + 1);
        return prev + 1;
    }

    template<typename T>
    struct AttributeParentPair
    {
        IRLayoutDecoration* layoutDecor;
        T* attr;
    };

    IRLayoutDecoration* _replaceAttributeOfLayout(
        IRBuilder& builder,
        IRLayoutDecoration* parentLayoutDecor,
        IRInst* instToReplace,
        IRInst* instToReplaceWith)
    {
        // Replace `instToReplace` with a `instToReplaceWith`

        auto layout = parentLayoutDecor->getLayout();
        // Find the exact same decoration `instToReplace` in-case multiple of the same type exist
        List<IRInst*> opList;
        opList.add(instToReplaceWith);
        for (UInt i = 0; i < layout->getOperandCount(); i++)
        {
            if (layout->getOperand(i) != instToReplace)
                opList.add(layout->getOperand(i));
        }
        auto newLayoutDecor = builder.addLayoutDecoration(
            parentLayoutDecor->getParent(),
            builder.getVarLayout(opList));
        parentLayoutDecor->removeAndDeallocate();
        return newLayoutDecor;
    }

    IRLayoutDecoration* _simplifyUserSemanticNames(
        IRBuilder& builder,
        IRLayoutDecoration* layoutDecor)
    {
        // Ensure all 'ExplicitIndex' semantics such as "SV_TARGET0" are simplified into
        // ("SV_TARGET", 0) using 'IRUserSemanticAttr' This is done to ensure we can check semantic
        // groups using 'IRUserSemanticAttr1->getName() == IRUserSemanticAttr2->getName()'
        SLANG_ASSERT(layoutDecor);
        auto layout = layoutDecor->getLayout();
        List<IRInst*> layoutOps;
        layoutOps.reserve(3);
        bool changed = false;
        for (auto attr : layout->getAllAttrs())
        {
            if (auto userSemantic = as<IRUserSemanticAttr>(attr))
            {
                UnownedStringSlice outName;
                UnownedStringSlice outIndex;
                bool hasStringIndex = splitNameAndIndex(userSemantic->getName(), outName, outIndex);
                if (hasStringIndex)
                {
                    changed = true;
                    auto loweredName = String(outName).toLower();
                    auto loweredNameSlice = loweredName.getUnownedSlice();
                    auto newDecoration =
                        builder.getUserSemanticAttr(loweredNameSlice, stringToInt(outIndex));
                    userSemantic->replaceUsesWith(newDecoration);
                    userSemantic->removeAndDeallocate();
                    userSemantic = newDecoration;
                }
                layoutOps.add(userSemantic);
                continue;
            }
            layoutOps.add(attr);
        }
        if (changed)
        {
            auto parent = layoutDecor->parent;
            layoutDecor->removeAndDeallocate();
            builder.addLayoutDecoration(parent, builder.getVarLayout(layoutOps));
        }
        return layoutDecor;
    }

    // Find overlapping field semantics and legalize them
    void fixFieldSemanticsOfFlatStruct(IRStructType* structType)
    {
        // Goal is to ensure we do not have overlapping semantics for the user defined semantics:
        // Note that in WGSL, the semantics can be either `builtin` without index or `location` with
        // index.
        /*
            // Assume the following code
            struct Fragment
            {
                float4 p0 : SV_POSITION;
                float2 p1 : TEXCOORD0;
                float2 p2 : TEXCOORD1;
                float3 p3 : COLOR0;
                float3 p4 : COLOR1;
            };

            // Translates into
            struct Fragment
            {
                float4 p0 : BUILTIN_POSITION;
                float2 p1 : LOCATION_0;
                float2 p2 : LOCATION_1;
                float3 p3 : LOCATION_2;
                float3 p4 : LOCATION_3;
            };
        */

        // For Multi-Render-Target, the semantic index must be translated to `location` with
        // the same index. Assume the following code
        /*
            struct Fragment
            {
                float4 p0 : SV_TARGET1;
                float4 p1 : SV_TARGET0;
            };

            // Translates into
            struct Fragment
            {
                float4 p0 : LOCATION_1;
                float4 p1 : LOCATION_0;
            };
        */

        IRBuilder builder(this->m_module);

        List<IRSemanticDecoration*> overlappingSemanticsDecor;
        Dictionary<UnownedStringSlice, std::set<UInt, std::less<UInt>>>
            usedSemanticIndexSemanticDecor;

        List<AttributeParentPair<IRVarOffsetAttr>> overlappingVarOffset;
        Dictionary<UInt, std::set<UInt, std::less<UInt>>> usedSemanticIndexVarOffset;

        List<AttributeParentPair<IRUserSemanticAttr>> overlappingUserSemantic;
        Dictionary<UnownedStringSlice, std::set<UInt, std::less<UInt>>>
            usedSemanticIndexUserSemantic;

        // We store a map from old `IRLayoutDecoration*` to new `IRLayoutDecoration*` since when
        // legalizing we may destroy and remake a `IRLayoutDecoration*`
        Dictionary<IRLayoutDecoration*, IRLayoutDecoration*> oldLayoutDecorToNew;

        // Collect all "semantic info carrying decorations". Any collected decoration will
        // fill up their respective 'Dictionary<SEMANTIC_TYPE, OrderedHashSet<UInt>>'
        // to keep track of in-use offsets for a semantic type.
        // Example: IRSemanticDecoration with name of "SV_TARGET1".
        // * This will have SEMANTIC_TYPE of "sv_target".
        // * This will use up index '1'
        //
        // Now if a second equal semantic "SV_TARGET1" is found, we add this decoration to
        // a list of 'overlapping semantic info decorations' so we can legalize this
        // 'semantic info decoration' later.
        //
        // NOTE: this is a flat struct, all members are children of the initial
        // IRStructType.
        for (auto field : structType->getFields())
        {
            auto key = field->getKey();
            if (auto semanticDecoration = key->findDecoration<IRSemanticDecoration>())
            {
                auto semanticName = semanticDecoration->getSemanticName();

                // sv_target is treated as a user-semantic because it should be emitted with
                // @location like how the user semantics are emitted.
                // For fragment shader, only sv_target will user @location, and for non-fragment
                // shaders, sv_target is not valid.
                bool isUserSemantic =
                    (semanticName.startsWithCaseInsensitive(toSlice("sv_target")) ||
                     !semanticName.startsWithCaseInsensitive(toSlice("sv_")));

                // Ensure names are in a uniform lowercase format so we can bunch together simmilar
                // semantics.
                UnownedStringSlice outName;
                UnownedStringSlice outIndex;
                bool hasStringIndex = splitNameAndIndex(semanticName, outName, outIndex);

                auto loweredName = String(outName).toLower();
                auto loweredNameSlice = getUserSemanticNameSlice(loweredName, isUserSemantic);
                auto semanticIndex =
                    hasStringIndex ? stringToInt(outIndex) : semanticDecoration->getSemanticIndex();
                auto newDecoration =
                    builder.addSemanticDecoration(key, loweredNameSlice, semanticIndex);

                semanticDecoration->replaceUsesWith(newDecoration);
                semanticDecoration->removeAndDeallocate();
                semanticDecoration = newDecoration;

                auto& semanticUse =
                    usedSemanticIndexSemanticDecor[semanticDecoration->getSemanticName()];
                if (semanticUse.find(semanticDecoration->getSemanticIndex()) != semanticUse.end())
                    overlappingSemanticsDecor.add(semanticDecoration);
                else
                    semanticUse.insert(semanticDecoration->getSemanticIndex());
            }
            if (auto layoutDecor = key->findDecoration<IRLayoutDecoration>())
            {
                // Ensure names are in a uniform lowercase format so we can bunch together simmilar
                // semantics
                layoutDecor = _simplifyUserSemanticNames(builder, layoutDecor);
                oldLayoutDecorToNew[layoutDecor] = layoutDecor;
                auto layout = layoutDecor->getLayout();
                for (auto attr : layout->getAllAttrs())
                {
                    if (auto offset = as<IRVarOffsetAttr>(attr))
                    {
                        auto& semanticUse = usedSemanticIndexVarOffset[offset->getResourceKind()];
                        if (semanticUse.find(offset->getOffset()) != semanticUse.end())
                            overlappingVarOffset.add({layoutDecor, offset});
                        else
                            semanticUse.insert(offset->getOffset());
                    }
                    else if (auto userSemantic = as<IRUserSemanticAttr>(attr))
                    {
                        auto& semanticUse = usedSemanticIndexUserSemantic[userSemantic->getName()];
                        if (semanticUse.find(userSemantic->getIndex()) != semanticUse.end())
                            overlappingUserSemantic.add({layoutDecor, userSemantic});
                        else
                            semanticUse.insert(userSemantic->getIndex());
                    }
                }
            }
        }

        // Legalize all overlapping 'semantic info decorations'
        for (auto decor : overlappingSemanticsDecor)
        {
            auto newOffset = _returnNonOverlappingAttributeIndex(
                usedSemanticIndexSemanticDecor[decor->getSemanticName()]);
            builder.addSemanticDecoration(
                decor->getParent(),
                decor->getSemanticName(),
                (int)newOffset);
            decor->removeAndDeallocate();
        }
        for (auto& varOffset : overlappingVarOffset)
        {
            auto newOffset = _returnNonOverlappingAttributeIndex(
                usedSemanticIndexVarOffset[varOffset.attr->getResourceKind()]);
            auto newVarOffset = builder.getVarOffsetAttr(
                varOffset.attr->getResourceKind(),
                newOffset,
                varOffset.attr->getSpace());
            oldLayoutDecorToNew[varOffset.layoutDecor] = _replaceAttributeOfLayout(
                builder,
                oldLayoutDecorToNew[varOffset.layoutDecor],
                varOffset.attr,
                newVarOffset);
        }
        for (auto& userSemantic : overlappingUserSemantic)
        {
            auto newOffset = _returnNonOverlappingAttributeIndex(
                usedSemanticIndexUserSemantic[userSemantic.attr->getName()]);
            auto newUserSemantic =
                builder.getUserSemanticAttr(userSemantic.attr->getName(), newOffset);
            oldLayoutDecorToNew[userSemantic.layoutDecor] = _replaceAttributeOfLayout(
                builder,
                oldLayoutDecorToNew[userSemantic.layoutDecor],
                userSemantic.attr,
                newUserSemantic);
        }
    }

    void wrapReturnValueInStruct(EntryPointInfo entryPoint)
    {
        // Wrap return value into a struct if it is not already a struct.
        // For example, given this entry point:
        // ```
        // float4 main() : SV_Target { return float3(1,2,3); }
        // ```
        // We are going to transform it into:
        // ```
        // struct Output {
        //     float4 value : SV_Target;
        // };
        // Output main() { return {float3(1,2,3)}; }

        auto func = entryPoint.entryPointFunc;

        auto returnType = func->getResultType();
        if (as<IRVoidType>(returnType))
            return;
        auto entryPointLayoutDecor = func->findDecoration<IRLayoutDecoration>();
        if (!entryPointLayoutDecor)
            return;
        auto entryPointLayout = as<IREntryPointLayout>(entryPointLayoutDecor->getLayout());
        if (!entryPointLayout)
            return;
        auto resultLayout = entryPointLayout->getResultLayout();

        // If return type is already a struct, just make sure every field has a semantic.
        if (auto returnStructType = as<IRStructType>(returnType))
        {
            IRBuilder builder(func);
            MapStructToFlatStruct mapOldFieldToNewField;
            // Flatten result struct type to ensure we do not have nested semantics
            auto flattenedStruct = maybeFlattenNestedStructs(
                builder,
                returnStructType,
                mapOldFieldToNewField,
                semanticInfoToRemove);
            if (returnStructType != flattenedStruct)
            {
                // Replace all return-values with the flattenedStruct we made.
                _replaceAllReturnInst(
                    builder,
                    func,
                    flattenedStruct,
                    [&](IRBuilder& copyBuilder, IRStructType* dstType, IRInst* srcVal) -> IRInst*
                    {
                        auto srcStructType = as<IRStructType>(srcVal->getDataType());
                        SLANG_ASSERT(srcStructType);
                        auto dstVal = copyBuilder.emitVar(dstType);
                        mapOldFieldToNewField.emitCopy<(
                            int)MapStructToFlatStruct::CopyOptions::StructIntoFlatStruct>(
                            copyBuilder,
                            dstVal,
                            srcVal);
                        return builder.emitLoad(dstVal);
                    });
                fixUpFuncType(func, flattenedStruct);
            }
            // Ensure non-overlapping semantics
            fixFieldSemanticsOfFlatStruct(flattenedStruct);
            ensureStructHasUserSemantic<LayoutResourceKind::VaryingOutput>(
                flattenedStruct,
                resultLayout);
            return;
        }

        IRBuilder builder(func);
        builder.setInsertBefore(func);
        IRStructType* structType = builder.createStructType();
        auto stageText = getStageText(entryPoint.entryPointDecor->getProfile().getStage());
        builder.addNameHintDecoration(
            structType,
            (String(stageText) + toSlice("Output")).getUnownedSlice());
        auto key = builder.createStructKey();
        builder.addNameHintDecoration(key, toSlice("output"));
        builder.addLayoutDecoration(key, resultLayout);
        builder.createStructField(structType, key, returnType);
        IRStructTypeLayout::Builder structTypeLayoutBuilder(&builder);
        structTypeLayoutBuilder.addField(key, resultLayout);
        auto typeLayout = structTypeLayoutBuilder.build();
        IRVarLayout::Builder varLayoutBuilder(&builder, typeLayout);
        auto varLayout = varLayoutBuilder.build();
        ensureStructHasUserSemantic<LayoutResourceKind::VaryingOutput>(structType, varLayout);

        _replaceAllReturnInst(
            builder,
            func,
            structType,
            [](IRBuilder& copyBuilder, IRStructType* dstType, IRInst* srcVal) -> IRInst*
            { return copyBuilder.emitMakeStruct(dstType, 1, &srcVal); });

        // Assign an appropriate system value semantic for stage output
        auto stage = entryPoint.entryPointDecor->getProfile().getStage();
        switch (stage)
        {
        case Stage::Compute:
        case Stage::Fragment:
            {
                addFragmentShaderReturnValueDecoration(builder, key);
                break;
            }
        case Stage::Vertex:
            {
                builder.addTargetSystemValueDecoration(key, toSlice("position"));
                break;
            }
        default:
            SLANG_ASSERT(false);
            return;
        }

        fixUpFuncType(func, structType);
    }

    IRInst* tryConvertValue(IRBuilder& builder, IRInst* val, IRType* toType)
    {
        auto fromType = val->getFullType();
        if (auto fromVector = as<IRVectorType>(fromType))
        {
            if (auto toVector = as<IRVectorType>(toType))
            {
                if (fromVector->getElementCount() != toVector->getElementCount())
                {
                    fromType = builder.getVectorType(
                        fromVector->getElementType(),
                        toVector->getElementCount());
                    val = builder.emitVectorReshape(fromType, val);
                }
            }
            else if (as<IRBasicType>(toType))
            {
                UInt index = 0;
                val = builder.emitSwizzle(fromVector->getElementType(), val, 1, &index);
                if (toType->getOp() == kIROp_VoidType)
                    return nullptr;
            }
        }
        else if (auto fromBasicType = as<IRBasicType>(fromType))
        {
            if (fromBasicType->getOp() == kIROp_VoidType)
                return nullptr;
            if (!as<IRBasicType>(toType))
                return nullptr;
            if (toType->getOp() == kIROp_VoidType)
                return nullptr;
        }
        else
        {
            return nullptr;
        }
        return builder.emitCast(toType, val);
    }

    void legalizeSystemValueParameters(EntryPointInfo entryPoint)
    {
        List<SystemValLegalizationWorkItem> systemValWorkItems =
            collectSystemValFromEntryPoint(entryPoint);

        for (auto index = 0; index < systemValWorkItems.getCount(); index++)
        {
            legalizeSystemValue(entryPoint, systemValWorkItems[index]);
        }
        fixUpFuncType(entryPoint.entryPointFunc);
    }

    void legalizeEntryPoint(EntryPointInfo entryPoint)
    {
        // If the entrypoint is receiving varying inputs as a pointer, turn it into a value.
        depointerizeInputParams(entryPoint.entryPointFunc);

        // Input Parameter Legalize
        hoistEntryPointParameterFromStruct(entryPoint);
        packStageInParameters(entryPoint);
        flattenInputParameters(entryPoint);

        // System Value Legalize
        legalizeSystemValueParameters(entryPoint);

        // Output Value Legalize
        wrapReturnValueInStruct(entryPoint);


        // Other Legalize
        switch (entryPoint.entryPointDecor->getProfile().getStage())
        {
        case Stage::Amplification:
            legalizeAmplificationStageEntryPoint(entryPoint);
            break;
        case Stage::Mesh:
            legalizeMeshStageEntryPoint(entryPoint);
            break;
        default:
            break;
        }
    }
};

class LegalizeMetalEntryPointContext : public LegalizeShaderEntryPointContext
{
public:
    LegalizeMetalEntryPointContext(IRModule* module, DiagnosticSink* sink)
        : LegalizeShaderEntryPointContext(module, sink)
    {
        generatePermittedTypes_sv_target();
    }

protected:
    SystemValueInfo getSystemValueInfo(
        String inSemanticName,
        String* optionalSemanticIndex,
        IRInst* parentVar) const SLANG_OVERRIDE
    {
        IRBuilder builder(m_module);
        SystemValueInfo result = {};
        UnownedStringSlice semanticName;
        UnownedStringSlice semanticIndex;

        auto hasExplicitIndex =
            splitNameAndIndex(inSemanticName.getUnownedSlice(), semanticName, semanticIndex);
        if (!hasExplicitIndex && optionalSemanticIndex)
            semanticIndex = optionalSemanticIndex->getUnownedSlice();

        result.systemValueNameEnum = convertSystemValueSemanticNameToEnum(semanticName);

        switch (result.systemValueNameEnum)
        {
        case SystemValueSemanticName::Position:
            {
                result.systemValueName = toSlice("position");
                result.permittedTypes.add(builder.getVectorType(
                    builder.getBasicType(BaseType::Float),
                    builder.getIntValue(builder.getIntType(), 4)));
                break;
            }
        case SystemValueSemanticName::ClipDistance:
            {
                result.isSpecial = true;
                break;
            }
        case SystemValueSemanticName::CullDistance:
            {
                result.isSpecial = true;
                break;
            }
        case SystemValueSemanticName::Coverage:
            {
                result.systemValueName = toSlice("sample_mask");
                result.permittedTypes.add(builder.getBasicType(BaseType::UInt));
                break;
            }
        case SystemValueSemanticName::InnerCoverage:
            {
                result.isSpecial = true;
                break;
            }
        case SystemValueSemanticName::Depth:
            {
                result.systemValueName = toSlice("depth(any)");
                result.permittedTypes.add(builder.getBasicType(BaseType::Float));
                break;
            }
        case SystemValueSemanticName::DepthGreaterEqual:
            {
                result.systemValueName = toSlice("depth(greater)");
                result.permittedTypes.add(builder.getBasicType(BaseType::Float));
                break;
            }
        case SystemValueSemanticName::DepthLessEqual:
            {
                result.systemValueName = toSlice("depth(less)");
                result.permittedTypes.add(builder.getBasicType(BaseType::Float));
                break;
            }
        case SystemValueSemanticName::DispatchThreadID:
            {
                result.systemValueName = toSlice("thread_position_in_grid");
                result.permittedTypes.add(builder.getVectorType(
                    builder.getBasicType(BaseType::UInt),
                    builder.getIntValue(builder.getIntType(), 3)));
                break;
            }
        case SystemValueSemanticName::DomainLocation:
            {
                result.systemValueName = toSlice("position_in_patch");
                result.permittedTypes.add(builder.getVectorType(
                    builder.getBasicType(BaseType::Float),
                    builder.getIntValue(builder.getIntType(), 3)));
                result.permittedTypes.add(builder.getVectorType(
                    builder.getBasicType(BaseType::Float),
                    builder.getIntValue(builder.getIntType(), 2)));
                break;
            }
        case SystemValueSemanticName::GroupID:
            {
                result.systemValueName = toSlice("threadgroup_position_in_grid");
                result.permittedTypes.add(builder.getVectorType(
                    builder.getBasicType(BaseType::UInt),
                    builder.getIntValue(builder.getIntType(), 3)));
                break;
            }
        case SystemValueSemanticName::GroupIndex:
            {
                result.isSpecial = true;
                break;
            }
        case SystemValueSemanticName::GroupThreadID:
            {
                result.systemValueName = toSlice("thread_position_in_threadgroup");
                result.permittedTypes.add(getGroupThreadIdType(builder));
                break;
            }
        case SystemValueSemanticName::GSInstanceID:
            {
                result.isUnsupported = true;
                break;
            }
        case SystemValueSemanticName::InstanceID:
            {
                result.systemValueName = toSlice("instance_id");
                result.permittedTypes.add(builder.getBasicType(BaseType::UInt));
                break;
            }
        case SystemValueSemanticName::IsFrontFace:
            {
                result.systemValueName = toSlice("front_facing");
                result.permittedTypes.add(builder.getBasicType(BaseType::Bool));
                break;
            }
        case SystemValueSemanticName::OutputControlPointID:
            {
                // In metal, a hull shader is just a compute shader.
                // This needs to be handled separately, by lowering into an ordinary buffer.
                break;
            }
        case SystemValueSemanticName::PointSize:
            {
                result.systemValueName = toSlice("point_size");
                result.permittedTypes.add(builder.getBasicType(BaseType::Float));
                break;
            }
        case SystemValueSemanticName::PointCoord:
            {
                result.systemValueName = toSlice("point_coord");
                result.permittedTypes.add(builder.getVectorType(
                    builder.getBasicType(BaseType::Float),
                    builder.getIntValue(builder.getIntType(), 2)));
                break;
            }
        case SystemValueSemanticName::PrimitiveID:
            {
                result.systemValueName = toSlice("primitive_id");
                result.permittedTypes.add(builder.getBasicType(BaseType::UInt));
                result.permittedTypes.add(builder.getBasicType(BaseType::UInt16));
                break;
            }
        case SystemValueSemanticName::RenderTargetArrayIndex:
            {
                result.systemValueName = toSlice("render_target_array_index");
                result.permittedTypes.add(builder.getBasicType(BaseType::UInt));
                result.permittedTypes.add(builder.getBasicType(BaseType::UInt16));
                break;
            }
        case SystemValueSemanticName::SampleIndex:
            {
                result.systemValueName = toSlice("sample_id");
                result.permittedTypes.add(builder.getBasicType(BaseType::UInt));
                break;
            }
        case SystemValueSemanticName::StencilRef:
            {
                result.systemValueName = toSlice("stencil");
                result.permittedTypes.add(builder.getBasicType(BaseType::UInt));
                break;
            }
        case SystemValueSemanticName::TessFactor:
            {
                // Tessellation factor outputs should be lowered into a write into a normal buffer.
                break;
            }
        case SystemValueSemanticName::VertexID:
            {
                result.systemValueName = toSlice("vertex_id");
                result.permittedTypes.add(builder.getBasicType(BaseType::UInt));
                break;
            }
        case SystemValueSemanticName::ViewID:
            {
                result.systemValueName = toSlice("amplification_id");
                result.permittedTypes.add(builder.getBasicType(BaseType::UInt));
                result.permittedTypes.add(builder.getBasicType(BaseType::UInt16));
                break;
            }
        case SystemValueSemanticName::ViewportArrayIndex:
            {
                result.systemValueName = toSlice("viewport_array_index");
                result.permittedTypes.add(builder.getBasicType(BaseType::UInt));
                result.permittedTypes.add(builder.getBasicType(BaseType::UInt16));
                break;
            }
        case SystemValueSemanticName::Target:
            {
                result.systemValueName =
                    (StringBuilder()
                     << "color(" << (semanticIndex.getLength() != 0 ? semanticIndex : toSlice("0"))
                     << ")")
                        .produceString();
                result.permittedTypes = permittedTypes_sv_target;

                break;
            }
        case SystemValueSemanticName::StartVertexLocation:
            {
                result.systemValueName = toSlice("base_vertex");
                result.permittedTypes.add(builder.getBasicType(BaseType::UInt));
                break;
            }
        case SystemValueSemanticName::StartInstanceLocation:
            {
                result.systemValueName = toSlice("base_instance");
                result.permittedTypes.add(builder.getBasicType(BaseType::UInt));
                break;
            }
        case SystemValueSemanticName::WaveLaneCount:
            {
                result.systemValueName = toSlice("threads_per_simdgroup");
                result.permittedTypes.add(builder.getUIntType());
                result.permittedTypes.add(builder.getUInt16Type());
                break;
            }
        case SystemValueSemanticName::WaveLaneIndex:
            {
                result.systemValueName = toSlice("thread_index_in_simdgroup");
                result.permittedTypes.add(builder.getUIntType());
                result.permittedTypes.add(builder.getUInt16Type());
                break;
            }
        case SystemValueSemanticName::QuadLaneIndex:
            {
                result.systemValueName = toSlice("thread_index_in_quadgroup");
                result.permittedTypes.add(builder.getUInt16Type());
                result.permittedTypes.add(builder.getUIntType());
                break;
            }
        default:
            m_sink->diagnose(
                parentVar,
                Diagnostics::unimplementedSystemValueSemantic,
                semanticName);
            return result;
        }
        return result;
    }


    List<SystemValLegalizationWorkItem> collectSystemValFromEntryPoint(
        EntryPointInfo entryPoint) const SLANG_OVERRIDE
    {
        List<SystemValLegalizationWorkItem> systemValWorkItems;
        for (auto param : entryPoint.entryPointFunc->getParams())
        {
            auto maybeWorkItem = tryToMakeSystemValWorkItem(param, param->getFullType());
            if (maybeWorkItem.has_value())
                systemValWorkItems.add(std::move(maybeWorkItem.value()));
        }

        return systemValWorkItems;
    }

    void flattenNestedStructsTransferKeyDecorations(IRInst* newKey, IRInst* oldKey) const
        SLANG_OVERRIDE
    {
        copyNameHintAndDebugDecorations(newKey, oldKey);
    }

    UnownedStringSlice getUserSemanticNameSlice(String& loweredName, bool isUserSemantic) const
        SLANG_OVERRIDE
    {
        SLANG_UNUSED(isUserSemantic);
        return loweredName.getUnownedSlice();
    };

    void addFragmentShaderReturnValueDecoration(IRBuilder& builder, IRInst* returnValueStructKey)
        const SLANG_OVERRIDE
    {
        builder.addTargetSystemValueDecoration(returnValueStructKey, toSlice("color(0)"));
    }

    IRVarLayout* handleGeometryStageParameterVarLayout(
        IRBuilder& builder,
        IRVarLayout* paramVarLayout) const SLANG_OVERRIDE
    {
        // For Metal geometric stages, we need to translate VaryingInput offsets to
        // MetalAttribute offsets.
        IRVarLayout::Builder elementVarLayoutBuilder(&builder, paramVarLayout->getTypeLayout());
        elementVarLayoutBuilder.cloneEverythingButOffsetsFrom(paramVarLayout);
        for (auto offsetAttr : paramVarLayout->getOffsetAttrs())
        {
            auto resourceKind = offsetAttr->getResourceKind();
            if (resourceKind == LayoutResourceKind::VaryingInput)
            {
                resourceKind = LayoutResourceKind::MetalAttribute;
            }
            auto resInfo = elementVarLayoutBuilder.findOrAddResourceInfo(resourceKind);
            resInfo->offset = offsetAttr->getOffset();
            resInfo->space = offsetAttr->getSpace();
        }

        return elementVarLayoutBuilder.build();
    }

    void handleSpecialSystemValue(
        const EntryPointInfo& entryPoint,
        SystemValLegalizationWorkItem& workItem,
        const SystemValueInfo& info,
        IRBuilder& builder) SLANG_OVERRIDE
    {
        const auto var = workItem.var;

        if (info.systemValueNameEnum == SystemValueSemanticName::InnerCoverage)
        {
            // Metal does not support conservative rasterization, so this is always false.
            auto val = builder.getBoolValue(false);
            var->replaceUsesWith(val);
            var->removeAndDeallocate();
        }
        else if (info.systemValueNameEnum == SystemValueSemanticName::GroupIndex)
        {
            // Ensure we have a cached "sv_groupthreadid" in our entry point
            if (!entryPointToGroupThreadId.containsKey(entryPoint.entryPointFunc))
            {
                auto systemValWorkItems = collectSystemValFromEntryPoint(entryPoint);
                for (auto i : systemValWorkItems)
                {
                    auto indexAsStringGroupThreadId = String(i.attrIndex);
                    if (getSystemValueInfo(i.attrName, &indexAsStringGroupThreadId, i.var)
                            .systemValueNameEnum == SystemValueSemanticName::GroupThreadID)
                    {
                        entryPointToGroupThreadId[entryPoint.entryPointFunc] = i.var;
                    }
                }
                if (!entryPointToGroupThreadId.containsKey(entryPoint.entryPointFunc))
                {
                    // Add the missing groupthreadid needed to compute sv_groupindex
                    IRBuilder groupThreadIdBuilder(builder);
                    groupThreadIdBuilder.setInsertInto(entryPoint.entryPointFunc->getFirstBlock());
                    auto groupThreadId = groupThreadIdBuilder.emitParamAtHead(
                        getGroupThreadIdType(groupThreadIdBuilder));
                    entryPointToGroupThreadId[entryPoint.entryPointFunc] = groupThreadId;
                    groupThreadIdBuilder.addNameHintDecoration(groupThreadId, groupThreadIDString);

                    // Since "sv_groupindex" will be translated out to a global var and no
                    // longer be considered a system value we can reuse its layout and
                    // semantic info
                    Index foundRequiredDecorations = 0;
                    IRLayoutDecoration* layoutDecoration = nullptr;
                    UInt semanticIndex = 0;
                    for (auto decoration : var->getDecorations())
                    {
                        if (auto layoutDecorationTmp = as<IRLayoutDecoration>(decoration))
                        {
                            layoutDecoration = layoutDecorationTmp;
                            foundRequiredDecorations++;
                        }
                        else if (auto semanticDecoration = as<IRSemanticDecoration>(decoration))
                        {
                            semanticIndex = semanticDecoration->getSemanticIndex();
                            groupThreadIdBuilder.addSemanticDecoration(
                                groupThreadId,
                                groupThreadIDString,
                                (int)semanticIndex);
                            foundRequiredDecorations++;
                        }
                        if (foundRequiredDecorations >= 2)
                            break;
                    }
                    SLANG_ASSERT(layoutDecoration);
                    layoutDecoration->removeFromParent();
                    layoutDecoration->insertAtStart(groupThreadId);
                    SystemValLegalizationWorkItem newWorkItem = {
                        groupThreadId,
                        groupThreadId->getFullType(),
                        groupThreadIDString,
                        semanticIndex};
                    legalizeSystemValue(entryPoint, newWorkItem);
                }
            }

            IRBuilder svBuilder(builder.getModule());
            svBuilder.setInsertBefore(entryPoint.entryPointFunc->getFirstOrdinaryInst());
            auto uint3Type = builder.getVectorType(
                builder.getUIntType(),
                builder.getIntValue(builder.getIntType(), 3));
            auto computeExtent =
                emitCalcGroupExtents(svBuilder, entryPoint.entryPointFunc, uint3Type);
            if (!computeExtent)
            {
                m_sink->diagnose(
                    entryPoint.entryPointFunc,
                    Diagnostics::unsupportedSpecializationConstantForNumThreads);

                // Fill in placeholder values.
                static const int kAxisCount = 3;
                IRInst* groupExtentAlongAxis[kAxisCount] = {};
                for (int axis = 0; axis < kAxisCount; axis++)
                    groupExtentAlongAxis[axis] =
                        builder.getIntValue(uint3Type->getElementType(), 1);
                computeExtent = builder.emitMakeVector(uint3Type, kAxisCount, groupExtentAlongAxis);
            }
            auto groupIndexCalc = emitCalcGroupIndex(
                svBuilder,
                entryPointToGroupThreadId[entryPoint.entryPointFunc],
                computeExtent);
            svBuilder.addNameHintDecoration(groupIndexCalc, UnownedStringSlice("sv_groupindex"));

            var->replaceUsesWith(groupIndexCalc);
            var->removeAndDeallocate();
        }
    }

    void legalizeAmplificationStageEntryPoint(const EntryPointInfo& entryPoint) const SLANG_OVERRIDE
    {
        // Find out DispatchMesh function
        IRGlobalValueWithCode* dispatchMeshFunc = nullptr;
        for (const auto globalInst : entryPoint.entryPointFunc->getModule()->getGlobalInsts())
        {
            if (const auto func = as<IRGlobalValueWithCode>(globalInst))
            {
                if (const auto dec = func->findDecoration<IRKnownBuiltinDecoration>())
                {
                    if (dec->getName() == "DispatchMesh")
                    {
                        SLANG_ASSERT(!dispatchMeshFunc && "Multiple DispatchMesh functions found");
                        dispatchMeshFunc = func;
                    }
                }
            }
        }

        if (!dispatchMeshFunc)
            return;

        IRBuilder builder{entryPoint.entryPointFunc->getModule()};

        // We'll rewrite the call to use mesh_grid_properties.set_threadgroups_per_grid
        traverseUses(
            dispatchMeshFunc,
            [&](const IRUse* use)
            {
                if (const auto call = as<IRCall>(use->getUser()))
                {
                    SLANG_ASSERT(call->getArgCount() == 4);
                    const auto payload = call->getArg(3);

                    const auto payloadPtrType =
                        composeGetters<IRPtrType>(payload, &IRInst::getDataType);
                    SLANG_ASSERT(payloadPtrType);
                    const auto payloadType = payloadPtrType->getValueType();
                    SLANG_ASSERT(payloadType);

                    builder.setInsertBefore(
                        entryPoint.entryPointFunc->getFirstBlock()->getFirstOrdinaryInst());
                    const auto annotatedPayloadType = builder.getPtrType(
                        kIROp_RefType,
                        payloadPtrType->getValueType(),
                        AddressSpace::MetalObjectData);
                    auto packedParam = builder.emitParam(annotatedPayloadType);
                    builder.addExternCppDecoration(packedParam, toSlice("_slang_mesh_payload"));
                    IRVarLayout::Builder varLayoutBuilder(
                        &builder,
                        IRTypeLayout::Builder{&builder}.build());

                    // Add the MetalPayload resource info, so we can emit [[payload]]
                    varLayoutBuilder.findOrAddResourceInfo(LayoutResourceKind::MetalPayload);
                    auto paramVarLayout = varLayoutBuilder.build();
                    builder.addLayoutDecoration(packedParam, paramVarLayout);

                    // Now we replace the call to DispatchMesh with a call to the mesh grid
                    // properties But first we need to create the parameter
                    const auto meshGridPropertiesType = builder.getMetalMeshGridPropertiesType();
                    auto mgp = builder.emitParam(meshGridPropertiesType);
                    builder.addExternCppDecoration(mgp, toSlice("_slang_mgp"));
                }
            });
    }

    void legalizeMeshStageEntryPoint(const EntryPointInfo& entryPoint) const SLANG_OVERRIDE
    {
        auto func = entryPoint.entryPointFunc;

        IRBuilder builder{func->getModule()};
        for (auto param : func->getParams())
        {
            if (param->findDecorationImpl(kIROp_HLSLMeshPayloadDecoration))
            {
                IRVarLayout::Builder varLayoutBuilder(
                    &builder,
                    IRTypeLayout::Builder{&builder}.build());

                varLayoutBuilder.findOrAddResourceInfo(LayoutResourceKind::MetalPayload);
                auto paramVarLayout = varLayoutBuilder.build();
                builder.addLayoutDecoration(param, paramVarLayout);

                IRPtrTypeBase* type = as<IRPtrTypeBase>(param->getDataType());

                const auto annotatedPayloadType = builder.getPtrType(
                    kIROp_ConstRefType,
                    type->getValueType(),
                    AddressSpace::MetalObjectData);

                param->setFullType(annotatedPayloadType);
            }
        }
        IROutputTopologyDecoration* outputDeco =
            entryPoint.entryPointFunc->findDecoration<IROutputTopologyDecoration>();
        if (outputDeco == nullptr)
        {
            SLANG_UNEXPECTED("Mesh shader output decoration missing");
            return;
        }

        const auto topologyEnum = outputDeco->getTopologyType();
        IRInst* topologyConst = builder.getIntValue(builder.getIntType(), topologyEnum);

        IRType* vertexType = nullptr;
        IRType* indicesType = nullptr;
        IRType* primitiveType = nullptr;

        IRInst* maxVertices = nullptr;
        IRInst* maxPrimitives = nullptr;

        IRInst* verticesParam = nullptr;
        IRInst* indicesParam = nullptr;
        IRInst* primitivesParam = nullptr;
        for (auto param : func->getParams())
        {
            if (param->findDecorationImpl(kIROp_HLSLMeshPayloadDecoration))
            {
                IRVarLayout::Builder varLayoutBuilder(
                    &builder,
                    IRTypeLayout::Builder{&builder}.build());

                varLayoutBuilder.findOrAddResourceInfo(LayoutResourceKind::MetalPayload);
                auto paramVarLayout = varLayoutBuilder.build();
                builder.addLayoutDecoration(param, paramVarLayout);
            }
            if (param->findDecorationImpl(kIROp_VerticesDecoration))
            {
                auto vertexRefType = as<IRPtrTypeBase>(param->getDataType());
                auto vertexOutputType = as<IRVerticesType>(vertexRefType->getValueType());
                vertexType = vertexOutputType->getElementType();
                maxVertices = vertexOutputType->getMaxElementCount();
                SLANG_ASSERT(vertexType);

                verticesParam = param;
                auto vertStruct = as<IRStructType>(vertexType);
                for (auto field : vertStruct->getFields())
                {
                    auto key = field->getKey();
                    if (auto deco = key->findDecoration<IRSemanticDecoration>())
                    {
                        if (deco->getSemanticName().caseInsensitiveEquals(toSlice("sv_position")))
                        {
                            builder.addTargetSystemValueDecoration(key, toSlice("position"));
                        }
                    }
                }
            }
            if (param->findDecorationImpl(kIROp_IndicesDecoration))
            {
                auto indicesRefType = (IRConstRefType*)param->getDataType();
                auto indicesOutputType = (IRIndicesType*)indicesRefType->getValueType();
                indicesType = indicesOutputType->getElementType();
                maxPrimitives = indicesOutputType->getMaxElementCount();
                SLANG_ASSERT(indicesType);

                indicesParam = param;
            }
            if (param->findDecorationImpl(kIROp_PrimitivesDecoration))
            {
                auto primitivesRefType = (IRConstRefType*)param->getDataType();
                auto primitivesOutputType = (IRPrimitivesType*)primitivesRefType->getValueType();
                primitiveType = primitivesOutputType->getElementType();
                SLANG_ASSERT(primitiveType);

                primitivesParam = param;
                auto primStruct = as<IRStructType>(primitiveType);
                for (auto field : primStruct->getFields())
                {
                    auto key = field->getKey();
                    if (auto deco = key->findDecoration<IRSemanticDecoration>())
                    {
                        if (deco->getSemanticName().caseInsensitiveEquals(
                                toSlice("sv_primitiveid")))
                        {
                            builder.addTargetSystemValueDecoration(key, toSlice("primitive_id"));
                        }
                    }
                }
            }
        }
        if (primitiveType == nullptr)
        {
            primitiveType = builder.getVoidType();
        }
        builder.setInsertBefore(entryPoint.entryPointFunc->getFirstBlock()->getFirstOrdinaryInst());

        auto meshParam = builder.emitParam(builder.getMetalMeshType(
            vertexType,
            primitiveType,
            maxVertices,
            maxPrimitives,
            topologyConst));
        builder.addExternCppDecoration(meshParam, toSlice("_slang_mesh"));


        verticesParam->removeFromParent();
        verticesParam->removeAndDeallocate();

        indicesParam->removeFromParent();
        indicesParam->removeAndDeallocate();

        if (primitivesParam != nullptr)
        {
            primitivesParam->removeFromParent();
            primitivesParam->removeAndDeallocate();
        }
    }

private:
    ShortList<IRType*> permittedTypes_sv_target;
    Dictionary<IRFunc*, IRInst*> entryPointToGroupThreadId;
    const UnownedStringSlice groupThreadIDString = UnownedStringSlice("sv_groupthreadid");

    static IRType* getGroupThreadIdType(IRBuilder& builder)
    {
        return builder.getVectorType(
            builder.getBasicType(BaseType::UInt),
            builder.getIntValue(builder.getIntType(), 3));
    }

    void generatePermittedTypes_sv_target()
    {
        IRBuilder builder(m_module);
        permittedTypes_sv_target.reserveOverflowBuffer(5 * 4);
        if (permittedTypes_sv_target.getCount() == 0)
        {
            for (auto baseType :
                 {BaseType::Float,
                  BaseType::Half,
                  BaseType::Int,
                  BaseType::UInt,
                  BaseType::Int16,
                  BaseType::UInt16})
            {
                for (IRIntegerValue i = 1; i <= 4; i++)
                {
                    permittedTypes_sv_target.add(
                        builder.getVectorType(builder.getBasicType(baseType), i));
                }
            }
        }
    }
};


class LegalizeWGSLEntryPointContext : public LegalizeShaderEntryPointContext
{
public:
    LegalizeWGSLEntryPointContext(IRModule* module, DiagnosticSink* sink)
        : LegalizeShaderEntryPointContext(module, sink)
    {
    }

protected:
    SystemValueInfo getSystemValueInfo(
        String inSemanticName,
        String* optionalSemanticIndex,
        IRInst* parentVar) const SLANG_OVERRIDE
    {
        IRBuilder builder(m_module);
        SystemValueInfo result = {};
        UnownedStringSlice semanticName;
        UnownedStringSlice semanticIndex;

        auto hasExplicitIndex =
            splitNameAndIndex(inSemanticName.getUnownedSlice(), semanticName, semanticIndex);
        if (!hasExplicitIndex && optionalSemanticIndex)
            semanticIndex = optionalSemanticIndex->getUnownedSlice();

        result.systemValueNameEnum = convertSystemValueSemanticNameToEnum(semanticName);

        switch (result.systemValueNameEnum)
        {

        case SystemValueSemanticName::CullDistance:
            {
                result.isUnsupported = true;
            }
            break;

        case SystemValueSemanticName::ClipDistance:
            {
                // TODO: Implement this based on the 'clip-distances' feature in WGSL
                // https: // www.w3.org/TR/webgpu/#dom-gpufeaturename-clip-distances
                result.isUnsupported = true;
            }
            break;

        case SystemValueSemanticName::Coverage:
            {
                result.systemValueName = toSlice("sample_mask");
                result.permittedTypes.add(builder.getUIntType());
            }
            break;

        case SystemValueSemanticName::Depth:
            {
                result.systemValueName = toSlice("frag_depth");
                result.permittedTypes.add(builder.getBasicType(BaseType::Float));
            }
            break;

        case SystemValueSemanticName::DepthGreaterEqual:
        case SystemValueSemanticName::DepthLessEqual:
            {
                result.isUnsupported = true;
            }
            break;

        case SystemValueSemanticName::DispatchThreadID:
            {
                result.systemValueName = toSlice("global_invocation_id");
                result.permittedTypes.add(builder.getVectorType(
                    builder.getBasicType(BaseType::UInt),
                    builder.getIntValue(builder.getIntType(), 3)));
            }
            break;

        case SystemValueSemanticName::DomainLocation:
            {
                result.isUnsupported = true;
            }
            break;

        case SystemValueSemanticName::GroupID:
            {
                result.systemValueName = toSlice("workgroup_id");
                result.permittedTypes.add(builder.getVectorType(
                    builder.getBasicType(BaseType::UInt),
                    builder.getIntValue(builder.getIntType(), 3)));
            }
            break;

        case SystemValueSemanticName::GroupIndex:
            {
                result.systemValueName = toSlice("local_invocation_index");
                result.permittedTypes.add(builder.getUIntType());
            }
            break;

        case SystemValueSemanticName::GroupThreadID:
            {
                result.systemValueName = toSlice("local_invocation_id");
                result.permittedTypes.add(builder.getVectorType(
                    builder.getBasicType(BaseType::UInt),
                    builder.getIntValue(builder.getIntType(), 3)));
            }
            break;

        case SystemValueSemanticName::GSInstanceID:
            {
                // No Geometry shaders in WGSL
                result.isUnsupported = true;
            }
            break;

        case SystemValueSemanticName::InnerCoverage:
            {
                result.isUnsupported = true;
            }
            break;

        case SystemValueSemanticName::InstanceID:
            {
                result.systemValueName = toSlice("instance_index");
                result.permittedTypes.add(builder.getUIntType());
            }
            break;

        case SystemValueSemanticName::IsFrontFace:
            {
                result.systemValueName = toSlice("front_facing");
                result.permittedTypes.add(builder.getBoolType());
            }
            break;

        case SystemValueSemanticName::OutputControlPointID:
        case SystemValueSemanticName::PointSize:
        case SystemValueSemanticName::PointCoord:
            {
                result.isUnsupported = true;
            }
            break;

        case SystemValueSemanticName::Position:
            {
                result.systemValueName = toSlice("position");
                result.permittedTypes.add(builder.getVectorType(
                    builder.getBasicType(BaseType::Float),
                    builder.getIntValue(builder.getIntType(), 4)));
                break;
            }

        case SystemValueSemanticName::PrimitiveID:
        case SystemValueSemanticName::RenderTargetArrayIndex:
            {
                result.isUnsupported = true;
                break;
            }

        case SystemValueSemanticName::SampleIndex:
            {
                result.systemValueName = toSlice("sample_index");
                result.permittedTypes.add(builder.getUIntType());
                break;
            }

        case SystemValueSemanticName::StencilRef:
        case SystemValueSemanticName::Target:
        case SystemValueSemanticName::TessFactor:
            {
                result.isUnsupported = true;
                break;
            }

        case SystemValueSemanticName::VertexID:
            {
                result.systemValueName = toSlice("vertex_index");
                result.permittedTypes.add(builder.getUIntType());
                break;
            }

        case SystemValueSemanticName::WaveLaneCount:
            {
                result.systemValueName = toSlice("subgroup_size");
                result.permittedTypes.add(builder.getUIntType());
                break;
            }

        case SystemValueSemanticName::WaveLaneIndex:
            {
                result.systemValueName = toSlice("subgroup_invocation_id");
                result.permittedTypes.add(builder.getUIntType());
                break;
            }

        case SystemValueSemanticName::ViewID:
        case SystemValueSemanticName::ViewportArrayIndex:
        case SystemValueSemanticName::StartVertexLocation:
        case SystemValueSemanticName::StartInstanceLocation:
            {
                result.isUnsupported = true;
                break;
            }
        default:
            {
                m_sink->diagnose(
                    parentVar,
                    Diagnostics::unimplementedSystemValueSemantic,
                    semanticName);
                return result;
            }
        }

        return result;
    }
    void flattenNestedStructsTransferKeyDecorations(IRInst* newKey, IRInst* oldKey) const
        SLANG_OVERRIDE
    {
        oldKey->transferDecorationsTo(newKey);
    }

    UnownedStringSlice getUserSemanticNameSlice(String& loweredName, bool isUserSemantic) const
        SLANG_OVERRIDE
    {
        return isUserSemantic ? userSemanticName : loweredName.getUnownedSlice();
    }

    void addFragmentShaderReturnValueDecoration(IRBuilder& builder, IRInst* returnValueStructKey)
        const SLANG_OVERRIDE
    {
        IRInst* operands[] = {
            builder.getStringValue(userSemanticName),
            builder.getIntValue(builder.getIntType(), 0)};
        builder.addDecoration(
            returnValueStructKey,
            kIROp_SemanticDecoration,
            operands,
            SLANG_COUNT_OF(operands));
    };

    List<SystemValLegalizationWorkItem> collectSystemValFromEntryPoint(
        EntryPointInfo entryPoint) const SLANG_OVERRIDE
    {
        List<SystemValLegalizationWorkItem> systemValWorkItems;
        for (auto param : entryPoint.entryPointFunc->getParams())
        {
            if (auto structType = as<IRStructType>(param->getDataType()))
            {
                for (auto field : structType->getFields())
                {
                    // Nested struct-s are flattened already by flattenInputParameters().
                    SLANG_ASSERT(!as<IRStructType>(field->getFieldType()));

                    auto key = field->getKey();
                    auto fieldType = field->getFieldType();
                    auto maybeWorkItem = tryToMakeSystemValWorkItem(key, fieldType);
                    if (maybeWorkItem.has_value())
                        systemValWorkItems.add(std::move(maybeWorkItem.value()));
                }
                continue;
            }

            auto maybeWorkItem = tryToMakeSystemValWorkItem(param, param->getFullType());
            if (maybeWorkItem.has_value())
                systemValWorkItems.add(std::move(maybeWorkItem.value()));
        }

        return systemValWorkItems;
    }

private:
    const UnownedStringSlice userSemanticName = toSlice("user_semantic");
};

void legalizeEntryPointVaryingParamsForMetal(
    IRModule* module,
    DiagnosticSink* sink,
    List<EntryPointInfo>& entryPoints)
{
    LegalizeMetalEntryPointContext context(module, sink);
    context.legalizeEntryPoints(entryPoints);
}

void legalizeEntryPointVaryingParamsForWGSL(
    IRModule* module,
    DiagnosticSink* sink,
    List<EntryPointInfo>& entryPoints)
{
    LegalizeWGSLEntryPointContext context(module, sink);
    context.legalizeEntryPoints(entryPoints);
}

} // namespace Slang
