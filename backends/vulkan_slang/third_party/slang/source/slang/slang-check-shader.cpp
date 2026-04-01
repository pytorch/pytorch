// slang-check-shader.cpp
#include "slang-check-impl.h"

// This file encapsulates semantic checking logic primarily
// related to shaders, including validating entry points,
// enumerating specialization parameters, and validating
// attempts to specialize shader code.

#include "slang-lookup.h"

namespace Slang
{
static bool isValidThreadDispatchIDType(Type* type)
{
    // Can accept a single int/unit
    {
        auto basicType = as<BasicExpressionType>(type);
        if (basicType)
        {
            return (
                basicType->getBaseType() == BaseType::Int ||
                basicType->getBaseType() == BaseType::UInt);
        }
    }
    // Can be an int/uint vector from size 1 to 3
    {
        auto vectorType = as<VectorExpressionType>(type);
        if (!vectorType)
        {
            return false;
        }
        auto elemCount = as<ConstantIntVal>(vectorType->getElementCount());
        if (elemCount->getValue() < 1 || elemCount->getValue() > 3)
        {
            return false;
        }
        // Must be a basic type
        auto basicType = as<BasicExpressionType>(vectorType->getElementType());
        if (!basicType)
        {
            return false;
        }

        // Must be integral
        auto baseType = basicType->getBaseType();
        return (baseType == BaseType::Int || baseType == BaseType::UInt);
    }
}

/// Recursively walk `paramDeclRef` and add any existential/interface specialization parameters to
/// `ioSpecializationParams`.
static void _collectExistentialSpecializationParamsRec(
    ASTBuilder* astBuilder,
    SpecializationParams& ioSpecializationParams,
    DeclRef<VarDeclBase> paramDeclRef);

/// Recursively walk `type` and add any existential/interface specialization parameters to
/// `ioSpecializationParams`.
static void _collectExistentialSpecializationParamsRec(
    ASTBuilder* astBuilder,
    SpecializationParams& ioSpecializationParams,
    Type* type,
    SourceLoc loc)
{
    // Whether or not something is an array does not affect
    // the number of existential slots it introduces.
    //
    while (auto arrayType = as<ArrayExpressionType>(type))
    {
        type = arrayType->getElementType();
    }

    if (auto parameterGroupType = as<ParameterGroupType>(type))
    {
        _collectExistentialSpecializationParamsRec(
            astBuilder,
            ioSpecializationParams,
            parameterGroupType->getElementType(),
            loc);
        return;
    }

    if (auto declRefType = as<DeclRefType>(type))
    {
        auto typeDeclRef = declRefType->getDeclRef();
        if (auto interfaceDeclRef = typeDeclRef.as<InterfaceDecl>())
        {
            // Each leaf parameter of interface type adds a specialization
            // parameter, which determines the concrete type(s) that may
            // be provided as arguments for that parameter.
            //
            SpecializationParam specializationParam;
            specializationParam.flavor = SpecializationParam::Flavor::ExistentialType;
            specializationParam.loc = loc;
            specializationParam.object = type;
            ioSpecializationParams.add(specializationParam);
        }
        else if (auto structDeclRef = typeDeclRef.as<StructDecl>())
        {
            // A structure type should recursively introduce
            // existential slots for its fields.
            //
            for (auto fieldDeclRef :
                 getFields(astBuilder, structDeclRef, MemberFilterStyle::Instance))
            {
                _collectExistentialSpecializationParamsRec(
                    astBuilder,
                    ioSpecializationParams,
                    fieldDeclRef);
            }
        }
    }

    // TODO: We eventually need to handle cases like constant
    // buffers and parameter blocks that may have existential
    // element types.
}

static void _collectExistentialSpecializationParamsRec(
    ASTBuilder* astBuilder,
    SpecializationParams& ioSpecializationParams,
    DeclRef<VarDeclBase> paramDeclRef)
{
    _collectExistentialSpecializationParamsRec(
        astBuilder,
        ioSpecializationParams,
        getType(astBuilder, paramDeclRef),
        paramDeclRef.getLoc());
}


/// Collect any interface/existential specialization parameters for `paramDeclRef` into
/// `ioParamInfo` and `ioSpecializationParams`
static void _collectExistentialSpecializationParamsForShaderParam(
    ASTBuilder* astBuilder,
    ShaderParamInfo& ioParamInfo,
    SpecializationParams& ioSpecializationParams,
    DeclRef<VarDeclBase> paramDeclRef)
{
    Index beginParamIndex = ioSpecializationParams.getCount();
    _collectExistentialSpecializationParamsRec(astBuilder, ioSpecializationParams, paramDeclRef);
    Index endParamIndex = ioSpecializationParams.getCount();

    ioParamInfo.firstSpecializationParamIndex = beginParamIndex;
    ioParamInfo.specializationParamCount = endParamIndex - beginParamIndex;
}

void EntryPoint::_collectGenericSpecializationParamsRec(Decl* decl)
{
    if (!decl)
        return;

    _collectGenericSpecializationParamsRec(decl->parentDecl);

    auto genericDecl = as<GenericDecl>(decl);
    if (!genericDecl)
        return;

    for (auto m : genericDecl->members)
    {
        if (auto genericTypeParam = as<GenericTypeParamDecl>(m))
        {
            SpecializationParam param;
            param.flavor = SpecializationParam::Flavor::GenericType;
            param.loc = genericTypeParam->loc;
            param.object = genericTypeParam;
            m_genericSpecializationParams.add(param);
        }
        else if (auto genericValParam = as<GenericValueParamDecl>(m))
        {
            SpecializationParam param;
            param.flavor = SpecializationParam::Flavor::GenericValue;
            param.loc = genericValParam->loc;
            param.object = genericValParam;
            m_genericSpecializationParams.add(param);
        }
    }
}

/// Enumerate the existential-type parameters of an `EntryPoint`.
///
/// Any parameters found will be added to the list of existential slots on `this`.
///
void EntryPoint::_collectShaderParams()
{
    // We don't currently treat an entry point as having any
    // *global* shader parameters.
    //
    // TODO: We could probably clean up the code a bit by treating
    // an entry point as introducing a global shader parameter
    // that is based on the implicit "parameters struct" type
    // of the entry point itself.

    // We collect the generic parameters of the entry point,
    // along with those of any outer generics first.
    //
    _collectGenericSpecializationParamsRec(getFuncDecl());

    // After geneic specialization parameters have been collected,
    // we look through the value parameters of the entry point
    // function and see if any of them introduce existential/interface
    // specialization parameters.
    //
    // Note: we defensively test whether there is a function decl-ref
    // because this routine gets called from the constructor, and
    // a "dummy" entry point will have a null pointer for the function.
    //
    if (auto funcDeclRef = getFuncDeclRef())
    {
        for (auto paramDeclRef : getParameters(getLinkage()->getASTBuilder(), funcDeclRef))
        {
            ShaderParamInfo shaderParamInfo;
            shaderParamInfo.paramDeclRef = paramDeclRef;

            _collectExistentialSpecializationParamsForShaderParam(
                getLinkage()->getASTBuilder(),
                shaderParamInfo,
                m_existentialSpecializationParams,
                paramDeclRef);

            m_shaderParams.add(shaderParamInfo);
        }
    }
}

bool isPrimaryDecl(CallableDecl* decl)
{
    SLANG_ASSERT(decl);
    return (!decl->primaryDecl) || (decl == decl->primaryDecl);
}

DeclRef<FuncDecl> findFunctionDeclByName(Module* translationUnit, Name* name, DiagnosticSink* sink)
{
    DeclRef<FuncDecl> entryPointFuncDeclRef;

    auto expr = translationUnit->findDeclFromString(getText(name), sink);
    if (auto declRefExpr = as<DeclRefExpr>(expr))
    {
        entryPointFuncDeclRef = declRefExpr->declRef.as<FuncDecl>();

        if (entryPointFuncDeclRef && getModule(entryPointFuncDeclRef.getDecl()) != translationUnit)
            entryPointFuncDeclRef = DeclRef<FuncDecl>();
    }

    if (!entryPointFuncDeclRef)
    {
        auto translationUnitSyntax = translationUnit->getModuleDecl();
        sink->diagnose(translationUnitSyntax, Diagnostics::entryPointFunctionNotFound, name);
    }
    return entryPointFuncDeclRef;
}

// Is a entry pointer parmaeter of `type` always a uniform parameter?
bool isUniformParameterType(Type* type)
{
    if (as<ResourceType>(type))
        return true;
    if (as<SubpassInputType>(type))
        return true;
    if (as<HLSLStructuredBufferTypeBase>(type))
        return true;
    if (as<UntypedBufferResourceType>(type))
        return true;
    if (as<UniformParameterGroupType>(type))
        return true;
    if (as<GLSLShaderStorageBufferType>(type))
        return true;
    if (as<SamplerStateType>(type))
        return true;
    if (as<PtrType>(type))
        return true;
    if (auto arrayType = as<ArrayExpressionType>(type))
        return isUniformParameterType(arrayType->getElementType());
    if (auto modType = as<ModifiedType>(type))
        return isUniformParameterType(modType->getBase());
    return false;
}

bool isBuiltinParameterType(Type* type)
{
    if (!as<BuiltinType>(type))
        return false;
    if (as<BasicExpressionType>(type))
        return false;
    if (as<VectorExpressionType>(type))
        return false;
    if (as<MatrixExpressionType>(type))
        return false;
    if (auto arrayType = as<ArrayExpressionType>(type))
        return isBuiltinParameterType(arrayType->getElementType());
    return true;
}

bool doStructFieldsHaveSemanticImpl(Type* type, HashSet<Type*>& seenTypes)
{
    auto declRefType = as<DeclRefType>(type);
    if (!declRefType)
        return false;
    auto structDecl = as<StructDecl>(declRefType->getDeclRef().getDecl());
    if (!structDecl)
        return false;
    seenTypes.add(type);
    bool hasFields = false;
    for (auto field : structDecl->getFields())
    {
        hasFields = true;
        if (!field->findModifier<HLSLSemantic>())
        {
            if (!seenTypes.contains(field->getType()))
            {
                if (!doStructFieldsHaveSemanticImpl(field->getType(), seenTypes))
                    return false;
            }
        }
    }
    return hasFields;
}

bool doStructFieldsHaveSemantic(Type* type)
{
    HashSet<Type*> seenTypes;
    return doStructFieldsHaveSemanticImpl(type, seenTypes);
}

// Validate that an entry point function conforms to any additional
// constraints based on the stage (and profile?) it specifies.
void validateEntryPoint(EntryPoint* entryPoint, DiagnosticSink* sink)
{
    auto entryPointFuncDecl = entryPoint->getFuncDecl();
    auto stage = entryPoint->getStage();

    // TODO: We currently do minimal checking here, but this is the
    // right place to perform the following validation checks:
    //

    // * Are the function input/output parameters and result type
    //   all valid for the chosen stage? (e.g., there shouldn't be
    //   an `OutputStream<X>` type in a vertex shader signature)
    //
    // * For any varying input/output, are there semantics specified
    //   (Note: this potentially overlaps with layout logic...), and
    //   are the system-value semantics valid for the given stage?
    //
    //   There's actually a lot of detail to semantic checking, in
    //   that the AST-level code should probably be validating the
    //   use of system-value semantics by linking them to explicit
    //   declarations in the core module. We should also be
    //   using profile information on those declarations to infer
    //   appropriate profile restrictions on the entry point.
    //
    // * Is the entry point actually usable on the given stage/profile?
    //   E.g., if we have a vertex shader that (transitively) calls
    //   `Texture2D.Sample`, then that should produce an error because
    //   that function is specific to the fragment profile/stage.
    //

    auto entryPointName = entryPointFuncDecl->getName();

    auto module = getModule(entryPointFuncDecl);
    auto linkage = module->getLinkage();

    // Every entry point needs to have a stage specified either via
    // command-line/API options, or via an explicit `[shader("...")]` attribute.
    //
    if (stage == Stage::Unknown)
    {
        sink->diagnose(entryPointFuncDecl, Diagnostics::entryPointHasNoStage, entryPointName);
    }

    if (stage == Stage::Hull)
    {
        // TODO: We could consider *always* checking any `[patchconsantfunc("...")]`
        // attributes, so that they need to resolve to a function.

        auto attr = entryPointFuncDecl->findModifier<PatchConstantFuncAttribute>();

        if (attr)
        {
            if (attr->args.getCount() != 1)
            {
                sink->diagnose(attr, Diagnostics::badlyDefinedPatchConstantFunc, entryPointName);
                return;
            }

            Expr* expr = attr->args[0];
            StringLiteralExpr* stringLit = as<StringLiteralExpr>(expr);

            if (!stringLit)
            {
                sink->diagnose(expr, Diagnostics::badlyDefinedPatchConstantFunc, entryPointName);
                return;
            }

            // We look up the patch-constant function by its name in the module
            // scope of the translation unit that declared the HS entry point.
            //
            // TODO: Eventually we probably want to do the lookup in the scope
            // of the parent declarations of the entry point. E.g., if the entry
            // point is a member function of a `struct`, then its patch-constant
            // function should be allowed to be another member function of
            // the same `struct`.
            //
            // In the extremely long run we may want to support an alternative to
            // this attribute-based linkage between the two functions that
            // make up the entry point.
            //
            Name* name = linkage->getNamePool()->getName(stringLit->value);
            DeclRef<FuncDecl> patchConstantFuncDeclRef = findFunctionDeclByName(module, name, sink);
            if (!patchConstantFuncDeclRef)
            {
                sink->diagnose(
                    expr,
                    Diagnostics::attributeFunctionNotFound,
                    name,
                    "patchconstantfunc");
                return;
            }

            attr->patchConstantFuncDecl = patchConstantFuncDeclRef.getDecl();
        }
    }
    else if (stage == Stage::Compute)
    {
        for (const auto& param : entryPointFuncDecl->getParameters())
        {
            if (auto semantic = param->findModifier<HLSLSimpleSemantic>())
            {
                const auto& semanticToken = semantic->name;

                String lowerName = String(semanticToken.getContent()).toLower();

                if (lowerName == "sv_dispatchthreadid")
                {
                    Type* paramType = param->getType();

                    if (!isValidThreadDispatchIDType(paramType))
                    {
                        String typeString = paramType->toString();
                        sink->diagnose(
                            param->loc,
                            Diagnostics::invalidDispatchThreadIDType,
                            typeString);
                        return;
                    }
                }
            }
        }
    }

    bool canHaveVaryingInput = false;
    bool shouldWarnOnNonUniformParam = true;
    switch (stage)
    {
    case Stage::Vertex:
    case Stage::Fragment:
    case Stage::Miss:
    case Stage::AnyHit:
    case Stage::ClosestHit:
    case Stage::Callable:
    case Stage::Geometry:
    case Stage::Mesh:
    case Stage::Hull:
    case Stage::Domain:
        canHaveVaryingInput = true;
        break;
    case Stage::Dispatch:
        shouldWarnOnNonUniformParam = false;
        break;
    default:
        break;
    }

    for (const auto& param : entryPointFuncDecl->getParameters())
    {
        if (isUniformParameterType(param->getType()))
        {
            // Automatically add `uniform` modifier to entry point parameters.
            if (!param->hasModifier<HLSLUniformModifier>())
            {
                addModifier(param, getCurrentASTBuilder()->create<HLSLUniformModifier>());
                continue;
            }
        }

        if (canHaveVaryingInput)
            continue;

        // If the stage doesn't allow varying input/output,
        // we require the parameter to be associated with a system value semantic.
        if (param->hasModifier<HLSLUniformModifier>())
            continue;
        if (param->findModifier<HLSLSemantic>())
            continue;

        bool isBuiltinType = isBuiltinParameterType(param->getType());
        if (isBuiltinType)
            continue;

        if (doStructFieldsHaveSemantic(param->getType()))
            continue;

        // The user is defining a parameter with no 'uniform' modifier for a stage that doesn't
        // support varying input/output. We will automatically convert it to a 'uniform' parameter,
        // and diagnose a warning.
        addModifier(param, getCurrentASTBuilder()->create<HLSLUniformModifier>());
        if (shouldWarnOnNonUniformParam)
        {
            sink->diagnose(
                param,
                Diagnostics::nonUniformEntryPointParameterTreatedAsUniform,
                param->getName());
        }
    }

    for (auto target : linkage->targets)
    {
        auto targetCaps = target->getTargetCaps();
        auto stageCapabilitySet = entryPoint->getProfile().getCapabilityName();
        targetCaps.join(stageCapabilitySet);
        if (targetCaps.isIncompatibleWith(entryPointFuncDecl->inferredCapabilityRequirements))
        {
            // Incompatable means we don't support a set of abstract atoms.
            // Diagnose that we lack support for 'stage' and 'target' atoms with our provided
            // entry-point
            auto compileTarget = target->getTargetCaps().getCompileTarget();
            auto stageTarget = stageCapabilitySet.getTargetStage();
            maybeDiagnose(
                sink,
                linkage->m_optionSet,
                DiagnosticCategory::Capability,
                entryPointFuncDecl,
                Diagnostics::entryPointUsesUnavailableCapability,
                entryPointFuncDecl,
                compileTarget,
                stageTarget);

            // Find out what is incompatible (ancestor missing a super set of 'target+stage')
            CapabilitySet failedSet({(CapabilityName)compileTarget, (CapabilityName)stageTarget});
            diagnoseMissingCapabilityProvenance(
                linkage->m_optionSet,
                sink,
                entryPointFuncDecl,
                failedSet);
        }
        else
        {
            auto& targetOptionSet = target->getOptionSet();
            bool specificProfileRequested =
                targetOptionSet.hasOption(CompilerOptionName::Profile) &&
                (targetOptionSet.getIntOption(CompilerOptionName::Profile) !=
                 SLANG_PROFILE_UNKNOWN);
            bool specificCapabilityRequested =
                targetOptionSet.hasOption(CompilerOptionName::Capability) &&
                (targetOptionSet.getIntOption(CompilerOptionName::Capability) !=
                 SLANG_CAPABILITY_UNKNOWN);
            // Only attempt to error if a specific profile or capability is requested
            if ((specificCapabilityRequested || specificProfileRequested) &&
                targetCaps.atLeastOneSetImpliedInOther(
                    entryPointFuncDecl->inferredCapabilityRequirements) ==
                    CapabilitySet::ImpliesReturnFlags::NotImplied)
            {
                CapabilitySet combinedSets = targetCaps;
                combinedSets.join(entryPointFuncDecl->inferredCapabilityRequirements);
                CapabilityAtomSet addedAtoms{};
                if (auto targetCapSet = targetCaps.getAtomSets())
                {
                    if (auto combinedSet = combinedSets.getAtomSets())
                    {
                        CapabilityAtomSet::calcSubtract(
                            addedAtoms,
                            (*combinedSet),
                            (*targetCapSet));
                    }
                }
                maybeDiagnoseWarningOrError(
                    sink,
                    target->getOptionSet(),
                    DiagnosticCategory::Capability,
                    entryPointFuncDecl->loc,
                    Diagnostics::profileImplicitlyUpgraded,
                    Diagnostics::profileImplicitlyUpgradedRestrictive,
                    entryPointFuncDecl,
                    target->getOptionSet().getProfile().getName(),
                    addedAtoms.getElements<CapabilityAtom>());
            }
        }
    }
}

bool resolveStageOfProfileWithEntryPoint(
    Profile& entryPointProfile,
    CompilerOptionSet& optionSet,
    const List<RefPtr<TargetRequest>>& targets,
    FuncDecl* entryPointFuncDecl,
    DiagnosticSink* sink)
{
    if (auto entryPointAttr = entryPointFuncDecl->findModifier<EntryPointAttribute>())
    {
        auto entryPointProfileStage = entryPointProfile.getStage();
        auto entryPointStage = getStageFromAtom(entryPointAttr->capabilitySet.getTargetStage());

        // Ensure every target is specifying the same stage as an entry-point
        // if a profile+stage was set, else user will not be aware that their
        // code is requiring `fragment` on a `vertex` shader
        for (auto target : targets)
        {
            auto targetProfile = target->getOptionSet().getProfile();
            auto profileStage = targetProfile.getStage();
            if (profileStage != Stage::Unknown && profileStage != entryPointStage)
                maybeDiagnose(
                    sink,
                    optionSet,
                    DiagnosticCategory::Capability,
                    entryPointAttr,
                    Diagnostics::entryPointAndProfileAreIncompatible,
                    entryPointFuncDecl,
                    entryPointStage,
                    targetProfile.getName());
        }
        if (entryPointProfileStage == Stage::Unknown)
            entryPointProfile = Profile(entryPointStage);
        else if (
            entryPointProfileStage != Stage::Unknown && entryPointProfileStage != entryPointStage)
            maybeDiagnose(
                sink,
                optionSet,
                DiagnosticCategory::Capability,
                entryPointFuncDecl,
                Diagnostics::specifiedStageDoesntMatchAttribute,
                entryPointFuncDecl->getName(),
                entryPointProfileStage,
                entryPointStage);
        entryPointProfile.additionalCapabilities.add(entryPointAttr->capabilitySet);
        return true;
    }
    return false;
}

// Given an entry point specified via API or command line options,
// attempt to find a matching AST declaration that implements the specified
// entry point. If such a function is found, then validate that it actually
// meets the requirements for the selected stage/profile.
//
// Returns an `EntryPoint` object representing the (unspecialized)
// entry point if it is found and validated, and null otherwise.
//
RefPtr<EntryPoint> findAndValidateEntryPoint(FrontEndEntryPointRequest* entryPointReq)
{
    // The first step in validating the entry point is to find
    // the (unique) function declaration that matches its name.
    //
    // TODO: We may eventually want/need to extend this to
    // account for nested names like `SomeStruct.vsMain`, or
    // indeed even to handle generics.
    //
    auto compileRequest = entryPointReq->getCompileRequest();
    auto translationUnit = entryPointReq->getTranslationUnit();
    auto linkage = compileRequest->getLinkage();
    auto sink = compileRequest->getSink();

    auto entryPointName = entryPointReq->getName();
    DeclRef<FuncDecl> entryPointFuncDeclRef =
        findFunctionDeclByName(translationUnit->getModule(), entryPointName, sink);

    // Did we find a function declaration in our search?
    if (!entryPointFuncDeclRef)
    {
        return nullptr;
    }

    // TODO: it is possible that the entry point was declared with
    // profile or target overloading. Is there anything that we need
    // to do at this point to filter out declarations that aren't
    // relevant to the selected profile for the entry point?

    // We found something, and can start doing some basic checking.
    //
    // If the entry point specifies a stage via a `[shader("...")]` attribute,
    // then we might be able to infer a stage for the entry point request if
    // it didn't have one, *or* issue a diagnostic if there is a mismatch with the profile.

    auto entryPointProfile = entryPointReq->getProfile();
    resolveStageOfProfileWithEntryPoint(
        entryPointProfile,
        linkage->m_optionSet,
        linkage->targets,
        entryPointFuncDeclRef.getDecl(),
        sink);
    // TODO: Should we attach a `[shader(...)]` attribute to an
    // entry point that didn't have one, so that we can have
    // a more uniform representation in the AST?

    RefPtr<EntryPoint> entryPoint =
        EntryPoint::create(linkage, entryPointFuncDeclRef, entryPointProfile);

    // Now that we've *found* the entry point, it is time to validate
    // that it actually meets the constraints for the chosen stage/profile.
    //
    validateEntryPoint(entryPoint, sink);

    // We should return nullptr if entry point fails to validate
    if (sink->getErrorCount())
    {
        return nullptr;
    }

    return entryPoint;
}

/// Get the name a variable will use for reflection purposes
Name* getReflectionName(VarDeclBase* varDecl)
{
    if (auto reflectionNameModifier = varDecl->findModifier<ParameterGroupReflectionName>())
        return reflectionNameModifier->nameAndLoc.name;

    return varDecl->getName();
}

Type* getParamType(ASTBuilder* astBuilder, DeclRef<VarDeclBase> paramDeclRef)
{
    auto paramType = getType(astBuilder, paramDeclRef);
    if (paramDeclRef.getDecl()->findModifier<NoDiffModifier>())
    {
        auto modifierVal = static_cast<Val*>(astBuilder->getOrCreate<NoDiffModifierVal>());
        paramType = astBuilder->getModifiedType(paramType, 1, &modifierVal);
    }
    return paramType;
}

Type* getParamTypeWithDirectionWrapper(ASTBuilder* astBuilder, DeclRef<VarDeclBase> paramDeclRef)
{
    auto result = getParamType(astBuilder, paramDeclRef);
    auto direction = getParameterDirection(paramDeclRef.getDecl());
    switch (direction)
    {
    case kParameterDirection_In:
        return result;
    case kParameterDirection_ConstRef:
        return astBuilder->getConstRefType(result);
    case kParameterDirection_Out:
        return astBuilder->getOutType(result);
    case kParameterDirection_InOut:
        return astBuilder->getInOutType(result);
    case kParameterDirection_Ref:
        return astBuilder->getRefType(result, AddressSpace::Generic);
    default:
        return result;
    }
}

void Module::_collectShaderParams()
{
    // We are going to walk the global declarations in the body of the
    // module, and use those to build up our lists of:
    //
    // * Global shader parameters
    // * Specialization parameters (both generic and interface/existential)
    // * Requirements (`import`ed modules)
    //
    // For requirements, we want to be careful to only
    // add each required module once (in case the same
    // module got `import`ed multiple times), so we
    // will keep a set of the modules we've already
    // seen and processed.
    //

    // We need to use a work list to traverse through all global scopes,
    // including the top level `moduleDecl` and all the included `FileDecl`s.

    List<ContainerDecl*> workList;
    workList.add(m_moduleDecl);

    HashSet<Module*> requiredModuleSet;
    for (Index i = 0; i < workList.getCount(); i++)
    {
        auto moduleDecl = workList[i];
        for (auto globalDecl : moduleDecl->members)
        {
            if (auto globalVar = as<VarDecl>(globalDecl))
            {
                // We do not want to consider global variable declarations
                // that don't represents shader parameters. This includes
                // things like `static` globals and `groupshared` variables.
                //
                if (!isGlobalShaderParameter(globalVar))
                {
                    bool isVarying = false;
                    for (auto m : globalVar->modifiers)
                    {
                        if (as<InModifier>(m) || as<OutModifier>(m))
                        {
                            isVarying = true;
                            break;
                        }
                    }
                    if (!isVarying)
                        continue;
                }

                // At this point we know we have a global shader parameter.

                ShaderParamInfo shaderParamInfo;
                shaderParamInfo.paramDeclRef = makeDeclRef(globalVar);

                // We need to consider what specialization parameters
                // are introduced by this shader parameter. This step
                // fills in fields on `shaderParamInfo` so that we
                // can assocaite specialization arguments supplied later
                // with the correct parameter.
                //
                _collectExistentialSpecializationParamsForShaderParam(
                    getLinkage()->getASTBuilder(),
                    shaderParamInfo,
                    m_specializationParams,
                    makeDeclRef(globalVar));

                m_shaderParams.add(shaderParamInfo);
            }
            else if (auto globalGenericParam = as<GlobalGenericParamDecl>(globalDecl))
            {
                // A global generic type parameter declaration introduces
                // a suitable specialization parameter.
                //
                SpecializationParam specializationParam;
                specializationParam.flavor = SpecializationParam::Flavor::GenericType;
                specializationParam.loc = globalGenericParam->loc;
                specializationParam.object = globalGenericParam;
                m_specializationParams.add(specializationParam);
            }
            else if (auto globalGenericValueParam = as<GlobalGenericValueParamDecl>(globalDecl))
            {
                // A global generic type parameter declaration introduces
                // a suitable specialization parameter.
                //
                SpecializationParam specializationParam;
                specializationParam.flavor = SpecializationParam::Flavor::GenericValue;
                specializationParam.loc = globalGenericValueParam->loc;
                specializationParam.object = globalGenericValueParam;
                m_specializationParams.add(specializationParam);
            }
            else if (auto importDecl = as<ImportDecl>(globalDecl))
            {
                // An `import` declaration creates a requirement dependency
                // from this module to another module.
                //
                auto importedModule = getModule(importDecl->importedModuleDecl);
                if (!requiredModuleSet.contains(importedModule))
                {
                    requiredModuleSet.add(importedModule);
                    m_requirements.add(importedModule);
                }
            }
            else if (auto fileDecl = as<FileDecl>(globalDecl))
            {
                // If we see a `FileDecl`, we need to recursively look into its
                // scope.
                workList.add(fileDecl);
            }
            else if (auto namespaceDecl = as<NamespaceDecl>(globalDecl))
            {
                workList.add(namespaceDecl);
            }
        }
    }
}

Index Module::getRequirementCount()
{
    return m_requirements.getCount();
}

RefPtr<ComponentType> Module::getRequirement(Index index)
{
    return m_requirements[index];
}

void Module::acceptVisitor(ComponentTypeVisitor* visitor, SpecializationInfo* specializationInfo)
{
    visitor->visitModule(this, as<ModuleSpecializationInfo>(specializationInfo));
}


/// Create a new component type based on `inComponentType`, but with all its requiremetns filled.
RefPtr<ComponentType> fillRequirements(ComponentType* inComponentType)
{
    auto linkage = inComponentType->getLinkage();

    // We are going to simplify things by solving the problem iteratively.
    // If the current `componentType` has requirements for `A`, `B`, ... etc.
    // then we will create a composite of `componentType`, `A`, `B`, ...
    // and then see if the resulting composite has any requirements.
    //
    // This avoids the problem of trying to compute teh transitive closure
    // of the requirements relationship (while dealing with deduplication,
    // etc.)

    RefPtr<ComponentType> componentType = inComponentType;
    for (;;)
    {
        auto requirementCount = componentType->getRequirementCount();
        if (requirementCount == 0)
            break;

        List<RefPtr<ComponentType>> allComponents;
        allComponents.add(componentType);

        for (Index rr = 0; rr < requirementCount; ++rr)
        {
            auto requirement = componentType->getRequirement(rr);
            allComponents.add(requirement);
        }

        componentType = CompositeComponentType::create(linkage, allComponents);
    }
    return componentType;
}

/// Create a component type to represent the "global scope" of a compile request.
///
/// This component type will include all the modules and their global
/// parameters from the compile request, but not anything specific
/// to any entry point functions.
///
/// The layout for this component type will thus represent the things that
/// a user is likely to want to have stay the same across all compiled
/// entry points.
///
/// The component type that this function creates is unspecialized, in
/// that it doesn't take into account any specialization arguments
/// that might have been supplied as part of the compile request.
///
RefPtr<ComponentType> createUnspecializedGlobalComponentType(FrontEndCompileRequest* compileRequest)
{
    // We want our resulting program to depend on
    // all the translation units the user specified,
    // even if some of them don't contain entry points
    // (this is important for parameter layout/binding).
    //
    // We also want to ensure that the modules for the
    // translation units comes first in the enumerated
    // order for dependencies, to match the pre-existing
    // compiler behavior (at least for now).
    //
    auto linkage = compileRequest->getLinkage();

    RefPtr<ComponentType> globalComponentType;
    if (compileRequest->translationUnits.getCount() == 1)
    {
        // The common case is that a compilation only uses
        // a single translation unit, and thus results in
        // a single `Module`. We can then use that module
        // as the component type that represents the global scope.
        //
        globalComponentType = compileRequest->translationUnits[0]->getModule();
    }
    else
    {
        List<RefPtr<ComponentType>> translationUnitComponentTypes;
        for (auto tu : compileRequest->translationUnits)
        {
            translationUnitComponentTypes.add(tu->getModule());
        }

        globalComponentType =
            CompositeComponentType::create(linkage, translationUnitComponentTypes);
    }

    return fillRequirements(globalComponentType);
}

void FrontEndCompileRequest::checkEntryPoints()
{
    auto linkage = getLinkage();
    SLANG_AST_BUILDER_RAII(linkage->getASTBuilder());

    auto sink = getSink();

    // The validation of entry points here will be modal, and controlled
    // by whether the user specified any entry points directly via
    // API or command-line options.
    //
    // TODO: We may want to make this choice explicit rather than implicit.
    //
    // First, check if the user requested any entry points explicitly via
    // the API or command line.
    //
    bool anyExplicitEntryPoints = getEntryPointReqCount() != 0;

    if (anyExplicitEntryPoints)
    {
        // If there were any explicit requests for entry points to be
        // checked, then we will *only* check those.
        //
        for (auto entryPointReq : getEntryPointReqs())
        {
            auto entryPoint = findAndValidateEntryPoint(entryPointReq);
            if (entryPoint)
            {
                // TODO: We need to implement an explicit policy
                // for what should happen if the user specified
                // entry points via the command-line (or API),
                // but didn't specify any groups (since the current
                // compilation API doesn't allow for grouping).
                //
                entryPointReq->getTranslationUnit()->module->_addEntryPoint(entryPoint);
            }
        }

        // TODO: We should consider always processing both categories,
        // and just making sure to only check each entry point function
        // declaration once...
    }
    else
    {
        // Otherwise, scan for any `[shader(...)]` attributes in
        // the user's code, and construct `EntryPoint`s to
        // represent them.
        //
        // This ensures that downstream code only has to consider
        // the central list of entry point requests, and doesn't
        // have to know where they came from.

        // TODO: A comprehensive approach here would need to search
        // recursively for entry points, because they might appear
        // as, e.g., member function of a `struct` type.
        //
        // For now we'll start with an extremely basic approach that
        // should work for typical HLSL code.
        //
        Index translationUnitCount = translationUnits.getCount();
        for (Index tt = 0; tt < translationUnitCount; ++tt)
        {
            auto translationUnit = translationUnits[tt];
            translationUnit->getModule()->_discoverEntryPoints(sink, this->getLinkage()->targets);
        }
    }
}


/// Create a component type that represents the global scope for a compile request,
/// along with any entry point functions.
///
/// The resulting component type will include the global-scope information
/// first, so its layout will be compatible with the result of
/// `createUnspecializedGlobalComponentType`.
///
/// The new component type will also add on any entry-point functions
/// that were requested and will thus include space for their `uniform` parameters.
/// If multiple entry points were requested then they will be given non-overlapping
/// parameter bindings, consistent with them being used together in
/// a single pipeline state, hit group, etc.
///
/// The result of this function is unspecialized and doesn't take into
/// account any specialization arguments the user might have supplied.
///
RefPtr<ComponentType> createUnspecializedGlobalAndEntryPointsComponentType(
    FrontEndCompileRequest* compileRequest,
    List<RefPtr<ComponentType>>& outUnspecializedEntryPoints)
{
    auto linkage = compileRequest->getLinkage();

    auto globalComponentType = compileRequest->getGlobalComponentType();

    List<RefPtr<ComponentType>> allComponentTypes;
    allComponentTypes.add(globalComponentType);

    Index translationUnitCount = compileRequest->translationUnits.getCount();
    for (Index tt = 0; tt < translationUnitCount; ++tt)
    {
        auto translationUnit = compileRequest->translationUnits[tt];
        auto module = translationUnit->getModule();

        for (auto entryPoint : module->getEntryPoints())
        {
            outUnspecializedEntryPoints.add(entryPoint);
            allComponentTypes.add(entryPoint);
        }
    }

    // Also consider entry points that were introduced via adding
    // a library reference...
    //
    for (auto extraEntryPoint : compileRequest->m_extraEntryPoints)
    {
        auto entryPoint = EntryPoint::createDummyForDeserialize(
            linkage,
            extraEntryPoint.name,
            extraEntryPoint.profile,
            extraEntryPoint.mangledName);
        allComponentTypes.add(entryPoint);
    }

    if (allComponentTypes.getCount() > 1)
    {
        auto composite = CompositeComponentType::create(linkage, allComponentTypes);
        return composite;
    }
    else
    {
        return globalComponentType;
    }
}

RefPtr<ComponentType::SpecializationInfo> Module::_validateSpecializationArgsImpl(
    SpecializationArg const* args,
    Index argCount,
    DiagnosticSink* sink)
{
    SLANG_ASSERT(argCount == getSpecializationParamCount());

    SharedSemanticsContext semanticsContext(getLinkage(), this, sink);
    SemanticsVisitor visitor(&semanticsContext);

    RefPtr<Module::ModuleSpecializationInfo> specializationInfo =
        new Module::ModuleSpecializationInfo();

    for (Index ii = 0; ii < argCount; ++ii)
    {
        auto& arg = args[ii];
        auto& param = m_specializationParams[ii];

        switch (param.flavor)
        {
        case SpecializationParam::Flavor::GenericType:
            {
                auto genericTypeParamDecl = as<GlobalGenericParamDecl>(param.object);
                SLANG_ASSERT(genericTypeParamDecl);

                Type* argType = as<Type>(arg.val);
                if (!argType)
                {
                    sink->diagnose(
                        param.loc,
                        Diagnostics::expectedTypeForSpecializationArg,
                        genericTypeParamDecl);
                    argType = getLinkage()->getASTBuilder()->getErrorType();
                }

                // TODO: There is a serious flaw to this checking logic if we ever have cases where
                // the constraints on one `type_param` can depend on another `type_param`, e.g.:
                //
                //      type_param A;
                //      type_param B : ISidekick<A>;
                //
                // In that case, if a user tries to set `B` to `Robin` and `Robin` conforms to
                // `ISidekick<Batman>`, then the compiler needs to know whether `A` is being
                // set to `Batman` to know whether the setting for `B` is valid. In this limit
                // the constraints can be mutually recursive (so `A : IMentor<B>`).
                //
                // The only way to check things correctly is to validate each conformance under
                // a set of assumptions (substitutions) that includes all the type substitutions,
                // and possibly also all the other constraints *except* the one to be validated.
                //
                // We will punt on this for now, and just check each constraint in isolation.

                // As a quick sanity check, see if the argument that is being supplied for a
                // global generic type parameter is a reference to *another* global generic
                // type parameter, since that should always be an error.
                //
                if (auto argDeclRefType = as<DeclRefType>(argType))
                {
                    auto argDeclRef = argDeclRefType->getDeclRef();
                    if (auto argGenericParamDeclRef = argDeclRef.as<GlobalGenericParamDecl>())
                    {
                        if (argGenericParamDeclRef.getDecl() == genericTypeParamDecl)
                        {
                            // We are trying to specialize a generic parameter using itself.
                            sink->diagnose(
                                genericTypeParamDecl,
                                Diagnostics::cannotSpecializeGlobalGenericToItself,
                                genericTypeParamDecl->getName());
                            continue;
                        }
                        else
                        {
                            // We are trying to specialize a generic parameter using a *different*
                            // global generic type parameter.
                            sink->diagnose(
                                genericTypeParamDecl,
                                Diagnostics::cannotSpecializeGlobalGenericToAnotherGenericParam,
                                genericTypeParamDecl->getName(),
                                argGenericParamDeclRef.getName());
                            continue;
                        }
                    }
                }

                ModuleSpecializationInfo::GenericArgInfo genericArgInfo;
                genericArgInfo.paramDecl = genericTypeParamDecl;
                genericArgInfo.argVal = argType;
                specializationInfo->genericArgs.add(genericArgInfo);

                // Walk through the declared constraints for the parameter,
                // and check that the argument actually satisfies them.
                for (auto constraintDecl :
                     genericTypeParamDecl->getMembersOfType<GenericTypeConstraintDecl>())
                {
                    // Get the type that the constraint is enforcing conformance to
                    auto interfaceType = getSup(
                        getLinkage()->getASTBuilder(),
                        DeclRef<GenericTypeConstraintDecl>(constraintDecl));

                    // Use our semantic-checking logic to search for a witness to the required
                    // conformance
                    auto witness =
                        visitor.isSubtype(argType, interfaceType, IsSubTypeOptions::None);
                    if (!witness)
                    {
                        // If no witness was found, then we will be unable to satisfy
                        // the conformances required.
                        sink->diagnose(
                            genericTypeParamDecl,
                            Diagnostics::typeArgumentForGenericParameterDoesNotConformToInterface,
                            argType,
                            genericTypeParamDecl->nameAndLoc.name,
                            interfaceType);
                    }

                    ModuleSpecializationInfo::GenericArgInfo constraintArgInfo;
                    constraintArgInfo.paramDecl = constraintDecl;
                    constraintArgInfo.argVal = witness;
                    specializationInfo->genericArgs.add(constraintArgInfo);
                }
            }
            break;

        case SpecializationParam::Flavor::ExistentialType:
            {
                auto interfaceType = as<Type>(param.object);
                SLANG_ASSERT(interfaceType);

                Type* argType = as<Type>(arg.val);
                if (!argType)
                {
                    sink->diagnose(
                        param.loc,
                        Diagnostics::expectedTypeForSpecializationArg,
                        interfaceType);
                    argType = getLinkage()->getASTBuilder()->getErrorType();
                }

                auto witness = visitor.isSubtype(argType, interfaceType, IsSubTypeOptions::None);
                if (!witness)
                {
                    // If no witness was found, then we will be unable to satisfy
                    // the conformances required.
                    sink->diagnose(
                        SourceLoc(),
                        Diagnostics::typeArgumentDoesNotConformToInterface,
                        argType,
                        interfaceType);
                }

                ExpandedSpecializationArg expandedArg;
                expandedArg.val = argType;
                expandedArg.witness = witness;

                specializationInfo->existentialArgs.add(expandedArg);
            }
            break;

        case SpecializationParam::Flavor::GenericValue:
            {
                auto paramDecl = as<GlobalGenericValueParamDecl>(param.object);
                SLANG_ASSERT(paramDecl);

                // Now we need to check that the argument `Val` has the
                // appropriate type expected by the parameter.

                IntVal* intVal = as<IntVal>(arg.val);
                if (!intVal)
                {
                    sink->diagnose(
                        param.loc,
                        Diagnostics::expectedValueOfTypeForSpecializationArg,
                        paramDecl->getType(),
                        paramDecl);
                    intVal =
                        getLinkage()->getASTBuilder()->getIntVal(m_astBuilder->getIntType(), 0);
                }

                ModuleSpecializationInfo::GenericArgInfo expandedArg;
                expandedArg.paramDecl = paramDecl;
                expandedArg.argVal = intVal;

                specializationInfo->genericArgs.add(expandedArg);
            }
            break;

        default:
            SLANG_UNEXPECTED("unhandled specialization parameter flavor");
        }
    }

    return specializationInfo;
}


static void _extractSpecializationArgs(
    ComponentType* componentType,
    List<Expr*> const& argExprs,
    List<SpecializationArg>& outArgs,
    DiagnosticSink* sink)
{
    auto linkage = componentType->getLinkage();

    SharedSemanticsContext semanticsContext(linkage, nullptr, sink);
    SemanticsVisitor semanticsVisitor(&semanticsContext);

    auto argCount = argExprs.getCount();
    for (Index ii = 0; ii < argCount; ++ii)
    {
        auto argExpr = argExprs[ii];

        SpecializationArg arg;
        arg.val = semanticsVisitor.ExtractGenericArgVal(argExpr);
        outArgs.add(arg);
    }
}

RefPtr<ComponentType::SpecializationInfo> EntryPoint::_validateSpecializationArgsImpl(
    SpecializationArg const* inArgs,
    Index inArgCount,
    DiagnosticSink* sink)
{
    auto args = inArgs;
    auto argCount = inArgCount;

    SharedSemanticsContext sharedSemanticsContext(getLinkage(), nullptr, sink);
    SemanticsVisitor visitor(&sharedSemanticsContext);

    // The first N arguments will be for the explicit generic parameters
    // of the entry point (if it has any).
    //
    auto genericSpecializationParamCount = getGenericSpecializationParamCount();
    SLANG_ASSERT(argCount >= genericSpecializationParamCount);

    RefPtr<EntryPointSpecializationInfo> info = new EntryPointSpecializationInfo();

    DeclRef<FuncDecl> specializedFuncDeclRef = m_funcDeclRef;
    if (genericSpecializationParamCount)
    {
        // We need to construct a generic application and use
        // the semantic checking machinery to expand out
        // the rest of the arguments via inference...

        auto genericDeclRef = m_funcDeclRef.getParent().as<GenericDecl>();
        SLANG_ASSERT(genericDeclRef); // otherwise we wouldn't have generic parameters

        List<Val*> genericArgs;

        for (Index ii = 0; ii < genericSpecializationParamCount; ++ii)
        {
            auto specializationArg = args[ii];
            genericArgs.add(specializationArg.val);
        }
        auto astBuilder = getLinkage()->getASTBuilder();
        for (auto constraintDecl : getMembersOfType<GenericTypeConstraintDecl>(
                 getLinkage()->getASTBuilder(),
                 DeclRef<ContainerDecl>(genericDeclRef)))
        {
            DeclRef<GenericTypeConstraintDecl> constraintDeclRef =
                astBuilder->getDirectDeclRef(constraintDecl.getDecl());
            int argIndex = -1;
            int ii = 0;

            // Find the generic parameter type (T) that this constraint (T:IFoo) is applying to.
            auto genericParamType = getSub(astBuilder, constraintDeclRef);
            auto genParamDeclRefType = as<DeclRefType>(genericParamType);
            if (!genParamDeclRefType)
            {
                continue;
            }
            auto genParamDeclRef = genParamDeclRefType->getDeclRef();

            // Find the generic argument index of the corresponding generic parameter type in the
            // generic parameter set.
            //
            for (auto member : genericDeclRef.getDecl()->getMembersOfType<GenericTypeParamDecl>())
            {
                if (member == genParamDeclRef.getDecl())
                {
                    argIndex = ii;
                    break;
                }
                ii++;
            }
            if (argIndex == -1)
            {
                SLANG_ASSERT(!"generic parameter not found in generic decl");
                continue;
            }
            auto sub = as<Type>(args[argIndex].val);
            if (!sub)
            {
                sink->diagnose(
                    constraintDecl,
                    Diagnostics::expectedTypeForSpecializationArg,
                    argIndex);
                continue;
            }

            auto sup = getSup(astBuilder, constraintDeclRef);
            auto subTypeWitness = visitor.isSubtype(sub, sup, IsSubTypeOptions::None);
            if (subTypeWitness)
            {
                genericArgs.add(subTypeWitness);
            }
            else
            {
                // TODO: diagnose a problem here
                sink->diagnose(
                    constraintDecl,
                    Diagnostics::typeArgumentDoesNotConformToInterface,
                    sub,
                    sup);
                continue;
            }
        }

        specializedFuncDeclRef =
            getLinkage()
                ->getASTBuilder()
                ->getGenericAppDeclRef(genericDeclRef, genericArgs.getArrayView())
                .as<FuncDecl>();
        SLANG_ASSERT(specializedFuncDeclRef);
    }

    info->specializedFuncDeclRef = specializedFuncDeclRef;

    // Once the generic parameters (if any) have been dealt with,
    // any remaining specialization arguments are for existential/interface
    // specialization parameters, attached to the value parameters
    // of the entry point.
    //
    args += genericSpecializationParamCount;
    argCount -= genericSpecializationParamCount;

    auto existentialSpecializationParamCount = getExistentialSpecializationParamCount();
    SLANG_ASSERT(argCount == existentialSpecializationParamCount);

    for (Index ii = 0; ii < existentialSpecializationParamCount; ++ii)
    {
        auto& param = m_existentialSpecializationParams[ii];
        auto& specializationArg = args[ii];

        // TODO: We need to handle all the cases of "flavor" for the `param`s (not just types)

        auto paramType = as<Type>(param.object);
        auto argType = as<Type>(specializationArg.val);

        auto witness = visitor.isSubtype(argType, paramType, IsSubTypeOptions::None);
        if (!witness)
        {
            // If no witness was found, then we will be unable to satisfy
            // the conformances required.
            sink->diagnose(
                SourceLoc(),
                Diagnostics::typeArgumentDoesNotConformToInterface,
                argType,
                paramType);
            continue;
        }

        ExpandedSpecializationArg expandedArg;
        expandedArg.val = specializationArg.val;
        expandedArg.witness = witness;
        info->existentialSpecializationArgs.add(expandedArg);
    }

    return info;
}

/// Create a specialization an existing entry point based on specialization argument expressions.
RefPtr<ComponentType> createSpecializedEntryPoint(
    EntryPoint* unspecializedEntryPoint,
    List<Expr*> const& argExprs,
    DiagnosticSink* sink)
{
    // We need to convert all of the `Expr` arguments
    // into `SpecializationArg`s, so that we can bottleneck
    // through the shared logic.
    //
    List<SpecializationArg> args;
    _extractSpecializationArgs(unspecializedEntryPoint, argExprs, args, sink);
    if (sink->getErrorCount())
        return nullptr;

    return ((ComponentType*)unspecializedEntryPoint)
        ->specialize(args.getBuffer(), args.getCount(), sink);
}

Scope* ComponentType::_getOrCreateScopeForLegacyLookup(ASTBuilder* astBuilder)
{
    // The shape of this logic is dictated by the legacy
    // behavior for name-based lookup/parsing of types
    // specified via the API or command line.
    //
    // We begin with a dummy scope that has as its parent
    // the scope that provides the "base" langauge
    // definitions (that scope is necessary because
    // it defines keywords like `true` and `false`).
    //
    if (m_lookupScope)
        return m_lookupScope;

    Scope* scope = astBuilder->create<Scope>();
    scope->parent = getLinkage()->getSessionImpl()->slangLanguageScope;
    //
    // Next, the scope needs to include all of the
    // modules in the program as peers, as if they
    // were `import`ed into the scope.
    //
    for (auto module : getModuleDependencies())
    {
        for (auto srcScope = module->getModuleDecl()->ownedScope; srcScope;
             srcScope = srcScope->nextSibling)
        {
            if (srcScope->containerDecl != module->getModuleDecl() &&
                srcScope->containerDecl->parentDecl != module->getModuleDecl())
                continue; // Skip scopes that is not part of current module.

            Scope* moduleScope = astBuilder->create<Scope>();
            moduleScope->containerDecl = srcScope->containerDecl;

            moduleScope->nextSibling = scope->nextSibling;
            scope->nextSibling = moduleScope;
        }
    }
    m_lookupScope = scope;
    return scope;
}

/// Parse an array of strings as specialization arguments.
///
/// Names in the strings will be parsed in the context of
/// the code loaded into the given compile request.
///
void parseSpecializationArgStrings(
    EndToEndCompileRequest* endToEndReq,
    List<String> const& genericArgStrings,
    List<Expr*>& outGenericArgs)
{
    auto unspecialiedProgram = endToEndReq->getUnspecializedGlobalComponentType();

    // TODO(JS):
    //
    // We create the scopes on the linkages ASTBuilder. We might want to create a temporary
    // ASTBuilder, and let that memory get freed, but is like this because it's not clear if the
    // scopes in ASTNode members will dangle if we do.
    Scope* scope = unspecialiedProgram->_getOrCreateScopeForLegacyLookup(
        endToEndReq->getLinkage()->getASTBuilder());

    // We are going to do some semantic checking, so we need to
    // set up a `SemanticsVistitor` that we can use.
    //
    auto linkage = endToEndReq->getLinkage();
    auto sink = endToEndReq->getSink();

    SharedSemanticsContext sharedSemanticsContext(linkage, nullptr, sink);
    SemanticsVisitor semantics(&sharedSemanticsContext);

    // We will be looping over the generic argument strings
    // that the user provided via the API (or command line),
    // and parsing+checking each into an `Expr`.
    //
    // This loop will *not* handle coercing the arguments
    // to be types.
    //
    for (auto name : genericArgStrings)
    {
        Expr* argExpr = linkage->parseTermString(name, scope);
        argExpr = semantics.CheckTerm(argExpr);

        if (!argExpr)
        {
            sink->diagnose(
                SourceLoc(),
                Diagnostics::internalCompilerError,
                "couldn't parse specialization argument");
            return;
        }

        outGenericArgs.add(argExpr);
    }
}

Type* Linkage::specializeType(
    Type* unspecializedType,
    Int argCount,
    Type* const* args,
    DiagnosticSink* sink)
{
    SLANG_ASSERT(unspecializedType);

    // TODO: We should cache and re-use specialized types
    // when the exact same arguments are provided again later.

    SharedSemanticsContext sharedSemanticsContext(this, nullptr, sink);
    SemanticsVisitor visitor(&sharedSemanticsContext);

    SpecializationParams specializationParams;
    _collectExistentialSpecializationParamsRec(
        getASTBuilder(),
        specializationParams,
        unspecializedType,
        SourceLoc());

    assert(specializationParams.getCount() == argCount);

    ExpandedSpecializationArgs specializationArgs;
    for (Int aa = 0; aa < argCount; ++aa)
    {
        auto paramType = as<Type>(specializationParams[aa].object);
        auto argType = args[aa];

        ExpandedSpecializationArg arg;
        arg.val = argType;
        arg.witness = visitor.isSubtype(argType, paramType, IsSubTypeOptions::None);
        specializationArgs.add(arg);
    }

    ExistentialSpecializedType* specializedType =
        m_astBuilder->getOrCreate<ExistentialSpecializedType>(
            unspecializedType,
            specializationArgs);

    m_specializedTypes.add(specializedType);

    return specializedType;
}

/// Shared implementation logic for the `_createSpecializedProgram*` entry points.
static RefPtr<ComponentType> _createSpecializedProgramImpl(
    Linkage* linkage,
    ComponentType* unspecializedProgram,
    List<Expr*> const& specializationArgExprs,
    DiagnosticSink* sink)
{
    // If there are no specialization arguments,
    // then the the result of specialization should
    // be the same as the input.
    //
    auto specializationArgCount = specializationArgExprs.getCount();
    if (specializationArgCount == 0)
    {
        return unspecializedProgram;
    }

    auto specializationParamCount = unspecializedProgram->getSpecializationParamCount();
    if (specializationArgCount != specializationParamCount)
    {
        sink->diagnose(
            SourceLoc(),
            Diagnostics::mismatchSpecializationArguments,
            specializationParamCount,
            specializationArgCount);
        return nullptr;
    }

    // We have an appropriate number of arguments for the global specialization parameters,
    // and now we need to check that the arguments conform to the declared constraints.
    //
    SharedSemanticsContext visitor(linkage, nullptr, sink);

    List<SpecializationArg> specializationArgs;
    _extractSpecializationArgs(
        unspecializedProgram,
        specializationArgExprs,
        specializationArgs,
        sink);
    if (sink->getErrorCount())
        return nullptr;

    auto specializedProgram = unspecializedProgram->specialize(
        specializationArgs.getBuffer(),
        specializationArgs.getCount(),
        sink);

    return specializedProgram;
}

/// Specialize an entry point that was checked by the front-end, based on specialization arguments.
///
/// If the end-to-end compile request included specialization argument strings
/// for this entry point, then they will be parsed, checked, and used
/// as arguments to the generic entry point.
///
/// Returns a specialized entry point if everything worked as expected.
/// Returns null and diagnoses errors if anything goes wrong.
///
RefPtr<ComponentType> createSpecializedEntryPoint(
    EndToEndCompileRequest* endToEndReq,
    EntryPoint* unspecializedEntryPoint,
    EndToEndCompileRequest::EntryPointInfo const& entryPointInfo)
{
    auto sink = endToEndReq->getSink();

    // If the user specified generic arguments for the entry point,
    // then we will need to parse the arguments first.
    //
    List<Expr*> specializationArgExprs;
    parseSpecializationArgStrings(
        endToEndReq,
        entryPointInfo.specializationArgStrings,
        specializationArgExprs);

    // Next we specialize the entry point function given the parsed
    // generic argument expressions.
    //
    auto entryPoint =
        createSpecializedEntryPoint(unspecializedEntryPoint, specializationArgExprs, sink);

    return entryPoint;
}

/// Create a specialized component type for the global scope of the given compile request.
///
/// The specialized program will be consistent with that created by
/// `createUnspecializedGlobalComponentType`, and will simply fill in
/// its specialization parameters with the arguments (if any) supllied
/// as part fo the end-to-end compile request.
///
/// The layout of the new component type will be consistent with that
/// of the original *if* there are no global generic type parameters
/// (only interface/existential parameters).
///
RefPtr<ComponentType> createSpecializedGlobalComponentType(EndToEndCompileRequest* endToEndReq)
{
    // The compile request must have already completed front-end processing,
    // so that we have an unspecialized program available, and now only need
    // to parse and check any generic arguments that are being supplied for
    // global or entry-point generic parameters.
    //
    auto unspecializedProgram = endToEndReq->getUnspecializedGlobalComponentType();
    auto linkage = endToEndReq->getLinkage();
    auto sink = endToEndReq->getSink();

    // First, let's parse the specialization argument strings that were
    // provided via the API, so that we can match them
    // against what was declared in the program.
    //
    List<Expr*> globalSpecializationArgs;
    parseSpecializationArgStrings(
        endToEndReq,
        endToEndReq->m_globalSpecializationArgStrings,
        globalSpecializationArgs);

    // Don't proceed further if anything failed to parse.
    if (sink->getErrorCount())
        return nullptr;

    // Now we create the initial specialized program by
    // applying the global generic arguments (if any) to the
    // unspecialized program.
    //
    auto specializedProgram = _createSpecializedProgramImpl(
        linkage,
        unspecializedProgram,
        globalSpecializationArgs,
        sink);

    // If anything went wrong with the global generic
    // arguments, then bail out now.
    //
    if (!specializedProgram)
        return nullptr;

    // Next we will deal with the entry points for the
    // new specialized program.
    //
    // If the user specified explicit entry points as part of the
    // end-to-end request, then we only want to process those (and
    // ignore any other `[shader(...)]`-attributed entry points).
    //
    // However, if the user specified *no* entry points as part
    // of the end-to-end request, then we would like to go
    // ahead and consider all the entry points that were found
    // by the front-end.
    //
    Index entryPointCount = endToEndReq->m_entryPoints.getCount();
    if (entryPointCount == 0)
    {
        entryPointCount = unspecializedProgram->getEntryPointCount();
        endToEndReq->m_entryPoints.setCount(entryPointCount);
    }

    return specializedProgram;
}

/// Create a specialized program based on the given compile request.
///
/// The specialized program created here includes both the global
/// scope for all the translation units involved and all the entry
/// points, and it also includes any specialization arguments
/// that were supplied.
///
/// It is important to note that this function specializes
/// the global scope and the entry points in isolation and then
/// composes them, and that this can lead to different layout
/// from the result of `createUnspecializedGlobalAndEntryPointsComponentType`.
///
/// If we have a module `M` with entry point `E`, and each has one
/// specialization parameter, then `createUnspecialized...` will yield:
///
///     compose(M,E)
///
/// That composed type will have two specialization parameters (the one
/// from `M` plus the one from `E`) and so we might specialize it to get:
///
///     specialize(compose(M,E), X, Y)
///
/// while if we use `createSpecialized...` we will get:
///
///     compose(specialize(M,X), specialize(E,Y))
///
/// While these options are semantically equivalent, they would not lay
/// out the same way in memory.
///
/// There are many reasons why an application might prefer one over the
/// other, and an application that cares should use the more explicit
/// APIs to construct what they want. The behavior of this function
/// is just to provide a reasonable default for use by end-to-end
/// compilation (e.g., from the command line).
///
RefPtr<ComponentType> createSpecializedGlobalAndEntryPointsComponentType(
    EndToEndCompileRequest* endToEndReq,
    List<RefPtr<ComponentType>>& outSpecializedEntryPoints)
{
    auto specializedGlobalComponentType = endToEndReq->getSpecializedGlobalComponentType();

    List<RefPtr<ComponentType>> allComponentTypes;
    allComponentTypes.add(specializedGlobalComponentType);

    auto unspecializedGlobalAndEntryPointsComponentType =
        endToEndReq->getUnspecializedGlobalAndEntryPointsComponentType();

    // It is possible that there were entry points other than those specified
    // vai the original end-to-end compile request. In particular:
    //
    // * It is possible to compile with *no* entry points specified, in which
    //   case the current compiler behavior is to use any entry points marked
    //   via `[shader(...)]` attributes in the AST.
    //
    // * It is possible for entry points to come into play via serialized libraries
    //   loaded with `-r` on the command line (or the equivalent API).
    //
    // We will thus draw a distinction between the "specified" entry points,
    // and the "found" entry points.
    //
    auto specifiedEntryPointCount = endToEndReq->m_entryPoints.getCount();
    auto foundEntryPointCount =
        unspecializedGlobalAndEntryPointsComponentType->getEntryPointCount();

    SLANG_ASSERT(foundEntryPointCount >= specifiedEntryPointCount);

    // For any entry points that were specified, we can use the specialization
    // argument information provided via API or command line.
    //
    for (Index ii = 0; ii < specifiedEntryPointCount; ++ii)
    {
        auto& entryPointInfo = endToEndReq->m_entryPoints[ii];
        auto unspecializedEntryPoint =
            unspecializedGlobalAndEntryPointsComponentType->getEntryPoint(ii);

        auto specializedEntryPoint =
            createSpecializedEntryPoint(endToEndReq, unspecializedEntryPoint, entryPointInfo);
        allComponentTypes.add(specializedEntryPoint);

        outSpecializedEntryPoints.add(specializedEntryPoint);
    }

    // There might have been errors during the specialization above,
    // so we will bail out early if anything went wrong, rather
    // then try to create a composite where some of the constituent
    // component types might be null.
    //
    if (endToEndReq->getSink()->getErrorCount() != 0)
        return nullptr;

    // Any entry points beyond those that were specified up front will be
    // assumed to not need/want specialization.
    //
    for (Index ii = specifiedEntryPointCount; ii < foundEntryPointCount; ++ii)
    {
        auto unspecializedEntryPoint =
            unspecializedGlobalAndEntryPointsComponentType->getEntryPoint(ii);
        allComponentTypes.add(unspecializedEntryPoint);
        outSpecializedEntryPoints.add(unspecializedEntryPoint);
    }

    RefPtr<ComponentType> composed =
        CompositeComponentType::create(endToEndReq->getLinkage(), allComponentTypes);
    return composed;
}


} // namespace Slang
