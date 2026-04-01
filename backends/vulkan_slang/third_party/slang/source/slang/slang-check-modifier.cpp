// slang-check-modifier.cpp
#include "../core/slang-char-util.h"
#include "slang-check-impl.h"

// This file implements semantic checking behavior for
// modifiers.
//
// At present, the semantic checking we do on modifiers is primarily
// focused on `[attributes]`.

#include "slang-lookup.h"

namespace Slang
{
IntVal* SemanticsVisitor::checkLinkTimeConstantIntVal(Expr* expr)
{
    expr = CheckExpr(expr);
    return CheckIntegerConstantExpression(
        expr,
        IntegerConstantExpressionCoercionType::AnyInteger,
        nullptr,
        ConstantFoldingKind::LinkTime);
}

ConstantIntVal* SemanticsVisitor::checkConstantIntVal(Expr* expr)
{
    // First type-check the expression as normal
    expr = CheckExpr(expr);

    auto intVal = CheckIntegerConstantExpression(
        expr,
        IntegerConstantExpressionCoercionType::AnyInteger,
        nullptr,
        ConstantFoldingKind::CompileTime);

    if (!intVal)
        return nullptr;

    auto constIntVal = as<ConstantIntVal>(intVal);
    if (!constIntVal)
    {
        getSink()->diagnose(expr->loc, Diagnostics::expectedIntegerConstantNotLiteral);
        return nullptr;
    }
    return constIntVal;
}

ConstantIntVal* SemanticsVisitor::checkConstantEnumVal(Expr* expr)
{
    // First type-check the expression as normal
    expr = CheckExpr(expr);

    auto intVal = CheckEnumConstantExpression(expr, ConstantFoldingKind::CompileTime);
    if (!intVal)
        return nullptr;

    auto constIntVal = as<ConstantIntVal>(intVal);
    if (!constIntVal)
    {
        getSink()->diagnose(expr->loc, Diagnostics::expectedIntegerConstantNotLiteral);
        return nullptr;
    }
    return constIntVal;
}

// Check an expression, coerce it to the `String` type, and then
// ensure that it has a literal (not just compile-time constant) value.
bool SemanticsVisitor::checkLiteralStringVal(Expr* expr, String* outVal)
{
    // TODO: This should actually perform semantic checking, etc.,
    // but for now we are just going to look for a direct string
    // literal AST node.

    if (auto stringLitExpr = as<StringLiteralExpr>(expr))
    {
        if (outVal)
        {
            *outVal = stringLitExpr->value;
        }
        return true;
    }

    getSink()->diagnose(expr, Diagnostics::expectedAStringLiteral);

    return false;
}

bool SemanticsVisitor::checkCapabilityName(Expr* expr, CapabilityName& outCapabilityName)
{
    if (auto varExpr = as<VarExpr>(expr))
    {
        if (!varExpr->name)
            return false;
        if (varExpr->name == getSession()->getCompletionRequestTokenName())
        {
            auto& suggestions = getLinkage()->contentAssistInfo.completionSuggestions;
            suggestions.clear();
            suggestions.scopeKind = CompletionSuggestions::ScopeKind::Capabilities;
        }
        outCapabilityName = findCapabilityName(varExpr->name->text.getUnownedSlice());
        if (outCapabilityName == CapabilityName::Invalid)
        {
            getSink()->diagnose(expr, Diagnostics::unknownCapability, varExpr->name);
            return false;
        }
        return true;
    }
    getSink()->diagnose(expr, Diagnostics::expectCapability);
    return false;
}

void SemanticsVisitor::visitModifier(Modifier*)
{
    // Do nothing with modifiers for now
}

DeclRef<VarDeclBase> SemanticsVisitor::tryGetIntSpecializationConstant(Expr* expr)
{
    // First type-check the expression as normal
    expr = CheckExpr(expr);

    if (IsErrorExpr(expr))
        return DeclRef<VarDeclBase>();

    if (!isScalarIntegerType(expr->type))
        return DeclRef<VarDeclBase>();

    auto specConstVar = as<VarExpr>(expr);
    if (!specConstVar || !specConstVar->declRef)
        return DeclRef<VarDeclBase>();

    auto decl = specConstVar->declRef.getDecl();
    if (!decl)
        return DeclRef<VarDeclBase>();

    for (auto modifier : decl->modifiers)
    {
        if (as<SpecializationConstantAttribute>(modifier) || as<VkConstantIdAttribute>(modifier))
        {
            return specConstVar->declRef.as<VarDeclBase>();
        }
    }

    return DeclRef<VarDeclBase>();
}

static bool _isDeclAllowedAsAttribute(DeclRef<Decl> declRef)
{
    if (as<AttributeDecl>(declRef.getDecl()))
        return true;
    auto structDecl = as<StructDecl>(declRef.getDecl());
    if (!structDecl)
        return false;
    auto attrUsageAttr = structDecl->findModifier<AttributeUsageAttribute>();
    if (!attrUsageAttr)
        return false;
    return true;
}

AttributeDecl* SemanticsVisitor::lookUpAttributeDecl(Name* attributeName, Scope* scope)
{
    if (!attributeName)
        return nullptr;
    // We start by looking for an existing attribute matching
    // the name `attributeName`.
    //
    {
        // Look up the name and see what attributes we find.
        //
        LookupMask lookupMask = LookupMask::Attribute;
        if (attributeName == getSession()->getCompletionRequestTokenName())
        {
            lookupMask = LookupMask((uint32_t)LookupMask::Attribute | (uint32_t)LookupMask::type);
        }

        auto lookupResult = lookUp(m_astBuilder, this, attributeName, scope, lookupMask);

        if (attributeName == getSession()->getCompletionRequestTokenName())
        {
            // If this is a completion request, add the lookup result to linkage.
            auto& suggestions = getLinkage()->contentAssistInfo.completionSuggestions;
            suggestions.clear();
            suggestions.scopeKind = CompletionSuggestions::ScopeKind::Attribute;
            for (auto& item : lookupResult)
            {
                if (_isDeclAllowedAsAttribute(item.declRef))
                {
                    suggestions.candidateItems.add(item);
                }
            }
        }

        // If the result was overloaded, then that means there
        // are multiple attributes matching the name, and we
        // aren't going to be able to narrow it down.
        //
        if (lookupResult.isOverloaded())
            return nullptr;

        // If there is a single valid result, and it names
        // an existing attribute declaration, then we can
        // use it as the result.
        //
        if (lookupResult.isValid())
        {
            auto decl = lookupResult.item.declRef.getDecl();
            if (auto attributeDecl = as<AttributeDecl>(decl))
            {
                return attributeDecl;
            }
        }
    }

    // If there wasn't already an attribute matching the
    // given name, then we will look for a `struct` type
    // matching the name scheme for user-defined attributes.
    //
    // If the attribute was `[Something(...)]` then we will
    // look for a `struct` named `SomethingAttribute`.
    //
    LookupResult lookupResult = lookUp(
        m_astBuilder,
        this,
        m_astBuilder->getGlobalSession()->getNameObj(attributeName->text + "Attribute"),
        scope,
        LookupMask::type);
    //
    // If we didn't find a matching type name, then we give up.
    //
    if (!lookupResult.isValid() || lookupResult.isOverloaded())
        return nullptr;


    // We only allow a `struct` type to be used as an attribute
    // if the type itself has an `[AttributeUsage(...)]` attribute
    // attached to it.
    //
    auto structDecl = lookupResult.item.declRef.as<StructDecl>().getDecl();
    if (!structDecl)
        return nullptr;
    auto attrUsageAttr = structDecl->findModifier<AttributeUsageAttribute>();
    if (!attrUsageAttr)
        return nullptr;

    // We will now synthesize a new `AttributeDecl` to mirror
    // what was declared on the `struct` type.
    //
    AttributeDecl* attrDecl = m_astBuilder->create<AttributeDecl>();
    attrDecl->nameAndLoc.name = attributeName;
    attrDecl->nameAndLoc.loc = structDecl->nameAndLoc.loc;
    attrDecl->loc = structDecl->loc;

    while (attrUsageAttr)
    {
        AttributeTargetModifier* targetModifier = m_astBuilder->create<AttributeTargetModifier>();
        targetModifier->syntaxClass = attrUsageAttr->targetSyntaxClass;
        targetModifier->loc = attrUsageAttr->loc;
        addModifier(attrDecl, targetModifier);
        attrUsageAttr = as<AttributeUsageAttribute>(attrUsageAttr->next);
    }

    // Every attribute declaration is associated with the type
    // of syntax nodes it constructs (via reflection/RTTI).
    //
    // User-defined attributes create instances of
    // `UserDefinedAttribute`.
    //
    attrDecl->syntaxClass =
        m_astBuilder->findSyntaxClass(UnownedStringSlice::fromLiteral("UserDefinedAttribute"));

    // The fields of the user-defined `struct` type become
    // the parameters of the new attribute.
    //
    // TODO: This step should skip `static` fields.
    //
    for (auto member : structDecl->members)
    {
        if (auto varMember = as<VarDecl>(member))
        {
            ensureDecl(varMember, DeclCheckState::CanUseTypeOfValueDecl);

            ParamDecl* paramDecl = m_astBuilder->create<ParamDecl>();
            paramDecl->nameAndLoc = member->nameAndLoc;
            paramDecl->type = varMember->type;
            paramDecl->loc = member->loc;
            paramDecl->setCheckState(DeclCheckState::DefinitionChecked);

            attrDecl->addMember(paramDecl);
        }
    }

    // We need to end by putting the new attribute declaration
    // into the AST, so that it can be found via lookup.
    //
    auto parentDecl = structDecl->parentDecl;
    //
    // TODO: handle the case where `parentDecl` is generic?
    //
    parentDecl->addMember(attrDecl);

    SLANG_ASSERT(!parentDecl->isMemberDictionaryValid());

    // Finally, we perform any required semantic checks on
    // the newly constructed attribute decl.
    //
    // TODO: what check state is relevant here?
    //
    ensureDecl(attrDecl, DeclCheckState::DefinitionChecked);

    return attrDecl;
}

bool SemanticsVisitor::hasFloatArgs(Attribute* attr, int numArgs)
{
    if (int(attr->args.getCount()) != numArgs)
    {
        return false;
    }
    for (int i = 0; i < numArgs; ++i)
    {
        if (!as<FloatingPointLiteralExpr>(attr->args[i]) && !as<IntegerLiteralExpr>(attr->args[i]))
        {
            return false;
        }
    }
    return true;
}

bool SemanticsVisitor::hasIntArgs(Attribute* attr, int numArgs)
{
    if (int(attr->args.getCount()) != numArgs)
    {
        return false;
    }
    for (int i = 0; i < numArgs; ++i)
    {
        if (!as<IntegerLiteralExpr>(attr->args[i]))
        {
            return false;
        }
    }
    return true;
}

bool SemanticsVisitor::hasStringArgs(Attribute* attr, int numArgs)
{
    if (int(attr->args.getCount()) != numArgs)
    {
        return false;
    }
    for (int i = 0; i < numArgs; ++i)
    {
        if (!as<StringLiteralExpr>(attr->args[i]))
        {
            return false;
        }
    }
    return true;
}

bool SemanticsVisitor::getAttributeTargetSyntaxClasses(
    SyntaxClass<NodeBase>& cls,
    uint32_t typeFlags)
{
    if (typeFlags == (int)UserDefinedAttributeTargets::Struct)
    {
        cls = m_astBuilder->findSyntaxClass(UnownedStringSlice::fromLiteral("StructDecl"));
        return true;
    }
    if (typeFlags == (int)UserDefinedAttributeTargets::Var)
    {
        cls = m_astBuilder->findSyntaxClass(UnownedStringSlice::fromLiteral("VarDecl"));
        return true;
    }
    if (typeFlags == (int)UserDefinedAttributeTargets::Function)
    {
        cls = m_astBuilder->findSyntaxClass(UnownedStringSlice::fromLiteral("FuncDecl"));
        return true;
    }
    if (typeFlags == (int)UserDefinedAttributeTargets::Param)
    {
        cls = m_astBuilder->findSyntaxClass(UnownedStringSlice::fromLiteral("ParamDecl"));
        return true;
    }
    return false;
}

Modifier* SemanticsVisitor::validateAttribute(
    Attribute* attr,
    AttributeDecl* attribClassDecl,
    ModifiableSyntaxNode* attrTarget)
{
    if (auto numThreadsAttr = as<NumThreadsAttribute>(attr))
    {
        SLANG_ASSERT(attr->args.getCount() == 3);

        for (int i = 0; i < 3; ++i)
        {
            IntVal* value = nullptr;

            auto arg = attr->args[i];
            if (arg)
            {
                auto specConstDecl = tryGetIntSpecializationConstant(arg);
                if (specConstDecl)
                {
                    numThreadsAttr->extents[i] = nullptr;
                    numThreadsAttr->specConstExtents[i] = specConstDecl;
                    continue;
                }

                auto intValue = checkLinkTimeConstantIntVal(arg);
                if (!intValue)
                {
                    return nullptr;
                }
                if (auto constIntVal = as<ConstantIntVal>(intValue))
                {
                    if (constIntVal->getValue() < 1)
                    {
                        getSink()->diagnose(
                            attr,
                            Diagnostics::nonPositiveNumThreads,
                            constIntVal->getValue());
                        return nullptr;
                    }
                    if (intValue->getType() != m_astBuilder->getIntType())
                    {
                        intValue = m_astBuilder->getIntVal(
                            m_astBuilder->getIntType(),
                            constIntVal->getValue());
                    }
                }
                // Make sure we always canonicalize the type to int.
                value = intValue;
                if (value->getType() != m_astBuilder->getIntType())
                    value = m_astBuilder->getTypeCastIntVal(m_astBuilder->getIntType(), value);
            }
            else
            {
                value = m_astBuilder->getIntVal(m_astBuilder->getIntType(), 1);
            }
            numThreadsAttr->extents[i] = value;
        }
    }
    else if (auto waveSizeAttr = as<WaveSizeAttribute>(attr))
    {
        SLANG_ASSERT(attr->args.getCount() == 1);

        IntVal* value = nullptr;

        auto arg = attr->args[0];
        if (arg)
        {
            auto intValue = checkLinkTimeConstantIntVal(arg);
            if (!intValue)
            {
                return nullptr;
            }
            if (auto constIntVal = as<ConstantIntVal>(intValue))
            {
                bool isValidWaveSize = false;
                const IntegerLiteralValue waveSize = constIntVal->getValue();
                for (int validWaveSize : {4, 8, 16, 32, 64, 128})
                {
                    if (validWaveSize == waveSize)
                    {
                        isValidWaveSize = true;
                        break;
                    }
                }
                if (!isValidWaveSize)
                {
                    getSink()->diagnose(
                        attr,
                        Diagnostics::invalidWaveSize,
                        constIntVal->getValue());
                    return nullptr;
                }
            }
            value = intValue;
        }
        else
        {
            value = m_astBuilder->getIntVal(m_astBuilder->getIntType(), 1);
        }

        waveSizeAttr->numLanes = value;
    }
    else if (auto anyValueSizeAttr = as<AnyValueSizeAttribute>(attr))
    {
        // This case handles GLSL-oriented layout attributes
        // that take a single integer argument.

        if (attr->args.getCount() != 1)
        {
            return nullptr;
        }

        auto value = checkConstantIntVal(attr->args[0]);
        if (value == nullptr)
        {
            return nullptr;
        }

        const IRIntegerValue kMaxAnyValueSize = 0x7FFF;
        if (value->getValue() > kMaxAnyValueSize)
        {
            getSink()->diagnose(
                anyValueSizeAttr->loc,
                Diagnostics::anyValueSizeExceedsLimit,
                kMaxAnyValueSize);
            return nullptr;
        }

        anyValueSizeAttr->size = int32_t(value->getValue());
    }
    else if (
        auto glslRequireShaderInputParameter = as<GLSLRequireShaderInputParameterAttribute>(attr))
    {
        if (attr->args.getCount() != 1)
        {
            return nullptr;
        }
        auto value = checkConstantIntVal(attr->args[0]);
        if (value == nullptr)
        {
            return nullptr;
        }
        if (value->getValue() < 0)
        {
            return nullptr;
        }
        glslRequireShaderInputParameter->parameterNumber = int32_t(value->getValue());
    }
    else if (auto overloadRankAttr = as<OverloadRankAttribute>(attr))
    {
        if (attr->args.getCount() != 1)
        {
            return nullptr;
        }
        auto rank = checkConstantIntVal(attr->args[0]);
        if (rank == nullptr)
        {
            return nullptr;
        }
        overloadRankAttr->rank = int32_t(rank->getValue());
    }
    else if (
        auto inputAttachmentIndexLayoutAttribute =
            as<GLSLInputAttachmentIndexLayoutAttribute>(attr))
    {
        if (attr->args.getCount() != 1)
            return nullptr;

        auto location = checkConstantIntVal(attr->args[0]);
        if (!location)
            return nullptr;

        inputAttachmentIndexLayoutAttribute->location = location->getValue();
    }
    else if (auto locationLayoutAttr = as<GLSLLocationAttribute>(attr))
    {
        if (attr->args.getCount() != 1)
            return nullptr;

        auto location = checkConstantIntVal(attr->args[0]);
        if (!location)
            return nullptr;

        locationLayoutAttr->value = int32_t(location->getValue());
    }
    else if (auto bindingAttr = as<GLSLBindingAttribute>(attr))
    {
        // This must be vk::binding or gl::binding (as specified in core.meta.slang under
        // vk_binding/gl_binding) Must have 2 int parameters. Ideally this would all be checked from
        // the specification in core.meta.slang, but that's not completely implemented. So for now
        // we check here.
        if (attr->args.getCount() != 2)
        {
            return nullptr;
        }

        // TODO(JS): Prior validation currently doesn't ensure both args are ints (as specified in
        // core.meta.slang), so check here to make sure they both are.
        //
        // Binding attribute may also be created from GLSL style layout qualifiers where only one of
        // the args are specified, hence check for each individually.
        ConstantIntVal* binding = nullptr;
        if (attr->args[0])
            binding = checkConstantIntVal(attr->args[0]);

        ConstantIntVal* set = nullptr;
        if (attr->args[1])
            set = checkConstantIntVal(attr->args[1]);

        if (!binding && !set)
        {
            return nullptr;
        }

        if (binding)
        {
            bindingAttr->binding = int32_t(binding->getValue());
        }

        if (set)
        {
            bindingAttr->set = int32_t(set->getValue());
        }
    }
    else if (auto simpleLayoutAttr = as<GLSLSimpleIntegerLayoutAttribute>(attr))
    {
        // This case handles GLSL-oriented layout attributes
        // that take a single integer argument.

        if (attr->args.getCount() != 1)
        {
            return nullptr;
        }

        auto value = checkConstantIntVal(attr->args[0]);
        if (value == nullptr)
        {
            return nullptr;
        }

        simpleLayoutAttr->value = int32_t(value->getValue());
    }
    else if (auto maxVertexCountAttr = as<MaxVertexCountAttribute>(attr))
    {
        SLANG_ASSERT(attr->args.getCount() == 1);
        auto val = checkConstantIntVal(attr->args[0]);

        if (!val)
            return nullptr;

        maxVertexCountAttr->value = (int32_t)val->getValue();
    }
    else if (auto instanceAttr = as<InstanceAttribute>(attr))
    {
        SLANG_ASSERT(attr->args.getCount() == 1);
        auto val = checkConstantIntVal(attr->args[0]);

        if (!val)
            return nullptr;

        instanceAttr->value = (int32_t)val->getValue();
    }
    else if (auto entryPointAttr = as<EntryPointAttribute>(attr))
    {
        SLANG_ASSERT(attr->args.getCount() == 1);

        String capNameString;
        if (!checkLiteralStringVal(attr->args[0], &capNameString))
        {
            return nullptr;
        }

        CapabilityName capName = findCapabilityName(capNameString.getUnownedSlice());
        if (capName != CapabilityName::Invalid)
        {
            if (isInternalCapabilityName(capName))
                maybeDiagnose(
                    getSink(),
                    this->getOptionSet(),
                    DiagnosticCategory::Capability,
                    attr,
                    Diagnostics::usingInternalCapabilityName,
                    attr,
                    capName);

            // Ensure this capability only defines 1 stage per target, else diagnose an error.
            // This is a fatal error, do not allow toggling this error off.
            entryPointAttr->capabilitySet = CapabilitySet(capName);
            HashSet<CapabilityAtom> stageToBeUsed;
            for (auto& targetSet : entryPointAttr->capabilitySet.getCapabilityTargetSets())
            {
                for (auto& stageSet : targetSet.second.shaderStageSets)
                    stageToBeUsed.add(stageSet.first);
            }

            // TODO: Once profiles are removed in favor for `CapabilitySet`s we will beable to use
            // more complex relationships, Until then we have an artificial limit that any
            // capabilites used inside '[shader(...)]' must only specify 1 stage type uniformly
            // across targets.
            if (stageToBeUsed.getCount() > 1)
            {
                List<CapabilityAtom> atomsToPrint;
                atomsToPrint.reserve(stageToBeUsed.getCount());
                for (auto i : stageToBeUsed)
                    atomsToPrint.add(i);
                getSink()->diagnose(
                    attr,
                    Diagnostics::capabilityHasMultipleStages,
                    capNameString,
                    atomsToPrint);
            }
            return entryPointAttr;
        }
        else
        {
            // always diagnose this error since nothing can compile with an invalid capability
            getSink()->diagnose(attr, Diagnostics::unknownCapability, capNameString);
            return nullptr;
        }
    }
    else if (
        (as<DomainAttribute>(attr)) || (as<OutputTopologyAttribute>(attr)) ||
        (as<PartitioningAttribute>(attr)) || (as<PatchConstantFuncAttribute>(attr)))
    {
        // Let it go thru iff single string attribute
        if (!hasStringArgs(attr, 1))
        {
            getSink()->diagnose(attr, Diagnostics::expectedSingleStringArg, attr->keywordName);
        }
    }
    else if (auto opAttr = as<SPIRVInstructionOpAttribute>(attr))
    {
        auto sink = getSink();
        const auto argsCount = opAttr->args.getCount();
        if (argsCount < 1 || argsCount > 2)
        {
            sink->diagnose(
                attr,
                Diagnostics::attributeArgumentCountMismatch,
                attr->keywordName,
                "1...2",
                argsCount);
        }
        else if (!as<IntegerLiteralExpr>(opAttr->args[0]))
        {
            sink->diagnose(attr, Diagnostics::attributeExpectedIntArg, attr->keywordName, 0);
        }
        else if (argsCount > 1 && !as<StringLiteralExpr>(opAttr->args[1]))
        {
            sink->diagnose(attr, Diagnostics::attributeExpectedStringArg, attr->keywordName, 1);
        }
    }
    else if (as<MaxTessFactorAttribute>(attr))
    {
        if (!hasFloatArgs(attr, 1))
        {
            getSink()->diagnose(attr, Diagnostics::expectedSingleFloatArg, attr->keywordName);
        }
    }
    else if (as<OutputControlPointsAttribute>(attr))
    {
        // Let it go thru iff single integral attribute
        if (!hasIntArgs(attr, 1))
        {
            getSink()->diagnose(attr, Diagnostics::expectedSingleIntArg, attr->keywordName);
        }
    }
    else if (auto attrUsageAttr = as<AttributeUsageAttribute>(attr))
    {
        uint32_t targetClassId = (uint32_t)UserDefinedAttributeTargets::None;
        if (attr->args.getCount() == 1)
        {
            // IntVal* outIntVal;
            if (auto cInt = checkConstantEnumVal(attr->args[0]))
            {
                targetClassId = (uint32_t)(cInt->getValue());
            }
            else
            {
                getSink()->diagnose(attr, Diagnostics::expectedSingleIntArg, attr->keywordName);
                return nullptr;
            }
        }
        if (!getAttributeTargetSyntaxClasses(attrUsageAttr->targetSyntaxClass, targetClassId))
        {
            getSink()->diagnose(attr, Diagnostics::invalidAttributeTarget);
            return nullptr;
        }
    }
    else if (const auto unrollAttr = as<UnrollAttribute>(attr))
    {
        // Check has an argument. We need this because default behavior is to give an error
        // if an attribute has arguments, but not handled explicitly (and the default param will
        // come through as 1 arg if nothing is specified)
        SLANG_ASSERT(attr->args.getCount() == 1);
    }
    else if (auto forceUnrollAttr = as<ForceUnrollAttribute>(attr))
    {
        if (forceUnrollAttr->args.getCount() < 1)
        {
            getSink()->diagnose(attr, Diagnostics::notEnoughArguments, attr->args.getCount(), 1);
        }
        auto cint = checkConstantIntVal(attr->args[0]);
        if (cint)
            forceUnrollAttr->maxIterations = (int32_t)cint->getValue();
    }
    else if (auto maxItersAttrs = as<MaxItersAttribute>(attr))
    {
        if (attr->args.getCount() < 1)
        {
            getSink()->diagnose(attr, Diagnostics::notEnoughArguments, attr->args.getCount(), 1);
        }
        else
        {
            maxItersAttrs->value = checkLinkTimeConstantIntVal(attr->args[0]);
        }
    }
    else if (const auto userDefAttr = as<UserDefinedAttribute>(attr))
    {
        // check arguments against attribute parameters defined in attribClassDecl
        Index paramIndex = 0;
        auto params = attribClassDecl->getMembersOfType<ParamDecl>();
        for (auto paramDecl : params)
        {
            ensureDecl(paramDecl, DeclCheckState::CanUseTypeOfValueDecl);

            if (paramIndex < attr->args.getCount())
            {
                auto& arg = attr->args[paramIndex];
                bool typeChecked = false;
                if (isValidCompileTimeConstantType(paramDecl->getType()))
                {
                    if (auto cint = checkConstantIntVal(arg))
                    {
                        for (Index ci = attr->intArgVals.getCount(); ci < paramIndex + 1; ci++)
                            attr->intArgVals.add(nullptr);
                        attr->intArgVals[(uint32_t)paramIndex] = cint;
                    }
                    typeChecked = true;
                }
                if (!typeChecked)
                {
                    arg = CheckTerm(arg);
                    arg = coerce(CoercionSite::Argument, paramDecl->getType(), arg);
                }
            }
            paramIndex++;
        }
        if (params.getCount() < attr->args.getCount())
        {
            getSink()->diagnose(
                attr,
                Diagnostics::tooManyArguments,
                attr->args.getCount(),
                params.getCount());
        }
        else if (params.getCount() > attr->args.getCount())
        {
            getSink()->diagnose(
                attr,
                Diagnostics::notEnoughArguments,
                attr->args.getCount(),
                params.getCount());
        }
    }
    else if (auto diffAttr = as<BackwardDifferentiableAttribute>(attr))
    {
        SLANG_ASSERT(attr->args.getCount() == 1);
        auto cint = checkConstantIntVal(attr->args[0]);
        if (cint)
            diffAttr->maxOrder = (int32_t)cint->getValue();
    }
    else if (auto formatAttr = as<FormatAttribute>(attr))
    {
        SLANG_ASSERT(attr->args.getCount() == 1);

        String formatName;
        if (!checkLiteralStringVal(attr->args[0], &formatName))
        {
            return nullptr;
        }

        ImageFormat format = ImageFormat::unknown;

        if (attr->keywordName->text.getUnownedSlice() == toSlice("image"))
        {
            if (!findImageFormatByName(formatName.getUnownedSlice(), &format))
            {
                getSink()->diagnose(attr->args[0], Diagnostics::unknownImageFormatName, formatName);
            }
        }
        else
        {
            if (!findVkImageFormatByName(formatName.getUnownedSlice(), &format))
            {
                getSink()->diagnose(attr->args[0], Diagnostics::unknownImageFormatName, formatName);
            }
        }

        formatAttr->format = format;
    }
    else if (auto allowAttr = as<AllowAttribute>(attr))
    {
        SLANG_ASSERT(attr->args.getCount() == 1);

        String diagnosticName;
        if (!checkLiteralStringVal(attr->args[0], &diagnosticName))
        {
            return nullptr;
        }

        auto diagnosticInfo = findDiagnosticByName(diagnosticName.getUnownedSlice());
        if (!diagnosticInfo)
        {
            getSink()->diagnose(attr->args[0], Diagnostics::unknownDiagnosticName, diagnosticName);
        }

        allowAttr->diagnostic = diagnosticInfo;
    }
    else if (auto dllImportAttr = as<DllImportAttribute>(attr))
    {
        SLANG_ASSERT(attr->args.getCount() == 1 || attr->args.getCount() == 2);

        String libraryName;
        if (!checkLiteralStringVal(dllImportAttr->args[0], &libraryName))
        {
            return nullptr;
        }
        dllImportAttr->modulePath = libraryName;

        String functionName;
        if (dllImportAttr->args.getCount() == 2 &&
            !checkLiteralStringVal(dllImportAttr->args[1], &functionName))
        {
            return nullptr;
        }
        dllImportAttr->functionName = functionName;
    }
    else if (auto rayPayloadAttr = as<VulkanRayPayloadAttribute>(attr))
    {
        SLANG_ASSERT(attr->args.getCount() == 1);
        auto val = checkConstantIntVal(attr->args[0]);

        if (!val)
            return nullptr;

        rayPayloadAttr->location = (int32_t)val->getValue();
    }
    else if (auto rayPayloadInAttr = as<VulkanRayPayloadInAttribute>(attr))
    {
        SLANG_ASSERT(attr->args.getCount() == 1);
        auto val = checkConstantIntVal(attr->args[0]);
        if (!val)
            return nullptr;
        rayPayloadInAttr->location = (int32_t)val->getValue();
    }
    else if (auto callablePayloadAttr = as<VulkanCallablePayloadAttribute>(attr))
    {
        SLANG_ASSERT(attr->args.getCount() == 1);
        auto val = checkConstantIntVal(attr->args[0]);

        if (!val)
            return nullptr;

        callablePayloadAttr->location = (int32_t)val->getValue();
    }
    else if (auto callablePayloadInAttr = as<VulkanCallablePayloadInAttribute>(attr))
    {
        SLANG_ASSERT(attr->args.getCount() == 1);
        auto val = checkConstantIntVal(attr->args[0]);
        if (!val)
            return nullptr;
        callablePayloadInAttr->location = (int32_t)val->getValue();
    }
    else if (auto hitObjectAttributesAttr = as<VulkanHitObjectAttributesAttribute>(attr))
    {
        SLANG_ASSERT(attr->args.getCount() == 1);
        auto val = checkConstantIntVal(attr->args[0]);

        if (!val)
            return nullptr;

        hitObjectAttributesAttr->location = (int32_t)val->getValue();
    }
    else if (auto constantIdAttr = as<VkConstantIdAttribute>(attr))
    {
        SLANG_ASSERT(attr->args.getCount() == 1);
        auto val = checkConstantIntVal(attr->args[0]);

        if (!val)
            return nullptr;
        constantIdAttr->location = (int32_t)val->getValue();
    }
    else if (as<UserDefinedDerivativeAttribute>(attr) || as<PrimalSubstituteAttribute>(attr))
    {
        SLANG_ASSERT(attr->args.getCount() == 1);
        SLANG_ASSERT(as<Decl>(attrTarget));
        if (auto derivativeAttr = as<UserDefinedDerivativeAttribute>(attr))
            derivativeAttr->funcExpr = attr->args[0];
        else if (auto primalSubstAttr = as<PrimalSubstituteAttribute>(attr))
            primalSubstAttr->funcExpr = attr->args[0];
    }
    else if (as<DerivativeOfAttribute>(attr) || as<PrimalSubstituteOfAttribute>(attr))
    {
        SLANG_ASSERT(attr->args.getCount() == 1);
        SLANG_ASSERT(as<Decl>(attrTarget));
        if (auto derivativeOfAttr = as<DerivativeOfAttribute>(attr))
            derivativeOfAttr->funcExpr = attr->args[0];
        else if (auto primalOfAttr = as<PrimalSubstituteOfAttribute>(attr))
            primalOfAttr->funcExpr = attr->args[0];
    }
    else if (auto preferRecomputeAttr = as<PreferRecomputeAttribute>(attr))
    {
        SLANG_ASSERT(attr->args.getCount() == 1);
        SLANG_ASSERT(as<Decl>(attrTarget));

        auto val = checkConstantIntVal(attr->args[0]);
        if (!val)
            return nullptr;

        preferRecomputeAttr->sideEffectBehavior =
            (PreferRecomputeAttribute::SideEffectBehavior)val->getValue();
    }
    else if (auto comInterfaceAttr = as<ComInterfaceAttribute>(attr))
    {
        SLANG_ASSERT(attr->args.getCount() == 1);
        String guid;
        if (!checkLiteralStringVal(comInterfaceAttr->args[0], &guid))
        {
            return nullptr;
        }
        StringBuilder resultGUID;
        for (auto ch : guid)
        {
            if (CharUtil::isHexDigit(ch))
            {
                resultGUID.appendChar(ch);
            }
            else if (ch == '-')
            {
                continue;
            }
            else
            {
                getSink()->diagnose(attr, Diagnostics::invalidGUID, guid);
                return nullptr;
            }
        }
        comInterfaceAttr->guid = resultGUID.toString();
        if (comInterfaceAttr->guid.getLength() != 32)
        {
            getSink()->diagnose(attr, Diagnostics::invalidGUID, guid);
            return nullptr;
        }
    }
    else if (const auto derivativeMemberAttr = as<DerivativeMemberAttribute>(attr))
    {
        auto varDecl = as<VarDeclBase>(attrTarget);
        if (!varDecl)
        {
            getSink()->diagnose(attr, Diagnostics::attributeNotApplicable, attr->getKeywordName());
            return nullptr;
        }
    }
    else if (auto deprecatedAttr = as<DeprecatedAttribute>(attr))
    {
        SLANG_ASSERT(attr->args.getCount() == 1);

        String message;
        if (!checkLiteralStringVal(attr->args[0], &message))
        {
            return nullptr;
        }

        deprecatedAttr->message = message;
    }
    else if (auto knownBuiltinAttr = as<KnownBuiltinAttribute>(attr))
    {
        SLANG_ASSERT(attr->args.getCount() == 1);

        String name;
        if (!checkLiteralStringVal(attr->args[0], &name))
        {
            return nullptr;
        }

        knownBuiltinAttr->name = name;
    }
    else if (auto pyExportAttr = as<PyExportAttribute>(attr))
    {
        // Check name string.
        SLANG_ASSERT(attr->args.getCount() == 1);

        String name;
        if (!checkLiteralStringVal(attr->args[0], &name))
        {
            return nullptr;
        }

        pyExportAttr->name = name;
    }
    else if (auto requireCapAttr = as<RequireCapabilityAttribute>(attr))
    {
        List<CapabilityName> capabilityNames;
        for (auto& arg : attr->args)
        {
            CapabilityName capName;
            if (checkCapabilityName(arg, capName))
            {
                capabilityNames.add(capName);
                if (isInternalCapabilityName(capName))
                    maybeDiagnose(
                        getSink(),
                        this->getOptionSet(),
                        DiagnosticCategory::Capability,
                        attr,
                        Diagnostics::usingInternalCapabilityName,
                        attr,
                        capName);
            }
        }
        requireCapAttr->capabilitySet = CapabilitySet(capabilityNames);
        if (requireCapAttr->capabilitySet.isInvalid())
            maybeDiagnose(
                getSink(),
                this->getOptionSet(),
                DiagnosticCategory::Capability,
                attr,
                Diagnostics::unexpectedCapability,
                attr,
                CapabilityName::Invalid);
    }
    else if (auto requirePreludeAttr = as<RequirePreludeAttribute>(attr))
    {
        if (attr->args.getCount() > 2)
        {
            getSink()->diagnose(attr, Diagnostics::tooManyArguments, attr->args.getCount(), 0);
            return nullptr;
        }
        else if (attr->args.getCount() < 2)
        {
            getSink()->diagnose(attr, Diagnostics::notEnoughArguments, attr->args.getCount(), 2);
            return nullptr;
        }
        CapabilityName capName;
        if (!checkCapabilityName(attr->args[0], capName))
        {
            return nullptr;
        }
        requirePreludeAttr->capabilitySet = CapabilitySet(capName);
        if (auto stringLitExpr = as<StringLiteralExpr>(attr->args[1]))
        {
            requirePreludeAttr->prelude = getStringLiteralTokenValue(stringLitExpr->token);
        }
        else
        {
            getSink()->diagnose(attr->args[1], Diagnostics::expectedAStringLiteral);
            return nullptr;
        }
        return attr;
    }
    else
    {
        if (attr->args.getCount() == 0)
        {
            // If the attribute took no arguments, then we will
            // assume it is valid as written.
        }
        else
        {
            // We should be special-casing the checking of any attribute
            // with a non-zero number of arguments.
            getSink()->diagnose(attr, Diagnostics::tooManyArguments, attr->args.getCount(), 0);
            return nullptr;
        }
    }

    return attr;
}

AttributeBase* SemanticsVisitor::checkAttribute(
    UncheckedAttribute* uncheckedAttr,
    ModifiableSyntaxNode* attrTarget)
{
    auto attrName = uncheckedAttr->getKeywordName();
    auto attrDecl = lookUpAttributeDecl(attrName, uncheckedAttr->scope);

    if (!attrDecl)
    {
        getSink()->diagnose(uncheckedAttr, Diagnostics::unknownAttributeName, attrName);
        return uncheckedAttr;
    }

    if (!attrDecl->syntaxClass.isSubClassOf<Attribute>())
    {
        SLANG_DIAGNOSE_UNEXPECTED(
            getSink(),
            attrDecl,
            "attribute declaration does not reference an attribute class");
        return uncheckedAttr;
    }

    // Manage scope
    NodeBase* attrInstance = attrDecl->syntaxClass.createInstance(m_astBuilder);
    auto attr = as<Attribute>(attrInstance);
    if (!attr)
    {
        SLANG_DIAGNOSE_UNEXPECTED(
            getSink(),
            attrDecl,
            "attribute class did not yield an attribute object");
        return uncheckedAttr;
    }

    // We are going to replace the unchecked attribute with the checked one.

    // First copy all of the state over from the original attribute.
    attr->keywordName = uncheckedAttr->keywordName;
    attr->originalIdentifierToken = uncheckedAttr->originalIdentifierToken;
    attr->args = uncheckedAttr->args;
    attr->loc = uncheckedAttr->loc;
    attr->attributeDecl = attrDecl;

    // We will start with checking steps that can be applied independent
    // of the concrete attribute type that was selected. These only need
    // us to look at the attribute declaration itself.
    //
    // Start by doing argument/parameter matching
    UInt argCount = attr->args.getCount();
    UInt paramCounter = 0;
    bool mismatch = false;
    for (auto paramDecl : attrDecl->getMembersOfType<ParamDecl>())
    {
        UInt paramIndex = paramCounter++;
        if (paramIndex < argCount)
        {
            // TODO: support checking the argument against the declared
            // type for the parameter.
        }
        else
        {
            // We didn't have enough arguments for the
            // number of parameters declared.
            if (const auto defaultArg = paramDecl->initExpr)
            {
                // The attribute declaration provided a default,
                // so we should use that.
                //
                // TODO: we need to figure out how to hook up
                // default arguments as needed.
                // For now just copy the expression over.

                attr->args.add(paramDecl->initExpr);
            }
            else
            {
                mismatch = true;
            }
        }
    }
    UInt paramCount = paramCounter;

    if (mismatch)
    {
        getSink()->diagnose(
            attr,
            Diagnostics::attributeArgumentCountMismatch,
            attrName,
            paramCount,
            argCount);
        return uncheckedAttr;
    }

    // The next bit of validation that we can apply semi-generically
    // is to validate that the target for this attribute is a valid
    // one for the chosen attribute.
    //
    // The attribute declaration will have one or more `AttributeTargetModifier`s
    // that each specify a syntax class that the attribute can be applied to.
    //
    bool validTarget = false;
    for (auto attrTargetMod : attrDecl->getModifiersOfType<AttributeTargetModifier>())
    {
        if (attrTarget->getClass().isSubClassOf(attrTargetMod->syntaxClass))
        {
            validTarget = true;
            break;
        }
    }

    // Some attributes impose constraints on where they can be placed that cannot be captured by the
    // only checking the syntax class. Perform more checks here.
    switch (attr->astNodeType)
    {
    // Allowed only on struct fields.
    case ASTNodeType::VkStructOffsetAttribute:
        auto targetDecl = as<Decl>(attrTarget);
        validTarget = validTarget && targetDecl && as<StructDecl>(getParentDecl(targetDecl));
        break;
    };

    if (!validTarget)
    {
        getSink()->diagnose(attr, Diagnostics::attributeNotApplicable, attrName);
        return uncheckedAttr;
    }

    // Now apply type-specific validation to the attribute.
    if (!validateAttribute(attr, attrDecl, attrTarget))
    {
        return uncheckedAttr;
    }


    return attr;
}

ASTNodeType getModifierConflictGroupKind(ASTNodeType modifierType)
{
    switch (modifierType)
    {
        // Allowed only on parameters and global variables.
    case ASTNodeType::InModifier:
        return modifierType;
    case ASTNodeType::OutModifier:
    case ASTNodeType::RefModifier:
    case ASTNodeType::ConstRefModifier:
    case ASTNodeType::InOutModifier:
        return ASTNodeType::OutModifier;

        // Modifiers that are their own exclusive group.
    case ASTNodeType::GLSLInputAttachmentIndexLayoutAttribute:
    case ASTNodeType::GLSLOffsetLayoutAttribute:
    case ASTNodeType::GLSLUnparsedLayoutModifier:
    case ASTNodeType::UncheckedGLSLBindingLayoutAttribute:
    case ASTNodeType::UncheckedGLSLSetLayoutAttribute:
    case ASTNodeType::UncheckedGLSLOffsetLayoutAttribute:
    case ASTNodeType::UncheckedGLSLInputAttachmentIndexLayoutAttribute:
    case ASTNodeType::UncheckedGLSLLocationLayoutAttribute:
    case ASTNodeType::UncheckedGLSLIndexLayoutAttribute:
    case ASTNodeType::UncheckedGLSLConstantIdAttribute:
    case ASTNodeType::UncheckedGLSLRayPayloadAttribute:
    case ASTNodeType::UncheckedGLSLRayPayloadInAttribute:
    case ASTNodeType::UncheckedGLSLHitObjectAttributesAttribute:
    case ASTNodeType::UncheckedGLSLCallablePayloadAttribute:
    case ASTNodeType::UncheckedGLSLCallablePayloadInAttribute:
    case ASTNodeType::GLSLBufferDataLayoutModifier:
    case ASTNodeType::GLSLLayoutModifierGroupMarker:
    case ASTNodeType::GLSLLayoutModifierGroupBegin:
    case ASTNodeType::GLSLLayoutModifierGroupEnd:
    case ASTNodeType::GLSLBufferModifier:
    case ASTNodeType::VkStructOffsetAttribute:
    case ASTNodeType::MemoryQualifierSetModifier:
    case ASTNodeType::GLSLWriteOnlyModifier:
    case ASTNodeType::GLSLReadOnlyModifier:
    case ASTNodeType::GLSLVolatileModifier:
    case ASTNodeType::GLSLRestrictModifier:
    case ASTNodeType::GLSLPatchModifier:
    case ASTNodeType::RayPayloadAccessSemantic:
    case ASTNodeType::RayPayloadReadSemantic:
    case ASTNodeType::RayPayloadWriteSemantic:
    case ASTNodeType::GloballyCoherentModifier:
    case ASTNodeType::PreciseModifier:
    case ASTNodeType::IntrinsicOpModifier:
    case ASTNodeType::InlineModifier:
    case ASTNodeType::HLSLExportModifier:
    case ASTNodeType::ExternCppModifier:
    case ASTNodeType::ExportedModifier:
    case ASTNodeType::ConstModifier:
    case ASTNodeType::ConstExprModifier:
    case ASTNodeType::MatrixLayoutModifier:
    case ASTNodeType::RowMajorLayoutModifier:
    case ASTNodeType::HLSLRowMajorLayoutModifier:
    case ASTNodeType::GLSLColumnMajorLayoutModifier:
    case ASTNodeType::ColumnMajorLayoutModifier:
    case ASTNodeType::HLSLColumnMajorLayoutModifier:
    case ASTNodeType::GLSLRowMajorLayoutModifier:
    case ASTNodeType::HLSLEffectSharedModifier:
    case ASTNodeType::HLSLVolatileModifier:
    case ASTNodeType::GLSLPrecisionModifier:
    case ASTNodeType::HLSLGroupSharedModifier:
        return modifierType;

    case ASTNodeType::HLSLStaticModifier:
    case ASTNodeType::ActualGlobalModifier:
    case ASTNodeType::HLSLUniformModifier:
        return ASTNodeType::HLSLStaticModifier;

    case ASTNodeType::HLSLNoInterpolationModifier:
    case ASTNodeType::HLSLNoPerspectiveModifier:
    case ASTNodeType::HLSLLinearModifier:
    case ASTNodeType::HLSLSampleModifier:
    case ASTNodeType::HLSLCentroidModifier:
    case ASTNodeType::PerVertexModifier:
        return ASTNodeType::InterpolationModeModifier;

    case ASTNodeType::PrefixModifier:
    case ASTNodeType::PostfixModifier:
        return ASTNodeType::PrefixModifier;

    case ASTNodeType::BuiltinModifier:
    case ASTNodeType::PublicModifier:
    case ASTNodeType::PrivateModifier:
    case ASTNodeType::InternalModifier:
        return ASTNodeType::VisibilityModifier;

    default:
        return ASTNodeType::NodeBase;
    }
}

bool isModifierAllowedOnDecl(bool isGLSLInput, ASTNodeType modifierType, Decl* decl)
{
    switch (modifierType)
    {
        // In addition to the above cases, these are also present on empty
        // global declarations, for instance
        // layout(local_size_x=1) in;
    case ASTNodeType::InModifier:
    case ASTNodeType::InOutModifier:
    case ASTNodeType::OutModifier:
    case ASTNodeType::GLSLInputAttachmentIndexLayoutAttribute:
    case ASTNodeType::GLSLOffsetLayoutAttribute:
    case ASTNodeType::GLSLUnparsedLayoutModifier:
    case ASTNodeType::UncheckedGLSLLayoutAttribute:
    case ASTNodeType::GLSLLayoutModifierGroupMarker:
    case ASTNodeType::GLSLLayoutModifierGroupBegin:
    case ASTNodeType::GLSLLayoutModifierGroupEnd:
        // If we are in GLSL mode, also allow these but otherwise fall to
        // the regular check
        if (isGLSLInput && as<EmptyDecl>(decl) && isGlobalDecl(decl))
            return true;
        [[fallthrough]];

    case ASTNodeType::RefModifier:
    case ASTNodeType::ConstRefModifier:
    case ASTNodeType::GLSLBufferModifier:
    case ASTNodeType::GLSLPatchModifier:
        return (as<VarDeclBase>(decl) && isGlobalDecl(decl)) || as<ParamDecl>(decl) ||
               as<GLSLInterfaceBlockDecl>(decl);
    case ASTNodeType::RayPayloadAccessSemantic:
    case ASTNodeType::RayPayloadReadSemantic:
    case ASTNodeType::RayPayloadWriteSemantic:
        // Allow on struct fields if the parent struct has the [raypayload] attribute
        if (auto varDecl = as<VarDeclBase>(decl))
        {
            if (auto structDecl = as<StructDecl>(varDecl->parentDecl))
            {
                if (structDecl->findModifier<RayPayloadAttribute>())
                    return true;
            }
        }
        return (as<VarDeclBase>(decl) && isGlobalDecl(decl)) || as<ParamDecl>(decl) ||
               as<GLSLInterfaceBlockDecl>(decl);

    case ASTNodeType::GLSLWriteOnlyModifier:
    case ASTNodeType::GLSLReadOnlyModifier:
    case ASTNodeType::GLSLVolatileModifier:
    case ASTNodeType::GLSLRestrictModifier:
        if (isGLSLInput)
            return (as<VarDeclBase>(decl) && (isGlobalDecl(decl)) || as<ParamDecl>(decl) ||
                    as<GLSLInterfaceBlockDecl>(decl)) ||
                   as<StructDecl>(getParentDecl(decl)) && isGlobalDecl(getParentDecl(decl));
        return (
            as<VarDeclBase>(decl) && (isGlobalDecl(decl)) || as<ParamDecl>(decl) ||
            as<GLSLInterfaceBlockDecl>(decl));

    case ASTNodeType::GloballyCoherentModifier:
    case ASTNodeType::HLSLVolatileModifier:
        if (isGLSLInput)
            return as<VarDecl>(decl) &&
                       (isGlobalDecl(decl) || as<StructDecl>(getParentDecl(decl)) ||
                        as<GLSLInterfaceBlockDecl>(decl)) ||
                   as<VarDeclBase>(decl) && isGlobalDecl(decl) || as<ParamDecl>(decl) ||
                   (as<StructDecl>(getParentDecl(decl)) && isGlobalDecl(getParentDecl(decl)));
        return as<VarDecl>(decl) && (isGlobalDecl(decl) || as<StructDecl>(getParentDecl(decl)) ||
                                     as<GLSLInterfaceBlockDecl>(decl));

        // Allowed only on parameters, struct fields and global variables.
    case ASTNodeType::InterpolationModeModifier:
    case ASTNodeType::HLSLNoInterpolationModifier:
    case ASTNodeType::HLSLNoPerspectiveModifier:
    case ASTNodeType::HLSLLinearModifier:
    case ASTNodeType::HLSLSampleModifier:
    case ASTNodeType::HLSLCentroidModifier:
    case ASTNodeType::PerVertexModifier:
    case ASTNodeType::HLSLUniformModifier:
    case ASTNodeType::DynamicUniformModifier:
        return (as<VarDeclBase>(decl) &&
                (isGlobalDecl(decl) || as<StructDecl>(getParentDecl(decl)))) ||
               as<ParamDecl>(decl);

    case ASTNodeType::HLSLSemantic:
    case ASTNodeType::HLSLLayoutSemantic:
    case ASTNodeType::HLSLRegisterSemantic:
    case ASTNodeType::HLSLPackOffsetSemantic:
    case ASTNodeType::HLSLSimpleSemantic:
        return (as<VarDeclBase>(decl) &&
                (isGlobalDecl(decl) || as<StructDecl>(getParentDecl(decl)))) ||
               as<ParamDecl>(decl) || as<FuncDecl>(decl);

        // Allowed only on functions
    case ASTNodeType::IntrinsicOpModifier:
    case ASTNodeType::SpecializedForTargetModifier:
    case ASTNodeType::InlineModifier:
    case ASTNodeType::PrefixModifier:
    case ASTNodeType::PostfixModifier:
        return as<CallableDecl>(decl);

    case ASTNodeType::BuiltinModifier:
    case ASTNodeType::PublicModifier:
    case ASTNodeType::PrivateModifier:
    case ASTNodeType::InternalModifier:
    case ASTNodeType::ExternModifier:
    case ASTNodeType::HLSLExportModifier:
    case ASTNodeType::ExternCppModifier:
        return as<VarDeclBase>(decl) || as<AggTypeDeclBase>(decl) || as<NamespaceDeclBase>(decl) ||
               as<CallableDecl>(decl) || as<TypeDefDecl>(decl) || as<PropertyDecl>(decl) ||
               as<SyntaxDecl>(decl) || as<AttributeDecl>(decl) || as<InheritanceDecl>(decl);

    case ASTNodeType::ExportedModifier:
        return as<ImportDecl>(decl);

    case ASTNodeType::ConstModifier:
    case ASTNodeType::HLSLStaticModifier:
    case ASTNodeType::ConstExprModifier:
    case ASTNodeType::PreciseModifier:
        return as<VarDeclBase>(decl) || as<CallableDecl>(decl);

    case ASTNodeType::ActualGlobalModifier:
    case ASTNodeType::MatrixLayoutModifier:
    case ASTNodeType::RowMajorLayoutModifier:
    case ASTNodeType::HLSLRowMajorLayoutModifier:
    case ASTNodeType::GLSLColumnMajorLayoutModifier:
    case ASTNodeType::ColumnMajorLayoutModifier:
    case ASTNodeType::HLSLColumnMajorLayoutModifier:
    case ASTNodeType::GLSLRowMajorLayoutModifier:
    case ASTNodeType::HLSLEffectSharedModifier:
        return as<VarDeclBase>(decl) || as<GLSLInterfaceBlockDecl>(decl);

    case ASTNodeType::GLSLPrecisionModifier:
        return as<VarDeclBase>(decl) || as<GLSLInterfaceBlockDecl>(decl) || as<CallableDecl>(decl);
    case ASTNodeType::HLSLGroupSharedModifier:
        // groupshared must be global or static.
        if (!as<VarDeclBase>(decl))
            return false;
        return isGlobalDecl(decl) || isEffectivelyStatic(decl);
    default:
        return true;
    }
}

void GLSLBindingOffsetTracker::setBindingOffset(int binding, int64_t byteOffset)
{
    bindingToByteOffset.set(binding, byteOffset);
}

int64_t GLSLBindingOffsetTracker::getNextBindingOffset(int binding)
{
    int64_t currentOffset;
    if (bindingToByteOffset.addIfNotExists(binding, 0))
        currentOffset = 0;
    else
        currentOffset = bindingToByteOffset.getValue(binding) + sizeof(uint32_t);

    bindingToByteOffset.set(binding, currentOffset + sizeof(uint32_t));
    return currentOffset;
}

AttributeBase* SemanticsVisitor::checkGLSLLayoutAttribute(
    UncheckedGLSLLayoutAttribute* uncheckedAttr,
    ModifiableSyntaxNode* attrTarget)
{
    SLANG_ASSERT(uncheckedAttr->args.getCount() == 1);

    Attribute* attr = nullptr;

    // False if the current unchecked attribute node is deleted and does not result in a new checked
    // attribute.
    bool addNode = true;

    if (as<UncheckedGLSLBindingLayoutAttribute>(uncheckedAttr) ||
        as<UncheckedGLSLSetLayoutAttribute>(uncheckedAttr))
    {
        // Binding and set are coupled together as a descriptor table slot resource for codegen.
        // Attempt to retrieve and annotate an existing binding attribute or create a new one.
        attr = attrTarget->findModifier<GLSLBindingAttribute>();
        if (!attr)
        {
            attr = m_astBuilder->create<GLSLBindingAttribute>();
        }
        else
        {
            addNode = false;
        }

        // `validateAttribute`, which will be called to parse the binding arguments, also accepts
        // modifiers from vk::binding and gl::binding where both set and binding are specified.
        // Binding is the first and set is the second argument - specify them explicitly here.
        if (as<UncheckedGLSLBindingLayoutAttribute>(uncheckedAttr))
        {
            uncheckedAttr->args.add(nullptr);
        }
        else
        {
            uncheckedAttr->args.add(uncheckedAttr->args[0]);
            uncheckedAttr->args[0] = nullptr;
        }

        SLANG_ASSERT(uncheckedAttr->args.getCount() == 2);
    }

#define CASE(UncheckedType, CheckedType)            \
    else if (as<UncheckedType>(uncheckedAttr))      \
    {                                               \
        attr = m_astBuilder->create<CheckedType>(); \
    }

    CASE(UncheckedGLSLOffsetLayoutAttribute, GLSLOffsetLayoutAttribute)
    CASE(UncheckedGLSLInputAttachmentIndexLayoutAttribute, GLSLInputAttachmentIndexLayoutAttribute)
    CASE(UncheckedGLSLLocationLayoutAttribute, GLSLLocationAttribute)
    CASE(UncheckedGLSLIndexLayoutAttribute, GLSLIndexAttribute)
    CASE(UncheckedGLSLConstantIdAttribute, VkConstantIdAttribute)
    CASE(UncheckedGLSLRayPayloadAttribute, VulkanRayPayloadAttribute)
    CASE(UncheckedGLSLRayPayloadInAttribute, VulkanRayPayloadInAttribute)
    CASE(UncheckedGLSLHitObjectAttributesAttribute, VulkanHitObjectAttributesAttribute)
    CASE(UncheckedGLSLCallablePayloadAttribute, VulkanCallablePayloadAttribute)
    CASE(UncheckedGLSLCallablePayloadInAttribute, VulkanCallablePayloadInAttribute)
    else
    {
        getSink()->diagnose(uncheckedAttr, Diagnostics::unrecognizedGLSLLayoutQualifier);
    }
#undef CASE

    if (attr)
    {
        attr->keywordName = uncheckedAttr->keywordName;
        attr->originalIdentifierToken = uncheckedAttr->originalIdentifierToken;
        attr->args = uncheckedAttr->args;
        attr->loc = uncheckedAttr->loc;

        // Offset constant folding computation is deferred until all other modifiers are checked to
        // ensure bindinig is checked first.
        if (!as<GLSLOffsetLayoutAttribute>(attr))
        {
            validateAttribute(attr, nullptr, attrTarget);
        }
    }

    if (!addNode)
    {
        attr = nullptr;
    }

    return attr;
}

Modifier* SemanticsVisitor::checkModifier(
    Modifier* m,
    ModifiableSyntaxNode* syntaxNode,
    bool ignoreUnallowedModifier)
{
    if (auto hlslUncheckedAttribute = as<UncheckedAttribute>(m))
    {
        // We have an HLSL `[name(arg,...)]` attribute, and we'd like
        // to check that it is provides all the expected arguments
        //
        // First, look up the attribute name in the current scope to find
        // the right syntax class to instantiate.
        //

        auto checkedAttr = checkAttribute(hlslUncheckedAttribute, syntaxNode);

        if (as<UnscopedEnumAttribute>(checkedAttr))
        {
            if (auto parentDecl = as<ContainerDecl>(getParentDecl(as<Decl>(syntaxNode))))
                parentDecl->invalidateMemberDictionary();
            return getASTBuilder()->create<TransparentModifier>();
        }
        return checkedAttr;
    }

    if (auto decl = as<Decl>(syntaxNode))
    {
        auto moduleDecl = getModuleDecl(decl);
        bool isGLSLInput = getOptionSet().getBoolOption(CompilerOptionName::AllowGLSL);
        if (!isGLSLInput && moduleDecl && moduleDecl->findModifier<GLSLModuleModifier>())
            isGLSLInput = true;
        if (!isModifierAllowedOnDecl(isGLSLInput, m->astNodeType, decl))
        {
            if (!ignoreUnallowedModifier)
            {
                getSink()->diagnose(m, Diagnostics::modifierNotAllowed, m);
                return nullptr;
            }
            return m;
        }
    }

    if (auto glslLayoutAttribute = as<UncheckedGLSLLayoutAttribute>(m))
    {
        return checkGLSLLayoutAttribute(glslLayoutAttribute, syntaxNode);
    }

    if (const auto glslImplicitOffsetAttribute = as<GLSLImplicitOffsetLayoutAttribute>(m))
    {
        auto offsetAttr = m_astBuilder->create<GLSLOffsetLayoutAttribute>();
        offsetAttr->loc = glslImplicitOffsetAttribute->loc;

        // Offset constant folding computation is deferred until all other modifiers are checked to
        // ensure bindinig is checked first.
        return offsetAttr;
    }

    MemoryQualifierSetModifier::Flags::MemoryQualifiersBit memoryQualifierBit =
        MemoryQualifierSetModifier::Flags::kNone;
    if (as<GloballyCoherentModifier>(m))
        memoryQualifierBit = MemoryQualifierSetModifier::Flags::kCoherent;
    else if (as<GLSLReadOnlyModifier>(m))
        memoryQualifierBit = MemoryQualifierSetModifier::Flags::kReadOnly;
    else if (as<GLSLWriteOnlyModifier>(m))
        memoryQualifierBit = MemoryQualifierSetModifier::Flags::kWriteOnly;
    else if (as<GLSLVolatileModifier>(m))
        memoryQualifierBit = MemoryQualifierSetModifier::Flags::kVolatile;
    else if (as<GLSLRestrictModifier>(m))
        memoryQualifierBit = MemoryQualifierSetModifier::Flags::kRestrict;
    if (memoryQualifierBit != MemoryQualifierSetModifier::Flags::kNone)
    {
        bool newModifier = false;
        MemoryQualifierSetModifier* memoryQualifiers =
            syntaxNode->findModifier<MemoryQualifierSetModifier>();
        if (!memoryQualifiers)
        {
            newModifier = true;
            memoryQualifiers = getASTBuilder()->create<MemoryQualifierSetModifier>();
        }
        memoryQualifiers->addQualifier(m, memoryQualifierBit);
        if (newModifier)
        {
            m->next = memoryQualifiers;
            return memoryQualifiers;
        }
        return nullptr;
    }

    if (auto hlslSemantic = as<HLSLSimpleSemantic>(m))
    {
        if (hlslSemantic->name.getName() == getSession()->getCompletionRequestTokenName())
        {
            getLinkage()->contentAssistInfo.completionSuggestions.scopeKind =
                CompletionSuggestions::ScopeKind::HLSLSemantics;
        }
    }

    if (const auto externModifier = as<ExternModifier>(m))
    {
        if (auto varDecl = as<VarDeclBase>(syntaxNode))
        {
            if (auto parentExtension = as<ExtensionDecl>(varDecl->parentDecl))
            {
                auto originalMemberLookup = lookUpMember(
                    m_astBuilder,
                    this,
                    varDecl->getName(),
                    parentExtension->targetType,
                    parentExtension->ownedScope);
                LookupResult filteredResult;
                for (auto item : originalMemberLookup.items)
                {
                    if (item.declRef.getDecl() != varDecl)
                        AddToLookupResult(filteredResult, item);
                }
                if (filteredResult.isValid() && !filteredResult.isOverloaded())
                {
                    auto extensionExternMemberModifier =
                        m_astBuilder->create<ExtensionExternVarModifier>();
                    extensionExternMemberModifier->originalDecl = filteredResult.item.declRef;
                    return extensionExternMemberModifier;
                }
                else if (filteredResult.isOverloaded())
                {
                    getSink()->diagnose(
                        varDecl,
                        Diagnostics::ambiguousOriginalDefintionOfExternDecl,
                        varDecl);
                }
                else
                {
                    getSink()->diagnose(
                        varDecl,
                        Diagnostics::missingOriginalDefintionOfExternDecl,
                        varDecl);
                }
            }
            // The next part of the check is to make sure the type defined here is consistent with
            // the original definition. Since we haven't checked the type of this decl yet, we defer
            // that until we have fully checked decl. See
            // SemanticsDeclHeaderVisitor::checkExtensionExternVarAttribute.
        }
    }

    if (auto packOffsetModifier = as<HLSLPackOffsetSemantic>(m))
    {
        if (!packOffsetModifier->registerName.getContent().startsWith("c"))
        {
            getSink()->diagnose(
                packOffsetModifier,
                Diagnostics::unknownRegisterClass,
                packOffsetModifier->registerName);
            return m;
        }
        auto uniformOffset =
            stringToInt(packOffsetModifier->registerName.getContent().tail(1)) * 16;
        if (packOffsetModifier->componentMask.getContentLength())
        {
            switch (packOffsetModifier->componentMask.getContent()[0])
            {
            case 'x':
                uniformOffset += 0;
                break;
            case 'y':
                uniformOffset += 4;
                break;
            case 'z':
                uniformOffset += 8;
                break;
            case 'w':
                uniformOffset += 12;
                break;
            default:
                getSink()->diagnose(
                    packOffsetModifier,
                    Diagnostics::invalidComponentMask,
                    packOffsetModifier->componentMask);
                break;
            }
        }
        packOffsetModifier->uniformOffset = uniformOffset;
        return packOffsetModifier;
    }

    if (auto targetIntrinsic = as<TargetIntrinsicModifier>(m))
    {
        // TODO: verify that the predicate is one we understand
        if (targetIntrinsic->scrutinee.name)
        {
            if (auto genDecl = as<ContainerDecl>(syntaxNode))
            {
                auto scrutineeResults = lookUp(
                    m_astBuilder,
                    this,
                    targetIntrinsic->scrutinee.name,
                    genDecl->ownedScope);
                if (!scrutineeResults.isValid())
                {
                    getSink()->diagnose(
                        targetIntrinsic->scrutinee.loc,
                        Diagnostics::undefinedIdentifier2,
                        targetIntrinsic->scrutinee.name);
                }
                if (scrutineeResults.isOverloaded())
                {
                    getSink()->diagnose(
                        targetIntrinsic->scrutinee.loc,
                        Diagnostics::ambiguousReference,
                        targetIntrinsic->scrutinee.name);
                }
                targetIntrinsic->scrutineeDeclRef = scrutineeResults.item.declRef;
            }
        }
    }

    if (as<PrivateModifier>(m))
    {
        if (auto decl = as<Decl>(syntaxNode))
        {
            if (isGlobalDecl(decl))
            {
                getSink()->diagnose(
                    m,
                    Diagnostics::invalidUseOfPrivateVisibility,
                    as<Decl>(syntaxNode));
                return m;
            }
        }
        if (as<NamespaceDeclBase>(syntaxNode))
        {
            getSink()->diagnose(
                m,
                Diagnostics::invalidVisibilityModifierOnTypeOfDecl,
                syntaxNode->astNodeType);
            return m;
        }
        else if (auto decl = as<Decl>(syntaxNode))
        {
            // Interface requirements can't be private.
            if (isInterfaceRequirement(decl))
            {
                getSink()->diagnose(
                    m,
                    Diagnostics::invalidUseOfPrivateVisibility,
                    as<Decl>(syntaxNode));
            }
        }
    }
    else if (as<InternalModifier>(m))
    {
        if (as<NamespaceDeclBase>(syntaxNode))
        {
            getSink()->diagnose(
                m,
                Diagnostics::invalidVisibilityModifierOnTypeOfDecl,
                syntaxNode->astNodeType);
            return m;
        }
    }

    if (auto attr = as<GLSLLayoutLocalSizeAttribute>(m))
    {
        SLANG_ASSERT(attr->args.getCount() == 3);

        // GLSLLayoutLocalSizeAttribute is always attached to an EmptyDecl.
        auto decl = as<EmptyDecl>(syntaxNode);
        SLANG_ASSERT(decl);

        for (int i = 0; i < 3; ++i)
        {
            attr->extents[i] = nullptr;

            auto arg = attr->args[i];
            if (arg)
            {
                auto specConstDecl = tryGetIntSpecializationConstant(arg);
                if (specConstDecl)
                {
                    attr->specConstExtents[i] = specConstDecl;
                    continue;
                }

                auto intValue = checkConstantIntVal(arg);
                if (!intValue)
                {
                    return nullptr;
                }
                if (auto cintVal = as<ConstantIntVal>(intValue))
                {
                    if (attr->axisIsSpecConstId[i])
                    {
                        // This integer should actually be a reference to a
                        // specialization constant with this ID.
                        Int specConstId = cintVal->getValue();

                        for (auto member : decl->parentDecl->members)
                        {
                            auto constantId = member->findModifier<VkConstantIdAttribute>();
                            if (constantId)
                            {
                                SLANG_ASSERT(constantId->args.getCount() == 1);
                                auto id = checkConstantIntVal(constantId->args[0]);
                                if (id->getValue() == specConstId)
                                {
                                    attr->specConstExtents[i] =
                                        DeclRef<VarDeclBase>(member->getDefaultDeclRef());
                                    break;
                                }
                            }
                        }

                        // If not found, we need to create a new specialization
                        // constant with this ID.
                        if (!attr->specConstExtents[i])
                        {
                            auto specConstVarDecl = getASTBuilder()->create<VarDecl>();
                            auto constantIdModifier =
                                getASTBuilder()->create<VkConstantIdAttribute>();
                            constantIdModifier->location = (int32_t)specConstId;
                            specConstVarDecl->type.type = getASTBuilder()->getIntType();
                            addModifier(specConstVarDecl, constantIdModifier);
                            decl->parentDecl->addMember(specConstVarDecl);
                            attr->specConstExtents[i] =
                                DeclRef<VarDeclBase>(specConstVarDecl->getDefaultDeclRef());
                        }
                        continue;
                    }
                    else if (cintVal->getValue() < 1)
                    {
                        getSink()->diagnose(
                            attr,
                            Diagnostics::nonPositiveNumThreads,
                            cintVal->getValue());
                        return nullptr;
                    }
                }
                attr->extents[i] = intValue;
            }
            else
            {
                attr->extents[i] = m_astBuilder->getIntVal(m_astBuilder->getIntType(), 1);
            }
        }
    }

    // Default behavior is to leave things as they are,
    // and assume that modifiers are mostly already checked.
    //
    // TODO: This would be a good place to validate that
    // a modifier is actually valid for the thing it is
    // being applied to, and potentially to check that
    // it isn't in conflict with any other modifiers
    // on the same declaration.

    return m;
}

void SemanticsVisitor::checkVisibility(Decl* decl)
{
    if (as<AccessorDecl>(decl))
    {
        return;
    }
    ShortList<Type*> typesToVerify;
    if (auto varDecl = as<VarDeclBase>(decl))
    {
        typesToVerify.add(varDecl->type);
    }
    else if (auto callable = as<CallableDecl>(decl))
    {
        typesToVerify.add(callable->returnType);
        typesToVerify.add(callable->errorType);
        for (auto param : callable->getParameters())
        {
            typesToVerify.add(param->type);
        }
    }
    else if (auto propertyDecl = as<PropertyDecl>(decl))
    {
        typesToVerify.add(propertyDecl->type);
    }
    else if (as<AggTypeDeclBase>(decl))
    {
    }
    else if (auto typeDecl = as<TypeDefDecl>(decl))
    {
        typesToVerify.add(typeDecl->type);
    }
    else
    {
        return;
    }
    auto thisVisibility = getDeclVisibility(decl);

    // First, we check that the decl's type does not have lower visibility.
    for (auto type : typesToVerify)
    {
        if (!type)
            continue;
        DeclVisibility typeVisibility = getTypeVisibility(type);
        if (typeVisibility < thisVisibility)
        {
            getSink()->diagnose(decl, Diagnostics::useOfLessVisibleType, decl, type);
            break;
        }
    }

    // Next, we check that the decl does not have higher visiblity than its parent.
    Decl* parentDecl = decl;
    for (; parentDecl; parentDecl = parentDecl->parentDecl)
    {
        if (as<AggTypeDeclBase>(parentDecl))
            break;
    }
    if (!parentDecl)
        return;
    auto parentVisibility = getDeclVisibility(parentDecl);
    if (thisVisibility > parentVisibility)
    {
        getSink()->diagnose(decl, Diagnostics::declCannotHaveHigherVisibility, decl, parentDecl);
    }
}

void postProcessingOnModifiers(Modifiers& modifiers)
{
    // compress all `require` nodes into 1 `require` modifier
    RequireCapabilityAttribute* firstRequire = nullptr;
    Modifier* previous = nullptr;
    Modifier* next = nullptr;
    for (auto m = modifiers.first; m != nullptr; m = next)
    {
        next = m->next;
        //

        if (auto req = as<RequireCapabilityAttribute>(m))
        {
            if (!firstRequire)
            {
                firstRequire = req;
                previous = m;
                continue;
            }
            firstRequire->capabilitySet.unionWith(req->capabilitySet);
            if (previous)
                previous->next = next;
            continue;
        }

        //
        previous = m;
    }
}

void SemanticsVisitor::checkModifiers(ModifiableSyntaxNode* syntaxNode)
{
    // TODO(tfoley): need to make sure this only
    // performs semantic checks on a `SharedModifier` once...

    // The process of checking a modifier may produce a new modifier in its place,
    // so we will build up a new linked list of modifiers that will replace
    // the old list.
    Modifier* resultModifiers = nullptr;
    Modifier** resultModifierLink = &resultModifiers;

    // We will keep track of the modifiers for each conflict group.
    Dictionary<ASTNodeType, Modifier*> mapExclusiveGroupToModifier;

    Modifier* modifier = syntaxNode->modifiers.first;
    bool ignoreUnallowedModifier = false;
    while (modifier)
    {
        // Check if a modifier belonging to the same conflict group is already
        // defined.
        Modifier* existingModifier = nullptr;
        auto conflictGroup = getModifierConflictGroupKind(modifier->astNodeType);
        if (conflictGroup != ASTNodeType::NodeBase)
        {
            if (mapExclusiveGroupToModifier.tryGetValue(conflictGroup, existingModifier))
            {
                getSink()->diagnose(
                    modifier->loc,
                    Diagnostics::duplicateModifier,
                    modifier,
                    existingModifier);
            }
            mapExclusiveGroupToModifier[conflictGroup] = modifier;
        }

        // Because we are rewriting the list in place, we need to extract
        // the next modifier here (not at the end of the loop).
        auto next = modifier->next;

        // We also go ahead and clobber the `next` field on the modifier
        // itself, so that the default behavior of `checkModifier()` can
        // be to return a single unlinked modifier.
        modifier->next = nullptr;

        // For any modifiers appears after "SharedModifiers", we will not diagnose
        // an error if the modifier is not allowed on the declaration.
        if (as<SharedModifiers>(modifier))
            ignoreUnallowedModifier = true;

        // may return a list of modifiers
        auto checkedModifier = checkModifier(modifier, syntaxNode, ignoreUnallowedModifier);

        if (checkedModifier)
        {
            // If checking gave us a modifier to add, then we
            // had better add it.

            // Just in case `checkModifier` ever returns multiple
            // modifiers, lets advance to the end of the list we
            // are building.
            while (*resultModifierLink)
                resultModifierLink = &(*resultModifierLink)->next;

            // attach the new modifier at the end of the list,
            // and now set the "link" to point to its `next` field
            *resultModifierLink = checkedModifier;
            resultModifierLink = &checkedModifier->next;
        }

        // Move along to the next modifier
        modifier = next;
    }

    // Whether we actually re-wrote anything or note, lets
    // install the new list of modifiers on the declaration
    syntaxNode->modifiers.first = resultModifiers;

    // GLSL offset layout qualifiers are resolved after all other modifiers are checked to ensure
    // binding layout qualifiers are processed first.
    if (auto glslOffsetAttribute = syntaxNode->findModifier<GLSLOffsetLayoutAttribute>())
    {
        if (const auto glslBindingAttribute = syntaxNode->findModifier<GLSLBindingAttribute>())
        {
            if (glslOffsetAttribute->args.getCount() == 0)
            {
                glslOffsetAttribute->offset = getGLSLBindingOffsetTracker()->getNextBindingOffset(
                    glslBindingAttribute->binding);
            }
            else if (const auto constVal = checkConstantIntVal(glslOffsetAttribute->args[0]))
            {
                glslOffsetAttribute->offset = uint64_t(constVal->getValue());
                getGLSLBindingOffsetTracker()->setBindingOffset(
                    glslBindingAttribute->binding,
                    glslOffsetAttribute->offset);
            }
        }
        else
        {
            getSink()->diagnose(glslOffsetAttribute, Diagnostics::missingLayoutBindingModifier);
        }
    }

    postProcessingOnModifiers(syntaxNode->modifiers);
}

void SemanticsVisitor::checkRayPayloadStructFields(StructDecl* structDecl)
{
    // Only check structs with the [raypayload] attribute
    if (!structDecl->findModifier<RayPayloadAttribute>())
    {
        return;
    }

    // Check each field in the struct
    for (auto member : structDecl->members)
    {
        auto fieldVarDecl = as<VarDeclBase>(member);
        if (!fieldVarDecl)
        {
            continue;
        }

        bool hasReadModifier = fieldVarDecl->findModifier<RayPayloadReadSemantic>() != nullptr;
        bool hasWriteModifier = fieldVarDecl->findModifier<RayPayloadWriteSemantic>() != nullptr;

        if (!hasReadModifier && !hasWriteModifier)
        {
            // Emit the diagnostic error
            getSink()->diagnose(
                fieldVarDecl,
                Diagnostics::rayPayloadFieldMissingAccessQualifiers,
                fieldVarDecl->getName());
        }
    }
}


} // namespace Slang
