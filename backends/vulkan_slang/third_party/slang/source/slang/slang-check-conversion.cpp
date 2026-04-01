// slang-check-conversion.cpp
#include "slang-check-impl.h"

// This file contains semantic-checking logic for dealing
// with conversion (both implicit and explicit) of expressions
// from one type to another.
//
// Type conversion is also the point at which a C-style initializer
// list (e.g., `float4 a = { 1, 2, 3, 4 };`) is validated against
// the desired type, so this file also contains all of the logic
// associated with validating initializer lists.

namespace Slang
{
ConversionCost SemanticsVisitor::getImplicitConversionCost(Decl* decl)
{
    if (auto modifier = decl->findModifier<ImplicitConversionModifier>())
    {
        return modifier->cost;
    }

    return kConversionCost_Explicit;
}

BuiltinConversionKind SemanticsVisitor::getImplicitConversionBuiltinKind(Decl* decl)
{
    if (auto modifier = decl->findModifier<ImplicitConversionModifier>())
    {
        return modifier->builtinConversionKind;
    }

    return kBuiltinConversion_Unknown;
}

bool SemanticsVisitor::isEffectivelyScalarForInitializerLists(Type* type)
{
    if (as<CoopVectorExpressionType>(type))
        return false;
    if (as<ArrayExpressionType>(type))
        return false;
    if (as<VectorExpressionType>(type))
        return false;
    if (as<MatrixExpressionType>(type))
        return false;

    if (as<BasicExpressionType>(type))
    {
        return true;
    }

    if (as<ResourceType>(type))
    {
        return true;
    }
    if (as<UntypedBufferResourceType>(type))
    {
        return true;
    }
    if (as<SamplerStateType>(type))
    {
        return true;
    }

    if (auto declRefType = as<DeclRefType>(type))
    {
        if (as<StructDecl>(declRefType->getDeclRef()))
            return false;
    }

    return true;
}

bool SemanticsVisitor::shouldUseInitializerDirectly(Type* toType, Expr* fromExpr)
{
    // A nested initializer list should always be used directly.
    //
    if (as<InitializerListExpr>(fromExpr))
    {
        return true;
    }

    // If the desired type is a scalar, then we should always initialize
    // directly, since it isn't an aggregate.
    //
    if (isEffectivelyScalarForInitializerLists(toType))
        return true;

    // If the type we are initializing isn't effectively scalar,
    // but the initialization expression *is*, then it doesn't
    // seem like direct initialization is intended.
    //
    if (isEffectivelyScalarForInitializerLists(fromExpr->type))
        return false;

    // Once the above cases are handled, the main thing
    // we want to check for is whether a direct initialization
    // is possible (a type conversion exists).
    //
    return canCoerce(toType, fromExpr->type, fromExpr);
}

bool SemanticsVisitor::_readValueFromInitializerList(
    Type* toType,
    Expr** outToExpr,
    InitializerListExpr* fromInitializerListExpr,
    UInt& ioInitArgIndex)
{
    // First, we will check if we have run out of arguments
    // on the initializer list.
    //
    UInt initArgCount = fromInitializerListExpr->args.getCount();
    if (ioInitArgIndex >= initArgCount)
    {
        // If we are at the end of the initializer list,
        // then our ability to read an argument depends
        // on whether the type we are trying to read
        // is default-initializable.
        //
        // For now, we will just pretend like everything
        // is default-initializable and move along.
        return true;
    }

    // Okay, we have at least one initializer list expression,
    // so we will look at the next expression and decide
    // whether to use it to initialize the desired type
    // directly (possibly via casts), or as the first sub-expression
    // for aggregate initialization.
    //
    auto firstInitExpr = fromInitializerListExpr->args[ioInitArgIndex];
    if (shouldUseInitializerDirectly(toType, firstInitExpr))
    {
        ioInitArgIndex++;
        return _coerce(
            CoercionSite::Initializer,
            toType,
            outToExpr,
            firstInitExpr->type,
            firstInitExpr,
            nullptr);
    }

    // If there is somehow an error in one of the initialization
    // expressions, then everything could be thrown off and we
    // shouldn't keep trying to read arguments.
    //
    if (IsErrorExpr(firstInitExpr))
    {
        // Stop reading arguments, as if we'd reached
        // the end of the list.
        //
        ioInitArgIndex = initArgCount;
        return true;
    }

    // The fallback case is to recursively read the
    // type from the same list as an aggregate.
    //
    return _readAggregateValueFromInitializerList(
        toType,
        outToExpr,
        fromInitializerListExpr,
        ioInitArgIndex);
}

DeclRefType* findBaseStructType(ASTBuilder* astBuilder, DeclRef<StructDecl> structTypeDeclRef)
{
    auto inheritanceDecl =
        getMembersOfType<InheritanceDecl>(astBuilder, structTypeDeclRef).getFirstOrNull();
    if (!inheritanceDecl)
        return nullptr;

    auto baseType = getBaseType(astBuilder, inheritanceDecl);
    auto baseDeclRefType = as<DeclRefType>(baseType);
    if (!baseDeclRefType)
        return nullptr;

    auto baseDeclRef = baseDeclRefType->getDeclRef();
    auto baseStructDeclRef = baseDeclRef.as<StructDecl>();
    if (!baseStructDeclRef)
        return nullptr;

    return baseDeclRefType;
}

DeclRef<StructDecl> findBaseStructDeclRef(
    ASTBuilder* astBuilder,
    DeclRef<StructDecl> structTypeDeclRef)
{
    auto inheritanceDecl =
        getMembersOfType<InheritanceDecl>(astBuilder, structTypeDeclRef).getFirstOrNull();
    if (!inheritanceDecl)
        return DeclRef<StructDecl>();

    auto baseType = getBaseType(astBuilder, inheritanceDecl);
    auto baseDeclRefType = as<DeclRefType>(baseType);
    if (!baseDeclRefType)
        return DeclRef<StructDecl>();

    auto baseDeclRef = baseDeclRefType->getDeclRef();
    auto baseStructDeclRef = baseDeclRef.as<StructDecl>();
    if (!baseStructDeclRef)
        return DeclRef<StructDecl>();

    return baseStructDeclRef;
}

ConstructorDecl* SemanticsVisitor::_getSynthesizedConstructor(
    StructDecl* structDecl,
    ConstructorDecl::ConstructorFlavor flavor)
{
    for (auto ctor : structDecl->getMembersOfType<ConstructorDecl>())
    {
        if (ctor->containsFlavor(flavor))
            return ctor;
    }
    return nullptr;
}

bool SemanticsVisitor::isCStyleType(Type* type, HashSet<Type*>& isVisit)
{
    isVisit.add(type);
    auto cacheResult = [&](bool result)
    {
        getShared()->cacheCStyleType(type, result);
        return result;
    };

    // Check cache first
    if (bool* isCStyle = getShared()->isCStyleType(type))
    {
        return *isCStyle;
    }

    // 1. It has to be basic scalar, vector or matrix type, or user-defined struct.
    if (as<VectorExpressionType>(type) || as<MatrixExpressionType>(type) ||
        as<BasicExpressionType>(type) || isDeclRefTypeOf<EnumDecl>(type).getDecl())
        return cacheResult(true);


    // A tuple type is C-style if all of its members are C-style.
    if (auto tupleType = as<TupleType>(type))
    {
        for (Index i = 0; i < tupleType->getMemberCount(); i++)
        {
            auto elementType = tupleType->getMember(i);
            // Avoid infinite loop in case of circular reference.
            if (isVisit.contains(elementType))
                return cacheResult(false);
            if (!isCStyleType(elementType, isVisit))
                return cacheResult(false);
        }
        return cacheResult(true);
    }

    if (auto structDecl = isDeclRefTypeOf<StructDecl>(type).getDecl())
    {
        // 2. It cannot have inheritance, but inherit from interface is fine.
        for (auto inheritanceDecl : structDecl->getMembersOfType<InheritanceDecl>())
        {
            if (!isDeclRefTypeOf<InterfaceDecl>(inheritanceDecl->base.type))
            {
                return cacheResult(false);
            }
        }

        // 3. It cannot have explicit constructor
        if (_hasExplicitConstructor(structDecl, true))
            return cacheResult(false);

        // 4. All of its members have to have the same visibility as the struct itself.
        DeclVisibility structVisibility = getDeclVisibility(structDecl);
        for (auto varDecl : structDecl->getMembersOfType<VarDeclBase>())
        {
            if (getDeclVisibility(varDecl) != structVisibility)
            {
                return cacheResult(false);
            }
        }

        for (auto varDecl : structDecl->getMembersOfType<VarDeclBase>())
        {
            Type* varType = varDecl->getType();

            if (isDeclRefTypeOf<StructDecl>(varType))
            {
                // Avoid infinite loop in case of circular reference.
                if (isVisit.contains(varType))
                    continue;
            }

            // Recursively check the type of the member.
            if (!isCStyleType(varType, isVisit))
                return cacheResult(false);
        }
    }

    // 5. All its members are legacy C-Style structs or arrays of legacy C-style structs
    if (auto arrayType = as<ArrayExpressionType>(type))
    {
        if (arrayType->isUnsized())
        {
            return cacheResult(false);
        }

        auto elementType = arrayType->getElementType();
        if (isDeclRefTypeOf<StructDecl>(elementType))
        {
            // Avoid infinite loop in case of circular reference.
            if (isVisit.contains(elementType))
                cacheResult(true);
        }

        if (!isCStyleType(elementType, isVisit))
            return cacheResult(false);
    }

    return cacheResult(true);
}

Expr* SemanticsVisitor::_createCtorInvokeExpr(
    Type* toType,
    const SourceLoc& loc,
    const List<Expr*>& coercedArgs)
{
    auto* varExpr = getASTBuilder()->create<VarExpr>();
    varExpr->type = (QualType)getASTBuilder()->getTypeType(toType);
    varExpr->declRef = isDeclRefTypeOf<Decl>(toType);

    auto* constructorExpr = getASTBuilder()->create<ExplicitCtorInvokeExpr>();
    constructorExpr->functionExpr = varExpr;
    constructorExpr->arguments.addRange(coercedArgs);
    constructorExpr->loc = loc;

    return constructorExpr;
}

// translation from initializer list to constructor invocation if the struct has constructor.
bool SemanticsVisitor::createInvokeExprForExplicitCtor(
    Type* toType,
    InitializerListExpr* fromInitializerListExpr,
    Expr** outExpr)
{
    if (auto toStructDeclRef = isDeclRefTypeOf<StructDecl>(toType))
    {
        // TODO: This is just a special case for a backwards-compatibility feature
        // for HLSL, this flag will imply that the initializer list is synthesized
        // for a type cast from a literal zero to a 'struct'. In this case, we will fall
        // back to legacy initializer list logic.
        if (!fromInitializerListExpr->useCStyleInitialization)
        {
            HashSet<Type*> isVisit;
            if (!isCStyleType(toType, isVisit))
                return false;
        }

        if (_hasExplicitConstructor(toStructDeclRef.getDecl(), false))
        {
            auto ctorInvokeExpr = _createCtorInvokeExpr(
                toType,
                fromInitializerListExpr->loc,
                fromInitializerListExpr->args);

            DiagnosticSink tempSink(getSourceManager(), nullptr);
            SemanticsVisitor subVisitor(withSink(&tempSink));
            ctorInvokeExpr = subVisitor.CheckTerm(ctorInvokeExpr);

            if (tempSink.getErrorCount())
            {
                HashSet<Type*> isVisit;
                if (!isCStyleType(toType, isVisit))
                {
                    Slang::ComPtr<ISlangBlob> blob;
                    tempSink.getBlobIfNeeded(blob.writeRef());
                    getSink()->diagnoseRaw(
                        Severity::Error,
                        static_cast<char const*>(blob->getBufferPointer()));
                }
                return false;
            }

            if (outExpr)
            {
                *outExpr = ctorInvokeExpr;
                return true;
            }
        }
    }
    return false;
}

bool SemanticsVisitor::createInvokeExprForSynthesizedCtor(
    Type* toType,
    InitializerListExpr* fromInitializerListExpr,
    Expr** outExpr)
{
    StructDecl* structDecl = isDeclRefTypeOf<StructDecl>(toType).getDecl();

    if (!structDecl)
        return false;

    HashSet<Type*> isVisit;
    bool isCStyle = false;
    if (!_getSynthesizedConstructor(
            structDecl,
            ConstructorDecl::ConstructorFlavor::SynthesizedDefault))
    {
        // When a struct has no constructor and it's not a C-style type, the initializer list is
        // invalid.
        isCStyle = isCStyleType(toType, isVisit);

        // WAR: We currently still has to allow legacy initializer list for array type until we have
        // more proper solution for array initialization, so if the right hand side is an array
        // type, we will not report error and fall-back to legacy initializer list logic.
        bool isArrayType = as<ArrayExpressionType>(toType) != nullptr;
        if (!isCStyle && !isArrayType)
        {
            getSink()->diagnose(
                fromInitializerListExpr->loc,
                Diagnostics::cannotUseInitializerListForType,
                toType);
        }

        return false;
    }

    isCStyle = isCStyleType(toType, isVisit);
    // TODO: This is just a special case for a backwards-compatibility feature
    // for HLSL, this flag will imply that the initializer list is synthesized
    // for a type cast from a literal zero to a 'struct'. In this case, we will fall
    // back to legacy initializer list logic.
    if (!fromInitializerListExpr->useCStyleInitialization)
    {
        if (isCStyle)
            return false;
    }

    DiagnosticSink tempSink(getSourceManager(), nullptr);
    SemanticsVisitor subVisitor(withSink(&tempSink));

    // First make sure the struct is fully checked, otherwise the synthesized constructor may not be
    // created yet.
    subVisitor.ensureDecl(structDecl, DeclCheckState::DefinitionChecked);

    List<Expr*> coercedArgs;
    auto ctorInvokeExpr =
        _createCtorInvokeExpr(toType, fromInitializerListExpr->loc, fromInitializerListExpr->args);

    ctorInvokeExpr = subVisitor.CheckExpr(ctorInvokeExpr);

    if (ctorInvokeExpr)
    {
        if (!tempSink.getErrorCount())
        {
            if (outExpr)
                *outExpr = ctorInvokeExpr;

            return true;
        }
        else if (!isCStyle)
        {
            Slang::ComPtr<ISlangBlob> blob;
            tempSink.getBlobIfNeeded(blob.writeRef());
            getSink()->diagnoseRaw(
                Severity::Error,
                static_cast<char const*>(blob->getBufferPointer()));
            return false;
        }
    }
    return false;
}

bool SemanticsVisitor::_readAggregateValueFromInitializerList(
    Type* inToType,
    Expr** outToExpr,
    InitializerListExpr* fromInitializerListExpr,
    UInt& ioArgIndex)
{
    auto toType = inToType;
    UInt argCount = fromInitializerListExpr->args.getCount();

    // In the case where we need to build a result expression,
    // we will collect the new arguments here
    List<Expr*> coercedArgs;

    if (isEffectivelyScalarForInitializerLists(toType))
    {
        // For any type that is effectively a non-aggregate,
        // we expect to read a single value from the initializer list
        //
        if (ioArgIndex < argCount)
        {
            auto arg = fromInitializerListExpr->args[ioArgIndex++];
            return _coerce(CoercionSite::Initializer, toType, outToExpr, arg->type, arg, nullptr);
        }
        else
        {
            // If there wasn't an initialization
            // expression to be found, then we need
            // to perform default initialization here.
            //
            // We will let this case come through the front-end
            // as an `InitializerListExpr` with zero arguments,
            // and then have the IR generation logic deal with
            // synthesizing default values.
        }
    }
    else if (auto toVecType = as<VectorExpressionType>(toType))
    {
        auto toElementCount = toVecType->getElementCount();
        auto toElementType = toVecType->getElementType();

        UInt elementCount = 0;
        if (auto constElementCount = as<ConstantIntVal>(toElementCount))
        {
            elementCount = (UInt)constElementCount->getValue();
        }
        else
        {
            // We don't know the element count statically,
            // so what are we supposed to be doing?
            //
            if (outToExpr)
            {
                getSink()->diagnose(
                    fromInitializerListExpr,
                    Diagnostics::cannotUseInitializerListForVectorOfUnknownSize,
                    toElementCount);
            }
            return false;
        }

        for (UInt ee = 0; ee < elementCount; ++ee)
        {
            Expr* coercedArg = nullptr;
            bool argResult = _readValueFromInitializerList(
                toElementType,
                outToExpr ? &coercedArg : nullptr,
                fromInitializerListExpr,
                ioArgIndex);

            // No point in trying further if any argument fails
            if (!argResult)
                return false;

            if (coercedArg)
            {
                coercedArgs.add(coercedArg);
            }
        }
    }
    else if (auto toCoopVectorType = as<CoopVectorExpressionType>(toType))
    {
        auto toElementCount = toCoopVectorType->getElementCount();
        auto toElementType = toCoopVectorType->getElementType();

        UInt elementCount = 0;
        if (auto constElementCount = as<ConstantIntVal>(toElementCount))
        {
            elementCount = (UInt)constElementCount->getValue();
        }
        else
        {
            // We don't know the element count statically,
            // so what are we supposed to be doing?
            //
            if (outToExpr)
            {
                getSink()->diagnose(
                    fromInitializerListExpr,
                    Diagnostics::cannotUseInitializerListForCoopVectorOfUnknownSize,
                    toElementCount);
            }
            return false;
        }

        for (UInt ee = 0; ee < elementCount; ++ee)
        {
            Expr* coercedArg = nullptr;
            bool argResult = _readValueFromInitializerList(
                toElementType,
                outToExpr ? &coercedArg : nullptr,
                fromInitializerListExpr,
                ioArgIndex);

            // No point in trying further if any argument fails
            if (!argResult)
                return false;

            if (coercedArg)
            {
                coercedArgs.add(coercedArg);
            }
        }
    }
    else if (auto toArrayType = as<ArrayExpressionType>(toType))
    {
        // TODO(tfoley): If we can compute the size of the array statically,
        // then we want to check that there aren't too many initializers present

        auto toElementType = toArrayType->getElementType();
        if (!toArrayType->isUnsized())
        {
            auto toElementCount = toArrayType->getElementCount();

            // In the case of a sized array, we need to check that the number
            // of elements being initialized matches what was declared.
            //
            UInt elementCount = 0;
            if (auto constElementCount = as<ConstantIntVal>(toElementCount))
            {
                elementCount = (UInt)constElementCount->getValue();
            }
            else
            {
                // We don't know the element count statically,
                // so what are we supposed to be doing?
                //
                if (outToExpr)
                {
                    getSink()->diagnose(
                        fromInitializerListExpr,
                        Diagnostics::cannotUseInitializerListForArrayOfUnknownSize,
                        toElementCount);
                }
                return false;
            }

            for (UInt ee = 0; ee < elementCount; ++ee)
            {
                Expr* coercedArg = nullptr;
                bool argResult = _readValueFromInitializerList(
                    toElementType,
                    outToExpr ? &coercedArg : nullptr,
                    fromInitializerListExpr,
                    ioArgIndex);

                // No point in trying further if any argument fails
                if (!argResult)
                    return false;

                if (coercedArg)
                {
                    coercedArgs.add(coercedArg);
                }
            }
        }
        else
        {
            // In the case of an unsized array type, we will use the
            // number of arguments to the initializer to determine
            // the element count.
            //
            UInt elementCount = 0;
            while (ioArgIndex < argCount)
            {
                Expr* coercedArg = nullptr;
                bool argResult = _readValueFromInitializerList(
                    toElementType,
                    outToExpr ? &coercedArg : nullptr,
                    fromInitializerListExpr,
                    ioArgIndex);

                // No point in trying further if any argument fails
                if (!argResult)
                    return false;

                elementCount++;

                if (coercedArg)
                {
                    coercedArgs.add(coercedArg);
                }
            }

            // We have a new type for the conversion, based on what
            // we learned.
            toType = m_astBuilder->getArrayType(
                toElementType,
                m_astBuilder->getIntVal(m_astBuilder->getIntType(), elementCount));
        }
    }
    else if (auto toMatrixType = as<MatrixExpressionType>(toType))
    {
        // In the general case, the initializer list might comprise
        // both vectors and scalars.
        //
        // The traditional HLSL compilers treat any vectors in
        // the initializer list exactly equivalent to their sequence
        // of scalar elements, and don't care how this might, or
        // might not, align with the rows of the matrix.
        //
        // We will draw a line in the sand and say that an initializer
        // list for a matrix will act as if the matrix type were an
        // array of vectors for the rows.


        UInt rowCount = 0;
        auto toRowType =
            createVectorType(toMatrixType->getElementType(), toMatrixType->getColumnCount());

        if (auto constRowCount = as<ConstantIntVal>(toMatrixType->getRowCount()))
        {
            rowCount = (UInt)constRowCount->getValue();
        }
        else
        {
            // We don't know the element count statically,
            // so what are we supposed to be doing?
            //
            if (outToExpr)
            {
                getSink()->diagnose(
                    fromInitializerListExpr,
                    Diagnostics::cannotUseInitializerListForMatrixOfUnknownSize,
                    toMatrixType->getRowCount());
            }
            return false;
        }

        for (UInt rr = 0; rr < rowCount; ++rr)
        {
            Expr* coercedArg = nullptr;
            bool argResult = _readValueFromInitializerList(
                toRowType,
                outToExpr ? &coercedArg : nullptr,
                fromInitializerListExpr,
                ioArgIndex);

            // No point in trying further if any argument fails
            if (!argResult)
                return false;

            if (coercedArg)
            {
                coercedArgs.add(coercedArg);
            }
        }
    }
    else if (auto tupleType = as<TupleType>(toType))
    {
        for (Index ee = 0; ee < tupleType->getMemberCount(); ++ee)
        {
            auto elementType = tupleType->getMember(ee);
            Expr* coercedArg = nullptr;
            bool argResult = _readValueFromInitializerList(
                elementType,
                outToExpr ? &coercedArg : nullptr,
                fromInitializerListExpr,
                ioArgIndex);

            // No point in trying further if any argument fails
            if (!argResult)
                return false;

            if (coercedArg)
            {
                coercedArgs.add(coercedArg);
            }
        }
    }
    else if (auto toDeclRefType = as<DeclRefType>(toType))
    {
        auto toTypeDeclRef = toDeclRefType->getDeclRef();
        if (auto toStructDeclRef = toTypeDeclRef.as<StructDecl>())
        {
            // Trying to initialize a `struct` type given an initializer list.
            //
            // Before we iterate over the fields, we want to check if this struct
            // inherits from another `struct` type. If so, we want to read
            // an initializer for that base type first.
            //
            if (auto baseStructType = findBaseStructType(m_astBuilder, toStructDeclRef))
            {
                Expr* coercedArg = nullptr;
                bool argResult = _readValueFromInitializerList(
                    baseStructType,
                    outToExpr ? &coercedArg : nullptr,
                    fromInitializerListExpr,
                    ioArgIndex);

                // No point in trying further if any argument fails
                if (!argResult)
                    return false;

                if (coercedArg)
                {
                    coercedArgs.add(coercedArg);
                }
            }

            // We will go through the fields in order and try to match them
            // up with initializer arguments.
            //
            for (auto fieldDeclRef : getMembersOfType<VarDecl>(
                     m_astBuilder,
                     toStructDeclRef,
                     MemberFilterStyle::Instance))
            {
                Expr* coercedArg = nullptr;
                bool argResult = _readValueFromInitializerList(
                    getType(m_astBuilder, fieldDeclRef),
                    outToExpr ? &coercedArg : nullptr,
                    fromInitializerListExpr,
                    ioArgIndex);

                // No point in trying further if any argument fails
                if (!argResult)
                    return false;

                if (coercedArg)
                {
                    coercedArgs.add(coercedArg);
                }
            }
        }
    }
    else
    {
        // We shouldn't get to this case in practice,
        // but just in case we'll consider an initializer
        // list invalid if we are trying to read something
        // off of it that wasn't handled by the cases above.
        //
        if (outToExpr)
        {
            getSink()->diagnose(
                fromInitializerListExpr,
                Diagnostics::cannotUseInitializerListForType,
                inToType);
        }
        return false;
    }

    // We were able to coerce all the arguments given, and so
    // we need to construct a suitable expression to remember the result
    //
    if (outToExpr)
    {
        auto toInitializerListExpr = m_astBuilder->create<InitializerListExpr>();
        toInitializerListExpr->loc = fromInitializerListExpr->loc;
        toInitializerListExpr->type = QualType(toType);
        toInitializerListExpr->args = coercedArgs;

        // Wrap initalizer list args if we're creating a non-differentiable struct within a
        // differentiable function.
        //
        if (auto func = getParentFuncOfVisitor())
        {
            if (func->findModifier<DifferentiableAttribute>() && !isTypeDifferentiable(toType))
            {
                for (auto& arg : toInitializerListExpr->args)
                {
                    if (isTypeDifferentiable(arg->type.type))
                    {
                        auto detachedArg = m_astBuilder->create<DetachExpr>();
                        detachedArg->inner = arg;
                        detachedArg->type = arg->type;
                        arg = detachedArg;
                    }
                }
            }
        }

        *outToExpr = toInitializerListExpr;
    }

    return true;
}

bool SemanticsVisitor::_coerceInitializerList(
    Type* toType,
    Expr** outToExpr,
    InitializerListExpr* fromInitializerListExpr)
{
    UInt argCount = fromInitializerListExpr->args.getCount();
    UInt argIndex = 0;

    // TODO: we should handle the special case of `{0}` as an initializer
    // for arbitrary `struct` types here.

    // If this initializer list has a more specific type than just
    // InitializerListType (i.e. it's already undergone a coercion) we
    // should ensure that we're allowed to coerce from that type to our
    // desired type.
    // If this isn't prohibited, then we can proceed to try and coerce from
    // the initializer list itself; assuming that coercion is closed under
    // composition this shouldn't fail.
    if (!as<InitializerListType>(fromInitializerListExpr->type) &&
        !canCoerce(toType, fromInitializerListExpr->type, nullptr))
        return _failedCoercion(toType, outToExpr, fromInitializerListExpr);

    // Try to invoke the user-defined constructor if it exists. This call will
    // report error diagnostics if the used-defined constructor exists but does not
    // match the initialize list.
    if (createInvokeExprForExplicitCtor(toType, fromInitializerListExpr, outToExpr))
    {
        return true;
    }

    // Try to invoke the synthesized constructor if it exists
    if (createInvokeExprForSynthesizedCtor(toType, fromInitializerListExpr, outToExpr))
    {
        return true;
    }

    // We will fall back to the legacy logic of initialize list.
    if (!_readAggregateValueFromInitializerList(
            toType,
            outToExpr,
            fromInitializerListExpr,
            argIndex))
        return false;

    if (argIndex != argCount)
    {
        if (outToExpr)
        {
            getSink()->diagnose(
                fromInitializerListExpr,
                Diagnostics::tooManyInitializers,
                argIndex,
                argCount);
        }
    }

    return true;
}

bool SemanticsVisitor::_failedCoercion(Type* toType, Expr** outToExpr, Expr* fromExpr)
{
    if (outToExpr)
    {
        // As a special case, if the expression we are trying to convert
        // from is overloaded (implying an ambiguous reference), then we
        // will try to produce a more appropriately tailored error message.
        //
        auto fromType = fromExpr->type.type;
        if (as<OverloadGroupType>(fromType))
        {
            diagnoseAmbiguousReference(fromExpr);
        }
        else
        {
            getSink()->diagnose(fromExpr->loc, Diagnostics::typeMismatch, toType, fromExpr->type);
        }
    }
    return false;
}

/// Do the `left` and `right` modifiers represent the same thing?
static bool _doModifiersMatch(Val* left, Val* right)
{
    if (left == right)
        return true;

    if (left->equals(right))
        return true;

    return false;
}

/// Does `type` have a modifier that matches `modifier`?
static bool _hasMatchingModifier(ModifiedType* type, Val* modifier)
{
    if (!type)
        return false;

    for (Index m = 0; m < type->getModifierCount(); m++)
    {
        if (_doModifiersMatch(type->getModifier(m), modifier))
            return true;
    }

    return false;
}

/// Can `modifier` be added to a type as part of a coercion?
///
/// For example, it is generally safe to convert from a value
/// of type `T` to a value of type `const T` in C/C++.
///
static bool _canModifierBeAddedDuringCoercion(Val* modifier)
{
    switch (modifier->astNodeType)
    {
    default:
        return false;

    case ASTNodeType::UNormModifierVal:
    case ASTNodeType::SNormModifierVal:
    case ASTNodeType::NoDiffModifierVal:
        return true;
    }
}

/// Can `modifier` be dropped from a type as part of a coercion?
///
/// For example, it is generally safe to convert from a value
/// of type `const T` to a value of type `T` in C/C++.
///
static bool _canModifierBeDroppedDuringCoercion(Val* modifier)
{
    switch (modifier->astNodeType)
    {
    default:
        return false;

    case ASTNodeType::UNormModifierVal:
    case ASTNodeType::SNormModifierVal:
    case ASTNodeType::NoDiffModifierVal:
        return true;
    }
}

static bool isSigned(Type* t)
{
    auto basicType = as<BasicExpressionType>(t);
    if (!basicType)
        return false;
    switch (basicType->getBaseType())
    {
    case BaseType::Int8:
    case BaseType::Int16:
    case BaseType::Int:
    case BaseType::Int64:
    case BaseType::IntPtr:
        return true;
    default:
        return false;
    }
}

int getTypeBitSize(Type* t)
{
    auto basicType = as<BasicExpressionType>(t);
    if (!basicType)
        return 0;

    switch (basicType->getBaseType())
    {
    case BaseType::Int8:
    case BaseType::UInt8:
        return 8;
    case BaseType::Int16:
    case BaseType::UInt16:
        return 16;
    case BaseType::Int:
    case BaseType::UInt:
        return 32;
    case BaseType::Int64:
    case BaseType::UInt64:
        return 64;
    case BaseType::IntPtr:
    case BaseType::UIntPtr:
#if SLANG_PTR_IS_32
        return 32;
#else
        return 64;
#endif
    default:
        return 0;
    }
}

ConversionCost SemanticsVisitor::getImplicitConversionCostWithKnownArg(
    DeclRef<Decl> decl,
    Type* toType,
    Expr* arg)
{
    ConversionCost candidateCost = getImplicitConversionCost(decl.getDecl());

    if (candidateCost == kConversionCost_TypeCoercionConstraint ||
        candidateCost == kConversionCost_TypeCoercionConstraintPlusScalarToVector)
    {
        if (auto genApp = as<GenericAppDeclRef>(decl.declRefBase))
        {
            for (auto genArg : genApp->getArgs())
            {
                if (auto wit = as<TypeCoercionWitness>(genArg))
                {
                    candidateCost -= kConversionCost_TypeCoercionConstraint;
                    candidateCost += getConversionCost(wit->getToType(), wit->getFromType());
                    break;
                }
            }
        }
    }

    // Fix up the cost if the operand is a const lit.
    if (isScalarIntegerType(toType))
    {
        auto knownVal = as<IntegerLiteralExpr>(arg);
        if (!knownVal)
            return candidateCost;
        if (getIntValueBitSize(knownVal->value) <= getTypeBitSize(toType))
        {
            bool toTypeIsSigned = isSigned(toType);
            bool fromTypeIsSigned = isSigned(knownVal->type);
            if (toTypeIsSigned == fromTypeIsSigned)
                candidateCost = kConversionCost_InRangeIntLitConversion;
            else if (toTypeIsSigned)
                candidateCost = kConversionCost_InRangeIntLitUnsignedToSignedConversion;
            else
                candidateCost = kConversionCost_InRangeIntLitSignedToUnsignedConversion;
        }
    }
    return candidateCost;
}

bool SemanticsVisitor::_coerce(
    CoercionSite site,
    Type* toType,
    Expr** outToExpr,
    QualType fromType,
    Expr* fromExpr,
    ConversionCost* outCost)
{
    // If we are about to try and coerce an overloaded expression,
    // then we should start by trying to resolve the ambiguous reference
    // based on prioritization of the different candidates.
    //
    // TODO: A more powerful model would be to try to coerce each
    // of the constituent overload candidates, filtering down to
    // those that are coercible, and then disambiguating the result.
    // Such an approach would let us disambiguate between overloaded
    // symbols based on their type (e.g., by casting the name of
    // an overloaded function to the type of the overload we mean
    // to reference).
    //
    if (auto fromOverloadedExpr = as<OverloadedExpr>(fromExpr))
    {
        auto resolvedExpr =
            maybeResolveOverloadedExpr(fromOverloadedExpr, LookupMask::Default, nullptr);

        fromExpr = resolvedExpr;
        fromType = resolvedExpr->type;
    }

    // An important and easy case is when the "to" and "from" types are equal.
    //
    if (toType->equals(fromType))
    {
        if (outToExpr)
            *outToExpr = fromExpr;
        if (outCost)
            *outCost = kConversionCost_None;
        return true;
    }

    // Assume string literals are convertible to any string type.
    if (as<StringLiteralExpr>(fromExpr) && as<StringTypeBase>(toType))
    {
        if (outToExpr)
            *outToExpr = fromExpr;
        if (outCost)
            *outCost = kConversionCost_None;
        return true;
    }

    // Allow implicit conversion from sized array to unsized array when
    // calling a function.
    // Note: we implement the logic here instead of an implicit_conversion
    // intrinsic in the core module because we only want to allow this conversion
    // when calling a function.
    //
    if (site == CoercionSite::Argument)
    {
        if (auto fromArrayType = as<ArrayExpressionType>(fromType))
        {
            if (auto toArrayType = as<ArrayExpressionType>(toType))
            {
                if (fromArrayType->getElementType()->equals(toArrayType->getElementType()) &&
                    toArrayType->isUnsized())
                {
                    if (outToExpr)
                        *outToExpr = fromExpr;
                    if (outCost)
                        *outCost = kConversionCost_SizedArrayToUnsizedArray;
                    return true;
                }
            }
        }
    }

    // Another important case is when either the "to" or "from" type
    // represents an error. In such a case we must have already
    // reporeted the error, so it is better to allow the conversion
    // to pass than to report a "cascading" error that might not
    // make any sense.
    //
    if (as<ErrorType>(toType) || as<ErrorType>(fromType))
    {
        if (outToExpr)
            *outToExpr = CreateImplicitCastExpr(toType, fromExpr);
        if (outCost)
            *outCost = kConversionCost_None;
        return true;
    }

    {
        // It is possible that one or more of the types involved might have modifiers
        // on it, but the underlying types are otherwise the same.
        //
        auto toModified = as<ModifiedType>(toType);
        auto toBase = toModified ? toModified->getBase() : toType;
        //
        auto fromModified = as<ModifiedType>(fromType);
        auto fromBase =
            fromModified ? QualType(fromModified->getBase(), fromType.isLeftValue) : fromType;


        if ((toModified || fromModified) && toBase->equals(fromBase))
        {
            // We need to check each modifier present on either `toType`
            // or `fromType`. For each modifier, it will either be:
            //
            // * Present on both types; these are a non-issue
            // * Present only on `toType`
            // * Present only on `fromType`
            //
            if (toModified)
            {
                for (Index m = 0; m < toModified->getModifierCount(); m++)
                {
                    auto modifier = toModified->getModifier(m);
                    if (_hasMatchingModifier(fromModified, modifier))
                        continue;

                    // If `modifier` is present on `toType`, but not `fromType`,
                    // then we need to know whether this modifier can be added
                    // to the type of an expression as part of coercion.
                    //
                    if (!_canModifierBeAddedDuringCoercion(modifier))
                    {
                        return _failedCoercion(toType, outToExpr, fromExpr);
                    }
                }
            }
            if (fromModified)
            {
                for (Index m = 0; m < fromModified->getModifierCount(); m++)
                {
                    auto modifier = fromModified->getModifier(m);

                    if (_hasMatchingModifier(toModified, modifier))
                        continue;

                    // If `modifier` is present on `fromType`, but not `toType`,
                    // then we need to know whether this modifier can be dropped
                    // to the type of an expression as part of coercion.
                    //
                    if (!_canModifierBeDroppedDuringCoercion(modifier))
                    {
                        return _failedCoercion(toType, outToExpr, fromExpr);
                    }
                }
            }

            // If all the modifiers were okay, we can convert.

            // TODO: we may need a cost to allow disambiguation of overloads based on modifiers?
            if (outCost)
            {
                *outCost = kConversionCost_None;
            }
            if (outToExpr)
            {
                *outToExpr = createModifierCastExpr(toType, fromExpr);
            }

            return true;
        }
    }

    // Coercion from an initializer list is allowed for many types,
    // so we will farm that out to its own subroutine.
    //
    if (fromExpr && as<InitializerListType>(fromExpr->type.type))
    {
        if (auto fromInitializerListExpr = as<InitializerListExpr>(fromExpr))
        {
            if (!_coerceInitializerList(toType, outToExpr, fromInitializerListExpr))
            {
                return false;
            }

            // For now, we treat coercion from an initializer list
            // as having  no cost, so that all conversions from initializer
            // lists are equally valid. This is fine given where initializer
            // lists are allowed to appear now, but might need to be made
            // more strict if we allow for initializer lists in more
            // places in the language (e.g., as function arguments).
            //
            if (outCost)
            {
                *outCost = kConversionCost_None;
            }

            return true;
        }
    }

    // nullptr_t can be cast into any pointer type.
    if (as<NullPtrType>(fromType) && as<PtrType>(toType))
    {
        if (outCost)
        {
            *outCost = kConversionCost_NullPtrToPtr;
        }
        if (outToExpr)
        {
            auto* defaultExpr = getASTBuilder()->create<DefaultConstructExpr>();
            defaultExpr->type = QualType(toType);
            *outToExpr = defaultExpr;
        }
        return true;
    }
    // none_t can be cast into any Optional<T> type.
    if (as<NoneType>(fromType) && as<OptionalType>(toType))
    {
        if (outCost)
        {
            *outCost = kConversionCost_NoneToOptional;
        }
        if (outToExpr)
        {
            auto resultExpr = getASTBuilder()->create<MakeOptionalExpr>();
            resultExpr->loc = fromExpr->loc;
            resultExpr->type = toType;
            resultExpr->checked = true;
            *outToExpr = resultExpr;
        }
        return true;
    }

    // A enum type can be converted into its underlying tag type.
    if (auto enumDecl = isEnumType(fromType))
    {
        Type* tagType = enumDecl->tagType;
        if (tagType == toType)
        {
            if (outCost)
            {
                *outCost = kConversionCost_RankPromotion;
            }
            if (outToExpr)
            {
                auto rsExpr = getASTBuilder()->create<BuiltinCastExpr>();
                rsExpr->type = toType;
                rsExpr->loc = fromExpr->loc;
                rsExpr->base = fromExpr;
                *outToExpr = rsExpr;
            }
            return true;
        }
    }

    // matrix type with different layouts are convertible
    if (auto fromMatrixType = as<MatrixExpressionType>(fromType))
    {
        if (auto toMatrixType = as<MatrixExpressionType>(toType))
        {
            if (fromMatrixType->getElementType()->equals(toMatrixType->getElementType()) &&
                fromMatrixType->getRowCount()->equals(toMatrixType->getRowCount()) &&
                fromMatrixType->getColumnCount()->equals(toMatrixType->getColumnCount()))
            {
                if (outCost)
                {
                    *outCost = kConversionCost_MatrixLayout;
                }
                if (outToExpr)
                {
                    *outToExpr = fromExpr;
                }
                return true;
            }
        }
    }

    // A type is always convertible to any of its supertypes.
    //
    if (auto witness = tryGetSubtypeWitness(fromType, toType))
    {
        if (outToExpr)
        {
            *outToExpr = createCastToSuperTypeExpr(toType, fromExpr, witness);

            // If the original expression was an l-value, then the result
            // of the cast may be an l-value itself. We want to be able
            // to invoke `[mutating]` methods on a value that is cast to
            // an interface it conforms to, and we also expect to be able
            // to pass a value of a derived `struct` type into methods that
            // expect a value of its base type.
            //
            if (fromExpr && fromExpr->type.isLeftValue)
            {
                // If the original type is a concrete type and toType is an interface type,
                // we need to wrap the original expression into a MakeExistential, and the
                // result of MakeExistential is not an l-value.
                bool toTypeIsInterface = isInterfaceType(toType);
                bool fromTypeIsInterface = isInterfaceType(fromType);
                if (!toTypeIsInterface || toTypeIsInterface == fromTypeIsInterface)
                    (*outToExpr)->type.isLeftValue = true;
            }
        }
        if (outCost)
            *outCost = kConversionCost_CastToInterface;
        return true;
    }
    else if (auto fromIsToWitness = tryGetSubtypeWitness(toType, fromType))
    {
        // Is toType and fromType the same via some type equality witness?
        // If so there is no need to do any conversion.
        //
        if (isTypeEqualityWitness(fromIsToWitness))
        {
            if (outToExpr)
            {
                *outToExpr = createCastToSuperTypeExpr(toType, fromExpr, fromIsToWitness);
            }
            if (outCost)
                *outCost = 0;
            return true;
        }
    }

    // Disallow converting to a ParameterGroupType.
    //
    // TODO(tfoley): Under what circumstances would this check ever be needed?
    //
    if (as<ParameterGroupType>(toType))
    {
        return _failedCoercion(toType, outToExpr, fromExpr);
    }

    // We allow implicit conversion of a parameter group type like
    // `ConstantBuffer<X>` or `ParameterBlock<X>` to its element
    // type `X`.
    //
    if (auto fromParameterGroupType = as<ParameterGroupType>(fromType))
    {
        auto fromElementType = fromParameterGroupType->getElementType();

        // If we convert, e.g., `ConstantBuffer<A> to `A`, we will allow
        // subsequent conversion of `A` to `B` if such a conversion
        // is possible.
        //
        ConversionCost subCost = kConversionCost_None;

        DerefExpr* derefExpr = nullptr;
        if (outToExpr)
        {
            derefExpr = m_astBuilder->create<DerefExpr>();
            derefExpr->base = fromExpr;
            derefExpr->type = QualType(fromElementType);
            derefExpr->checked = true;
        }

        if (!_coerce(site, toType, outToExpr, fromElementType, derefExpr, &subCost))
        {
            return false;
        }

        if (outCost)
            *outCost = subCost + kConversionCost_ImplicitDereference;
        return true;
    }

    if (auto refType = as<RefTypeBase>(toType))
    {
        ConversionCost cost;
        if (!canCoerce(refType->getValueType(), fromType, fromExpr, &cost))
            return false;
        if (as<RefType>(toType) && !fromExpr->type.isLeftValue)
            return false;
        ConversionCost subCost = kConversionCost_GetRef;

        MakeRefExpr* refExpr = nullptr;
        if (outToExpr)
        {
            refExpr = m_astBuilder->create<MakeRefExpr>();
            refExpr->base = fromExpr;
            refExpr->type = QualType(refType);
            refExpr->type.isLeftValue = false;
            refExpr->checked = true;
            *outToExpr = refExpr;
        }
        if (outCost)
            *outCost = subCost;
        return true;
    }


    // Allow implicit dereferencing a reference type.
    if (auto fromRefType = as<RefTypeBase>(fromType))
    {
        auto fromValueType = fromRefType->getValueType();

        // If we convert, e.g., `ConstantBuffer<A> to `A`, we will allow
        // subsequent conversion of `A` to `B` if such a conversion
        // is possible.
        //
        ConversionCost subCost = kConversionCost_None;

        Expr* openRefExpr = nullptr;
        if (outToExpr)
        {
            openRefExpr = maybeOpenRef(fromExpr);
        }

        if (!_coerce(site, toType, outToExpr, fromValueType, openRefExpr, &subCost))
        {
            return false;
        }

        if (outCost)
            *outCost = subCost + kConversionCost_ImplicitDereference;
        return true;
    }


    // The main general-purpose approach for conversion is
    // using suitable marked initializer ("constructor")
    // declarations on the target type.
    //
    // This is treated as a form of overload resolution,
    // since we are effectively forming an overloaded
    // call to one of the initializers in the target type.

    OverloadResolveContext overloadContext;
    overloadContext.disallowNestedConversions = (site != CoercionSite::ExplicitCoercion);
    overloadContext.argCount = 1;
    List<Expr*> args;
    args.add(fromExpr);
    overloadContext.argTypes = &fromType.type;
    overloadContext.args = &args;
    overloadContext.sourceScope = m_outerScope;
    overloadContext.originalExpr = nullptr;
    if (fromExpr)
    {
        overloadContext.loc = fromExpr->loc;
        overloadContext.funcLoc = fromExpr->loc;
    }

    overloadContext.baseExpr = nullptr;
    overloadContext.mode = OverloadResolveContext::Mode::JustTrying;

    // Since the lookup and resolution of all possible implicit conversions
    // can be very costly, we use a cache to store the checking results.
    ImplicitCastMethodKey implicitCastKey = ImplicitCastMethodKey(fromType, toType, fromExpr);
    ImplicitCastMethod* cachedMethod = getShared()->tryGetImplicitCastMethod(implicitCastKey);

    if (cachedMethod)
    {
        if (cachedMethod->conversionFuncOverloadCandidate.status !=
            OverloadCandidate::Status::Applicable)
        {
            return _failedCoercion(toType, outToExpr, fromExpr);
        }
        overloadContext.bestCandidateStorage = cachedMethod->conversionFuncOverloadCandidate;
        overloadContext.bestCandidate = &overloadContext.bestCandidateStorage;
        if (!outToExpr)
        {
            // If we are not requesting to create an expression, we can return early.
            if (outCost)
                *outCost = cachedMethod->cost;
            return true;
        }
        else
        {
            if (cachedMethod->isAmbiguous)
            {
                overloadContext.bestCandidate = nullptr;
                overloadContext.bestCandidates.add(cachedMethod->conversionFuncOverloadCandidate);
            }
        }
    }

    if (!overloadContext.bestCandidate)
    {
        AddTypeOverloadCandidates(toType, overloadContext);
    }

    // After all of the overload candidates have been added
    // to the context and processed, we need to see whether
    // there was one best overload or not.
    //
    if (overloadContext.bestCandidates.getCount() != 0)
    {
        // In this case there were multiple equally-good candidates to call.
        //
        // We will start by checking if the candidates are
        // even applicable, because if not, then we shouldn't
        // consider the conversion as possible.
        //
        if (overloadContext.bestCandidates[0].status != OverloadCandidate::Status::Applicable)
        {
            if (!cachedMethod)
            {
                getShared()->cacheImplicitCastMethod(implicitCastKey, ImplicitCastMethod{});
            }
            return _failedCoercion(toType, outToExpr, fromExpr);
        }

        // If all of the candidates in `bestCandidates` are applicable,
        // then we have an ambiguity.
        //
        // We will compute a nominal conversion cost as the minimum over
        // all the conversions available.
        //
        ConversionCost bestCost = kConversionCost_Explicit;
        ImplicitCastMethod method;
        for (auto candidate : overloadContext.bestCandidates)
        {
            ConversionCost candidateCost =
                getImplicitConversionCostWithKnownArg(candidate.item.declRef, toType, fromExpr);
            if (candidateCost < bestCost)
            {
                method.conversionFuncOverloadCandidate = candidate;
                bestCost = candidateCost;
            }
        }

        // Conceptually, we want to treat the conversion as
        // possible, but report it as ambiguous if we actually
        // need to reify the result as an expression.
        //
        if (outToExpr)
        {
            getSink()->diagnose(fromExpr, Diagnostics::ambiguousConversion, fromType, toType);

            *outToExpr = CreateErrorExpr(fromExpr);
        }

        if (!cachedMethod)
        {
            method.isAmbiguous = true;
            method.cost = bestCost;
            getShared()->cacheImplicitCastMethod(implicitCastKey, method);
        }

        if (outCost)
            *outCost = bestCost;
        return true;
    }
    else if (overloadContext.bestCandidate)
    {
        // If there is a single best candidate for conversion,
        // then we want to use it.
        //
        // It is possible that there was a single best candidate,
        // but it wasn't actually usable, so we will check for
        // that case first.
        //
        if (overloadContext.bestCandidate->status != OverloadCandidate::Status::Applicable)
        {
            if (!cachedMethod)
            {
                getShared()->cacheImplicitCastMethod(implicitCastKey, ImplicitCastMethod{});
            }
            return _failedCoercion(toType, outToExpr, fromExpr);
        }

        // Next, we need to look at the implicit conversion
        // cost associated with the initializer we are invoking.
        //
        ConversionCost cost = getImplicitConversionCostWithKnownArg(
            overloadContext.bestCandidate->item.declRef,
            toType,
            fromExpr);

        // If the cost is too high to be usable as an
        // implicit conversion, then we will report the
        // conversion as possible (so that an overload involving
        // this conversion will be selected over one without),
        // but then emit a diagnostic when actually reifying
        // the result expression.
        //
        if (outToExpr && site != CoercionSite::ExplicitCoercion)
        {
            if (cost >= kConversionCost_Explicit)
            {
                getSink()->diagnose(fromExpr, Diagnostics::typeMismatch, toType, fromType);
                getSink()->diagnoseWithoutSourceView(
                    fromExpr,
                    Diagnostics::noteExplicitConversionPossible,
                    fromType,
                    toType);
            }
            else if (cost >= kConversionCost_Default)
            {
                // For general types of implicit conversions, we issue a warning, unless `fromExpr`
                // is a known constant and we know it won't cause a problem.
                bool shouldEmitGeneralWarning = true;
                if (isScalarIntegerType(toType) || isHalfType(toType))
                {
                    if (auto intVal = tryFoldIntegerConstantExpression(
                            fromExpr,
                            ConstantFoldingKind::CompileTime,
                            nullptr))
                    {
                        if (auto val = as<ConstantIntVal>(intVal))
                        {
                            if (isIntValueInRangeOfType(val->getValue(), toType))
                            {
                                // OK.
                                shouldEmitGeneralWarning = false;
                            }
                        }
                    }
                }
                if (shouldEmitGeneralWarning)
                {
                    getSink()->diagnose(
                        fromExpr,
                        Diagnostics::unrecommendedImplicitConversion,
                        fromType,
                        toType);
                }
            }

            if (site == CoercionSite::Argument)
            {
                auto builtinConversionKind = getImplicitConversionBuiltinKind(
                    overloadContext.bestCandidate->item.declRef.getDecl());
                if (builtinConversionKind == kBuiltinConversion_FloatToDouble)
                {
                    if (!as<FloatingPointLiteralExpr>(fromExpr))
                        getSink()->diagnose(fromExpr, Diagnostics::implicitConversionToDouble);
                }
            }
        }
        if (fromType.isLeftValue)
        {
            // If we are implicitly casting the type of an l-value, we need to impose additional
            // cost.
            cost += kConversionCost_LValueCast;
        }
        if (outCost)
            *outCost = cost;

        if (outToExpr)
        {
            // The logic here is a bit ugly, to deal with the fact that
            // `CompleteOverloadCandidate` will, left to its own devices,
            // construct a vanilla `InvokeExpr` to represent the call
            // to the initializer we found, while we *want* it to
            // create some variety of `ImplicitCastExpr`.
            //
            // Now, it just so happens that `CompleteOverloadCandidate`
            // will use the "original" expression if one is available,
            // so we'll create one and initialize it here.
            // We fill in the location and arguments, but not the
            // base expression (the callee), since that will come
            // from the selected overload candidate.
            //
            InvokeExpr* castExpr = (site == CoercionSite::ExplicitCoercion)
                                       ? m_astBuilder->create<ExplicitCastExpr>()
                                       : createImplicitCastExpr();
            castExpr->loc = fromExpr->loc;
            castExpr->arguments.add(fromExpr);
            //
            // Next we need to set our cast expression as the "original"
            // expression and then complete the overload process.
            //
            overloadContext.originalExpr = castExpr;
            *outToExpr = CompleteOverloadCandidate(overloadContext, *overloadContext.bestCandidate);
            //
            // However, the above isn't *quite* enough, because
            // the process of completing the overload candidate
            // might overwrite the argument list that was passed
            // in to overload resolution, and in this case that
            // "argument list" was just a pointer to `fromExpr`.
            //
            // That means we need to clear the argument list and
            // reload it from `args[0]` to make sure that we
            // got the arguments *after* any transformations
            // were applied.
            // For right now this probably doesn't matter,
            // because we don't allow nested implicit conversions,
            // but I'd rather play it safe.
            //
            castExpr->arguments.clear();
            castExpr->arguments.add(args[0]);
        }
        if (!cachedMethod)
            getShared()->cacheImplicitCastMethod(
                implicitCastKey,
                ImplicitCastMethod{*overloadContext.bestCandidate, cost});
        return true;
    }
    if (!cachedMethod)
    {
        getShared()->cacheImplicitCastMethod(implicitCastKey, ImplicitCastMethod{});
    }
    return _failedCoercion(toType, outToExpr, fromExpr);
}

bool SemanticsVisitor::canCoerce(
    Type* toType,
    QualType fromType,
    Expr* fromExpr,
    ConversionCost* outCost)
{
    // As an optimization, we will maintain a cache of conversion results
    // for basic types such as scalars and vectors.
    //

    bool shouldAddToCache = false;
    ConversionCost cost;
    TypeCheckingCache* typeCheckingCache = getLinkage()->getTypeCheckingCache();

    BasicTypeKeyPair cacheKey;
    cacheKey.type1 = makeBasicTypeKey(toType);
    cacheKey.type2 = makeBasicTypeKey(fromType, fromExpr);

    if (cacheKey.isValid())
    {
        if (typeCheckingCache->conversionCostCache.tryGetValue(cacheKey, cost))
        {
            if (outCost)
                *outCost = cost;
            return cost != kConversionCost_Impossible;
        }
        else
            shouldAddToCache = true;
    }

    // If there was no suitable entry in the cache,
    // then we fall back to the general-purpose
    // conversion checking logic.
    //
    // Note that we are passing in `nullptr` as
    // the output expression to be constructed,
    // which suppresses emission of any diagnostics
    // during the coercion process.
    //
    bool rs = _coerce(CoercionSite::General, toType, nullptr, fromType, fromExpr, &cost);

    if (outCost)
        *outCost = cost;

    if (shouldAddToCache)
    {
        if (!rs)
            cost = kConversionCost_Impossible;
        typeCheckingCache->conversionCostCache[cacheKey] = cost;
    }

    return rs;
}

TypeCastExpr* SemanticsVisitor::createImplicitCastExpr()
{
    return m_astBuilder->create<ImplicitCastExpr>();
}

Expr* SemanticsVisitor::CreateImplicitCastExpr(Type* toType, Expr* fromExpr)
{
    TypeCastExpr* castExpr = createImplicitCastExpr();

    auto typeType = m_astBuilder->getTypeType(toType);

    auto typeExpr = m_astBuilder->create<SharedTypeExpr>();
    typeExpr->type.type = typeType;
    typeExpr->base.type = toType;

    castExpr->loc = fromExpr->loc;
    castExpr->functionExpr = typeExpr;
    castExpr->type = QualType(toType);
    castExpr->arguments.add(fromExpr);
    return castExpr;
}

Expr* SemanticsVisitor::createCastToSuperTypeExpr(Type* toType, Expr* fromExpr, Val* witness)
{
    CastToSuperTypeExpr* expr = m_astBuilder->create<CastToSuperTypeExpr>();
    expr->loc = fromExpr->loc;
    expr->type = QualType(toType);
    expr->valueArg = fromExpr;
    expr->witnessArg = witness;
    return expr;
}

Expr* SemanticsVisitor::createModifierCastExpr(Type* toType, Expr* fromExpr)
{
    ModifierCastExpr* expr = m_astBuilder->create<ModifierCastExpr>();
    expr->loc = fromExpr->loc;
    expr->type = QualType(toType);
    expr->valueArg = fromExpr;
    return expr;
}


Expr* SemanticsVisitor::coerce(CoercionSite site, Type* toType, Expr* fromExpr)
{
    Expr* expr = nullptr;
    if (!_coerce(site, toType, &expr, fromExpr->type, fromExpr, nullptr))
    {
        // Note(tfoley): We don't call `CreateErrorExpr` here, because that would
        // clobber the type on `fromExpr`, and an invariant here is that coercion
        // really shouldn't *change* the expression that is passed in, but should
        // introduce new AST nodes to coerce its value to a different type...
        return CreateImplicitCastExpr(m_astBuilder->getErrorType(), fromExpr);
    }

    return expr;
}

bool SemanticsVisitor::canConvertImplicitly(ConversionCost conversionCost)
{
    // Is the conversion cheap enough to be done implicitly?
    if (conversionCost >= kConversionCost_GeneralConversion)
        return false;
    return true;
}

bool SemanticsVisitor::canConvertImplicitly(Type* toType, QualType fromType)
{
    auto conversionCost = getConversionCost(toType, fromType);

    // Is the conversion cheap enough to be done implicitly?
    if (canConvertImplicitly(conversionCost))
        return false;

    return true;
}

ConversionCost SemanticsVisitor::getConversionCost(Type* toType, QualType fromType)
{
    ConversionCost conversionCost = kConversionCost_Impossible;
    if (!canCoerce(toType, fromType, nullptr, &conversionCost))
        return kConversionCost_Impossible;
    return conversionCost;
}
} // namespace Slang
