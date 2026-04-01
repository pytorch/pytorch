// slang-ast-print.cpp
#include "slang-ast-print.h"

#include "core/slang-char-util.h"
#include "slang-check-impl.h"

namespace Slang
{

ASTPrinter::Part::Kind ASTPrinter::Part::getKind(ASTPrinter::Part::Type type)
{
    typedef ASTPrinter::Part::Kind Kind;
    typedef ASTPrinter::Part::Type Type;

    switch (type)
    {
    case Type::ParamType:
        return Kind::Type;
    case Type::ParamName:
        return Kind::Name;
    case Type::ReturnType:
        return Kind::Type;
    case Type::DeclPath:
        return Kind::Name;
    case Type::GenericParamType:
        return Kind::Type;
    case Type::GenericParamValue:
        return Kind::Value;
    case Type::GenericParamValueType:
        return Kind::Type;
    default:
        break;
    }
    return Kind::None;
}

void ASTPrinter::addType(Type* type)
{
    if (!type)
    {
        m_builder << "<error>";
        return;
    }
    type = type->getCanonicalType();
    if (m_optionFlags & OptionFlag::SimplifiedBuiltinType)
    {
        if (auto vectorType = as<VectorExpressionType>(type))
        {
            if (as<BasicExpressionType>(vectorType->getElementType()))
            {
                vectorType->getElementType()->toText(m_builder);
                if (as<ConstantIntVal>(vectorType->getElementCount()))
                {
                    m_builder << vectorType->getElementCount();
                    return;
                }
            }
        }
        else if (auto matrixType = as<MatrixExpressionType>(type))
        {
            auto elementType = matrixType->getElementType();
            if (as<BasicExpressionType>(elementType))
            {
                matrixType->getElementType()->toText(m_builder);
                if (as<ConstantIntVal>(matrixType->getRowCount()) &&
                    as<ConstantIntVal>(matrixType->getColumnCount()))
                {
                    m_builder << matrixType->getRowCount() << "x" << matrixType->getColumnCount();
                    return;
                }
            }
        }
    }
    type->toText(m_builder);
}

void ASTPrinter::addExpr(Expr* expr)
{
    if (!expr)
    {
        m_builder << "<error>";
        return;
    }

    auto& sb = m_builder;

    if (const auto incompleteExpr = as<IncompleteExpr>(expr))
    {
        sb << "<incomplete>";
    }
    else if (const auto varExpr = as<VarExpr>(expr))
    {
        if (varExpr->declRef)
        {
            addDeclPath(varExpr->declRef);
        }
        else if (varExpr->name)
        {
            sb << varExpr->name->text;
        }
        else
        {
            sb << "<unknown-var>";
        }
    }
    else if (const auto memberExpr = as<MemberExpr>(expr))
    {
        if (memberExpr->baseExpression)
        {
            addExpr(memberExpr->baseExpression);
            sb << ".";
        }

        if (memberExpr->declRef)
        {
            _addDeclName(memberExpr->declRef.getDecl());
        }
        else if (memberExpr->name)
        {
            sb << memberExpr->name->text;
        }
        else
        {
            sb << "<unknown-member>";
        }
    }
    else if (const auto staticMemberExpr = as<StaticMemberExpr>(expr))
    {
        if (staticMemberExpr->baseExpression)
        {
            addExpr(staticMemberExpr->baseExpression);
            sb << "::";
        }

        if (staticMemberExpr->declRef)
        {
            _addDeclName(staticMemberExpr->declRef.getDecl());
        }
        else if (staticMemberExpr->name)
        {
            sb << staticMemberExpr->name->text;
        }
        else
        {
            sb << "<unknown-static-member>";
        }
    }
    else if (const auto derefMemberExpr = as<DerefMemberExpr>(expr))
    {
        if (derefMemberExpr->baseExpression)
        {
            addExpr(derefMemberExpr->baseExpression);
            sb << "->";
        }

        if (derefMemberExpr->declRef)
        {
            _addDeclName(derefMemberExpr->declRef.getDecl());
        }
        else if (derefMemberExpr->name)
        {
            sb << derefMemberExpr->name->text;
        }
        else
        {
            sb << "<unknown-deref-member>";
        }
    }
    else if (const auto intLit = as<IntegerLiteralExpr>(expr))
    {
        sb << intLit->value;
        // Handle suffix types without using getBaseTypeName
        switch (intLit->suffixType)
        {
        case BaseType::Int:
            // No suffix for default int
            break;
        case BaseType::UInt:
            sb << "u";
            break;
        case BaseType::Int64:
            sb << "l";
            break;
        case BaseType::UInt64:
            sb << "ul";
            break;
        case BaseType::Int16:
            sb << "s";
            break;
        case BaseType::UInt16:
            sb << "us";
            break;
        case BaseType::Int8:
            sb << "b";
            break;
        case BaseType::UInt8:
            sb << "ub";
            break;
        default:
            // Don't add a suffix for other types
            break;
        }
    }
    else if (const auto floatLit = as<FloatingPointLiteralExpr>(expr))
    {
        sb << floatLit->value;
        // Handle suffix types without using getBaseTypeName
        switch (floatLit->suffixType)
        {
        case BaseType::Float:
            sb << "f";
            break;
        case BaseType::Double:
            // No suffix for default double
            break;
        case BaseType::Half:
            sb << "h";
            break;
        default:
            // Don't add a suffix for other types
            break;
        }
    }
    else if (const auto boolLit = as<BoolLiteralExpr>(expr))
    {
        sb << (boolLit->value ? "true" : "false");
    }
    else if (as<NullPtrLiteralExpr>(expr))
    {
        sb << "nullptr";
    }
    else if (as<NoneLiteralExpr>(expr))
    {
        sb << "none";
    }
    else if (const auto stringLit = as<StringLiteralExpr>(expr))
    {
        sb << "\"" << stringLit->value << "\"";
    }
    else if (const auto initList = as<InitializerListExpr>(expr))
    {
        sb << "{";
        bool first = true;
        for (auto arg : initList->args)
        {
            if (!first)
                sb << ", ";
            addExpr(arg);
            first = false;
        }
        sb << "}";
    }
    else if (const auto arrayLengthExpr = as<GetArrayLengthExpr>(expr))
    {
        if (arrayLengthExpr->arrayExpr)
        {
            addExpr(arrayLengthExpr->arrayExpr);
            sb << ".Length";
        }
        else
        {
            sb << "<unknown-array-length>";
        }
    }
    else if (const auto expandExpr = as<ExpandExpr>(expr))
    {
        sb << "...";
        if (expandExpr->baseExpr)
        {
            addExpr(expandExpr->baseExpr);
        }
    }
    else if (const auto eachExpr = as<EachExpr>(expr))
    {
        sb << "each ";
        if (eachExpr->baseExpr)
        {
            addExpr(eachExpr->baseExpr);
        }
    }
    else if (const auto aggTypeCtorExpr = as<AggTypeCtorExpr>(expr))
    {
        addType(aggTypeCtorExpr->base.type);
        sb << "(";
        bool first = true;
        for (auto arg : aggTypeCtorExpr->arguments)
        {
            if (!first)
                sb << ", ";
            addExpr(arg);
            first = false;
        }
        sb << ")";
    }
    else if (const auto invokeExpr = as<InvokeExpr>(expr))
    {
        if (const auto operatorExpr = as<OperatorExpr>(invokeExpr))
        {
            if (const auto infixExpr = as<InfixExpr>(operatorExpr))
            {
                // Binary operator
                if (invokeExpr->arguments.getCount() == 2)
                {
                    sb << "(";
                    addExpr(invokeExpr->arguments[0]);
                    if (operatorExpr->functionExpr && as<VarExpr>(operatorExpr->functionExpr) &&
                        as<VarExpr>(operatorExpr->functionExpr)->name)
                    {
                        sb << " " << as<VarExpr>(operatorExpr->functionExpr)->name->text << " ";
                    }
                    else
                    {
                        sb << " <op> ";
                    }
                    addExpr(invokeExpr->arguments[1]);
                    sb << ")";
                    return;
                }
            }
            else if (const auto prefixExpr = as<PrefixExpr>(operatorExpr))
            {
                // Prefix operator
                if (operatorExpr->functionExpr && as<VarExpr>(operatorExpr->functionExpr) &&
                    as<VarExpr>(operatorExpr->functionExpr)->name)
                {
                    sb << as<VarExpr>(operatorExpr->functionExpr)->name->text;
                }
                else
                {
                    sb << "<op>";
                }

                if (invokeExpr->arguments.getCount() > 0)
                {
                    addExpr(invokeExpr->arguments[0]);
                }
                return;
            }
            else if (const auto postfixExpr = as<PostfixExpr>(operatorExpr))
            {
                // Postfix operator
                if (invokeExpr->arguments.getCount() > 0)
                {
                    addExpr(invokeExpr->arguments[0]);
                }

                if (operatorExpr->functionExpr && as<VarExpr>(operatorExpr->functionExpr) &&
                    as<VarExpr>(operatorExpr->functionExpr)->name)
                {
                    sb << as<VarExpr>(operatorExpr->functionExpr)->name->text;
                }
                else
                {
                    sb << "<op>";
                }
                return;
            }
            else if (const auto selectExpr = as<SelectExpr>(operatorExpr))
            {
                // Ternary operator: cond ? ifTrue : ifFalse
                if (invokeExpr->arguments.getCount() == 3)
                {
                    addExpr(invokeExpr->arguments[0]);
                    sb << " ? ";
                    addExpr(invokeExpr->arguments[1]);
                    sb << " : ";
                    addExpr(invokeExpr->arguments[2]);
                    return;
                }
            }
            else if (const auto logicExpr = as<LogicOperatorShortCircuitExpr>(operatorExpr))
            {
                // Logical operators with short-circuit behavior
                if (invokeExpr->arguments.getCount() == 2)
                {
                    addExpr(invokeExpr->arguments[0]);
                    sb
                        << (logicExpr->flavor == LogicOperatorShortCircuitExpr::And ? " && "
                                                                                    : " || ");
                    addExpr(invokeExpr->arguments[1]);
                    return;
                }
            }
        }

        // Regular function call
        if (invokeExpr->functionExpr)
        {
            addExpr(invokeExpr->functionExpr);
        }
        else
        {
            sb << "<unknown-func>";
        }

        sb << "(";
        bool first = true;
        for (auto arg : invokeExpr->arguments)
        {
            if (!first)
                sb << ", ";
            addExpr(arg);
            first = false;
        }
        sb << ")";
    }
    else if (const auto indexExpr = as<IndexExpr>(expr))
    {
        if (indexExpr->baseExpression)
        {
            addExpr(indexExpr->baseExpression);
        }

        sb << "[";
        bool first = true;
        for (auto i : indexExpr->indexExprs)
        {
            if (!first)
                sb << ", ";
            addExpr(i);
            first = false;
        }
        sb << "]";
    }
    else if (const auto swizzleExpr = as<SwizzleExpr>(expr))
    {
        if (swizzleExpr->base)
        {
            addExpr(swizzleExpr->base);
        }

        sb << ".";

        // Print swizzle components (like .xyzw or .rgba)
        static const char* xyzwComponents = "xyzw";

        // Choose component naming based on type if possible
        const char* components = xyzwComponents;

        for (auto index : swizzleExpr->elementIndices)
        {
            if (index < 4)
            {
                sb << components[index];
            }
            else
            {
                sb << "?";
            }
        }
    }
    else if (const auto matrixSwizzleExpr = as<MatrixSwizzleExpr>(expr))
    {
        if (matrixSwizzleExpr->base)
        {
            addExpr(matrixSwizzleExpr->base);
        }

        sb << ".";

        // Print matrix swizzle components (like _m00, _m01, etc.)
        for (int i = 0; i < matrixSwizzleExpr->elementCount; ++i)
        {
            if (i > 0)
                sb << "";
            sb << "_m" << matrixSwizzleExpr->elementCoords[i].row
               << matrixSwizzleExpr->elementCoords[i].col;
        }
    }
    else if (const auto makeRefExpr = as<MakeRefExpr>(expr))
    {
        sb << "&";
        if (makeRefExpr->base)
        {
            addExpr(makeRefExpr->base);
        }
    }
    else if (const auto derefExpr = as<DerefExpr>(expr))
    {
        sb << "*";
        if (derefExpr->base)
        {
            addExpr(derefExpr->base);
        }
    }
    else if (const auto typeCastExpr = as<TypeCastExpr>(expr))
    {
        if (as<ExplicitCastExpr>(typeCastExpr))
        {
            sb << "(";
            addType(expr->type);
            sb << ")";

            if (typeCastExpr->arguments.getCount() > 0)
            {
                addExpr(typeCastExpr->arguments[0]);
            }
        }
        else if (
            as<ImplicitCastExpr>(typeCastExpr) || as<LValueImplicitCastExpr>(typeCastExpr) ||
            as<OutImplicitCastExpr>(typeCastExpr) || as<InOutImplicitCastExpr>(typeCastExpr))
        {
            // For implicit casts, just print the inner expression
            if (typeCastExpr->arguments.getCount() > 0)
            {
                addExpr(typeCastExpr->arguments[0]);
            }
            else
            {
                sb << "<implicit-cast>";
            }
        }
    }
    else if (const auto builtinCastExpr = as<BuiltinCastExpr>(expr))
    {
        if (builtinCastExpr->base)
        {
            addExpr(builtinCastExpr->base);
        }
        else
        {
            sb << "<builtin-cast>";
        }
    }
    else if (const auto castToSuperTypeExpr = as<CastToSuperTypeExpr>(expr))
    {
        sb << "((";
        addType(expr->type);
        sb << ")";

        if (castToSuperTypeExpr->valueArg)
        {
            addExpr(castToSuperTypeExpr->valueArg);
        }

        sb << ")";
    }
    else if (const auto isTypeExpr = as<IsTypeExpr>(expr))
    {
        if (isTypeExpr->value)
        {
            addExpr(isTypeExpr->value);
        }

        sb << " is ";

        if (isTypeExpr->typeExpr.type)
        {
            addType(isTypeExpr->typeExpr.type);
        }
        else
        {
            sb << "<unknown-type>";
        }
    }
    else if (const auto asTypeExpr = as<AsTypeExpr>(expr))
    {
        if (asTypeExpr->value)
        {
            addExpr(asTypeExpr->value);
        }

        sb << " as ";

        if (asTypeExpr->typeExpr)
        {
            addExpr(asTypeExpr->typeExpr);
        }
        else
        {
            sb << "<unknown-type>";
        }
    }
    else if (const auto sizeOfExpr = as<SizeOfExpr>(expr))
    {
        sb << "sizeof(";
        if (sizeOfExpr->sizedType)
        {
            addType(sizeOfExpr->sizedType);
        }
        else if (sizeOfExpr->value)
        {
            addExpr(sizeOfExpr->value);
        }
        sb << ")";
    }
    else if (const auto alignOfExpr = as<AlignOfExpr>(expr))
    {
        sb << "alignof(";
        if (alignOfExpr->sizedType)
        {
            addType(alignOfExpr->sizedType);
        }
        else if (alignOfExpr->value)
        {
            addExpr(alignOfExpr->value);
        }
        sb << ")";
    }
    else if (const auto countOfExpr = as<CountOfExpr>(expr))
    {
        sb << "countof(";
        if (countOfExpr->sizedType)
        {
            addType(countOfExpr->sizedType);
        }
        else if (countOfExpr->value)
        {
            addExpr(countOfExpr->value);
        }
        sb << ")";
    }
    else if (const auto makeOptionalExpr = as<MakeOptionalExpr>(expr))
    {
        if (makeOptionalExpr->value)
        {
            sb << "Optional(";
            addExpr(makeOptionalExpr->value);
            sb << ")";
        }
        else
        {
            sb << "Optional<";
            if (makeOptionalExpr->typeExpr)
            {
                addExpr(makeOptionalExpr->typeExpr);
            }
            else
            {
                addType(expr->type);
            }
            sb << ">.none";
        }
    }
    else if (const auto modifierCastExpr = as<ModifierCastExpr>(expr))
    {
        sb << "(";
        addType(expr->type);
        sb << ")";

        if (modifierCastExpr->valueArg)
        {
            addExpr(modifierCastExpr->valueArg);
        }
    }
    else if (const auto assignExpr = as<AssignExpr>(expr))
    {
        if (assignExpr->left)
        {
            addExpr(assignExpr->left);
        }

        sb << " = ";

        if (assignExpr->right)
        {
            addExpr(assignExpr->right);
        }
    }
    else if (const auto parenExpr = as<ParenExpr>(expr))
    {
        sb << "(";
        if (parenExpr->base)
        {
            addExpr(parenExpr->base);
        }
        sb << ")";
    }
    else if (as<ThisExpr>(expr))
    {
        sb << "this";
    }
    else if (as<ReturnValExpr>(expr))
    {
        sb << "__return_val";
    }
    else if (const auto letExpr = as<LetExpr>(expr))
    {
        sb << "let ";
        if (letExpr->decl)
        {
            _addDeclName(letExpr->decl);

            if (letExpr->decl->type.type)
            {
                sb << " : ";
                addType(letExpr->decl->type.type);
            }

            if (letExpr->decl->initExpr)
            {
                sb << " = ";
                addExpr(letExpr->decl->initExpr);
            }
        }

        if (letExpr->body)
        {
            sb << " in ";
            addExpr(letExpr->body);
        }
    }
    else if (const auto extractExistentialValueExpr = as<ExtractExistentialValueExpr>(expr))
    {
        if (extractExistentialValueExpr->declRef)
        {
            addDeclPath(extractExistentialValueExpr->declRef);
        }
        else if (extractExistentialValueExpr->originalExpr)
        {
            addExpr(extractExistentialValueExpr->originalExpr);
        }
        else
        {
            sb << "<extract-existential>";
        }
    }
    else if (const auto openRefExpr = as<OpenRefExpr>(expr))
    {
        sb << "open(";
        if (openRefExpr->innerExpr)
        {
            addExpr(openRefExpr->innerExpr);
        }
        sb << ")";
    }
    else if (const auto detachExpr = as<DetachExpr>(expr))
    {
        sb << "detach(";
        if (detachExpr->inner)
        {
            addExpr(detachExpr->inner);
        }
        sb << ")";
    }
    else if (const auto higherOrderInvokeExpr = as<HigherOrderInvokeExpr>(expr))
    {
        if (const auto primalSubstituteExpr = as<PrimalSubstituteExpr>(higherOrderInvokeExpr))
        {
            sb << "__primal(";
        }
        else if (const auto forwardDiffExpr = as<ForwardDifferentiateExpr>(higherOrderInvokeExpr))
        {
            sb << "__fwd_diff(";
        }
        else if (const auto backwardDiffExpr = as<BackwardDifferentiateExpr>(higherOrderInvokeExpr))
        {
            sb << "__bwd_diff(";
        }
        else if (const auto dispatchKernelExpr = as<DispatchKernelExpr>(higherOrderInvokeExpr))
        {
            sb << "__dispatch_kernel(";
        }
        else
        {
            sb << "<higher-order>(";
        }

        if (higherOrderInvokeExpr->baseFunction)
        {
            addExpr(higherOrderInvokeExpr->baseFunction);
        }

        // Add additional parameters for specific higher-order expressions
        if (const auto dispatchKernelExpr = as<DispatchKernelExpr>(higherOrderInvokeExpr))
        {
            sb << ", ";
            if (dispatchKernelExpr->threadGroupSize)
            {
                addExpr(dispatchKernelExpr->threadGroupSize);
            }
            else
            {
                sb << "<unknown-thread-group-size>";
            }

            sb << ", ";

            if (dispatchKernelExpr->dispatchSize)
            {
                addExpr(dispatchKernelExpr->dispatchSize);
            }
            else
            {
                sb << "<unknown-dispatch-size>";
            }
        }

        sb << ")";
    }
    else if (const auto treatAsDiffExpr = as<TreatAsDifferentiableExpr>(expr))
    {
        if (treatAsDiffExpr->flavor == TreatAsDifferentiableExpr::NoDiff)
        {
            sb << "no_diff(";
        }
        else
        {
            sb << "differentiable(";
        }

        if (treatAsDiffExpr->innerExpr)
        {
            addExpr(treatAsDiffExpr->innerExpr);
        }

        sb << ")";
    }
    else if (as<ThisTypeExpr>(expr))
    {
        sb << "This";
    }
    else if (const auto andTypeExpr = as<AndTypeExpr>(expr))
    {
        if (andTypeExpr->left.type)
        {
            addType(andTypeExpr->left.type);
        }
        else
        {
            sb << "<unknown-type>";
        }

        sb << " & ";

        if (andTypeExpr->right.type)
        {
            addType(andTypeExpr->right.type);
        }
        else
        {
            sb << "<unknown-type>";
        }
    }
    else if (const auto modifiedTypeExpr = as<ModifiedTypeExpr>(expr))
    {
        // Print modifiers
        for (auto modifier : modifiedTypeExpr->modifiers)
        {
            if (modifier->getKeywordName())
            {
                sb << modifier->getKeywordName()->text << " ";
            }
        }

        if (modifiedTypeExpr->base.type)
        {
            addType(modifiedTypeExpr->base.type);
        }
        else
        {
            sb << "<unknown-type>";
        }
    }
    else if (const auto pointerTypeExpr = as<PointerTypeExpr>(expr))
    {
        if (pointerTypeExpr->base.type)
        {
            addType(pointerTypeExpr->base.type);
        }
        else
        {
            sb << "<unknown-type>";
        }

        sb << "*";
    }
    else if (const auto funcTypeExpr = as<FuncTypeExpr>(expr))
    {
        sb << "(";
        bool first = true;
        for (auto& param : funcTypeExpr->parameters)
        {
            if (!first)
                sb << ", ";

            if (param.type)
            {
                addType(param.type);
            }
            else
            {
                sb << "<unknown-type>";
            }

            first = false;
        }
        sb << ") -> ";

        if (funcTypeExpr->result.type)
        {
            addType(funcTypeExpr->result.type);
        }
        else
        {
            sb << "<unknown-type>";
        }
    }
    else if (const auto tupleTypeExpr = as<TupleTypeExpr>(expr))
    {
        sb << "(";
        bool first = true;
        for (auto& member : tupleTypeExpr->members)
        {
            if (!first)
                sb << ", ";

            if (member.type)
            {
                addType(member.type);
            }
            else
            {
                sb << "<unknown-type>";
            }

            first = false;
        }
        sb << ")";
    }
    else if (const auto partiallyAppliedGenericExpr = as<PartiallyAppliedGenericExpr>(expr))
    {
        if (partiallyAppliedGenericExpr->baseGenericDeclRef)
        {
            addDeclPath(partiallyAppliedGenericExpr->baseGenericDeclRef);
        }
        else if (partiallyAppliedGenericExpr->originalExpr)
        {
            addExpr(partiallyAppliedGenericExpr->originalExpr);
        }
        else
        {
            sb << "<partially-applied-generic>";
        }

        sb << "<";
        bool first = true;
        for (auto arg : partiallyAppliedGenericExpr->knownGenericArgs)
        {
            if (!first)
                sb << ", ";

            addVal(arg);

            first = false;
        }
        sb << ", ...>";
    }
    else if (const auto packExpr = as<PackExpr>(expr))
    {
        sb << "(";
        bool first = true;
        for (auto arg : packExpr->args)
        {
            if (!first)
                sb << ", ";

            addExpr(arg);

            first = false;
        }
        sb << ")";
    }
    else if (const auto spirvAsmExpr = as<SPIRVAsmExpr>(expr))
    {
        sb << "spirv_asm {";
        bool first = true;
        for (auto& inst : spirvAsmExpr->insts)
        {
            if (!first)
                sb << "\n  ";
            else
                sb << " ";

            // Print opcode
            if (inst.opcode.flavor == SPIRVAsmOperand::NamedValue &&
                inst.opcode.token.type != TokenType::Unknown)
            {
                sb << inst.opcode.token.getContent();
            }
            else
            {
                sb << "<unknown-opcode>";
            }

            // Print operands
            for (auto& operand : inst.operands)
            {
                sb << " ";

                switch (operand.flavor)
                {
                case SPIRVAsmOperand::Literal:
                    if (operand.token.type != TokenType::Unknown)
                        sb << operand.token.getContent();
                    else
                        sb << operand.knownValue;
                    break;

                case SPIRVAsmOperand::Id:
                    sb << "%" << operand.token.getContent();
                    break;

                case SPIRVAsmOperand::ResultMarker:
                    sb << "result";
                    break;

                case SPIRVAsmOperand::NamedValue:
                    sb << operand.token.getContent();
                    break;

                case SPIRVAsmOperand::SlangValue:
                case SPIRVAsmOperand::SlangValueAddr:
                case SPIRVAsmOperand::SlangImmediateValue:
                    if (operand.expr)
                        addExpr(operand.expr);
                    else
                        sb << "<unknown-slang-value>";
                    break;

                case SPIRVAsmOperand::SlangType:
                case SPIRVAsmOperand::SampledType:
                case SPIRVAsmOperand::ImageType:
                case SPIRVAsmOperand::SampledImageType:
                    if (operand.type.type)
                        addType(operand.type.type);
                    else
                        sb << "<unknown-type>";
                    break;

                default:
                    sb << "<unknown-operand>";
                    break;
                }
            }

            first = false;
        }
        sb << " }";
    }
    else if (const auto genericAppExpr = as<GenericAppExpr>(expr))
    {
        if (genericAppExpr->functionExpr)
        {
            addExpr(genericAppExpr->functionExpr);
        }
        else
        {
            sb << "<unknown-generic>";
        }

        sb << "<";
        bool first = true;
        for (auto arg : genericAppExpr->arguments)
        {
            if (!first)
                sb << ", ";
            addExpr(arg);
            first = false;
        }
        sb << ">";
    }
    else if (const auto tryExpr = as<TryExpr>(expr))
    {
        switch (tryExpr->tryClauseType)
        {
        case TryClauseType::Standard:
            sb << "try ";
            break;
        case TryClauseType::Optional:
            sb << "try? ";
            break;
        case TryClauseType::Assert:
            sb << "try! ";
            break;
        default:
            break;
        }

        if (tryExpr->base)
        {
            addExpr(tryExpr->base);
        }
    }
    else if (const auto defaultConstructExpr = as<DefaultConstructExpr>(expr))
    {
        sb << "default(";
        addType(expr->type);
        sb << ")";
    }
    else if (const auto overloadedExpr = as<OverloadedExpr>(expr))
    {
        if (overloadedExpr->base)
        {
            addExpr(overloadedExpr->base);
            sb << ".";
        }

        if (overloadedExpr->name)
        {
            sb << overloadedExpr->name->text;
        }
        else
        {
            sb << "<overloaded>";
        }
    }
    else if (const auto overloadedExpr2 = as<OverloadedExpr2>(expr))
    {
        if (overloadedExpr2->base)
        {
            addExpr(overloadedExpr2->base);
            sb << ".";
        }

        sb << "<overloaded>";
    }
    else
    {
        // For any other expression types
        sb << "<expr>";
    }
}

void ASTPrinter::addVal(Val* val)
{
    val->toText(m_builder);
}

/* static */ void ASTPrinter::appendDeclName(Decl* decl, StringBuilder& out)
{
    if (as<ConstructorDecl>(decl))
    {
        out << "init";
    }
    else if (as<SubscriptDecl>(decl))
    {
        out << "subscript";
    }
    else
    {
        auto text = getText(decl->getName());
        if (text.getLength() && !(CharUtil::isAlphaOrDigit(text[0]) || text[0] == '_'))
            out << "operator" << text;
        else
            out << text;
    }
}

void ASTPrinter::_addDeclName(Decl* decl)
{
    appendDeclName(decl, m_builder);
}

void ASTPrinter::addOverridableDeclPath(const DeclRef<Decl>& declRef)
{
    ScopePart scopePart(this, Part::Type::DeclPath);
    _addDeclPathRec(declRef, 0);
}

void ASTPrinter::addDeclPath(const DeclRef<Decl>& declRef)
{
    ScopePart scopePart(this, Part::Type::DeclPath);
    _addDeclPathRec(declRef, 1);
}

void ASTPrinter::_addDeclPathRec(const DeclRef<Decl>& declRef, Index depth)
{
    auto& sb = m_builder;

    // Find the parent declaration
    auto parentDeclRef = declRef.getParent();

    // If the immediate parent is a generic, then we probably
    // want the declaration above that...
    auto parentGenericDeclRef = parentDeclRef.as<GenericDecl>();
    if (parentGenericDeclRef)
    {
        parentDeclRef = parentGenericDeclRef.getParent();
    }

    // Depending on what the parent is, we may want to format things specially
    if (auto aggTypeDeclRef = parentDeclRef.as<AggTypeDecl>())
    {
        _addDeclPathRec(aggTypeDeclRef, depth + 1);
        sb << toSlice(".");
    }
    else if (auto namespaceDeclRef = parentDeclRef.as<NamespaceDecl>())
    {
        _addDeclPathRec(namespaceDeclRef, depth + 1);
        // Hmm, it could be argued that we follow the . as seen in AggType as is followed in some
        // other languages like Java. That it is useful to have a distinction between something that
        // is a member/method and something that is in a scope (such as a namespace), and is
        // something that has returned to later languages probably for that reason (Slang accepts .
        // or ::). So for now this is follows the :: convention.
        //
        // It could be argued them that the previous '.' use should vary depending on that
        // distinction.

        sb << toSlice("::");
    }
    else if (auto extensionDeclRef = parentDeclRef.as<ExtensionDecl>())
    {
        ExtensionDecl* extensionDecl = as<ExtensionDecl>(parentDeclRef.getDecl());
        Type* type = extensionDecl->targetType.type;
        if (m_optionFlags & OptionFlag::NoSpecializedExtensionTypeName)
        {
            if (auto unspecializedDeclRef = isDeclRefTypeOf<Decl>(type))
            {
                type = DeclRefType::create(
                    m_astBuilder,
                    unspecializedDeclRef.getDecl()->getDefaultDeclRef());
            }
        }
        addType(type);
        sb << toSlice(".");
    }
    else if (auto moduleDecl = as<ModuleDecl>(parentDeclRef.getDecl()))
    {
        Name* moduleName = moduleDecl->getName();
        if ((m_optionFlags & OptionFlag::ModuleName) && moduleName)
        {
            // Use to say in modules scope
            sb << moduleName->text << toSlice("::");
        }
    }

    // If this decl is the module, we only output it's name if that feature is enabled
    if (ModuleDecl* moduleDecl = as<ModuleDecl>(declRef.getDecl()))
    {
        Name* moduleName = moduleDecl->getName();
        if ((m_optionFlags & OptionFlag::ModuleName) && moduleName)
        {
            sb << moduleName->text;
        }
        return;
    }

    _addDeclName(declRef.getDecl());

    // If the parent declaration is a generic, then we need to print out its
    // signature
    if (parentGenericDeclRef && !declRef.as<GenericValueParamDecl>() &&
        !declRef.as<GenericTypeParamDeclBase>())
    {
        auto substArgs =
            tryGetGenericArguments(SubstitutionSet(declRef), parentGenericDeclRef.getDecl());
        if (substArgs.getCount())
        {
            // If the name we printed previously was an operator
            // that ends with `<`, then immediately printing the
            // generic arguments inside `<...>` may cause it to
            // be hard to parse the operator name visually.
            //
            // We thus include a space between the declaration name
            // and its generic arguments in this case.
            //
            if (sb.endsWith("<"))
            {
                sb << " ";
            }

            sb << "<";
            bool first = true;
            for (auto arg : substArgs)
            {
                // When printing the representation of a specialized
                // generic declaration we don't want to include the
                // argument values for subtype witnesses since these
                // do not correspond to parameters of the generic
                // as the user sees it.
                //
                if (as<Witness>(arg))
                    continue;

                if (!first)
                    sb << ", ";
                addVal(arg);
                first = false;
            }
            sb << ">";
        }
        else if (depth > 0)
        {
            // Write out the generic parameters (only if the depth allows it)
            addGenericParams(parentGenericDeclRef);
        }
    }
}

void ASTPrinter::addGenericParams(const DeclRef<GenericDecl>& genericDeclRef)
{
    auto& sb = m_builder;

    sb << "<";
    bool first = true;
    for (auto paramDeclRef : getMembers(m_astBuilder, genericDeclRef))
    {
        if (auto genericTypeParam = paramDeclRef.as<GenericTypeParamDecl>())
        {
            if (!first)
                sb << ", ";
            first = false;

            {
                ScopePart scopePart(this, Part::Type::GenericParamType);
                sb << getText(genericTypeParam.getName());
            }
        }
        else if (auto genericValParam = paramDeclRef.as<GenericValueParamDecl>())
        {
            if (!first)
                sb << ", ";
            first = false;

            {
                ScopePart scopePart(this, Part::Type::GenericParamValue);
                sb << getText(genericValParam.getName());
            }

            sb << ":";

            {
                ScopePart scopePart(this, Part::Type::GenericParamValueType);
                addType(getType(m_astBuilder, genericValParam));
            }
        }
        else if (auto genericTypePackParam = paramDeclRef.as<GenericTypePackParamDecl>())
        {
            if (!first)
                sb << ", ";
            first = false;
            {
                ScopePart scopePart(this, Part::Type::GenericParamType);
                sb << "each ";
                sb << getText(genericTypePackParam.getName());
            }
        }
        else
        {
        }
    }
    sb << ">";
}

void ASTPrinter::addDeclParams(const DeclRef<Decl>& declRef, List<Range<Index>>* outParamRange)
{
    auto& sb = m_builder;

    if (auto funcDeclRef = declRef.as<CallableDecl>())
    {
        // This is something callable, so we need to also print parameter types for overloading
        sb << "(";

        bool first = true;
        for (auto paramDeclRef : getParameters(m_astBuilder, funcDeclRef))
        {
            auto rangeStart = sb.getLength();

            ParamDecl* paramDecl = paramDeclRef.getDecl();
            auto paramType = getType(m_astBuilder, paramDeclRef);

            auto addParamElement = [&](Type* type, Index elementIndex)
            {
                if (!first)
                {
                    sb << ", ";
                    rangeStart += 2;
                }

                // Type part.
                {
                    ScopePart scopePart(this, Part::Type::ParamType);

                    // Seems these apply to parameters/VarDeclBase and are not part of the 'type'
                    // but seems more appropriate to put in the Type Part

                    if (paramDecl->hasModifier<InOutModifier>())
                    {
                        sb << toSlice("inout ");
                    }
                    else if (paramDecl->hasModifier<OutModifier>())
                    {
                        sb << toSlice("out ");
                    }
                    else if (paramDecl->hasModifier<InModifier>())
                    {
                        sb << toSlice("in ");
                    }

                    // And this to params/variables (not the type)
                    if (paramDecl->hasModifier<ConstModifier>())
                    {
                        sb << toSlice("const ");
                    }

                    addType(type);
                }

                // Output the parameter name if there is one, and it's enabled in the options
                if (m_optionFlags & OptionFlag::ParamNames && paramDecl->getName())
                {
                    sb << " ";
                    {
                        ScopePart scopePart(this, Part::Type::ParamName);
                        sb << paramDecl->getName()->text;
                        if (elementIndex != -1)
                            sb << "_" << elementIndex;
                    }
                }

                if (m_optionFlags & OptionFlag::DefaultParamValues && paramDecl->initExpr)
                {
                    sb << " = ";
                    addExpr(paramDecl->initExpr);
                }

                auto rangeEnd = sb.getLength();

                if (outParamRange)
                    outParamRange->add(makeRange<Index>(rangeStart, rangeEnd));
                first = false;
            };
            if (auto typePack = as<ConcreteTypePack>(paramType))
            {
                for (Index elementIndex = 0; elementIndex < typePack->getTypeCount();
                     ++elementIndex)
                {
                    addParamElement(typePack->getElementType(elementIndex), elementIndex);
                }
            }
            else
            {
                addParamElement(paramType, -1);
            }
        }

        sb << ")";
    }
    else if (auto genericDeclRef = declRef.as<GenericDecl>())
    {
        addGenericParams(genericDeclRef);

        addDeclParams(
            m_astBuilder->getMemberDeclRef(genericDeclRef, genericDeclRef.getDecl()->inner),
            outParamRange);
    }
    else
    {
    }
}

void ASTPrinter::addDeclKindPrefix(Decl* decl)
{
    if (auto genericDecl = as<GenericDecl>(decl))
    {
        decl = genericDecl->inner;
    }
    for (auto modifier : decl->modifiers)
    {
        if (modifier->getKeywordName())
        {
            if (m_optionFlags & OptionFlag::NoInternalKeywords)
            {
                if (as<TargetIntrinsicModifier>(modifier))
                    continue;
                if (as<MagicTypeModifier>(modifier))
                    continue;
                if (as<IntrinsicOpModifier>(modifier))
                    continue;
                if (as<IntrinsicTypeModifier>(modifier))
                    continue;
                if (as<BuiltinModifier>(modifier))
                    continue;
                if (as<BuiltinRequirementModifier>(modifier))
                    continue;
                if (as<BuiltinTypeModifier>(modifier))
                    continue;
                if (as<SpecializedForTargetModifier>(modifier))
                    continue;
                if (as<AttributeTargetModifier>(modifier))
                    continue;
                if (as<RequiredCUDASMVersionModifier>(modifier))
                    continue;
                if (as<RequiredSPIRVVersionModifier>(modifier))
                    continue;
                if (as<RequiredGLSLVersionModifier>(modifier))
                    continue;
                if (as<RequiredGLSLExtensionModifier>(modifier))
                    continue;
                if (as<RequiredWGSLExtensionModifier>(modifier))
                    continue;
                if (as<GLSLLayoutModifierGroupMarker>(modifier))
                    continue;
                if (as<HLSLLayoutSemantic>(modifier))
                    continue;
            }
            // Don't print out attributes.
            if (as<AttributeBase>(modifier))
                continue;
            m_builder << modifier->getKeywordName()->text << " ";
        }
    }
    if (as<FuncDecl>(decl))
    {
        m_builder << "func ";
    }
    else if (as<StructDecl>(decl))
    {
        m_builder << "struct ";
    }
    else if (as<InterfaceDecl>(decl))
    {
        m_builder << "interface ";
    }
    else if (as<ClassDecl>(decl))
    {
        m_builder << "class ";
    }
    else if (auto typedefDecl = as<TypeDefDecl>(decl))
    {
        m_builder << "typedef ";
        if (typedefDecl->type.type)
        {
            addType(typedefDecl->type.type);
            m_builder << " ";
        }
    }
    else if (const auto propertyDecl = as<PropertyDecl>(decl))
    {
        m_builder << "property ";
    }
    else if (as<NamespaceDecl>(decl))
    {
        m_builder << "namespace ";
    }
    else if (auto varDecl = as<VarDeclBase>(decl))
    {
        if (varDecl->getType())
        {
            addType(varDecl->getType());
            m_builder << " ";
        }
    }
    else if (as<EnumDecl>(decl))
    {
        m_builder << "enum ";
    }
    else if (auto enumCase = as<EnumCaseDecl>(decl))
    {
        if (enumCase->getType())
        {
            addType(enumCase->getType());
            m_builder << " ";
        }
    }
    else if (const auto assocType = as<AssocTypeDecl>(decl))
    {
        m_builder << "associatedtype ";
    }
    else if (const auto attribute = as<AttributeDecl>(decl))
    {
        m_builder << "attribute ";
    }
}

void ASTPrinter::addDeclResultType(const DeclRef<Decl>& inDeclRef)
{
    DeclRef<Decl> declRef = inDeclRef;
    if (auto genericDeclRef = declRef.as<GenericDecl>())
    {
        declRef =
            m_astBuilder->getMemberDeclRef<Decl>(genericDeclRef, genericDeclRef.getDecl()->inner);
    }

    if (declRef.as<ConstructorDecl>())
    {
    }
    else if (auto callableDeclRef = declRef.as<CallableDecl>())
    {
        m_builder << " -> ";

        {
            ScopePart scopePart(this, Part::Type::ReturnType);
            addType(getResultType(m_astBuilder, callableDeclRef));
        }
    }
    else if (auto propertyDecl = declRef.as<PropertyDecl>())
    {
        if (propertyDecl.getDecl()->type.type)
        {
            m_builder << " : ";
            addType(declRef.substitute(m_astBuilder, propertyDecl.getDecl()->type.type));
        }
    }
}

/* static */ void ASTPrinter::addDeclSignature(const DeclRef<Decl>& declRef)
{
    addDeclKindPrefix(declRef.getDecl());
    addDeclPath(declRef);
    addDeclParams(declRef);
    addDeclResultType(declRef);
}

/* static */ String ASTPrinter::getDeclSignatureString(
    DeclRef<Decl> declRef,
    ASTBuilder* astBuilder)
{
    ASTPrinter astPrinter(
        astBuilder,
        ASTPrinter::OptionFlag::NoInternalKeywords | ASTPrinter::OptionFlag::SimplifiedBuiltinType);
    astPrinter.addDeclSignature(declRef);
    return astPrinter.getString();
}

/* static */ String ASTPrinter::getDeclSignatureString(
    const LookupResultItem& item,
    ASTBuilder* astBuilder)
{
    return getDeclSignatureString(item.declRef, astBuilder);
}

/* static */ UnownedStringSlice ASTPrinter::getPart(
    Part::Type partType,
    const UnownedStringSlice& slice,
    const List<Part>& parts)
{
    const Index index =
        parts.findFirstIndex([&](const Part& part) -> bool { return part.type == partType; });
    return index >= 0 ? getPart(slice, parts[index]) : UnownedStringSlice();
}

UnownedStringSlice ASTPrinter::getPartSlice(Part::Type partType) const
{
    return m_parts ? getPart(partType, getSlice(), *m_parts) : UnownedStringSlice();
}


} // namespace Slang
