// slang-intrinsic-expand.cpp
#include "slang-intrinsic-expand.h"

#include "../core/slang-char-util.h"
#include "slang-emit-cuda.h"
#include "slang-ir-util.h"

namespace Slang
{

void IntrinsicExpandContext::emit(
    IRCall* inst,
    IRUse* args,
    Int argCount,
    const UnownedStringSlice& intrinsicText,
    IRInst* intrinsicInst)
{
    m_args = args;
    m_argCount = argCount;
    m_text = intrinsicText;
    m_callInst = inst;
    m_intrinsicInst = intrinsicInst;

    const auto returnType = inst->getDataType();

    // If it returns void -> then we don't need parenthesis
    if (as<IRVoidType>(returnType) == nullptr)
    {
        m_writer->emit("(");
        m_openParenCount++;
    }

    {
        char const* spanStart = intrinsicText.begin();
        char const* cursor = spanStart;
        char const* end = intrinsicText.end();

        while (cursor < end)
        {
            // Indicates the start of a 'special' sequence
            if (*cursor == '$')
            {
                // Flush any chars not yet output
                if (spanStart < cursor)
                {
                    m_writer->emit(spanStart, cursor);
                }

                // Requires special processing for output (ie we don't just copy chars verbatim)
                cursor = _emitSpecial(cursor);
                // The start is now after 'special' handling
                spanStart = cursor;
            }
            else
            {
                cursor++;
            }
        }

        // Flush any non escaped
        if (spanStart < end)
        {
            m_writer->emit(spanStart, end);
        }
    }

    // Close any remaining open parens
    for (Index i = 0; i < m_openParenCount; ++i)
    {
        m_writer->emit(")");
    }
}

static BaseType _getBaseTypeFromScalarType(SlangScalarType type)
{
    switch (type)
    {
    case SLANG_SCALAR_TYPE_INT32:
        return BaseType::Int;
    case SLANG_SCALAR_TYPE_UINT32:
        return BaseType::UInt;
    case SLANG_SCALAR_TYPE_INT16:
        return BaseType::Int16;
    case SLANG_SCALAR_TYPE_UINT16:
        return BaseType::UInt16;
    case SLANG_SCALAR_TYPE_INT64:
        return BaseType::Int64;
    case SLANG_SCALAR_TYPE_INTPTR:
        return BaseType::IntPtr;
    case SLANG_SCALAR_TYPE_UINT64:
        return BaseType::UInt64;
    case SLANG_SCALAR_TYPE_UINTPTR:
        return BaseType::UIntPtr;
    case SLANG_SCALAR_TYPE_INT8:
        return BaseType::Int8;
    case SLANG_SCALAR_TYPE_UINT8:
        return BaseType::UInt8;
    case SLANG_SCALAR_TYPE_FLOAT16:
        return BaseType::Half;
    case SLANG_SCALAR_TYPE_FLOAT32:
        return BaseType::Float;
    case SLANG_SCALAR_TYPE_FLOAT64:
        return BaseType::Double;
    case SLANG_SCALAR_TYPE_BOOL:
        return BaseType::Bool;
    default:
        return BaseType::Void;
    }
}

// TODO(JS): There is an inherent problem here:
//
// TimF: The big gotcha you'd have with trying to look up the IRVar or whatever from an intrinsic is
// that it is very easy for the user to "smuggle" a resource-type value through an intermediate
// function:
//
// ```
// Imagine this is user code...
// void f(RWTexture2D t) { t.YourOpThatYouAdded(...); }[attributeYouCareAbout(...)]
// RWTexture2D gTex;
// ...
// f(gTex);
//
// ```
//
// So when emitting IR code for f, there is no way to trace t back to gTex and get
// at[attributeYouCareAbout(...)] Structurally, you can get back to the IRParam for t and that's it.
// And even if there was some magic way to trace back through the call site, you would run into the
// problem that some call sites might call f(gTex) and other might call f(gSomeOtherTex) and there
// is no guarantee the attributes on those two textures would match.
//
// The VK back-end gets away with this kind of coincidentally, since the "legalization" we have to
// do for resources means that there wouldn't be a single f() function any more. But for CUDA and
// C++ that's not the case or generally desirable.

static IRFormatDecoration* _findImageFormatDecoration(IRInst* resourceInst)
{
    // JS(TODO):
    // There could perhaps be other situations, that need to be covered

    // If this is a load, we need to get the decoration from the field key
    if (IRLoad* load = as<IRLoad>(resourceInst))
    {
        if (IRFieldAddress* fieldAddress = as<IRFieldAddress>(load->getOperand(0)))
        {
            IRInst* field = fieldAddress->getField();
            return field->findDecoration<IRFormatDecoration>();
        }
    }
    // Otherwise just try on the instruction
    return resourceInst->findDecoration<IRFormatDecoration>();
}

// Returns true if dataType and imageFormat are compatible - that they have the same representation,
// and no conversion is required.
static bool _isImageFormatCompatible(ImageFormat imageFormat, IRType* dataType)
{
    int numElems = 1;

    if (auto vecType = as<IRVectorType>(dataType))
    {
        numElems = int(getIntVal(vecType->getElementCount()));
        dataType = vecType->getElementType();
    }

    BaseType baseType = BaseType::Void;
    if (auto basicType = as<IRBasicType>(dataType))
    {
        baseType = basicType->getBaseType();
    }

    const auto& imageFormatInfo = getImageFormatInfo(imageFormat);
    const BaseType formatBaseType = _getBaseTypeFromScalarType(imageFormatInfo.scalarType);

    if (numElems != imageFormatInfo.channelCount)
    {
        SLANG_ASSERT(!"Format doesn't match channel count");
        return false;
    }

    return formatBaseType == baseType;
}

static bool _isConvertRequired(ImageFormat imageFormat, IRInst* callee)
{
    auto textureType = as<IRTextureTypeBase>(callee->getDataType());
    IRType* elementType = textureType ? textureType->getElementType() : nullptr;
    return elementType && !_isImageFormatCompatible(imageFormat, elementType);
}

static size_t _calcBackingElementSizeInBytes(IRInst* resourceInst)
{
    // First see if there is a format associated with the resource
    if (IRFormatDecoration* formatDecoration = _findImageFormatDecoration(resourceInst))
    {
        return getImageFormatInfo(formatDecoration->getFormat()).sizeInBytes;
    }
    else
    {
        // If not we *assume* the backing format is the same as the element type used for access.
        /// Ie in RWTexture<T>, this would return sizeof(T)

        auto textureType = as<IRTextureTypeBase>(resourceInst->getDataType());
        IRType* elementType = textureType ? textureType->getElementType() : nullptr;

        if (elementType)
        {
            int numElems = 1;

            if (auto vecType = as<IRVectorType>(elementType))
            {
                numElems = int(getIntVal(vecType->getElementCount()));
                elementType = vecType->getElementType();
            }

            BaseType baseType = BaseType::Void;
            if (auto basicType = as<IRBasicType>(elementType))
            {
                baseType = basicType->getBaseType();
            }

            const auto& info = BaseTypeInfo::getInfo(baseType);
            return info.sizeInBytes * numElems;
        }
    }

    // When in doubt 4 is not a terrible guess based on limitations around DX11 etc
    return 4;
}

static bool _isResourceRead(IRCall* call)
{
    IRType* returnType = call->getDataType();
    return returnType && (as<IRVoidType>(returnType) == nullptr);
}

static bool _isResourceWrite(IRCall* call)
{
    IRType* returnType = call->getDataType();
    return returnType && (as<IRVoidType>(returnType) != nullptr);
}

static Index parseNumber(const char*& cursor, const char* end)
{
    char d = *cursor;
    SLANG_RELEASE_ASSERT(CharUtil::isDigit(d));
    Index n = 0;
    while (CharUtil::isDigit(d))
    {
        n = n * 10 + (d - '0');
        cursor++;
        if (cursor == end)
            break;
        d = *cursor;
    }
    return n;
}

const char* IntrinsicExpandContext::_emitSpecial(const char* cursor)
{
    const char* const end = m_text.end();

    // Check we are at start of 'special' sequence
    SLANG_ASSERT(*cursor == '$');
    cursor++;

    SLANG_RELEASE_ASSERT(cursor < end);

    char d = *cursor++;
    auto parseNat = [&]() -> Index { return parseNumber(cursor, end); };

    // Takes the first character of the number, parses the rest and returns the
    // total value.

    switch (d)
    {
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
        {
            --cursor;
            Index argIndex = parseNat() + m_argIndexOffset;
            // Simple case: emit one of the direct arguments to the call
            SLANG_RELEASE_ASSERT((0 <= argIndex) && (argIndex < m_argCount));
            m_writer->emit("(");
            m_emitter->emitOperand(m_args[argIndex].get(), getInfo(EmitOp::General));
            m_writer->emit(")");
        }
        break;

    case 'G':
        {
            // Get the type/value at the index of the specialization of this generic

            Index argIndex = parseNat() + m_argIndexOffset;

            IRSpecialize* specialize = as<IRSpecialize>(m_callInst->getCallee());
            SLANG_ASSERT(specialize);

            {
                auto argCount = Index(specialize->getArgCount());
                SLANG_UNUSED(argCount);
                SLANG_ASSERT(argIndex < argCount);

                auto arg = specialize->getArg(argIndex);

                if (auto type = as<IRType>(arg))
                {
                    m_emitter->emitType(type);
                }
                else
                {
                    m_emitter->emitVal(arg, getInfo(EmitOp::General));
                }
            }
        }
        break;

    case 'T':
        // Get the 'element' or `return` type for the type of the param at the index
        {
            IRType* type = nullptr;
            if (*cursor == 'R')
            {
                // Get the return type of the call
                cursor++;
                type = m_callInst->getDataType();
            }
            else
            {
                Index argIndex = parseNat() + m_argIndexOffset;
                SLANG_RELEASE_ASSERT(m_argCount > argIndex);

                type = m_args[argIndex].get()->getDataType();
                if (auto baseTextureType = as<IRTextureType>(type))
                {
                    type = baseTextureType->getElementType();
                }
            }
            SLANG_RELEASE_ASSERT(type);
            m_emitter->emitType(type);
        }
        break;

    case 'S':
        // Get the scalar type of a generic at specified index
        {
            Index argIndex = parseNat() + m_argIndexOffset;
            SLANG_RELEASE_ASSERT(m_argCount > argIndex);

            IRType* type = m_args[argIndex].get()->getDataType();
            if (auto baseTextureType = as<IRTextureType>(type))
            {
                type = baseTextureType->getElementType();
            }

            IRBasicType* underlyingType = nullptr;
            if (auto basicType = as<IRBasicType>(type))
            {
                underlyingType = basicType;
            }
            else if (auto vectorType = as<IRVectorType>(type))
            {
                underlyingType = as<IRBasicType>(vectorType->getElementType());
            }
            else if (auto matrixType = as<IRMatrixType>(type))
            {
                underlyingType = as<IRBasicType>(matrixType->getElementType());
            }

            SLANG_ASSERT(underlyingType);

            m_emitter->emitSimpleType(underlyingType);
        }
        break;
    case 'p':
        {
            // If we are calling a D3D texturing operation in the form t.Foo(s, ...),
            // then this form will pair up the t and s arguments as needed for a GLSL
            // texturing operation.
            SLANG_RELEASE_ASSERT(m_argCount >= 1);

            auto textureArg = m_args[0].get();

            if (auto baseTextureType = as<IRTextureType>(textureArg->getDataType()))
            {
                if (baseTextureType->isCombined())
                {
                    // If the base object (first argument) has a combined texture-sampler type,
                    // then we can simply use that argument as-is.
                    //
                    m_emitter->emitOperand(textureArg, getInfo(EmitOp::General));

                    // HACK: Because the target intrinsic strings are using indices based on the
                    // parameter list with the `SamplerState` parameter in place, seeing this opcode
                    // and this type tells us that we are about to have an off-by-one sort of
                    // situation. We quietly patch it up by offseting the index-based access for
                    // all the other operands.
                    //
                    m_argIndexOffset -= 1;
                }
                else
                {
                    // If the base object (first argument) has a texture type, then we expect
                    // the next argument to be a sampler to pair with it.
                    //
                    SLANG_RELEASE_ASSERT(m_argCount >= 2);
                    auto samplerArg = m_args[1].get();

                    // We will emit GLSL code to construct the corresponding combined
                    // texture/sampler type from the separate pieces.
                    //
                    m_emitter->emitTextureOrTextureSamplerType(baseTextureType, "sampler");
                    if (auto samplerType = as<IRSamplerStateTypeBase>(samplerArg->getDataType()))
                    {
                        if (as<IRSamplerComparisonStateType>(samplerType))
                        {
                            m_writer->emit("Shadow");
                        }
                    }

                    m_writer->emit("(");
                    m_emitter->emitOperand(textureArg, getInfo(EmitOp::General));
                    m_writer->emit(",");
                    m_emitter->emitOperand(samplerArg, getInfo(EmitOp::General));
                    m_writer->emit(")");
                }
            }
            else
            {
                SLANG_UNEXPECTED("bad format in intrinsic definition");
            }
        }
        break;

    case 'C':
        {
            // The $C intrinsic is a mechanism to change the name of an invocation depending on if
            // there is a format conversion required between the type associated by the resource and
            // the backing ImageFormat. Currently this is only implemented on CUDA, where there are
            // specialized versions of the RWTexture writes that will do a format conversion.
            if (m_emitter->getTarget() == CodeGenTarget::CUDASource)
            {
                IRInst* resourceInst = m_callInst->getArg(0);

                if (IRFormatDecoration* formatDecoration = _findImageFormatDecoration(resourceInst))
                {
                    const ImageFormat imageFormat = formatDecoration->getFormat();
                    if (_isConvertRequired(imageFormat, resourceInst))
                    {
                        // If the function returns something it's a reader so we may need to convert
                        // and in doing so require half
                        if (_isResourceRead(m_callInst))
                        {
                            // If the source format if half derived, then we need to enable half
                            switch (imageFormat)
                            {
                            case ImageFormat::r16f:
                            case ImageFormat::rg16f:
                            case ImageFormat::rgba16f:
                                {
                                    CUDAExtensionTracker* extensionTracker =
                                        as<CUDAExtensionTracker>(m_emitter->getExtensionTracker());
                                    if (extensionTracker)
                                    {
                                        extensionTracker->requireBaseType(BaseType::Half);
                                    }
                                    break;
                                }
                            default:
                                break;
                            }
                        }

                        // Append _convert on the name to signify we need to use a code path, that
                        // will automatically do the format conversion.
                        m_writer->emit("_convert");
                    }
                }
            }
            break;
        }

    case 'E':
        {
            /// Sometimes accesses need to be scaled. For example in CUDA the x coordinate for
            /// surface access is byte addressed. $E will return the byte size of the *backing
            /// element*.

            IRInst* resourceInst = m_callInst->getArg(0);
            size_t elemSizeInBytes = _calcBackingElementSizeInBytes(resourceInst);

            // If we have a format converstion and its a *write* we don't need to scale
            if (IRFormatDecoration* formatDecoration = _findImageFormatDecoration(resourceInst))
            {
                const ImageFormat imageFormat = formatDecoration->getFormat();
                if (_isConvertRequired(imageFormat, resourceInst) && _isResourceWrite(m_callInst))
                {
                    // If there is a conversion *and* it's a write we don't need to scale.
                    elemSizeInBytes = 1;
                }
            }

            SLANG_ASSERT(elemSizeInBytes > 0);
            m_writer->emitUInt64(UInt64(elemSizeInBytes));
            break;
        }

    case 'c':
        {
            // When doing texture access in glsl the result may need to be cast.
            // In particular if the underlying texture is 'half' based, glsl only accesses
            // (read/write) as float. So we need to cast to a half type on output. When storing into
            // a texture it is still the case the value written must be half - but we don't need to
            // do any casting there as half is coerced to float without a problem.
            SLANG_RELEASE_ASSERT(m_argCount >= 1);

            auto textureArg = m_args[0].get();
            if (auto baseTextureType = as<IRTextureType>(textureArg->getDataType()))
            {
                auto elementType = baseTextureType->getElementType();
                IRBasicType* underlyingType = nullptr;
                if (auto basicType = as<IRBasicType>(elementType))
                {
                    underlyingType = basicType;
                }
                else if (auto vectorType = as<IRVectorType>(elementType))
                {
                    underlyingType = as<IRBasicType>(vectorType->getElementType());
                }

                // We only need to output a cast if the underlying type is half.
                if (underlyingType && underlyingType->getOp() == kIROp_HalfType)
                {
                    m_emitter->emitSimpleType(elementType);
                    m_writer->emit("(");
                    m_openParenCount++;
                }
            }
        }
        break;

    case 'z':
        {
            // If we are calling a D3D texturing operation in the form t.Foo(s, ...),
            // where `t` is a `Texture*<T>`, then this is the step where we try to
            // properly swizzle the output of the equivalent GLSL call into the right
            // shape.
            SLANG_RELEASE_ASSERT(m_argCount >= 1);

            auto textureArg = m_args[0].get();
            IRType* elementType = nullptr;
            bool isShadowTexture = false;

            if (auto baseTextureType = as<IRTextureTypeBase>(textureArg->getDataType()))
            {
                elementType = baseTextureType->getElementType();
                isShadowTexture = baseTextureType->isShadow();
            }
            else
            {
                SLANG_UNEXPECTED("bad format in intrinsic definition");
            }

            // unorm and snorm attributes are currently ignored by glslang
            // https://github.com/KhronosGroup/glslang/blob/main/glslang/HLSL/hlslGrammar.cpp#L1476
            elementType = dropNormAttributes(elementType);

            SLANG_ASSERT(elementType);
            if (const auto basicType = as<IRBasicType>(elementType))
            {
                // A scalar result is expected

                // If the texture type is a scalar type, the return type from the target intrinsic
                // probably returns a scalar type already. For GLSL, ".x" swizzling happened to
                // work fine on a scalar value like "1.x", but it is invalid for WGSL.
                if (!isShadowTexture)
                {
                    m_writer->emit(".x");
                }
            }
            else if (auto vectorType = as<IRVectorType>(elementType))
            {
                // A vector result is expected
                auto elementCount = getIntVal(vectorType->getElementCount());

                if (elementCount < 4)
                {
                    char const* swiz[] = {"", ".x", ".xy", ".xyz", ""};
                    m_writer->emit(swiz[elementCount]);
                }
            }
            else if (const auto attrType = as<IRAttributedType>(elementType))
            {
                SLANG_UNEXPECTED("unhandled attributed type in intrinsic definition");
            }
            else
            {
                // What other cases are possible?
                SLANG_UNEXPECTED("bad format in intrinsic definition");
            }
        }
        break;

    case 'N':
        {
            // Extract the element count from a vector argument so that
            // we can use it in the constructed expression.

            Index argIndex = parseNat() + m_argIndexOffset;
            SLANG_RELEASE_ASSERT(m_argCount > argIndex);

            auto vectorArg = m_args[argIndex].get();
            if (auto vectorType = as<IRVectorType>(vectorArg->getDataType()))
            {
                auto elementCount = getIntVal(vectorType->getElementCount());
                m_writer->emit(elementCount);
            }
            else
            {
                SLANG_UNEXPECTED("bad format in intrinsic definition");
            }
        }
        break;

    case 'V':
        {
            // Take an argument of some scalar/vector type and pad
            // it out to a 4-vector with the same element type
            // (this is the inverse of `$z`).
            //
            Index argIndex = parseNat() + m_argIndexOffset;
            SLANG_RELEASE_ASSERT(m_argCount > argIndex);

            auto arg = m_args[argIndex].get();
            IRIntegerValue elementCount = 1;
            IRType* elementType = arg->getDataType();
            if (auto vectorType = as<IRVectorType>(elementType))
            {
                elementCount = getIntVal(vectorType->getElementCount());
                elementType = vectorType->getElementType();
            }

            if (elementCount == 4)
            {
                // In the simple case, the operand is already a 4-vector,
                // so we can just emit it as-is.
                m_emitter->emitOperand(arg, getInfo(EmitOp::General));
            }
            else
            {
                // Otherwise, we need to construct a 4-vector from the
                // value we have, padding it out with zero elements as
                // needed.
                //
                m_emitter->emitVectorTypeName(elementType, 4);
                m_writer->emit("(");
                m_emitter->emitOperand(arg, getInfo(EmitOp::General));
                for (IRIntegerValue ii = elementCount; ii < 4; ++ii)
                {
                    m_writer->emit(", ");
                    if (m_emitter->getSourceLanguage() == SourceLanguage::GLSL)
                    {
                        m_emitter->emitSimpleType(elementType);
                        m_writer->emit("(0)");
                    }
                    else
                    {
                        m_writer->emit("0");
                    }
                }
                m_writer->emit(")");
            }
        }
        break;
    case 'P':
        // Type-based prefix as used for CUDA and C++ targets
        {
            Index argIndex = 0;
            SLANG_RELEASE_ASSERT(m_argCount > argIndex);
            auto arg = m_args[argIndex].get();
            auto argType = arg->getDataType();

            const char* str = "";
            switch (argType->getOp())
            {
#define CASE(OP, STR) \
    case kIROp_##OP:  \
        str = #STR;   \
        break

                CASE(Int8Type, I8);
                CASE(Int16Type, I16);
                CASE(IntType, I32);
                CASE(Int64Type, I64);
                CASE(UInt8Type, U8);
                CASE(UInt16Type, U16);
                CASE(UIntType, U32);
                CASE(UInt64Type, U64);
                CASE(IntPtrType, IPTR);
                CASE(UIntPtrType, UPTR);
                CASE(HalfType, F16);
                CASE(FloatType, F32);
                CASE(DoubleType, F64);

#undef CASE

            default:
                SLANG_UNEXPECTED("unexpected type in intrinsic definition");
                break;
            }
            m_writer->emit(str);
        }
        break;

    case '*':
        {
            // An escape like `$*3` indicates that all arguments
            // from index 3 (in this example) and up should be
            // emitted as comma-separated expressions.
            //
            // We therefore expect the next byte to be a digit:
            //
            Index firstArgIndex = parseNat() + m_argIndexOffset;
            SLANG_RELEASE_ASSERT(m_argCount > firstArgIndex);

            for (Index argIndex = firstArgIndex; argIndex < m_argCount; ++argIndex)
            {
                if (argIndex != firstArgIndex)
                {
                    m_writer->emit(", ");
                }
                m_emitter->emitOperand(m_args[argIndex].get(), getInfo(EmitOp::General));
            }
        }
        break;

    case 'w':
        {
            // Emit a swizzle of the argument.
            Index argIndex = parseNat() + m_argIndexOffset;
            SLANG_RELEASE_ASSERT((0 <= argIndex) && (argIndex < m_argCount));
            auto arg = m_args[argIndex];
            d = *cursor++;
            auto vectorType = as<IRVectorType>(arg.get()->getDataType());
            auto vectorSize = as<IRIntLit>(vectorType->getElementCount())->getValue();
            const char swizzleNames[] = {'x', 'y', 'z', 'w'};
            if (d == 'b')
            {
                // Emit the first half the vector up to the n-1'th element.
                for (int i = 0; i < Math::Min(4, (int)vectorSize - 1); i++)
                    m_writer->emitChar(swizzleNames[i]);
            }
            else if (d == 'e')
            {
                // Emit the swizzle to get the last element.
                if (vectorSize <= 4)
                    m_writer->emitChar(swizzleNames[vectorSize - 1]);
            }
            break;
        }
    case '!':
        {
            // Emit a literal directly without any prefix/postfix/casts.
            Index argIndex = parseNat() + m_argIndexOffset;
            SLANG_RELEASE_ASSERT((0 <= argIndex) && (argIndex < m_argCount));
            auto arg = m_args[argIndex];
            if (auto intLit = as<IRIntLit>(arg.get()))
            {
                m_writer->emitInt64(intLit->getValue());
            }
            else if (auto stringLit = as<IRStringLit>(arg.get()))
            {
                m_writer->emit(stringLit->getStringSlice());
            }
            else
            {
                SLANG_UNEXPECTED("unexpected literal argument in intrinsic call.");
            }
            break;
        }
    case '[':
        {
            Index argIndex = parseNat();
            auto arg = m_intrinsicInst->getOperand((UInt)(1 + argIndex));
            if (!arg->getDataType())
            {
                m_emitter->emitSimpleType((IRType*)arg);
            }
            else
            {
                switch (arg->getDataType()->getOp())
                {
                case kIROp_TypeKind:
                case kIROp_TypeType:
                    m_emitter->emitType((IRType*)arg);
                    break;
                default:
                    m_emitter->emitOperand(
                        m_intrinsicInst->getOperand((UInt)(1 + argIndex)),
                        getInfo(EmitOp::General));
                    break;
                }
            }
            SLANG_ASSERT(*cursor == ']');
            cursor++;
            break;
        }
    default:
        SLANG_UNEXPECTED("bad format in intrinsic definition");
        break;
    }

    // Return the cursor position. Will be next character after characters processed
    // for the 'special'
    return cursor;
}

} // namespace Slang
