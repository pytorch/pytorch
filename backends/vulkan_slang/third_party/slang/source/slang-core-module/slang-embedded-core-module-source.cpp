#include "slang-compiler.h"
#include "slang-core-module-textures.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

#define STRINGIZE(x) STRINGIZE2(x)
#define STRINGIZE2(x) #x
#define LINE_STRING STRINGIZE(__LINE__)

namespace Slang
{
// We are going to generate the core module source code from a more compact
// description. For example, we need to generate all the `operator`
// declarations for the basic unary and binary math operations on
// builtin types. To do this, we will make a big array of all these
// types, and associate them with data on their categories/capabilities
// so that we generate only the correct operations.
//
enum
{
    SINT_MASK = 1 << 0,
    FLOAT_MASK = 1 << 1,
    BOOL_RESULT = 1 << 2,
    BOOL_MASK = 1 << 3,
    UINT_MASK = 1 << 4,

    INT_MASK = SINT_MASK | UINT_MASK,
    ARITHMETIC_MASK = INT_MASK | FLOAT_MASK,
    LOGICAL_MASK = INT_MASK | BOOL_MASK,
    ANY_MASK = INT_MASK | FLOAT_MASK | BOOL_MASK,
};

// We are going to declare initializers that allow for conversion between
// all of our base types, and we need a way to priotize those conversion
// by giving them different costs. Rather than maintain a hard-coded table
// of N^2 costs for N basic types, we are going to try to do things a bit
// more systematically.
//
// Every base type will be given a "kind" and a "rank" for conversion.
// The kind will classify it as signed/unsigned/float, and the rank will
// classify it by its logical bit size (with a distinct rank for pointer-sized
// types that logically sits between 32- and 64-bit types).
//
enum BaseTypeConversionKind : uint8_t
{
    kBaseTypeConversionKind_Signed,
    kBaseTypeConversionKind_Unsigned,
    kBaseTypeConversionKind_Float,
    kBaseTypeConversionKind_Error,
};
enum BaseTypeConversionRank : uint8_t
{
    kBaseTypeConversionRank_Bool,
    kBaseTypeConversionRank_Int8,
    kBaseTypeConversionRank_Int16,
    kBaseTypeConversionRank_Int32,
    kBaseTypeConversionRank_IntPtr,
    kBaseTypeConversionRank_Int64,
    kBaseTypeConversionRank_Error,
};

// Here we declare the table of all our builtin types, so that we can generate all the relevant
// declarations.
//
struct BaseTypeConversionInfo
{
    char const* name;
    BaseType tag;
    unsigned flags;
    BaseTypeConversionKind conversionKind;
    BaseTypeConversionRank conversionRank;
};
static const BaseTypeConversionInfo kBaseTypes[] = {
    // TODO: `void` really shouldn't be in the `BaseType` enumeration, since it behaves so
    // differently across the board
    {"void", BaseType::Void, 0, kBaseTypeConversionKind_Error, kBaseTypeConversionRank_Error},

    {"bool",
     BaseType::Bool,
     BOOL_MASK,
     kBaseTypeConversionKind_Unsigned,
     kBaseTypeConversionRank_Bool},

    {"int8_t",
     BaseType::Int8,
     SINT_MASK,
     kBaseTypeConversionKind_Signed,
     kBaseTypeConversionRank_Int8},
    {"int16_t",
     BaseType::Int16,
     SINT_MASK,
     kBaseTypeConversionKind_Signed,
     kBaseTypeConversionRank_Int16},
    {"int",
     BaseType::Int,
     SINT_MASK,
     kBaseTypeConversionKind_Signed,
     kBaseTypeConversionRank_Int32},
    {"int64_t",
     BaseType::Int64,
     SINT_MASK,
     kBaseTypeConversionKind_Signed,
     kBaseTypeConversionRank_Int64},
    {"intptr_t",
     BaseType::IntPtr,
     SINT_MASK,
     kBaseTypeConversionKind_Signed,
     kBaseTypeConversionRank_IntPtr},


    {"half",
     BaseType::Half,
     FLOAT_MASK,
     kBaseTypeConversionKind_Float,
     kBaseTypeConversionRank_Int16},
    {"float",
     BaseType::Float,
     FLOAT_MASK,
     kBaseTypeConversionKind_Float,
     kBaseTypeConversionRank_Int32},
    {"double",
     BaseType::Double,
     FLOAT_MASK,
     kBaseTypeConversionKind_Float,
     kBaseTypeConversionRank_Int64},

    {"uint8_t",
     BaseType::UInt8,
     UINT_MASK,
     kBaseTypeConversionKind_Unsigned,
     kBaseTypeConversionRank_Int8},
    {"uint16_t",
     BaseType::UInt16,
     UINT_MASK,
     kBaseTypeConversionKind_Unsigned,
     kBaseTypeConversionRank_Int16},
    {"uint",
     BaseType::UInt,
     UINT_MASK,
     kBaseTypeConversionKind_Unsigned,
     kBaseTypeConversionRank_Int32},
    {"uint64_t",
     BaseType::UInt64,
     UINT_MASK,
     kBaseTypeConversionKind_Unsigned,
     kBaseTypeConversionRank_Int64},
    {"uintptr_t",
     BaseType::UIntPtr,
     UINT_MASK,
     kBaseTypeConversionKind_Unsigned,
     kBaseTypeConversionRank_IntPtr},
};

void Session::finalizeSharedASTBuilder()
{
    // Force creation of all builtin types so we can make sure
    // they are created by the builtin AST builder instead of
    // some user linkage's ast builder. This avoid the problem
    // of storing a reference to these global types that are
    // owned by a user linkage that gets deleted with the linkage.
    //
    globalAstBuilder->getNoneType();
    globalAstBuilder->getNullPtrType();
    globalAstBuilder->getBottomType();
    globalAstBuilder->getErrorType();
    globalAstBuilder->getInitializerListType();
    globalAstBuilder->getOverloadedType();
    globalAstBuilder->getStringType();
    globalAstBuilder->getEnumTypeType();
    globalAstBuilder->getDiffInterfaceType();
    globalAstBuilder->getSharedASTBuilder()->getDynamicType();
    globalAstBuilder->getSharedASTBuilder()->getDiffInterfaceType();
    globalAstBuilder->getSharedASTBuilder()->getNativeStringType();
    for (auto& baseType : kBaseTypes)
        globalAstBuilder->getBuiltinType(baseType.tag);
}

// Given two base types, we need to be able to compute the cost of converting between them.
ConversionCost getBaseTypeConversionCost(
    BaseTypeConversionInfo const& toInfo,
    BaseTypeConversionInfo const& fromInfo)
{
    if (toInfo.conversionKind == fromInfo.conversionKind &&
        toInfo.conversionRank == fromInfo.conversionRank)
    {
        // Thse should represent the exact same type.
        return kConversionCost_None;
    }

    // Conversions within the same kind are easist to handle
    if (toInfo.conversionKind == fromInfo.conversionKind)
    {
        // If we are converting to a "larger" type, then
        // we are doing a lossless promotion, and otherwise
        // we are doing a demotion.
        if (toInfo.conversionRank > fromInfo.conversionRank)
            return kConversionCost_RankPromotion;
        else
            return kConversionCost_GeneralConversion;
    }
    else if (fromInfo.tag == BaseType::Bool && toInfo.tag == BaseType::Int)
    {
        return kConversionCost_BoolToInt;
    }

    // If we are converting from an unsigned integer type to
    // a signed integer type that is guaranteed to be larger,
    // then that is also a lossless promotion.
    //
    // There is one additional wrinkle here, which is that
    // a conversion from a 32-bit unsigned integer to a
    // "pointer-sized" signed integer should be treated
    // as unsafe, because the pointer size might also be
    // 32 bits.
    //
    // The same basic exemption applied when converting
    // *from* a pointer-sized unsigned integer.
    else if (
        toInfo.conversionKind == kBaseTypeConversionKind_Signed &&
        fromInfo.conversionKind == kBaseTypeConversionKind_Unsigned &&
        toInfo.conversionRank > fromInfo.conversionRank &&
        toInfo.conversionRank != kBaseTypeConversionRank_IntPtr &&
        fromInfo.conversionRank != kBaseTypeConversionRank_IntPtr)
    {
        return kConversionCost_UnsignedToSignedPromotion;
    }
    // Same-size unsigned to signed integer conversion.
    else if (
        toInfo.conversionKind == kBaseTypeConversionKind_Signed &&
        fromInfo.conversionKind == kBaseTypeConversionKind_Unsigned &&
        toInfo.conversionRank == fromInfo.conversionRank &&
        toInfo.conversionRank != kBaseTypeConversionRank_IntPtr &&
        fromInfo.conversionRank != kBaseTypeConversionRank_IntPtr)
    {
        return kConversionCost_SameSizeUnsignedToSignedConversion;
    }

    // Conversion from signed to unsigned is always lossy,
    // but it is preferred over conversions from unsigned
    // to signed, for same-size types.
    else if (
        toInfo.conversionKind == kBaseTypeConversionKind_Unsigned &&
        fromInfo.conversionKind == kBaseTypeConversionKind_Signed &&
        toInfo.conversionRank >= fromInfo.conversionRank)
    {
        return kConversionCost_SignedToUnsignedConversion;
    }

    // Conversion from an integer to a floating-point type
    // is never considered a promotion (even when the value
    // would fit in the available mantissa bits).
    // If the destination type is at least 32 bits we consider
    // this a reasonably good conversion, though.
    //
    // Note that this means we do *not* consider implicit
    // conversion to `half` as a good conversion, even for small
    // types. This makes sense because we relaly want to prefer
    // conversion to `float` as the default.
    else if (
        toInfo.conversionKind == kBaseTypeConversionKind_Float &&
        toInfo.conversionRank >= kBaseTypeConversionRank_Int32 &&
        fromInfo.conversionRank >= kBaseTypeConversionRank_Int8)
    {
        return kConversionCost_IntegerToFloatConversion;
    }
    else if (
        toInfo.conversionKind == kBaseTypeConversionKind_Float &&
        toInfo.conversionRank >= kBaseTypeConversionRank_Int16 &&
        fromInfo.conversionRank >= kBaseTypeConversionRank_Int8)
    {
        return kConversionCost_IntegerToHalfConversion;
    }
    // All other cases are considered as "general" conversions,
    // where we don't consider any one conversion better than
    // any others.
    else
    {
        return kConversionCost_GeneralConversion;
    }
}

IROp getBaseTypeConversionOp(
    BaseTypeConversionInfo const& toInfo,
    BaseTypeConversionInfo const& fromInfo)
{
    if (toInfo.tag == fromInfo.tag)
        return kIROp_Nop;

    IROp intrinsicOpCode = kIROp_Nop;
    auto toStyle = getTypeStyle(toInfo.tag);
    auto fromStyle = getTypeStyle(fromInfo.tag);
    if (toStyle == kIROp_BoolType)
        toStyle = kIROp_IntType;
    if (fromStyle == kIROp_BoolType)
        fromStyle = kIROp_IntType;
    if (toStyle == kIROp_IntType && fromStyle == kIROp_IntType)
        intrinsicOpCode = kIROp_IntCast;
    if (toStyle == kIROp_IntType && fromStyle == kIROp_FloatType)
        intrinsicOpCode = kIROp_CastFloatToInt;
    if (toStyle == kIROp_FloatType && fromStyle == kIROp_IntType)
        intrinsicOpCode = kIROp_CastIntToFloat;
    if (toStyle == kIROp_FloatType && fromStyle == kIROp_FloatType)
        intrinsicOpCode = kIROp_FloatCast;
    return intrinsicOpCode;
}

struct IntrinsicOpInfo
{
    IROp opCode;
    char const* funcName;
    char const* opName;
    char const* interface;
    unsigned flags;
};

[[maybe_unused]] static const IntrinsicOpInfo intrinsicUnaryOps[] = {
    {kIROp_Neg, "neg", "-", "__BuiltinArithmeticType", ARITHMETIC_MASK},
    {kIROp_Not, "logicalNot", "!", nullptr, BOOL_MASK | BOOL_RESULT},
    {kIROp_BitNot, "not", "~", "__BuiltinLogicalType", INT_MASK},
};

[[maybe_unused]] static const IntrinsicOpInfo intrinsicBinaryOps[] = {
    {kIROp_Add, "add", "+", "__BuiltinArithmeticType", ARITHMETIC_MASK},
    {kIROp_Sub, "sub", "-", "__BuiltinArithmeticType", ARITHMETIC_MASK},
    {kIROp_Mul, "mul", "*", "__BuiltinArithmeticType", ARITHMETIC_MASK},
    {kIROp_Div, "div", "/", "__BuiltinArithmeticType", ARITHMETIC_MASK},
    {kIROp_IRem, "irem", "%", "__BuiltinIntegerType", INT_MASK},
    {kIROp_FRem, "frem", "%", "__BuiltinFloatingPointType", FLOAT_MASK},
    {kIROp_And, "logicalAnd", "&&", nullptr, BOOL_MASK | BOOL_RESULT},
    {kIROp_Or, "logicalOr", "||", nullptr, BOOL_MASK | BOOL_RESULT},
    {kIROp_BitAnd, "and", "&", "__BuiltinLogicalType", LOGICAL_MASK},
    {kIROp_BitOr, "or", "|", "__BuiltinLogicalType", LOGICAL_MASK},
    {kIROp_BitXor, "xor", "^", "__BuiltinLogicalType", LOGICAL_MASK},
    {kIROp_Eql, "eql", "==", "__BuiltinType", ANY_MASK | BOOL_RESULT},
    {kIROp_Neq, "neq", "!=", "__BuiltinType", ANY_MASK | BOOL_RESULT},
    {kIROp_Greater, "greater", ">", "__BuiltinArithmeticType", ARITHMETIC_MASK | BOOL_RESULT},
    {kIROp_Less, "less", "<", "__BuiltinArithmeticType", ARITHMETIC_MASK | BOOL_RESULT},
    {kIROp_Geq, "geq", ">=", "__BuiltinArithmeticType", ARITHMETIC_MASK | BOOL_RESULT},
    {kIROp_Leq, "leq", "<=", "__BuiltinArithmeticType", ARITHMETIC_MASK | BOOL_RESULT},
};

// Integer types that can be used in atomic operations in CUDA.
[[maybe_unused]] static const char* kCudaAtomicIntegerTypes[] =
    {"int", "uint", "uint64_t", "int64_t"};

// Both the following functions use these macros.
// NOTE! They require a variable named path to emit the #line correctly if in source file.
#define SLANG_RAW(TEXT) sb << TEXT;
#define SLANG_SPLICE(EXPR) sb << (EXPR);

#define EMIT_LINE_DIRECTIVE() sb << "#line " << (__LINE__ + 1) << " \"" << path << "\"\n"

ComPtr<ISlangBlob> Session::getCoreLibraryCode()
{
#if SLANG_EMBED_CORE_MODULE_SOURCE
    if (!coreLibraryCode)
    {
        StringBuilder sb;
        const String path = getCoreModulePath();
#include "core.meta.slang.h"
        coreLibraryCode = StringBlob::moveCreate(sb);
    }
#endif
    return coreLibraryCode;
}

ComPtr<ISlangBlob> Session::getHLSLLibraryCode()
{
#if SLANG_EMBED_CORE_MODULE_SOURCE
    if (!hlslLibraryCode)
    {
        const String path = getCoreModulePath();
        StringBuilder sb;
#include "hlsl.meta.slang.h"
        hlslLibraryCode = StringBlob::moveCreate(sb);
    }
#endif
    return hlslLibraryCode;
}

ComPtr<ISlangBlob> Session::getAutodiffLibraryCode()
{
#if SLANG_EMBED_CORE_MODULE_SOURCE
    if (!autodiffLibraryCode)
    {
        const String path = getCoreModulePath();
        StringBuilder sb;
#include "diff.meta.slang.h"
        autodiffLibraryCode = StringBlob::moveCreate(sb);
    }
#endif
    return autodiffLibraryCode;
}

ComPtr<ISlangBlob> Session::getGLSLLibraryCode()
{
    if (!glslLibraryCode)
    {
        const String path = getCoreModulePath();
        StringBuilder sb;
#include "glsl.meta.slang.h"
        glslLibraryCode = StringBlob::moveCreate(sb);
    }
    return glslLibraryCode;
}
} // namespace Slang
