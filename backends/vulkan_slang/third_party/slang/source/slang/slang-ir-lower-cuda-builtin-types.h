#ifndef SLANG_IR_LOWER_BUILTIN_TYPES_H
#define SLANG_IR_LOWER_BUILTIN_TYPES_H

#include "slang-ir-clone.h"
#include "slang-ir-insts.h"
#include "slang-ir-layout.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{
struct LoweredBuiltinTypeInfo
{
    IRType* originalType;
    IRType* loweredType;
    IRType* loweredInnerArrayType =
        nullptr; // For matrix/array types that are lowered into a struct type, this is the inner
                 // array type of the data field.
    IRStructKey* loweredInnerStructKey =
        nullptr; // For matrix/array types that are lowered into a struct type, this is the struct
                 // key of the data field.
    IRFunc* convertOriginalToLowered = nullptr;
    IRFunc* convertLoweredToOriginal = nullptr;
};

struct BuiltinTypeLoweringEnv
{
    Dictionary<IRType*, LoweredBuiltinTypeInfo> loweredTypes;
};

LoweredBuiltinTypeInfo lowerMatrixType(
    IRBuilder* builder,
    IRMatrixType* matrixType,
    String nameSuffix = "");

LoweredBuiltinTypeInfo lowerVectorType(
    IRBuilder* builder,
    IRVectorType* vectorType,
    String nameSuffix = "");

LoweredBuiltinTypeInfo lowerStructType(
    BuiltinTypeLoweringEnv* env,
    IRBuilder* builder,
    IRStructType* structType,
    String nameSuffix = "");

LoweredBuiltinTypeInfo lowerType(
    BuiltinTypeLoweringEnv* env,
    IRBuilder* builder,
    IRType* type,
    String nameSuffix = "");

} // namespace Slang

#endif // SLANG_IR_LOWER_BUILTIN_TYPES_H