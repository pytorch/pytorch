#include "slang-ir-lower-cuda-builtin-types.h"

#include "slang-ir-clone.h"
#include "slang-ir-insts.h"
#include "slang-ir-layout.h"
#include "slang-ir-util.h"
#include "slang-ir.h"
namespace Slang
{

IRFunc* createMatrixUnpackFunc(
    IRMatrixType* matrixType,
    IRStructType* structType,
    IRStructKey* dataKey,
    IRArrayType* arrayType)
{
    IRBuilder builder(structType);
    builder.setInsertAfter(structType);
    auto func = builder.createFunc();
    auto funcType = builder.getFuncType(1, (IRType**)&structType, matrixType);
    func->setFullType(funcType);
    builder.addNameHintDecoration(func, UnownedStringSlice("unpackStorage"));
    builder.setInsertInto(func);
    builder.emitBlock();
    auto rowCount = (Index)getIntVal(matrixType->getRowCount());
    auto colCount = (Index)getIntVal(matrixType->getColumnCount());
    auto packedParam = builder.emitParam(structType);
    auto matrixArray = builder.emitFieldExtract(arrayType, packedParam, dataKey);
    List<IRInst*> args;
    args.setCount(rowCount * colCount);
    if (getIntVal(matrixType->getLayout()) == SLANG_MATRIX_LAYOUT_COLUMN_MAJOR)
    {
        for (IRIntegerValue c = 0; c < colCount; c++)
            for (IRIntegerValue r = 0; r < rowCount; r++)
                args[(Index)(r * colCount + c)] =
                    builder.emitElementExtract(matrixArray, (Index)(r * colCount + c));
    }
    else
    {
        for (IRIntegerValue c = 0; c < colCount; c++)
            for (IRIntegerValue r = 0; r < rowCount; r++)
                args[(Index)(c * rowCount + r)] =
                    builder.emitElementExtract(matrixArray, (Index)(r * colCount + c));
    }
    IRInst* result = builder.emitMakeMatrix(matrixType, (UInt)args.getCount(), args.getBuffer());
    builder.emitReturn(result);
    return func;
}

IRFunc* createMatrixPackFunc(
    IRMatrixType* matrixType,
    IRStructType* structType,
    IRArrayType* arrayType)
{
    IRBuilder builder(structType);
    builder.setInsertAfter(structType);
    auto func = builder.createFunc();
    auto funcType = builder.getFuncType(1, (IRType**)&matrixType, structType);
    func->setFullType(funcType);
    builder.addNameHintDecoration(func, UnownedStringSlice("packMatrix"));
    builder.setInsertInto(func);
    builder.emitBlock();
    auto rowCount = getIntVal(matrixType->getRowCount());
    auto colCount = getIntVal(matrixType->getColumnCount());
    auto originalParam = builder.emitParam(matrixType);
    List<IRInst*> elements;
    elements.setCount((Index)(rowCount * colCount));
    for (IRIntegerValue r = 0; r < rowCount; r++)
    {
        auto vector = builder.emitElementExtract(originalParam, r);
        for (IRIntegerValue c = 0; c < colCount; c++)
        {
            auto element = builder.emitElementExtract(vector, c);
            elements[(Index)(r * colCount + c)] = element;
        }
    }

    auto matrixArray =
        builder.emitMakeArray(arrayType, (UInt)elements.getCount(), elements.getBuffer());
    auto result = builder.emitMakeStruct(structType, 1, &matrixArray);
    builder.emitReturn(result);
    return func;
}

IRFunc* createVectorUnpackFunc(
    IRVectorType* vectorType,
    IRStructType* structType,
    IRStructKey* dataKey,
    IRArrayType* arrayType)
{
    IRBuilder builder(structType);
    builder.setInsertAfter(structType);
    auto func = builder.createFunc();
    auto funcType = builder.getFuncType(1, (IRType**)&structType, vectorType);
    func->setFullType(funcType);
    builder.addNameHintDecoration(func, UnownedStringSlice("unpackVector"));
    builder.setInsertInto(func);
    builder.emitBlock();
    auto packedParam = builder.emitParam(structType);
    auto packedArray = builder.emitFieldExtract(arrayType, packedParam, dataKey);
    auto count = getIntVal(vectorType->getElementCount());
    List<IRInst*> args;
    args.setCount((Index)count);
    for (IRIntegerValue ii = 0; ii < count; ++ii)
    {
        args[(Index)ii] = builder.emitElementExtract(packedArray, ii);
    }
    auto result = builder.emitMakeVector(vectorType, (UInt)args.getCount(), args.getBuffer());
    builder.emitReturn(result);
    return func;
}

IRFunc* createVectorPackFunc(
    IRVectorType* vectorType,
    IRStructType* structType,
    IRArrayType* arrayType)
{
    IRBuilder builder(structType);
    builder.setInsertAfter(structType);
    auto func = builder.createFunc();
    auto funcType = builder.getFuncType(1, (IRType**)&vectorType, structType);
    func->setFullType(funcType);
    builder.addNameHintDecoration(func, UnownedStringSlice("packVector"));
    builder.setInsertInto(func);
    builder.emitBlock();
    auto originalParam = builder.emitParam(vectorType);
    auto count = getIntVal(vectorType->getElementCount());
    List<IRInst*> args;
    args.setCount((Index)count);
    for (IRIntegerValue ii = 0; ii < count; ++ii)
    {
        args[(Index)ii] = builder.emitElementExtract(originalParam, ii);
    }
    auto packedArray = builder.emitMakeArray(arrayType, (UInt)args.getCount(), args.getBuffer());
    auto result = builder.emitMakeStruct(structType, 1, &packedArray);
    builder.emitReturn(result);
    return func;
}

LoweredBuiltinTypeInfo lowerMatrixType(
    IRBuilder* builder,
    IRMatrixType* matrixType,
    String nameSuffix)
{
    LoweredBuiltinTypeInfo info;

    auto loweredType = builder->createStructType();
    StringBuilder nameSB;
    bool isColMajor = getIntVal(matrixType->getLayout()) == SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;
    nameSB << "_MatrixStorage_";
    getTypeNameHint(nameSB, matrixType->getElementType());
    nameSB << getIntVal(matrixType->getRowCount()) << "x"
           << getIntVal(matrixType->getColumnCount());
    if (isColMajor)
        nameSB << "_ColMajor";
    nameSB << nameSuffix;
    builder->addNameHintDecoration(loweredType, nameSB.produceString().getUnownedSlice());
    auto structKey = builder->createStructKey();
    builder->addNameHintDecoration(structKey, UnownedStringSlice("data"));
    // auto vectorType = builder->getVectorType(matrixType->getElementType(),
    //     isColMajor?matrixType->getRowCount():matrixType->getColumnCount());

    auto arrayType = builder->getArrayType(
        matrixType->getElementType(),
        builder->getIntValue(
            builder->getUIntType(),
            getIntVal(matrixType->getColumnCount()) * getIntVal(matrixType->getRowCount())));

    builder->createStructField(loweredType, structKey, arrayType);

    info.originalType = matrixType;
    info.loweredType = loweredType;
    info.loweredInnerArrayType = arrayType;
    info.loweredInnerStructKey = structKey;
    info.convertLoweredToOriginal =
        createMatrixUnpackFunc(matrixType, loweredType, structKey, arrayType);
    info.convertOriginalToLowered = createMatrixPackFunc(matrixType, loweredType, arrayType);
    return info;
}

LoweredBuiltinTypeInfo lowerVectorType(
    IRBuilder* builder,
    IRVectorType* vectorType,
    String nameSuffix)
{
    LoweredBuiltinTypeInfo info;

    auto loweredType = builder->createStructType();

    StringBuilder nameSB;
    nameSB << "_VectorStorage_";
    getTypeNameHint(nameSB, vectorType->getElementType());
    nameSB << getIntVal(vectorType->getElementCount()) << "_";
    nameSB << nameSuffix;
    builder->addNameHintDecoration(loweredType, nameSB.produceString().getUnownedSlice());


    info.originalType = vectorType;
    info.loweredType = loweredType;

    auto structKey = builder->createStructKey();
    builder->addNameHintDecoration(structKey, UnownedStringSlice("data"));

    auto arrayType =
        builder->getArrayType(vectorType->getElementType(), vectorType->getElementCount());

    builder->createStructField(loweredType, structKey, arrayType);

    info.convertLoweredToOriginal =
        createVectorUnpackFunc(vectorType, loweredType, structKey, arrayType);
    info.convertOriginalToLowered = createVectorPackFunc(vectorType, loweredType, arrayType);

    return info;
}

LoweredBuiltinTypeInfo lowerStructType(
    BuiltinTypeLoweringEnv* env,
    IRBuilder* builder,
    IRStructType* structType,
    String nameSuffix)
{
    // Recursively lower the fields of the struct type
    List<IRType*> fieldTypes;
    List<IRStructField*> fields;
    for (auto field : structType->getFields())
    {
        fieldTypes.add(field->getFieldType());
        fields.add(field);
    }

    auto loweredType = builder->createStructType();
    StringBuilder nameSB;
    nameSB << "_StructStorage_";

    // Find a name hint for the struct type
    for (auto decoration : structType->getDecorations())
        if (auto nameHint = as<IRNameHintDecoration>(decoration))
            nameSB << nameHint->getName();

    nameSB << nameSuffix;
    builder->addNameHintDecoration(loweredType, nameSB.produceString().getUnownedSlice());

    bool changesRequired = false;

    // Lower field types.
    List<LoweredBuiltinTypeInfo> loweredFieldInfos;
    for (auto field : fields)
    {
        // Lower the field type
        auto loweredFieldInfo = lowerType(env, builder, field->getFieldType(), nameSuffix);
        loweredFieldInfos.add(loweredFieldInfo);

        // Add the lowered field type to the lowered struct type
        builder->createStructField(loweredType, field->getKey(), loweredFieldInfo.loweredType);

        if (loweredFieldInfo.convertLoweredToOriginal != nullptr)
            changesRequired = true;
    }

    if (!changesRequired)
    {
        // If no changes are required, then we can just return the original struct type
        LoweredBuiltinTypeInfo info;
        info.originalType = structType;
        info.loweredType = structType;
        info.convertLoweredToOriginal = nullptr;
        info.convertOriginalToLowered = nullptr;
        return info;
    }

    LoweredBuiltinTypeInfo info;
    info.originalType = structType;
    info.loweredType = loweredType;

    // Create the conversion function from the lowered struct type to the original struct type
    {
        builder->setInsertAfter(loweredType);
        auto func = builder->createFunc();
        auto funcType = builder->getFuncType(1, (IRType**)&loweredType, structType);
        func->setFullType(funcType);
        builder->addNameHintDecoration(func, UnownedStringSlice("convertLoweredToOriginal"));
        builder->setInsertInto(func);
        builder->emitBlock();
        auto loweredParam = builder->emitParam(loweredType);
        List<IRInst*> args;
        args.setCount((Index)fields.getCount());
        for (Index i = 0; i < fields.getCount(); i++)
        {
            auto loweredField = builder->emitFieldExtract(
                loweredFieldInfos[i].loweredType,
                loweredParam,
                fields[i]->getKey());
            List<IRInst*> callArgs;
            callArgs.add(loweredField);

            if (loweredFieldInfos[i].convertLoweredToOriginal == nullptr)
            {
                args[i] = loweredField;
                continue;
            }

            args[i] = builder->emitCallInst(
                loweredFieldInfos[i].originalType,
                loweredFieldInfos[i].convertLoweredToOriginal,
                callArgs);
        }

        auto result = builder->emitMakeStruct(structType, (UInt)args.getCount(), args.getBuffer());
        builder->emitReturn(result);
        info.convertLoweredToOriginal = func;
    }

    // Create the conversion function from the original struct type to the lowered struct type
    {
        builder->setInsertAfter(structType);
        auto func = builder->createFunc();
        auto funcType = builder->getFuncType(1, (IRType**)&structType, loweredType);
        func->setFullType(funcType);
        builder->addNameHintDecoration(func, UnownedStringSlice("convertOriginalToLowered"));
        builder->setInsertInto(func);
        builder->emitBlock();
        auto originalParam = builder->emitParam(structType);
        List<IRInst*> args;
        args.setCount((Index)fields.getCount());
        for (Index i = 0; i < fields.getCount(); i++)
        {
            auto originalField = builder->emitFieldExtract(
                loweredFieldInfos[i].originalType,
                originalParam,
                fields[i]->getKey());
            List<IRInst*> callArgs;
            callArgs.add(originalField);

            if (loweredFieldInfos[i].convertOriginalToLowered == nullptr)
            {
                args[i] = originalField;
                continue;
            }

            args[i] = builder->emitCallInst(
                loweredFieldInfos[i].loweredType,
                loweredFieldInfos[i].convertOriginalToLowered,
                callArgs);
        }

        auto result = builder->emitMakeStruct(loweredType, (UInt)args.getCount(), args.getBuffer());
        builder->emitReturn(result);
        info.convertOriginalToLowered = func;
    }

    return info;
}

LoweredBuiltinTypeInfo lowerArrayType(
    BuiltinTypeLoweringEnv* env,
    IRBuilder* builder,
    IRArrayType* arrayType,
    String nameSuffix)
{
    auto loweredElementTypeInfo = lowerType(env, builder, arrayType->getElementType(), nameSuffix);
    auto loweredType =
        builder->getArrayType(loweredElementTypeInfo.loweredType, arrayType->getElementCount());

    LoweredBuiltinTypeInfo info;
    info.originalType = arrayType;
    info.loweredType = loweredType;

    // If the element type was lowered, then we need to create conversion functions
    if (loweredElementTypeInfo.convertLoweredToOriginal != nullptr)
    {
        builder->setInsertAfter(loweredType);
        auto func = builder->createFunc();
        auto funcType = builder->getFuncType(1, (IRType**)&loweredType, arrayType);
        func->setFullType(funcType);
        builder->addNameHintDecoration(func, UnownedStringSlice("convertLoweredToOriginal"));
        builder->setInsertInto(func);
        builder->emitBlock();
        auto loweredParam = builder->emitParam(loweredType);

        auto count = getIntVal(arrayType->getElementCount());
        List<IRInst*> args;
        args.setCount((Index)count);
        for (IRIntegerValue ii = 0; ii < count; ++ii)
        {
            auto loweredElement = builder->emitElementExtract(loweredParam, ii);
            List<IRInst*> callArgs;
            callArgs.add(loweredElement);
            args[(Index)ii] = builder->emitCallInst(
                arrayType->getElementType(),
                loweredElementTypeInfo.convertLoweredToOriginal,
                callArgs);
        }

        auto result = builder->emitMakeArray(arrayType, (UInt)args.getCount(), args.getBuffer());
        builder->emitReturn(result);
        info.convertLoweredToOriginal = func;
    }

    if (loweredElementTypeInfo.convertOriginalToLowered != nullptr)
    {
        builder->setInsertAfter(arrayType);
        auto func = builder->createFunc();
        auto funcType = builder->getFuncType(1, (IRType**)&arrayType, loweredType);
        func->setFullType(funcType);
        builder->addNameHintDecoration(func, UnownedStringSlice("convertOriginalToLowered"));
        builder->setInsertInto(func);
        builder->emitBlock();
        auto originalParam = builder->emitParam(arrayType);
        auto count = getIntVal(arrayType->getElementCount());
        List<IRInst*> args;
        args.setCount((Index)count);
        for (IRIntegerValue ii = 0; ii < count; ++ii)
        {
            auto originalElement = builder->emitElementExtract(originalParam, ii);
            List<IRInst*> callArgs;
            callArgs.add(originalElement);
            args[(Index)ii] = builder->emitCallInst(
                loweredElementTypeInfo.loweredType,
                loweredElementTypeInfo.convertOriginalToLowered,
                callArgs);
        }

        auto result = builder->emitMakeArray(loweredType, (UInt)args.getCount(), args.getBuffer());
        builder->emitReturn(result);
        info.convertOriginalToLowered = func;
    }

    return info;
}

LoweredBuiltinTypeInfo lowerType(
    BuiltinTypeLoweringEnv* env,
    IRBuilder* builder,
    IRType* type,
    String nameSuffix)
{
    if (env->loweredTypes.containsKey(type))
        return env->loweredTypes[type];

    if (auto matrixType = as<IRMatrixType>(type))
    {
        auto loweredInfo = lowerMatrixType(builder, matrixType, nameSuffix);
        env->loweredTypes[type] = loweredInfo;
        return loweredInfo;
    }
    else if (auto vectorType = as<IRVectorType>(type))
    {
        auto loweredInfo = lowerVectorType(builder, vectorType, nameSuffix);
        env->loweredTypes[type] = loweredInfo;
        return loweredInfo;
    }
    else if (auto structType = as<IRStructType>(type))
    {
        auto loweredInfo = lowerStructType(env, builder, structType, nameSuffix);
        env->loweredTypes[type] = loweredInfo;
        return loweredInfo;
    }
    else if (auto arrayType = as<IRArrayType>(type))
    {
        auto loweredInfo = lowerArrayType(env, builder, arrayType, nameSuffix);
        env->loweredTypes[type] = loweredInfo;
        return loweredInfo;
    }

    LoweredBuiltinTypeInfo info;
    info.originalType = type;
    info.loweredType = type;
    info.convertLoweredToOriginal = nullptr;
    info.convertOriginalToLowered = nullptr;

    return info;
}
}; // namespace Slang