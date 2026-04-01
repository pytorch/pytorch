#include "slang-ir-lower-buffer-element-type.h"

#include "slang-ir-clone.h"
#include "slang-ir-insts.h"
#include "slang-ir-layout.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{

struct TypeLoweringConfig
{
    AddressSpace addressSpace;
    IRTypeLayoutRules* layoutRule;
    bool operator==(const TypeLoweringConfig& other) const
    {
        return addressSpace == other.addressSpace && layoutRule == other.layoutRule;
    }
    HashCode getHashCode() const
    {
        return combineHash(Slang::getHashCode(addressSpace), Slang::getHashCode(layoutRule));
    }
};
TypeLoweringConfig getTypeLoweringConfigForBuffer(TargetProgram* target, IRType* bufferType);

struct LoweredElementTypeContext
{
    static const IRIntegerValue kMaxArraySizeToUnroll = 32;

    enum ConversionMethodKind
    {
        Func,
        Opcode
    };
    struct ConversionMethod
    {
        ConversionMethodKind kind = ConversionMethodKind::Func;
        union
        {
            IRFunc* func;
            IROp op;
        };
        ConversionMethod() { func = nullptr; }
        operator bool()
        {
            return kind == ConversionMethodKind::Func ? func != nullptr : op != kIROp_Nop;
        }
        ConversionMethod& operator=(IRFunc* f)
        {
            kind = ConversionMethodKind::Func;
            this->func = f;
            return *this;
        }
        ConversionMethod& operator=(IROp irop)
        {
            kind = ConversionMethodKind::Opcode;
            this->op = irop;
            return *this;
        }
        IRInst* apply(IRBuilder& builder, IRType* resultType, IRInst* operandAddr)
        {
            if (!*this)
                return builder.emitLoad(operandAddr);
            if (kind == ConversionMethodKind::Func)
                return builder.emitCallInst(resultType, func, 1, &operandAddr);
            else
            {
                auto val = builder.emitLoad(operandAddr);
                return builder.emitIntrinsicInst(resultType, op, 1, &val);
            }
        }
        void applyDestinationDriven(IRBuilder& builder, IRInst* dest, IRInst* operand)
        {
            if (!*this)
            {
                builder.emitStore(dest, operand);
                return;
            }
            if (kind == ConversionMethodKind::Func)
            {
                IRInst* operands[] = {dest, operand};
                builder.emitCallInst(builder.getVoidType(), func, 2, operands);
            }
            else
            {
                auto val = builder.emitIntrinsicInst(
                    tryGetPointedToType(&builder, dest->getDataType()),
                    op,
                    1,
                    &operand);
                builder.emitStore(dest, val);
            }
        }
    };

    struct LoweredElementTypeInfo
    {
        IRType* originalType;
        IRType* loweredType;
        IRType* loweredInnerArrayType =
            nullptr; // For matrix/array types that are lowered into a struct type, this is the
                     // inner array type of the data field.
        IRStructKey* loweredInnerStructKey =
            nullptr; // For matrix/array types that are lowered into a struct type, this is the
                     // struct key of the data field.
        ConversionMethod convertOriginalToLowered;
        ConversionMethod convertLoweredToOriginal;
    };

    struct LoweredTypeMap : RefObject
    {
        Dictionary<IRType*, LoweredElementTypeInfo> loweredTypeInfo;
        Dictionary<IRType*, LoweredElementTypeInfo> mapLoweredTypeToInfo;
    };

    Dictionary<TypeLoweringConfig, RefPtr<LoweredTypeMap>> loweredTypeInfoMaps;

    struct ConversionMethodKey
    {
        IRType* toType;
        IRType* fromType;
        bool operator==(const ConversionMethodKey& other) const
        {
            return toType == other.toType && fromType == other.fromType;
        }
        HashCode64 getHashCode() const
        {
            return combineHash(Slang::getHashCode(toType), Slang::getHashCode(fromType));
        }
    };

    Dictionary<ConversionMethodKey, ConversionMethod> conversionMethodMap;
    ConversionMethod getConversionMethod(IRType* toType, IRType* fromType)
    {
        ConversionMethodKey key;
        key.toType = toType;
        key.fromType = fromType;
        ConversionMethod method;
        conversionMethodMap.tryGetValue(key, method);
        return method;
    }

    SlangMatrixLayoutMode defaultMatrixLayout = SLANG_MATRIX_LAYOUT_ROW_MAJOR;
    TargetProgram* target;
    BufferElementTypeLoweringOptions options;

    LoweredElementTypeContext(
        TargetProgram* target,
        BufferElementTypeLoweringOptions inOptions,
        SlangMatrixLayoutMode inDefaultMatrixLayout)
        : target(target), defaultMatrixLayout(inDefaultMatrixLayout), options(inOptions)
    {
    }

    IRFunc* createMatrixUnpackFunc(
        IRMatrixType* matrixType,
        IRStructType* structType,
        IRStructKey* dataKey)
    {
        IRBuilder builder(structType);
        builder.setInsertAfter(structType);
        auto func = builder.createFunc();
        auto refStructType = builder.getRefType(structType, AddressSpace::Generic);
        auto funcType = builder.getFuncType(1, (IRType**)&refStructType, matrixType);
        func->setFullType(funcType);
        builder.addNameHintDecoration(func, UnownedStringSlice("unpackStorage"));
        builder.addForceInlineDecoration(func);
        builder.setInsertInto(func);
        builder.emitBlock();
        auto rowCount = (Index)getIntVal(matrixType->getRowCount());
        auto colCount = (Index)getIntVal(matrixType->getColumnCount());
        auto packedParamRef = builder.emitParam(refStructType);
        auto packedParam = builder.emitLoad(packedParamRef);
        auto vectorArray = builder.emitFieldExtract(packedParam, dataKey);
        List<IRInst*> args;
        args.setCount(rowCount * colCount);
        if (getIntVal(matrixType->getLayout()) == SLANG_MATRIX_LAYOUT_COLUMN_MAJOR)
        {
            for (IRIntegerValue c = 0; c < colCount; c++)
            {
                auto vector = builder.emitElementExtract(vectorArray, c);
                for (IRIntegerValue r = 0; r < rowCount; r++)
                {
                    auto element = builder.emitElementExtract(vector, r);
                    args[(Index)(r * colCount + c)] = element;
                }
            }
        }
        else
        {
            for (IRIntegerValue r = 0; r < rowCount; r++)
            {
                auto vector = builder.emitElementExtract(vectorArray, r);
                for (IRIntegerValue c = 0; c < colCount; c++)
                {
                    auto element = builder.emitElementExtract(vector, c);
                    args[(Index)(r * colCount + c)] = element;
                }
            }
        }
        IRInst* result =
            builder.emitMakeMatrix(matrixType, (UInt)args.getCount(), args.getBuffer());
        builder.emitReturn(result);
        return func;
    }

    IRFunc* createMatrixPackFunc(
        IRMatrixType* matrixType,
        IRStructType* structType,
        IRVectorType* vectorType,
        IRArrayType* arrayType)
    {
        IRBuilder builder(structType);
        builder.setInsertAfter(structType);
        auto func = builder.createFunc();
        auto outStructType = builder.getRefType(structType, AddressSpace::Generic);
        IRType* paramTypes[] = {outStructType, matrixType};
        auto funcType = builder.getFuncType(2, paramTypes, builder.getVoidType());
        func->setFullType(funcType);
        builder.addNameHintDecoration(func, UnownedStringSlice("packMatrix"));
        builder.addForceInlineDecoration(func);
        builder.setInsertInto(func);
        builder.emitBlock();
        auto rowCount = getIntVal(matrixType->getRowCount());
        auto colCount = getIntVal(matrixType->getColumnCount());
        auto outParam = builder.emitParam(outStructType);
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
        List<IRInst*> vectors;
        if (getIntVal(matrixType->getLayout()) == SLANG_MATRIX_LAYOUT_COLUMN_MAJOR)
        {
            for (IRIntegerValue c = 0; c < colCount; c++)
            {
                List<IRInst*> vecArgs;
                for (IRIntegerValue r = 0; r < rowCount; r++)
                {
                    auto element = elements[(Index)(r * colCount + c)];
                    vecArgs.add(element);
                }
                // Fill in default values for remaining elements in the vector.
                for (IRIntegerValue r = rowCount; r < getIntVal(vectorType->getElementCount()); r++)
                {
                    vecArgs.add(builder.emitDefaultConstruct(vectorType->getElementType()));
                }
                auto colVector = builder.emitMakeVector(
                    vectorType,
                    (UInt)vecArgs.getCount(),
                    vecArgs.getBuffer());
                vectors.add(colVector);
            }
        }
        else
        {
            for (IRIntegerValue r = 0; r < rowCount; r++)
            {
                List<IRInst*> vecArgs;
                for (IRIntegerValue c = 0; c < colCount; c++)
                {
                    auto element = elements[(Index)(r * colCount + c)];
                    vecArgs.add(element);
                }
                // Fill in default values for remaining elements in the vector.
                for (IRIntegerValue c = colCount; c < getIntVal(vectorType->getElementCount()); c++)
                {
                    vecArgs.add(builder.emitDefaultConstruct(vectorType->getElementType()));
                }
                auto rowVector = builder.emitMakeVector(
                    vectorType,
                    (UInt)vecArgs.getCount(),
                    vecArgs.getBuffer());
                vectors.add(rowVector);
            }
        }

        auto vectorArray =
            builder.emitMakeArray(arrayType, (UInt)vectors.getCount(), vectors.getBuffer());
        auto result = builder.emitMakeStruct(structType, 1, &vectorArray);
        builder.emitStore(outParam, result);
        builder.emitReturn();
        return func;
    }

    IRFunc* createArrayUnpackFunc(
        IRArrayType* arrayType,
        IRStructType* structType,
        IRStructKey* dataKey,
        LoweredElementTypeInfo innerTypeInfo)
    {
        IRBuilder builder(structType);
        builder.setInsertAfter(structType);
        auto func = builder.createFunc();
        auto refStructType = builder.getRefType(structType, AddressSpace::Generic);
        auto funcType = builder.getFuncType(1, (IRType**)&refStructType, arrayType);
        func->setFullType(funcType);
        builder.addNameHintDecoration(func, UnownedStringSlice("unpackStorage"));
        builder.addForceInlineDecoration(func);
        builder.setInsertInto(func);
        builder.emitBlock();
        auto packedParam = builder.emitParam(refStructType);
        auto packedArray = builder.emitFieldAddress(packedParam, dataKey);
        auto count = getIntVal(arrayType->getElementCount());
        IRInst* result = nullptr;
        if (count <= kMaxArraySizeToUnroll)
        {
            // If the array is small enough, just process each element directly.
            List<IRInst*> args;
            args.setCount((Index)count);
            for (IRIntegerValue ii = 0; ii < count; ++ii)
            {
                auto packedElementAddr = builder.emitElementAddress(packedArray, ii);
                auto originalElement = innerTypeInfo.convertLoweredToOriginal.apply(
                    builder,
                    innerTypeInfo.originalType,
                    packedElementAddr);
                args[(Index)ii] = originalElement;
            }
            result = builder.emitMakeArray(arrayType, (UInt)args.getCount(), args.getBuffer());
        }
        else
        {
            // The general case for large arrays is to emit a loop through the elements.
            IRVar* resultVar = builder.emitVar(arrayType);
            IRBlock* loopBodyBlock;
            IRBlock* loopBreakBlock;
            auto loopParam = emitLoopBlocks(
                &builder,
                builder.getIntValue(builder.getIntType(), 0),
                builder.getIntValue(builder.getIntType(), count),
                loopBodyBlock,
                loopBreakBlock);

            builder.setInsertBefore(loopBodyBlock->getFirstOrdinaryInst());
            auto packedElementAddr = builder.emitElementAddress(packedArray, loopParam);
            auto originalElement = innerTypeInfo.convertLoweredToOriginal.apply(
                builder,
                innerTypeInfo.originalType,
                packedElementAddr);
            auto varPtr = builder.emitElementAddress(resultVar, loopParam);
            builder.emitStore(varPtr, originalElement);
            builder.setInsertInto(loopBreakBlock);
            result = builder.emitLoad(resultVar);
        }
        builder.emitReturn(result);
        return func;
    }

    IRFunc* createArrayPackFunc(
        IRArrayType* arrayType,
        IRStructType* structType,
        IRStructKey* arrayStructKey,
        LoweredElementTypeInfo innerTypeInfo)
    {
        IRBuilder builder(structType);
        builder.setInsertAfter(structType);
        auto func = builder.createFunc();
        auto outLoweredType = builder.getRefType(structType, AddressSpace::Generic);
        IRType* paramTypes[] = {outLoweredType, structType};
        auto funcType = builder.getFuncType(2, paramTypes, builder.getVoidType());
        func->setFullType(funcType);
        builder.addNameHintDecoration(func, UnownedStringSlice("packStorage"));
        builder.addForceInlineDecoration(func);
        builder.setInsertInto(func);
        builder.emitBlock();
        auto outParam = builder.emitParam(outLoweredType);
        auto originalParam = builder.emitParam(arrayType);
        auto count = getIntVal(arrayType->getElementCount());
        auto destArray = builder.emitFieldAddress(outParam, arrayStructKey);
        if (count <= kMaxArraySizeToUnroll)
        {
            // If the array is small enough, just process each element directly.
            List<IRInst*> args;
            args.setCount((Index)count);
            for (IRIntegerValue ii = 0; ii < count; ++ii)
            {
                auto originalElement = builder.emitElementExtract(originalParam, ii);
                auto destArrayElement = builder.emitElementAddress(destArray, ii);
                innerTypeInfo.convertOriginalToLowered.applyDestinationDriven(
                    builder,
                    destArrayElement,
                    originalElement);
            }
        }
        else
        {
            // The general case for large arrays is to emit a loop through the elements.
            IRBlock* loopBodyBlock;
            IRBlock* loopBreakBlock;
            auto loopParam = emitLoopBlocks(
                &builder,
                builder.getIntValue(builder.getIntType(), 0),
                builder.getIntValue(builder.getIntType(), count),
                loopBodyBlock,
                loopBreakBlock);

            builder.setInsertBefore(loopBodyBlock->getFirstOrdinaryInst());
            auto originalElement = builder.emitElementExtract(originalParam, loopParam);
            auto varPtr = builder.emitElementAddress(destArray, loopParam);
            innerTypeInfo.convertOriginalToLowered.applyDestinationDriven(
                builder,
                varPtr,
                originalElement);
            builder.setInsertInto(loopBreakBlock);
        }
        builder.emitReturn();
        return func;
    }

    const char* getLayoutName(IRTypeLayoutRuleName name)
    {
        switch (name)
        {
        case IRTypeLayoutRuleName::Std140:
            return "std140";
        case IRTypeLayoutRuleName::Std430:
            return "std430";
        case IRTypeLayoutRuleName::Natural:
            return "natural";
        default:
            return "default";
        }
    }

    // Returns the number of elements N that ensures the IRVectorType(elementType,N)
    // has 16-byte aligned size and N is no less than `minCount`.
    IRIntegerValue get16ByteAlignedVectorElementCount(IRType* elementType, IRIntegerValue minCount)
    {
        IRSizeAndAlignment sizeAlignment;
        getNaturalSizeAndAlignment(target->getOptionSet(), elementType, &sizeAlignment);
        if (sizeAlignment.size)
            return align(sizeAlignment.size * minCount, 16) / sizeAlignment.size;
        return 4;
    }

    bool shouldLowerMatrixType(IRMatrixType* matrixType, TypeLoweringConfig config)
    {
        // For spirv, we always want to lower all matrix types, because SPIRV does not support
        // specifying matrix layout/stride if the matrix type is used in places other than
        // defining a struct field. This means that if a matrix is used to define a varying
        // parameter, we always want to wrap it in a struct.
        //
        if (target->shouldEmitSPIRVDirectly())
        {
            return true;
        }

        if (getIntVal(matrixType->getLayout()) == defaultMatrixLayout &&
            config.layoutRule->ruleName == IRTypeLayoutRuleName::Natural)
        {
            // For other targets, we only lower the matrix types if they differ from the default
            // matrix layout.
            return false;
        }
        return true;
    }

    LoweredElementTypeInfo getLoweredTypeInfoImpl(IRType* type, TypeLoweringConfig config)
    {
        IRBuilder builder(type);
        builder.setInsertAfter(type);

        LoweredElementTypeInfo info;
        info.originalType = type;

        if (auto matrixType = as<IRMatrixType>(type))
        {
            if (!shouldLowerMatrixType(matrixType, config))
            {
                info.loweredType = type;
                return info;
            }

            auto loweredType = builder.createStructType();
            builder.addPhysicalTypeDecoration(loweredType);

            StringBuilder nameSB;
            bool isColMajor =
                getIntVal(matrixType->getLayout()) == SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;
            nameSB << "_MatrixStorage_";
            getTypeNameHint(nameSB, matrixType->getElementType());
            nameSB << getIntVal(matrixType->getRowCount()) << "x"
                   << getIntVal(matrixType->getColumnCount());
            if (isColMajor)
                nameSB << "_ColMajor";
            nameSB << getLayoutName(config.layoutRule->ruleName);
            builder.addNameHintDecoration(loweredType, nameSB.produceString().getUnownedSlice());
            auto structKey = builder.createStructKey();
            builder.addNameHintDecoration(structKey, UnownedStringSlice("data"));
            auto vectorSize = isColMajor ? matrixType->getRowCount() : matrixType->getColumnCount();
            if (config.layoutRule->ruleName == IRTypeLayoutRuleName::Std140 &&
                options.use16ByteArrayElementForConstantBuffer)
            {
                // For constant buffer layout, we need to use 16-byte aligned vector if
                // we are required to ensure array element types has 16-byte stride.
                vectorSize = builder.getIntValue(get16ByteAlignedVectorElementCount(
                    matrixType->getElementType(),
                    getIntVal(vectorSize)));
            }

            auto vectorType = builder.getVectorType(matrixType->getElementType(), vectorSize);
            IRSizeAndAlignment elementSizeAlignment;
            getSizeAndAlignment(
                target->getOptionSet(),
                config.layoutRule,
                vectorType,
                &elementSizeAlignment);
            elementSizeAlignment = config.layoutRule->alignCompositeElement(elementSizeAlignment);

            auto arrayType = builder.getArrayType(
                vectorType,
                isColMajor ? matrixType->getColumnCount() : matrixType->getRowCount(),
                builder.getIntValue(builder.getIntType(), elementSizeAlignment.getStride()));
            builder.createStructField(loweredType, structKey, arrayType);

            info.loweredType = loweredType;
            info.loweredInnerArrayType = arrayType;
            info.loweredInnerStructKey = structKey;
            info.convertLoweredToOriginal =
                createMatrixUnpackFunc(matrixType, loweredType, structKey);
            info.convertOriginalToLowered =
                createMatrixPackFunc(matrixType, loweredType, vectorType, arrayType);
            return info;
        }
        else if (auto arrayTypeBase = as<IRArrayTypeBase>(type))
        {
            auto loweredInnerTypeInfo = getLoweredTypeInfo(arrayTypeBase->getElementType(), config);

            if (config.layoutRule->ruleName == IRTypeLayoutRuleName::Std140 &&
                options.use16ByteArrayElementForConstantBuffer)
            {
                // For constant buffer layout, we need to use 16-byte-aligned vector if
                // we are required to ensure array element types has 16-byte stride.
                // We only need to handle the case where the element type is a scalar or vector
                // type here, because if the element type is a matrix type or struct type,
                // the size promotion will be handled during lowering of the element type.
                IRType* packedVectorType = nullptr;
                if (auto vectorType = as<IRVectorType>(loweredInnerTypeInfo.loweredType))
                {
                    packedVectorType = builder.getVectorType(
                        vectorType->getElementType(),
                        builder.getIntValue(get16ByteAlignedVectorElementCount(
                            vectorType->getElementType(),
                            getIntVal(vectorType->getElementCount()))));
                    if (packedVectorType != loweredInnerTypeInfo.originalType)
                    {
                        loweredInnerTypeInfo.convertLoweredToOriginal = kIROp_VectorReshape;
                        loweredInnerTypeInfo.convertOriginalToLowered = kIROp_VectorReshape;
                    }
                }
                else if (auto scalarType = as<IRBasicType>(loweredInnerTypeInfo.loweredType))
                {
                    packedVectorType = builder.getVectorType(
                        loweredInnerTypeInfo.loweredType,
                        get16ByteAlignedVectorElementCount(scalarType, 1));
                    loweredInnerTypeInfo.convertLoweredToOriginal = kIROp_VectorReshape;
                    loweredInnerTypeInfo.convertOriginalToLowered = kIROp_MakeVectorFromScalar;
                }
                if (packedVectorType)
                {
                    loweredInnerTypeInfo.loweredType = packedVectorType;
                    if (loweredInnerTypeInfo.convertLoweredToOriginal)
                        conversionMethodMap[ConversionMethodKey{
                            packedVectorType,
                            loweredInnerTypeInfo.originalType}] =
                            loweredInnerTypeInfo.convertOriginalToLowered;
                    if (loweredInnerTypeInfo.convertOriginalToLowered)
                        conversionMethodMap[ConversionMethodKey{
                            loweredInnerTypeInfo.originalType,
                            packedVectorType}] = loweredInnerTypeInfo.convertLoweredToOriginal;
                }
            }

            // For spirv backend, we always want to lower all array types for non-varying
            // parameters, even if the element type comes out the same. This is because different
            // layout rules may have different array stride requirements.
            if (!target->shouldEmitSPIRVDirectly() || config.addressSpace == AddressSpace::Input)
            {
                if (!loweredInnerTypeInfo.convertLoweredToOriginal)
                {
                    info.loweredType = type;
                    return info;
                }
            }

            auto arrayType = as<IRArrayType>(arrayTypeBase);
            if (arrayType)
            {
                auto loweredType = builder.createStructType();
                builder.addPhysicalTypeDecoration(loweredType);

                info.loweredType = loweredType;
                StringBuilder nameSB;
                nameSB << "_Array_" << getLayoutName(config.layoutRule->ruleName) << "_";
                getTypeNameHint(nameSB, arrayType->getElementType());
                nameSB << getIntVal(arrayType->getElementCount());
                builder.addNameHintDecoration(
                    loweredType,
                    nameSB.produceString().getUnownedSlice());
                auto structKey = builder.createStructKey();
                builder.addNameHintDecoration(structKey, UnownedStringSlice("data"));
                IRSizeAndAlignment elementSizeAlignment;
                getSizeAndAlignment(
                    target->getOptionSet(),
                    config.layoutRule,
                    loweredInnerTypeInfo.loweredType,
                    &elementSizeAlignment);
                elementSizeAlignment =
                    config.layoutRule->alignCompositeElement(elementSizeAlignment);
                auto innerArrayType = builder.getArrayType(
                    loweredInnerTypeInfo.loweredType,
                    arrayType->getElementCount(),
                    builder.getIntValue(builder.getIntType(), elementSizeAlignment.getStride()));
                builder.createStructField(loweredType, structKey, innerArrayType);
                info.loweredInnerArrayType = innerArrayType;
                info.loweredInnerStructKey = structKey;
                info.convertLoweredToOriginal =
                    createArrayUnpackFunc(arrayType, loweredType, structKey, loweredInnerTypeInfo);
                info.convertOriginalToLowered =
                    createArrayPackFunc(arrayType, loweredType, structKey, loweredInnerTypeInfo);
            }
            else
            {
                IRSizeAndAlignment elementSizeAlignment;
                getSizeAndAlignment(
                    target->getOptionSet(),
                    config.layoutRule,
                    loweredInnerTypeInfo.loweredType,
                    &elementSizeAlignment);
                elementSizeAlignment =
                    config.layoutRule->alignCompositeElement(elementSizeAlignment);
                auto innerArrayType = builder.getArrayTypeBase(
                    arrayTypeBase->getOp(),
                    loweredInnerTypeInfo.loweredType,
                    nullptr,
                    builder.getIntValue(builder.getIntType(), elementSizeAlignment.getStride()));
                info.loweredType = innerArrayType;
            }
            return info;
        }
        else if (auto structType = as<IRStructType>(type))
        {
            List<LoweredElementTypeInfo> fieldLoweredTypeInfo;
            bool isTrivial = true;
            for (auto field : structType->getFields())
            {
                auto loweredFieldTypeInfo = getLoweredTypeInfo(field->getFieldType(), config);
                fieldLoweredTypeInfo.add(loweredFieldTypeInfo);
                if (loweredFieldTypeInfo.convertLoweredToOriginal ||
                    config.layoutRule->ruleName != IRTypeLayoutRuleName::Natural)
                    isTrivial = false;
            }

            // For spirv backend, we always want to lower all array types, even if the element type
            // comes out the same. This is because different layout rules may have different array
            // stride requirements.
            if (!target->shouldEmitSPIRVDirectly())
            {
                // For non-spirv target, we skip lowering this type if all field types are
                // unchanged.
                if (isTrivial)
                {
                    info.loweredType = type;
                    return info;
                }
            }
            auto loweredType = builder.createStructType();
            builder.addPhysicalTypeDecoration(loweredType);

            StringBuilder nameSB;
            getTypeNameHint(nameSB, type);
            nameSB << "_" << getLayoutName(config.layoutRule->ruleName);
            builder.addNameHintDecoration(loweredType, nameSB.produceString().getUnownedSlice());
            info.loweredType = loweredType;
            // Create fields.
            {
                Index fieldId = 0;
                for (auto field : structType->getFields())
                {
                    auto& loweredFieldTypeInfo = fieldLoweredTypeInfo[fieldId];
                    // When lowering type for user pointer, skip fields that are unsized array.
                    if (config.addressSpace == AddressSpace::UserPointer &&
                        as<IRUnsizedArrayType>(loweredFieldTypeInfo.loweredType))
                    {
                        fieldId++;
                        loweredFieldTypeInfo.loweredType = builder.getVoidType();
                        continue;
                    }
                    builder.createStructField(
                        loweredType,
                        field->getKey(),
                        loweredFieldTypeInfo.loweredType);
                    fieldId++;
                }
            }

            // Create unpack func.
            {
                builder.setInsertAfter(loweredType);
                info.convertLoweredToOriginal = builder.createFunc();
                builder.setInsertInto(info.convertLoweredToOriginal.func);
                builder.addNameHintDecoration(
                    info.convertLoweredToOriginal.func,
                    UnownedStringSlice("unpackStorage"));
                builder.addForceInlineDecoration(info.convertLoweredToOriginal.func);
                auto refLoweredType = builder.getRefType(loweredType, AddressSpace::Generic);
                info.convertLoweredToOriginal.func->setFullType(
                    builder.getFuncType(1, (IRType**)&refLoweredType, type));
                builder.emitBlock();
                auto loweredParam = builder.emitParam(refLoweredType);
                List<IRInst*> args;
                Index fieldId = 0;
                for (auto field : structType->getFields())
                {
                    if (as<IRVoidType>(fieldLoweredTypeInfo[fieldId].loweredType))
                    {
                        fieldId++;
                        continue;
                    }
                    auto storageField = builder.emitFieldAddress(loweredParam, field->getKey());
                    auto unpackedField =
                        fieldLoweredTypeInfo[fieldId].convertLoweredToOriginal.apply(
                            builder,
                            field->getFieldType(),
                            storageField);
                    args.add(unpackedField);
                    fieldId++;
                }
                auto result = builder.emitMakeStruct(type, args);
                builder.emitReturn(result);
            }

            // Create pack func.
            {
                builder.setInsertAfter(info.convertLoweredToOriginal.func);
                info.convertOriginalToLowered = builder.createFunc();
                builder.setInsertInto(info.convertOriginalToLowered.func);
                builder.addNameHintDecoration(
                    info.convertOriginalToLowered.func,
                    UnownedStringSlice("packStorage"));
                builder.addForceInlineDecoration(info.convertOriginalToLowered.func);

                auto outLoweredType = builder.getRefType(loweredType, AddressSpace::Generic);
                IRType* paramTypes[] = {outLoweredType, type};
                info.convertOriginalToLowered.func->setFullType(
                    builder.getFuncType(2, paramTypes, builder.getVoidType()));
                builder.emitBlock();
                auto outParam = builder.emitParam(outLoweredType);
                auto param = builder.emitParam(type);
                List<IRInst*> args;
                Index fieldId = 0;
                for (auto field : structType->getFields())
                {
                    if (as<IRVoidType>(fieldLoweredTypeInfo[fieldId].loweredType))
                    {
                        fieldId++;
                        continue;
                    }
                    auto fieldVal =
                        builder.emitFieldExtract(field->getFieldType(), param, field->getKey());
                    auto destAddr = builder.emitFieldAddress(outParam, field->getKey());

                    fieldLoweredTypeInfo[fieldId].convertOriginalToLowered.applyDestinationDriven(
                        builder,
                        destAddr,
                        fieldVal);
                    fieldId++;
                }
                builder.emitReturn();
            }

            return info;
        }

        if (target->shouldEmitSPIRVDirectly())
        {
            switch (target->getTargetReq()->getTarget())
            {
            case CodeGenTarget::SPIRV:
            case CodeGenTarget::SPIRVAssembly:
                {
                    auto scalarType = type;
                    auto vectorType = as<IRVectorType>(scalarType);
                    if (vectorType)
                        scalarType = vectorType->getElementType();

                    if (as<IRBoolType>(scalarType))
                    {
                        // Bool is an abstract type in SPIRV, so we need to lower them into an int.
                        info.loweredType = builder.getIntType();
                        if (vectorType)
                            info.loweredType = builder.getVectorType(
                                info.loweredType,
                                vectorType->getElementCount());
                        info.convertLoweredToOriginal = kIROp_BuiltinCast;
                        info.convertOriginalToLowered = kIROp_BuiltinCast;
                        return info;
                    }
                }
            default:
                break;
            }
        }

        info.loweredType = type;
        return info;
    }

    LoweredTypeMap& getTypeLoweringMap(TypeLoweringConfig config)
    {
        RefPtr<LoweredTypeMap> map;
        if (loweredTypeInfoMaps.tryGetValue(config, map))
            return *map;
        map = new LoweredTypeMap();
        loweredTypeInfoMaps.add(config, map);
        return *map;
    }

    LoweredElementTypeInfo getLoweredTypeInfo(IRType* type, TypeLoweringConfig config)
    {
        // If `type` is already a lowered type, no more lowering is required.
        LoweredElementTypeInfo info;
        auto& map = getTypeLoweringMap(config);
        auto& mapLoweredTypeToInfo = map.mapLoweredTypeToInfo;
        auto& loweredTypeInfo = map.loweredTypeInfo;
        if (mapLoweredTypeToInfo.tryGetValue(type))
        {
            info.originalType = type;
            info.loweredType = type;
            return info;
        }
        if (loweredTypeInfo.tryGetValue(type, info))
            return info;
        info = getLoweredTypeInfoImpl(type, config);
        IRSizeAndAlignment sizeAlignment;
        getSizeAndAlignment(
            target->getOptionSet(),
            config.layoutRule,
            info.loweredType,
            &sizeAlignment);
        loweredTypeInfo.set(type, info);
        mapLoweredTypeToInfo.set(info.loweredType, info);
        conversionMethodMap[{info.originalType, info.loweredType}] = info.convertLoweredToOriginal;
        conversionMethodMap[{info.loweredType, info.originalType}] = info.convertOriginalToLowered;
        return info;
    }

    IRType* getLoweredPtrLikeType(IRType* originalPtrLikeType, IRType* newElementType)
    {
        if (as<IRPointerLikeType>(originalPtrLikeType) || as<IRPtrTypeBase>(originalPtrLikeType) ||
            as<IRHLSLStructuredBufferTypeBase>(originalPtrLikeType) ||
            as<IRGLSLShaderStorageBufferType>(originalPtrLikeType))
        {
            IRBuilder builder(newElementType);
            builder.setInsertAfter(newElementType);
            ShortList<IRInst*> operands;
            for (UInt i = 0; i < originalPtrLikeType->getOperandCount(); i++)
                operands.add(originalPtrLikeType->getOperand(i));
            operands[0] = newElementType;
            return builder.getType(
                originalPtrLikeType->getOp(),
                (UInt)operands.getCount(),
                operands.getArrayView().getBuffer());
        }
        SLANG_UNREACHABLE("unhandled ptr like or buffer type");
    }

    IRInst* getStoreVal(IRInst* storeInst)
    {
        if (auto store = as<IRStore>(storeInst))
            return store->getVal();
        else if (auto sbStore = as<IRRWStructuredBufferStore>(storeInst))
            return sbStore->getVal();
        return nullptr;
    }

    struct MatrixAddrWorkItem
    {
        IRInst* matrixAddrInst;
        TypeLoweringConfig config;
    };

    IRInst* getBufferAddr(IRBuilder& builder, IRInst* loadStoreInst)
    {
        switch (loadStoreInst->getOp())
        {
        case kIROp_Load:
        case kIROp_Store:
            return loadStoreInst->getOperand(0);
        case kIROp_StructuredBufferLoad:
        case kIROp_StructuredBufferLoadStatus:
        case kIROp_RWStructuredBufferLoad:
        case kIROp_RWStructuredBufferLoadStatus:
        case kIROp_RWStructuredBufferStore:
            return builder.emitRWStructuredBufferGetElementPtr(
                loadStoreInst->getOperand(0),
                loadStoreInst->getOperand(1));
        default:
            return nullptr;
        }
    }

    void processModule(IRModule* module)
    {
        IRBuilder builder(module);
        struct BufferTypeInfo
        {
            IRType* bufferType;
            IRType* elementType;
            bool shouldWrapArrayInStruct = false;
        };
        List<BufferTypeInfo> bufferTypeInsts;
        for (auto globalInst : module->getGlobalInsts())
        {
            IRType* elementType = nullptr;

            if (auto ptrType = as<IRPtrTypeBase>(globalInst))
            {
                switch (ptrType->getAddressSpace())
                {
                case AddressSpace::UserPointer:
                    if (!options.lowerBufferPointer)
                        continue;
                    [[fallthrough]];
                case AddressSpace::Input:
                case AddressSpace::Output:
                    elementType = ptrType->getValueType();
                    break;
                }
            }
            if (auto structBuffer = as<IRHLSLStructuredBufferTypeBase>(globalInst))
                elementType = structBuffer->getElementType();
            else if (auto constBuffer = as<IRUniformParameterGroupType>(globalInst))
                elementType = constBuffer->getElementType();
            else if (auto storageBuffer = as<IRGLSLShaderStorageBufferType>(globalInst))
                elementType = storageBuffer->getElementType();
            if (as<IRTextureBufferType>(globalInst))
                continue;
            if (!as<IRStructType>(elementType) && !as<IRMatrixType>(elementType) &&
                !as<IRArrayType>(elementType) && !as<IRBoolType>(elementType))
                continue;
            bufferTypeInsts.add(BufferTypeInfo{(IRType*)globalInst, elementType});
        }

        // Maintain a pending work list of all matrix addresses, and try to lower them out of
        // existance after everything else has been lowered.

        List<MatrixAddrWorkItem> matrixAddrInsts;

        for (auto bufferTypeInfo : bufferTypeInsts)
        {
            auto bufferType = bufferTypeInfo.bufferType;
            auto elementType = bufferTypeInfo.elementType;

            if (elementType->findDecoration<IRPhysicalTypeDecoration>())
                continue;

            auto config = getTypeLoweringConfigForBuffer(target, bufferType);
            auto loweredBufferElementTypeInfo = getLoweredTypeInfo(elementType, config);

            // If the lowered type is the same as original type, no change is required.
            if (loweredBufferElementTypeInfo.loweredType ==
                loweredBufferElementTypeInfo.originalType)
                continue;

            builder.setInsertBefore(bufferType);

            ShortList<IRInst*> typeOperands;
            for (UInt i = 0; i < bufferType->getOperandCount(); i++)
                typeOperands.add(bufferType->getOperand(i));
            typeOperands[0] = loweredBufferElementTypeInfo.loweredType;
            auto loweredBufferType = builder.getType(
                bufferType->getOp(),
                (UInt)typeOperands.getCount(),
                typeOperands.getArrayView().getBuffer());

            // We treat a value of a buffer type as a pointer, and use a work list to translate
            // all loads and stores through the pointer values that needs lowering.

            List<IRInst*> ptrValsWorkList;
            traverseUses(
                bufferType,
                [&](IRUse* use)
                {
                    auto user = use->getUser();
                    if (use != &user->typeUse)
                        return;
                    ptrValsWorkList.add(use->getUser());
                });

            // Translate the values to use new lowered buffer type instead.
            for (Index i = 0; i < ptrValsWorkList.getCount(); i++)
            {
                auto ptrVal = ptrValsWorkList[i];
                auto oldPtrType = ptrVal->getFullType();
                auto originalElementType = oldPtrType->getOperand(0);

                // If we are accessing an unsized array element from a pointer, we need to compute
                // the trailing ptr that points to the first element of the array.
                // And then replace all getElementPtr(arrayPtr, index) with
                // getOffsetPtr(trailingPtr, index).
                if (auto fieldAddr = as<IRFieldAddress>(ptrVal))
                {
                    auto handleUnsizedArrayAccess = [&]() -> bool
                    {
                        auto ptrType = as<IRPtrType>(ptrVal->getDataType());
                        if (!ptrType)
                            return false;
                        if (ptrType->getAddressSpace() != AddressSpace::UserPointer)
                            return false;
                        if (auto unsizedArrayType = as<IRUnsizedArrayType>(ptrType->getValueType()))
                        {
                            builder.setInsertBefore(ptrVal);
                            auto newArrayPtrVal = fieldAddr->getBase();
                            auto loweredInnerType =
                                getLoweredTypeInfo(unsizedArrayType->getElementType(), config);

                            IRSizeAndAlignment arrayElementSizeAlignment;
                            getSizeAndAlignment(
                                target->getOptionSet(),
                                config.layoutRule,
                                loweredInnerType.loweredType,
                                &arrayElementSizeAlignment);
                            IRSizeAndAlignment baseSizeAlignment;
                            getSizeAndAlignment(
                                target->getOptionSet(),
                                config.layoutRule,
                                tryGetPointedToType(&builder, fieldAddr->getBase()->getDataType()),
                                &baseSizeAlignment);

                            // Convert pointer to uint64 and adjust offset.
                            IRIntegerValue offset = baseSizeAlignment.size;
                            offset = align(offset, arrayElementSizeAlignment.alignment);
                            if (offset != 0)
                            {
                                auto rawPtr =
                                    builder.emitBitCast(builder.getUInt64Type(), newArrayPtrVal);
                                newArrayPtrVal = builder.emitAdd(
                                    rawPtr->getFullType(),
                                    rawPtr,
                                    builder.getIntValue(builder.getUInt64Type(), offset));
                            }
                            newArrayPtrVal = builder.emitBitCast(
                                builder.getPtrType(
                                    loweredInnerType.loweredType,
                                    ptrType->getAddressSpace()),
                                newArrayPtrVal);
                            traverseUses(
                                ptrVal,
                                [&](IRUse* use)
                                {
                                    auto user = use->getUser();
                                    if (user->getOp() == kIROp_GetElementPtr)
                                    {
                                        builder.setInsertBefore(user);
                                        auto newElementPtr = builder.emitGetOffsetPtr(
                                            newArrayPtrVal,
                                            user->getOperand(1));
                                        user->replaceUsesWith(newElementPtr);
                                        user->removeAndDeallocate();
                                        ptrValsWorkList.add(newElementPtr);
                                    }
                                    else if (user->getOp() == kIROp_GetOffsetPtr)
                                    {
                                    }
                                    else
                                    {
                                        SLANG_UNEXPECTED(
                                            "unknown use of pointer to unsized array.");
                                    }
                                });
                            SLANG_ASSERT(!ptrVal->hasUses());
                            ptrVal->removeAndDeallocate();
                            return true;
                        }
                        return false;
                    };
                    if (handleUnsizedArrayAccess())
                        continue;
                }

                LoweredElementTypeInfo loweredElementTypeInfo = {};
                if (auto getElementPtr = as<IRGetElementPtr>(ptrVal))
                {
                    if (auto arrayType = as<IRArrayTypeBase>(
                            tryGetPointedToType(&builder, getElementPtr->getBase()->getDataType())))
                    {
                        // For WGSL, an array of scalar or vector type will always be converted to
                        // an array of 16-byte aligned vector type. In this case, we will run into a
                        // GetElementPtr where the result type is different from the element type of
                        // the base array.
                        // We should setup loweredElementTypeInfo so the remaining logic can handle
                        // this case and insert proper packing/unpacking logic around it.
                        if (arrayType->getElementType() != originalElementType &&
                            isScalarOrVectorType(originalElementType))
                        {
                            loweredElementTypeInfo.loweredType = arrayType->getElementType();
                            loweredElementTypeInfo.originalType = (IRType*)originalElementType;
                            loweredElementTypeInfo.convertLoweredToOriginal = getConversionMethod(
                                loweredElementTypeInfo.originalType,
                                loweredElementTypeInfo.loweredType);
                            loweredElementTypeInfo.convertOriginalToLowered = getConversionMethod(
                                loweredElementTypeInfo.loweredType,
                                loweredElementTypeInfo.originalType);
                        }
                    }
                }

                // For general cases we simply check if the element type needs lowering.
                // If so we will insert packing/unpacking logic if necessary.
                //
                if (!loweredElementTypeInfo.loweredType)
                {
                    loweredElementTypeInfo =
                        getLoweredTypeInfo((IRType*)originalElementType, config);
                }

                if (loweredElementTypeInfo.loweredType == loweredElementTypeInfo.originalType)
                    continue;

                ptrVal->setFullType(getLoweredPtrLikeType(
                    ptrVal->getFullType(),
                    loweredElementTypeInfo.loweredType));

                traverseUses(
                    ptrVal,
                    [&](IRUse* use)
                    {
                        auto user = use->getUser();
                        if (as<IRDecoration>(user))
                            return;
                        switch (user->getOp())
                        {
                        case kIROp_Load:
                        case kIROp_StructuredBufferLoad:
                        case kIROp_StructuredBufferLoadStatus:
                        case kIROp_RWStructuredBufferLoad:
                        case kIROp_RWStructuredBufferLoadStatus:
                        case kIROp_StructuredBufferConsume:
                            {
                                builder.setInsertBefore(user);
                                auto addr = getBufferAddr(builder, user);
                                if (!addr)
                                {
                                    IRCloneEnv cloneEnv = {};
                                    builder.setInsertBefore(user);
                                    auto newLoad = cloneInst(&cloneEnv, &builder, user);
                                    newLoad->setFullType(loweredElementTypeInfo.loweredType);
                                    addr = builder.emitVar(loweredElementTypeInfo.loweredType);
                                    builder.emitStore(addr, newLoad);
                                }
                                if (auto alignedAttr = user->findAttr<IRAlignedAttr>())
                                {
                                    builder.addAlignedAddressDecoration(
                                        addr,
                                        alignedAttr->getAlignment());
                                }
                                auto unpackedVal =
                                    loweredElementTypeInfo.convertLoweredToOriginal.apply(
                                        builder,
                                        loweredElementTypeInfo.originalType,
                                        addr);
                                user->replaceUsesWith(unpackedVal);
                                user->removeAndDeallocate();
                                break;
                            }
                        case kIROp_Store:
                        case kIROp_RWStructuredBufferStore:
                        case kIROp_StructuredBufferAppend:
                            {
                                // Use must be the dest operand of the store inst.
                                if (use != user->getOperands() + 0)
                                    break;
                                IRCloneEnv cloneEnv = {};
                                builder.setInsertBefore(user);
                                auto originalVal = getStoreVal(user);
                                IRInst* addr = getBufferAddr(builder, user);
                                if (addr)
                                {
                                    if (auto alignedAttr = user->findAttr<IRAlignedAttr>())
                                    {
                                        builder.addAlignedAddressDecoration(
                                            addr,
                                            alignedAttr->getAlignment());
                                    }

                                    loweredElementTypeInfo.convertOriginalToLowered
                                        .applyDestinationDriven(builder, addr, originalVal);
                                    user->removeAndDeallocate();
                                }
                                else if (auto sbAppend = as<IRStructuredBufferAppend>(user))
                                {
                                    builder.setInsertBefore(sbAppend);
                                    addr = builder.emitVar(loweredElementTypeInfo.loweredType);
                                    loweredElementTypeInfo.convertOriginalToLowered
                                        .applyDestinationDriven(builder, addr, originalVal);
                                    auto packedVal = builder.emitLoad(addr);
                                    sbAppend->setOperand(1, packedVal);
                                }
                                else
                                {
                                    SLANG_UNREACHABLE("unhandled store type");
                                }
                                break;
                            }
                        case kIROp_GetElementPtr:
                        case kIROp_FieldAddress:
                            {
                                // If original type is an array, the lowered type will be a struct.
                                // In that case, all existing address insts should be appended with
                                // a field extract.
                                if (as<IRArrayType>(originalElementType))
                                {
                                    builder.setInsertBefore(user);
                                    List<IRInst*> args;
                                    for (UInt i = 0; i < user->getOperandCount(); i++)
                                        args.add(user->getOperand(i));
                                    auto newArrayPtrVal = builder.emitFieldAddress(
                                        builder.getPtrType(
                                            loweredElementTypeInfo.loweredInnerArrayType),
                                        ptrVal,
                                        loweredElementTypeInfo.loweredInnerStructKey);
                                    builder.replaceOperand(use, newArrayPtrVal);
                                    ptrValsWorkList.add(user);
                                }
                                else if (as<IRMatrixType>(originalElementType))
                                {
                                    // We are tring to get a pointer to a lowered matrix element.
                                    // We process this insts at a later phase.
                                    SLANG_ASSERT(user->getOp() == kIROp_GetElementPtr);
                                    matrixAddrInsts.add(MatrixAddrWorkItem{user, config});
                                }
                                else
                                {
                                    // If we getting a derived address from the pointer, we need
                                    // to recursively lower the new address. We do so by pushing
                                    // the address inst into the work list.
                                    ptrValsWorkList.add(user);
                                }
                            }
                            break;
                        case kIROp_RWStructuredBufferGetElementPtr:
                        case kIROp_GetOffsetPtr:
                            ptrValsWorkList.add(user);
                            break;
                        case kIROp_StructuredBufferGetDimensions:
                            break;
                        case kIROp_Call:
                            {
                                // If a structured buffer or pointer typed value is used directly as
                                // an argument, we don't need to do any marshalling here.
                                if (as<IRHLSLStructuredBufferTypeBase>(ptrVal->getDataType()))
                                    break;
                                if (options.lowerBufferPointer &&
                                    as<IRPtrType>(ptrVal->getDataType()))
                                    break;
                                // If we are calling a function with an l-value pointer from buffer
                                // access, we need to materialize the object as a local variable,
                                // and pass the address of the local variable to the function.
                                builder.setInsertBefore(user);
                                auto unpackedVal =
                                    loweredElementTypeInfo.convertLoweredToOriginal.apply(
                                        builder,
                                        (IRType*)originalElementType,
                                        ptrVal);
                                auto var = builder.emitVar((IRType*)originalElementType);
                                builder.emitStore(var, unpackedVal);
                                use->set(var);
                                builder.setInsertAfter(user);
                                auto newVal = builder.emitLoad(var);
                                loweredElementTypeInfo.convertOriginalToLowered
                                    .applyDestinationDriven(builder, ptrVal, newVal);
                            }
                            break;
                        default:
                            break;
                        }
                    });
            }

            // Replace all remaining uses of bufferType to loweredBufferType, these uses are
            // non-operational and should be directly replaceable, such as uses in `IRFuncType`.
            bufferType->replaceUsesWith(loweredBufferType);
            bufferType->removeAndDeallocate();
        }

        // Process all matrix address uses.
        lowerMatrixAddresses(module, matrixAddrInsts);
    }

    // Lower all getElementPtr insts of a lowered matrix out of existance.
    void lowerMatrixAddresses(IRModule* module, List<MatrixAddrWorkItem>& matrixAddrInsts)
    {
        IRBuilder builder(module);
        for (auto workItem : matrixAddrInsts)
        {
            auto majorAddr = workItem.matrixAddrInst;
            auto majorGEP = as<IRGetElementPtr>(majorAddr);
            SLANG_ASSERT(majorGEP);
            auto loweredMatrixType =
                cast<IRPtrTypeBase>(majorGEP->getBase()->getFullType())->getValueType();
            auto matrixTypeInfo = getTypeLoweringMap(workItem.config)
                                      .mapLoweredTypeToInfo.tryGetValue(loweredMatrixType);
            SLANG_ASSERT(matrixTypeInfo);
            auto matrixType = as<IRMatrixType>(matrixTypeInfo->originalType);
            auto rowCount = getIntVal(matrixType->getRowCount());
            traverseUses(
                majorAddr,
                [&](IRUse* use)
                {
                    auto user = use->getUser();
                    builder.setInsertBefore(user);
                    switch (user->getOp())
                    {
                    case kIROp_Load:
                        {
                            IRInst* resultInst = nullptr;
                            auto dataPtr = builder.emitFieldAddress(
                                getLoweredPtrLikeType(
                                    majorAddr->getDataType(),
                                    matrixTypeInfo->loweredInnerArrayType),
                                majorGEP->getBase(),
                                matrixTypeInfo->loweredInnerStructKey);
                            if (getIntVal(matrixType->getLayout()) ==
                                SLANG_MATRIX_LAYOUT_COLUMN_MAJOR)
                            {
                                List<IRInst*> args;
                                for (IRIntegerValue i = 0; i < rowCount; i++)
                                {
                                    auto vector =
                                        builder.emitLoad(builder.emitElementAddress(dataPtr, i));
                                    auto element =
                                        builder.emitElementExtract(vector, majorGEP->getIndex());
                                    args.add(element);
                                }
                                resultInst = builder.emitMakeVector(
                                    builder.getVectorType(
                                        matrixType->getElementType(),
                                        (IRIntegerValue)args.getCount()),
                                    args);
                            }
                            else
                            {
                                auto element =
                                    builder.emitElementAddress(dataPtr, majorGEP->getIndex());
                                resultInst = builder.emitLoad(element);
                            }
                            user->replaceUsesWith(resultInst);
                            user->removeAndDeallocate();
                        }
                        break;
                    case kIROp_Store:
                        {
                            auto storeInst = cast<IRStore>(user);
                            if (storeInst->getOperand(0) != majorAddr)
                                break;
                            auto dataPtr = builder.emitFieldAddress(
                                getLoweredPtrLikeType(
                                    majorAddr->getDataType(),
                                    matrixTypeInfo->loweredInnerArrayType),
                                majorGEP->getBase(),
                                matrixTypeInfo->loweredInnerStructKey);
                            if (getIntVal(matrixType->getLayout()) ==
                                SLANG_MATRIX_LAYOUT_COLUMN_MAJOR)
                            {
                                for (IRIntegerValue i = 0; i < rowCount; i++)
                                {
                                    auto vectorAddr = builder.emitElementAddress(dataPtr, i);
                                    auto elementAddr = builder.emitElementAddress(
                                        vectorAddr,
                                        majorGEP->getIndex());
                                    builder.emitStore(
                                        elementAddr,
                                        builder.emitElementExtract(storeInst->getVal(), i));
                                }
                            }
                            else
                            {
                                auto rowAddr =
                                    builder.emitElementAddress(dataPtr, majorGEP->getIndex());
                                builder.emitStore(rowAddr, storeInst->getVal());
                                user->removeAndDeallocate();
                            }
                            break;
                        }
                    case kIROp_GetElementPtr:
                        {
                            auto gep2 = cast<IRGetElementPtr>(user);
                            auto rowIndex = majorGEP->getIndex();
                            auto colIndex = gep2->getIndex();
                            if (getIntVal(matrixType->getLayout()) ==
                                SLANG_MATRIX_LAYOUT_COLUMN_MAJOR)
                            {
                                Swap(rowIndex, colIndex);
                            }
                            auto dataPtr = builder.emitFieldAddress(
                                getLoweredPtrLikeType(
                                    majorAddr->getDataType(),
                                    matrixTypeInfo->loweredInnerArrayType),
                                majorGEP->getBase(),
                                matrixTypeInfo->loweredInnerStructKey);
                            auto vectorAddr = builder.emitElementAddress(dataPtr, rowIndex);
                            auto elementAddr = builder.emitElementAddress(vectorAddr, colIndex);
                            gep2->replaceUsesWith(elementAddr);
                            gep2->removeAndDeallocate();
                            break;
                        }
                    default:
                        SLANG_UNREACHABLE("unhandled inst of a matrix address inst that needs "
                                          "storage lowering.");
                        break;
                    }
                });
        }
    }
};

void lowerBufferElementTypeToStorageType(
    TargetProgram* target,
    IRModule* module,
    BufferElementTypeLoweringOptions options)
{
    SlangMatrixLayoutMode defaultMatrixMode =
        (SlangMatrixLayoutMode)target->getOptionSet().getMatrixLayoutMode();
    if ((isCPUTarget(target->getTargetReq()) || isCUDATarget(target->getTargetReq()) ||
         isMetalTarget(target->getTargetReq())))
        defaultMatrixMode = SLANG_MATRIX_LAYOUT_ROW_MAJOR;
    else if (defaultMatrixMode == SLANG_MATRIX_LAYOUT_MODE_UNKNOWN)
        defaultMatrixMode = SLANG_MATRIX_LAYOUT_ROW_MAJOR;
    LoweredElementTypeContext context(target, options, defaultMatrixMode);
    context.processModule(module);
}

IRTypeLayoutRules* getTypeLayoutRulesFromOp(IROp layoutTypeOp, IRTypeLayoutRules* defaultLayout)
{
    switch (layoutTypeOp)
    {
    case kIROp_DefaultBufferLayoutType:
        return defaultLayout;
    case kIROp_Std140BufferLayoutType:
        return IRTypeLayoutRules::getStd140();
    case kIROp_Std430BufferLayoutType:
        return IRTypeLayoutRules::getStd430();
    case kIROp_ScalarBufferLayoutType:
        return IRTypeLayoutRules::getNatural();
    }
    return defaultLayout;
}

IRTypeLayoutRules* getTypeLayoutRuleForBuffer(TargetProgram* target, IRType* bufferType)
{
    if (target->getTargetReq()->getTarget() != CodeGenTarget::WGSL)
    {
        if (!isKhronosTarget(target->getTargetReq()))
            return IRTypeLayoutRules::getNatural();

        // If we are just emitting GLSL, we can just use the general layout rule.
        if (!target->shouldEmitSPIRVDirectly())
            return IRTypeLayoutRules::getNatural();

        // If the user specified a scalar buffer layout, then just use that.
        if (target->getOptionSet().shouldUseScalarLayout())
            return IRTypeLayoutRules::getNatural();
    }

    if (target->getOptionSet().shouldUseDXLayout())
    {
        if (as<IRUniformParameterGroupType>(bufferType))
        {
            return IRTypeLayoutRules::getConstantBuffer();
        }
        else
            return IRTypeLayoutRules::getNatural();
    }

    // The default behavior is to use std140 for constant buffers and std430 for other buffers.
    switch (bufferType->getOp())
    {
    case kIROp_HLSLStructuredBufferType:
    case kIROp_HLSLRWStructuredBufferType:
    case kIROp_HLSLAppendStructuredBufferType:
    case kIROp_HLSLConsumeStructuredBufferType:
    case kIROp_HLSLRasterizerOrderedStructuredBufferType:
        {
            auto structBufferType = as<IRHLSLStructuredBufferTypeBase>(bufferType);
            auto layoutTypeOp = structBufferType->getDataLayout()
                                    ? structBufferType->getDataLayout()->getOp()
                                    : kIROp_DefaultBufferLayoutType;
            return getTypeLayoutRulesFromOp(layoutTypeOp, IRTypeLayoutRules::getStd430());
        }
    case kIROp_ConstantBufferType:
    case kIROp_ParameterBlockType:
        {
            auto parameterGroupType = as<IRUniformParameterGroupType>(bufferType);

            auto layoutTypeOp = parameterGroupType->getDataLayout()
                                    ? parameterGroupType->getDataLayout()->getOp()
                                    : kIROp_DefaultBufferLayoutType;
            return getTypeLayoutRulesFromOp(layoutTypeOp, IRTypeLayoutRules::getStd140());
        }
    case kIROp_GLSLShaderStorageBufferType:
        {
            auto storageBufferType = as<IRGLSLShaderStorageBufferType>(bufferType);
            auto layoutTypeOp = storageBufferType->getDataLayout()
                                    ? storageBufferType->getDataLayout()->getOp()
                                    : kIROp_Std430BufferLayoutType;
            return getTypeLayoutRulesFromOp(layoutTypeOp, IRTypeLayoutRules::getStd430());
        }
    case kIROp_PtrType:
        return IRTypeLayoutRules::getNatural();
    }
    return IRTypeLayoutRules::getNatural();
}

TypeLoweringConfig getTypeLoweringConfigForBuffer(TargetProgram* target, IRType* bufferType)
{
    AddressSpace addrSpace = AddressSpace::Generic;
    if (auto ptrType = as<IRPtrTypeBase>(bufferType))
    {
        switch (ptrType->getAddressSpace())
        {
        case AddressSpace::Input:
        case AddressSpace::Output:
            addrSpace = AddressSpace::Input;
            break;
        case AddressSpace::UserPointer:
            addrSpace = AddressSpace::UserPointer;
            break;
        }
    }
    auto rules = getTypeLayoutRuleForBuffer(target, bufferType);
    return TypeLoweringConfig{addrSpace, rules};
}

} // namespace Slang
