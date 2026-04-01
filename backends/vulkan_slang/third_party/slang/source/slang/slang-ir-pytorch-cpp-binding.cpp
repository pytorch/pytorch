#include "slang-ir-pytorch-cpp-binding.h"

#include "slang-diagnostics.h"
#include "slang-ir-autodiff.h"
#include "slang-ir-insts.h"
#include "slang-ir-lower-cuda-builtin-types.h"
#include "slang-ir.h"

namespace Slang
{
// Convert a type to a target tuple type.
static IRType* translateToTupleType(IRBuilder& builder, IRType* type)
{
    if (as<IRVoidType>(type))
        return type;
    else if (as<IRBasicType>(type))
        return type;
    else if (as<IRTorchTensorType>(type))
        return type;
    else if (auto matrixType = as<IRMatrixType>(type))
    {
        auto rowCount = as<IRIntLit>(matrixType->getRowCount());
        auto colCount = as<IRIntLit>(matrixType->getColumnCount());
        if (!rowCount || !colCount)
        {
            return nullptr;
        }
        List<IRType*> elementTypes;
        for (IRIntegerValue i = 0; i < rowCount->getValue(); i++)
        {
            elementTypes.addRange(matrixType->getElementType());
        }
        auto elementTupleType =
            builder.getTargetTupleType((UInt)elementTypes.getCount(), elementTypes.getBuffer());
        List<IRType*> rowTypes;
        for (IRIntegerValue i = 0; i < colCount->getValue(); i++)
        {
            rowTypes.add(elementTupleType);
        }
        return builder.getTargetTupleType((UInt)rowTypes.getCount(), rowTypes.getBuffer());
    }
    else if (auto vectorType = as<IRVectorType>(type))
    {
        auto count = as<IRIntLit>(vectorType->getElementCount());
        if (!count)
        {
            return nullptr;
        }
        List<IRType*> elementTypes;
        for (IRIntegerValue i = 0; i < count->getValue(); i++)
        {
            elementTypes.addRange(vectorType->getElementType());
        }
        return builder.getTargetTupleType((UInt)elementTypes.getCount(), elementTypes.getBuffer());
    }
    else if (auto arrayType = as<IRArrayType>(type))
    {
        auto arraySize = as<IRIntLit>(arrayType->getElementCount());
        if (!arraySize)
        {
            return nullptr;
        }
        List<IRType*> subElementTypes;
        auto subElementType = translateToTupleType(builder, arrayType->getElementType());
        for (IRIntegerValue i = 0; i < arraySize->getValue(); i++)
        {
            subElementTypes.addRange(subElementType);
        }
        return builder.getTargetTupleType(
            (UInt)subElementTypes.getCount(),
            subElementTypes.getBuffer());
    }
    else if (auto structType = as<IRStructType>(type))
    {
        List<IRType*> elementTypes;
        for (auto field : structType->getFields())
        {
            auto fieldType = translateToTupleType(builder, field->getFieldType());
            if (!fieldType)
            {
                return nullptr;
            }
            elementTypes.addRange(fieldType);
        }
        return builder.getTargetTupleType((UInt)elementTypes.getCount(), elementTypes.getBuffer());
    }
    else if (as<IRTargetTupleType>(type))
    {
        return type;
    }
    else
    {
        return nullptr;
    }
}

// Convert a value to a target tuple type.
static IRInst* makeTargetTuple(IRBuilder& builder, IRInst* val)
{
    auto type = val->getDataType();
    if (as<IRVoidType>(type))
        return val;
    if (as<IRBasicType>(type))
        return val;
    else if (as<IRTorchTensorType>(type))
        return val;
    else if (auto matrixType = as<IRMatrixType>(type))
    {
        auto rowCount = as<IRIntLit>(matrixType->getRowCount());
        auto colCount = as<IRIntLit>(matrixType->getColumnCount());
        if (!rowCount || !colCount)
        {
            return nullptr;
        }
        List<IRInst*> rowElements;
        List<IRType*> rowTypes;
        for (IRIntegerValue i = 0; i < rowCount->getValue(); i++)
        {
            List<IRInst*> colElements;
            List<IRType*> colTypes;
            for (IRIntegerValue j = 0; j < colCount->getValue(); j++)
            {
                auto elementVal =
                    builder.emitElementExtract(val, builder.getIntValue(builder.getIntType(), i));
                auto tupleElement = makeTargetTuple(builder, elementVal);
                if (!tupleElement)
                    return nullptr;
                colElements.add(tupleElement);
                colTypes.add(tupleElement->getFullType());
            }
            auto rowType =
                builder.getTargetTupleType((UInt)colTypes.getCount(), colTypes.getBuffer());
            rowTypes.add(rowType);
            rowElements.add(builder.emitMakeTargetTuple(
                rowType,
                (UInt)colElements.getCount(),
                colElements.getBuffer()));
        }
        return builder.emitMakeTargetTuple(
            builder.getTargetTupleType((UInt)rowTypes.getCount(), rowTypes.getBuffer()),
            (UInt)rowElements.getCount(),
            rowElements.getBuffer());
    }
    else if (auto vectorType = as<IRVectorType>(type))
    {
        auto count = as<IRIntLit>(vectorType->getElementCount());
        if (!count)
        {
            return nullptr;
        }
        List<IRInst*> resultElements;
        List<IRType*> elementTypes;
        for (IRIntegerValue i = 0; i < count->getValue(); i++)
        {
            auto elementVal =
                builder.emitElementExtract(val, builder.getIntValue(builder.getIntType(), i));
            auto tupleElement = makeTargetTuple(builder, elementVal);
            if (!tupleElement)
                return nullptr;
            resultElements.add(tupleElement);
            elementTypes.add(tupleElement->getFullType());
        }
        auto resultType =
            builder.getTargetTupleType((UInt)elementTypes.getCount(), elementTypes.getBuffer());
        return builder.emitMakeTargetTuple(
            resultType,
            (UInt)resultElements.getCount(),
            resultElements.getBuffer());
    }
    else if (auto arrayType = as<IRArrayType>(type))
    {
        auto arraySize = as<IRIntLit>(arrayType->getElementCount());
        if (!arraySize)
        {
            return nullptr;
        }
        List<IRInst*> resultElements;
        List<IRType*> elementTypes;
        for (IRIntegerValue i = 0; i < arraySize->getValue(); i++)
        {
            auto elementVal =
                builder.emitElementExtract(val, builder.getIntValue(builder.getIntType(), i));
            auto tupleElement = makeTargetTuple(builder, elementVal);
            if (!tupleElement)
                return nullptr;
            resultElements.add(tupleElement);
            elementTypes.add(tupleElement->getFullType());
        }
        auto resultType =
            builder.getTargetTupleType((UInt)elementTypes.getCount(), elementTypes.getBuffer());
        return builder.emitMakeTargetTuple(
            resultType,
            (UInt)resultElements.getCount(),
            resultElements.getBuffer());
    }
    else if (auto structType = as<IRStructType>(type))
    {
        List<IRInst*> resultElements;
        List<IRType*> elementTypes;
        for (auto field : structType->getFields())
        {
            auto elementVal = builder.emitFieldExtract(field->getFieldType(), val, field->getKey());
            auto tupleElement = makeTargetTuple(builder, elementVal);
            if (!tupleElement)
                return nullptr;
            resultElements.add(tupleElement);
            elementTypes.add(tupleElement->getFullType());
        }
        auto resultType =
            builder.getTargetTupleType((UInt)elementTypes.getCount(), elementTypes.getBuffer());
        return builder.emitMakeTargetTuple(
            resultType,
            (UInt)resultElements.getCount(),
            resultElements.getBuffer());
    }
    else if (as<IRTargetTupleType>(type))
    {
        return val;
    }
    else
    {
        return nullptr;
    }
}

// Convert a target tuple type to a value.
static IRInst* makeValueFromTargetTuple(IRBuilder& builder, IRType* type, IRInst* val)
{
    if (as<IRVoidType>(type))
        return val;
    if (as<IRBasicType>(type))
        return val;
    else if (as<IRTorchTensorType>(type))
        return val;
    else if (auto matrixType = as<IRMatrixType>(type))
    {
        auto rowCount = as<IRIntLit>(matrixType->getRowCount());
        auto colCount = as<IRIntLit>(matrixType->getColumnCount());
        SLANG_ASSERT(rowCount && colCount);

        List<IRInst*> resultElements;
        auto rowType = builder.getTargetTupleType(
            (UInt)colCount->getValue(),
            List<IRType*>()
                .makeRepeated(matrixType->getElementType(), (Index)colCount->getValue())
                .getBuffer());
        for (IRIntegerValue i = 0; i < rowCount->getValue(); i++)
        {
            auto rowElement = builder.emitTargetTupleGetElement(
                rowType,
                val,
                builder.getIntValue(builder.getIntType(), i));
            for (IRIntegerValue j = 0; j < colCount->getValue(); j++)
            {
                auto element = builder.emitTargetTupleGetElement(
                    matrixType->getElementType(),
                    rowElement,
                    builder.getIntValue(builder.getIntType(), j));
                resultElements.add(element);
            }
        }
        return builder.emitMakeMatrix(
            type,
            (UInt)resultElements.getCount(),
            resultElements.getBuffer());
    }
    else if (auto vectorType = as<IRVectorType>(type))
    {
        auto count = as<IRIntLit>(vectorType->getElementCount());
        if (!count)
        {
            return nullptr;
        }
        List<IRInst*> resultElements;
        auto elementType = vectorType->getElementType();
        for (IRIntegerValue i = 0; i < count->getValue(); i++)
        {
            auto tupleElement = builder.emitTargetTupleGetElement(
                elementType,
                val,
                builder.getIntValue(builder.getIntType(), i));
            auto convertedElement = makeValueFromTargetTuple(builder, elementType, tupleElement);
            if (!convertedElement)
                return nullptr;
            resultElements.add(convertedElement);
        }
        return builder.emitMakeVector(
            type,
            (UInt)resultElements.getCount(),
            resultElements.getBuffer());
    }
    else if (auto arrayType = as<IRArrayType>(type))
    {
        auto arraySize = as<IRIntLit>(arrayType->getElementCount());
        if (!arraySize)
        {
            return nullptr;
        }
        List<IRInst*> resultElements;
        auto elementType = arrayType->getElementType();
        auto tupleElementType = translateToTupleType(builder, elementType);
        for (IRIntegerValue i = 0; i < arraySize->getValue(); i++)
        {
            auto tupleElement = builder.emitTargetTupleGetElement(
                tupleElementType,
                val,
                builder.getIntValue(builder.getIntType(), i));

            // Make a name hint: <valname>_<i>
            if (auto nameHint = val->findDecoration<IRNameHintDecoration>())
            {
                StringBuilder newName;
                newName << nameHint->getName() << "_" << i;
                builder.addNameHintDecoration(tupleElement, newName.getUnownedSlice());
            }

            auto convertedElement = makeValueFromTargetTuple(builder, elementType, tupleElement);
            if (!convertedElement)
                return nullptr;
            resultElements.add(convertedElement);
        }
        return builder.emitMakeArray(
            type,
            (UInt)resultElements.getCount(),
            resultElements.getBuffer());
    }
    else if (auto structType = as<IRStructType>(type))
    {
        List<IRInst*> resultElements;
        IRIntegerValue i = 0;
        for (auto field : structType->getFields())
        {
            auto tupleElement = builder.emitTargetTupleGetElement(
                translateToTupleType(builder, field->getFieldType()),
                val,
                builder.getIntValue(builder.getIntType(), i));

            // Make a name hint: <valname>_<fieldname>
            if (auto nameHint = val->findDecoration<IRNameHintDecoration>())
            {
                if (auto fieldHint = field->getKey()->findDecoration<IRNameHintDecoration>())
                {
                    StringBuilder newName;
                    newName << nameHint->getName() << "_" << fieldHint->getName();
                    builder.addNameHintDecoration(tupleElement, newName.getUnownedSlice());
                }
            }

            auto convertedElement =
                makeValueFromTargetTuple(builder, field->getFieldType(), tupleElement);
            if (!convertedElement)
                return nullptr;
            resultElements.add(convertedElement);
            i++;
        }
        return builder.emitMakeStruct(
            type,
            (UInt)resultElements.getCount(),
            resultElements.getBuffer());
    }
    else if (as<IRTargetTupleType>(type))
    {
        return val;
    }
    else
    {
        return nullptr;
    }
}

static void generateCppBindingForFunc(IRFunc* func, DiagnosticSink* sink)
{
    IRBuilder builder(func);

    builder.setInsertBefore(func);
    auto hostReturnType = translateToTupleType(builder, func->getResultType());
    if (!hostReturnType)
    {
        sink->diagnose(
            func->sourceLoc,
            Diagnostics::invalidTorchKernelReturnType,
            func->getResultType());
        return;
    }
    List<IRType*> hostParamTypes;
    auto funcType = as<IRFuncType>(func->getDataType());
    for (UInt i = 0; i < funcType->getParamCount(); i++)
    {
        hostParamTypes.add(translateToTupleType(builder, funcType->getParamType(i)));
    }
    auto bindingFuncType = builder.getFuncType(hostParamTypes, hostReturnType);
    func->setFullType(bindingFuncType);

    builder.setInsertBefore(func->getFirstBlock()->getFirstOrdinaryInst());

    List<IRInst*> instsToRemove;
    List<IRInst*> oldParams;
    for (auto param : func->getFirstBlock()->getParams())
    {
        oldParams.add(param);
    }

    List<IRInst*> newParams;
    for (auto param : oldParams)
    {
        auto paramType = param->getFullType();
        auto newParamType = translateToTupleType(builder, paramType);
        if (!newParamType)
        {
            sink->diagnose(param->sourceLoc, Diagnostics::invalidTorchKernelParamType, paramType);
            return;
        }
        auto newParam = builder.emitParam(newParamType);
        param->transferDecorationsTo(newParam);
        newParams.add(newParam);
    }

    // Convert all new parameters from tuples to their original types.
    for (Index i = 0; i < newParams.getCount(); i++)
    {
        auto oldParam = oldParams[i];
        auto newParam = newParams[i];
        auto convertedParam = makeValueFromTargetTuple(builder, oldParam->getFullType(), newParam);
        if (!convertedParam)
        {
            return;
        }
        oldParam->replaceUsesWith(convertedParam);
        oldParam->removeAndDeallocate();
    }

    for (auto block : func->getBlocks())
    {
        for (auto inst : block->getChildren())
        {
            if (auto kernelDispatch = as<IRDispatchKernel>(inst))
            {
                builder.setInsertBefore(kernelDispatch);
                List<IRInst*> kernelArgs;
                auto kernelArgCount = kernelDispatch->getArgCount();
                auto argArrayType = builder.getArrayType(
                    builder.getPtrType(builder.getVoidType()),
                    builder.getIntValue(builder.getIntType(), kernelArgCount));
                auto argArrayVar = builder.emitVar(argArrayType);
                for (UInt i = 0; i < kernelArgCount; i++)
                {
                    auto arg = kernelDispatch->getArg(i);
                    auto argVar = builder.emitVar(arg->getFullType());
                    builder.emitStore(argVar, arg);
                    auto addr = builder.emitElementAddress(
                        argArrayVar,
                        builder.getIntValue(builder.getIntType(), i));
                    builder.emitStore(addr, argVar);
                }
                auto argArrayPtr = builder.emitElementAddress(
                    argArrayVar,
                    builder.getIntValue(builder.getIntType(), 0));
                builder.emitCudaKernelLaunch(
                    kernelDispatch->getBaseFn(),
                    kernelDispatch->getDispatchSize(),
                    kernelDispatch->getThreadGroupSize(),
                    argArrayPtr,
                    builder.emitGetTorchCudaStream());
                instsToRemove.add(inst);
            }
            else if (auto getView = as<IRTorchTensorGetView>(inst))
            {
                builder.setInsertBefore(getView);
                auto makeView =
                    builder.emitMakeTensorView(getView->getFullType(), inst->getOperand(0));
                getView->replaceUsesWith(makeView);
                instsToRemove.add(getView);
            }
            else if (auto ret = as<IRReturn>(inst))
            {
                builder.setInsertBefore(ret);
                auto retVal = makeTargetTuple(builder, ret->getVal());
                ret->setOperand(0, retVal);
            }
        }
    }

    for (auto inst : instsToRemove)
        inst->removeAndDeallocate();
}

IRType* translateToHostType(
    IRBuilder* builder,
    IRType* type,
    IRInst* func,
    DiagnosticSink* sink = nullptr)
{
    if (as<IRBasicType>(type) || as<IRVectorType>(type) || as<IRMatrixType>(type))
        return type;

    switch (type->getOp())
    {
    case kIROp_TensorViewType:
        return builder->getTorchTensorType(as<IRTensorViewType>(type)->getElementType());
    case kIROp_StructType:
        {
            // Create a new struct type with translated fields.
            List<IRType*> fieldTypes;
            List<IRNameHintDecoration*> fieldNames;
            for (auto field : as<IRStructType>(type)->getFields())
            {
                fieldTypes.add(translateToHostType(builder, field->getFieldType(), func, sink));
                fieldNames.add(field->getKey()->findDecoration<IRNameHintDecoration>());
            }
            auto hostStructType = builder->createStructType();

            // Add fields to the struct.
            for (UInt i = 0; i < (UInt)fieldTypes.getCount(); i++)
            {
                auto structKey = builder->createStructKey();
                if (fieldNames[i])
                    builder->addNameHintDecoration(structKey, fieldNames[i]->getName());
                builder->createStructField(hostStructType, structKey, fieldTypes[i]);
            }

            return hostStructType;
        }
    case kIROp_ArrayType:
        {
            auto elementType =
                translateToHostType(builder, as<IRArrayType>(type)->getElementType(), func, sink);
            if (!elementType)
                return nullptr;
            return builder->getArrayType(elementType, as<IRArrayType>(type)->getElementCount());
        }
    default:
        break;
    }

    if (sink)
        sink->diagnose(type->sourceLoc, Diagnostics::unableToAutoMapCUDATypeToHostType, type, func);
    return nullptr;
}

// Propagates name hints through field extracts.
IRInst* propagateNameHint(IRBuilder* builder, IRFieldExtract* inst)
{
    // If the field has a name hint, propagate it to the inst by appending the field name to the
    // inst name (which must be fetched from the inst's name hint decoration).
    //
    // This is useful for propagating the name hint from a struct field to the inst that extracts
    // the field.
    if (auto nameHint = inst->getField()->findDecoration<IRNameHintDecoration>())
    {
        if (auto instNameHint = inst->getBase()->findDecoration<IRNameHintDecoration>())
        {
            StringBuilder newName;
            newName << instNameHint->getName() << "_" << nameHint->getName();
            builder->addNameHintDecoration(inst, newName.getUnownedSlice());
        }
    }

    return inst;
}

// Propagates name hints through array indexing
IRInst* propagateNameHint(IRBuilder* builder, IRGetElement* inst)
{
    // If the index is a constant, we can propagate the name hint from the inst to the index.
    if (auto intLit = as<IRIntLit>(inst->getIndex()))
    {
        if (auto nameHint = inst->getBase()->findDecoration<IRNameHintDecoration>())
        {
            StringBuilder newName;
            newName << nameHint->getName() << "_" << intLit->getValue();
            builder->addNameHintDecoration(inst, newName.getUnownedSlice());
        }
    }

    return inst;
}

IRInst* castHostToCUDAType(IRBuilder* builder, IRType* hostType, IRType* cudaType, IRInst* inst)
{
    if (hostType == cudaType)
        return inst;
    if (as<IRBasicType>(hostType) && as<IRBasicType>(cudaType))
        return inst;

    switch (cudaType->getOp())
    {
    case kIROp_TensorViewType:
        return builder->emitMakeTensorView(cudaType, inst);
    case kIROp_StructType:
        {
            auto cudaStructType = cast<IRStructType>(cudaType);
            auto hostStructType = cast<IRStructType>(hostType);

            List<IRStructField*> cudaFields;
            for (auto field : cudaStructType->getFields())
                cudaFields.add(field);

            List<IRStructField*> hostFields;
            for (auto field : hostStructType->getFields())
                hostFields.add(field);

            List<IRInst*> resultFields;
            for (auto ii = 0; ii < cudaFields.getCount(); ii++)
            {
                auto cudaField = cudaFields[ii];
                auto hostField = hostFields[ii];
                auto cudaFieldType = cudaField->getFieldType();
                auto hostFieldType = hostField->getFieldType();
                auto castedField = castHostToCUDAType(
                    builder,
                    hostFieldType,
                    cudaFieldType,
                    propagateNameHint(
                        builder,
                        cast<IRFieldExtract>(
                            builder->emitFieldExtract(hostFieldType, inst, hostField->getKey()))));

                SLANG_RELEASE_ASSERT(castedField);
                resultFields.add(castedField);
            }

            return builder->emitMakeStruct(
                cudaType,
                (UInt)resultFields.getCount(),
                resultFields.getBuffer());
        }
    case kIROp_ArrayType:
        {
            auto cudaArrayType = cast<IRArrayType>(cudaType);
            auto hostArrayType = cast<IRArrayType>(hostType);

            List<IRInst*> resultElements;
            for (UInt i = 0; i < (UInt)cast<IRIntLit>(cudaArrayType->getElementCount())->getValue();
                 i++)
            {
                auto cudaElementType = cudaArrayType->getElementType();
                auto hostElementType = hostArrayType->getElementType();
                auto castedElement = castHostToCUDAType(
                    builder,
                    hostElementType,
                    cudaElementType,
                    propagateNameHint(
                        builder,
                        cast<IRGetElement>(builder->emitElementExtract(
                            inst,
                            builder->getIntValue(builder->getIntType(), i)))));

                SLANG_RELEASE_ASSERT(castedElement);
                resultElements.add(castedElement);
            }

            return builder->emitMakeArray(
                cudaType,
                (UInt)resultElements.getCount(),
                resultElements.getBuffer());
        }

    default:
        break;
    }

    // If translateToHostType worked correctly, there should be no unhandled cases here.
    // However, we won't diagnose here since its already diagnosed in translateToHostType()
    return nullptr;
}

void generateReflectionFunc(IRBuilder* builder, IRFunc* kernelFunc, IRFunc* hostFunc)
{
    // Given a func with torch binding, we'll generate a reflection function that returns
    // a tuple where the first element is another tuple of parameter names, the second
    // element is a string containing the name of the fwd-diff function, and the third
    // element is a string containing the name of the bwd-diff function.
    //

    // Create a new function.
    auto reflectionFunc = builder->createFunc();
    builder->setInsertInto(reflectionFunc);
    builder->emitBlock();

    // Go through func & generate a tuple of parameter names.
    List<IRInst*> paramNames;
    List<IRInst*> paramTypeNames;
    UIndex paramCount = 0;
    for (auto param : hostFunc->getFirstBlock()->getParams())
    {
        if (auto nameHint = param->findDecoration<IRNameHintDecoration>())
        {
            paramNames.add(
                builder->emitGetNativeString(builder->getStringValue(nameHint->getName())));
        }
        else
        {
            StringBuilder argNameBuilder;
            argNameBuilder << "param";
            argNameBuilder << paramCount;

            paramNames.add(builder->emitGetNativeString(
                builder->getStringValue(argNameBuilder.getUnownedSlice())));
        }
        paramCount++;
    }

    for (auto param : kernelFunc->getParams())
    {
        // Check for py-export decoration.
        if (auto pyExportHint = param->getDataType()->findDecoration<IRPyExportDecoration>())
        {
            paramTypeNames.add(builder->emitGetNativeString(
                builder->getStringValue(pyExportHint->getExportName())));
        }
        else
        {
            paramTypeNames.add(
                builder->emitGetNativeString(builder->getStringValue(UnownedStringSlice(""))));
        }
    }

    // Create a target-tuple-type for the names
    auto paramNamesTupleType = builder->getTargetTupleType(
        (UInt)paramNames.getCount(),
        List<IRType*>()
            .makeRepeated(builder->getNativeStringType(), paramNames.getCount())
            .getBuffer());
    auto paramNamesTuple = builder->emitMakeTargetTuple(
        paramNamesTupleType,
        paramNames.getCount(),
        paramNames.getBuffer());

    // Create a target-tuple-type for the type names
    auto paramTypeNamesTupleType = builder->getTargetTupleType(
        (UInt)paramTypeNames.getCount(),
        List<IRType*>()
            .makeRepeated(builder->getNativeStringType(), paramTypeNames.getCount())
            .getBuffer());
    auto paramTypeNamesTuple = builder->emitMakeTargetTuple(
        paramTypeNamesTupleType,
        paramTypeNames.getCount(),
        paramTypeNames.getBuffer());

    // Find the fwd-diff function name (blank string indicates no fwd-diff)
    IRInst* fwdDiffName = builder->getStringValue(UnownedStringSlice(""));
    if (auto fwdDiffHint = kernelFunc->findDecoration<IRCudaKernelForwardDerivativeDecoration>())
    {
        auto fwdDiffFunc = fwdDiffHint->getForwardDerivativeFunc();

        if (auto fwdDiffFuncExternHint = fwdDiffFunc->findDecoration<IRExternCppDecoration>())
        {
            fwdDiffName = builder->emitGetNativeString(
                builder->getStringValue(fwdDiffFuncExternHint->getName()));
        }
    }

    // Find the bwd-diff function name (blank string indicates no bwd-diff)
    IRInst* bwdDiffName = builder->getStringValue(UnownedStringSlice(""));
    if (auto bwdDiffHint = kernelFunc->findDecoration<IRCudaKernelBackwardDerivativeDecoration>())
    {
        auto bwdDiffFunc = bwdDiffHint->getBackwardDerivativeFunc();

        if (auto bwdDiffFuncExternHint = bwdDiffFunc->findDecoration<IRExternCppDecoration>())
        {
            bwdDiffName = builder->emitGetNativeString(
                builder->getStringValue(bwdDiffFuncExternHint->getName()));
        }
    }

    auto stringType = builder->getNativeStringType();
    auto returnTupleType = builder->getTargetTupleType(
        4,
        List<IRType*>(paramNamesTupleType, paramTypeNamesTupleType, stringType, stringType)
            .getBuffer());

    // Create a target-tuple-type for the names
    auto returnTupleArgs =
        List<IRInst*>(paramNamesTuple, paramTypeNamesTuple, fwdDiffName, bwdDiffName);
    auto returnTuple = builder->emitMakeTargetTuple(
        returnTupleType,
        returnTupleArgs.getCount(),
        returnTupleArgs.getBuffer());
    builder->emitReturn(returnTuple);

    // Set function type.
    auto funcType = builder->getFuncType(List<IRType*>(), returnTupleType);
    reflectionFunc->setFullType(funcType);

    // Set function name.
    StringBuilder reflFuncExportName;
    auto hostFuncExportName = hostFunc->findDecoration<IRExternCppDecoration>()->getName();
    reflFuncExportName << "__funcinfo__" << hostFuncExportName;

    builder->addExternCppDecoration(reflectionFunc, reflFuncExportName.getUnownedSlice());
    builder->addTorchEntryPointDecoration(reflectionFunc, reflFuncExportName.getUnownedSlice());
    builder->addKeepAliveDecoration(reflectionFunc);
}

IRInst* generateHostParamForCUDAParam(
    IRBuilder* builder,
    IRParam* param,
    DiagnosticSink* sink,
    IRType** outType = nullptr)
{
    auto type = translateToHostType(builder, param->getDataType(), getParentFunc(param), sink);
    if (outType)
        *outType = type;

    if (!type || sink->getErrorCount() > 0)
    {
        return nullptr;
    }

    auto hostParam = builder->emitParam(type);

    // Add a namehint to the param
    if (auto nameHint = param->findDecoration<IRNameHintDecoration>())
    {
        builder->addNameHintDecoration(hostParam, nameHint->getName());
    }

    // Then cast the param to the appropriate type.
    if (auto castedParam = castHostToCUDAType(builder, type, param->getDataType(), hostParam))
        return castedParam;

    return nullptr;
}

void markTypeForPyExport(IRType* type, DiagnosticSink* sink)
{
    // If it's a basic type, we're done.
    if (as<IRBasicType>(type) || as<IRVoidType>(type))
        return;

    // If it's a struct type, mark for py-export.
    if (auto structType = as<IRStructType>(type))
    {
        IRBuilder builder(structType->getModule());

        // If it already has a py-export decoration, we're done.
        if (!structType->findDecoration<IRPyExportDecoration>())
        {
            // Look for a name hint.
            UnownedStringSlice nameHint;
            if (auto nameHintDecoration = structType->findDecoration<IRNameHintDecoration>())
                nameHint = nameHintDecoration->getName();
            else
            {
                // If there's no name hint, we can't export this type.
                SLANG_UNEXPECTED("struct marked for export has no name");
            }

            builder.addPyExportDecoration(structType, nameHint);
        }

        for (auto field : structType->getFields())
        {
            markTypeForPyExport(field->getFieldType(), sink);
        }
        return;
    }
    else if (auto arrayType = as<IRArrayType>(type))
    {
        IRBuilder builder(arrayType->getModule());
        if (!arrayType->findDecoration<IRPyExportDecoration>())
            builder.addPyExportDecoration(arrayType, UnownedStringSlice("Array"));

        markTypeForPyExport(arrayType->getElementType(), sink);
        return;
    }
}

String tryGetExportTypeName(IRBuilder* builder, IRType* type)
{
    if (as<IRStructType>(type))
    {
        if (auto pyExportDecoration = type->findDecoration<IRPyExportDecoration>())
            return String(pyExportDecoration->getExportName());
        else
            return String("");
    }
    else if (auto arrayType = as<IRArrayType>(type))
    {
        StringBuilder nameBuilder;
        nameBuilder << "Array_";
        nameBuilder << tryGetExportTypeName(builder, arrayType->getElementType());
        nameBuilder << "_";
        nameBuilder << cast<IRIntLit>(arrayType->getElementCount())->getValue();

        return nameBuilder.produceString();
    }
    else
        return String();
}

void generateReflectionForType(IRType* type, DiagnosticSink* sink)
{
    SLANG_UNUSED(sink);
    // Emit a function that returns a py::list.
    // The list will contain the names of all the fields of the type.
    //

    if (!type->findDecoration<IRPyExportDecoration>())
        return;

    IRBuilder builder(type->getModule());

    auto reflFunc = builder.createFunc();
    builder.setInsertInto(reflFunc);
    builder.emitBlock();

    List<IRInst*> fieldNames;
    List<IRInst*> fieldTypeNames;

    switch (type->getOp())
    {
    case kIROp_StructType:
        {
            for (auto field : as<IRStructType>(type)->getFields())
            {
                auto structKey = field->getKey();
                // Look for a name hint.
                if (auto nameHintDecoration = structKey->findDecoration<IRNameHintDecoration>())
                    fieldNames.add(builder.emitGetNativeString(
                        builder.getStringValue(nameHintDecoration->getName())));
                else
                    fieldNames.add(builder.emitGetNativeString(
                        builder.getStringValue(UnownedStringSlice(""))));

                auto fieldType = field->getFieldType();
                auto exportName = tryGetExportTypeName(&builder, fieldType);

                if (exportName.getLength() > 0)
                    fieldTypeNames.add(builder.emitGetNativeString(
                        builder.getStringValue(exportName.getUnownedSlice())));
                else
                    fieldTypeNames.add(builder.emitGetNativeString(
                        builder.getStringValue(UnownedStringSlice(""))));
            }
            break;
        }
    case kIROp_ArrayType:
        {
            auto elementType = as<IRArrayType>(type)->getElementType();
            fieldNames.add(
                builder.emitGetNativeString(builder.getStringValue(UnownedStringSlice("type"))));
            fieldTypeNames.add(builder.emitGetNativeString(builder.getStringValue(
                tryGetExportTypeName(&builder, elementType).getUnownedSlice())));

            auto elementCount = as<IRIntLit>(as<IRArrayType>(type)->getElementCount());
            fieldNames.add(
                builder.emitGetNativeString(builder.getStringValue(UnownedStringSlice("size"))));

            StringBuilder elementCountStr;
            elementCountStr << elementCount->getValue();
            fieldTypeNames.add(builder.emitGetNativeString(
                builder.getStringValue(elementCountStr.getUnownedSlice())));
            break;
        }
    default:
        break;
    }

    auto _nameListTupleType = builder.getTargetTupleType(
        (UInt)fieldNames.getCount(),
        List<IRType*>()
            .makeRepeated(builder.getNativeStringType(), fieldNames.getCount())
            .getBuffer());
    auto nameListTuple = builder.emitMakeTargetTuple(
        _nameListTupleType,
        (UInt)fieldNames.getCount(),
        fieldNames.getBuffer());

    auto _typeNameListTupleType = builder.getTargetTupleType(
        (UInt)fieldTypeNames.getCount(),
        List<IRType*>()
            .makeRepeated(builder.getNativeStringType(), fieldTypeNames.getCount())
            .getBuffer());
    auto typeNameListTuple = builder.emitMakeTargetTuple(
        _typeNameListTupleType,
        (UInt)fieldTypeNames.getCount(),
        fieldTypeNames.getBuffer());

    auto _nameAndTypeTupleType = builder.getTargetTupleType(
        2,
        List<IRType*>(_nameListTupleType, _typeNameListTupleType).getBuffer());
    auto nameAndTypeTuple = builder.emitMakeTargetTuple(
        _nameAndTypeTupleType,
        2,
        List<IRInst*>(nameListTuple, typeNameListTuple).getBuffer());
    builder.emitReturn(nameAndTypeTuple);

    // Set function type.
    auto funcType = builder.getFuncType(List<IRType*>(), _nameAndTypeTupleType);
    reflFunc->setFullType(funcType);

    // Set function name.
    StringBuilder reflFuncExportName;
    reflFuncExportName << "__typeinfo__" << tryGetExportTypeName(&builder, type).getUnownedSlice();

    builder.addTorchEntryPointDecoration(reflFunc, reflFuncExportName.getUnownedSlice());
    builder.addExternCppDecoration(reflFunc, reflFuncExportName.getUnownedSlice());
    builder.addKeepAliveDecoration(reflFunc);
}

IRFunc* generateCUDAWrapperForFunc(IRFunc* func, DiagnosticSink* sink)
{
    // Check that the function has an auto-bind decoration
    if (!func->findDecoration<IRAutoPyBindCudaDecoration>())
        return nullptr;

    // We will create a CudaHost function that will call func.
    // But before that, we need to determine the type of CudaHost.
    //
    // To determine the type, first we will append two uint3 parameters to the function.
    // with the names "__blockSize" and "__gridSize", these will serve as input block and
    // grid size parameters for the launch.
    //
    // Then, we will go over the parameters of func, and find a host-mapping for each type
    // by calling mapTypeToCudaHostType(IRType*), which turns structs into tuples, and
    // IRTensorViewType to IRTorchTensorType.
    //
    // Finally, we will create a CudaHost function and transfer the name of func over to
    // the generated method.
    //
    // The function body will first perform any conversion logic needed to convert the
    // parameters from the CudaHost types to the types of func, and then use dispatch_kernel
    // to dispatch func with the given block and grid size.
    //

    // Create new function.
    IRBuilder builder(func->getModule());

    auto hostFunc = builder.createFunc();
    builder.setInsertInto(hostFunc);
    builder.emitBlock();

    List<IRType*> hostParamTypes;

    // Add the two uint3 parameters
    auto uint3Type = builder.getVectorType(builder.getUIntType(), 3);

    auto blockSizeParam = builder.emitParam(uint3Type);
    hostParamTypes.add(uint3Type);
    builder.addNameHintDecoration(blockSizeParam, UnownedStringSlice("__blockSize"));

    auto gridSizeParam = builder.emitParam(uint3Type);
    hostParamTypes.add(uint3Type);
    builder.addNameHintDecoration(gridSizeParam, UnownedStringSlice("__gridSize"));

    List<IRInst*> mappedParams;
    for (auto param : func->getFirstBlock()->getParams())
    {
        IRType* hostParamType;
        mappedParams.add(generateHostParamForCUDAParam(&builder, param, sink, &hostParamType));
        hostParamTypes.add(hostParamType);
        markTypeForPyExport(param->getDataType(), sink); // Should we be marking the host type?
    }

    // Dispatch the original function.
    builder.emitDispatchKernelInst(
        builder.getVoidType(),
        func,
        blockSizeParam,
        gridSizeParam,
        mappedParams.getCount(),
        mappedParams.getBuffer());

    builder.emitReturn();

    IRFuncType* hostFuncType = builder.getFuncType(hostParamTypes, builder.getVoidType());
    hostFunc->setFullType(hostFuncType);

    // Add a torch entry point decoration to the host function to mark
    // for further processing.
    //
    if (auto pybindCudaHint = func->findDecoration<IRAutoPyBindCudaDecoration>())
    {
        // Mark for further processing of torch-specific insts.
        builder.addTorchEntryPointDecoration(hostFunc, pybindCudaHint->getFunctionName());
        // Mark for host-side emit logic.
        builder.addCudaHostDecoration(hostFunc);
        // Keep alive. This method will be accessed externally.
        builder.addKeepAliveDecoration(hostFunc);
    }

    if (auto externCppHint = func->findDecoration<IRExternCppDecoration>())
    {
        // Transfer to the host function.
        builder.addExternCppDecoration(hostFunc, externCppHint->getName());
    }

    if (func->findDecoration<IRAutoPyBindExportInfoDecoration>())
        generateReflectionFunc(&builder, func, hostFunc);

    return hostFunc;
}

void lowerBuiltinTypesForKernelEntryPoints(IRModule* module, DiagnosticSink*)
{
    List<IRFunc*> cudaKernels;
    for (auto globalInst : module->getGlobalInsts())
    {
        if (auto func = as<IRFunc>(globalInst))
        {
            if (func->findDecoration<IRCudaKernelDecoration>())
            {
                cudaKernels.add(func);
            }
        }
    }

    BuiltinTypeLoweringEnv typeLoweringEnv;
    IRBuilder builder(module);
    for (auto func : cudaKernels)
    {
        // Go through parameters and replace any built-in types with their equivalent.
        List<IRParam*> params;
        for (auto param : func->getFirstBlock()->getParams())
        {
            params.add(param);
        }

        bool changed = false;
        List<LoweredBuiltinTypeInfo> loweredParamTypes;
        for (auto param : params)
        {
            LoweredBuiltinTypeInfo info =
                lowerType(&typeLoweringEnv, &builder, param->getDataType());
            loweredParamTypes.add(info);

            if (info.convertLoweredToOriginal != nullptr)
            {
                // Replace parameter with the lowered type.
                auto originalType = param->getDataType();
                param->setFullType(info.loweredType);

                // Call the conversion function to convert the lowered parameter to the original
                // parameter.
                List<IRInst*> args;
                args.add(param);

                setInsertAfterOrdinaryInst(&builder, param);
                auto convertedParam =
                    builder.emitCallInst(originalType, info.convertLoweredToOriginal, args);

                // Replace all uses of the lowered parameter with the converted parameter, except
                // for the call instruction.
                for (auto use = param->firstUse; use;)
                {
                    auto nextUse = use->nextUse;

                    if (use->getUser() == convertedParam)
                    {
                        use = nextUse;
                        continue;
                    }

                    use->set(convertedParam);
                    use = nextUse;
                }

                changed = true;
            }
        }

        if (!changed)
            continue;

        fixUpFuncType(func);

        // Go through any calls to this function and insert a call to converOriginalToLowered before
        // the call.
        for (auto use = func->firstUse; use;)
        {
            auto nextUse = use->nextUse;

            if (as<IRCall>(use->getUser()) || as<IRDispatchKernel>(use->getUser()))
            {
                auto user = use->getUser();
                IROperandList<IRInst> argsList;
                if (auto callInst = as<IRCall>(user))
                    argsList = callInst->getArgsList();
                else if (auto dispatchInst = as<IRDispatchKernel>(user))
                    argsList = dispatchInst->getArgsList();

                // Insert a call to convertOriginalToLowered before the call.
                List<IRInst*> convertedArgs;
                IRBuilder callBuilder(func->getModule());
                callBuilder.setInsertBefore(user);
                for (auto arg : argsList)
                {
                    if (loweredParamTypes[convertedArgs.getCount()].convertOriginalToLowered !=
                        nullptr)
                    {
                        auto convertedArg = callBuilder.emitCallInst(
                            loweredParamTypes[convertedArgs.getCount()].loweredType,
                            loweredParamTypes[convertedArgs.getCount()].convertOriginalToLowered,
                            List<IRInst*>(arg));
                        convertedArgs.add(convertedArg);
                    }
                    else
                    {
                        convertedArgs.add(arg);
                    }
                }

                // Rebuild the call/dispatch inst.
                IRInst* newCall = nullptr;

                if (as<IRCall>(user))
                    newCall = callBuilder.emitCallInst(user->getFullType(), func, convertedArgs);
                else if (auto dispatchInst = as<IRDispatchKernel>(user))
                    newCall = callBuilder.emitDispatchKernelInst(
                        user->getFullType(),
                        func,
                        dispatchInst->getThreadGroupSize(),
                        dispatchInst->getDispatchSize(),
                        convertedArgs.getCount(),
                        convertedArgs.getBuffer());

                // Replace the call instruction.
                user->replaceUsesWith(newCall);

                // Remove the call instruction.
                user->removeAndDeallocate();
            }

            use = nextUse;
        }
    }
}

void generateHostFunctionsForAutoBindCuda(IRModule* module, DiagnosticSink* sink)
{
    List<IRFunc*> autoBindRequests;
    for (auto globalInst : module->getGlobalInsts())
    {
        if (auto func = as<IRFunc>(globalInst))
        {
            if (func->findDecoration<IRAutoPyBindCudaDecoration>())
            {
                autoBindRequests.add(func);
            }
        }
    }

    for (auto func : autoBindRequests)
    {
        generateCUDAWrapperForFunc(func, sink);
    }
}

void generatePyTorchCppBinding(IRModule* module, DiagnosticSink* sink)
{
    List<IRFunc*> workList;
    List<IRFunc*> cudaKernels;
    List<IRType*> typesToExport;
    for (auto globalInst : module->getGlobalInsts())
    {
        if (auto func = as<IRFunc>(globalInst))
        {
            if (func->findDecoration<IRTorchEntryPointDecoration>())
            {
                workList.add(func);
            }
            else if (func->findDecoration<IRCudaKernelDecoration>())
            {
                cudaKernels.add(func);
            }
            else
            {
                // Remove all other export decorations if this is not a cuda host func.
                if (auto decor = func->findDecoration<IRPublicDecoration>())
                    decor->removeAndDeallocate();
                if (auto decor = func->findDecoration<IRHLSLExportDecoration>())
                    decor->removeAndDeallocate();
                if (auto decor = func->findDecoration<IRKeepAliveDecoration>())
                    decor->removeAndDeallocate();
                if (auto decor = func->findDecoration<IRDllExportDecoration>())
                    decor->removeAndDeallocate();
            }
        }
    }

    for (auto func : workList)
        generateCppBindingForFunc(func, sink);

    for (auto func : cudaKernels)
    {
        for (auto block = func->getFirstBlock(); block;)
        {
            auto nextBlock = block->getNextBlock();
            block->removeAndDeallocate();
            block = nextBlock;
        }
    }

    for (auto globalInst : module->getGlobalInsts())
    {
        if (auto type = as<IRType>(globalInst))
        {
            if (type->findDecoration<IRPyExportDecoration>())
            {
                typesToExport.add(type);
            }
        }
    }

    for (auto type : typesToExport)
        generateReflectionForType(type, sink);
}

// Remove all [TorchEntryPoint] functions when emitting CUDA source.
void removeTorchKernels(IRModule* module)
{
    List<IRInst*> toRemove;
    for (auto globalInst : module->getGlobalInsts())
    {
        if (!as<IRFunc>(globalInst))
            continue;
        if (globalInst->findDecoration<IRTorchEntryPointDecoration>())
            toRemove.add(globalInst);
    }
    for (auto inst : toRemove)
        inst->removeAndDeallocate();
}

void handleAutoBindNames(IRModule* module)
{
    // We need to rewrite extern-cpp names for functions that have an auto-bind decoration.
    // since the name needs to be used for the host function.
    //
    for (auto globalInst : module->getGlobalInsts())
    {
        if (globalInst->findDecoration<IRAutoPyBindCudaDecoration>())
        {
            // Find an extern decoration on the original function, and append a prefix to the name.
            if (auto externCppHint = globalInst->findDecoration<IRExternCppDecoration>())
            {
                IRBuilder builder(module);

                // Change the name of the original function.
                StringBuilder nameBuilder;
                nameBuilder << "__kernel__" << externCppHint->getName();
                externCppHint->removeAndDeallocate();
                builder.addExternCppDecoration(globalInst, nameBuilder.getUnownedSlice());
            }
        }
    }
}

void removeTorchAndCUDAEntryPoints(IRModule* module)
{
    // Go through global insts, find cuda & torch related entry points and remove the keep-alive
    // decoration.
    IRBuilder builder(module);
    for (auto globalInst : module->getGlobalInsts())
    {
        if (auto func = as<IRFunc>(globalInst))
        {
            if (func->findDecoration<IRAutoPyBindCudaDecoration>() ||
                func->findDecoration<IRTorchEntryPointDecoration>() ||
                func->findDecoration<IRCudaKernelDecoration>())
            {
                if (auto keepAlive = func->findDecoration<IRKeepAliveDecoration>())
                    keepAlive->removeAndDeallocate();
                if (auto hlslExport = func->findDecoration<IRHLSLExportDecoration>())
                    hlslExport->removeAndDeallocate();
            }
        }
    }
}

void generateDerivativeWrappers(IRModule* module, DiagnosticSink* sink)
{
    SLANG_UNUSED(sink);
    for (auto globalInst : module->getGlobalInsts())
    {
        if (!as<IRFunc>(globalInst))
            continue;

        // Look for methods marked with auto-bind and are differentiable.
        if (globalInst->findDecoration<IRAutoPyBindCudaDecoration>())
        {
            if (globalInst->findDecoration<IRForwardDifferentiableDecoration>() ||
                globalInst->findDecoration<IRBackwardDifferentiableDecoration>())
            {
                // We'll generate a wrapper for this method that calls fwd_diff(fn)
                // but an important thing to note is that we won't actually employ the usual
                // differentiable typing rules. We'll assume none of the parameters are
                // differentiable & throw a warning if some are. This is because, for the
                // auto-binding scenario, we expect to only see tensor types, and their
                // differentiation is handled using tensor _pair_ types which handle the
                // differentiable loads/stores through custom derivatives
                //
                // For now, the user is expected to explicitly use the tensor pair types, so we will
                // simply copy over the original function's signature. In the future, when we update
                // the type system to be able to specify the corresponding pair type, we can update
                // this logic.
                //

                // Create a new wrapper function.
                IRBuilder builder(module);
                auto func = cast<IRFunc>(globalInst);
                auto wrapperFunc = builder.createFunc();
                builder.setInsertInto(wrapperFunc);
                builder.emitBlock();

                // Clone the parameter list.
                List<IRInst*> params;
                for (auto param : func->getFirstBlock()->getParams())
                {
                    auto newParam = builder.emitParam(param->getFullType());

                    // Copy over the name hint.
                    if (auto nameHint = param->findDecoration<IRNameHintDecoration>())
                        builder.addNameHintDecoration(newParam, nameHint->getName());

                    params.add(newParam);
                }

                wrapperFunc->setFullType(func->getFullType());

                auto fwdDiffFunc = builder.emitForwardDifferentiateInst(func->getFullType(), func);
                auto fwdDiffCall = builder.emitCallInst(
                    func->getResultType(),
                    fwdDiffFunc,
                    params.getCount(),
                    params.getBuffer());

                builder.emitReturn(fwdDiffCall);

                // If the original func is a CUDA kernel, mark the wrapper as a CUDA kernel as well.
                if (func->findDecoration<IRCudaKernelDecoration>())
                {
                    builder.addCudaKernelDecoration(wrapperFunc);
                    builder.addExternCDecoration(wrapperFunc);
                }

                // Add an auto-pybind-cuda decoration to the wrapper function to further generate
                // the host-side binding for the derivative kernel.
                //
                {
                    auto autoPyBindCudaHint = func->findDecoration<IRAutoPyBindCudaDecoration>();
                    StringBuilder nameBuilder;
                    nameBuilder << autoPyBindCudaHint->getFunctionName() << "_fwd_diff";
                    builder.addAutoPyBindCudaDecoration(wrapperFunc, nameBuilder.getUnownedSlice());
                }

                // Build a name for the wrapper function: <original_name>_fwd_diff
                if (auto externCppHint = func->findDecoration<IRExternCppDecoration>())
                {
                    StringBuilder nameBuilder;
                    nameBuilder << externCppHint->getName() << "_fwd_diff";
                    builder.addExternCppDecoration(wrapperFunc, nameBuilder.getUnownedSlice());
                }

                builder.addKeepAliveDecoration(wrapperFunc);

                builder.addCudaKernelForwardDerivativeDecoration(func, wrapperFunc);
            }

            if (globalInst->findDecoration<IRBackwardDifferentiableDecoration>())
            {
                // The reasoning for the reverse-mode is the same as the forward-mode version
                // (see above)
                //

                // Create a new wrapper function.
                IRBuilder builder(module);
                auto func = cast<IRFunc>(globalInst);
                auto wrapperFunc = builder.createFunc();
                builder.setInsertInto(wrapperFunc);
                builder.emitBlock();

                // Clone the parameter list.
                List<IRInst*> params;
                for (auto param : func->getFirstBlock()->getParams())
                {
                    auto newParam = builder.emitParam(param->getFullType());

                    // Copy over the name hint.
                    if (auto nameHint = param->findDecoration<IRNameHintDecoration>())
                        builder.addNameHintDecoration(newParam, nameHint->getName());

                    params.add(newParam);
                }

                wrapperFunc->setFullType(func->getFullType());

                auto fwdDiffFunc = builder.emitBackwardDifferentiateInst(func->getFullType(), func);
                auto fwdDiffCall = builder.emitCallInst(
                    func->getResultType(),
                    fwdDiffFunc,
                    params.getCount(),
                    params.getBuffer());

                builder.emitReturn(fwdDiffCall);

                // If the original func is a CUDA kernel, mark the wrapper as a CUDA kernel as well.
                if (func->findDecoration<IRCudaKernelDecoration>())
                {
                    builder.addCudaKernelDecoration(wrapperFunc);
                    builder.addExternCDecoration(wrapperFunc);
                }

                // Add an auto-pybind-cuda decoration to the wrapper function to further generate
                // the host-side binding for the derivative kernel.
                //
                {
                    auto autoPyBindCudaHint = func->findDecoration<IRAutoPyBindCudaDecoration>();
                    StringBuilder nameBuilder;
                    nameBuilder << autoPyBindCudaHint->getFunctionName() << "_bwd_diff";
                    builder.addAutoPyBindCudaDecoration(wrapperFunc, nameBuilder.getUnownedSlice());
                }

                // Build a name for the wrapper function: <original_name>_bwd_diff
                if (auto externCppHint = func->findDecoration<IRExternCppDecoration>())
                {
                    StringBuilder nameBuilder;
                    nameBuilder << externCppHint->getName() << "_bwd_diff";
                    builder.addExternCppDecoration(wrapperFunc, nameBuilder.getUnownedSlice());
                }

                builder.addKeepAliveDecoration(wrapperFunc);

                builder.addCudaKernelBackwardDerivativeDecoration(func, wrapperFunc);
            }
        }
    }
}

} // namespace Slang
