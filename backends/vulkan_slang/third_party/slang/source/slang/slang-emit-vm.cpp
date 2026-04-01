#include "slang-emit-vm.h"

#include "slang-ir-call-graph.h"
#include "slang-ir-layout.h"
#include "slang-ir-util.h"

using namespace slang;

namespace Slang
{
class ByteCodeEmitter
{
public:
    Dictionary<IRInst*, String> mapInstToName;
    Dictionary<String, int> mapNameToUniqueId;
    Dictionary<IRInst*, VMOperand> mapInstToOperand;
    Dictionary<UnownedStringSlice, VMOperand> mapStringToOperand;
    struct ConstKey
    {
        uint64_t value;
        uint32_t size;
        bool operator==(const ConstKey& other) const
        {
            return value == other.value && size == other.size;
        }
        bool operator!=(const ConstKey& other) const { return !(*this == other); }
        HashCode getHashCode() const { return combineHash(value, size); }
    };
    Dictionary<ConstKey, VMOperand> mapConstantIntToOperand;
    Dictionary<IRFunc*, int> mapFuncToId;

    VMByteCodeBuilder& byteCodeBuilder;
    CodeGenContext* codeGenContext;

    ByteCodeEmitter(VMByteCodeBuilder& builder, CodeGenContext* codeGenContext)
        : byteCodeBuilder(builder), codeGenContext(codeGenContext)
    {
    }

    String getName(IRInst* inst)
    {
        String name;
        if (mapInstToName.tryGetValue(inst, name))
            return name;

        if (auto nameDecor = inst->findDecoration<IRNameHintDecoration>())
        {
            name = nameDecor->getName();
        }
        else if (auto linkageDecor = inst->findDecoration<IRLinkageDecoration>())
        {
            name = linkageDecor->getMangledName();
        }
        else
        {
            name = getIROpInfo(inst->getOp()).name;
        }
        if (int* id = mapNameToUniqueId.tryGetValue(name))
        {
            (*id)++;
            name = name + "_" + String(*id);
        }
        else
        {
            mapNameToUniqueId[name] = 0;
        }
        mapInstToName[inst] = name;
        return name;
    }

    struct InstRelocationEntry
    {
        Index offsetToOperand;
        IRBlock* block;
    };

    template<typename T>
    static T alignUp(T value, T alignment)
    {
        return (value + alignment - 1) / alignment * alignment;
    }

    VMOperand allocReg(VMByteCodeFunctionBuilder& funcBuilder, size_t size, size_t alignment)
    {
        VMOperand operand;
        operand.sectionId = kSlangByteCodeSectionWorkingSet;
        operand.offset = funcBuilder.workingSetSizeInBytes;
        funcBuilder.workingSetSizeInBytes =
            alignUp(funcBuilder.workingSetSizeInBytes, (uint32_t)alignment);
        operand.offset = funcBuilder.workingSetSizeInBytes;
        operand.size = size;
        funcBuilder.workingSetSizeInBytes += (uint32_t)size;
        return operand;
    }

    VMOperand ensureWorkingsetMemory(VMByteCodeFunctionBuilder& funcBuilder, IRInst* inst)
    {
        VMOperand operand;

        if (mapInstToOperand.tryGetValue(inst, operand))
            return operand;

        IRSizeAndAlignment sizeAlignment = {};
        getNaturalSizeAndAlignment(
            codeGenContext->getTargetProgram()->getOptionSet(),
            inst->getDataType(),
            &sizeAlignment);
        operand = allocReg(funcBuilder, sizeAlignment.size, sizeAlignment.alignment);
        mapInstToOperand[inst] = operand;
        return operand;
    }

    VMOperand addStringLiteral(UnownedStringSlice str)
    {
        if (auto operand = mapStringToOperand.tryGetValue(str))
            return *operand;
        VMOperand operand;
        operand.sectionId = kSlangByteCodeSectionStrings;
        operand.offset = (uint32_t)byteCodeBuilder.stringOffsets.getCount();

        byteCodeBuilder.stringOffsets.add((uint32_t)byteCodeBuilder.constantSection.getCount());
        byteCodeBuilder.constantSection.addRange((uint8_t*)str.begin(), str.getLength());
        byteCodeBuilder.constantSection.add(0);
        operand.setType(OperandDataType::String);
        operand.size = 0;
        mapStringToOperand[str] = operand;
        return operand;
    }

    void alignConstSection(int alignment)
    {
        int rem = (int)byteCodeBuilder.constantSection.getCount() % alignment;
        if (rem != 0)
        {
            int paddingSize = alignment - rem;
            for (int i = 0; i < paddingSize; i++)
            {
                byteCodeBuilder.constantSection.add(0);
            }
        }
    }

    template<typename IntType>
    VMOperand addConstantValue(IntType value)
    {
        ConstKey key;
        key.value = value;
        key.size = (uint32_t)sizeof(IntType);
        if (auto operand = mapConstantIntToOperand.tryGetValue(key))
            return *operand;
        VMOperand operand;
        operand.sectionId = kSlangByteCodeSectionConstants;
        // align constantSection
        alignConstSection((int)sizeof(IntType));
        operand.offset = (uint32_t)byteCodeBuilder.constantSection.getCount();
        byteCodeBuilder.constantSection.addRange((uint8_t*)&value, sizeof(value));
        mapConstantIntToOperand[key] = operand;

        operand.size = sizeof(IntType);
        if (operand.size == 4)
            operand.setType(OperandDataType::Int32);
        else if (operand.size == 8)
            operand.setType(OperandDataType::Int64);
        else
            operand.setType(OperandDataType::General);
        return operand;
    }

    VMOperand addConstantValue(IRConstant* inst)
    {
        VMOperand operand;
        operand.sectionId = kSlangByteCodeSectionConstants;

        // Align constantSection.
        IRSizeAndAlignment sizeAlignment;
        getNaturalSizeAndAlignment(
            codeGenContext->getTargetProgram()->getOptionSet(),
            inst->getDataType(),
            &sizeAlignment);
        alignConstSection(sizeAlignment.alignment);

        operand.offset = (uint32_t)byteCodeBuilder.constantSection.getCount();
        operand.size = sizeAlignment.size;

        switch (inst->getOp())
        {
        case kIROp_StringLit:
            {
                return addStringLiteral(static_cast<IRStringLit*>(inst)->getStringSlice());
            }
        case kIROp_IntLit:
            {
                int64_t value = static_cast<IRIntLit*>(inst)->getValue();
                byteCodeBuilder.constantSection.addRange((uint8_t*)&value, sizeAlignment.size);
                operand.setType(OperandDataType::General);
                if (sizeAlignment.size != 64)
                {
                    operand.setType(OperandDataType::Int32);
                }
                break;
            }
        case kIROp_FloatLit:
            {
                auto value = static_cast<IRFloatLit*>(inst)->getValue();
                if (inst->getDataType()->getOp() == kIROp_HalfType)
                {
                    auto halfValue = FloatToHalf((float)value);
                    byteCodeBuilder.constantSection.addRange(
                        (uint8_t*)&halfValue,
                        sizeof(halfValue));
                }
                else if (inst->getDataType()->getOp() == kIROp_FloatType)
                {
                    float floatValue = (float)value;
                    byteCodeBuilder.constantSection.addRange(
                        (uint8_t*)&floatValue,
                        sizeof(floatValue));
                    operand.setType(OperandDataType::Float32);
                }
                else
                {
                    byteCodeBuilder.constantSection.addRange((uint8_t*)&value, sizeof(value));
                    operand.setType(OperandDataType::Float64);
                }
                break;
            }
        case kIROp_PtrLit:
            {
                int64_t value = static_cast<IRIntLit*>(inst)->getValue();
                byteCodeBuilder.constantSection.addRange((uint8_t*)&value, sizeof(value));
                break;
            }
        case kIROp_VoidLit:
            break;
        }
        return operand;
    }

    VMOperand ensureInst(IRInst* inst)
    {
        VMOperand operand;
        if (mapInstToOperand.tryGetValue(inst, operand))
            return operand;

        if (auto constantInst = as<IRConstant>(inst))
        {
            operand = addConstantValue(constantInst);
            mapInstToOperand[inst] = operand;
        }
        else
        {
            SLANG_UNEXPECTED("unsupported global inst for vm bytecode emit");
        }
        return operand;
    }

    void writeInst(
        VMByteCodeFunctionBuilder& funcBuilder,
        VMOp op,
        uint32_t extOp,
        ArrayView<VMOperand> operands)
    {
        VMInstHeader instHeader;
        instHeader.opcode = op;
        instHeader.opcodeExtension = extOp;
        instHeader.operandCount = (uint16_t)operands.getCount();
        funcBuilder.instOffsets.add(funcBuilder.code.getCount());
        funcBuilder.code.addRange(reinterpret_cast<uint8_t*>(&instHeader), sizeof(instHeader));
        for (auto operand : operands)
        {
            funcBuilder.code.addRange(reinterpret_cast<uint8_t*>(&operand), sizeof(operand));
        }
    }

    void writeInst(VMByteCodeFunctionBuilder& funcBuilder, VMOp op, uint32_t extOp)
    {
        writeInst(funcBuilder, op, extOp, ArrayView<VMOperand>());
    }

    void writeInst(
        VMByteCodeFunctionBuilder& funcBuilder,
        VMOp op,
        uint32_t extOp,
        VMOperand operand)
    {
        writeInst(funcBuilder, op, extOp, makeArrayViewSingle(operand));
    }

    void writeInst(
        VMByteCodeFunctionBuilder& funcBuilder,
        VMOp op,
        uint32_t extOp,
        VMOperand operand1,
        VMOperand operand2)
    {
        writeInst(funcBuilder, op, extOp, makeArray(operand1, operand2).getView());
    }

    void writeInst(
        VMByteCodeFunctionBuilder& funcBuilder,
        VMOp op,
        uint32_t extOp,
        VMOperand operand1,
        VMOperand operand2,
        VMOperand operand3)
    {
        writeInst(funcBuilder, op, extOp, makeArray(operand1, operand2, operand3).getView());
    }

    uint32_t getExtCode(IRInst* type)
    {
        ArithmeticExtCode extCode = {};
        if (auto vecType = as<IRVectorType>(type))
        {
            extCode.vectorSize = getIntVal(vecType->getElementCount());
            type = vecType->getElementType();
        }
        else if (auto matType = as<IRMatrixType>(type))
        {
            extCode.vectorSize =
                getIntVal(matType->getRowCount()) * getIntVal(matType->getColumnCount());
            type = matType->getElementType();
        }
        switch (type->getOp())
        {
        case kIROp_IntType:
        case kIROp_BoolType:
            extCode.scalarType = kSlangByteCodeScalarTypeSignedInt;
            extCode.scalarBitWidth = 2;
            break;
        case kIROp_Int8Type:
            extCode.scalarType = kSlangByteCodeScalarTypeSignedInt;
            extCode.scalarBitWidth = 0;
            break;
        case kIROp_Int16Type:
            extCode.scalarType = kSlangByteCodeScalarTypeSignedInt;
            extCode.scalarBitWidth = 1;
            break;
        case kIROp_Int64Type:
        case kIROp_IntPtrType:
            extCode.scalarType = kSlangByteCodeScalarTypeSignedInt;
            extCode.scalarBitWidth = 3;
            break;
        case kIROp_UIntType:
            extCode.scalarType = kSlangByteCodeScalarTypeUnsignedInt;
            extCode.scalarBitWidth = 2;
            break;
        case kIROp_UInt8Type:
            extCode.scalarType = kSlangByteCodeScalarTypeUnsignedInt;
            extCode.scalarBitWidth = 0;
            break;
        case kIROp_UInt16Type:
            extCode.scalarType = kSlangByteCodeScalarTypeUnsignedInt;
            extCode.scalarBitWidth = 1;
            break;
        case kIROp_UInt64Type:
        case kIROp_UIntPtrType:
        case kIROp_PtrType:
        case kIROp_OutType:
        case kIROp_InOutType:
        case kIROp_RefType:
        case kIROp_NativePtrType:
            extCode.scalarType = kSlangByteCodeScalarTypeUnsignedInt;
            extCode.scalarBitWidth = 3;
            break;
        case kIROp_FloatType:
            extCode.scalarType = kSlangByteCodeScalarTypeFloat;
            extCode.scalarBitWidth = 2;
            break;
        case kIROp_HalfType:
            extCode.scalarType = kSlangByteCodeScalarTypeFloat;
            extCode.scalarBitWidth = 1;
            break;
        case kIROp_DoubleType:
            extCode.scalarType = kSlangByteCodeScalarTypeFloat;
            extCode.scalarBitWidth = 3;
            break;
        default:
            SLANG_UNEXPECTED("Unsupported type for arithmetic operation");
        }
        uint32_t result;
        memcpy(&result, &extCode, sizeof(extCode));
        return result;
    }

    VMInstHeader translateArithmeticOp(IRInst* inst)
    {
        VMInstHeader opInfo = {};

        switch (inst->getOp())
        {
        case kIROp_Add:
            opInfo.opcode = VMOp::Add;
            break;
        case kIROp_Sub:
            opInfo.opcode = VMOp::Sub;
            break;
        case kIROp_Mul:
            opInfo.opcode = VMOp::Mul;
            break;
        case kIROp_Div:
            opInfo.opcode = VMOp::Div;
            break;
        case kIROp_IRem:
        case kIROp_FRem:
            opInfo.opcode = VMOp::Rem;
            break;
        case kIROp_Neg:
            opInfo.opcode = VMOp::Neg;
            break;
        case kIROp_And:
            opInfo.opcode = VMOp::And;
            break;
        case kIROp_Or:
            opInfo.opcode = VMOp::Or;
            break;
        case kIROp_Not:
            opInfo.opcode = VMOp::Not;
            break;
        case kIROp_BitAnd:
            opInfo.opcode = VMOp::BitAnd;
            break;
        case kIROp_BitOr:
            opInfo.opcode = VMOp::BitOr;
            break;
        case kIROp_BitXor:
            opInfo.opcode = VMOp::BitXor;
            break;
        case kIROp_BitNot:
            opInfo.opcode = VMOp::BitNot;
            break;
        case kIROp_Lsh:
            opInfo.opcode = VMOp::Shl;
            break;
        case kIROp_Rsh:
            opInfo.opcode = VMOp::Shr;
            break;
        case kIROp_Less:
            opInfo.opcode = VMOp::Less;
            break;
        case kIROp_Leq:
            opInfo.opcode = VMOp::Leq;
            break;
        case kIROp_Greater:
            opInfo.opcode = VMOp::Greater;
            break;
        case kIROp_Geq:
            opInfo.opcode = VMOp::Geq;
            break;
        case kIROp_Eql:
            opInfo.opcode = VMOp::Equal;
            break;
        case kIROp_Neq:
            opInfo.opcode = VMOp::Neq;
            break;
        default:
            SLANG_UNEXPECTED("Unsupported operation");
            break;
        }
        opInfo.opcodeExtension = getExtCode(inst->getOperand(0)->getDataType());
        return opInfo;
    }

    void emitCast(VMByteCodeFunctionBuilder& funcBuilder, VMOp op, IRInst* inst)
    {
        auto extCode1 = getExtCode(inst->getDataType());
        auto extCode2 = getExtCode(inst->getOperand(0)->getDataType());
        auto extCode = extCode1 | (extCode2 << 16);
        writeInst(
            funcBuilder,
            op,
            extCode,
            ensureWorkingsetMemory(funcBuilder, inst),
            ensureInst(inst->getOperand(0)));
    }

    void emitInst(
        VMByteCodeFunctionBuilder& funcBuilder,
        IRInst* inst,
        List<InstRelocationEntry>& relocations)
    {
        switch (inst->getOp())
        {
        case kIROp_undefined:
            {
                ensureWorkingsetMemory(funcBuilder, inst);
            }
            break;
        case kIROp_Param:
            {
                auto operand = ensureWorkingsetMemory(funcBuilder, inst);
                if (isFirstBlock(inst->getParent()))
                {
                    funcBuilder.parameterOffsets.add(operand.offset);
                    IRSizeAndAlignment sizeAlignment = {};
                    getNaturalSizeAndAlignment(
                        codeGenContext->getTargetProgram()->getOptionSet(),
                        inst->getDataType(),
                        &sizeAlignment);
                    funcBuilder.parameterSize =
                        operand.offset + (uint32_t)sizeAlignment.getStride();
                }
            }
            break;
        case kIROp_Var:
            {
                IRBuilder builder(inst);
                auto type = tryGetPointedToType(&builder, inst->getDataType());
                IRSizeAndAlignment sizeAlignment = {};
                getNaturalSizeAndAlignment(
                    codeGenContext->getTargetProgram()->getOptionSet(),
                    type,
                    &sizeAlignment);
                auto varStorage = allocReg(
                    funcBuilder,
                    (size_t)sizeAlignment.size,
                    (size_t)sizeAlignment.alignment);
                writeInst(
                    funcBuilder,
                    VMOp::GetWorkingSetPtr,
                    varStorage.offset,
                    ensureWorkingsetMemory(funcBuilder, inst));
            }
            break;
        case kIROp_Load:
            {
                IRSizeAndAlignment sizeAlignment = {};
                getNaturalSizeAndAlignment(
                    codeGenContext->getTargetProgram()->getOptionSet(),
                    inst->getDataType(),
                    &sizeAlignment);
                writeInst(
                    funcBuilder,
                    VMOp::Load,
                    (uint32_t)sizeAlignment.getStride(),
                    ensureWorkingsetMemory(funcBuilder, inst),
                    ensureInst(inst->getOperand(0)));
            }
            break;
        case kIROp_Store:
            {
                IRSizeAndAlignment sizeAlignment = {};
                getNaturalSizeAndAlignment(
                    codeGenContext->getTargetProgram()->getOptionSet(),
                    inst->getOperand(1)->getDataType(),
                    &sizeAlignment);
                writeInst(
                    funcBuilder,
                    VMOp::Store,
                    (uint32_t)sizeAlignment.getStride(),
                    ensureInst(inst->getOperand(0)),
                    ensureInst(inst->getOperand(1)));
            }
            break;
        case kIROp_Add:
        case kIROp_Sub:
        case kIROp_Mul:
        case kIROp_Div:
        case kIROp_And:
        case kIROp_FRem:
        case kIROp_IRem:
        case kIROp_Or:
        case kIROp_BitAnd:
        case kIROp_BitOr:
        case kIROp_BitXor:
        case kIROp_Lsh:
        case kIROp_Rsh:
        case kIROp_Less:
        case kIROp_Leq:
        case kIROp_Greater:
        case kIROp_Geq:
        case kIROp_Eql:
        case kIROp_Neq:
            {
                auto opInfo = translateArithmeticOp(inst);
                IRSizeAndAlignment sizeAlignment = {};
                getNaturalSizeAndAlignment(
                    codeGenContext->getTargetProgram()->getOptionSet(),
                    inst->getDataType(),
                    &sizeAlignment);
                writeInst(
                    funcBuilder,
                    opInfo.opcode,
                    opInfo.opcodeExtension,
                    ensureWorkingsetMemory(funcBuilder, inst),
                    ensureInst(inst->getOperand(0)),
                    ensureInst(inst->getOperand(1)));
            }
            break;
        case kIROp_Neg:
        case kIROp_Not:
        case kIROp_BitNot:
            {
                auto opInfo = translateArithmeticOp(inst);
                IRSizeAndAlignment sizeAlignment = {};
                getNaturalSizeAndAlignment(
                    codeGenContext->getTargetProgram()->getOptionSet(),
                    inst->getDataType(),
                    &sizeAlignment);
                writeInst(
                    funcBuilder,
                    opInfo.opcode,
                    opInfo.opcodeExtension,
                    ensureWorkingsetMemory(funcBuilder, inst),
                    ensureInst(inst->getOperand(0)));
            }
            break;
        case kIROp_unconditionalBranch:
        case kIROp_loop:
            {
                // Write phi arguments into param registers.
                auto branch = as<IRUnconditionalBranch>(inst);
                auto params = branch->getTargetBlock()->getParams();
                List<IRInst*> paramList;
                for (auto param : params)
                {
                    paramList.add(param);
                }
                if (paramList.getCount() != (Index)branch->getArgCount())
                {
                    SLANG_UNEXPECTED("Invalid number of arguments for branch instruction");
                }
                for (UInt i = 0; i < branch->getArgCount(); i++)
                {
                    auto arg = branch->getArg(i);
                    auto param = paramList[i];
                    auto paramReg = ensureWorkingsetMemory(funcBuilder, param);
                    IRSizeAndAlignment sizeAlignment = {};
                    getNaturalSizeAndAlignment(
                        codeGenContext->getTargetProgram()->getOptionSet(),
                        param->getDataType(),
                        &sizeAlignment);
                    writeInst(
                        funcBuilder,
                        VMOp::Copy,
                        (uint32_t)sizeAlignment.getStride(),
                        paramReg,
                        ensureInst(arg));
                }
                // Write jump inst.
                VMOperand relocOperand = {};
                writeInst(funcBuilder, VMOp::Jump, 0, relocOperand);
                InstRelocationEntry entry;
                entry.block = (IRBlock*)inst->getOperand(0);
                entry.offsetToOperand = funcBuilder.code.getCount() - sizeof(VMOperand);
                relocations.add(entry);
            }
            break;
        case kIROp_ifElse:
            {
                VMOperand relocOperand = {};
                writeInst(
                    funcBuilder,
                    VMOp::JumpIf,
                    0,
                    ensureInst(inst->getOperand(0)),
                    relocOperand,
                    relocOperand);
                InstRelocationEntry entry;
                entry.block = (IRBlock*)inst->getOperand(1);
                entry.offsetToOperand = funcBuilder.code.getCount() - sizeof(VMOperand) * 2;
                relocations.add(entry);
                entry.block = (IRBlock*)inst->getOperand(2);
                entry.offsetToOperand = funcBuilder.code.getCount() - sizeof(VMOperand);
                relocations.add(entry);
            }
            break;
        case kIROp_Call:
            {
                auto callInst = as<IRCall>(inst);
                auto callee = as<IRFunc>(callInst->getCallee());
                UnownedStringSlice def;
                IRInst* intrinsicInst;
                if (findTargetIntrinsicDefinition(
                        callee,
                        codeGenContext->getTargetCaps(),
                        def,
                        intrinsicInst))
                {
                    auto calleeOperand = addStringLiteral(def);
                    List<VMOperand> operands;
                    operands.add(ensureWorkingsetMemory(funcBuilder, inst));
                    operands.add(calleeOperand);
                    for (UInt i = 0; i < callInst->getArgCount(); ++i)
                    {
                        operands.add(ensureInst(callInst->getArg(i)));
                    }
                    writeInst(funcBuilder, VMOp::CallExt, 0, operands.getArrayView());
                    break;
                }
                List<VMOperand> operands;
                int calleeId = -1;
                mapFuncToId.tryGetValue(callee, calleeId);
                SLANG_ASSERT(calleeId != -1);
                VMOperand calleeOperand = {};
                calleeOperand.sectionId = kSlangByteCodeSectionFuncs;
                calleeOperand.offset = calleeId;
                calleeOperand.setType(OperandDataType::Int32);
                operands.add(ensureWorkingsetMemory(funcBuilder, inst));
                operands.add(calleeOperand);
                for (UInt i = 0; i < callInst->getArgCount(); ++i)
                {
                    operands.add(ensureInst(callInst->getArg(i)));
                }
                IRSizeAndAlignment sizeAlignment = {};
                getNaturalSizeAndAlignment(
                    codeGenContext->getTargetProgram()->getOptionSet(),
                    inst->getDataType(),
                    &sizeAlignment);
                writeInst(
                    funcBuilder,
                    VMOp::Call,
                    (uint32_t)sizeAlignment.getStride(),
                    operands.getArrayView());
            }
            break;
        case kIROp_MissingReturn:
        case kIROp_Return:
            {
                auto returnInst = as<IRReturn>(inst);
                if (returnInst && returnInst->getVal()->getOp() != kIROp_VoidLit)
                {
                    IRSizeAndAlignment sizeAlignment = {};
                    getNaturalSizeAndAlignment(
                        codeGenContext->getTargetProgram()->getOptionSet(),
                        returnInst->getVal()->getDataType(),
                        &sizeAlignment);
                    writeInst(
                        funcBuilder,
                        VMOp::Ret,
                        (uint32_t)sizeAlignment.getStride(),
                        ensureInst(returnInst->getOperand(0)));
                }
                else
                {
                    writeInst(funcBuilder, VMOp::Ret, 0);
                }
            }
            break;
        case kIROp_GetElementPtr:
            {
                auto getElemInst = as<IRGetElementPtr>(inst);
                auto base = getElemInst->getBase();
                auto index = getElemInst->getIndex();
                IRBuilder builder(inst);
                auto elementType = tryGetPointedToType(&builder, getElemInst->getDataType());
                IRSizeAndAlignment sizeAlignment = {};
                getNaturalSizeAndAlignment(
                    codeGenContext->getTargetProgram()->getOptionSet(),
                    elementType,
                    &sizeAlignment);
                auto stride = sizeAlignment.getStride();
                auto baseOperand = ensureInst(base);
                auto indexOperand = ensureInst(index);
                writeInst(
                    funcBuilder,
                    VMOp::GetElementPtr,
                    (uint32_t)stride,
                    ensureWorkingsetMemory(funcBuilder, inst),
                    baseOperand,
                    indexOperand);
            }
            break;
        case kIROp_FieldAddress:
            {
                auto fieldAddrInst = as<IRFieldAddress>(inst);
                auto base = fieldAddrInst->getBase();
                auto fieldKey = (IRStructKey*)fieldAddrInst->getField();
                IRBuilder builder(base);

                auto structType =
                    as<IRStructType>(tryGetPointedToType(&builder, base->getDataType()));
                IRIntegerValue offset = 0;
                auto field = findStructField(structType, fieldKey);
                getNaturalOffset(
                    codeGenContext->getTargetProgram()->getOptionSet(),
                    field,
                    &offset);

                writeInst(
                    funcBuilder,
                    VMOp::Add,
                    getExtCode(inst->getDataType()),
                    ensureWorkingsetMemory(funcBuilder, inst),
                    ensureInst(base),
                    addConstantValue((uint64_t)offset));
            }
            break;
        case kIROp_GetOffsetPtr:
            {
                auto getOffsetPtrInst = as<IRGetOffsetPtr>(inst);
                auto base = getOffsetPtrInst->getBase();
                auto offset = getOffsetPtrInst->getOffset();
                IRSizeAndAlignment sizeAlignment = {};
                IRBuilder builder(inst);
                auto elementType = tryGetPointedToType(&builder, getOffsetPtrInst->getDataType());
                getNaturalSizeAndAlignment(
                    codeGenContext->getTargetProgram()->getOptionSet(),
                    elementType,
                    &sizeAlignment);
                writeInst(
                    funcBuilder,
                    VMOp::OffsetPtr,
                    (uint32_t)sizeAlignment.getStride(),
                    ensureWorkingsetMemory(funcBuilder, inst),
                    ensureInst(base),
                    ensureInst(offset));
            }
            break;
        case kIROp_FieldExtract:
            {
                auto fieldExtractInst = as<IRFieldExtract>(inst);
                auto base = fieldExtractInst->getBase();
                auto fieldKey = (IRStructKey*)fieldExtractInst->getField();

                auto structType = as<IRStructType>(base->getDataType());
                IRIntegerValue offset = 0;
                auto field = findStructField(structType, fieldKey);
                getNaturalOffset(
                    codeGenContext->getTargetProgram()->getOptionSet(),
                    field,
                    &offset);

                auto baseOperand = ensureInst(base);
                baseOperand.offset += (uint32_t)offset;
                mapInstToOperand[inst] = baseOperand;
            }
            break;
        case kIROp_GetElement:
            {
                auto getElemInst = as<IRGetElement>(inst);
                auto base = getElemInst->getBase();
                auto index = getElemInst->getIndex();
                auto elementType = getElemInst->getDataType();
                IRSizeAndAlignment sizeAlignment = {};
                getNaturalSizeAndAlignment(
                    codeGenContext->getTargetProgram()->getOptionSet(),
                    elementType,
                    &sizeAlignment);
                auto stride = sizeAlignment.getStride();
                auto baseOperand = ensureInst(base);
                if (as<IRIntLit>(index))
                {
                    baseOperand.offset += (uint32_t)(stride * getIntVal(index));
                    mapInstToOperand[inst] = baseOperand;
                    break;
                }
                writeInst(
                    funcBuilder,
                    VMOp::GetElement,
                    (uint32_t)stride,
                    ensureWorkingsetMemory(funcBuilder, inst),
                    baseOperand,
                    ensureInst(index));
            }
            break;
        case kIROp_BitCast:
            {
                auto operand = ensureInst(inst->getOperand(0));
                mapInstToOperand[inst] = operand;
            }
            break;
        case kIROp_IntCast:
        case kIROp_CastIntToPtr:
        case kIROp_CastPtrToInt:
        case kIROp_CastIntToFloat:
        case kIROp_CastFloatToInt:
        case kIROp_FloatCast:
            emitCast(funcBuilder, VMOp::Cast, inst);
            break;
        case kIROp_swizzle:
            {
                auto swizzleInst = as<IRSwizzle>(inst);
                auto base = swizzleInst->getBase();
                auto baseOperand = ensureInst(base);
                auto count = (uint32_t)swizzleInst->getElementCount();
                List<VMOperand> operands;
                operands.add(ensureWorkingsetMemory(funcBuilder, inst));
                operands.add(baseOperand);
                for (UInt i = 0; i < count; ++i)
                {
                    auto index = (uint32_t)getIntVal(swizzleInst->getElementIndex(i));
                    VMOperand operand;
                    operand.sectionId = kSlangByteCodeSectionImmediate;
                    operand.offset = index;
                    operands.add(operand);
                }
                writeInst(
                    funcBuilder,
                    VMOp::Swizzle,
                    getExtCode(inst->getDataType()),
                    operands.getArrayView());
            }
            break;
        case kIROp_MakeArray:
            {
                auto result = ensureWorkingsetMemory(funcBuilder, inst);
                auto arrayType = as<IRArrayTypeBase>(inst->getDataType());
                auto elementType = arrayType->getElementType();
                IRSizeAndAlignment sizeAlignment = {};
                getNaturalSizeAndAlignment(
                    codeGenContext->getTargetProgram()->getOptionSet(),
                    elementType,
                    &sizeAlignment);
                auto stride = (uint32_t)sizeAlignment.getStride();
                for (UInt i = 0; i < inst->getOperandCount(); ++i)
                {
                    VMOperand elementOperand = result;
                    elementOperand.offset += (uint32_t)(stride * i);
                    writeInst(
                        funcBuilder,
                        VMOp::Copy,
                        stride,
                        elementOperand,
                        ensureInst(inst->getOperand(i)));
                }
            }
            break;
        case kIROp_MakeArrayFromElement:
            {
                auto result = ensureWorkingsetMemory(funcBuilder, inst);
                auto arrayType = as<IRArrayTypeBase>(inst->getDataType());
                auto elementType = arrayType->getElementType();
                IRSizeAndAlignment sizeAlignment = {};
                getNaturalSizeAndAlignment(
                    codeGenContext->getTargetProgram()->getOptionSet(),
                    elementType,
                    &sizeAlignment);
                auto stride = (uint32_t)sizeAlignment.getStride();
                for (Index i = 0; i < getIntVal(arrayType->getElementCount()); ++i)
                {
                    VMOperand elementOperand = result;
                    elementOperand.offset += (uint32_t)(stride * i);
                    writeInst(
                        funcBuilder,
                        VMOp::Copy,
                        stride,
                        elementOperand,
                        ensureInst(inst->getOperand(0)));
                }
            }
            break;
        case kIROp_MakeStruct:
            {
                auto result = ensureWorkingsetMemory(funcBuilder, inst);
                auto structType = as<IRStructType>(inst->getDataType());
                List<IRStructField*> fields;
                for (auto field : structType->getFields())
                {
                    fields.add(field);
                }
                for (UInt i = 0; i < inst->getOperandCount(); ++i)
                {
                    auto field = fields[i];
                    IRIntegerValue offset = 0;
                    getNaturalOffset(
                        codeGenContext->getTargetProgram()->getOptionSet(),
                        field,
                        &offset);
                    IRSizeAndAlignment sizeAlignment = {};
                    getNaturalSizeAndAlignment(
                        codeGenContext->getTargetProgram()->getOptionSet(),
                        field->getFieldType(),
                        &sizeAlignment);
                    VMOperand elementOperand = result;
                    elementOperand.offset += (uint32_t)offset;
                    writeInst(
                        funcBuilder,
                        VMOp::Copy,
                        (uint32_t)sizeAlignment.getStride(),
                        elementOperand,
                        ensureInst(inst->getOperand(i)));
                }
            }
            break;
        case kIROp_MakeVector:
        case kIROp_MakeMatrix:
            {
                auto result = ensureWorkingsetMemory(funcBuilder, inst);
                for (UInt i = 0; i < inst->getOperandCount(); ++i)
                {
                    VMOperand elementOperand = result;
                    IRSizeAndAlignment sizeAlignment = {};
                    getNaturalSizeAndAlignment(
                        codeGenContext->getTargetProgram()->getOptionSet(),
                        inst->getOperand(i)->getDataType(),
                        &sizeAlignment);
                    writeInst(
                        funcBuilder,
                        VMOp::Copy,
                        (uint32_t)sizeAlignment.getStride(),
                        elementOperand,
                        ensureInst(inst->getOperand(i)));
                    result.offset += (uint32_t)sizeAlignment.getStride();
                }
            }
            break;
        case kIROp_MakeVectorFromScalar:
            {
                auto result = ensureWorkingsetMemory(funcBuilder, inst);
                auto vectorType = as<IRVectorType>(inst->getDataType());
                IRSizeAndAlignment sizeAlignment = {};
                getNaturalSizeAndAlignment(
                    codeGenContext->getTargetProgram()->getOptionSet(),
                    vectorType->getElementType(),
                    &sizeAlignment);
                auto stride = (uint32_t)sizeAlignment.getStride();
                for (Index i = 0; i < getIntVal(vectorType->getElementCount()); ++i)
                {
                    VMOperand elementOperand = result;
                    elementOperand.offset += (uint32_t)(stride * i);
                    writeInst(
                        funcBuilder,
                        VMOp::Copy,
                        stride,
                        elementOperand,
                        ensureInst(inst->getOperand(0)));
                }
            }
            break;
        case kIROp_MakeMatrixFromScalar:
            {
                auto result = ensureWorkingsetMemory(funcBuilder, inst);
                auto matrixType = as<IRMatrixType>(inst->getDataType());
                IRSizeAndAlignment sizeAlignment = {};
                getNaturalSizeAndAlignment(
                    codeGenContext->getTargetProgram()->getOptionSet(),
                    matrixType->getElementType(),
                    &sizeAlignment);
                auto stride = (uint32_t)sizeAlignment.getStride();
                for (Index i = 0; i < getIntVal(matrixType->getRowCount()); ++i)
                {
                    for (Index j = 0; j < getIntVal(matrixType->getColumnCount()); ++j)
                    {
                        writeInst(
                            funcBuilder,
                            VMOp::Copy,
                            stride,
                            result,
                            ensureInst(inst->getOperand(0)));
                        result.offset += stride;
                    }
                }
            }
            break;
        case kIROp_Printf:
            {
                List<VMOperand> operands;
                operands.add(ensureInst(inst->getOperand(0)));
                auto tuple = inst->getOperand(1);
                if (auto makeTuple = as<IRMakeStruct>(tuple))
                {
                    for (UInt i = 0; i < makeTuple->getOperandCount(); i++)
                    {
                        operands.add(ensureInst(makeTuple->getOperand(i)));
                    }
                }
                else
                {
                    // If not a tuple, it should be a single value.
                    operands.add(ensureInst(tuple));
                }
                writeInst(funcBuilder, VMOp::Print, 0, operands.getArrayView());
            }
            break;
        default:
            SLANG_UNIMPLEMENTED_X("VM bytecode gen for inst.");
        }
    }

    void emitFunction(IRFunc* func)
    {
        VMByteCodeFunctionBuilder funcBuilder;
        funcBuilder.name = addStringLiteral(getName(func).getUnownedSlice());

        IRSizeAndAlignment sizeAlignment = {};
        getNaturalSizeAndAlignment(
            codeGenContext->getTargetProgram()->getOptionSet(),
            func->getResultType(),
            &sizeAlignment);
        funcBuilder.resultSize = (uint32_t)sizeAlignment.getStride();

        Dictionary<IRBlock*, Index> mapBlockToByteOffset;
        List<InstRelocationEntry> relocations;

        for (auto block : func->getBlocks())
        {
            mapBlockToByteOffset[block] = funcBuilder.code.getCount();

            for (auto inst : block->getChildren())
            {
                funcBuilder.instOffsets.add(funcBuilder.code.getCount());
                emitInst(funcBuilder, inst, relocations);
            }
        }

        // Apply relocations for jump targets.
        for (auto reloc : relocations)
        {
            Index offset = mapBlockToByteOffset.getValue(reloc.block);
            uint8_t* codePtr = (funcBuilder.code.getBuffer() + reloc.offsetToOperand);
            VMOperand* operand = (VMOperand*)codePtr;
            operand->sectionId = kSlangByteCodeSectionInsts;
            operand->offset = (uint32_t)offset;
        }
        funcBuilder.workingSetSizeInBytes =
            alignUp(funcBuilder.workingSetSizeInBytes, (uint32_t)sizeof(uint64_t));

        byteCodeBuilder.functions.add(funcBuilder);
    }

    void emitEntryPoints(LinkedIR& linkedIR)
    {
        Dictionary<IRInst*, HashSet<IRFunc*>> referencingEntryPoints;
        buildEntryPointReferenceGraph(referencingEntryPoints, linkedIR.module);
        OrderedHashSet<IRFunc*> entryPointSet;
        for (auto entryPoint : linkedIR.entryPoints)
        {
            auto entryPointDecor = entryPoint->findDecoration<IREntryPointDecoration>();
            if (!entryPointDecor)
                continue;
            if (entryPointDecor->getProfile().getStage() != Stage::Dispatch)
                continue;
            entryPointSet.add(entryPoint);
        }

        List<IRFunc*> functionsToEmit;

        // Emit all entrypoints first.
        for (auto entryPoint : entryPointSet)
        {
            // Emit the function for the entry point.
            functionsToEmit.add(entryPoint);
        }

        // Emit remaining funcitons, if they are called by entry points.
        for (auto globalInst : linkedIR.module->getGlobalInsts())
        {
            auto func = as<IRFunc>(globalInst);

            if (!func)
                continue;

            // Skip if already emitted as an entry point.
            if (entryPointSet.contains(func))
                continue;

            HashSet<IRFunc*>* entryPointRefs = referencingEntryPoints.tryGetValue(func);
            if (!entryPointRefs)
                continue;

            // If the function is referenced by any entry point, emit it.
            bool referencedByHostEntryPoint = false;
            for (auto entryPoint : *entryPointRefs)
            {
                if (entryPointSet.contains(entryPoint))
                {
                    referencedByHostEntryPoint = true;
                    break;
                }
            }
            if (referencedByHostEntryPoint)
            {
                functionsToEmit.add(func);
            }
        }

        // Emit all functions.
        for (Index i = 0; i < functionsToEmit.getCount(); i++)
        {
            mapFuncToId[functionsToEmit[i]] = (int)i;
        }
        for (auto func : functionsToEmit)
        {
            emitFunction(func);
        }
    }
};

SlangResult emitVMByteCodeForEntryPoints(
    CodeGenContext* codeGenContext,
    LinkedIR& linkedIR,
    VMByteCodeBuilder& byteCode)
{
    ByteCodeEmitter emitter(byteCode, codeGenContext);
    emitter.emitEntryPoints(linkedIR);
    return SLANG_OK;
}

SlangResult VMByteCodeBuilder::serialize(slang::IBlob** outBlob)
{
    OwnedMemoryStream ms(FileAccess::Write);
    ms.write(&kSlangByteCodeFourCC, sizeof(uint32_t));
    ms.write(&kSlangByteCodeVersion, sizeof(uint32_t));

    // Write functions section.
    ms.write(&kSlangByteCodeFunctionsFourCC, sizeof(uint32_t));
    uint32_t functionChunkSizeStart = (uint32_t)ms.getPosition();
    uint32_t zero = 0;
    ms.write(&zero, sizeof(uint32_t)); // Reserve space for function chunk size.

    uint32_t functionCount = (uint32_t)functions.getCount();
    ms.write(&functionCount, sizeof(uint32_t));
    // Reserve space for function offsets.
    auto functionOffsetStart = ms.getPosition();
    for (uint32_t i = 0; i < functionCount; ++i)
    {
        ms.write(&zero, sizeof(uint32_t));
    }
    List<uint32_t> functionOffsets;
    for (uint32_t i = 0; i < functionCount; ++i)
    {
        functionOffsets.add((uint32_t)ms.getPosition());

        auto& function = functions[i];
        VMFuncHeader funcHeader;
        funcHeader.name = function.name;
        funcHeader.codeSize = (uint32_t)function.code.getCount();
        funcHeader.parameterCount = (uint32_t)function.parameterOffsets.getCount();
        funcHeader.workingSetSizeInBytes = function.workingSetSizeInBytes;
        funcHeader.returnValueSizeInBytes = function.resultSize;
        funcHeader.parameterSizeInBytes = function.parameterSize;
        ms.write(&funcHeader, sizeof(funcHeader));
        ms.write(
            function.parameterOffsets.getBuffer(),
            sizeof(uint32_t) * function.parameterOffsets.getCount());

        ms.write(function.code.begin(), funcHeader.codeSize);
    }
    uint32_t functionChunkSize =
        (uint32_t)(ms.getPosition() - functionChunkSizeStart - sizeof(uint32_t));

    // Write kernel Blob section.
    ms.write(&kSlangByteCodeKernelBlobFourCC, sizeof(uint32_t));
    uint32_t kernelBlobSize = (uint32_t)kernelBlob->getBufferSize();
    ms.write(&kernelBlobSize, sizeof(uint32_t));
    ms.write(kernelBlob->getBufferPointer(), kernelBlobSize);

    // Write constant section.
    ms.write(&kSlangByteCodeConstantsFourCC, sizeof(uint32_t));
    uint32_t constanBlobSize = (uint32_t)constantSection.getCount();
    ms.write(&constanBlobSize, sizeof(uint32_t));
    uint32_t stringCount = (uint32_t)stringOffsets.getCount();
    ms.write(&stringCount, sizeof(uint32_t));
    ms.write(stringOffsets.getBuffer(), sizeof(uint32_t) * stringCount);
    ms.write(constantSection.begin(), constanBlobSize);

    auto blob = RawBlob::create(ms.getContents().getBuffer(), ms.getContents().getCount());

    // Patch in the function chunk size.
    uint32_t* functionChunkSizePtr =
        (uint32_t*)((uint8_t*)blob->getBufferPointer() + functionChunkSizeStart);
    *functionChunkSizePtr = functionChunkSize;

    // Patch in the function offsets.
    auto funcOffsetTable = (uint32_t*)((uint8_t*)blob->getBufferPointer() + functionOffsetStart);
    for (uint32_t i = 0; i < functionCount; ++i)
    {
        funcOffsetTable[i] = functionOffsets[i];
    }

    *outBlob = blob.detach();
    return SLANG_OK;
}

} // namespace Slang
