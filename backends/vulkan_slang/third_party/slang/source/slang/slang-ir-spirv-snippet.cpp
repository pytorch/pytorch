// slang-ir-spirv-snippet.cpp

#include "slang-ir-spirv-snippet.h"

#include "../compiler-core/slang-spirv-core-grammar.h"
#include "../core/slang-token-reader.h"
#include "slang-lookup-spirv.h"

namespace Slang
{
static SpvStorageClass translateStorageClass(String name)
{
    if (name == "Uniform")
    {
        return SpvStorageClassUniform;
    }
    else if (name == "StorageBuffer")
    {
        return SpvStorageClassStorageBuffer;
    }
    return (SpvStorageClass)-1;
}

SpvSnippet::ASMType parseASMType(Slang::Misc::TokenReader& tokenReader)
{
    auto word = tokenReader.ReadWord();
    if (word == "float")
        return SpvSnippet::ASMType::Float;
    else if (word == "double")
        return SpvSnippet::ASMType::Double;
    else if (word == "uint2")
        return SpvSnippet::ASMType::UInt2;
    else if (word == "uint16_t")
        return SpvSnippet::ASMType::UInt16;
    else if (word == "float2")
        return SpvSnippet::ASMType::Float2;
    else if (word == "int")
        return SpvSnippet::ASMType::Int;
    else if (word == "uint")
        return SpvSnippet::ASMType::UInt;
    else if (word == "_p")
        return SpvSnippet::ASMType::FloatOrDouble;
    else if (word == "half")
        return SpvSnippet::ASMType::Half;
    return SpvSnippet::ASMType::None;
}

// Read an unsigned integer (a SPIR-V word) or a SPIR-V enum (currently those
// which are coded into this function).
//
// This also 'or's together a list of these words/enums separated by '|'
SpvWord readWordOrWordLiteral(Misc::TokenReader& reader)
{
    SpvWord ret = 0;
    do
    {
        switch (reader.NextToken().Type)
        {
        case Slang::Misc::TokenType::IntLiteral:
            ret = reader.ReadUInt();
            break;
        case Slang::Misc::TokenType::Identifier:
            {
                const auto i = reader.ReadWord();
#define GO(x)    \
    if (i == #x) \
    ret |= Spv##x
                GO(ScopeWorkgroup);
                else GO(ScopeDevice);
                else GO(MemorySemanticsMaskNone);
                else GO(MemorySemanticsAcquireReleaseMask);
                else GO(MemorySemanticsUniformMemoryMask);
                else GO(MemorySemanticsImageMemoryMask);
                else GO(MemorySemanticsAtomicCounterMemoryMask);
                else GO(MemorySemanticsWorkgroupMemoryMask);
#undef GO
                else
                {
                    reader.Back(1);
                    throw Misc::TextFormatException(
                        "Text parsing error: Unrecognized SPIR-V enum: " + i);
                }
            }
            break;
        default:
            throw Misc::TextFormatException("Text parsing error: Expected int or SPIR-V enum");
        }
    } while (reader.AdvanceIf(Misc::TokenType::OpBitOr));
    return ret;
}

RefPtr<SpvSnippet> SpvSnippet::parse(
    const SPIRVCoreGrammarInfo& spirvGrammar,
    UnownedStringSlice definition)
{
    RefPtr<SpvSnippet> snippet = new SpvSnippet();
    try
    {
        Dictionary<String, SpvWord> mapInstNameToIndex;
        Slang::Misc::TokenReader tokenReader(definition);
        // A leading "*" at the beginning of the snip modifies $resultType with
        // a storage class.
        if (tokenReader.AdvanceIf("*"))
        {
            auto storageToken = tokenReader.ReadWord();
            snippet->resultStorageClass = translateStorageClass(storageToken);
        }
        while (!tokenReader.IsEnd())
        {
            SpvSnippet::ASMInst inst;
            if (tokenReader.AdvanceIf("%"))
            {
                String instName = tokenReader.ReadToken().Content;
                mapInstNameToIndex.set(instName, (int)snippet->instructions.getCount());
                tokenReader.Read(Slang::Misc::TokenType::OpAssign);
            }
            SpvOp opCode;
            switch (tokenReader.NextToken().Type)
            {
            case Slang::Misc::TokenType::IntLiteral:
                opCode = (SpvOp)tokenReader.ReadInt();
                break;
            case Slang::Misc::TokenType::Identifier:
                {
                    auto opName = tokenReader.ReadWord();
                    const auto opCodeMaybe = spirvGrammar.opcodes.lookup(opName.getUnownedSlice());
                    if (!opCodeMaybe)
                    {
                        throw Misc::TextFormatException(
                            "Text parsing error: Unrecognized SPIR-V opcode: " + opName);
                    }
                    opCode = *opCodeMaybe;
                    break;
                }
            default:
                throw Misc::TextFormatException("Text parsing error: SPIR-V intrinsics must "
                                                "begin with an integer or opcode name");
            }
            inst.opCode = opCode;
            bool insideOperandList = true;
            const bool isExtInst = inst.opCode == SpvOpExtInst;
            bool isGLSLstd450OpcodeAllowed = false;
            auto readExtInstOpcode = [&]()
            {
                switch (tokenReader.NextToken().Type)
                {
                case Slang::Misc::TokenType::IntLiteral:
                    return (SpvWord)tokenReader.ReadInt();
                    break;
                case Slang::Misc::TokenType::Identifier:
                    {
                        if (isGLSLstd450OpcodeAllowed)
                        {
                            auto opName = tokenReader.ReadWord();
                            GLSLstd450 glslOpcode;
                            if (!lookupGLSLstd450(opName.getUnownedSlice(), glslOpcode))
                            {
                                throw Misc::TextFormatException(
                                    "Text parsing error: Unrecognized SPIR-V GLSLstd450 opcode: " +
                                    opName);
                            }
                            return (SpvWord)glslOpcode;
                        }
                    }
                // fallthrough
                default:
                    throw Misc::TextFormatException(
                        "Text parsing error: Failed to read SPIR-V ExtInst Opcode");
                }
            };
            while (insideOperandList)
            {
                ASMOperand operand = {ASMOperandType::SpvWord, 0, 0, 0};
                switch (tokenReader.NextToken().Type)
                {
                case Slang::Misc::TokenType::Semicolon:
                    insideOperandList = false;
                    tokenReader.ReadToken();
                    break;
                case Slang::Misc::TokenType::IntLiteral:
                    operand.type = SpvSnippet::ASMOperandType::SpvWord;
                    operand.content = tokenReader.ReadInt();
                    inst.operands.add(operand);
                    break;
                case Slang::Misc::TokenType::OpMod:
                    {
                        tokenReader.ReadToken();
                        operand.type = SpvSnippet::ASMOperandType::InstReference;
                        auto refName = tokenReader.ReadToken().Content;
                        if (!mapInstNameToIndex.tryGetValue(refName, operand.content))
                        {
                            SLANG_ASSERT(!"Invalid SPV ASM: referenced inst is not defined.");
                        }
                        inst.operands.add(operand);
                    }
                    break;
                case Slang::Misc::TokenType::Identifier:
                    {
                        auto identifier = tokenReader.ReadToken().Content;
                        if (identifier == "resultType")
                        {
                            operand.type = SpvSnippet::ASMOperandType::ResultTypeId;
                            operand.content = (SpvWord)0xFFFFFFFF;
                            if (tokenReader.AdvanceIf("*"))
                            {
                                // A "*" at operand qualifies the use of `resultType` as
                                // `ptr(resultType, storage class), but does
                                // not modify `resultType` itself.
                                auto storageClass = tokenReader.ReadWord();
                                auto spvStorageClass = translateStorageClass(storageClass);
                                operand.content = spvStorageClass;
                                snippet->usedPtrResultTypeStorageClasses.add(spvStorageClass);
                            }
                            inst.operands.add(operand);
                        }
                        else if (identifier == "resultId")
                        {
                            operand.type = SpvSnippet::ASMOperandType::ResultId;
                            inst.operands.add(operand);
                        }
                        else if (identifier == "glsl450")
                        {
                            operand.type = SpvSnippet::ASMOperandType::GLSL450ExtInstSet;
                            inst.operands.add(operand);
                            // Allow the next token to be parsed as a glslsstd450 opcode
                            isGLSLstd450OpcodeAllowed = isExtInst;
                        }
                        else if (identifier == "fi")
                        {
                            operand.type = SpvSnippet::ASMOperandType::FloatIntegerSelection;
                            tokenReader.Read("(");
                            operand.content = readExtInstOpcode();
                            tokenReader.Read(",");
                            operand.content2 = readExtInstOpcode();
                            tokenReader.Read(")");
                            inst.operands.add(operand);
                        }
                        else if (identifier == "fus")
                        {
                            operand.type = SpvSnippet::ASMOperandType::FloatUnsignedSignedSelection;
                            tokenReader.Read("(");
                            operand.content = readExtInstOpcode();
                            tokenReader.Read(",");
                            operand.content2 = readExtInstOpcode();
                            tokenReader.Read(",");
                            operand.content3 = readExtInstOpcode();
                            tokenReader.Read(")");
                            inst.operands.add(operand);
                        }
                        else if (identifier == "_type")
                        {
                            operand.type = SpvSnippet::ASMOperandType::TypeReference;
                            tokenReader.Read("(");
                            operand.content = (SpvWord)parseASMType(tokenReader);
                            tokenReader.Read(")");
                            inst.operands.add(operand);
                        }
                        else if (identifier.startsWith("_"))
                        {
                            operand.type = SpvSnippet::ASMOperandType::ObjectReference;
                            operand.content = (SpvWord)stringToInt(
                                identifier.subString(1, identifier.getLength() - 1));
                            inst.operands.add(operand);
                        }
                        else if (identifier == "const")
                        {
                            operand.type = SpvSnippet::ASMOperandType::ConstantReference;
                            ASMConstant constant;
                            memset(&constant, 0, sizeof(ASMConstant));
                            tokenReader.Read("(");
                            constant.type = parseASMType(tokenReader);
                            int i = 0;
                            while (tokenReader.AdvanceIf(","))
                            {
                                switch (constant.type)
                                {
                                case ASMType::Float:
                                case ASMType::Float2:
                                case ASMType::FloatOrDouble:
                                    constant.floatValues[i] = tokenReader.ReadFloat();
                                    ++i;
                                    break;

                                default:
                                    constant.intValues[i] = readWordOrWordLiteral(tokenReader);
                                    ++i;
                                    break;
                                }
                            }
                            tokenReader.Read(")");
                            snippet->constants.add(constant);
                            operand.content = (SpvWord)(snippet->constants.getCount() - 1);
                            inst.operands.add(operand);
                        }
                        else if (isGLSLstd450OpcodeAllowed)
                        {
                            GLSLstd450 glslstd450Opcode;
                            lookupGLSLstd450(identifier.getUnownedSlice(), glslstd450Opcode);
                            operand.type = SpvSnippet::ASMOperandType::SpvWord;
                            operand.content = (SpvWord)glslstd450Opcode;
                            inst.operands.add(operand);
                        }
                        else
                        {
                            SLANG_UNEXPECTED(
                                ("Invalid SPV ASM operand: \"" + identifier + "\"").getBuffer());
                        }
                    }
                    break;
                default:
                    insideOperandList = false;
                    break;
                }
            }
            snippet->instructions.add(inst);
        }
    }
    catch (const Slang::Misc::TextFormatException&)
    {
        return nullptr;
    }
    return snippet;
}


} // namespace Slang
