// slang-ir-spirv-legalize.h
#pragma once
#include "../core/slang-basic.h"
#include "spirv/unified1/spirv.h"

namespace Slang
{

struct SPIRVCoreGrammarInfo;

//
// [2.2: Terms]
//
// > Word: 32 bits.
//
// Despite the importance to SPIR-V, the `spirv.h` header doesn't
// define a type for words, so we'll do it here.

/// A SPIR-V word.
typedef uint32_t SpvWord;

/// Represents a parsed Spv ASM from intrinsic definition.
struct SpvSnippet : public RefObject
{
    enum class ASMOperandType
    {
        // Plain SpvWord to inline without modifications.
        SpvWord,
        // Represents the result type of the intrinsic.
        ResultTypeId,
        // Represents the result Id of the ASM inst.
        ResultId,
        // Represents a reference to an intrinsic argument (e.g. `_1`).
        ObjectReference,
        // Represents a reference to an ASM inst (e.g. `%t`).
        InstReference,
        // Refer to the GLSL450 Instruction Set.
        GLSL450ExtInstSet,
        // A select expression based on whether result type is float, e.g.
        // `fi(x,y)` selects `x` if resultType is `float`.
        FloatIntegerSelection,
        // A select expression based on whether result type is float, unsigned
        // or signed integer. e.g. `fus(f_opcode, u_opcode, s_opcode)`.
        FloatUnsignedSignedSelection,
        // Reference to a type defined in `ASMType`.
        TypeReference,
        // Reference to a Constant defined in `SpvSnippet::constants`.
        ConstantReference,
    };

    struct ASMOperand
    {
        ASMOperandType type;

        // The value of the spv word when type is `SpvWord`, or
        // the reference name when type is `ObjectReference`
        // (e.g. an argument reference (_1) has `content` == 1).
        SpvWord content;

        // Additional value contents.
        SpvWord content2;
        SpvWord content3;
    };

    enum class ASMType : SpvWord
    {
        None,
        Int,
        UInt,
        UInt16,
        Half,
        Float,
        Double,
        FloatOrDouble, // Float or double type, depending on the result type of the intrinsic.
        Float2,
        UInt2,
    };

    struct ASMConstant
    {
        ASMType type;
        SpvWord intValues[4];
        float floatValues[4];
        HashCode getHashCode() const
        {
            HashCode result = (HashCode)type;
            for (int i = 0; i < 4; i++)
            {
                switch (type)
                {
                case ASMType::Half:
                case ASMType::Float:
                case ASMType::Double:
                case ASMType::Float2:
                case ASMType::FloatOrDouble:
                    result = combineHash(result, Slang::getHashCode(floatValues[i]));
                    break;
                default:
                    result = combineHash(result, Slang::getHashCode(intValues[i]));
                    break;
                }
            }
            return result;
        }
        bool operator==(const ASMConstant& other) const
        {
            if (type != other.type)
                return false;
            switch (type)
            {
            case ASMType::Half:
            case ASMType::Float:
            case ASMType::Double:
            case ASMType::FloatOrDouble:
                return floatValues[0] == other.floatValues[0];
            case ASMType::Float2:
                return floatValues[0] == other.floatValues[0] &&
                       floatValues[1] == other.floatValues[1];
            case ASMType::Int:
                return intValues[0] == other.intValues[0];
            case ASMType::UInt:
            case ASMType::UInt16:
                return intValues[0] == other.intValues[0];
            case ASMType::UInt2:
                return intValues[0] == other.intValues[0] && intValues[1] == other.intValues[1];
            default:
                return false;
            }
        }
    };

    struct ASMInst
    {
        SpvWord opCode = 0;
        List<ASMOperand> operands;
    };

    List<ASMInst> instructions;
    HashSet<SpvStorageClass> usedPtrResultTypeStorageClasses;
    List<ASMConstant> constants;
    SpvStorageClass resultStorageClass = SpvStorageClassMax;

    static RefPtr<SpvSnippet> parse(
        const SPIRVCoreGrammarInfo& spirvGrammar,
        UnownedStringSlice definition);
};


} // namespace Slang
