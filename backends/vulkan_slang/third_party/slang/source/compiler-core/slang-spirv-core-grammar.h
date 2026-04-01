#pragma once

#include "../core/slang-dictionary.h"
#include "../core/slang-smart-pointer.h"
#include "../core/slang-string-slice-pool.h"
#include "../core/slang-string.h"

#include <optional>
#include <spirv/unified1/spirv.h>

namespace Slang
{
using SpvWord = uint32_t;
class DiagnosticSink;
class SourceView;

struct SPIRVCoreGrammarInfo : public RefObject
{
    static RefPtr<SPIRVCoreGrammarInfo> loadFromJSON(SourceView& source, DiagnosticSink& sink);
    static RefPtr<SPIRVCoreGrammarInfo>& getEmbeddedVersion();
    static inline void freeEmbeddedGrammerInfo() { getEmbeddedVersion() = nullptr; }

    template<typename K, typename T>
    struct Lookup
    {
        std::optional<T> lookup(const K& name) const
        {
            T ret;
            if (embedded ? embedded(name, ret) : dict.tryGetValue(name, ret))
                return ret;
            else
                return std::nullopt;
        }

        bool (*embedded)(const K&, T&) = nullptr;
        Dictionary<K, T> dict;
    };

    struct OperandKind
    {
        uint8_t index;
        SLANG_COMPONENTWISE_HASHABLE_1;
        SLANG_COMPONENTWISE_EQUALITY_1(OperandKind);
    };

    struct QualifiedEnumName
    {
        OperandKind kind;
        UnownedStringSlice name;
        SLANG_COMPONENTWISE_HASHABLE_2;
        SLANG_COMPONENTWISE_EQUALITY_2(QualifiedEnumName);
    };

    struct QualifiedEnumValue
    {
        OperandKind kind;
        SpvWord value;
        SLANG_COMPONENTWISE_HASHABLE_2;
        SLANG_COMPONENTWISE_EQUALITY_2(QualifiedEnumValue);
    };

    struct OpInfo
    {
        enum Class
        {
            // Unrecognized instructions go in here
            Other,

            // Adding to this? Don't forget to update the embedding generator
            Miscellaneous,
            Debug,
            Annotation,
            Extension,
            ModeSetting,
            TypeDeclaration,
            ConstantCreation,
            Memory,
            Function,
            Image,
            Conversion,
            Composite,
            Arithmetic,
            Bit,
            Relational_and_Logical,
            Derivative,
            ControlFlow,
            Atomic,
            Primitive,
            Barrier,
            Group,
            DeviceSideEnqueue,
            Pipe,
            NonUniform,
            Reserved,
        };
        constexpr static int8_t kNoResultTypeId = -1;
        constexpr static int8_t kNoResultId = -1;

        Class class_;
        // -1 or 0
        int8_t resultTypeIndex = kNoResultTypeId;
        // -1 or 0 or 1
        int8_t resultIdIndex = kNoResultId;
        // The range of valid operand counts for this instruction,
        // including any result type and id. Multi-word operands count as a
        // single operand.
        uint16_t minOperandCount;
        uint16_t maxOperandCount;
        // when looking up an operand type, clamp to this number-1 to
        // account for variable length operands at the end
        uint16_t numOperandTypes;
        const OperandKind* operandTypes;
    };

    //
    // Our tables:
    //

    // Instruction name to opcode
    Lookup<UnownedStringSlice, SpvOp> opcodes;
    // Capability name to value
    Lookup<UnownedStringSlice, SpvCapability> capabilities;
    // String-qualified enum name (one with the type prefix) to value
    Lookup<UnownedStringSlice, SpvWord> allEnumsWithTypePrefix;
    // kind * enum name to value
    Lookup<QualifiedEnumName, SpvWord> allEnums;
    // kine * enum value to unqualified name
    Lookup<QualifiedEnumValue, UnownedStringSlice> allEnumNames;
    // Any other information on instructions
    Lookup<SpvOp, OpInfo> opInfos;
    // Opcode to instruction name
    Lookup<SpvOp, UnownedStringSlice> opNames;
    // Operand kind string to numeric id
    Lookup<UnownedStringSlice, OperandKind> operandKinds;
    // Operand kind id to string
    Lookup<OperandKind, UnownedStringSlice> operandKindNames;
    // Operand kind to the "un-id" version of itself, for example IdMemorySemantics to
    // MemorySemantics
    Lookup<OperandKind, OperandKind> operandKindUnderneathIds;

private:
    // If this is loaded from JSON, we keep the strings around instead of
    // copying them as dictionary keys
    StringSlicePool strings = StringSlicePool(StringSlicePool::Style::Empty);
    List<OperandKind> operandTypesStorage;
};
} // namespace Slang
