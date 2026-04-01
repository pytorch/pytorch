// slang-serialize-ir-types.h
#ifndef SLANG_SERIALIZE_IR_TYPES_H_INCLUDED
#define SLANG_SERIALIZE_IR_TYPES_H_INCLUDED

#include "../compiler-core/slang-name.h"
#include "../compiler-core/slang-source-loc.h"
#include "../core/slang-array-view.h"
#include "../core/slang-riff.h"
#include "../core/slang-string-slice-pool.h"
#include "slang-ir.h"
#include "slang-serialize-source-loc.h"
#include "slang-serialize-types.h"

namespace Slang
{

// Pre-declare
class Name;

struct IRSerialBinary
{
    /// IR module list
    static const FourCC kIRModuleFourCc = SLANG_FOUR_CC('S', 'i', 'm', 'd');

    /* NOTE! All FourCC that can be compressed must start with capital 'S', because compressed
    version is the same FourCC with the 'S' replaced with 's' */

    static const FourCC kInstFourCc = SLANG_FOUR_CC('S', 'L', 'i', 'n');
    static const FourCC kChildRunFourCc = SLANG_FOUR_CC('S', 'L', 'c', 'r');
    static const FourCC kExternalOperandsFourCc = SLANG_FOUR_CC('S', 'L', 'e', 'o');

    static const FourCC kUInt32RawSourceLocFourCc = SLANG_FOUR_CC('S', 'r', 's', '4');

    /// Debug information is held elsewhere, but if this optional section exists, it maps
    /// instructions to locs
    static const FourCC kDebugSourceLocRunFourCc = SLANG_FOUR_CC('S', 'd', 's', 'r');
};

struct IRSerialData
{
    typedef IRSerialData ThisType;

    typedef SerialStringData::StringIndex StringIndex;

    enum class InstIndex : uint32_t;
    enum class ArrayIndex : uint32_t;

    enum class RawSourceLoc : SourceLoc::RawValue; ///< This is just to copy over source loc data
                                                   ///< (ie not strictly serialize)

    typedef uint32_t SizeType;

    /// A run of instructions
    struct InstRun
    {
        typedef InstRun ThisType;
        SLANG_FORCE_INLINE bool operator==(const ThisType& rhs) const
        {
            return m_parentIndex == rhs.m_parentIndex && m_startInstIndex == rhs.m_startInstIndex &&
                   m_numChildren == rhs.m_numChildren;
        }
        SLANG_FORCE_INLINE bool operator!=(const ThisType& rhs) const { return !(*this == rhs); }

        InstIndex m_parentIndex;    ///< The parent instruction
        InstIndex m_startInstIndex; ///< The index to the first instruction
        SizeType m_numChildren;     ///< The number of children
    };

    struct SourceLocRun
    {
        typedef SourceLocRun ThisType;

        bool operator==(const ThisType& rhs) const
        {
            return m_sourceLoc == rhs.m_sourceLoc && m_startInstIndex == rhs.m_startInstIndex &&
                   m_numInst == rhs.m_numInst;
        }
        bool operator!=(const ThisType& rhs) const { return !(*this == rhs); }
        bool operator<(const ThisType& rhs) const { return m_sourceLoc < rhs.m_sourceLoc; }

        SerialSourceLocData::SourceLoc m_sourceLoc; ///< The source location
        InstIndex m_startInstIndex;                 ///< The index to the first instruction
        SizeType m_numInst;                         ///< The number of children
    };

    struct PayloadInfo
    {
        uint8_t m_numOperands;
        uint8_t m_numStrings;
    };


    // Instruction...
    // We can store SourceLoc values separately. Just store per index information.
    // Parent information is stored in m_childRuns
    // Decoration information is stored in m_decorationRuns
    struct Inst
    {
        typedef Inst ThisType;
        enum
        {
            kMaxOperands = 2, ///< Maximum number of operands that can be held in an instruction
                              ///< (otherwise held 'externally')
        };

        // NOTE! Can't change order or list without changing appropriate s_payloadInfos
        enum class PayloadType : uint8_t
        {
            // First 3 must be in this order so a cast from 0-2 is directly represented as number of
            // operands
            Empty,     ///< Has no payload (or operands)
            Operand_1, ///< 1 Operand
            Operand_2, ///< 2 Operands

            OperandAndUInt32, ///< 1 Operand and a single UInt32
            OperandExternal,  ///< Operands are held externally
            String_1,         ///< 1 String
            String_2,         ///< 2 Strings
            UInt32,           ///< Holds an unsigned 32 bit integral (might represent a type)
            Float64,
            Int64,

            CountOf,
        };

        /// Get the number of operands
        SLANG_FORCE_INLINE int getNumOperands() const;

        bool operator==(const ThisType& rhs) const;

        SLANG_FORCE_INLINE bool operator!=(const ThisType& rhs) const { return !(*this == rhs); }

        uint16_t m_op;             ///< For now one of IROp
        PayloadType m_payloadType; ///< The type of payload
        uint8_t m_pad0;            ///< Not currently used

        InstIndex m_resultTypeIndex; //< 0 if has no type. The result type of this instruction

        struct ExternalOperandPayload
        {
            ArrayIndex m_arrayIndex; ///< Index into the m_externalOperands table
            SizeType m_size;         ///< The amount of entries in that table
        };

        struct OperandAndUInt32
        {
            InstIndex m_operand;
            uint32_t m_uint32;
        };

        union Payload
        {
            double m_float64;
            int64_t m_int64;
            uint32_t m_uint32;            ///< Unsigned integral value
            IRFloatingPointValue m_float; ///< Floating point value
            IRIntegerValue m_int;         ///< Integral value
            InstIndex
                m_operands[kMaxOperands]; ///< For items that 2 or less operands it can use this.
            StringIndex m_stringIndices[kMaxOperands];
            ExternalOperandPayload
                m_externalOperand; ///< Operands are stored in an an index of an operand array
            OperandAndUInt32 m_operandAndUInt32;
        };

        Payload m_payload;
    };

    /// Clear to initial state
    void clear();
    /// Get the operands of an instruction
    SLANG_FORCE_INLINE int getOperands(const Inst& inst, const InstIndex** operandsOut) const;

    /// ==
    bool operator==(const ThisType& rhs) const;
    SLANG_FORCE_INLINE bool operator!=(const ThisType& rhs) const { return !(*this == rhs); }

    /// Calculate the amount of memory used by this IRSerialData
    size_t calcSizeInBytes() const;

    /// Ctor
    IRSerialData();

    List<Inst> m_insts; ///< The instructions

    List<RawSourceLoc> m_rawSourceLocs; ///< A source location per instruction (saved without
                                        ///< modification from IRInst)

    List<InstRun>
        m_childRuns; ///< Holds the information about children that belong to an instruction

    List<InstIndex> m_externalOperands; ///< Holds external operands (for instructions with more
                                        ///< than kNumOperands)

    List<char> m_stringTable; ///< All strings. Indexed into by StringIndex

    List<SourceLocRun> m_debugSourceLocRuns; ///< Runs of instructions that use a source loc

    static const PayloadInfo s_payloadInfos[int(Inst::PayloadType::CountOf)];
};

// --------------------------------------------------------------------------
SLANG_FORCE_INLINE int IRSerialData::Inst::getNumOperands() const
{
    return (m_payloadType == PayloadType::OperandExternal)
               ? m_payload.m_externalOperand.m_size
               : s_payloadInfos[int(m_payloadType)].m_numOperands;
}

// --------------------------------------------------------------------------
SLANG_FORCE_INLINE bool IRSerialData::Inst::operator==(const ThisType& rhs) const
{
    if (m_op == rhs.m_op && m_payloadType == rhs.m_payloadType &&
        m_resultTypeIndex == rhs.m_resultTypeIndex)
    {
        switch (m_payloadType)
        {
        case PayloadType::Empty:
            {
                return true;
            }
        case PayloadType::Operand_1:
        case PayloadType::String_1:
        case PayloadType::UInt32:
            {
                return m_payload.m_operands[0] == rhs.m_payload.m_operands[0];
            }
        case PayloadType::OperandAndUInt32:
        case PayloadType::OperandExternal:
        case PayloadType::Operand_2:
        case PayloadType::String_2:
            {
                return m_payload.m_operands[0] == rhs.m_payload.m_operands[0] &&
                       m_payload.m_operands[1] == rhs.m_payload.m_operands[1];
            }
        case PayloadType::Float64:
        case PayloadType::Int64:
            {
                return m_payload.m_int64 == rhs.m_payload.m_int64;
            }
        default:
            break;
        }
    }

    return false;
}
// --------------------------------------------------------------------------
SLANG_FORCE_INLINE int IRSerialData::getOperands(const Inst& inst, const InstIndex** operandsOut)
    const
{
    if (inst.m_payloadType == Inst::PayloadType::OperandExternal)
    {
        *operandsOut =
            m_externalOperands.begin() + int(inst.m_payload.m_externalOperand.m_arrayIndex);
        return int(inst.m_payload.m_externalOperand.m_size);
    }
    else
    {
        *operandsOut = inst.m_payload.m_operands;
        return s_payloadInfos[int(inst.m_payloadType)].m_numOperands;
    }
}


} // namespace Slang

#endif
