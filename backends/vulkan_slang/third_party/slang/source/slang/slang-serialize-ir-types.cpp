// slang-serialize-ir-types.cpp
#include "slang-serialize-ir-types.h"

#include "../core/slang-byte-encode-util.h"
#include "../core/slang-math.h"
#include "../core/slang-text-io.h"
#include "slang-ir-insts.h"

namespace Slang
{

/* Note that an IRInst can be derived from, but when it derived from it's new members are IRUse
variables, and they in effect alias over the operands - and reflected in the operand count. There
_could_ be other members after these IRUse variables, but only a few types include extra data, and
these do not have any operands:

* IRConstant        - Needs special-case handling
* IRModuleInst      - Presumably we can just set to the module pointer on reconstruction

Note! That on an IRInst there is an IRType* variable (accessed as getFullType()). As it stands it
may NOT actually point to an IRType derived type. Its 'ok' as long as it's an instruction that can
be used in the place of the type. So this code does not bother to check if it's correct, and just
casts it.
*/

/* static */ const IRSerialData::PayloadInfo
    IRSerialData::s_payloadInfos[int(Inst::PayloadType::CountOf)] = {
        {0, 0}, // Empty
        {1, 0}, // Operand_1
        {2, 0}, // Operand_2
        {1, 0}, // OperandAndUInt32,
        {0, 0}, // OperandExternal - This isn't correct, Operand has to be specially handled
        {0, 1}, // String_1,
        {0, 2}, // String_2,
        {0, 0}, // UInt32,
        {0, 0}, // Float64,
        {0, 0}  // Int64,
};

struct PrefixString;

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! IRSerialData !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

template<typename T>
static size_t _calcArraySize(const List<T>& list)
{
    return list.getCount() * sizeof(T);
}

size_t IRSerialData::calcSizeInBytes() const
{
    return _calcArraySize(m_insts) + _calcArraySize(m_childRuns) +
           _calcArraySize(m_externalOperands) + _calcArraySize(m_stringTable) +
           /* Raw source locs */
           _calcArraySize(m_rawSourceLocs) +
           /* Debug */
           _calcArraySize(m_debugSourceLocRuns);
}

IRSerialData::IRSerialData()
{
    clear();
}

void IRSerialData::clear()
{
    // First Instruction is null
    m_insts.setCount(1);
    memset(&m_insts[0], 0, sizeof(Inst));

    m_childRuns.clear();
    m_externalOperands.clear();
    m_rawSourceLocs.clear();

    m_stringTable.clear();

    m_debugSourceLocRuns.clear();
}

bool IRSerialData::operator==(const ThisType& rhs) const
{
    return (this == &rhs) ||
           (SerialListUtil::isEqual(m_insts, rhs.m_insts) &&
            SerialListUtil::isEqual(m_childRuns, rhs.m_childRuns) &&
            SerialListUtil::isEqual(m_externalOperands, rhs.m_externalOperands) &&
            SerialListUtil::isEqual(m_rawSourceLocs, rhs.m_rawSourceLocs) &&
            SerialListUtil::isEqual(m_stringTable, rhs.m_stringTable) &&
            /* Debug */
            SerialListUtil::isEqual(m_debugSourceLocRuns, rhs.m_debugSourceLocRuns));
}

} // namespace Slang
