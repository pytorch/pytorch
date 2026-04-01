#include "flag-combiner.h"

namespace gfx
{
using namespace Slang;

void FlagCombiner::add(uint32_t flags, ChangeType type)
{
    // The flag/s must be set
    SLANG_ASSERT(flags);
    SLANG_ASSERT((flags & m_usedFlags) == 0);
    // Mark the flags used
    m_usedFlags |= flags;

    if (type == ChangeType::On || type == ChangeType::OnOff)
    {
        m_invertBits |= flags;
    }
    if (type == ChangeType::OnOff || type == ChangeType::OffOn)
    {
        m_changingBits[m_numChangingBits++] = flags;
    }
}

void FlagCombiner::calcCombinations(List<uint32_t>& outCombinations) const
{
    const int numCombinations = getNumCombinations();
    outCombinations.setCount(numCombinations);
    uint32_t* dstCombinations = outCombinations.getBuffer();
    for (int i = 0; i < numCombinations; ++i)
    {
        dstCombinations[i] = getCombination(i);
    }
}

uint32_t FlagCombiner::getCombination(int index) const
{
    SLANG_ASSERT(index >= 0 && index < getNumCombinations());

    uint32_t combination = 0;
    uint32_t bit = 1;
    for (int i = m_numChangingBits - 1; i >= 0; --i, bit += bit)
    {
        combination |= ((bit & index) ? m_changingBits[i] : 0);
    }
    return combination ^ m_invertBits;
}

void FlagCombiner::reset()
{
    m_numChangingBits = 0;
    m_usedFlags = 0;
    m_invertBits = 0;
}

} // namespace gfx
