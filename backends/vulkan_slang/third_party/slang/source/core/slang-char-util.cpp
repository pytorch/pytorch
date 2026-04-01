#include "slang-char-util.h"

namespace Slang
{

/* static */ CharUtil::CharFlagMap CharUtil::makeCharFlagMap()
{
    CharUtil::CharFlagMap map;
    memset(&map, 0, sizeof(map));

    {
        for (Index i = 'a'; i <= 'z'; ++i)
        {
            map.flags[i] |= Flag::Lower;
        }
    }
    {
        for (Index i = 'A'; i <= 'Z'; ++i)
        {
            map.flags[i] |= Flag::Upper;
        }
    }
    {
        for (Index i = '0'; i <= '9'; ++i)
        {
            map.flags[i] |= Flag::Digit | Flag::HexDigit;
        }
    }
    {
        for (Index i = 'a'; i <= 'f'; ++i)
        {
            map.flags[i] |= Flag::HexDigit;
            map.flags[size_t(CharUtil::toUpper(char(i)))] |= Flag::HexDigit;
        }
    }

    {
        map.flags[size_t(' ')] |= Flag::HorizontalWhitespace;
        map.flags[size_t('\t')] |= Flag::HorizontalWhitespace;
    }

    {
        map.flags[size_t('\n')] |= Flag::VerticalWhitespace;
        map.flags[size_t('\r')] |= Flag::VerticalWhitespace;
    }

    return map;
}

/* static */ int CharUtil::_ensureLink()
{
    return makeCharFlagMap().flags[0];
}

/* static */ const CharUtil::CharFlagMap CharUtil::g_charFlagMap = makeCharFlagMap();

} // namespace Slang
