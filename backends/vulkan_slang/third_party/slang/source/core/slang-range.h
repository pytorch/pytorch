#ifndef SLANG_CORE_RANGE_H
#define SLANG_CORE_RANGE_H

namespace Slang
{
template<typename T>
struct Range
{
    T begin = 0;
    T end = 0;

    bool inRange(T val) const { return val >= begin && val < end; }
};

template<typename T>
Range<T> makeRange(T begin, T end)
{
    Range<T> result;
    result.begin = begin;
    result.end = end;
    return result;
}

} // namespace Slang

#endif // SLANG_CORE_RANGE_H
