// unit-test-path.cpp

#include "core/slang-basic.h"
#include "core/slang-chunked-list.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

template<typename T>
static bool _checkArrayView(ArrayView<T> v0, ArrayView<T> v1)
{
    if (v0.getCount() != v1.getCount())
        return false;
    for (Index i = 0; i < v0.getCount(); i++)
        if (v0[i] != v1[i])
            return false;
    return true;
}

SLANG_UNIT_TEST(chunkedList)
{
    {
        ChunkedList<String> stringList;
        List<String*> ptrs;
        for (int i = 0; i < 256; i++)
        {
            ptrs.add(stringList.add(String(i)));
        }
        SLANG_CHECK(stringList.getCount() == 256);
        SLANG_CHECK(*(ptrs[128]) == "128");

        stringList.clearAndDeallocate();
        ptrs.clear();
        for (int i = 0; i < 64; i++)
        {
            ptrs.add(stringList.add(String(i)));
        }
        SLANG_CHECK(stringList.getCount() == 64);
        SLANG_CHECK(*(ptrs[32]) == "32");
    }
}
