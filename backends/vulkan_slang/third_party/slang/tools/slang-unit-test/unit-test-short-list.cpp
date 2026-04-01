// unit-test-path.cpp

#include "core/slang-basic.h"
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

SLANG_UNIT_TEST(shortList)
{
    {
        ShortList<String, 4> shortList = {"a", "b", "c"};
        shortList.add("d");
        auto arrayView = shortList.getArrayView();
        SLANG_CHECK(arrayView.ownsStorage == false);
        SLANG_CHECK(
            _checkArrayView(arrayView.arrayView, List<String>{"a", "b", "c", "d"}.getArrayView()));
        shortList.add("e");
        auto arrayView2 = shortList.getArrayView();
        SLANG_CHECK(arrayView2.ownsStorage == true);
        SLANG_CHECK(_checkArrayView(
            arrayView2.arrayView,
            List<String>{"a", "b", "c", "d", "e"}.getArrayView()));
        auto arrayView3 = shortList.getArrayView(0, 2);
        SLANG_CHECK(arrayView3.ownsStorage == false);
        SLANG_CHECK(_checkArrayView(arrayView3.arrayView, List<String>{"a", "b"}.getArrayView()));
        auto arrayView4 = shortList.getArrayView(4, 1);
        SLANG_CHECK(arrayView4.ownsStorage == false);
        SLANG_CHECK(_checkArrayView(arrayView4.arrayView, List<String>{"e"}.getArrayView()));
        auto arrayView5 = shortList.getArrayView(2, 3);
        SLANG_CHECK(arrayView5.ownsStorage == true);
        SLANG_CHECK(
            _checkArrayView(arrayView5.arrayView, List<String>{"c", "d", "e"}.getArrayView()));

        ShortList<String, 1> copy2;
        ShortList<String, 2> copy1;
        copy1 = shortList;
        for (auto item : copy1)
            copy2.add(item);
        SLANG_CHECK(_checkArrayView(
            copy2.getArrayView().arrayView,
            List<String>{"a", "b", "c", "d", "e"}.getArrayView()));

        SLANG_CHECK(copy2.indexOf("a") == 0);
        SLANG_CHECK(copy2.indexOf("e") == 4);

        SLANG_CHECK(copy2.lastIndexOf("a") == 0);
        SLANG_CHECK(copy2.lastIndexOf("e") == 4);

        copy2.compress();
        copy2.add("f");
        copy2.fastRemove("c");
        copy2.compress();
        SLANG_CHECK(_checkArrayView(
            copy2.getArrayView().arrayView,
            List<String>{"a", "b", "f", "d", "e"}.getArrayView()));

        shortList.removeLast();
        shortList.removeLast();
        shortList.compress();
        SLANG_CHECK(_checkArrayView(
            shortList.getArrayView().arrayView,
            List<String>{"a", "b", "c"}.getArrayView()));
        shortList.add("d");
        shortList.add("e");
        SLANG_CHECK(_checkArrayView(
            shortList.getArrayView().arrayView,
            List<String>{"a", "b", "c", "d", "e"}.getArrayView()));
    }
}
