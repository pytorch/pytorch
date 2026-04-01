// unit-test-rtti.cpp

#include "../../source/core/slang-rtti-info.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

namespace
{ // anonymous

struct SomeStruct
{
    int a = 0;
    float b = 2.0f;
    String s;
    List<String> list;

    static const StructRttiInfo g_rttiInfo;
};

} // namespace

static const StructRttiInfo _makeSomeStructRtti()
{
    SomeStruct obj;
    StructRttiBuilder builder(&obj, "SomeStruct", nullptr);

    builder.addField("a", &obj.a);
    builder.addField("b", &obj.b);
    builder.addField("s", &obj.s);
    builder.addField("list", &obj.list);

    return builder.make();
}
/* static */ const StructRttiInfo SomeStruct::g_rttiInfo = _makeSomeStructRtti();

SLANG_UNIT_TEST(Rtti)
{
    using namespace Slang;

    const RttiInfo* types[] = {
        GetRttiInfo<int32_t>::get(),
        GetRttiInfo<int32_t[10]>::get(),
        GetRttiInfo<String>::get(),
        GetRttiInfo<List<String>>::get(),
        GetRttiInfo<List<List<String>>>::get(),
        GetRttiInfo<int32_t[2][3]>::get(),
        GetRttiInfo<SomeStruct>::get(),
        GetRttiInfo<SomeStruct*>::get(),
        GetRttiInfo<const float* const>::get(),
    };

    StringBuilder buf;

    for (auto type : types)
    {
        RttiInfo::append(type, buf);
        buf << "\n";
    }

    const char expected[] = "int32_t\n"
                            "int32_t[10]\n"
                            "String\n"
                            "List<String>\n"
                            "List<List<String>>\n"
                            "int32_t[2][3]\n"
                            "SomeStruct\n"
                            "SomeStruct*\n"
                            "float*\n";

    SLANG_CHECK(buf == expected)
}
