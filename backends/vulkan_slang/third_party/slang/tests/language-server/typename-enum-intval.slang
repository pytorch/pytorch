//TEST:LANG_SERVER(filecheck=CHECK):

namespace ns {
enum Test : uint32_t
{
    A = 1,
    B = 2,
}

struct Foo<let T : Test>
{
}
}

void f()
{
//HOVER:18,25
    ns.Foo<ns.Test.A> first;
//HOVER:20,27
    ns.Foo<ns.Test(3)> second;
}

// CHECK: ns.Foo<ns.Test.A>
// CHECK: ns.Foo<ns.Test(3)>