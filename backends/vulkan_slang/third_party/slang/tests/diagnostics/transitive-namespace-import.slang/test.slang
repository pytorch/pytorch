//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

module test;

import b;

void testFunc()
{
    // CHECK-NOT: {{.*}}([[# @LINE + 1]]): error
    ns.f_b(); // OK.
    // CHECK: {{.*}}([[# @LINE + 1]]): error 30027:
    ns.f_a(); // Error.
}

namespace ns
{
    void testFunc2()
    {
        // CHECK-NOT: {{.*}}([[# @LINE + 1]]): error
        f_b(); // OK.
        // CHECK: {{.*}}([[# @LINE + 1]]): error 30015:
        f_a(); // Error.
    }
}

namespace ns2
{
    using namespace ns;

    void testFunc3()
    {
        // CHECK-NOT: {{.*}}([[# @LINE + 1]]): error
        f_b(); // OK.
        // CHECK: {{.*}}([[# @LINE + 1]]): error 30015:
        f_a(); // Error.
    }
}
