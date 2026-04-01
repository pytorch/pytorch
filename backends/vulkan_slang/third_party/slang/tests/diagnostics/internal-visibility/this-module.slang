//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

module "this-module";

import "that-module";

void test()
{
    PublicStruct s;
    // CHECK:{{.*}}(11): error 30600:
    s.x = 1; // Error.
    // CHECK-NOT:{{.*}}error
    s.y = 2; // OK.
    publicMethod(); // OK.
    publicFunc2();  // OK.
    Namespace.publicVar = 1; // OK.
    Namespace.publicVar2 = 1; // OK.
    // CHECK:{{.*}}(19): error 30600:
    Namespace.internalVar = 1; // error.
    // CHECK:{{.*}}(21): error 30600:
    InternalEnum e; // Error.
    // CHECK:{{.*}}(23): error 30600:
    InternalStruct s1; // Error.
    // CHECK:{{.*}}(25): error 30600:
    internalMethod(); // Error.
}
