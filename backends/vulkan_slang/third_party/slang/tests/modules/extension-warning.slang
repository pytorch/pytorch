//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):
module mymodule;

// This tests the issue described here: https://github.com/shader-slang/slang/issues/5995

//CHECK: result code = 0
//CHECK: tests/modules/implementing-with-ext.slang(1): warning 30506: implementing directive contains file extension in module name 'mymodule.slang'. Module names should not include extensions. The compiler will use 'mymodule' as the module name.
__include "implementing-with-ext";

int moduleFunction() { return 0; }
