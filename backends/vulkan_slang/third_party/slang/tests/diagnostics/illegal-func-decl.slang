// illegal-func-decl.slang

// This test checks that the in/out/inout modifiers in function declarations must
// be consistent with the function's definition, and slang can diagnose the inconsistency.

//TEST:COMPILE: tests/diagnostics/illegal-func-decl-module.slang -o tests/diagnostics/illegal-func-decl-module.slang-module

//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK1): -r tests/diagnostics/illegal-func-decl-module.slang-module -DTEST1 -target spirv -o illegal-func-decl.spv
//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK2): -r tests/diagnostics/illegal-func-decl-module.slang-module -DTEST2 -target spirv -o illegal-func-decl.spv -skip-spirv-validation
//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK3): -r tests/diagnostics/illegal-func-decl-module.slang-module -DTEST3 -target spirv -o illegal-func-decl.spv
//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK4): -r tests/diagnostics/illegal-func-decl-module.slang-module -DTEST4 -target spirv -o illegal-func-decl.spv -skip-spirv-validation

#ifdef TEST1
// CHECK1: ([[# @LINE+1]]): error 45001: unresolved external symbol 'libraryFunction'.
extern float libraryFunction(inout float a);    // invalid: 'a' is 'in'
#endif

#ifdef TEST2
// CHECK2-NOT: ([[# @LINE+1]]): error 45001: unresolved external symbol 'libraryFunction1'.
extern float libraryFunction1(inout float b);   // valid
#endif

#ifdef TEST3
// CHECK3: ([[# @LINE+1]]): error 45001: unresolved external symbol 'libraryFunction2'.
extern float libraryFunction2(inout float a, in float b, float c);   // valid: 'c' is 'inout'
#endif

#ifdef TEST4
// CHECK4-NOT: ([[# @LINE+1]]): error 45001: unresolved external symbol 'libraryFunction3'.
export float libraryFunction3(float a);         // valid: 'in' is the default is not specified
#endif

[shader("compute")]
[numthreads(1, 1, 1)]
void main(out float4 col : SV_Target0, bool isFrontHit)
{
    float a = 5;
    float b = 7;
    float c = 7;

#ifdef TEST1
    col.x = libraryFunction(a);
#endif

#ifdef TEST2
    col.y = libraryFunction1(b);
#endif

#ifdef TEST3
    col.z = libraryFunction2(a, b, c);
#endif

#ifdef TEST4
    col.w = libraryFunction3(a);
#endif
}


