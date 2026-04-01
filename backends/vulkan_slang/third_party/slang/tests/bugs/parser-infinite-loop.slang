//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

// CHECK: error 20001:
struct test
{
    float3 field;
}
void f()
{
    test x; x
    vector<int,2> v;
}
