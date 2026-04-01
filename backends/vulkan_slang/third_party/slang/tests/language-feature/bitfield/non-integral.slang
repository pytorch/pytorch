//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

struct S
{
    float foo : 10;
    // CHECK: non-integral.slang([[#@LINE-1]]): error 31301: bit-field type (float) must be an integral type
    S bar : 10;
    // CHECK: non-integral.slang([[#@LINE-1]]): error 31301: bit-field type (S) must be an integral type
};

[numthreads(1, 1, 1)]
void computeMain()
{
}
