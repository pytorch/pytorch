//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

struct S
{
    int foo : 33;
    // CHECK: too-large.slang([[#@LINE-1]]): error 31300: bit-field size (33) exceeds the width of its type int (32)
    int64_t bar : 999;
    // CHECK: too-large.slang([[#@LINE-1]]): error 31300: bit-field size (999) exceeds the width of its type int64_t (64)
};

[numthreads(1, 1, 1)]
void computeMain()
{
}
