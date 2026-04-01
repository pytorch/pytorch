//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target spirv -entry main -stage compute

extern static const int size = 1;

struct V
{
    // CHECK-DAG: ([[# @LINE+1]]): error 30070
    int c[];
    int b[size];
    int a[];
}

struct Q
{
    // CHECK-DAG: ([[# @LINE+1]]): error 30070
    V v1;
    V v2;
}

[numthreads(1,1,1)]
void main()
{
    V v;
    int i = 2;
}