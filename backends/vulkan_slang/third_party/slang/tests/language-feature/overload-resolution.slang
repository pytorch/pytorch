//TEST:SIMPLE(filecheck=CHECK): -target hlsl -stage compute -entry main
RWStructuredBuffer<float> result;

[ForceInline]
float myF(inout int a, int b)
{
    return a + b;
}

[ForceInline]
float myF(inout uint a, uint b)
{
    return a - b;
}

[ForceInline]
T myGenF<T : __BuiltinIntegerType>(inout T a, T b)
{
    if (__isSignedInt<T>())
    {
        return a + b;
    }
    else
    {
        return a - b;
    }
}
// CHECK: result{{.*}}[int(0)] = 1
// CHECK: result{{.*}}[int(1)] = 4
// CHECK: result{{.*}}[int(2)] = 1
// CHECK: result{{.*}}[int(3)] = 4
[numthreads(1,1,1)]
void main()
{
    int ic = 1;
    uint a = 2;
    result[0] = myF(a, ic);

    int b = 3;
    uint uc = 1;
    result[1] = myF(b, uc);

    result[2] = myGenF(a, ic);
    result[3] = myGenF(b, uc);
}