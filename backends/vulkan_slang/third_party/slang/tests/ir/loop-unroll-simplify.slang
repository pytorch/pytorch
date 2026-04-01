//TEST:SIMPLE(filecheck=CHECK): -entry main -target hlsl -profile cs_6_3

RWStructuredBuffer<int> output;

void main()
{
    // Test that the unrolled code does not have trivial single iteration loops.
    // CHECK-NOT: break
    [ForceUnroll]
    for (int i = 0; i < 2; i++)
    {
        output[i] = i;
    }
}
