//TEST:SIMPLE(filecheck=CHECK): -target hlsl -profile cs_5_0 -entry computeMain -line-directive-mode none


RWStructuredBuffer<float> outputBuffer;

int test1()
{
    float t = 0;
    for (int i = 0; i < 5; i++)
    {
        if (i < 3)
            t = t + 1;
        else
            t = t + 2;
        // we should coalesce the phi after the `if` the and phi of the `for` loop.
    }
    outputBuffer[0] = t;
    return 0;
}
// CHECK: int test1{{[_0-9]*}}()
// CHECK-NOT: float t_1
// CHECK: return

int test2()
{
    float v = 0;
    for (int i = 0; i < 5; i++)
    {
        float ov = v;
        if (i < 3)
            v = v + 1;
        else
            v = v + 2;
        // use of ot here means we can't coalesce the phis of the `if` and the `for` loop.
        outputBuffer[1] = ov;
    }
    outputBuffer[0] = v;
    return 0;
}
// CHECK: int test2{{[_0-9]*}}()
// CHECK: float v_1
// CHECK: return

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    test1();
    test2();
}
