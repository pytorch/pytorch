// Test tuple comparison.

//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-dx12 -use-dxil -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[1 2 3 4], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    var t = makeTuple(1, 2, 2.0);
    var u = makeTuple(1, 3, 2.0);

    // CHECK: 1
    if (t._0_2 == u._0_2 && t != u)
        outputBuffer[0] = 1;

    int result = 1;
    if (t < t) result = 0;
    if (!(t <= t)) result = 0;
    if (t != t) result = 0;
    if (t > t) result = 0;
    if (!(t >= t)) result = 0;

    if (!(t < u)) result = 0;
    if (!(t <= u)) result = 0;
    if (!(t != u)) result = 0;
    if (t == u) result = 0;
    if (t > u) result = 0;
    if (t >= u) result = 0;

    if (!(u > t)) result = 0;
    if (!(u >= t)) result = 0;
    if (!(u != t)) result = 0;
    if (u == t) result = 0;
    if (u < t) result = 0;
    if (u <= t) result = 0;

    // CHECK: 1
    outputBuffer[1] = result;
}
