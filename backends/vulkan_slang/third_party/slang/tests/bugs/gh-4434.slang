//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-dx12 -use-dxil
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-mtl
//DISABLE_TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu
//DISABLE_TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cuda

// CHECK:      1
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// CHECK-NEXT: 1

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint tid : SV_GroupIndex)
{
    bool K = (bool)outputBuffer[tid];

    bool c = and(K, K);

    bool1 K1 = K;
    bool2 K2 = K;
    bool3 K3 = K;
    bool4 K4 = K;

    bool1 k = and(K, K);
    bool2 n = and(K, K);
    bool3 q = and(K, K);
    bool4 t = and(K, K);

    k = !and(k, false);
    n = !and(n, false);
    q = !and(q, false);
    t = !and(t, false);

    outputBuffer[tid] = all(k) && all(n) && all(q) && all(t);
}
