//DISABLED_TEST(compute, vulkan):COMPARE_COMPUTE_EX():-vk -compute -shaderobj -output-using-type -render-features wave-ops
//DISABLED_TEST(compute):COMPARE_COMPUTE_EX():-dx12 -profile sm_6_5 -use-dxil -compute -shaderobj -output-using-type -render-features wave-ops
//TEST:SIMPLE(filecheck=CHECK):-target hlsl -profile cs_6_5 -entry computeMain -line-directive-mode none
//TEST:SIMPLE(filecheck=CHECK):-target glsl -profile cs_6_5 -entry computeMain -line-directive-mode none

//
// This test checks that we fuse calls to saturated cooperation, even with different input values
//

//TEST_INPUT:ubuffer(data=[0 3 2 2], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

static int count = 0;

int coopAdd(float x, int i)
{
    count += i;
    return int(x) * 2;
}

int fallback1(float x, int)
{
    count = -1;
    return -1;
}

int coopMul(float x, float f)
{
    count *= int(f);
    return int(x) * 2;
}

int fallback2(float x, float)
{
    count = -1;
    return -1;
}

int coopExp(float x, int i)
{
    int c = count;
    count = 1;
    for(int j = 0; j < i; ++j)
        count *= c;
    return int(x) * 2;
}

// Make sure that we have enough invocations to saturate the first workgroup
[numthreads(128, 1, 1)]
void computeMain(uint tig : SV_GroupIndex)
{
    // The values we're cooperating over are {0, 2, 3}
    // We track the number of sets evaluated in the "count" variable, and write
    // that at index 0
    //
    // If these are not fused, then we'd expect count to be incremented three
    // times then doubled three times. What we want to see is
    // 0, +1, *2, **2, +1, *2, **2, +1, *2, **2 = 40804
    // If they're not interleaved then we'll see
    // 0, +1, +1, +1, *2, *2, *2, **2, **2, **2 = -1593835520 (having overflowed)
    let i = tig < 4 ? float(outputBuffer[tig]) : 0;
    let j = i + 1;
    let k = j + 1;
    let y = saturated_cooperation(coopAdd, fallback1, i, 1);
    let m = 2.f;
    let x = saturated_cooperation(coopMul, fallback2, j, m);
    // Put some calculation between the calls to check that the fusion still takes place
    let e = min(m*100, 2);
    let z = saturated_cooperation(coopExp, fallback1, k, int(e));

    // check that all the piping functions we invent during fusing is inlined away
    // CHECK:      = coopAdd
    // CHECK-NEXT: = coopMul
    // there is a tuple pack here
    // CHECK-NEXT: Tuple
    // CHECK-NEXT: = coopExp

    // CHECK:      = fallback1
    // CHECK-NEXT: = fallback2
    // CHECK-NEXT: Tuple
    // CHECK-NEXT: = fallback1

    if(tig < 4)
        outputBuffer[tig] = tig == 0 ? count : z;
}
