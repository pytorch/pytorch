// pointer-self-reference.slang

//TEST:SIMPLE(filecheck=CHECK): -target cuda -entry computeMain -stage compute
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUFFER): -slang -compute -output-using-type -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUFFER): -vk -compute -output-using-type -shaderobj


//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

struct Thing
{
    int value;
    int bigArray[128];

    // Check that we are not inserting local variables that are copies of `this` parameter.
    
    // CHECK: __device__ int Thing_getSum{{.*}}(Thing{{.*}} * this{{.*}})
    // CHECK-NOT: Thing{{[a-zA-Z0-9_]*}} {{[a-zA-Z0-9_]+}}
    // CHECK: }
    [constref]
    int getSum()
    {
        int result = 0;
        for (int i = 0; i < 128; i++)
        {
            result += bigArray[i];
        }
        return result;
    }
};

// Check that we are not inserting local variables that are copies of `thing` parameter.

// CHECK: __device__ int test{{.*}}(Thing{{.*}} * thing{{.*}})
// CHECK-NOT: Thing{{[a-zA-Z0-9_]*}} {{[a-zA-Z0-9_]+}}
// CHECK: }

int test(__constref Thing thing)
{
    int sum0 = thing.getSum();
    AllMemoryBarrier();
    int sum1 = thing.getSum();
    return sum0 + sum1;
}

int caller(Thing thing)
{
    return test(thing);
}

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    int idx = dispatchThreadID.x;

    Thing thing = {};

    thing.bigArray[0] = 100;
    thing.bigArray[99] = 1;

    // BUFFER: 202
    outputBuffer[idx] = caller(thing);
}
