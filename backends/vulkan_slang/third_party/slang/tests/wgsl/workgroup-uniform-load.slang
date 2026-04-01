// groupshared-ref-param.slang

//TEST:SIMPLE(filecheck=CHECK): -target wgsl -entry computeMain -stage compute


//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

groupshared uint sharedVal;

// Expect sharedVal to be passed by reference.
//
// CHECK: fn computeMain({{.*}}({{.*}})
// CHECK: {{.*}}workgroupUniformLoad(&((sharedVal_{{[a-zA-Z0-9_]*}}))){{.*}};
// CHECK: }


[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    int idx = dispatchThreadID.x;

    sharedVal = 1;
    outputBuffer[0] = workgroupUniformLoad(sharedVal);
}