//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute,vulkan):COMPARE_COMPUTE_EX:-vk -slang -compute -shaderobj

// When a function is passed a parameter that contains an array, it specialized it as a performance
// improvement for VK. If the struct contained a structured buffer, though it meant that the 
// function was specialized to have the structured buffer passed as a parameter. For GLSL/VK based
// targets Slang cannot pass StructuredBuffer types and so would subsequently fail in 
// GLSLSourceEmitter::emitSimpleTypeImpl when trying to output the StructuredBuffer type as a parameter
//
// This test, checks that when passing a struct with an array and structured buffer
// code functions as expected. 

struct Params
{
    RWStructuredBuffer<int> buffer;
    int a[2];
};

//TEST_INPUT:ubuffer(data=[9 9 9 9], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

//TEST_INPUT:ubuffer(data=[1 2 3 4], stride=4):name=anotherBuffer
RWStructuredBuffer<int> anotherBuffer;
// 
int doSomething(Params params, int v)
{
    return params.a[v & 1] - v;
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    Params params;
    params.buffer = anotherBuffer;
    params.a[0] = 1;
    params.a[1] = 2;

	int tid = int(dispatchThreadID.x);

	outputBuffer[tid] = doSomething(params, tid);
}