// bound-check-zero-index.slang

// Check 'zero indexing' bound check feature, supported by CPU and CUDA

// Currently zero index bound checking doesn't appear to be working properly for CUDA.
//TEST(compute):COMPARE_COMPUTE:-cuda -shaderobj -Xslang... -DSLANG_ENABLE_BOUND_ZERO_INDEX -X.
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj -Xslang... -DSLANG_ENABLE_BOUND_ZERO_INDEX -X.

//TEST_INPUT:ubuffer(data=[1 2 3 4]):name=byteAddressBuffer
ByteAddressBuffer byteAddressBuffer;

//TEST_INPUT:ubuffer(data=[0x10 0x20 0x30 0x40]):name=rwByteAddressBuffer
RWByteAddressBuffer rwByteAddressBuffer;

//TEST_INPUT:ubuffer(data=[0x100 0x200 0x300 0x400], stride=4):name=structuredBuffer
StructuredBuffer<int> structuredBuffer;

//TEST_INPUT:ubuffer(data=[0x1000 0x2000 0x3000 0x4000], stride=4):name=rwStructuredBuffer
RWStructuredBuffer<int> rwStructuredBuffer;

//TEST_INPUT:ubuffer(data=[-1 -1 -1 -1], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

//TEST_INPUT:ubuffer(data=[-1 -1 -1 -1], stride=4):out,name=outputBuffer2
RWStructuredBuffer<int> outputBuffer2;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
	int tid = dispatchThreadID.x;
    
    int fixedArray[3] = { 2, 5, 9};
    
    int total = 0;
    total += byteAddressBuffer.Load<int>(tid * 4);
    total += byteAddressBuffer.Load<int>(-tid * 4);
    
    total += rwByteAddressBuffer.Load<int>(tid * 4);
    total += rwByteAddressBuffer.Load<int>(-tid * 4);
    
    total += structuredBuffer[tid];
    total += structuredBuffer[-tid];
    
    total += rwStructuredBuffer[tid];
    total += rwStructuredBuffer[-tid];
    
    total += fixedArray[tid];
    total += fixedArray[-tid];
    
    outputBuffer[tid] = total;
    
    // NOTE! Different threads could access this if being performed in parallel.
    // So undeterministic if we write to same index (because out of range) when running in parallel
    // By just adding one, all indices should be hit once
    outputBuffer2[tid + 1] = total;
}