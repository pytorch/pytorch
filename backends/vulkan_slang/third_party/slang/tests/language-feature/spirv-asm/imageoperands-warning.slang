//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -shaderobj -emit-spirv-directly
//DIAGNOSTIC_TEST:SIMPLE(filecheck=DIAG):

//TEST_INPUT:ubuffer(data=[1 2 3 4], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

// CHECK:      1
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// CHECK-NEXT: 1

//TEST_INPUT: Texture2D(size=4, content = one):name t2D
Texture2D<float> t2D;
//TEST_INPUT: Sampler:name samplerState
SamplerState samplerState;

//
// This test tests that we don't emit a warning for the variadic operand list
// to OpImageSample. The reason to test for this is that "ImageOperands" aren't
// specified as variadic in the json SPIR-V instruction descriptions.
//
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int i = dispatchThreadID.x;
    float2 loc = float2(0.5, 0.5);
    float lod = 0;
    float r = spirv_asm
    {
        // The type of our sampled image
        %sampledImageType = OpTypeSampledImage $$Texture2D<float>;
        // Combine the image with the sampler
        %sampledImage : %sampledImageType = OpSampledImage $t2D $samplerState;
        // Perform a sample
        %sampled : __sampledType(float) = OpImageSampleExplicitLod %sampledImage $loc
            // Put some sampling operands in here to check that they don't warn
            // DIAG-NOT: warning{{.*}}too many operands
            Lod $lod;
        // Samples in SPIR-V always return a 4-vector of the component type,
        // the __truncate function will drop elements so it fits into the
        // desired output type
        __truncate $$float result __sampledType(float) %sampled;
    };
    outputBuffer[i] = int(r);
}
