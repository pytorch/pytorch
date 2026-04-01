//TEST:SIMPLE(filecheck=METAL_ERROR): -target metal -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=METALLIB): -target metallib -entry computeMain -stage compute
// Metal currently lacks multisampled texture write support.
// Due to this, Metal compute test is disabled
//DISABLE_TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF): -slang -output-using-type -shaderobj -mtl
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF): -slang -output-using-type -shaderobj -vk

//METAL_ERROR: error 41402
//METALLIB: error 41402

//TEST_INPUT: RWTexture2D(format=RGBA8Sint, size=4, content = zero, sampleCount=two, mipMaps = 1):name outputTexture2DMS
RWTexture2DMS<int4> outputTexture2DMS;

//TEST_INPUT: RWTexture2D(format=RGBA8Sint, size=4, content = zero, arrayLength=2, sampleCount=two, mipMaps = 1):name outputTexture2DMSArray
RWTexture2DMSArray<int4> outputTexture2DMSArray;

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;


[numthreads(1,1,1)]
void computeMain()
{

    outputTexture2DMS[0, 0].xz = int2(1,2);
    outputTexture2DMS[int2(0, 0), 1].xz = int2(3,4);

    outputTexture2DMSArray[0, 0].xz = int2(1,2);
    outputTexture2DMSArray[int3(0, 0, 1), 1].xz = int2(3,4);

    outputBuffer[0] = uint(true
            && all(outputTexture2DMS[0, 0] == int4(1, 0, 2, 0)) == true
            && all(outputTexture2DMS[int2(0, 0), 1] == int4(3, 0, 4, 0))  == true

            && all(outputTexture2DMSArray[0, 0] == int4(1, 0, 2, 0)) == true
            && all(outputTexture2DMSArray[int3(0, 0, 1), 1] == int4(3, 0, 4, 0))  == true
            );
}

//BUF: 1
