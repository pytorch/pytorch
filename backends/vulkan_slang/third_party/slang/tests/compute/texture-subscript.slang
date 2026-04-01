//TEST:SIMPLE(filecheck=METAL): -target metal -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=METALLIB): -target metallib -entry computeMain -stage compute
// Metal lacks RWTexture GFX backend support. 
// Due to this, Metal compute test is disabled
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF): -slang -output-using-type -shaderobj -mtl
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF): -slang -output-using-type -shaderobj -vk

//METAL-NOT: error 41402
//METALLIB: @computeMain

//TEST_INPUT: RWTexture1D(format=RGBA8Sint, size=8, content = zero, mipMaps = 1):name outputTexture1D
RWTexture1D<int4> outputTexture1D;

//TEST_INPUT: RWTexture2D(format=RGBA8Sint, size=8, content = zero, mipMaps = 1):name outputTexture2D
RWTexture2D<int4> outputTexture2D;

//TEST_INPUT: RWTexture3D(format=RGBA8Sint, size=8, content = zero, mipMaps = 1):name outputTexture3D
RWTexture3D<int4> outputTexture3D;

//TEST_INPUT: RWTexture2D(format=RGBA8Sint, size=4, content = zero, arrayLength=2, mipMaps = 1):name outputTexture2DArray
RWTexture2DArray<int4> outputTexture2DArray;

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;


[numthreads(1,1,1)]
void computeMain()
{
    outputTexture1D[0].xz = int2(1, 2).xx;
    AllMemoryBarrier();
    outputTexture1D[1].x = int2(3, 4).y;
    AllMemoryBarrier();
    outputTexture2D[0].xz = int2(1, 2).xx;
    AllMemoryBarrier();
    outputTexture2D[int2(0, 1)].x = int2(3, 4).y;
    AllMemoryBarrier();
    outputTexture3D[0].xz = int2(1, 2).xx;
    AllMemoryBarrier();
    outputTexture3D[int3(0, 0, 1)].x = int2(3, 4).y;
    AllMemoryBarrier();
    outputTexture2DArray[0].xz = int2(1, 2);
    AllMemoryBarrier();
    outputTexture2DArray[int3(0, 0, 1)].xz = int2(3, 4);
    AllMemoryBarrier();
    outputBuffer[0] = uint(true
            && all(outputTexture1D[0] == int4(1, 0, 1, 0)) == true
            && all(outputTexture1D[1] == int4(4, 0, 0, 0)) == true

            && all(outputTexture2D[0] == int4(1, 0, 1, 0)) == true
            && all(outputTexture2D[int2(0, 1)] == int4(4, 0, 0, 0)) == true

            && all(outputTexture3D[0] == int4(1, 0, 1, 0)) == true
            && all(outputTexture3D[int3(0, 0, 1)] == int4(4, 0, 0, 0)) == true

            && all(outputTexture2DArray[0] == int4(1, 0, 2, 0)) == true
            && all(outputTexture2DArray[int3(0, 0, 1)] == int4(3, 0, 4, 0))  == true
            );
}

//BUF: 1
