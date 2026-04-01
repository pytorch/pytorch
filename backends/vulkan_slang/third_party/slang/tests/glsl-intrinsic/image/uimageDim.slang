//TEST:SIMPLE(filecheck=CHECK_GLSL):  -allow-glsl -stage compute -entry computeMain -target glsl
//TEST:SIMPLE(filecheck=CHECK_SPV):  -allow-glsl -stage compute -entry computeMain -target spirv -emit-spirv-directly
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl -emit-spirv-directly

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
buffer MyBlockName2
{
    uint data[1];
} outputBuffer;

layout(local_size_x = 1) in;

//TEST_INPUT: set image_1d = RWTexture1D(format=R32Uint, size=4, content=one, mipMaps = 1)
uniform layout(binding=0,r32ui) uimage1D image_1d;
//TEST_INPUT: set image_buffer = RWTextureBuffer(format=R32Uint, stride=8, data=[1 1 1 1])
uniform layout(binding=1,r32ui) uimageBuffer image_buffer;
//TEST_INPUT: set image_1dArray = RWTexture1D(format=R32Uint, size=4, content=one, mipMaps = 1, arrayLength=2)
uniform layout(binding=2,r32ui) uimage1DArray image_1dArray;
//TEST_INPUT: set image_2d = RWTexture2D(format=R32Uint, size=4, content=one, mipMaps = 1)
uniform layout(binding=3,r32ui) uimage2D image_2d;
//TEST_INPUT: set image_2dRect = RWTexture2D(format=R32Uint, size=4, content=one, mipMaps = 1)
uniform layout(binding=4,r32ui) uimage2DRect image_2dRect;
//TEST_INPUT: set image_2dMultiSample = RWTexture2D(format=R32Uint, size=4, content=one, mipMaps = 1, sampleCount = two)
uniform layout(binding=5,r32ui) uimage2DMS image_2dMultiSample;
//TEST_INPUT: set image_2dArray = RWTexture2D(format=R32Uint, size=4, content=one, mipMaps = 1, arrayLength=2)
uniform layout(binding=6,r32ui) uimage2DArray image_2dArray;
//TEST_INPUT: set image_3d = RWTexture3D(format=R32Uint, size=4, content=one, mipMaps = 1)
uniform layout(binding=7,r32ui) uimage3D image_3d;
//TEST_INPUT: set image_cube = RWTextureCube(format=R32Uint, size=4, content=one, mipMaps = 1)
uniform layout(binding=8,r32ui) uimageCube image_cube;
//TEST_INPUT: set image_cubeArray = RWTextureCube(format=R32Uint, size=4, content=one, mipMaps = 1, arrayLength=2)
uniform layout(binding=9,r32ui) uimageCubeArray image_cubeArray;
//TEST_INPUT: set image_2dMultiSampleArray = RWTexture2D(format=R32Uint, size=4, content=one, mipMaps = 1, arrayLength=2, sampleCount = two)
uniform layout(binding=10,r32ui) uimage2DMSArray image_2dMultiSampleArray;

bool checkAllImageSize()
{
    return true
        && imageSize(image_1d) == int(4)
        && imageSize(image_buffer) == int(4)
        && imageSize(image_1dArray) == ivec2(4, 2)
        && imageSize(image_2d) == ivec2(4)
        && imageSize(image_2dArray) == ivec3(4, 4, 2)
        && imageSize(image_2dRect) == ivec2(4)
        && imageSize(image_2dMultiSample) == ivec2(4)
        && imageSize(image_3d) == ivec3(4)
        && imageSize(image_cube) == ivec2(4)
        && imageSize(image_cubeArray) == ivec3(4, 4, 2)
        && imageSize(image_2dMultiSampleArray) == ivec3(4, 4, 2)
        ;
}
bool checkAllImageLoad()
{
    return true
        && imageLoad(image_1d, 0).x == 1
        && imageLoad(image_buffer, 0).x == 1
        && imageLoad(image_1dArray, ivec2(0)).x == 1
        && imageLoad(image_2d, ivec2(0)).x == 1
        && imageLoad(image_2dRect, ivec2(0)).x == 1
        && imageLoad(image_2dMultiSample, ivec2(0), 1).x == 1
        && imageLoad(image_2dArray, ivec3(0)).x == 1
        && imageLoad(image_3d, ivec3(0)).x == 1
        && imageLoad(image_cube, ivec3(0)).x == 1
        && imageLoad(image_cubeArray, ivec3(0)).x == 1
        && imageLoad(image_cubeArray, ivec3(0)).x == 1
        && imageLoad(image_2dMultiSampleArray, ivec3(0), 1).x == 1
        ;
}
bool resetAllImageValues()
{
    imageStore(image_1d, 0, uvec4(1));
    imageStore(image_buffer, 0, uvec4(1));
    imageStore(image_1dArray, ivec2(0), uvec4(1));
    imageStore(image_2d, ivec2(0), uvec4(1));
    imageStore(image_2dRect, ivec2(0), uvec4(1));
    imageStore(image_2dMultiSample, ivec2(0), 1, uvec4(1));
    imageStore(image_2dArray, ivec3(0), uvec4(1));
    imageStore(image_3d, ivec3(0), uvec4(1));
    imageStore(image_cube, ivec3(0), uvec4(1));
    imageStore(image_cubeArray, ivec3(0), uvec4(1));
    imageStore(image_2dMultiSampleArray, ivec3(0), 1, uvec4(1));
    return true;
}
uint load_image_1d()
{
    return imageLoad(image_1d, 0).x;
}
uint load_image_buffer()
{
    return imageLoad(image_buffer, 0).x;
}
uint load_image_1dArray()
{
    return imageLoad(image_1dArray, ivec2(0)).x;
}
uint load_image_2d()
{
    return imageLoad(image_2d, ivec2(0)).x;
}
uint load_image_2dRect()
{
    return imageLoad(image_2dRect, ivec2(0)).x;
}
uint load_image_2dMultiSample()
{
    return imageLoad(image_2dMultiSample, ivec2(0), 1).x;
}
uint load_image_2dArray()
{
    return imageLoad(image_2dArray, ivec3(0)).x;
}
uint load_image_3d()
{
    return imageLoad(image_3d, ivec3(0)).x;
}
uint load_image_cube()
{
    return imageLoad(image_cube, ivec3(0)).x;
}
uint load_image_cubeArray()
{
    return imageLoad(image_cubeArray, ivec3(0)).x;
}
uint load_image_2dMultiSampleArray()
{
    return imageLoad(image_2dMultiSampleArray, ivec3(0), 1).x;
}
// requires ImageLoad test to pass
bool checkAllImageStore()
{
    bool loadCheck = true;
    imageStore(image_1d, 0, uvec4(0));
    loadCheck = loadCheck && load_image_1d() == 0;
    imageStore(image_buffer, 0, uvec4(0));
    loadCheck = loadCheck && load_image_buffer() == 0;
    imageStore(image_1dArray, ivec2(0), uvec4(0));
    loadCheck = loadCheck && load_image_1dArray() == 0;
    imageStore(image_2d, ivec2(0), uvec4(0));
    loadCheck = loadCheck && load_image_2d() == 0;
    imageStore(image_2dRect, ivec2(0), uvec4(0));
    loadCheck = loadCheck && load_image_2dRect() == 0;
    imageStore(image_2dMultiSample, ivec2(0), 1, uvec4(0));
    loadCheck = loadCheck && load_image_2dMultiSample() == 0;
    imageStore(image_2dArray, ivec3(0), uvec4(0));
    loadCheck = loadCheck && load_image_2dArray() == 0;
    imageStore(image_3d, ivec3(0), uvec4(0));
    loadCheck = loadCheck && load_image_3d() == 0;
    imageStore(image_cube, ivec3(0), uvec4(0));
    loadCheck = loadCheck && load_image_cube() == 0;
    imageStore(image_cubeArray, ivec3(0), uvec4(0));
    loadCheck = loadCheck && load_image_cubeArray() == 0;
    imageStore(image_2dMultiSampleArray, ivec3(0), 1, uvec4(0));
    loadCheck = loadCheck && load_image_2dMultiSampleArray() == 0;
    resetAllImageValues();
    return loadCheck;
}
bool checkAllImageSamples()
{
    resetAllImageValues();
    return true
        && imageSamples(image_2dMultiSample) == 2
        && imageSamples(image_2dMultiSampleArray) == 2
        ;
}
bool checkAllImageAtomicAdd()
{
    resetAllImageValues();
    return true
        && imageAtomicAdd(image_1d, 0, 0) == 1
        && load_image_1d() == 1
        && imageAtomicAdd(image_buffer, 0, 2) == 1
        && load_image_buffer() == 3
        && imageAtomicAdd(image_1dArray, ivec2(0), 0) == 1
        && load_image_1dArray() == 1
        && imageAtomicAdd(image_2d, ivec2(0), 2) == 1
        && load_image_2d() == 3
        && imageAtomicAdd(image_2dRect, ivec2(0), 2) == 1
        && load_image_2dRect() == 3
        && imageAtomicAdd(image_2dMultiSample, ivec2(0), 1, 2) == 1
        && load_image_2dMultiSample() == 3
        && imageAtomicAdd(image_2dArray, ivec3(0), 0) == 1
        && load_image_2dArray() == 1
        && imageAtomicAdd(image_3d, ivec3(0), 2) == 1
        && load_image_3d() == 3
        && imageAtomicAdd(image_cube, ivec3(0), 2) == 1
        && load_image_cube() == 3
        && imageAtomicAdd(image_cubeArray, ivec3(0), 2) == 1
        && load_image_cubeArray() == 3
        && imageAtomicAdd(image_2dMultiSampleArray, ivec3(0), 1, 2) == 1
        && load_image_2dMultiSampleArray() == 3
        ;
}
bool checkAllImageAtomicExchange()
{
    resetAllImageValues();
    return true
        && imageAtomicExchange(image_1d, 0, 0) == 1
        && load_image_1d() == 0
        && imageAtomicExchange(image_buffer, 0, 2) == 1
        && load_image_buffer() == 2
        && imageAtomicExchange(image_1dArray, ivec2(0), 0) == 1
        && load_image_1dArray() == 0
        && imageAtomicExchange(image_2d, ivec2(0), 2) == 1
        && load_image_2d() == 2
        && imageAtomicExchange(image_2dRect, ivec2(0), 2) == 1
        && load_image_2dRect() == 2
        && imageAtomicExchange(image_2dMultiSample, ivec2(0), 1, 2) == 1
        && load_image_2dMultiSample() == 2
        && imageAtomicExchange(image_2dArray, ivec3(0), 0) == 1
        && load_image_2dArray() == 0
        && imageAtomicExchange(image_3d, ivec3(0), 2) == 1
        && load_image_3d() == 2
        && imageAtomicExchange(image_cube, ivec3(0), 2) == 1
        && load_image_cube() == 2
        && imageAtomicExchange(image_cubeArray, ivec3(0), 2) == 1
        && load_image_cubeArray() == 2
        && imageAtomicExchange(image_2dMultiSampleArray, ivec3(0), 1, 2) == 1
        && load_image_2dMultiSampleArray() == 2
        ;
}
bool checkAllImageAtomicMin()
{
    resetAllImageValues();
    return true
        && imageAtomicMin(image_1d, 0, 0) == 1
        && load_image_1d() == 0
        && imageAtomicMin(image_buffer, 0, 2) == 1
        && load_image_buffer() == 1
        && imageAtomicMin(image_1dArray, ivec2(0), 0) == 1
        && load_image_1dArray() == 0
        && imageAtomicMin(image_2d, ivec2(0), 2) == 1
        && load_image_2d() == 1
        && imageAtomicMin(image_2dRect, ivec2(0), 2) == 1
        && load_image_2dRect() == 1
        && imageAtomicMin(image_2dMultiSample, ivec2(0), 1, 2) == 1
        && load_image_2dMultiSample() == 1
        && imageAtomicMin(image_2dArray, ivec3(0), 0) == 1
        && load_image_2dArray() == 0
        && imageAtomicMin(image_3d, ivec3(0), 2) == 1
        && load_image_3d() == 1
        && imageAtomicMin(image_cube, ivec3(0), 2) == 1
        && load_image_cube() == 1
        && imageAtomicMin(image_cubeArray, ivec3(0), 2) == 1
        && load_image_cubeArray() == 1
        && imageAtomicMin(image_2dMultiSampleArray, ivec3(0), 1, 2) == 1
        && load_image_2dMultiSampleArray() == 1
        ;
}
bool checkAllImageAtomicMax()
{
    resetAllImageValues();
    return true
        && imageAtomicMax(image_1d, 0, 0) == 1
        && load_image_1d() == 1
        && imageAtomicMax(image_buffer, 0, 2) == 1
        && load_image_buffer() == 2
        && imageAtomicMax(image_1dArray, ivec2(0), 0) == 1
        && load_image_1dArray() == 1
        && imageAtomicMax(image_2d, ivec2(0), 2) == 1
        && load_image_2d() == 2
        && imageAtomicMax(image_2dRect, ivec2(0), 2) == 1
        && load_image_2dRect() == 2
        && imageAtomicMax(image_2dMultiSample, ivec2(0), 1, 2) == 1
        && load_image_2dMultiSample() == 2
        && imageAtomicMax(image_2dArray, ivec3(0), 0) == 1
        && load_image_2dArray() == 1
        && imageAtomicMax(image_3d, ivec3(0), 2) == 1
        && load_image_3d() == 2
        && imageAtomicMax(image_cube, ivec3(0), 2) == 1
        && load_image_cube() == 2
        && imageAtomicMax(image_cubeArray, ivec3(0), 2) == 1
        && load_image_cubeArray() == 2
        && imageAtomicMax(image_2dMultiSampleArray, ivec3(0), 1, 2) == 1
        && load_image_2dMultiSampleArray() == 2
        ;
}
bool checkAllImageAtomicAnd()
{
    resetAllImageValues();
    return true
        && imageAtomicAnd(image_1d, 0, 1) == 1
        && load_image_1d() == 1
        && imageAtomicAnd(image_buffer, 0, 2) == 1
        && load_image_buffer() == 0
        && imageAtomicAnd(image_1dArray, ivec2(0), 1) == 1
        && load_image_1dArray() == 1
        && imageAtomicAnd(image_2d, ivec2(0), 2) == 1
        && load_image_2d() == 0
        && imageAtomicAnd(image_2dRect, ivec2(0), 2) == 1
        && load_image_2dRect() == 0
        && imageAtomicAnd(image_2dMultiSample, ivec2(0), 1, 2) == 1
        && load_image_2dMultiSample() == 0
        && imageAtomicAnd(image_2dArray, ivec3(0), 1) == 1
        && load_image_2dArray() == 1
        && imageAtomicAnd(image_3d, ivec3(0), 2) == 1
        && load_image_3d() == 0
        && imageAtomicAnd(image_cube, ivec3(0), 2) == 1
        && load_image_cube() == 0
        && imageAtomicAnd(image_cubeArray, ivec3(0), 2) == 1
        && load_image_cubeArray() == 0
        && imageAtomicAnd(image_2dMultiSampleArray, ivec3(0), 1, 2) == 1
        && load_image_2dMultiSampleArray() == 0
        ;
}
bool checkAllImageAtomicOr()
{
    resetAllImageValues();
    return true
        && imageAtomicOr(image_1d, 0, 1) == 1
        && load_image_1d() == 1
        && imageAtomicOr(image_buffer, 0, 2) == 1
        && load_image_buffer() == 3
        && imageAtomicOr(image_1dArray, ivec2(0), 1) == 1
        && load_image_1dArray() == 1
        && imageAtomicOr(image_2d, ivec2(0), 2) == 1
        && load_image_2d() == 3
        && imageAtomicOr(image_2dRect, ivec2(0), 2) == 1
        && load_image_2dRect() == 3
        && imageAtomicOr(image_2dMultiSample, ivec2(0), 1, 2) == 1
        && load_image_2dMultiSample() == 3
        && imageAtomicOr(image_2dArray, ivec3(0), 1) == 1
        && load_image_2dArray() == 1
        && imageAtomicOr(image_3d, ivec3(0), 2) == 1
        && load_image_3d() == 3
        && imageAtomicOr(image_cube, ivec3(0), 2) == 1
        && load_image_cube() == 3
        && imageAtomicOr(image_cubeArray, ivec3(0), 2) == 1
        && load_image_cubeArray() == 3
        && imageAtomicOr(image_2dMultiSampleArray, ivec3(0), 1, 2) == 1
        && load_image_2dMultiSampleArray() == 3
        ;
}
bool checkAllImageAtomicXor()
{
    resetAllImageValues();
    return true
        && imageAtomicXor(image_1d, 0, 1) == 1
        && load_image_1d() == 0
        && imageAtomicXor(image_buffer, 0, 2) == 1
        && load_image_buffer() == 3
        && imageAtomicXor(image_1dArray, ivec2(0), 1) == 1
        && load_image_1dArray() == 0
        && imageAtomicXor(image_2d, ivec2(0), 2) == 1
        && load_image_2d() == 3
        && imageAtomicXor(image_2dRect, ivec2(0), 2) == 1
        && load_image_2dRect() == 3
        && imageAtomicXor(image_2dMultiSample, ivec2(0), 1, 2) == 1
        && load_image_2dMultiSample() == 3
        && imageAtomicXor(image_2dArray, ivec3(0), 1) == 1
        && load_image_2dArray() == 0
        && imageAtomicXor(image_3d, ivec3(0), 2) == 1
        && load_image_3d() == 3
        && imageAtomicXor(image_cube, ivec3(0), 2) == 1
        && load_image_cube() == 3
        && imageAtomicXor(image_cubeArray, ivec3(0), 2) == 1
        && load_image_cubeArray() == 3
        && imageAtomicXor(image_2dMultiSampleArray, ivec3(0), 1, 2) == 1
        && load_image_2dMultiSampleArray() == 3
        ;
}
bool checkAllImageAtomicCompSwap()
{
    resetAllImageValues();
    return true
        && imageAtomicCompSwap(image_1d, 0, 0, 2) == 1
        && load_image_1d() == 1
        && imageAtomicCompSwap(image_buffer, 0, 1, 2) == 1
        && load_image_buffer() == 2
        && imageAtomicCompSwap(image_1dArray, ivec2(0), 0, 2) == 1
        && load_image_1dArray() == 1
        && imageAtomicCompSwap(image_2d, ivec2(0), 1, 2) == 1
        && load_image_2d() == 2
        && imageAtomicCompSwap(image_2dRect, ivec2(0), 1, 2) == 1
        && load_image_2dRect() == 2
        && imageAtomicCompSwap(image_2dMultiSample, ivec2(0), 1, 1, 2) == 1
        && load_image_2dMultiSample() == 2
        && imageAtomicCompSwap(image_2dArray, ivec3(0), 0, 2) == 1
        && load_image_2dArray() == 1
        && imageAtomicCompSwap(image_3d, ivec3(0), 1, 2) == 1
        && load_image_3d() == 2
        && imageAtomicCompSwap(image_cube, ivec3(0), 1, 2) == 1
        && load_image_cube() == 2
        && imageAtomicCompSwap(image_cubeArray, ivec3(0), 1, 2) == 1
        && load_image_cubeArray() == 2
        && imageAtomicCompSwap(image_2dMultiSampleArray, ivec3(0), 1, 1, 2) == 1
        && load_image_2dMultiSampleArray() == 2
        ;
}

// CHECK_GLSL: void main(
// CHECK_SPV: OpEntryPoint
void computeMain()
{
    outputBuffer.data[0] = true
        && checkAllImageSize()
        && checkAllImageLoad()
        && checkAllImageStore()
        && checkAllImageSamples()
        && checkAllImageAtomicAdd()
        && checkAllImageAtomicExchange()
        && checkAllImageAtomicMin()
        && checkAllImageAtomicMax()
        && checkAllImageAtomicAnd()
        && checkAllImageAtomicOr()
        && checkAllImageAtomicXor()
        && checkAllImageAtomicCompSwap()
        ;
    // BUF: 1
}
