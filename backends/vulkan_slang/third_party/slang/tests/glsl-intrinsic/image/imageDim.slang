//TEST:SIMPLE(filecheck=CHECK_GLSL):  -allow-glsl -stage compute -entry computeMain -target glsl
//TEST:SIMPLE(filecheck=CHECK_SPV):  -allow-glsl -stage compute -entry computeMain -target spirv -emit-spirv-directly
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl -emit-spirv-directly

// shader_atomic_float2 is currently a very new extension; most hardware lacks
// this extension and will fail this test if attempting to use atomic_float2
// operations.
// #define TEST_when_shader_atomic_float2_is_available

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
buffer MyBlockName2
{
    uint data[1];
} outputBuffer;

layout(local_size_x = 1) in;

//TEST_INPUT: set image_1d = RWTexture1D(format=R32Float, size=4, content=one, mipMaps = 1)
uniform layout(binding=0,r32f) image1D image_1d;
//TEST_INPUT: set image_buffer = RWTextureBuffer(format=R32Float, stride=4, data=[1.0f 1.0f 1.0f 1.0f])
uniform layout(binding=1,r32f) imageBuffer image_buffer;
//TEST_INPUT: set image_1dArray = RWTexture1D(format=R32Float, size=4, content=one, mipMaps = 1, arrayLength=2)
uniform layout(binding=2,r32f) image1DArray image_1dArray;
//TEST_INPUT: set image_2d = RWTexture2D(format=R32Float, size=4, content=one, mipMaps = 1)
uniform layout(binding=3,r32f) image2D image_2d;
//TEST_INPUT: set image_2dRect = RWTexture2D(format=R32Float, size=4, content=one, mipMaps = 1)
uniform layout(binding=4,r32f) image2DRect image_2dRect;
//TEST_INPUT: set image_2dMultiSample = RWTexture2D(format=R32Float, size=4, content=one, mipMaps = 1, sampleCount = two)
uniform layout(binding=5,r32f) image2DMS image_2dMultiSample;
//TEST_INPUT: set image_2dArray = RWTexture2D(format=R32Float, size=4, content=one, mipMaps = 1, arrayLength=2)
uniform layout(binding=6,r32f) image2DArray image_2dArray;
//TEST_INPUT: set image_3d = RWTexture3D(format=R32Float, size=4, content=one, mipMaps = 1)
uniform layout(binding=7,r32f) image3D image_3d;
//TEST_INPUT: set image_cube = RWTextureCube(format=R32Float, size=4, content=one, mipMaps = 1)
uniform layout(binding=8,r32f) imageCube image_cube;
//TEST_INPUT: set image_cubeArray = RWTextureCube(format=R32Float, size=4, content=one, mipMaps = 1, arrayLength=2)
uniform layout(binding=9,r32f) imageCubeArray image_cubeArray;
//TEST_INPUT: set image_2dMultiSampleArray = RWTexture2D(format=R32Float, size=4, content=one, mipMaps = 1, arrayLength=2, sampleCount = two)
uniform layout(binding=10,r32f) image2DMSArray image_2dMultiSampleArray;

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
    imageStore(image_1d, 0, vec4(1));
    imageStore(image_buffer, 0, vec4(1));
    imageStore(image_1dArray, ivec2(0), vec4(1));
    imageStore(image_2d, ivec2(0), vec4(1));
    imageStore(image_2dRect, ivec2(0), vec4(1));
    imageStore(image_2dMultiSample, ivec2(0), 1,  vec4(1));
    imageStore(image_2dArray, ivec3(0), vec4(1));
    imageStore(image_3d, ivec3(0), vec4(1));
    imageStore(image_cube, ivec3(0), vec4(1));
    imageStore(image_cubeArray, ivec3(0), vec4(1));
    imageStore(image_2dMultiSampleArray, ivec3(0), 1, vec4(1));
    return true;
}
float load_image_1d()
{
    return imageLoad(image_1d, 0).x;
}
float load_image_buffer()
{
    return imageLoad(image_buffer, 0).x;
}
float load_image_1dArray()
{
    return imageLoad(image_1dArray, ivec2(0)).x;
}
float load_image_2d()
{
    return imageLoad(image_2d, ivec2(0)).x;
}
float load_image_2dRect()
{
    return imageLoad(image_2dRect, ivec2(0)).x;
}
float load_image_2dMultiSample()
{
    return imageLoad(image_2dMultiSample, ivec2(0), 1).x;
}
float load_image_2dArray()
{
    return imageLoad(image_2dArray, ivec3(0)).x;
}
float load_image_3d()
{
    return imageLoad(image_3d, ivec3(0)).x;
}
float load_image_cube()
{
    return imageLoad(image_cube, ivec3(0)).x;
}
float load_image_cubeArray()
{
    return imageLoad(image_cubeArray, ivec3(0)).x;
}
float load_image_2dMultiSampleArray()
{
    return imageLoad(image_2dMultiSampleArray, ivec3(0), 1).x;
}
// requires ImageLoad test to pass
bool checkAllImageStore()
{
    bool loadCheck = true;
    imageStore(image_1d, 0, vec4(0));
    loadCheck = loadCheck && load_image_1d() == 0;
    imageStore(image_buffer, 0, vec4(0));
    loadCheck = loadCheck && load_image_buffer() == 0;
    imageStore(image_1dArray, ivec2(0), vec4(0));
    loadCheck = loadCheck && load_image_1dArray() == 0;
    imageStore(image_2d, ivec2(0), vec4(0));
    loadCheck = loadCheck && load_image_2d() == 0;
    imageStore(image_2dRect, ivec2(0), vec4(0));
    loadCheck = loadCheck && load_image_2dRect() == 0;
    imageStore(image_2dMultiSample, ivec2(0), 1,  vec4(0));
    loadCheck = loadCheck && load_image_2dMultiSample() == 0;
    imageStore(image_2dArray, ivec3(0), vec4(0));
    loadCheck = loadCheck && load_image_2dArray() == 0;
    imageStore(image_3d, ivec3(0), vec4(0));
    loadCheck = loadCheck && load_image_3d() == 0;
    imageStore(image_cube, ivec3(0), vec4(0));
    loadCheck = loadCheck && load_image_cube() == 0;
    imageStore(image_cubeArray, ivec3(0), vec4(0));
    loadCheck = loadCheck && load_image_cubeArray() == 0;
    imageStore(image_2dMultiSampleArray, ivec3(0), 1, vec4(0));
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
        && imageAtomicAdd(image_1d, 0, 0.0f) == 1
        && load_image_1d() == 1
        && imageAtomicAdd(image_buffer, 0, 2.0f) == 1
        && load_image_buffer() == 3
        && imageAtomicAdd(image_1dArray, ivec2(0), 0.0f) == 1
        && load_image_1dArray() == 1
        && imageAtomicAdd(image_2d, ivec2(0), 2.0f) == 1
        && load_image_2d() == 3
        && imageAtomicAdd(image_2dRect, ivec2(0), 2.0f) == 1
        && load_image_2dRect() == 3
        && imageAtomicAdd(image_2dMultiSample, ivec2(0), 1, 2.0f) == 1
        && load_image_2dMultiSample() == 3
        && imageAtomicAdd(image_2dArray, ivec3(0), 0.0f) == 1
        && load_image_2dArray() == 1
        && imageAtomicAdd(image_3d, ivec3(0), 2.0f) == 1
        && load_image_3d() == 3
        && imageAtomicAdd(image_cube, ivec3(0), 2.0f) == 1
        && load_image_cube() == 3
        && imageAtomicAdd(image_cubeArray, ivec3(0), 2.0f) == 1
        && load_image_cubeArray() == 3
        && imageAtomicAdd(image_2dMultiSampleArray, ivec3(0), 1, 2) == 1
        && load_image_2dMultiSampleArray() == 3
        ;
}
bool checkAllImageAtomicExchange()
{
    resetAllImageValues();
    return true
        && imageAtomicExchange(image_1d, 0, 0.0f) == 1
        && load_image_1d() == 0
        && imageAtomicExchange(image_buffer, 0, 2.0f) == 1
        && load_image_buffer() == 2
        && imageAtomicExchange(image_1dArray, ivec2(0), 0.0f) == 1
        && load_image_1dArray() == 0
        && imageAtomicExchange(image_2d, ivec2(0), 2.0f) == 1
        && load_image_2d() == 2
        && imageAtomicExchange(image_2dRect, ivec2(0), 2.0f) == 1
        && load_image_2dRect() == 2
        && imageAtomicExchange(image_2dMultiSample, ivec2(0), 1, 2.0f) == 1
        && load_image_2dMultiSample() == 2
        && imageAtomicExchange(image_2dArray, ivec3(0), 0.0f) == 1
        && load_image_2dArray() == 0
        && imageAtomicExchange(image_3d, ivec3(0), 2.0f) == 1
        && load_image_3d() == 2
        && imageAtomicExchange(image_cube, ivec3(0), 2.0f) == 1
        && load_image_cube() == 2
        && imageAtomicExchange(image_cubeArray, ivec3(0), 2.0f) == 1
        && load_image_cubeArray() == 2
        && imageAtomicExchange(image_2dMultiSampleArray, ivec3(0), 1, 2.0f) == 1
        && load_image_2dMultiSampleArray() == 2
        ;
}
#ifdef TEST_when_shader_atomic_float2_is_available
bool checkAllImageAtomicMin()
{
    resetAllImageValues();
    return true
        && imageAtomicMin(image_1d, 0, 0.0f) == 1
        && load_image_1d() == 0
        && imageAtomicMin(image_buffer, 0, 2.0f) == 1
        && load_image_buffer() == 1
        && imageAtomicMin(image_1dArray, ivec2(0), 0.0f) == 1
        && load_image_1dArray() == 0
        && imageAtomicMin(image_2d, ivec2(0), 2.0f) == 1
        && load_image_2d() == 1
        && imageAtomicMin(image_2dRect, ivec2(0), 2.0f) == 1
        && load_image_2dRect() == 1
        && imageAtomicMin(image_2dMultiSample, ivec2(0), 1, 2.0f) == 1
        && load_image_2dMultiSample() == 1
        && imageAtomicMin(image_2dArray, ivec3(0), 0.0f) == 1
        && load_image_2dArray() == 0
        && imageAtomicMin(image_3d, ivec3(0), 2.0f) == 1
        && load_image_3d() == 1
        && imageAtomicMin(image_cube, ivec3(0), 2.0f) == 1
        && load_image_cube() == 1
        && imageAtomicMin(image_cubeArray, ivec3(0), 2.0f) == 1
        && load_image_cubeArray() == 1
        && imageAtomicMin(image_2dMultiSampleArray, ivec3(0), 1, 2.0f) == 1
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
        && imageAtomicMax(image_1dArray, ivec2(0), 0.0f) == 1
        && load_image_1dArray() == 1
        && imageAtomicMax(image_2d, ivec2(0), 2.0f) == 1
        && load_image_2d() == 2
        && imageAtomicMax(image_2dRect, ivec2(0), 2.0f) == 1
        && load_image_2dRect() == 2
        && imageAtomicMax(image_2dMultiSample, ivec2(0), 1, 2.0f) == 1
        && load_image_2dMultiSample() == 2
        && imageAtomicMax(image_2dArray, ivec3(0), 0.0f) == 1
        && load_image_2dArray() == 1
        && imageAtomicMax(image_3d, ivec3(0), 2.0f) == 1
        && load_image_3d() == 2
        && imageAtomicMax(image_cube, ivec3(0), 2.0f) == 1
        && load_image_cube() == 2
        && imageAtomicMax(image_cubeArray, ivec3(0), 2.0f) == 1
        && load_image_cubeArray() == 2
        && imageAtomicMax(image_2dMultiSampleArray, ivec3(0), 1, 2.0f) == 1
        && load_image_2dMultiSampleArray() == 2
        ;
}
#endif // #ifdef TEST_when_shader_atomic_float2_is_available

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
#ifdef TEST_when_shader_atomic_float2_is_available
        && checkAllImageAtomicMin()
        && checkAllImageAtomicMax()
#endif // #ifdef TEST_when_shader_atomic_float2_is_available
        ;
    // BUF: 1
}
