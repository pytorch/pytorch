//TEST:SIMPLE(filecheck=CHECK_GLSL):  -allow-glsl -stage compute -entry computeMain -target glsl
//TEST:SIMPLE(filecheck=CHECK_SPV):  -allow-glsl -stage compute -entry computeMain -target spirv -emit-spirv-directly
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl -emit-spirv-directly

// As per vulkan specification: https://registry.khronos.org/vulkan/specs/1.3/html/chap34.html#formats-definition
// i64/u64 images do not require sampled image support; support is currently unlikley on hardware
//#define test_when_hardware_supports_i64_and_u64_sampled_textures
// i64/u64 images do not require texel buffers support; support is currently unlikley on hardware
//#define TEST_when_hardware_supports_i64_and_u64_buffers

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
buffer MyBlockName2
{
    uint data[1];
} outputBuffer;

layout(local_size_x = 1) in;

//TEST_INPUT: set image_1d = RWTexture1D(format=R64Sint, size=4, content=one, mipMaps = 1)
uniform layout(binding=0,r64i) i64image1D image_1d;
#ifdef TEST_when_hardware_supports_i64_and_u64_buffers
//COM:TEST_INPUT: set image_buffer = RWTextureBuffer(format=R64Sint, stride=8, data=[0 1 0 1 0 1 0 1])
uniform layout(binding=1,r64i) i64imageBuffer image_buffer;
#endif // #ifdef TEST_when_hardware_supports_i64_and_u64_buffers
//TEST_INPUT: set image_1dArray = RWTexture1D(format=R64Sint, size=4, content=one, mipMaps = 1, arrayLength=2)
uniform layout(binding=2,r64i) i64image1DArray image_1dArray;
//TEST_INPUT: set image_2d = RWTexture2D(format=R64Sint, size=4, content=one, mipMaps = 1)
uniform layout(binding=3,r64i) i64image2D image_2d;
//TEST_INPUT: set image_2dRect = RWTexture2D(format=R64Sint, size=4, content=one, mipMaps = 1)
uniform layout(binding=4,r64i) i64image2DRect image_2dRect;
//TEST_INPUT: set image_2dArray = RWTexture2D(format=R64Sint, size=4, content=one, mipMaps = 1, arrayLength=2)
uniform layout(binding=6,r64i) i64image2DArray image_2dArray;
//TEST_INPUT: set image_3d = RWTexture3D(format=R64Sint, size=4, content=one, mipMaps = 1)
uniform layout(binding=7,r64i) i64image3D image_3d;
//TEST_INPUT: set image_cube = RWTextureCube(format=R64Sint, size=4, content=one, mipMaps = 1)
uniform layout(binding=8,r64i) i64imageCube image_cube;
//TEST_INPUT: set image_cubeArray = RWTextureCube(format=R64Sint, size=4, content=one, mipMaps = 1, arrayLength=2)
uniform layout(binding=9,r64i) i64imageCubeArray image_cubeArray;
#ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
//COM:TEST_INPUT: set image_2dMultiSample = RWTexture2D(format=R64Sint, size=4, content=one, mipMaps = 1, sampleCount = two)
uniform layout(binding=5,r64i) i64image2DMS image_2dMultiSample;
//COM:TEST_INPUT: set image_2dMultiSampleArray = RWTexture2D(format=R64Sint, size=4, content=one, mipMaps = 1, arrayLength=2, sampleCount = two)
uniform layout(binding=10,r64i) i64image2DMSArray image_2dMultiSampleArray;
#endif // #ifdef test_when_hardware_supports_i64_and_u64_sampled_textures

bool checkAllImageSize()
{
    return true
        && imageSize(image_1d) == int(4)
#ifdef TEST_when_hardware_supports_i64_and_u64_buffers
        && imageSize(image_buffer) == int(4)
#endif // #ifdef TEST_when_hardware_supports_i64_and_u64_buffers
        && imageSize(image_1dArray) == ivec2(4, 2)
        && imageSize(image_2d) == ivec2(4)
        && imageSize(image_2dArray) == ivec3(4, 4, 2)
        && imageSize(image_2dRect) == ivec2(4)
        && imageSize(image_3d) == ivec3(4)
        && imageSize(image_cube) == ivec2(4)
        && imageSize(image_cubeArray) == ivec3(4, 4, 2)
#ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
        && imageSize(image_2dMultiSample) == ivec2(4)
        && imageSize(image_2dMultiSampleArray) == ivec3(4, 4, 2)
#endif // #ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
        ;
}
bool checkAllImageLoad()
{
    return true
        && imageLoad(image_1d, 0).x == 1
#ifdef TEST_when_hardware_supports_i64_and_u64_buffers
        && imageLoad(image_buffer, 0).x == 1
#endif // #ifdef TEST_when_hardware_supports_i64_and_u64_buffers
        && imageLoad(image_1dArray, ivec2(0)).x == 1
        && imageLoad(image_2d, ivec2(0)).x == 1
        && imageLoad(image_2dRect, ivec2(0)).x == 1
        && imageLoad(image_2dArray, ivec3(0)).x == 1
        && imageLoad(image_3d, ivec3(0)).x == 1
        && imageLoad(image_cube, ivec3(0)).x == 1
        && imageLoad(image_cubeArray, ivec3(0)).x == 1
        && imageLoad(image_cubeArray, ivec3(0)).x == 1
#ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
        && imageLoad(image_2dMultiSample, ivec2(0), 1).x == 1
        && imageLoad(image_2dMultiSampleArray, ivec3(0), 1).x == 1
#endif // #ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
        ;
}
bool resetAllImageValues()
{
    imageStore(image_1d, 0,i64vec4(1));
#ifdef TEST_when_hardware_supports_i64_and_u64_buffers
    imageStore(image_buffer, 0,i64vec4(1));
#endif // #ifdef TEST_when_hardware_supports_i64_and_u64_buffers
    imageStore(image_1dArray, ivec2(0),i64vec4(1));
    imageStore(image_2d, ivec2(0),i64vec4(1));
    imageStore(image_2dRect, ivec2(0),i64vec4(1));
    imageStore(image_2dArray, ivec3(0),i64vec4(1));
    imageStore(image_3d, ivec3(0),i64vec4(1));
    imageStore(image_cube, ivec3(0),i64vec4(1));
    imageStore(image_cubeArray, ivec3(0),i64vec4(1));
#ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
    imageStore(image_2dMultiSample, ivec2(0), 1, i64vec4(1));
    imageStore(image_2dMultiSampleArray, ivec3(0), 1,i64vec4(1));
#endif // #ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
    return true;
}
int64_t load_image_1d()
{
    return imageLoad(image_1d, 0).x;
}
#ifdef TEST_when_hardware_supports_i64_and_u64_buffers
int64_t load_image_buffer()
{
    return imageLoad(image_buffer, 0).x;
}
#endif // #ifdef TEST_when_hardware_supports_i64_and_u64_buffers
int64_t load_image_1dArray()
{
    return imageLoad(image_1dArray, ivec2(0)).x;
}
int64_t load_image_2d()
{
    return imageLoad(image_2d, ivec2(0)).x;
}
int64_t load_image_2dRect()
{
    return imageLoad(image_2dRect, ivec2(0)).x;
}
int64_t load_image_2dArray()
{
    return imageLoad(image_2dArray, ivec3(0)).x;
}
int64_t load_image_3d()
{
    return imageLoad(image_3d, ivec3(0)).x;
}
int64_t load_image_cube()
{
    return imageLoad(image_cube, ivec3(0)).x;
}
int64_t load_image_cubeArray()
{
    return imageLoad(image_cubeArray, ivec3(0)).x;
}
#ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
int64_t load_image_2dMultiSample()
{
    return imageLoad(image_2dMultiSample, ivec2(0), 1).x;
}
int64_t load_image_2dMultiSampleArray()
{
    return imageLoad(image_2dMultiSampleArray, ivec3(0), 1).x;
}
#endif // #ifdef test_when_hardware_supports_i64_and_u64_sampled_textures

// requires ImageLoad test to pass
bool checkAllImageStore()
{
    bool loadCheck = true;

    imageStore(image_1d, 0,i64vec4(0));
    loadCheck = loadCheck && load_image_1d() == 0;
#ifdef TEST_when_hardware_supports_i64_and_u64_buffers
    imageStore(image_buffer, 0,i64vec4(0));
    loadCheck = loadCheck && load_image_buffer() == 0;
#endif // #ifdef TEST_when_hardware_supports_i64_and_u64_buffers
    imageStore(image_1dArray, ivec2(0),i64vec4(0));
    loadCheck = loadCheck && load_image_1dArray() == 0;
    imageStore(image_2d, ivec2(0),i64vec4(0));
    loadCheck = loadCheck && load_image_2d() == 0;
    imageStore(image_2dRect, ivec2(0),i64vec4(0));
    loadCheck = loadCheck && load_image_2dRect() == 0;
    imageStore(image_2dArray, ivec3(0),i64vec4(0));
    loadCheck = loadCheck && load_image_2dArray() == 0;
    imageStore(image_3d, ivec3(0),i64vec4(0));
    loadCheck = loadCheck && load_image_3d() == 0;
    imageStore(image_cube, ivec3(0),i64vec4(0));
    loadCheck = loadCheck && load_image_cube() == 0;
    imageStore(image_cubeArray, ivec3(0),i64vec4(0));
    loadCheck = loadCheck && load_image_cubeArray() == 0;
#ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
    imageStore(image_2dMultiSample, ivec2(0), 1, i64vec4(0));
    loadCheck = loadCheck && load_image_2dMultiSample() == 0;
    imageStore(image_2dMultiSampleArray, ivec3(0), 1,i64vec4(0));
    loadCheck = loadCheck && load_image_2dMultiSampleArray() == 0;
#endif // #ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
    resetAllImageValues();
    return loadCheck;
}
bool checkAllImageSamples()
{
    resetAllImageValues();
    return true
#ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
        && imageSamples(image_2dMultiSample) == 2
        && imageSamples(image_2dMultiSampleArray) == 2
#endif // #ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
        ;
}
bool checkAllImageAtomicAdd()
{
    resetAllImageValues();
    return true
        && imageAtomicAdd(image_1d, 0, 0) == 1
        && load_image_1d() == 1
#ifdef TEST_when_hardware_supports_i64_and_u64_buffers
        && imageAtomicAdd(image_buffer, 0, 2) == 1
        && load_image_buffer() == 3
#endif // #ifdef TEST_when_hardware_supports_i64_and_u64_buffers
        && imageAtomicAdd(image_1dArray, ivec2(0), 0) == 1
        && load_image_1dArray() == 1
        && imageAtomicAdd(image_2d, ivec2(0), 2) == 1
        && load_image_2d() == 3
        && imageAtomicAdd(image_2dRect, ivec2(0), 2) == 1
        && load_image_2dRect() == 3
        && imageAtomicAdd(image_2dArray, ivec3(0), 0) == 1
        && load_image_2dArray() == 1
        && imageAtomicAdd(image_3d, ivec3(0), 2) == 1
        && load_image_3d() == 3
        && imageAtomicAdd(image_cube, ivec3(0), 2) == 1
        && load_image_cube() == 3
        && imageAtomicAdd(image_cubeArray, ivec3(0), 2) == 1
        && load_image_cubeArray() == 3
#ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
        && imageAtomicAdd(image_2dMultiSample, ivec2(0), 1, 2) == 1
        && load_image_2dMultiSample() == 3
        && imageAtomicAdd(image_2dMultiSampleArray, ivec3(0), 1, 2) == 1
        && load_image_2dMultiSampleArray() == 3
#endif // #ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
        ;
}
bool checkAllImageAtomicExchange()
{
    resetAllImageValues();
    return true
        && imageAtomicExchange(image_1d, 0, 0) == 1
        && load_image_1d() == 0
#ifdef TEST_when_hardware_supports_i64_and_u64_buffers
        && imageAtomicExchange(image_buffer, 0, 2) == 1
        && load_image_buffer() == 2
#endif // #ifdef TEST_when_hardware_supports_i64_and_u64_buffers
        && imageAtomicExchange(image_1dArray, ivec2(0), 0) == 1
        && load_image_1dArray() == 0
        && imageAtomicExchange(image_2d, ivec2(0), 2) == 1
        && load_image_2d() == 2
        && imageAtomicExchange(image_2dRect, ivec2(0), 2) == 1
        && load_image_2dRect() == 2
        && imageAtomicExchange(image_2dArray, ivec3(0), 0) == 1
        && load_image_2dArray() == 0
        && imageAtomicExchange(image_3d, ivec3(0), 2) == 1
        && load_image_3d() == 2
        && imageAtomicExchange(image_cube, ivec3(0), 2) == 1
        && load_image_cube() == 2
        && imageAtomicExchange(image_cubeArray, ivec3(0), 2) == 1
        && load_image_cubeArray() == 2
#ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
        && imageAtomicExchange(image_2dMultiSample, ivec2(0), 1, 2) == 1
        && load_image_2dMultiSample() == 2
        && imageAtomicExchange(image_2dMultiSampleArray, ivec3(0), 1, 2) == 1
        && load_image_2dMultiSampleArray() == 2
#endif // #ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
        ;
}
bool checkAllImageAtomicMin()
{
    resetAllImageValues();
    return true
        && imageAtomicMin(image_1d, 0, 0) == 1
        && load_image_1d() == 0
#ifdef TEST_when_hardware_supports_i64_and_u64_buffers
        && imageAtomicMin(image_buffer, 0, 2) == 1
        && load_image_buffer() == 1
#endif // #ifdef TEST_when_hardware_supports_i64_and_u64_buffers
        && imageAtomicMin(image_1dArray, ivec2(0), 0) == 1
        && load_image_1dArray() == 0
        && imageAtomicMin(image_2d, ivec2(0), 2) == 1
        && load_image_2d() == 1
        && imageAtomicMin(image_2dRect, ivec2(0), 2) == 1
        && load_image_2dRect() == 1
        && imageAtomicMin(image_2dArray, ivec3(0), 0) == 1
        && load_image_2dArray() == 0
        && imageAtomicMin(image_3d, ivec3(0), 2) == 1
        && load_image_3d() == 1
        && imageAtomicMin(image_cube, ivec3(0), 2) == 1
        && load_image_cube() == 1
        && imageAtomicMin(image_cubeArray, ivec3(0), 2) == 1
        && load_image_cubeArray() == 1
#ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
        && imageAtomicMin(image_2dMultiSample, ivec2(0), 1, 2) == 1
        && load_image_2dMultiSample() == 1
        && imageAtomicMin(image_2dMultiSampleArray, ivec3(0), 1, 2) == 1
        && load_image_2dMultiSampleArray() == 1
#endif // #ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
        ;
}
bool checkAllImageAtomicMax()
{
    resetAllImageValues();
    return true
        && imageAtomicMax(image_1d, 0, 0) == 1
        && load_image_1d() == 1
#ifdef TEST_when_hardware_supports_i64_and_u64_buffers
        && imageAtomicMax(image_buffer, 0, 2) == 1
        && load_image_buffer() == 2
#endif // #ifdef TEST_when_hardware_supports_i64_and_u64_buffers
        && imageAtomicMax(image_1dArray, ivec2(0), 0) == 1
        && load_image_1dArray() == 1
        && imageAtomicMax(image_2d, ivec2(0), 2) == 1
        && load_image_2d() == 2
        && imageAtomicMax(image_2dRect, ivec2(0), 2) == 1
        && load_image_2dRect() == 2
        && imageAtomicMax(image_2dArray, ivec3(0), 0) == 1
        && load_image_2dArray() == 1
        && imageAtomicMax(image_3d, ivec3(0), 2) == 1
        && load_image_3d() == 2
        && imageAtomicMax(image_cube, ivec3(0), 2) == 1
        && load_image_cube() == 2
        && imageAtomicMax(image_cubeArray, ivec3(0), 2) == 1
        && load_image_cubeArray() == 2
#ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
        && imageAtomicMax(image_2dMultiSample, ivec2(0), 1, 2) == 1
        && load_image_2dMultiSample() == 2
        && imageAtomicMax(image_2dMultiSampleArray, ivec3(0), 1, 2) == 1
        && load_image_2dMultiSampleArray() == 2
#endif // #ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
        ;
}
bool checkAllImageAtomicAnd()
{
    resetAllImageValues();
    return true
        && imageAtomicAnd(image_1d, 0, 1) == 1
        && load_image_1d() == 1
#ifdef TEST_when_hardware_supports_i64_and_u64_buffers
        && imageAtomicAnd(image_buffer, 0, 2) == 1
        && load_image_buffer() == 0
#endif // #ifdef TEST_when_hardware_supports_i64_and_u64_buffers
        && imageAtomicAnd(image_1dArray, ivec2(0), 1) == 1
        && load_image_1dArray() == 1
        && imageAtomicAnd(image_2d, ivec2(0), 2) == 1
        && load_image_2d() == 0
        && imageAtomicAnd(image_2dRect, ivec2(0), 2) == 1
        && load_image_2dRect() == 0
        && imageAtomicAnd(image_2dArray, ivec3(0), 1) == 1
        && load_image_2dArray() == 1
        && imageAtomicAnd(image_3d, ivec3(0), 2) == 1
        && load_image_3d() == 0
        && imageAtomicAnd(image_cube, ivec3(0), 2) == 1
        && load_image_cube() == 0
        && imageAtomicAnd(image_cubeArray, ivec3(0), 2) == 1
        && load_image_cubeArray() == 0
#ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
        && imageAtomicAnd(image_2dMultiSample, ivec2(0), 1, 2) == 1
        && load_image_2dMultiSample() == 0
        && imageAtomicAnd(image_2dMultiSampleArray, ivec3(0), 1, 2) == 1
        && load_image_2dMultiSampleArray() == 0
#endif // #ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
        ;
}
bool checkAllImageAtomicOr()
{
    resetAllImageValues();
    return true
        && imageAtomicOr(image_1d, 0, 1) == 1
        && load_image_1d() == 1
#ifdef TEST_when_hardware_supports_i64_and_u64_buffers
        && imageAtomicOr(image_buffer, 0, 2) == 1
        && load_image_buffer() == 3
#endif // #ifdef TEST_when_hardware_supports_i64_and_u64_buffers
        && imageAtomicOr(image_1dArray, ivec2(0), 1) == 1
        && load_image_1dArray() == 1
        && imageAtomicOr(image_2d, ivec2(0), 2) == 1
        && load_image_2d() == 3
        && imageAtomicOr(image_2dRect, ivec2(0), 2) == 1
        && load_image_2dRect() == 3
        && imageAtomicOr(image_2dArray, ivec3(0), 1) == 1
        && load_image_2dArray() == 1
        && imageAtomicOr(image_3d, ivec3(0), 2) == 1
        && load_image_3d() == 3
        && imageAtomicOr(image_cube, ivec3(0), 2) == 1
        && load_image_cube() == 3
        && imageAtomicOr(image_cubeArray, ivec3(0), 2) == 1
        && load_image_cubeArray() == 3
#ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
        && imageAtomicOr(image_2dMultiSample, ivec2(0), 1, 2) == 1
        && load_image_2dMultiSample() == 3
        && imageAtomicOr(image_2dMultiSampleArray, ivec3(0), 1, 2) == 1
        && load_image_2dMultiSampleArray() == 3
#endif // #ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
        ;
}
bool checkAllImageAtomicXor()
{
    resetAllImageValues();
    return true
        && imageAtomicXor(image_1d, 0, 1) == 1
        && load_image_1d() == 0
#ifdef TEST_when_hardware_supports_i64_and_u64_buffers
        && imageAtomicXor(image_buffer, 0, 2) == 1
        && load_image_buffer() == 3
#endif // #ifdef TEST_when_hardware_supports_i64_and_u64_buffers
        && imageAtomicXor(image_1dArray, ivec2(0), 1) == 1
        && load_image_1dArray() == 0
        && imageAtomicXor(image_2d, ivec2(0), 2) == 1
        && load_image_2d() == 3
        && imageAtomicXor(image_2dRect, ivec2(0), 2) == 1
        && load_image_2dRect() == 3
        && imageAtomicXor(image_2dArray, ivec3(0), 1) == 1
        && load_image_2dArray() == 0
        && imageAtomicXor(image_3d, ivec3(0), 2) == 1
        && load_image_3d() == 3
        && imageAtomicXor(image_cube, ivec3(0), 2) == 1
        && load_image_cube() == 3
        && imageAtomicXor(image_cubeArray, ivec3(0), 2) == 1
        && load_image_cubeArray() == 3
#ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
        && imageAtomicXor(image_2dMultiSample, ivec2(0), 1, 2) == 1
        && load_image_2dMultiSample() == 3
        && imageAtomicXor(image_2dMultiSampleArray, ivec3(0), 1, 2) == 1
        && load_image_2dMultiSampleArray() == 3
#endif // #ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
        ;
}
bool checkAllImageAtomicCompSwap()
{
    resetAllImageValues();
    return true
        && imageAtomicCompSwap(image_1d, 0, 0, 2) == 1
        && load_image_1d() == 1
#ifdef TEST_when_hardware_supports_i64_and_u64_buffers
        && imageAtomicCompSwap(image_buffer, 0, 1, 2) == 1
        && load_image_buffer() == 2
#endif // #ifdef TEST_when_hardware_supports_i64_and_u64_buffers
        && imageAtomicCompSwap(image_1dArray, ivec2(0), 0, 2) == 1
        && load_image_1dArray() == 1
        && imageAtomicCompSwap(image_2d, ivec2(0), 1, 2) == 1
        && load_image_2d() == 2
        && imageAtomicCompSwap(image_2dRect, ivec2(0), 1, 2) == 1
        && load_image_2dRect() == 2
        && imageAtomicCompSwap(image_2dArray, ivec3(0), 0, 2) == 1
        && load_image_2dArray() == 1
        && imageAtomicCompSwap(image_3d, ivec3(0), 1, 2) == 1
        && load_image_3d() == 2
        && imageAtomicCompSwap(image_cube, ivec3(0), 1, 2) == 1
        && load_image_cube() == 2
        && imageAtomicCompSwap(image_cubeArray, ivec3(0), 1, 2) == 1
        && load_image_cubeArray() == 2
#ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
        && imageAtomicCompSwap(image_2dMultiSample, ivec2(0), 1, 1, 2) == 1
        && load_image_2dMultiSample() == 2
        && imageAtomicCompSwap(image_2dMultiSampleArray, ivec3(0), 1, 1, 2) == 1
        && load_image_2dMultiSampleArray() == 2
#endif // #ifdef test_when_hardware_supports_i64_and_u64_sampled_textures
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
