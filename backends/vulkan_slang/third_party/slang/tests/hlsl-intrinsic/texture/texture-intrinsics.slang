//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=DX11):-slang -compute -dx11 -shaderobj -output-using-type -render-feature hardware-device
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=DX12):-slang -compute -dx12 -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=DX12CS6):-slang -compute -dx12 -profile cs_6_0 -use-dxil -shaderobj -output-using-type -xslang -DCS60
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=VK):-vk -emit-spirv-directly -compute -shaderobj -output-using-type -render-feature hardware-device -xslang -DVK
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj -output-using-type

//TEST_INPUT: Texture1D(size=4, content = one):name t1D
Texture1D<float> t1D;
//TEST_INPUT: Texture2D(size=4, content = one):name t2D
Texture2D<float> t2D;
//TEST_INPUT: Texture3D(size=4, content = one):name t3D
Texture3D<float> t3D;
//TEST_INPUT: TextureCube(size=4, content = one):name tCube
TextureCube<float> tCube;

//TEST_INPUT: Texture1D(size=4, content = one, arrayLength=2):name t1DArray
Texture1DArray<float> t1DArray;
//TEST_INPUT: Texture2D(size=4, content = one, arrayLength=2):name t2DArray
Texture2DArray<float> t2DArray;
//TEST_INPUT: TextureCube(size=4, content = one, arrayLength=2):name tCubeArray
TextureCubeArray<float> tCubeArray;

//TEST_INPUT: Sampler:name samplerState
SamplerState samplerState;
//TEST_INPUT: Sampler:name shadowSampler
SamplerComparisonState shadowSampler;


//TEST_INPUT: ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    int idx = dispatchThreadID.x;
    float u = idx * (1.0f / 4);

    float val = 0.0f;

    uint width = 0, height = 0, depth = 0;
    float fwidth = 0.0f, fheight = 0.0f, fdepth = 0.0f;
    uint numLevels = 0, elements = 0;
    float fnumLevels = 0.0f, felements = 0.0f;

    float clamp = 0.0f;
    float bias = 0.0f;

    uint status;

    /*
        <Template Type> Object.SampleLevel()
    */
    val += t1D.SampleLevel(samplerState, u, 0);
    val += t2D.SampleLevel(samplerState, float2(u, u), 0);
    val += t3D.SampleLevel(samplerState, float3(u, u, u), 0);
    val += tCube.SampleLevel(samplerState, normalize(float3(u, 1 - u, u)), 0);

    val += t1DArray.SampleLevel(samplerState, float2(u, 0), 0);
    val += t2DArray.SampleLevel(samplerState, float3(u, u, 0), 0);
    val += tCubeArray.SampleLevel(samplerState, float4(u, u, u, 0), 0);

    // Offset variant
    //  NOTE: The "cpu" profile does not like these, so it's disabled for now
    val += t1D.SampleLevel(samplerState, u, 0, 1);
    val += t2D.SampleLevel(samplerState, float2(u, u), 0, int2(1, 1));
    val += t3D.SampleLevel(samplerState, float3(u, u, u), 0, int3(1, 1, 1));

    val += t1DArray.SampleLevel(samplerState, float2(u, 0), 0, 1);
    val += t2DArray.SampleLevel(samplerState, float3(u, u, 0), 0, int2(1, 1));

    // Status variant
#if !defined(VK)
    val += t1D.SampleLevel(samplerState, u, 0, 1, status);
    val += t2D.SampleLevel(samplerState, float2(u, u), 0, int2(1, 1), status);
    val += t3D.SampleLevel(samplerState, float3(u, u, u), 0, int3(1, 1, 1), status);

    val += t1DArray.SampleLevel(samplerState, float2(u, 0), 0, 1, status);
    val += t2DArray.SampleLevel(samplerState, float3(u, u, 0), 0, int2(1, 1), status);
#endif

    /*
        float Object.SampleCmpLevelZero()
    */
    // NOTE: These are having issues with vulkan (glsl)
    val += t1D.SampleCmpLevelZero(shadowSampler, u, 0);
    val += t2D.SampleCmpLevelZero(shadowSampler, float2(u, u), 0);
    val += t1DArray.SampleCmpLevelZero(shadowSampler, float2(u, u), 0);
    val += tCube.SampleCmpLevelZero(shadowSampler, normalize(float3(u, 1 - u, u)), 0);
    val += t2DArray.SampleCmpLevelZero(shadowSampler, normalize(float3(u, 1 - u, u)), 0);
    val += tCubeArray.SampleCmpLevelZero(shadowSampler, normalize(float4(u, 1-u, u, u)), 0);

    // Offset variant
    val += t1D.SampleCmpLevelZero(shadowSampler, u, 0, 0);
    val += t2D.SampleCmpLevelZero(shadowSampler, float2(u, u), 0, int2(0, 0));
    // TextureCube does not have an offset version of this

    // Status variant
#if !defined(VK)
    val += t1D.SampleCmpLevelZero(shadowSampler, u, 0, 0, status);
    val += t2D.SampleCmpLevelZero(shadowSampler, float2(u, u), 0, int2(0, 0), status);

    val += t1DArray.SampleCmpLevelZero(shadowSampler, float2(u, u), 0, 0, status);
    val += t2DArray.SampleCmpLevelZero(shadowSampler, normalize(float3(u, 1 - u, u)), 0, int2(0, 0), status);
#endif

    /*
        float Object.SampleCmpLevel().
        These require SM 6.7 for dx12 but functional compute tests currently do not run with the cs_6_7,
        hence only enable these for vk.
    */
#if defined(VK)
    float level = 1.0;
    val += t1D.SampleCmpLevel(shadowSampler, u, 0, level);
    val += t2D.SampleCmpLevel(shadowSampler, float2(u, u), 0, level);
    val += t1DArray.SampleCmpLevel(shadowSampler, float2(u, u), 0, level);
    val += tCube.SampleCmpLevel(shadowSampler, normalize(float3(u, 1 - u, u)), 0, level);
    val += t2DArray.SampleCmpLevel(shadowSampler, normalize(float3(u, 1 - u, u)), 0, level);
    val += tCubeArray.SampleCmpLevel(shadowSampler, normalize(float4(u, 1-u, u, u)), 0, level);

    // Offset variant
    val += t1D.SampleCmpLevel(shadowSampler, u, 0, level, 0);
    val += t2D.SampleCmpLevel(shadowSampler, float2(u, u), 0, level, int2(0, 0));
#endif

    /*
        void Object.GetDimensions()
    */
    t1D.GetDimensions(width);
    val += width;

    t1D.GetDimensions(fwidth);
    val += fwidth;

    t1D.GetDimensions(0, width, numLevels);
    val += width;
    val += numLevels;

    t1D.GetDimensions(0, fwidth, fnumLevels);
    val += fwidth;
    val += fnumLevels;

    t2D.GetDimensions(width, height);
    val += width;
    val += height;

    t2D.GetDimensions(fwidth, fheight);
    val += fwidth;
    val += fheight;

    t2D.GetDimensions(0, width, height, numLevels);
    val += width;
    val += height;
    val += numLevels;

    t2D.GetDimensions(0, fwidth, fheight, fnumLevels);
    val += fwidth;
    val += fheight;
    val += fnumLevels;

    t3D.GetDimensions(width, height, depth);
    val += width;
    val += height;
    val += depth;

    t3D.GetDimensions(fwidth, fheight, fdepth);
    val += fwidth;
    val += fheight;
    val += fdepth;

    t3D.GetDimensions(0, width, height, depth, numLevels);
    val += width;
    val += height;
    val += depth;
    val += numLevels;

    t3D.GetDimensions(0, fwidth, fheight, fdepth, fnumLevels);
    val += fwidth;
    val += fheight;
    val += fdepth;
    val += fnumLevels;

    tCube.GetDimensions(width, height);
    val += width;
    val += height;

    tCube.GetDimensions(fwidth, fheight);
    val += fwidth;
    val += fheight;

    tCube.GetDimensions(0, width, height, numLevels);
    val += width;
    val += height;
    val += numLevels;

    tCube.GetDimensions(0, fwidth, fheight, fnumLevels);
    val += fwidth;
    val += fheight;
    val += fnumLevels;

    t1DArray.GetDimensions(width, elements);
    val += width;
    val += elements;

    t1DArray.GetDimensions(fwidth, felements);
    val += fwidth;
    val += felements;

    t1DArray.GetDimensions(0, width, elements, numLevels);
    val += width;
    val += elements;
    val += numLevels;

    t1DArray.GetDimensions(0, fwidth, felements, fnumLevels);
    val += fwidth;
    val += felements;
    val += fnumLevels;

    t2DArray.GetDimensions(width, height, elements);
    val += width;
    val += height;
    val += elements;

    t2DArray.GetDimensions(fwidth, fheight, felements);
    val += fwidth;
    val += fheight;
    val += felements;

    t2DArray.GetDimensions(0, width, height, elements, numLevels);
    val += width;
    val += height;
    val += elements;
    val += numLevels;

    t2DArray.GetDimensions(0, fwidth, fheight, felements, fnumLevels);
    val += fwidth;
    val += fheight;
    val += felements;
    val += fnumLevels;

    tCubeArray.GetDimensions(width, height, elements);
    val += width;
    val += height;
    val += elements;

    // fxc 47.0: (95): error X4598:  The array element count of GetDimensions on TextureCubeArray objects is unavailable on cs_5_0
#if defined(CS60) || defined(VK)
    tCubeArray.GetDimensions(fwidth, fheight, felements);
    val += fwidth;
    val += fheight;
    val += felements;
#endif

    tCubeArray.GetDimensions(0, width, height, elements, numLevels);
    val += width;
    val += height;
    val += elements;
    val += numLevels;

#if defined(CS60) || defined(VK)
    tCubeArray.GetDimensions(0, fwidth, fheight, felements, fnumLevels);
    val += fwidth;
    val += fheight;
    val += felements;
    val += fnumLevels;
#endif

    /*
        <Template Type>4 Object.Gather()
    */
    float4 f4 = t2D.Gather(samplerState, float2(u, u));
    val += f4.x;  val += f4.y;  val += f4.z;  val += f4.w;

    f4 = tCube.Gather(samplerState, normalize(float3(u, 1 - u, u)));
    val += f4.x;  val += f4.y;  val += f4.z;  val += f4.w;

    f4 = t2DArray.Gather(samplerState, float3(u, u, 0));
    val += f4.x;  val += f4.y;  val += f4.z;  val += f4.w;

    f4 = tCubeArray.Gather(samplerState, float4(normalize(float3(u, 1 - u, u)), 0));
    val += f4.x;  val += f4.y;  val += f4.z;  val += f4.w;

    // Offset variant
    f4 = t2D.Gather(samplerState, float2(u, u), int2(0, 0));
    val += f4.x;  val += f4.y;  val += f4.z;  val += f4.w;

    f4 = t2DArray.Gather(samplerState, float3(u, u, 0), int2(0, 0));
    val += f4.x;  val += f4.y;  val += f4.z;  val += f4.w;

    // Object.GatherGreen()
    f4 = t2D.GatherGreen(samplerState, float2(u, u), int2(1, 2));
    val += f4.x;  val += f4.y;  val += f4.z;  val += f4.w;

    f4 = t2D.GatherGreen(samplerState, float2(u, u), int2(1, 2), int2(3, 4), int2(5, 6), int2(7, 8));
    val += f4.x;  val += f4.y;  val += f4.z;  val += f4.w;

    f4 = tCubeArray.GatherGreen(samplerState, float4(1.5, 1.5, 1.5, 1.5));
    val += f4.x;  val += f4.y;  val += f4.z;  val += f4.w;

    // status variant
#if !defined(VK)
    f4 = t2D.GatherGreen(samplerState, float2(u, u), int2(1, 2), status);
    val += f4.x;  val += f4.y;  val += f4.z;  val += f4.w;

    f4 = t2D.GatherGreen(samplerState, float2(u, u), int2(1, 2), int2(3, 4), int2(5, 6), int2(7, 8), status);
    val += f4.x;  val += f4.y;  val += f4.z;  val += f4.w;

    f4 = tCubeArray.GatherGreen(samplerState, float4(1.5, 1.5, 1.5, 1.5), status);
    val += f4.x;  val += f4.y;  val += f4.z;  val += f4.w;
#endif

    /*
        ret Object.Load()
    */
    val += t1D.Load(int2(0, 0));
    val += t2D.Load(int3(0, 0, 0));
    val += t3D.Load(int4(0, 0, 0, 0));

    val += t1DArray.Load(int3(0, 0, 0));
    val += t2DArray.Load(int4(0, 0, 0, 0));

    // Offset variant
    val += t1D.Load(int2(0, 0), 0);
    val += t2D.Load(int3(0, 0, 0), int2(0,0));
    val += t3D.Load(int4(0, 0, 0, 0), int3(0, 0, 0));

    val += t1DArray.Load(int3(0, 0, 0), 0);
    val += t2DArray.Load(int4(0, 0, 0, 0), int2(0, 0));

    /*
        <Template Type> Object.SampleGrad()
    */
    float _ddx = 0.0f, _ddy = 0.0f;
    val += t1D.SampleGrad(samplerState, 0.0f, _ddx, _ddy);
    val += t2D.SampleGrad(samplerState, float2(u, u), float2(_ddx, _ddx), float2(_ddy, _ddy));
    val += t3D.SampleGrad(samplerState, float3(u, u, u), float3(_ddx, _ddx, _ddx), float3(_ddy, _ddy, _ddy));
    val += tCube.SampleGrad(samplerState, normalize(float3(u, 1 - u, u)), float3(_ddx, _ddx, _ddx), float3(_ddy, _ddy, _ddy));

    val += t1DArray.SampleGrad(samplerState, float2(0.0f, 0.0f), _ddx, _ddy);
    val += t2DArray.SampleGrad(samplerState, float3(u, u, 0.0f), float2(_ddx, _ddx), float2(_ddy, _ddy));

    // Offset variant
    val += t1D.SampleGrad(samplerState, 0.0f, _ddx, _ddy, 0);
    val += t2D.SampleGrad(samplerState, float2(u, u), float2(_ddx, _ddx), float2(_ddy, _ddy), int2(0, 0));
    val += t3D.SampleGrad(samplerState, float3(u, u, u), float3(_ddx, _ddx, _ddx), float3(_ddy, _ddy, _ddy), int3(0, 0, 0));

    val += t1DArray.SampleGrad(samplerState, float2(0.0f, 0.0f), _ddx, _ddy, 0);
    val += t2DArray.SampleGrad(samplerState, float3(u, u, 0.0f), float2(_ddx, _ddx), float2(_ddy, _ddy), int2(0, 0));

    // Status variant
#if !defined(VK)
    val += t1D.SampleGrad(samplerState, 0.0f, _ddx, _ddy, 0, clamp,  status);
    val += t2D.SampleGrad(samplerState, float2(u, u), float2(_ddx, _ddx), float2(_ddy, _ddy), int2(0, 0), clamp, status);
    val += t3D.SampleGrad(samplerState, float3(u, u, u), float3(_ddx, _ddx, _ddx), float3(_ddy, _ddy, _ddy), int3(0, 0, 0), clamp, status);

    val += t1DArray.SampleGrad(samplerState, float2(0.0f, 0.0f), _ddx, _ddy, 0, clamp, status);
    val += t2DArray.SampleGrad(samplerState, float3(u, u, 0.0f), float2(_ddx, _ddx), float2(_ddy, _ddy), int2(0, 0), clamp, status);
#endif

    outputBuffer[idx] = val;
}

// DX11: 354
// DX11: 354
// DX11: 354
// DX11: 354
// DX12: 354
// DX12: 354
// DX12: 354
// DX12: 354
// DX12CS6: 377
// DX12CS6: 377
// DX12CS6: 377
// DX12CS6: 377
// VK: 359
// VK: 359
// VK: 359
// VK: 359
